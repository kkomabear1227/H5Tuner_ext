"""Command-line entry point.

    python3 -m autotuner run   --framework h5tuner --dry-run
    python3 -m autotuner sweep --space minimal --record traces/grid.jsonl
    python3 -m autotuner rank  traces/grid.jsonl
    python3 -m autotuner info

Every framework's paper-reported settings are applied as defaults; anything
given on the command line wins.  That is deliberate: the TunIO paper's own
comparison table is a set of ablations of one pipeline, so running them should
not require editing code.
"""

import argparse
import json
import os
import random
import sys

from . import apps, darshan, early_stopping, evaluate, frameworks
from . import launchers
from . import metrics, objective, rl
from . import space
from . import subset as subset_module
from . import trace as trace_module

PROGRAM = 'autotuner'


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _add_shared(parser):
    parser.add_argument('--space', choices=sorted(space.SPACES),
                        help='parameter space (default: framework\'s)')
    parser.add_argument('--objective', choices=sorted(objective.OBJECTIVES),
                        help='what to optimise (default: framework\'s)')
    parser.add_argument('--reduction',
                        choices=('best', 'min', 'max', 'mean'),
                        help='how to collapse repetitions of one candidate')
    parser.add_argument('--app-cmd',
                        default=os.environ.get('H5TUNER_APP_CMD'),
                        help='shell command that runs the application')
    parser.add_argument('--workdir', default=os.getcwd(),
                        help='where config.xml is written and the application '
                             'runs (default: cwd)')
    parser.add_argument('--reps', type=int,
                        help='measurements per candidate')
    parser.add_argument('--timeout', type=int, default=59 * 60,
                        help='per-candidate budget in seconds '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int,
                        help='seed the RNG for a reproducible run')
    parser.add_argument('--perf-regex',
                        help='regex scraping bandwidth from application '
                             'output; named groups bw_write, bw_read, '
                             'bytes_written, bytes_read are recognised')
    parser.add_argument('--perf-unit', default='MB/s',
                        choices=('B/s', 'KB/s', 'MB/s', 'GB/s', 'TB/s'),
                        help='unit of the scraped bandwidth '
                             '(default: %(default)s)')
    parser.add_argument('--cleanup-file', action='append', dest='cleanup',
                        help='remove this file before each measurement; '
                             'repeatable')
    parser.add_argument('--repetition-cost', choices=('one', 'all'),
                        default='one',
                        help="how repetitions of one candidate are charged to "
                             "the tuning budget. 'one' follows the paper (IV "
                             "Methodology: the extra runs are not accumulated) "
                             "and keeps RoTI comparable across frameworks that "
                             "repeat a different number of times; 'all' "
                             "charges every run (default: %(default)s)")
    parser.add_argument('--dry-run', action='store_true',
                        help='score candidates with a synthetic model, '
                             'launching nothing')
    parser.add_argument('--record', metavar='PATH',
                        help='append every measurement to a JSONL trace')
    parser.add_argument('--replay', metavar='PATH',
                        help='score candidates from a recorded trace instead '
                             'of launching the application')
    parser.add_argument('--no-cache', action='store_true',
                        help='re-measure repeated configurations')
    parser.add_argument('--hdf5-version', metavar='X.Y.Z',
                        help='HDF5 version the shim was built against, e.g. '
                             '1.14.1. Used to warn before a run that tunes a '
                             'parameter the build cannot inject; the shim also '
                             'warns at runtime')
    parser.add_argument('--launcher', choices=sorted(launchers.LAUNCHERS),
                        default='local',
                        help='how a candidate is executed: local runs it here '
                             '(correct inside an allocation), pbs submits one '
                             'PBS Pro job per candidate from a login node '
                             '(default: %(default)s)')
    parser.add_argument('--app', choices=sorted(apps.APPS),
                        help='application adapter; writes the application\'s '
                             'own input file and supplies --app-cmd and '
                             '--perf-regex unless you give them')
    parser.add_argument('--mpi-cmd', default='mpirun',
                        help='parallel launcher the adapter builds its command '
                             'around (default: %(default)s)')
    parser.add_argument('--ld-preload', metavar='PATH',
                        help='libautotuner.so, exported to the application '
                             'only. Prefer this to exporting LD_PRELOAD in '
                             'your shell, which would also load the shim into '
                             'python and qsub')

    dsh = parser.add_argument_group('Darshan (what the paper measures with)')
    dsh.add_argument('--darshan-lib', metavar='PATH',
                     help='libdarshan.so on the target machine. Setting this '
                          'switches bandwidth from output scraping to the '
                          'Darshan log, which is what TunIO III-E uses')
    dsh.add_argument('--darshan-parser', default='darshan-parser',
                     metavar='BIN',
                     help='darshan-parser executable, run where the tuner '
                          'runs (default: %(default)s)')
    dsh.add_argument('--darshan-first', action='store_true',
                     help='put libdarshan.so ahead of libautotuner.so in '
                          'LD_PRELOAD. Default is shim first, so Darshan '
                          'observes I/O after the parameters are injected')
    dsh.add_argument('--darshan-nonmpi', action='store_true',
                     help='set DARSHAN_ENABLE_NONMPI, for an application that '
                          'never calls MPI_Init')

    h5b = parser.add_argument_group('h5bench (with --app h5bench-write/read)')
    h5b.add_argument('--h5bench-bin', metavar='PATH',
                     help='path to h5bench_write / h5bench_read (default: '
                          'found on PATH)')
    h5b.add_argument('--h5bench-particles', default='2M',
                     help='NUM_PARTICLES per rank; 32 bytes each '
                          '(default: %(default)s)')
    h5b.add_argument('--h5bench-timesteps', type=int, default=4,
                     help='TIMESTEPS (default: %(default)s)')
    h5b.add_argument('--h5bench-mem-pattern', default='CONTIG',
                     choices=('CONTIG', 'INTERLEAVED', 'STRIDED'))
    h5b.add_argument('--h5bench-file-pattern', default='CONTIG',
                     choices=('CONTIG', 'INTERLEAVED', 'STRIDED'))
    h5b.add_argument('--h5bench-output', default=apps.DEFAULT_OUTPUT,
                     help='HDF5 file the benchmark writes, relative to the '
                          'candidate directory (default: %(default)s)')
    h5b.add_argument('--h5bench-set', action='append', metavar='KEY=VALUE',
                     dest='h5bench_set',
                     help='override or add one h5bench config key; repeatable')

    mac = parser.add_argument_group('MACSio (with --app macsio)')
    mac.add_argument('--macsio-bin', metavar='PATH',
                     help='path to the macsio binary')
    mac.add_argument('--macsio-part-size', default='16Mi',
                     help='bytes per mesh part, e.g. 16Mi '
                          '(default: %(default)s)')
    mac.add_argument('--macsio-num-dumps', type=int, default=5,
                     help='dumps to marshal (default: %(default)s)')
    mac.add_argument('--macsio-parts-per-rank', default='1',
                     help='avg_num_parts (default: %(default)s)')
    mac.add_argument('--macsio-part-dim', type=int, default=3,
                     choices=(1, 2, 3),
                     help='spatial dimension of parts (default: %(default)s)')
    mac.add_argument('--macsio-file-mode', default='SIF',
                     help='parallel_file_mode; SIF is one shared file and is '
                          'what exercises collective I/O (default: '
                          '%(default)s)')
    mac.add_argument('--macsio-json-lib', metavar='DIR',
                     help='directory holding libjson-cwx.so, needed when '
                          'macsio was built inside h5bench')
    mac.add_argument('--macsio-arg', action='append', dest='macsio_extra',
                     metavar='ARG', default=[],
                     help='extra macsio argument; repeatable. Do not pass '
                          '--sieve_buf_size, --meta_block_size or --alignment '
                          'here: those are tuned parameters and belong to the '
                          'shim')

    exr = parser.add_argument_group('exerciser (with --app exerciser)')
    exr.add_argument('--exerciser-bin', metavar='PATH',
                     help='path to h5bench_exerciser')
    exr.add_argument('--exerciser-numdims', type=int, default=1,
                     choices=(1, 2, 3, 4),
                     help='numDims; reshapes the hyperslab and moves the '
                          'optimum (default: %(default)s)')
    exr.add_argument('--exerciser-minels', default='16777216',
                     help='elements per rank per dimension, comma separated, '
                          'one per dimension or one value for all. Doubles, '
                          'so 16777216 is 128 MiB/rank '
                          '(default: %(default)s)')
    exr.add_argument('--exerciser-dimranks', metavar='N[,N...]',
                     help='rank grid, comma separated; the product must equal '
                          'the rank count the launcher provides')
    exr.add_argument('--exerciser-bufmult', default='1',
                     help='bufMult per dimension (default: %(default)s)')
    exr.add_argument('--exerciser-nsizes', type=int, default=1,
                     help='buffer-size loops (default: %(default)s)')
    exr.add_argument('--exerciser-indepio', action='store_true',
                     help='set the transfer property list to independent I/O. '
                          'Not a tuned parameter, so it is a context knob')
    exr.add_argument('--exerciser-chunked', action='store_true',
                     help='call H5Pset_chunk, which makes chunk_cache live')
    exr.add_argument('--exerciser-derivedtype', action='store_true',
                     help='add the compound-type dataset (8x the bytes)')
    exr.add_argument('--exerciser-addattr', action='store_true',
                     help='add attributes, which exercises the metadata path')
    exr.add_argument('--exerciser-keepfile', action='store_true',
                     help='keep the HDF5 file. Off by default: the kernel '
                          'unlinks it, which a long campaign needs')

    pbs = parser.add_argument_group('PBS Pro (with --launcher pbs)')
    pbs.add_argument('--pbs-queue', default='normal',
                     help='(default: %(default)s)')
    pbs.add_argument('--pbs-account', metavar='CODE',
                     help='the -A application code, mandatory on Nurion')
    pbs.add_argument('--pbs-select', type=int, default=1,
                     help='nodes per job (default: %(default)s)')
    pbs.add_argument('--pbs-ncpus', type=int, default=64,
                     help='cores per node (default: %(default)s)')
    pbs.add_argument('--pbs-mpiprocs', type=int, default=64,
                     help='MPI ranks per node (default: %(default)s)')
    pbs.add_argument('--pbs-ompthreads', type=int, default=1,
                     help='(default: %(default)s)')
    pbs.add_argument('--pbs-walltime', default='00:30:00',
                     help='per-job walltime (default: %(default)s)')
    pbs.add_argument('--pbs-module', action='append', dest='pbs_modules',
                     metavar='NAME',
                     help='module load NAME inside the job; repeatable')
    pbs.add_argument('--pbs-preamble', action='append', dest='pbs_preamble',
                     metavar='LINE',
                     help='shell line to run inside the job before the '
                          'application; repeatable')
    pbs.add_argument('--pbs-poll', type=float, default=15.0,
                     help='seconds between completion checks '
                          '(default: %(default)s)')
    pbs.add_argument('--pbs-wait-minutes', type=float, default=240.0,
                     help='give up on a candidate after this long in queue '
                          'plus execution (default: %(default)s)')

    parser.add_argument('--levels', type=int, metavar='K',
                        help='thin every parameter to K candidate values. '
                             'A full grid over the thinned space is small '
                             'enough to sweep, which is what makes a trace '
                             'replayable without misses; use the same K for '
                             'the sweep and the replay')


def build_parser():
    parser = argparse.ArgumentParser(
        prog=PROGRAM, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # `required=True` is Python 3.7+.  Nurion's compute nodes run the system
    # python 3.6 even where the login node has 3.9, so the flag is applied
    # after the fact and main() reports a missing command itself.
    sub = parser.add_subparsers(dest='command')

    run = sub.add_parser('run', help='run an autotuning framework',
                         formatter_class=argparse.RawDescriptionHelpFormatter,
                         epilog='frameworks:\n' + frameworks.describe())
    run.add_argument('--framework', required=True, choices=frameworks.NAMES)
    _add_shared(run)
    run.add_argument('--generations', type=int)
    run.add_argument('--population', type=int)
    run.add_argument('--elite', type=int)
    run.add_argument('--mutation-rate', type=float)
    run.add_argument('--crossover-rate', type=float)
    run.add_argument('--scale-mutation', action='store_true',
                     help='divide the mutation rate by the active gene count. '
                          'A deviation from both papers, which use a fixed '
                          'per-gene rate; matters above ~10 parameters')
    run.add_argument('--stopper', choices=early_stopping.STOPPERS,
                     help='early stopping policy (default: framework\'s)')
    run.add_argument('--max-minutes', type=float,
                     help='hard tuning budget; stops the run when spent')
    run.add_argument('--subset', choices=subset_module.SUBSET_SELECTORS,
                     help='parameter subset policy (default: framework\'s)')
    run.add_argument('--subset-size', type=int,
                     help='parameters tuned per iteration when --subset static')
    run.add_argument('--subset-grow-every', type=int,
                     help='add one parameter to the subset every N iterations')
    run.add_argument('--ranking', metavar='PATH',
                     help='impact ranking for --subset static: one parameter '
                          'name per line, most impactful first')
    run.add_argument('--ranking-from-trace', metavar='PATH',
                     help='derive the impact ranking from a recorded trace')
    run.add_argument('--size-penalty', type=float, default=1.0,
                     help='weight of the subset-size term in the RL picker\'s '
                          'reward. The paper gives the terms equal weight '
                          '(default: %(default)s)')
    run.add_argument('--rl-seed', type=int,
                     help='seed for RL weight initialisation and offline '
                          'training; defaults to --seed')
    run.add_argument('--rl-curves', type=int, default=240,
                     help='synthetic tuning curves used to train the RL '
                          'stopper offline (default: %(default)s)')
    run.add_argument('--rl-rounds', type=int, default=60,
                     help='maximum fitted-Q rounds; training stops earlier '
                          'when the held-out score stagnates '
                          '(default: %(default)s)')
    run.add_argument('--rl-verbose', action='store_true',
                     help='print RL offline training progress')
    run.add_argument('--summary-json', metavar='PATH',
                     help='write the campaign summary as JSON')
    run.add_argument('--quiet', action='store_true',
                     help='only print the final summary')

    sweep = sub.add_parser('sweep',
                           help='measure a grid or random sample and record it')
    _add_shared(sweep)
    sweep.add_argument('--samples', type=int,
                       help='random sample size; omit for the full grid')
    sweep.add_argument('--max-points', type=int, default=100000,
                       help='refuse a grid larger than this '
                            '(default: %(default)s)')

    rank = sub.add_parser('rank', help='rank parameter impact from a trace')
    rank.add_argument('trace')
    rank.add_argument('--space', choices=sorted(space.SPACES),
                      default='paper12')
    rank.add_argument('--objective', choices=sorted(objective.OBJECTIVES),
                      default='bandwidth')
    rank.add_argument('--out', metavar='PATH',
                      help='write the ranking as one name per line')

    discover = sub.add_parser(
        'discover', help="reduce an application's source to its I/O kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="TunIO III-B. Needs the clang python bindings "
               "(pip install libclang) and the application's include paths.")
    discover.add_argument('source', help='C source file to reduce')
    discover.add_argument('--out', metavar='PATH',
                          help='write the kernel here (default: stdout)')
    discover.add_argument('--include-dir', action='append', metavar='DIR',
                          dest='include_dirs', default=[],
                          help='add DIR to the clang include path; '
                               'repeatable. Almost always needed: on the '
                               'cluster $HDF5ROOT/include, for testing '
                               'test/stubs')
    discover.add_argument('--clang-arg', action='append', dest='clang_args',
                          metavar='ARG', default=[],
                          help='extra clang argument; repeatable. Write it as '
                               '--clang-arg=-DFOO so argparse does not read a '
                               'leading dash as a flag of its own')
    discover.add_argument('--io-prefix', action='append', dest='io_prefixes',
                          metavar='PREFIX',
                          help='treat calls with this prefix as I/O '
                               '(default: H5); repeatable')
    discover.add_argument('--keep-region', action='append',
                          dest='keep_regions', metavar='START:END',
                          help='keep these source lines whatever the marking '
                               'decides; repeatable. The paper allows '
                               '"manually indicated keep regions"')
    discover.add_argument('--loop-keep', type=int, metavar='N',
                          help='cap every loop containing I/O at N iterations '
                               '(loop reduction). This is where the paper\'s '
                               'largest RoTI gain comes from, and it trades '
                               'locality and caching information')
    discover.add_argument('--memory-path', metavar='DIR',
                          help='redirect I/O paths under DIR, e.g. /dev/shm '
                               '(I/O path switching). Faster, but no longer '
                               'tunes for the target storage')
    discover.add_argument('--no-clang-format', action='store_true',
                          help='skip the clang-format pass. The paper formats '
                               'first so one line holds one statement')
    discover.add_argument('--allow-parse-errors', action='store_true',
                          help='continue despite clang errors. Unsafe: calls '
                               'whose types are unknown are invisible to the '
                               'AST, so the kernel may silently miss I/O')
    discover.add_argument('--workdir', default=os.getcwd(),
                          help='where the formatted source is written '
                               '(default: cwd)')

    sub.add_parser('info', help='list frameworks and parameter spaces')

    return parser


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def _hdf5_version(args):
    raw = getattr(args, 'hdf5_version', None)
    if not raw:
        return None
    parts = []
    for piece in raw.replace('-', '.').split('.'):
        if piece.isdigit():
            parts.append(int(piece))
    if not parts:
        raise ValueError('could not read a version out of {0!r}'.format(raw))
    return tuple(parts[:3])


def _space_for(args, name):
    the_space = space.get_space(name)
    if getattr(args, 'levels', None):
        the_space = space.coarsen(the_space, args.levels)
    return the_space


def _resolve(args, settings, key, fallback=None):
    """Command line wins, then the framework's default, then `fallback`."""
    value = getattr(args, key.replace('-', '_'), None)
    if value is not None:
        return value
    if key in settings:
        return settings[key]
    return fallback


def _build_app(args):
    """Instantiate the application adapter named by --app, if any."""
    name = getattr(args, 'app', None)
    if not name:
        return None

    extra = {}
    for item in getattr(args, 'h5bench_set', None) or []:
        if '=' not in item:
            raise ValueError(
                '--h5bench-set expects KEY=VALUE, got {0!r}'.format(item))
        key, value = item.split('=', 1)
        extra[key.strip()] = value.strip()

    if name == 'exerciser':
        def _ints(text):
            if text is None:
                return None
            values = [item.strip() for item in str(text).split(',')
                      if item.strip()]
            if not values:
                return None
            return [int(item) for item in values] if len(values) > 1 \
                else int(values[0])

        return apps.APPS[name](
            binary=args.exerciser_bin,
            numdims=args.exerciser_numdims,
            minels=_ints(args.exerciser_minels),
            dimranks=_ints(args.exerciser_dimranks),
            bufmult=_ints(args.exerciser_bufmult),
            nsizes=args.exerciser_nsizes,
            indepio=args.exerciser_indepio,
            usechunked=args.exerciser_chunked,
            derivedtype=args.exerciser_derivedtype,
            addattr=args.exerciser_addattr,
            keepfile=args.exerciser_keepfile)

    if name == 'macsio':
        return apps.APPS[name](
            binary=args.macsio_bin,
            part_size=args.macsio_part_size,
            num_dumps=args.macsio_num_dumps,
            avg_num_parts=args.macsio_parts_per_rank,
            part_dim=args.macsio_part_dim,
            file_mode=args.macsio_file_mode,
            json_lib_dir=args.macsio_json_lib,
            extra=args.macsio_extra)

    return apps.APPS[name](
        binary=args.h5bench_bin,
        particles=args.h5bench_particles,
        timesteps=args.h5bench_timesteps,
        mem_pattern=args.h5bench_mem_pattern,
        file_pattern=args.h5bench_file_pattern,
        output=args.h5bench_output,
        extra=extra)


def _build_launcher(args):
    name = getattr(args, 'launcher', 'local') or 'local'
    if name == 'local':
        return launchers.LocalLauncher()
    if name != 'pbs':
        raise ValueError('unknown launcher {0!r}'.format(name))
    if not args.pbs_account:
        raise ValueError(
            '--launcher pbs needs --pbs-account: PBS Pro on Nurion rejects a '
            'job with no -A application code. Use "etc" if none of the '
            'software-specific codes apply.')
    return launchers.PBSLauncher(
        queue=args.pbs_queue, account=args.pbs_account,
        select=args.pbs_select, ncpus=args.pbs_ncpus,
        mpiprocs=args.pbs_mpiprocs, ompthreads=args.pbs_ompthreads,
        walltime=args.pbs_walltime,
        modules=args.pbs_modules or (),
        preamble=args.pbs_preamble or (),
        poll_seconds=args.pbs_poll,
        wait_minutes=args.pbs_wait_minutes)


def _build_darshan(args):
    path = getattr(args, 'darshan_lib', None)
    if not path:
        return None
    return darshan.Darshan(library=path, parser=args.darshan_parser,
                           first=args.darshan_first,
                           nonmpi=args.darshan_nonmpi)


def _build_evaluator(args, the_space, the_objective, recorder, replay):
    app = _build_app(args)
    launcher = _build_launcher(args)
    reader = _build_darshan(args)


    app_cmd = args.app_cmd
    if app is not None and not app_cmd:
        app_cmd = app.command(args.mpi_cmd)

    scraper = None
    if args.perf_regex:
        scraper = objective.OutputScraper(args.perf_regex, args.perf_unit)
    elif app is not None and app.perf_regex:
        # Kept even with Darshan on: the scraped figure costs nothing and is
        # what tells you the two disagree.
        scraper = objective.OutputScraper(app.perf_regex, app.perf_unit)
    elif reader is not None:
        pass
    elif app is not None and app.perf_regex is None and not args.dry_run \
            and replay is None:
        raise ValueError(
            '{0} does not print a bandwidth this tuner scrapes, so it needs '
            '--darshan-lib. Without it every candidate scores the penalty '
            'value and the search has nothing to follow.'.format(app.name))
    elif the_objective.sense == objective.SENSE_MAX and not args.dry_run \
            and replay is None:
        print('WARNING: --objective bandwidth without --perf-regex; no '
              'bandwidth can be measured and every candidate will score the '
              'penalty value.', file=sys.stderr)

    extra_env = {}
    if args.ld_preload:
        extra_env['LD_PRELOAD'] = args.ld_preload

    return evaluate.Evaluator(
        space=the_space, objective=the_objective, workdir=args.workdir,
        app_cmd=app_cmd, reps=getattr(args, 'reps', None) or 1,
        timeout=args.timeout, scraper=scraper, cleanup_files=args.cleanup,
        cache=not args.no_cache, recorder=recorder, replay=replay,
        dry_run=args.dry_run, launcher=launcher, app=app,
        extra_env=extra_env, darshan=reader,
        repetition_cost=args.repetition_cost)


def _load_ranking(args, the_space, the_objective):
    """Return (ordered names, {name: impact score} or None).

    The scores are what the RL picker warm-starts from; the order alone is
    enough for the static selector.
    """
    if args.ranking:
        with open(args.ranking) as stream:
            names = [line.strip() for line in stream
                     if line.strip() and not line.startswith('#')]
        # A plain name list carries order but no magnitudes.  Synthesise a
        # linearly decreasing prior so the warm start reproduces the order.
        total = max(1, len(names))
        return names, {name: (total - index) / float(total)
                       for index, name in enumerate(names)}
    if args.ranking_from_trace:
        loaded = trace_module.load(args.ranking_from_trace)
        effects = subset_module.rank_from_trace(the_space, loaded,
                                               the_objective)
        return ([name for name, _, _ in effects],
                {name: effect for name, effect, _ in effects})
    return list(the_space.free_names), None


def _build_subset_selector(args, settings, the_space, the_objective,
                           horizon):
    choice = _resolve(args, settings, 'subset', 'all')
    if choice == 'all':
        return subset_module.AllParameters()

    have_prior = bool(args.ranking or args.ranking_from_trace)
    names, prior = _load_ranking(args, the_space, the_objective)

    if choice == 'static':
        if not have_prior:
            print('WARNING: --subset static without --ranking or '
                  '--ranking-from-trace; falling back to declaration order, '
                  'which is not an impact ranking.', file=sys.stderr)
        return subset_module.StaticRanking(
            the_space, names, size=args.subset_size,
            grow_every=args.subset_grow_every)

    if choice == 'rl':
        if not have_prior:
            print('WARNING: --subset rl without --ranking or '
                  '--ranking-from-trace; the picker starts with no offline '
                  'prior and has to learn the ranking from scratch inside the '
                  'campaign. The paper pretrains on a parameter sweep.',
                  file=sys.stderr)
        return subset_module.build_rl_picker(
            space=the_space, objective=the_objective, horizon=horizon,
            size=args.subset_size, prior=prior,
            size_penalty=args.size_penalty,
            seed=args.rl_seed if args.rl_seed is not None else args.seed)

    raise ValueError('unknown subset policy {0!r}'.format(choice))


def _report_rl(args, stopper, subset_selector):
    """Print what the learned components were trained on.

    Anything reported out of a run needs this: the fallback approximator is
    weaker than the PyTorch one, and a picker with no offline prior is a
    different experiment from one with a sweep behind it.
    """
    if args.quiet:
        return
    report = getattr(stopper, 'training_report', None)
    if report:
        print('  stopper: {0} backend, {1} fitted-Q rounds on {2} curves; '
              'held-out return ratio {3:.3f}, mean stop {4:.1f}/{5}'.format(
                  report['backend'], report['rounds'], report['curves'],
                  report['return_ratio'], report['mean_stop'],
                  report['horizon']))
    warm = getattr(subset_selector, 'warm_report', None)
    if warm:
        print('  picker:  {0} backend, warm-started from the offline ranking; '
              'initial top {1}: {2}'.format(
                  warm['backend'], len(warm['prior_top']),
                  ', '.join(warm['prior_top'])))
    elif getattr(subset_selector, 'name', None) == 'rl':
        print('  picker:  {0} backend, no offline prior'.format(rl.BACKEND))


def _make_reporter(args, campaign, the_space, the_objective):
    if args.quiet:
        return None

    def reporter(event, payload):
        if event == 'baseline':
            result = payload['result']
            print('baseline ({0}): {1:.3f} {2}'.format(
                result.source, result.score, the_objective.unit))
        elif event == 'generation_start':
            mask = payload['mask']
            if mask is not None:
                print('--- generation {0}  tuning: {1}'.format(
                    payload['iteration'], ', '.join(sorted(mask))))
            else:
                print('--- generation {0}'.format(payload['iteration']))
        elif event == 'generation_end':
            generation = payload['generation']
            line = ('    best {0:.3f}  mean {1:.3f}  evaluated {2}  '
                    'budget {3:.1f} min'.format(
                        generation.best, generation.mean,
                        generation.evaluations,
                        generation.tuning_seconds / 60.0))
            roti = campaign.roti(campaign.best_score,
                                 generation.tuning_seconds)
            if roti is not None:
                line += '  RoTI {0:.3f}'.format(roti)
            print(line)
            sys.stdout.flush()
        elif event == 'stopped':
            print('stopped at generation {0}: {1}'.format(
                payload['iteration'], payload['reason']))

    return reporter


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _warn_about_environment(args, evaluator):
    """Complain about the two setups that silently produce a flat landscape."""
    preload = args.ld_preload or os.environ.get('LD_PRELOAD', '')
    if 'autotuner' not in preload:
        print('WARNING: no libautotuner.so in --ld-preload or LD_PRELOAD; the '
              'application will run untuned and every candidate will score '
              'the same.', file=sys.stderr)

    if args.launcher == 'local' and not os.environ.get('PBS_JOBID'):
        print('WARNING: --launcher local outside a PBS job. The application '
              'will run wherever this process is -- on a login node if that '
              'is where you are. Use --launcher pbs, or start this inside '
              'qsub / qsub -I.', file=sys.stderr)

    app = evaluator.app
    if app is not None and args.launcher == 'pbs':
        ranks = args.pbs_select * args.pbs_mpiprocs
        described = app.describe(ranks=ranks)
        if described:
            print('workload: {0}'.format(described))


def command_info(args):
    print('frameworks')
    print(frameworks.describe())
    print()
    print('parameter spaces')
    for name in sorted(space.SPACES):
        the_space = space.get_space(name)
        print('  {0:<12}{1:>3} parameters, {2:>3} free, {3:,} configurations'
              .format(name, len(the_space), len(the_space.free_names),
                      the_space.size()))
        unsupported = the_space.unsupported()
        if unsupported:
            print('  {0:<12}not injectable by the current C shim: {1}'.format(
                '', ', '.join(p.name for p, _ in unsupported)))
    return 0


def command_rank(args):
    the_space = space.get_space(args.space)
    the_objective = objective.get_objective(args.objective)
    loaded = trace_module.load(args.trace)
    effects = subset_module.rank_from_trace(the_space, loaded, the_objective)
    print('trace: {0} configurations'.format(len(loaded)))
    print(subset_module.format_ranking(effects, the_objective))
    if args.out:
        with open(args.out, 'w') as stream:
            stream.write('# impact ranking from {0}\n'.format(args.trace))
            for name, _, _ in effects:
                stream.write(name + '\n')
        print('\nwrote {0}'.format(args.out))
    return 0


def command_sweep(args):
    the_space = _space_for(args, args.space or 'minimal')
    the_objective = objective.get_objective(args.objective or 'bandwidth',
                                            args.reduction or 'best')
    rng = random.Random(args.seed)

    if args.samples:
        points = [the_space.sample(rng) for _ in range(args.samples)]
    else:
        total = the_space.size()
        if total > args.max_points:
            print('grid has {0:,} points, above --max-points {1:,}.\n'
                  'Use --samples N for a random sample or raise --max-points.'
                  .format(total, args.max_points), file=sys.stderr)
            return 2
        points = list(space.grid(the_space))

    recorder = None
    if args.record:
        recorder = trace_module.Recorder(args.record, metadata={
            'space': the_space.name,
            'objective': the_objective.name,
            'parameters': [p.name for p in the_space.parameters],
            'mode': 'sweep',
        })

    replay = trace_module.load(args.replay) if args.replay else None
    if replay is not None:
        replay.check_compatible(the_space)

    evaluator = _build_evaluator(args, the_space, the_objective, recorder,
                                 replay)
    print('sweeping {0} points over space {1} ({2})'.format(
        len(points), the_space.name, evaluator.mode))
    try:
        for index, point in enumerate(points, 1):
            result = evaluator.evaluate(point)
            print('{0:>6}/{1}  {2:.3f} {3}  {4}'.format(
                index, len(points), result.score, the_objective.unit,
                the_space.describe(point)))
            sys.stdout.flush()
    finally:
        if recorder is not None:
            recorder.close()
            print('recorded {0} measurements to {1}'.format(
                recorder.count, args.record))
    return 0


def command_discover(args):
    from . import discovery

    regions = []
    for raw in getattr(args, 'keep_regions', None) or []:
        if ':' not in raw:
            raise ValueError(
                '--keep-region expects START:END, got {0!r}'.format(raw))
        start, end = raw.split(':', 1)
        regions.append((int(start), int(end)))

    try:
        kernel = discovery.discover(
            args.source, workdir=args.workdir,
            io_prefixes=args.io_prefixes,
            clang_args=(['-I' + d for d in args.include_dirs]
                        + list(args.clang_args)),
            keep_regions=regions,
            use_clang_format=not args.no_clang_format,
            allow_parse_errors=args.allow_parse_errors)
    except discovery.DiscoveryError as error:
        # The paper's fallback: "if the I/O kernel of the application causes an
        # error, TunIO will revert to using the full application."  Saying so
        # explicitly matters, because a silent failure here would be tuned
        # against a kernel that is missing I/O.
        print('discovery failed: {0}'.format(error), file=sys.stderr)
        print('\nfall back to tuning the full application: run without '
              '--app-cmd pointing at a kernel.', file=sys.stderr)
        return 3

    source = discovery.reconstruct(kernel)
    source, notes = discovery.apply_reductions(
        kernel, source, keep_iterations=args.loop_keep,
        memory_path=args.memory_path)
    kernel.reductions.extend(str(note) for note in notes)

    print(kernel.report(), file=sys.stderr)
    for name, line in kernel.io_calls[:20]:
        print('  I/O call {0} at line {1}'.format(name, line),
              file=sys.stderr)
    if len(kernel.io_calls) > 20:
        print('  ... and {0} more'.format(len(kernel.io_calls) - 20),
              file=sys.stderr)

    if args.out:
        with open(args.out, 'w') as stream:
            stream.write(source)
        print('wrote {0}'.format(args.out), file=sys.stderr)
        print('\nCompile it and pass the binary as --app-cmd. Verify it '
              'writes what you expect before tuning against it.',
              file=sys.stderr)
    else:
        sys.stdout.write(source)
    return 0


def command_run(args):
    framework_class, settings = frameworks.get(args.framework)

    the_space = _space_for(args, _resolve(args, settings, 'space', 'paper12'))
    the_objective = objective.get_objective(
        _resolve(args, settings, 'objective', 'bandwidth'),
        _resolve(args, settings, 'reduction', 'best'))
    args.reps = _resolve(args, settings, 'reps', 1)

    rng = random.Random(args.seed)

    replay = trace_module.load(args.replay) if args.replay else None
    if replay is not None:
        replay.check_compatible(the_space)

    recorder = None
    if args.record:
        recorder = trace_module.Recorder(args.record, metadata={
            'space': the_space.name,
            'objective': the_objective.name,
            'framework': args.framework,
            'parameters': [p.name for p in the_space.parameters],
            'mode': 'run',
        })

    evaluator = _build_evaluator(args, the_space, the_objective, recorder,
                                 replay)
    campaign = metrics.Campaign(the_objective, args.framework, the_space)

    horizon = _resolve(args, settings, 'generations', 40)
    stopper_name = _resolve(args, settings, 'stopper', 'never')
    if stopper_name == 'rl' and not args.quiet:
        print('training the RL early stopper offline on synthetic curves '
              '(backend: {0})...'.format(rl.BACKEND))
    stopper = early_stopping.build_stopper(
        stopper_name, the_objective, campaign=campaign,
        max_minutes=args.max_minutes, horizon=horizon,
        rl_kwargs={
            'seed': args.rl_seed if args.rl_seed is not None else args.seed,
            'curves': args.rl_curves,
            'rounds': args.rl_rounds,
            'verbose': args.rl_verbose,
        })
    subset_selector = _build_subset_selector(args, settings, the_space,
                                             the_objective, horizon)
    _report_rl(args, stopper, subset_selector)

    if not args.quiet:
        print('{0}: {1}'.format(args.framework, framework_class.description))
        print('space {0} -- {1} parameters, {2} free, {3:,} configurations'
              .format(the_space.name, len(the_space),
                      len(the_space.free_names), the_space.size()))
        print('objective {0} ({1}imise), mode {2}'.format(
            the_objective.name, the_objective.sense, evaluator.mode))
        unsupported = the_space.unsupported(_hdf5_version(args))
        if unsupported and evaluator.mode == 'live':
            print('WARNING: {0} parameter(s) cannot be injected by the current '
                  'C shim and will have no effect:'.format(len(unsupported)),
                  file=sys.stderr)
            for parameter, reason in unsupported:
                print('  {0:<24}{1}'.format(parameter.name, reason),
                      file=sys.stderr)
        if evaluator.mode == 'live':
            _warn_about_environment(args, evaluator)
        print()

    framework = framework_class(
        space=the_space, objective=the_objective, evaluator=evaluator,
        campaign=campaign, rng=rng,
        generations=_resolve(args, settings, 'generations', 40),
        population=_resolve(args, settings, 'population', 15),
        elite=_resolve(args, settings, 'elite', 3),
        mutation_rate=_resolve(args, settings, 'mutation-rate', 0.15),
        crossover_rate=_resolve(args, settings, 'crossover-rate', 0.9),
        scale_mutation=args.scale_mutation,
        stopper=stopper, subset_selector=subset_selector,
        reporter=_make_reporter(args, campaign, the_space, the_objective))

    try:
        framework.run()
    except KeyboardInterrupt:
        if hasattr(evaluator.launcher, 'cancel_all'):
            print('\ninterrupted; deleting submitted jobs', file=sys.stderr)
            evaluator.launcher.cancel_all()
        raise
    finally:
        if recorder is not None:
            recorder.close()

    print()
    print(metrics.format_summary(campaign))
    if evaluator.mode == 'live':
        print()
        print('application launches: {0}'.format(evaluator.launches))
        if args.repetition_cost == 'one' and \
                evaluator.wall_seconds > evaluator.tuning_seconds * 1.05:
            print('charged budget:       {0:.1f} min of {1:.1f} min actually '
                  'spent'.format(evaluator.tuning_seconds / 60.0,
                                 evaluator.wall_seconds / 60.0))
            print('                      (--repetition-cost one: repetitions '
                  'of a candidate are not accumulated, per the paper)')
        if evaluator.darshan is not None:
            failures = len(evaluator.darshan_failures)
            if failures:
                print('WARNING: {0} of {1} candidates had no readable Darshan '
                      'log; those fell back to scraped output or scored the '
                      'penalty. First: {2}'.format(
                          failures, evaluator.evaluations,
                          evaluator.darshan_failures[0][:160]),
                      file=sys.stderr)
            else:
                print('bandwidth source:     Darshan '
                      '({0} candidates)'.format(evaluator.evaluations))

        jobs = getattr(evaluator.launcher, 'jobs', None)
        if jobs is not None:
            print('scheduler jobs:       {0}'.format(len(jobs)))
        if evaluator.queued_seconds > 0:
            # Reported, not charged: RoTI above is computed from execution
            # time only, so it does not move with how busy the machine was.
            print('scheduler queue time: {0:.1f} min (excluded from the '
                  'tuning budget and from RoTI)'.format(
                      evaluator.queued_seconds / 60.0))

    # Leave the winning configuration in place, as evo/evolve.py did.
    if campaign.best_point is not None and evaluator.mode == 'live':
        from . import config_writer
        config_writer.write(the_space, campaign.best_point, args.workdir)

    if args.summary_json:
        with open(args.summary_json, 'w') as stream:
            json.dump(campaign.summary(), stream, indent=2, sort_keys=True,
                      default=str)
        print('wrote {0}'.format(args.summary_json))
    return 0


COMMANDS = {
    'run': command_run,
    'discover': command_discover,
    'sweep': command_sweep,
    'rank': command_rank,
    'info': command_info,
}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'command', None):
        parser.error('a command is required: {0}'.format(
            ', '.join(sorted(COMMANDS))))
    try:
        return COMMANDS[args.command](args)
    except NotImplementedError as error:
        print('not implemented: {0}'.format(error), file=sys.stderr)
        return 3
    except (ValueError, evaluate.TraceMiss) as error:
        print('error: {0}'.format(error), file=sys.stderr)
        return 2
