#!/usr/bin/env python3
#
# Copyright by The HDF Group.
# All rights reserved.
#
# This file is part of h5tuner. The full h5tuner copyright notice,
#  including terms governing use, modification, and redistribution, is
# contained in the file COPYING, which can be found at the root of the
# source code distribution tree.  If you do not have access to this file,
# you may request a copy from help@hdfgroup.org.
#
"""h5evolve -- genetic-algorithm search over H5Tuner I/O parameters.

For each candidate configuration this driver rewrites ./config.xml, relaunches
the target application from scratch, and minimises the application's wall-clock
runtime.  The H5Tuner shim reads ./config.xml once per H5Fcreate() call, so a
configuration only takes effect on a fresh process -- hence one app launch per
candidate, not one long-lived run that re-reads the file.

LD_PRELOAD must already point at libautotuner.so when this script starts;
subprocesses inherit it.

    export LD_PRELOAD=/path/to/H5Tuner/lib/libautotuner.so
    python3 evo/evolve.py --app-cmd 'mpiexec -n 32 ./my_app'

Total app launches are population x generations x reps (default 15 x 40 x 5 =
3000), so size the job accordingly.  Use --dry-run to exercise the search and
the XML writer with a synthetic cost function and no launches at all -- it
needs neither MPI nor HDF5.
"""

import argparse
import datetime
import math
import os
import os.path
import random
import signal
import subprocess
import sys
import time
from xml.dom.minidom import Document

# ---------------------------------------------------------------------------
# Search settings.  Defaults reproduce the original experiment's scale.
# ---------------------------------------------------------------------------

NUM_POP = 15                # ga.setPopulationSize(15)
NUM_GENS = 40               # ga.setGenerations(40)
NUM_ELITE = 3               # ga.setElitismReplacement(3)
REPS = 5                    # measurements per candidate; fitness is min(times)
TIMEOUT_SECONDS = 59 * 60   # per-candidate budget
TIMEOUT_PENALTY = 10000.0   # fitness assigned when a candidate times out

CONFIG_FILENAME = 'config.xml'
DEFAULT_APP_CMD = '$SCRATCH/h5_write'

# Output files left behind by the application, removed before each measurement
# so a candidate never benefits from a previous run's data.
CLEANUP_FILES = ('SDS.h5', 'sample_dataset.h5part', 'vorpalio.h5', 'prs.h5')

# ---------------------------------------------------------------------------
# Tunable parameters.  One allele list per gene; the GA only ever picks values
# from these lists, so every candidate is a legal configuration.
# ---------------------------------------------------------------------------

# Striping.  Default stripe_count is 1; 1 and 4 measured very badly, so they
# are skipped.  -1 means stripe over every available OST.
strp_fac = [4, 8, 16, 24, 32, 48, 64, 96, 128, -1]

# Stripe size must be a multiple of the 64KB page size.  Good sequential-I/O
# values sit between 1MB and 4MB; the hard limits are 512KB and 4GB.
strp_unt = [1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864,
            134217728]

# Collective buffering: number of aggregators.
cb_nds = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256]

# HDF5 alignment, as "threshold, alignment" pairs.
alignment = ["1, 1", "0, 4096", "0, 16384", "0, 65536", "0, 262144",
             "1024, 4096", "1024, 16384", "1024, 65536", "1024, 262144",
             "4096, 16384", "4096, 65536", "4096, 262144", "16384, 65536",
             "16384, 262144"]

# Data sieving buffer size.  Single value, so this gene is effectively a
# constant -- widen the list to search it.
siv_buf_size = [536870912]

ALLELES = (strp_fac, strp_unt, cb_nds, alignment, siv_buf_size)

# Gene order, for readability at the call sites below.
GENE_STRP_FAC, GENE_STRP_UNT, GENE_CB_NDS, GENE_ALIGN, GENE_SIV_BUF = range(5)

# ---------------------------------------------------------------------------
# Genetic algorithm.
#
# Replaces pyevolve (Python 2 only, unmaintained since ~2012) with a stdlib
# implementation of the configuration the original script asked for:
#
#   G1DList over per-gene allele lists         -> genomes are plain tuples
#   G1DListInitializatorAllele                 -> random_genome()
#   G1DListMutatorAllele, rate 0.15 per gene   -> mutate()
#   G1DListCrossoverSinglePoint, rate 0.9      -> single_point()
#     pyevolve's G1DList default -- the original never set a crossover
#     operator, so it inherited both the operator and GSimpleGA's 0.9 rate.
#   GRouletteWheel                             -> roulette()
#   setElitism(True) + setElitismReplacement(3)-> evolve()
#   setMinimax(minimize)                       -> lower score wins throughout
#
# pyevolve fed roulette through its linear-scaling scheme (multiplier 1.2).
# roulette() reproduces that selection ratio directly rather than cloning
# pyevolve's internal scaling, so scores will not match the original runs
# number-for-number.
# ---------------------------------------------------------------------------

CROSSOVER_RATE = 0.9        # pyevolve Consts.CDefGACrossoverRate
MUTATION_RATE = 0.15        # ga.setMutationRate(0.15)
SELECTION_PRESSURE = 1.2    # pyevolve Consts.CDefScaleLinearMultiplier


def random_genome(alleles, rng):
    return tuple(rng.choice(a) for a in alleles)


def mutate(genome, alleles, rng, rate=MUTATION_RATE):
    """Replace each gene with another of its alleles with probability `rate`."""
    genes = list(genome)
    for i, allele in enumerate(alleles):
        if rng.random() < rate:
            genes[i] = rng.choice(allele)
    return tuple(genes)


def single_point(mom, dad, rng):
    """Single-point crossover; returns both children."""
    cut = rng.randint(1, len(mom) - 1)
    return mom[:cut] + dad[cut:], dad[:cut] + mom[cut:]


def roulette(scored, rng):
    """Roulette-wheel pick from [(genome, score), ...].  Lower score is better.

    Weights are linearly scaled so the best individual's share is
    SELECTION_PRESSURE times the worst individual's, which keeps the worst
    candidate reachable instead of zeroing it out.
    """
    best = min(score for _, score in scored)
    worst = max(score for _, score in scored)
    spread = worst - best
    if spread <= 0.0:                       # whole population tied
        return rng.choice(scored)[0]
    floor = spread / (SELECTION_PRESSURE - 1.0)
    weights = [(worst - score) + floor for _, score in scored]
    target = rng.uniform(0.0, sum(weights))
    running = 0.0
    for (genome, _), weight in zip(scored, weights):
        running += weight
        if running >= target:
            return genome
    return scored[-1][0]                    # float rounding fallback


def next_population(scored, alleles, pop_size, rng):
    children = []
    while len(children) + 1 < pop_size:
        mom = roulette(scored, rng)
        dad = roulette(scored, rng)
        if rng.random() < CROSSOVER_RATE:
            sister, brother = single_point(mom, dad, rng)
        else:
            sister, brother = mom, dad
        children.append(mutate(sister, alleles, rng))
        children.append(mutate(brother, alleles, rng))
    if len(children) < pop_size:            # odd population size
        children.append(mutate(roulette(scored, rng), alleles, rng))
    return children[:pop_size]


def evolve(alleles, evaluate, pop_size, generations, num_elite, rng,
           on_generation=None):
    """Run the search; return the best (genome, score) found."""
    population = [random_genome(alleles, rng) for _ in range(pop_size)]
    scored = []
    for generation in range(generations):
        previous, scored = scored, []
        for genome in population:
            scored.append((genome, evaluate(genome, generation)))
        scored.sort(key=lambda pair: pair[1])
        # Elitism: carry the previous generation's best individuals over this
        # generation's worst, but only where they are actually better.
        for i in range(min(num_elite, len(previous))):
            if previous[i][1] < scored[-1][1]:
                scored[-1] = previous[i]
                scored.sort(key=lambda pair: pair[1])
        if on_generation is not None:
            on_generation(generation, scored)
        if generation < generations - 1:
            population = next_population(scored, alleles, pop_size, rng)
    return scored[0]


# ---------------------------------------------------------------------------
# Candidate -> config.xml
# ---------------------------------------------------------------------------

def decode(genome):
    """Turn a genome into the parameter values written to config.xml."""
    threshold, align = (int(part.strip())
                        for part in genome[GENE_ALIGN].split(','))
    return {
        'striping_factor': genome[GENE_STRP_FAC],
        'striping_unit': genome[GENE_STRP_UNT],
        'cb_nodes': genome[GENE_CB_NDS],
        # Tied to striping_unit rather than searched independently, preserved
        # from the original ("Ruth(David's) Suggestion").
        'cb_buffer_size': genome[GENE_STRP_UNT],
        'align_threshold': threshold,
        'alignment': align,
        'sieve_buf_size': genome[GENE_SIV_BUF],
    }


def build_config_xml(params):
    """Render params as an H5Tuner config document.

    The section elements are documentation only: H5Tuner searches the whole
    tree by tag name and ignores this nesting.
    """
    doc = Document()
    root = doc.createElement('Parameters')
    doc.appendChild(root)

    def section(name):
        node = doc.createElement(name)
        root.appendChild(node)
        return node

    def setting(parent, name, value):
        node = doc.createElement(name)
        parent.appendChild(node)
        node.appendChild(doc.createTextNode(str(value)))

    high = section('High_Level_IO_Library')
    setting(high, 'alignment',
            '{0},{1}'.format(params['align_threshold'], params['alignment']))
    setting(high, 'sieve_buf_size', params['sieve_buf_size'])

    middle = section('Middleware_Layer')
    setting(middle, 'cb_nodes', params['cb_nodes'])
    setting(middle, 'cb_buffer_size', params['cb_buffer_size'])

    low = section('Parallel_File_System')
    setting(low, 'striping_factor', params['striping_factor'])
    setting(low, 'striping_unit', params['striping_unit'])

    return doc.toprettyxml(indent='  ')


def write_config(params, directory):
    """Write config.xml where the application will look for it.

    H5Tuner opens a bare relative "config.xml", so it must land in the working
    directory the application inherits -- ours.
    """
    path = os.path.join(directory, CONFIG_FILENAME)
    with open(path, 'w') as handle:
        handle.write(build_config_xml(params))
    return path


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

class Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise Timeout


def cleanup_outputs(directory):
    for name in CLEANUP_FILES:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            os.remove(path)


def measure(app_cmd, reps, timeout_seconds):
    """Launch the application `reps` times.

    Returns (times, ok, output).  `times` is None if the candidate exceeded
    timeout_seconds; `ok` is True when every launch exited zero.
    """
    times = []
    ok = True
    output = ''
    running = None
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout_seconds)
    try:
        for _ in range(reps):
            start = time.time()
            running = subprocess.Popen(app_cmd, shell=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
            raw, _unused = running.communicate()
            times.append(time.time() - start)
            output = raw.decode('utf-8', 'replace')
            if running.returncode != 0:
                ok = False
            running = None
    except Timeout:
        if running is not None and running.poll() is None:
            running.kill()
            running.wait()
        return None, False, output
    finally:
        signal.alarm(0)
    return times, ok, output


def synthetic_cost(params):
    """A fake, deterministic runtime for --dry-run.

    This is a smooth landscape with one broad optimum so the search visibly
    converges without launching anything.  It is not a performance model and
    says nothing about real I/O behaviour.
    """
    def distance(value, ideal):
        return abs(math.log2(value / float(ideal)))

    stripes = params['striping_factor']
    if stripes < 0:                         # -1 means "all OSTs"
        stripes = 64
    return (60.0
            + 4.0 * distance(stripes, 32)
            + 3.0 * distance(params['striping_unit'], 4194304)
            + 2.0 * distance(params['cb_nodes'], 32)
            + 1.5 * distance(max(params['alignment'], 1), 65536))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--app-cmd', default=os.environ.get(
        'H5TUNER_APP_CMD', DEFAULT_APP_CMD),
        help='shell command that runs the application under test '
             '(default: %(default)s)')
    parser.add_argument('--population', type=int, default=NUM_POP,
                        help='individuals per generation (default: %(default)s)')
    parser.add_argument('--generations', type=int, default=NUM_GENS,
                        help='number of generations (default: %(default)s)')
    parser.add_argument('--elite', type=int, default=NUM_ELITE,
                        help='individuals carried over unchanged '
                             '(default: %(default)s)')
    parser.add_argument('--reps', type=int, default=REPS,
                        help='measurements per candidate; fitness is the '
                             'fastest (default: %(default)s)')
    parser.add_argument('--timeout', type=int, default=TIMEOUT_SECONDS,
                        help='per-candidate budget in seconds '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed the RNG for a reproducible search')
    parser.add_argument('--no-cache', action='store_true',
                        help='re-measure repeated configurations instead of '
                             'reusing their recorded time')
    parser.add_argument('--dry-run', action='store_true',
                        help='exercise the search and the XML writer with a '
                             'synthetic cost function, launching nothing')
    return parser.parse_args(argv)


def run_main(argv=None):
    args = parse_args(argv)
    workdir = os.getcwd()
    scratch = os.environ.setdefault('SCRATCH', workdir)

    if not args.dry_run:
        preload = os.environ.get('LD_PRELOAD', '')
        if 'autotuner' not in preload:
            print('WARNING: LD_PRELOAD does not mention autotuner '
                  '({0!r}).'.format(preload), file=sys.stderr)
            print('         The application will run untuned and every '
                  'candidate will score the same.', file=sys.stderr)

    total = args.population * args.generations * args.reps
    print('h5evolve: {0} generations x {1} individuals x {2} reps '
          '= up to {3} app launches'.format(
              args.generations, args.population, args.reps, total))
    print('h5evolve: config written to {0}'.format(
        os.path.join(workdir, CONFIG_FILENAME)))
    if args.dry_run:
        print('h5evolve: DRY RUN -- synthetic cost, no application launched')
    else:
        print('h5evolve: app command: {0}'.format(args.app_cmd))
        print('h5evolve: SCRATCH={0}'.format(scratch))

    rng = random.Random(args.seed)
    cache = {}
    launches = 0

    result_output = open('./result_output.txt', 'w')
    config_feat_file = open('./config_feat.txt', 'w')
    running_time_file = open('./running_time.txt', 'w')

    def evaluate(genome, generation):
        nonlocal launches
        params = decode(genome)
        summary = ('{striping_factor}, {striping_unit}, {cb_nodes}, '
                   '{cb_buffer_size}, {align_threshold}, {alignment}, '
                   '{sieve_buf_size}').format(**params)

        if not args.no_cache and genome in cache:
            return cache[genome]

        write_config(params, workdir)
        print('Evaluating ({0})'.format(summary))
        sys.stdout.flush()

        stamp = datetime.datetime.now()
        if args.dry_run:
            elapsed = synthetic_cost(params)
            ok = True
        else:
            cleanup_outputs(scratch)
            times, ok, output = measure(args.app_cmd, args.reps, args.timeout)
            launches += args.reps
            if times is None:
                print('  timed out after {0}s, penalising'.format(args.timeout))
                elapsed = TIMEOUT_PENALTY
            else:
                elapsed = min(times)
                print('  times: {0}  -> {1:.3f}s'.format(
                    ', '.join('{0:.3f}'.format(t) for t in times), elapsed))
                if output.strip():
                    print(output.rstrip())
            if not ok:
                print('  WARNING: application exited non-zero')

        record = '{0}: {1}: {2}'.format(generation, summary, elapsed)
        config_feat_file.write(summary + '\n')
        result_output.write(record + '\n')
        running_time_file.write('[{0}] {1}={2}\n'.format(
            stamp, record, 1 if ok else 0))
        for handle in (config_feat_file, result_output, running_time_file):
            handle.flush()
        sys.stdout.flush()

        cache[genome] = elapsed
        return elapsed

    def report(generation, scored):
        best_genome, best_score = scored[0]
        average = sum(score for _, score in scored) / len(scored)
        print('--- generation {0}: best {1:.3f}  avg {2:.3f}  '
              'evaluated {3} configs'.format(
                  generation, best_score, average, len(cache)))
        print('    best config: {0}'.format(
            ', '.join('{0}={1}'.format(k, v)
                      for k, v in sorted(decode(best_genome).items()))))
        sys.stdout.flush()

    try:
        best_genome, best_score = evolve(
            ALLELES, evaluate,
            pop_size=args.population,
            generations=args.generations,
            num_elite=args.elite,
            rng=rng,
            on_generation=report)
    finally:
        for handle in (result_output, config_feat_file, running_time_file):
            handle.close()

    print('')
    print('Best solution: {0:.3f}s'.format(best_score))
    for key, value in sorted(decode(best_genome).items()):
        print('  {0} = {1}'.format(key, value))
    print('Distinct configurations evaluated: {0}'.format(len(cache)))
    if not args.dry_run:
        print('Application launches: {0}'.format(launches))

    # Leave the winning configuration in place for the next run.
    write_config(decode(best_genome), workdir)
    return 0


if __name__ == '__main__':
    sys.exit(run_main())
