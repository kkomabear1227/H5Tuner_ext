"""Launch the application and measure one candidate configuration.

Every framework goes through this class, which is the point: if H5Tuner and
TunIO measured candidates differently their scores would not be comparable.

Four modes, in order of increasing detachment from reality:

    live      write config.xml, run the application, measure it
    replay    look the candidate up in a recorded trace; run nothing
    dry-run   score the candidate with a synthetic model; run nothing
    cached    the candidate was already scored in this session

Replay is what makes the reinforcement-learning work tractable.  A TunIO agent
needs offline pretraining, and running the target application for every
training episode is not affordable.  Record a sweep once on the cluster, then
train and compare search algorithms against that trace on a laptop.

A live evaluation delegates the running to a launchers.Launcher, so the same
campaign can execute candidates as local mpirun steps inside an allocation or
as one PBS job each from a login node.  Which one is in use changes the
wall-clock enormously and the measurement not at all, which is the point.
"""

import math
import os
import subprocess
import time

from . import config_writer
from . import darshan as darshan_module
from . import launchers as launchers_module
from .objective import Measurement

# Output files the applications in this repository leave behind.  Removed
# before each repetition so a candidate never benefits from a previous run's
# data.  evo/evolve.py meant to do this but passed a list to Popen with
# shell=True, so only the first element ran and the cleanup never happened.
DEFAULT_CLEANUP_FILES = ('SDS.h5', 'sample_dataset.h5part', 'vorpalio.h5',
                         'prs.h5')


class EvalResult:
    """One scored candidate."""

    __slots__ = ('point', 'score', 'measurement', 'source', 'wall_cost')

    def __init__(self, point, score, measurement, source, wall_cost):
        self.point = point
        self.score = score
        self.measurement = measurement
        # 'live' | 'replay' | 'dry-run' | 'cache'
        self.source = source
        # Seconds of tuning budget this evaluation consumed.  Zero for cache
        # hits; the recorded cost for replays, so RoTI stays meaningful.
        self.wall_cost = wall_cost


class TraceMiss(Exception):
    """Replay was asked for a candidate the trace does not contain."""


class Evaluator:
    """Scores candidates and keeps the books.

    space       the parameter space being searched
    objective   turns a Measurement into a score
    workdir     where config.xml goes; the application inherits this cwd
    app_cmd     shell command that runs the application
    reps        repetitions per candidate
    timeout     per-candidate budget in seconds, across all repetitions
    scraper     optional OutputScraper for bandwidth
    recorder    optional trace.Recorder; every live measurement is written
    replay      optional trace.Trace; when set nothing is launched
    dry_run     score with the synthetic model instead of launching
    launcher    optional launchers.Launcher; defaults to running locally
    app         optional apps.Application that writes the application's own
                input file and knows what to clean up between candidates
    extra_env   environment given to the application on top of this process's.
                LD_PRELOAD belongs here rather than in the tuner's own
                environment: exporting the shim in the shell that runs the
                tuner would also load it into python, and into qsub.
    darshan     optional darshan.Darshan.  When set, bandwidth comes from the
                Darshan log rather than from the application's own output,
                which is what the paper measures with.
    repetition_cost
                'one' charges the tuning budget for a single run of a
                candidate however many repetitions were measured, which is the
                paper's accounting; 'all' charges every repetition.  See
                _charge.
    """

    def __init__(self, space, objective, workdir, app_cmd=None, reps=1,
                 timeout=None, scraper=None, cleanup_files=None, cache=True,
                 recorder=None, replay=None, dry_run=False,
                 synthetic=None, on_evaluated=None, launcher=None, app=None,
                 extra_env=None, darshan=None, repetition_cost='one'):
        self.space = space
        self.objective = objective
        self.workdir = workdir
        self.app_cmd = app_cmd
        self.reps = reps
        self.timeout = timeout
        self.scraper = scraper
        self.cleanup_files = (DEFAULT_CLEANUP_FILES if cleanup_files is None
                              else tuple(cleanup_files))
        self.use_cache = cache
        self.recorder = recorder
        self.replay = replay
        self.dry_run = dry_run
        self.synthetic = synthetic or SyntheticApp(space, objective)
        self.on_evaluated = on_evaluated
        self.launcher = launcher or launchers_module.LocalLauncher()
        self.app = app
        self.extra_env = dict(extra_env or {})
        self.darshan = darshan
        # Candidates whose Darshan log could not be read.  Counted rather than
        # raised: one unreadable log should not end a campaign, but a campaign
        # where most of them failed is not a measurement of anything.
        self.darshan_failures = []
        if repetition_cost not in ('one', 'all'):
            raise ValueError(
                "repetition_cost must be 'one' or 'all', not {0!r}".format(
                    repetition_cost))
        self.repetition_cost = repetition_cost

        self._cache = {}
        self._candidates = 0
        self.launches = 0
        self.evaluations = 0
        # Budget charged, per the repetition_cost policy.  RoTI divides by it.
        self.tuning_seconds = 0.0
        # Everything the campaign really spent running candidates.  Reported
        # alongside, so choosing the paper's accounting does not hide the bill.
        self.wall_seconds = 0.0
        # Scheduler queue time, accumulated but deliberately kept out of
        # tuning_seconds: see launchers.PBSLauncher on why RoTI must not
        # depend on how busy the machine was.
        self.queued_seconds = 0.0

        if not dry_run and replay is None and not app_cmd:
            raise ValueError('app_cmd is required for live evaluation')

    # -- public API ---------------------------------------------------------

    @property
    def mode(self):
        if self.dry_run:
            return 'dry-run'
        if self.replay is not None:
            return 'replay'
        return 'live'

    def evaluate(self, point):
        key = self.space.canonical(point)

        if self.use_cache and key in self._cache:
            cached = self._cache[key]
            result = EvalResult(point, cached.score, cached.measurement,
                                'cache', 0.0)
            self._announce(result)
            return result

        started = time.time()
        if self.dry_run:
            measurement = self.synthetic.measure(point)
            source = 'dry-run'
            cost = self.synthetic.cost_seconds(measurement)
            wall = cost
        elif self.replay is not None:
            measurement, cost = self._from_trace(key)
            source = 'replay'
            wall = cost
        else:
            measurement = self._launch(point)
            source = 'live'
            wall = time.time() - started
            cost = self._charge(measurement, wall)

        score = self.objective.score(measurement)
        self.evaluations += 1
        self.tuning_seconds += cost
        self.wall_seconds += wall

        result = EvalResult(point, score, measurement, source, cost)
        self._cache[key] = result

        if self.recorder is not None and source in ('live', 'dry-run'):
            self.recorder.add(self.space, point, score, measurement, cost)

        self._announce(result)
        return result

    def _charge(self, measurement, wall):
        """Seconds of tuning budget one candidate costs.

        The paper charges a single run.  IV Methodology: "each application run
        is performed 3 times and bandwidths are averaged.  The time cost of
        running the application is not accumulated across runs, because
        different systems have different volatility, and the extra runs can be
        seen as a necessary expense for a given platform."

        This is not a detail.  RoTI divides a bandwidth gain by tuning time, so
        charging all repetitions divides by three for TunIO and by five for
        H5Tuner -- and the two frameworks pick different repetition counts.
        Charging everything would make H5Tuner look worse than TunIO for a
        reason that has nothing to do with either algorithm, which is precisely
        the confound this codebase exists to avoid.

        'all' is kept for the honest-bookkeeping view.  wall_seconds records it
        either way.
        """
        if self.repetition_cost == 'all':
            return wall
        if not measurement.elapsed:
            return wall
        # The mean of the repetitions is what one run of this candidate costs.
        # Scaled up by the launch overhead the repetitions shared, so a
        # candidate is not charged less than it could possibly have taken.
        per_run = sum(measurement.elapsed) / float(len(measurement.elapsed))
        overhead = max(0.0, wall - sum(measurement.elapsed))
        return per_run + overhead / float(len(measurement.elapsed))

    def evaluate_many(self, points, tag='batch'):
        """Evaluate several candidates, returning one EvalResult each in order.

        Where the launcher can batch, every candidate that still needs running
        goes into one submission.  This is what turns a 40-generation campaign
        from 600 queued jobs into 40 -- see launchers.PBSLauncher.batch_script.
        Candidates already in the cache never reach the launcher, so a
        generation of repeats submits nothing at all.
        """
        batching = (getattr(self.launcher, 'batches_candidates', False)
                    and not self.dry_run and self.replay is None
                    and len(points) > 1)
        if not batching:
            return [self.evaluate(point) for point in points]

        results = [None] * len(points)
        pending = []
        for index, point in enumerate(points):
            key = self.space.canonical(point)
            if self.use_cache and key in self._cache:
                cached = self._cache[key]
                results[index] = EvalResult(point, cached.score,
                                            cached.measurement, 'cache', 0.0)
                self._announce(results[index])
            else:
                pending.append((index, point, key))

        if not pending:
            return results

        parent = self.launcher.prepare_batch(self.workdir, tag)
        staged = []
        for index, point, key in pending:
            rundir, spec = self._stage(point, parent=parent)
            staged.append((index, point, key, rundir, spec))

        started = time.time()
        launched = self.launcher.launch_many(
            [spec for _i, _p, _k, _r, spec in staged],
            timeout=self.timeout, tag=tag)
        batch_wall = time.time() - started

        for (index, point, key, rundir, _spec), result in zip(staged,
                                                              launched):
            self.launches += len(result.repetitions or [1])
            self.queued_seconds += result.queued_seconds
            if result.timed_out:
                measurement = Measurement(output=result.output, timed_out=True)
            else:
                samples = result.repetitions or [
                    (result.returncode, result.elapsed, result.output)]
                measurement = self._combine(samples, rundir=rundir)

            score = self.objective.score(measurement)
            self.evaluations += 1
            # Each candidate is charged for its own execution.  The queue wait
            # the batch shared is not charged at all, as with a single job.
            cost = self._charge(measurement, sum(measurement.elapsed)
                                if measurement.elapsed else 0.0)
            self.tuning_seconds += cost
            self.wall_seconds += (sum(measurement.elapsed)
                                  if measurement.elapsed else 0.0)

            evaluated = EvalResult(point, score, measurement, 'live', cost)
            self._cache[key] = evaluated
            if self.recorder is not None:
                self.recorder.add(self.space, point, score, measurement, cost)
            self._announce(evaluated)
            results[index] = evaluated

        del batch_wall
        return results

    def baseline(self):
        """Score the default configuration.

        RoTI needs perf_achieved(0), the performance before any tuning.  The
        cost of this evaluation is bookkept like any other, because it is a
        real charge against the tuning budget.
        """
        return self.evaluate(self.space.default_point())

    # -- internals ----------------------------------------------------------

    def _announce(self, result):
        if self.on_evaluated is not None:
            self.on_evaluated(result)

    def _from_trace(self, key):
        entry = self.replay.lookup(self.space, key)
        if entry is None:
            raise TraceMiss(
                'no recorded measurement for {0}\n'
                '  the trace covers {1} configurations; restrict the search '
                'to the recorded grid or record a wider sweep'.format(
                    self.space.describe(key), len(self.replay)))
        return entry.measurement, entry.wall_cost

    def _cleanup(self, directory):
        for name in self.cleanup_files:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _stage(self, point, parent=None):
        """Write a candidate's inputs and return (rundir, LaunchSpec).

        Split out of _launch so that a batch of candidates can be staged first
        and submitted together.
        """
        self._candidates += 1
        tag = 'cand-{0:06d}'.format(self._candidates)
        preamble = self.app.preamble() if self.app is not None else []

        if parent is not None:
            rundir = os.path.join(parent, tag)
            if not os.path.isdir(rundir):
                os.makedirs(rundir)
            for name in os.listdir(rundir):
                if name in (launchers_module.SENTINEL,
                            launchers_module.SENTINEL_PARTIAL) or (
                        name.startswith('run.') and name.endswith('.out')):
                    os.remove(os.path.join(rundir, name))
        else:
            rundir = self.launcher.prepare(self.workdir, tag)

        config_path = config_writer.write(self.space, point, rundir)
        if self.app is not None:
            self.app.prepare(rundir)
        self._cleanup(rundir)

        env = dict(self.extra_env)
        env['AT_CONFIG_FILE'] = os.path.abspath(config_path)
        if self.darshan is not None:
            self.darshan.prepare(rundir)
            env.update(self.darshan.environment(
                rundir, preload=env.get('LD_PRELOAD')))

        return rundir, launchers_module.LaunchSpec(
            rundir=rundir, command=self.app_cmd, env=env,
            preamble=preamble, reps=self.reps)

    def _launch(self, point):
        """Run one candidate `reps` times and fold the results into one."""
        self._candidates += 1
        tag = 'cand-{0:06d}'.format(self._candidates)
        preamble = self.app.preamble() if self.app is not None else []

        rundir = self.launcher.prepare(self.workdir, tag)
        config_path = config_writer.write(self.space, point, rundir)
        if self.app is not None:
            self.app.prepare(rundir)

        # The shim falls back to a bare relative "config.xml" in the process
        # working directory.  Pointing AT_CONFIG_FILE at an absolute path
        # removes the dependence on where the scheduler decided to start the
        # job, and on every rank agreeing about it.
        env = dict(os.environ)
        env['AT_CONFIG_FILE'] = os.path.abspath(config_path)
        env.update(self.extra_env)

        if self.darshan is not None:
            self.darshan.prepare(rundir)
            env.update(self.darshan.environment(
                rundir, preload=env.get('LD_PRELOAD')))

        deadline = None if self.timeout is None else time.time() + self.timeout
        samples = []

        if self.launcher.batches_reps:
            # One launch covers every repetition.  Under PBS this is the
            # difference between one queue wait per candidate and one per
            # repetition -- five of them, with H5Tuner's own settings.
            self._cleanup(rundir)
            result = self.launcher.launch(rundir, self.app_cmd, env,
                                          self.timeout, preamble, self.reps)
            self.launches += len(result.repetitions or [1])
            self.queued_seconds += result.queued_seconds
            if result.timed_out:
                return Measurement(output=result.output, timed_out=True)
            samples = result.repetitions or [
                (result.returncode, result.elapsed, result.output)]
        else:
            for _ in range(self.reps):
                self._cleanup(rundir)

                remaining = None
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return self._combine(samples, timed_out=True,
                                             rundir=rundir)

                result = self.launcher.launch(rundir, self.app_cmd, env,
                                              remaining, preamble)
                self.launches += 1
                self.queued_seconds += result.queued_seconds
                if result.timed_out:
                    return self._combine(samples, timed_out=True,
                                         output=result.output,
                                         rundir=rundir)
                samples.append((result.returncode, result.elapsed,
                                result.output))

        return self._combine(samples, rundir=rundir)

    def _read_darshan(self, rundir):
        """Read this candidate's Darshan log, or None."""
        if self.darshan is None or rundir is None:
            return None
        try:
            return self.darshan.measure(rundir)
        except darshan_module.DarshanError as error:
            self.darshan_failures.append(str(error))
            return None

    def _combine(self, samples, timed_out=False, output=None, rundir=None):
        """Fold per-repetition results into a single Measurement.

        Bandwidth is scraped once per repetition and then reduced the same way
        wall-clock is.  It has to be done here because Measurement carries
        bandwidth as one field, not a list: before this, a campaign with
        reps=5 kept only the last repetition's bandwidth and --reduction mean
        averaged nothing.  TunIO's three-runs-per-candidate setting depends on
        the fold actually happening.
        """
        elapsed = []
        returncodes = []
        writes = []
        reads = []
        last_output = output or ''
        bytes_written = None
        bytes_read = None

        darshan_reading = self._read_darshan(rundir)

        for returncode, seconds, text in samples:
            one = Measurement(elapsed=[seconds], returncodes=[returncode],
                              output=text)
            if self.scraper is not None:
                self.scraper.apply(one)
            if darshan_reading is not None:
                # Darshan wins over the scraper.  One log covers the whole
                # candidate, repetitions included, so every repetition is
                # given the same figure rather than the reading being folded
                # as if it were per-repetition evidence.
                one.bw_write = darshan_reading.get('bw_write')
                one.bw_read = darshan_reading.get('bw_read')
                if darshan_reading.get('bytes_written') is not None:
                    one.bytes_written = darshan_reading['bytes_written']
                if darshan_reading.get('bytes_read') is not None:
                    one.bytes_read = darshan_reading['bytes_read']
            elapsed.append(seconds)
            returncodes.append(returncode)
            if one.bw_write is not None:
                writes.append(one.bw_write)
            if one.bw_read is not None:
                reads.append(one.bw_read)
            if one.bytes_written is not None:
                bytes_written = one.bytes_written
            if one.bytes_read is not None:
                bytes_read = one.bytes_read
            if output is None:
                last_output = text

        reduce = self.objective.reduce_samples
        return Measurement(
            elapsed=elapsed, returncodes=returncodes, output=last_output,
            bw_write=reduce(writes) if writes else None,
            bw_read=reduce(reads) if reads else None,
            bytes_written=bytes_written, bytes_read=bytes_read,
            timed_out=timed_out)


# ---------------------------------------------------------------------------
# Synthetic application, for --dry-run
# ---------------------------------------------------------------------------

class SyntheticApp:
    """A deterministic stand-in for the application.

    This exists to exercise the search, the config writer and the stopping
    logic without MPI, HDF5 or a filesystem.  It is NOT a performance model and
    says nothing about real I/O behaviour: it is a smooth landscape with one
    broad optimum, so a working search visibly converges on it.

    Parameters the model has no opinion about are ignored, which means a
    dry-run cannot tell you whether tuning them matters.
    """

    # Where the synthetic optimum sits, and how sharply each parameter matters.
    IDEAL = {
        'striping_factor': (32, 4.0),
        'striping_unit': (4 * 1024 * 1024, 3.0),
        'cb_nodes': (32, 2.0),
        'cb_buffer_size': (16 * 1024 * 1024, 1.5),
        'alignment': (64 * 1024, 1.5),
        'sieve_buf_size': (256 * 1024, 1.0),
        'meta_block_size': (256 * 1024, 1.0),
        'chunk_cache': (16 * 1024 * 1024, 0.8),
    }
    # Flat preferences for the categorical knobs.
    PREFERRED = {
        'coll_metadata_write': (1, 2.0),
        'col_meta_ops': (1, 1.5),
        'mdc_conf': ('aggressive', 1.0),
    }

    PEAK_BANDWIDTH_MB = 4000.0
    BYTES_WRITTEN = 64 * 1024 * 1024 * 1024
    COMPUTE_SECONDS = 12.0

    def __init__(self, space, objective):
        self.space = space
        self.objective = objective

    @staticmethod
    def _numeric(name, value):
        """Coerce a parameter value to a number the model can reason about.

        Most values are already integers.  The `original` space keeps HDF5
        alignment as a "threshold,alignment" string, so the alignment field is
        taken from it.  Anything that will not coerce is skipped.
        """
        if isinstance(value, str):
            tail = value.split(',')[-1].strip()
            try:
                value = float(tail)
            except ValueError:
                return None
        if not isinstance(value, (int, float)):
            return None
        if name == 'striping_factor' and value < 0:
            return 64.0             # -1 means "every OST"
        return float(value) if value > 0 else 1.0

    def _distance(self, values):
        total = 0.0
        for name, (ideal, weight) in self.IDEAL.items():
            if name not in values:
                continue
            value = self._numeric(name, values[name])
            if value is None:
                continue
            total += weight * abs(math.log2(value / float(ideal)))
        for name, (preferred, weight) in self.PREFERRED.items():
            if name in values and values[name] != preferred:
                total += weight
        return total

    def measure(self, point):
        values = self.space.decode(point)
        # Bandwidth decays smoothly away from the optimum and never reaches
        # zero, so selection always has a gradient to follow.
        fraction = 1.0 / (1.0 + 0.18 * self._distance(values))
        bw_write = self.PEAK_BANDWIDTH_MB * fraction
        elapsed = self.COMPUTE_SECONDS + (
            self.BYTES_WRITTEN / (1024.0 * 1024.0)) / bw_write
        return Measurement(elapsed=[elapsed], returncodes=[0], output='',
                           bw_write=bw_write, bytes_written=self.BYTES_WRITTEN,
                           bytes_read=0)

    def cost_seconds(self, measurement):
        """Tuning budget a real run of this candidate would have consumed.

        Dry runs launch nothing, but RoTI is a per-minute figure, so charging
        the modelled runtime keeps the reported curves the right shape.
        """
        return sum(measurement.elapsed) if measurement.elapsed else 0.0
