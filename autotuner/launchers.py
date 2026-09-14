"""Where a candidate actually runs.

The evaluator does not care whether a measurement came from a local process or
from a batch job that sat in a queue for two hours, but the mechanics differ
enough that they do not belong in evaluate.py.  A launcher is handed a command
and gives back what the run printed and what it cost.

    local   fork the command here and wait.  This is what you want inside an
            interactive allocation (qsub -I) or inside a batch script that
            already holds the nodes: one candidate is one mpirun.

    pbs     write a PBS Pro job script per candidate, qsub it, wait for it.
            The tuner itself lives on the login node.  Every candidate pays
            the queue wait, so this is for campaigns too long to fit in a
            single allocation.

Both give each candidate its own directory under `.autotuner-jobs/`, holding
the config.xml it was given, the job script, and the captured output.  A run
that goes wrong can then be opened and read after the fact, which matters most
for the runs you cannot reproduce interactively.

Completion is detected by a sentinel file the job script writes as its last
act, not by polling qstat for a terminal state.  Two reasons.  PBS copies the
job's own stdout back to the submission directory only after the script exits,
so a job seen as finished may not have its output on disk yet; and a job that
never starts, is held, or is deleted by the administrator leaves qstat saying
different things on different sites.  A sentinel that carries the exit status
and the measured runtime is unambiguous, and qstat is consulted only to notice
that a job has vanished without writing one.
"""

import os
import shlex
import subprocess
import time

# Written by the job script once the application has exited, one line per
# repetition: "<index> <rc> <seconds>".
SENTINEL = '.autotuner-done'
# Appended to while the job runs, then renamed onto SENTINEL, so a poller
# never reads a partially written result.
SENTINEL_PARTIAL = '.autotuner-done.partial'
# Where each repetition's stdout+stderr is redirected, by the job script itself
# rather than by PBS, so that it is complete when SENTINEL appears.
RUN_OUTPUT = 'run.{0}.out'
FIRST_OUTPUT = RUN_OUTPUT.format(1)
JOB_SCRIPT = 'job.sh'
JOBS_DIR = '.autotuner-jobs'
# Written by a batch job once every candidate in it has finished.
BATCH_SENTINEL = '.autotuner-batch-done'

# Grace period after a job leaves the queue without a sentinel.  A shared
# filesystem can lag behind the scheduler by a few seconds.
VANISHED_GRACE_SECONDS = 30.0


class LaunchResult:
    """One execution of the application."""

    __slots__ = ('output', 'returncode', 'elapsed', 'timed_out',
                 'queued_seconds', 'repetitions')

    def __init__(self, output='', returncode=0, elapsed=0.0, timed_out=False,
                 queued_seconds=0.0, repetitions=None):
        self.output = output
        self.returncode = returncode
        # Seconds the application ran.  For PBS this excludes the queue wait,
        # which is measured separately: see queued_seconds.
        self.elapsed = elapsed
        self.timed_out = timed_out
        # Seconds spent waiting for the scheduler.  Reported, but deliberately
        # not charged to the tuning budget -- see PBSLauncher.
        self.queued_seconds = queued_seconds
        # [(returncode, elapsed, output), ...] when the launcher ran the
        # repetitions itself; None when it ran the command exactly once and
        # the caller is looping.
        self.repetitions = repetitions


class LaunchSpec:
    """One candidate's run, as handed to a launcher.

    Batched submission needs the whole generation described up front, so what
    used to be launch()'s arguments becomes a value.
    """

    __slots__ = ('rundir', 'command', 'env', 'preamble', 'reps')

    def __init__(self, rundir, command, env=None, preamble=(), reps=1):
        self.rundir = rundir
        self.command = command
        self.env = dict(env or {})
        self.preamble = list(preamble)
        self.reps = max(1, reps)


class LaunchError(Exception):
    """The launcher could not run the candidate at all."""


class Launcher:
    """Base class."""

    name = None
    # True when the launcher isolates each candidate in its own directory.
    per_candidate_dir = False
    # True when the launcher can run a candidate's repetitions itself.  Worth
    # doing only where a launch has a fixed overhead the repetitions can
    # share: under PBS that overhead is the queue wait, and batching turns
    # five waits per candidate into one.
    batches_reps = False
    # True when the launcher can run several candidates in one submission.
    # Same reasoning one level up: a generation's candidates are evaluated
    # sequentially either way, so putting them in one job changes nothing
    # about the measurements and removes a queue wait per candidate.
    batches_candidates = False

    def prepare(self, workdir, tag):
        """Return the directory this candidate should run in.

        Created if missing.  --workdir naming a directory that does not exist
        yet is the normal case for a fresh campaign, and failing on it wastes
        a scheduled job.
        """
        if workdir and not os.path.isdir(workdir):
            os.makedirs(workdir)
        return workdir

    def launch_many(self, specs, timeout=None, tag='batch'):
        """Run several candidates and return one LaunchResult each, in order.

        The default runs them one at a time through launch(); a launcher with
        batches_candidates set overrides this to submit once.
        """
        return [self.launch(spec.rundir, spec.command, spec.env, timeout,
                            spec.preamble, spec.reps) for spec in specs]

    def launch(self, rundir, command, env, timeout, preamble=(), reps=1):
        """Run `command` in `rundir` and return a LaunchResult.

        `preamble` is shell lines the application adapter wants run first, in
        the same directory and the same environment -- removing the last
        candidate's output file, typically.  It runs before every repetition,
        not once per launch, because a stale output file would let a candidate
        inherit the previous one's Lustre stripe layout.

        `reps` is honoured only by launchers with batches_reps set; the others
        ignore it and the caller loops.
        """
        raise NotImplementedError


class LocalLauncher(Launcher):
    """Run the command here, in this process's machine.

    Correct inside an allocation.  On a login node it will run the application
    on the login node, which is both wrong and antisocial, so the CLI warns
    when it cannot see a job environment.
    """

    name = 'local'

    def launch(self, rundir, command, env, timeout, preamble=(), reps=1):
        if preamble:
            command = '; '.join(list(preamble) + [command])
        started = time.time()
        process = subprocess.Popen(command, shell=True, cwd=rundir, env=env,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        try:
            raw, _unused = process.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            process.kill()
            raw, _unused = process.communicate()
            timed_out = True

        return LaunchResult(output=raw.decode('utf-8', 'replace'),
                            returncode=process.returncode,
                            elapsed=time.time() - started,
                            timed_out=timed_out)


class PBSLauncher(Launcher):
    """Submit each candidate as a PBS Pro job and wait for it.

    Written against the Nurion (KISTI) guide, which is PBS Pro with a mandatory
    -A application code and an exclusive-node policy.  Nothing here is Nurion
    specific beyond the defaults.

    On charging the tuning budget: the queue wait is measured and reported but
    is NOT added to it.  RoTI divides a bandwidth gain by tuning time, and a
    budget that included queue waits would make every reported RoTI a function
    of how busy the machine was that afternoon.  Two runs of the same framework
    would then differ for reasons that have nothing to do with the framework,
    and comparing H5Tuner against TunIO -- the whole point of this codebase --
    would stop meaning anything.  Wall-clock is kept alongside so the real cost
    of a campaign is still recoverable.
    """

    name = 'pbs'
    per_candidate_dir = True
    batches_reps = True
    batches_candidates = True

    def __init__(self, queue='normal', account='etc', select=1, ncpus=64,
                 mpiprocs=64, ompthreads=1, walltime='00:30:00', modules=(),
                 preamble=(), poll_seconds=15.0, wait_minutes=240.0,
                 job_prefix='at', qsub='qsub', qstat='qstat', qdel='qdel'):
        self.queue = queue
        self.account = account
        self.select = select
        self.ncpus = ncpus
        self.mpiprocs = mpiprocs
        self.ompthreads = ompthreads
        self.walltime = walltime
        self.modules = list(modules)
        # Arbitrary shell lines injected before the application runs, for
        # anything site specific the flags do not cover (lfs setstripe on the
        # output directory, LD_LIBRARY_PATH, a conda activate).
        self.preamble = list(preamble)
        self.poll_seconds = poll_seconds
        self.wait_minutes = wait_minutes
        self.job_prefix = job_prefix
        self.qsub = qsub
        self.qstat = qstat
        self.qdel = qdel

        self.jobs = []

    # -- directory ----------------------------------------------------------

    def prepare(self, workdir, tag):
        rundir = os.path.join(workdir, JOBS_DIR, tag)
        if not os.path.isdir(rundir):
            os.makedirs(rundir)
        # A stale sentinel from an interrupted campaign would be read as this
        # candidate's result.
        for name in os.listdir(rundir):
            if name in (SENTINEL, SENTINEL_PARTIAL) or (
                    name.startswith('run.') and name.endswith('.out')):
                os.remove(os.path.join(rundir, name))
        return rundir

    # -- submission ---------------------------------------------------------

    def _select_line(self):
        parts = ['select={0}'.format(self.select),
                 'ncpus={0}'.format(self.ncpus),
                 'mpiprocs={0}'.format(self.mpiprocs)]
        if self.ompthreads:
            parts.append('ompthreads={0}'.format(self.ompthreads))
        return ':'.join(parts)

    def _directives(self, directory, tag):
        """The #PBS block, shared by the single and batched scripts.

        -m n is not optional in practice.  A campaign submits hundreds of jobs
        and the site default mails on every one of them.
        """
        return ['#!/bin/sh',
                '#PBS -N {0}-{1}'.format(self.job_prefix, tag),
                '#PBS -q {0}'.format(self.queue),
                '#PBS -A {0}'.format(self.account),
                '#PBS -l {0}'.format(self._select_line()),
                '#PBS -l walltime={0}'.format(self.walltime),
                '#PBS -m n',
                '#PBS -o {0}'.format(os.path.join(directory, 'pbs.out')),
                '#PBS -e {0}'.format(os.path.join(directory, 'pbs.err')),
                '']

    def script(self, rundir, command, env, tag, preamble=(), reps=1):
        """Render the PBS job script for one candidate.

        The environment is exported inside the script rather than inherited
        through `#PBS -V`.  LD_PRELOAD is the reason: exporting the shim in the
        shell that runs qsub would make qsub itself load it, and -V would then
        carry any login-node contamination into the job.
        """
        lines = self._directives(rundir, tag)
        lines.extend(['cd {0}'.format(shlex.quote(rundir)), ''])

        if self.modules:
            lines.append('module purge')
            for module in self.modules:
                lines.append('module load {0}'.format(module))
            lines.append('')

        for name in sorted(env or {}):
            lines.append('export {0}={1}'.format(name,
                                                 shlex.quote(env[name])))
        if env:
            lines.append('')

        # Site-wide preamble runs once; the adapter's runs before every
        # repetition, inside the loop below.
        lines.extend(self.preamble)
        if self.preamble:
            lines.append('')
        preamble_lines = list(preamble)

        # Redirect here rather than relying on `#PBS -o`: PBS copies its own
        # stdout back only after the script exits, which would race the
        # sentinel this script writes on its last line.
        #
        # The repetitions run inside one job.  Submitting them separately
        # would make a candidate wait in the queue once per repetition, and
        # with the frameworks' own settings -- five for H5Tuner, three for
        # TunIO -- that is most of a campaign's wall-clock spent queueing for
        # measurements of a configuration already on the machine.
        body = ['__at_worst=0',
                'rm -f {0}'.format(SENTINEL_PARTIAL),
                'for __at_i in {0}; do'.format(
                    ' '.join(str(i + 1) for i in range(max(1, reps))))]
        for line in preamble_lines:
            body.append('    ' + line)
        body.extend([
            '    __at_start=$(date +%s.%N)',
            '    {0} > {1} 2>&1'.format(command, RUN_OUTPUT.format(
                '$__at_i')),
            '    __at_rc=$?',
            '    __at_end=$(date +%s.%N)',
            '    __at_elapsed=$(awk -v s="$__at_start" -v e="$__at_end" '
            "'BEGIN{printf \"%.3f\", e - s}')",
            '    printf "%s %s %s\\n" "$__at_i" "$__at_rc" "$__at_elapsed" '
            '>> {0}'.format(SENTINEL_PARTIAL),
            '    if [ "$__at_rc" -ne 0 ]; then __at_worst=$__at_rc; fi',
            'done',
            # Renaming into place last makes the sentinel atomic: the poller
            # never sees a half-written list of repetitions.
            'mv {0} {1}'.format(SENTINEL_PARTIAL, SENTINEL),
            'exit $__at_worst',
            '',
        ])
        lines.extend(body)
        return '\n'.join(lines)

    def _submit(self, rundir, script_text):
        path = os.path.join(rundir, JOB_SCRIPT)
        with open(path, 'w') as stream:
            stream.write(script_text)
        os.chmod(path, 0o755)

        process = subprocess.Popen([self.qsub, path], cwd=rundir,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        raw, _unused = process.communicate()
        text = raw.decode('utf-8', 'replace').strip()
        if process.returncode != 0 or not text:
            raise LaunchError('qsub failed ({0}): {1}'.format(
                process.returncode, text or '<no output>'))
        # PBS Pro answers with the job id alone, e.g. "1234567.pbs".
        return text.splitlines()[-1].strip()

    # -- waiting ------------------------------------------------------------

    def _in_queue(self, jobid):
        """True while the scheduler still admits to knowing this job."""
        process = subprocess.Popen([self.qstat, jobid],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        process.communicate()
        return process.returncode == 0

    def _read_sentinel(self, rundir):
        """Return [(index, returncode, elapsed), ...] or None if not done."""
        path = os.path.join(rundir, SENTINEL)
        try:
            with open(path) as stream:
                raw = stream.read()
        except (IOError, OSError):
            return None

        repetitions = []
        for line in raw.splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            try:
                repetitions.append((int(fields[0]), int(fields[1]),
                                    float(fields[2])))
            except ValueError:
                continue
        return repetitions or None

    def _read_file(self, rundir, name):
        path = os.path.join(rundir, name)
        if not os.path.exists(path):
            return ''
        with open(path, 'rb') as stream:
            return stream.read().decode('utf-8', 'replace')

    def _read_output(self, rundir, index=1):
        return (self._read_file(rundir, RUN_OUTPUT.format(index))
                or self._read_file(rundir, 'pbs.out'))

    def launch(self, rundir, command, env, timeout, preamble=(), reps=1):
        reps = max(1, reps)
        jobid = self._submit(rundir, self.script(rundir, command, env,
                                                 os.path.basename(rundir),
                                                 preamble, reps))
        self.jobs.append(jobid)

        submitted = time.time()
        deadline = submitted + self.wait_minutes * 60.0
        started_running = None
        vanished_at = None

        while True:
            sentinel = self._read_sentinel(rundir)
            if sentinel is not None:
                repetitions = [
                    (returncode, elapsed,
                     self._read_output(rundir, index))
                    for index, returncode, elapsed in sentinel]
                total = sum(elapsed for _, elapsed, _ in repetitions)
                queued = max(0.0, (time.time() - submitted) - total)
                worst = max((returncode for returncode, _, _ in repetitions),
                            key=abs)
                return LaunchResult(output=repetitions[-1][2],
                                    returncode=worst, elapsed=total,
                                    timed_out=False, queued_seconds=queued,
                                    repetitions=repetitions)

            # The application redirects its own output, so the file appearing
            # is the moment the job stopped queueing and started running.
            if started_running is None and os.path.exists(
                    os.path.join(rundir, FIRST_OUTPUT)):
                started_running = time.time()

            if not self._in_queue(jobid):
                # Gone from the queue with no sentinel.  Give the filesystem a
                # moment before calling it a failure.
                if vanished_at is None:
                    vanished_at = time.time()
                elif time.time() - vanished_at > VANISHED_GRACE_SECONDS:
                    return LaunchResult(
                        output=self._read_output(rundir) or
                        'job {0} left the queue without completing'.format(
                            jobid),
                        returncode=-1,
                        elapsed=(0.0 if started_running is None
                                 else time.time() - started_running),
                        timed_out=True,
                        queued_seconds=max(0.0, (started_running or
                                                 time.time()) - submitted))
            else:
                vanished_at = None

            if time.time() > deadline:
                self.cancel(jobid)
                return LaunchResult(
                    output=self._read_output(rundir),
                    returncode=-1,
                    elapsed=(0.0 if started_running is None
                             else time.time() - started_running),
                    timed_out=True,
                    queued_seconds=max(0.0, (started_running or time.time()) -
                                       submitted))

            time.sleep(self.poll_seconds)

    # -- batched submission -------------------------------------------------

    def batch_script(self, batchdir, specs, tag):
        """Render one job that evaluates every candidate in `specs`.

        Why this exists.  Nurion's normal queue allows 1000 queued jobs, so the
        count was never the constraint -- the wait is.  At 88% utilisation with
        hundreds of jobs already queued, a campaign of 40 generations by 15
        individuals pays 600 queue waits if each candidate is its own job, and
        one per generation if they share.  The candidates in a generation are
        evaluated sequentially either way, so nothing about the measurement
        changes.

        Each candidate writes its own sentinel, exactly as in the single-
        candidate script, so the reading code is shared.  The job writes one
        more at the end, which is what the poller waits on.
        """
        lines = self._directives(batchdir, tag)
        lines.extend(['cd {0}'.format(shlex.quote(batchdir)), ''])

        if self.modules:
            lines.append('module purge')
            for module in self.modules:
                lines.append('module load {0}'.format(module))
            lines.append('')

        lines.extend(self.preamble)
        if self.preamble:
            lines.append('')

        lines.append('__at_worst=0')
        lines.append('')

        for spec in specs:
            name = os.path.basename(spec.rundir)
            lines.append('# ---- {0} ----'.format(name))
            lines.append('cd {0}'.format(shlex.quote(spec.rundir)))
            for key in sorted(spec.env):
                lines.append('export {0}={1}'.format(
                    key, shlex.quote(spec.env[key])))
            lines.append('rm -f {0}'.format(SENTINEL_PARTIAL))
            lines.append('for __at_i in {0}; do'.format(
                ' '.join(str(i + 1) for i in range(spec.reps))))
            for line in spec.preamble:
                lines.append('    ' + line)
            lines.extend([
                '    __at_start=$(date +%s.%N)',
                '    {0} > {1} 2>&1'.format(
                    spec.command, RUN_OUTPUT.format('$__at_i')),
                '    __at_rc=$?',
                '    __at_end=$(date +%s.%N)',
                '    __at_elapsed=$(awk -v s="$__at_start" -v e="$__at_end" '
                "'BEGIN{printf \"%.3f\", e - s}')",
                '    printf "%s %s %s\\n" "$__at_i" "$__at_rc" '
                '"$__at_elapsed" >> {0}'.format(SENTINEL_PARTIAL),
                '    if [ "$__at_rc" -ne 0 ]; then __at_worst=$__at_rc; fi',
                'done',
                'mv {0} {1}'.format(SENTINEL_PARTIAL, SENTINEL),
                '',
            ])

        lines.extend([
            'cd {0}'.format(shlex.quote(batchdir)),
            'printf "%s\\n" "$__at_worst" > {0}'.format(BATCH_SENTINEL),
            'exit $__at_worst',
            '',
        ])
        return '\n'.join(lines)

    def prepare_batch(self, workdir, tag):
        batchdir = os.path.join(workdir, JOBS_DIR, tag)
        if not os.path.isdir(batchdir):
            os.makedirs(batchdir)
        stale = os.path.join(batchdir, BATCH_SENTINEL)
        if os.path.exists(stale):
            os.remove(stale)
        return batchdir

    def launch_many(self, specs, timeout=None, tag='batch'):
        if not specs:
            return []
        if len(specs) == 1:
            spec = specs[0]
            return [self.launch(spec.rundir, spec.command, spec.env, timeout,
                                spec.preamble, spec.reps)]

        batchdir = os.path.dirname(specs[0].rundir)
        stale = os.path.join(batchdir, BATCH_SENTINEL)
        if os.path.exists(stale):
            os.remove(stale)

        jobid = self._submit(batchdir,
                             self.batch_script(batchdir, specs, tag))
        self.jobs.append(jobid)

        submitted = time.time()
        deadline = submitted + self.wait_minutes * 60.0
        vanished_at = None
        started_running = None

        while True:
            if os.path.exists(os.path.join(batchdir, BATCH_SENTINEL)):
                return self._collect_batch(specs, submitted, started_running)

            if started_running is None:
                first = os.path.join(specs[0].rundir, FIRST_OUTPUT)
                if os.path.exists(first):
                    started_running = time.time()

            if not self._in_queue(jobid):
                if vanished_at is None:
                    vanished_at = time.time()
                elif time.time() - vanished_at > VANISHED_GRACE_SECONDS:
                    return self._collect_batch(specs, submitted,
                                               started_running,
                                               incomplete=True, jobid=jobid)
            else:
                vanished_at = None

            if time.time() > deadline:
                self.cancel(jobid)
                return self._collect_batch(specs, submitted, started_running,
                                           incomplete=True, jobid=jobid)

            time.sleep(self.poll_seconds)

    def _collect_batch(self, specs, submitted, started_running,
                       incomplete=False, jobid=None):
        """Read each candidate's sentinel; the queue wait is charged once.

        A candidate with no sentinel did not run -- the job died or was killed
        partway -- and is returned as timed out so the campaign scores it the
        penalty rather than silently treating it as fast.
        """
        queue_wait = max(0.0, (started_running or time.time()) - submitted)
        results = []
        for index, spec in enumerate(specs):
            sentinel = self._read_sentinel(spec.rundir)
            if sentinel is None:
                results.append(LaunchResult(
                    output=self._read_output(spec.rundir) or
                    'candidate did not run{0}'.format(
                        ' (job {0} ended early)'.format(jobid)
                        if incomplete else ''),
                    returncode=-1, elapsed=0.0, timed_out=True,
                    queued_seconds=queue_wait if index == 0 else 0.0))
                continue
            repetitions = [(rc, seconds, self._read_output(spec.rundir, i))
                           for i, rc, seconds in sentinel]
            total = sum(seconds for _, seconds, _ in repetitions)
            worst = max((rc for rc, _, _ in repetitions), key=abs)
            results.append(LaunchResult(
                output=repetitions[-1][2], returncode=worst, elapsed=total,
                timed_out=False,
                # The whole batch waited once; charging it to the first
                # candidate keeps the campaign total right without pretending
                # every candidate queued separately.
                queued_seconds=queue_wait if index == 0 else 0.0,
                repetitions=repetitions))
        return results

    def cancel(self, jobid):
        try:
            subprocess.Popen([self.qdel, jobid],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT).communicate()
        except OSError:
            pass

    def cancel_all(self):
        """Delete every job this launcher submitted.

        A campaign interrupted with Ctrl-C on the login node would otherwise
        leave its current job to run to completion against the allocation.
        """
        for jobid in self.jobs:
            self.cancel(jobid)


LAUNCHERS = {
    'local': LocalLauncher,
    'pbs': PBSLauncher,
}
