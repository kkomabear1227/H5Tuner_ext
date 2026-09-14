"""Read bandwidth from Darshan instead of scraping the application's output.

This is what the paper measures with.  III-E, describing the tuning pipeline:
"calls Python subprocess() to spawn an I/O kernel job with the appropriate
configurations set and monitor bandwidth (using monitoring hooks such as
Darshan [54]) within its fitness function".

Why it is better than a regex, beyond fidelity.  A scraped number is whatever
the application chose to print: h5bench reports its own view of write time, and
an application that prints nothing cannot be tuned at all.  Darshan measures
the I/O the process actually performed, at the POSIX and MPI-IO layers, for any
application -- including an I/O kernel produced by discovery, which has had its
print statements removed.

Two things have to line up for this to work on a machine.

LD_PRELOAD holds two libraries at once.  libautotuner.so injects the tuned
parameters; libdarshan.so measures the result.  The order matters and this
module puts the shim first: the shim rewrites the file access property list and
then calls into HDF5, so Darshan sitting behind it observes the I/O as
configured.  With Darshan first, it is the shim's call that gets instrumented
before the parameters are in place on some paths.  --darshan-first is there for
sites where that turns out to be wrong.

The log has to be findable.  Darshan writes to a central directory chosen at
build time, laid out by date, unless DARSHAN_LOGPATH points somewhere else.
This module points it at the candidate's own directory, which both makes the log
easy to find and keeps one candidate's log from being mistaken for another's.
"""

import glob
import os
import re
import shutil
import subprocess

# Subdirectory of a candidate's run directory that receives its logs.
LOG_DIRNAME = 'darshan'

# darshan-parser --perf prints this; it is the aggregate rate over the slowest
# rank, which is the figure that corresponds to what a user would call the
# application's I/O bandwidth.
_AGG_PERF = re.compile(
    r'agg_perf_by_slowest:\s*([0-9.eE+-]+)')
# darshan-parser --total prints one line per counter.
_TOTAL = re.compile(r'^total_([A-Z0-9_]+):\s*([0-9.eE+-]+)\s*$', re.MULTILINE)

# Modules to read byte counts and times from, most preferred first.  MPI-IO
# sits closest to what HDF5 actually issues for a collective write; POSIX is
# the fallback and always present.
_MODULES = ('MPIIO', 'POSIX')


class DarshanError(Exception):
    """Darshan was asked for a measurement it could not provide."""


class Darshan:
    """Configures a run to be instrumented, then reads what it produced.

    library     path to libdarshan.so on the target machine
    parser      darshan-parser executable; it runs where the tuner runs, so on
                a login node, not inside the job
    first       put libdarshan.so ahead of libautotuner.so in LD_PRELOAD
    nonmpi      set DARSHAN_ENABLE_NONMPI, needed when the application does not
                call MPI_Init
    """

    def __init__(self, library, parser='darshan-parser', first=False,
                 nonmpi=False):
        self.library = library
        self.parser = parser
        self.first = first
        self.nonmpi = nonmpi

    # -- setting up a run ---------------------------------------------------

    def log_dir(self, rundir):
        return os.path.join(rundir, LOG_DIRNAME)

    def prepare(self, rundir):
        """Create the log directory and clear any log left from a retry."""
        directory = self.log_dir(rundir)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        for stale in glob.glob(os.path.join(directory, '**', '*.darshan'),
                               recursive=True):
            os.remove(stale)
        return directory

    def environment(self, rundir, preload=None):
        """Environment additions that turn instrumentation on.

        `preload` is whatever LD_PRELOAD would otherwise be -- the tuner's own
        shim.  Both libraries end up in it, in the order this object was
        configured for.
        """
        parts = [part for part in (preload or '').split(':') if part]
        if self.library not in parts:
            if self.first:
                parts.insert(0, self.library)
            else:
                parts.append(self.library)

        env = {
            'LD_PRELOAD': ':'.join(parts),
            # Flat directory, one candidate's logs per candidate.
            'DARSHAN_LOGPATH': self.log_dir(rundir),
            # Darshan otherwise skips files it considers uninteresting; the
            # tuned parameters change exactly how those accesses are issued.
            'DARSHAN_LOGHINTS': '',
        }
        if self.nonmpi:
            env['DARSHAN_ENABLE_NONMPI'] = '1'
        return env

    # -- reading the result -------------------------------------------------

    def find_log(self, rundir):
        """The newest .darshan log under this candidate's log directory."""
        pattern = os.path.join(self.log_dir(rundir), '**', '*.darshan*')
        logs = [path for path in glob.glob(pattern, recursive=True)
                if os.path.isfile(path)]
        if not logs:
            return None
        return max(logs, key=os.path.getmtime)

    def _run_parser(self, log, extra):
        binary = shutil.which(self.parser) or self.parser
        try:
            process = subprocess.Popen([binary] + list(extra) + [log],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
        except OSError as error:
            raise DarshanError(
                'could not run {0}: {1}. It must be on PATH where the tuner '
                'runs, which for --launcher pbs is the login node, not the '
                'compute node.'.format(self.parser, error))
        out, err = process.communicate()
        if process.returncode != 0:
            raise DarshanError('{0} failed on {1}: {2}'.format(
                self.parser, os.path.basename(log),
                err.decode('utf-8', 'replace').strip()[:200]))
        return out.decode('utf-8', 'replace')

    def measure(self, rundir):
        """Return a dict of bandwidth and byte counts for one candidate.

        Keys: bw_write, bw_read (MiB/s, or None), bytes_written, bytes_read.

        On a workload that only writes -- every h5bench pattern is either pure
        write or pure read -- the aggregate rate IS the write bandwidth, and
        TunIO's perf reduces to it exactly, because alpha is 1.  For a mixed
        workload the aggregate cannot be split without assuming how reads and
        writes overlapped in time, so the two are derived from per-direction
        byte counts and times instead, and that path is approximate.
        """
        log = self.find_log(rundir)
        if log is None:
            raise DarshanError(
                'no .darshan log in {0}. Darshan writes its log when the '
                'process exits through MPI_Finalize; check that '
                'libdarshan.so was in LD_PRELOAD and that the application '
                'did not abort.'.format(self.log_dir(rundir)))

        text = self._run_parser(log, ['--perf', '--total'])
        totals = {name: float(value) for name, value in _TOTAL.findall(text)}

        aggregate = None
        match = _AGG_PERF.search(text)
        if match:
            aggregate = float(match.group(1))

        written = read = None
        write_time = read_time = None
        for module in _MODULES:
            if written is None:
                written = totals.get('{0}_BYTES_WRITTEN'.format(module))
                write_time = totals.get('{0}_F_WRITE_TIME'.format(module))
            if read is None:
                read = totals.get('{0}_BYTES_READ'.format(module))
                read_time = totals.get('{0}_F_READ_TIME'.format(module))

        result = {'bytes_written': written, 'bytes_read': read,
                  'bw_write': None, 'bw_read': None, 'log': log,
                  'aggregate': aggregate}

        pure_write = bool(written) and not read
        pure_read = bool(read) and not written

        if aggregate is not None and pure_write:
            result['bw_write'] = aggregate
        elif aggregate is not None and pure_read:
            result['bw_read'] = aggregate
        else:
            # Mixed, or --perf gave nothing.  Fall back to bytes over the
            # summed per-rank time.  This understates bandwidth on a parallel
            # run, because those times add across ranks while the I/O overlaps.
            if written and write_time:
                result['bw_write'] = written / write_time / (1024.0 * 1024.0)
            if read and read_time:
                result['bw_read'] = read / read_time / (1024.0 * 1024.0)
            result['approximate'] = True

        if result['bw_write'] is None and result['bw_read'] is None:
            raise DarshanError(
                'Darshan log {0} has no usable bandwidth. Parsed counters: '
                '{1}. If the application uses only HDF5 calls and Darshan was '
                'built without its POSIX or MPI-IO modules, there is nothing '
                'to read.'.format(os.path.basename(log),
                                  ', '.join(sorted(totals)[:8]) or 'none'))
        return result
