"""What "better" means.

Two objectives are supported and they point in opposite directions, so every
consumer must go through `Objective.sense` rather than assuming one of them.

    walltime    minimise the application's wall-clock time.  What
                evo/evolve.py used.  Includes compute, so it is not an I/O
                measurement.
    bandwidth   maximise TunIO's perf.  This is the objective the paper's whole
                design assumes:

                    perf = (1 - alpha) * BW_read + alpha * BW_write
                    alpha = bytes written / bytes transferred

Getting the objective wrong is not a detail.  TunIO's early stopper is trained
on synthetic curves that *rise* logarithmically, its reward normalises perf by
1 / (BW_single * num_nodes), and RoTI divides a bandwidth gain by tuning time.
Feed wall-clock seconds into any of those and the sign flips.

Bandwidth has to come from somewhere.  Three sources, in decreasing fidelity:

    darshan     what the paper uses.  Not implemented yet -- needs the profiler
                on the target machine and coexistence with libautotuner.so in
                LD_PRELOAD.
    regex       scrape the number the application already prints.  Works today
                because evaluate.py captures stdout anyway.
    derived     output file size divided by elapsed time.  No dependencies, but
                it includes compute time and cannot see reads.
"""

import re

SENSE_MIN = 'min'
SENSE_MAX = 'max'

# Fitness assigned to a candidate that timed out or crashed.  Deliberately far
# outside any plausible measurement so it cannot be mistaken for a real result.
PENALTY = {
    SENSE_MIN: 1e4,
    SENSE_MAX: 0.0,
}

_UNIT_TO_MB = {
    'B/s': 1.0 / (1024 * 1024),
    'KB/s': 1.0 / 1024,
    'MB/s': 1.0,
    'GB/s': 1024.0,
    'TB/s': 1024.0 * 1024.0,
}

# Multipliers for a unit prefix captured from the output rather than declared
# on the command line.  h5bench needs this: it formats every rate through
# format_human_readable(), which divides by 1024 until the number fits and
# prints the matching prefix, so the same benchmark says MB/s on one run and
# GB/s on a faster one.  Assuming a fixed unit would understate a fast
# candidate by 1024x -- an error that grows precisely as the search succeeds,
# and one the search would read as the candidate being terrible.
_PREFIX_TO_MB = {
    '': 1.0 / (1024 * 1024),
    'B': 1.0 / (1024 * 1024),
    'K': 1.0 / 1024,
    'M': 1.0,
    'G': 1024.0,
    'T': 1024.0 * 1024.0,
}


class Measurement:
    """Everything one candidate evaluation produced.

    elapsed         wall-clock seconds, one entry per repetition
    returncodes     process exit codes, one per repetition
    output          stdout+stderr of the last repetition
    bw_read         read bandwidth in MB/s, or None if unknown
    bw_write        write bandwidth in MB/s, or None if unknown
    bytes_read      bytes read, or None
    bytes_written   bytes written, or None
    timed_out       True if the candidate exceeded its budget
    """

    __slots__ = ('elapsed', 'returncodes', 'output', 'bw_read', 'bw_write',
                 'bytes_read', 'bytes_written', 'timed_out')

    def __init__(self, elapsed=None, returncodes=None, output='',
                 bw_read=None, bw_write=None, bytes_read=None,
                 bytes_written=None, timed_out=False):
        self.elapsed = list(elapsed or [])
        self.returncodes = list(returncodes or [])
        self.output = output
        self.bw_read = bw_read
        self.bw_write = bw_write
        self.bytes_read = bytes_read
        self.bytes_written = bytes_written
        self.timed_out = timed_out

    @property
    def ok(self):
        return (not self.timed_out
                and bool(self.returncodes)
                and all(code == 0 for code in self.returncodes))

    @property
    def alpha(self):
        """Write share of transferred data, as TunIO defines it.

        Falls back to 1.0 (pure write) when byte counts are unavailable, which
        is the case for every workload in this repository.
        """
        if self.bytes_read is None or self.bytes_written is None:
            return 1.0
        total = self.bytes_read + self.bytes_written
        if total <= 0:
            return 1.0
        return float(self.bytes_written) / total

    def as_dict(self):
        return {
            'elapsed': self.elapsed,
            'returncodes': self.returncodes,
            'bw_read': self.bw_read,
            'bw_write': self.bw_write,
            'bytes_read': self.bytes_read,
            'bytes_written': self.bytes_written,
            'timed_out': self.timed_out,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(elapsed=data.get('elapsed'),
                   returncodes=data.get('returncodes'),
                   output='',
                   bw_read=data.get('bw_read'),
                   bw_write=data.get('bw_write'),
                   bytes_read=data.get('bytes_read'),
                   bytes_written=data.get('bytes_written'),
                   timed_out=data.get('timed_out', False))


class Objective:
    """Base class.  Subclasses turn a Measurement into a single number."""

    name = None
    sense = None
    unit = ''

    def __init__(self, reduction='best'):
        if reduction not in ('best', 'min', 'max', 'mean'):
            raise ValueError('unknown reduction {0!r}'.format(reduction))
        self.reduction = reduction

    @property
    def penalty(self):
        return PENALTY[self.sense]

    def better(self, left, right):
        """True if `left` is a better score than `right`."""
        return left < right if self.sense == SENSE_MIN else left > right

    def best_of(self, scores):
        scores = list(scores)
        if not scores:
            raise ValueError('no scores')
        return min(scores) if self.sense == SENSE_MIN else max(scores)

    def _reduce(self, samples):
        """Collapse per-repetition samples into one number.

        'best' resolves to min for a minimised objective and max for a
        maximised one, which is what evo/evolve.py did with min(times): the
        fastest run is the one least polluted by cold caches and neighbours.
        TunIO averages three runs instead, so 'mean' reproduces the paper.
        """
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        if self.reduction == 'mean':
            return sum(samples) / float(len(samples))
        if self.reduction == 'min':
            return min(samples)
        if self.reduction == 'max':
            return max(samples)
        return min(samples) if self.sense == SENSE_MIN else max(samples)

    def reduce_samples(self, samples):
        """Collapse per-repetition measurements of one quantity.

        Public because the evaluator now scrapes bandwidth once per
        repetition and has to fold the samples itself: unlike wall-clock,
        which Measurement carries as a list, bandwidth is a single field.
        """
        return self._reduce(samples)

    def score(self, measurement):
        raise NotImplementedError


class Walltime(Objective):
    """Minimise application wall-clock time.  evo/evolve.py's objective."""

    name = 'walltime'
    sense = SENSE_MIN
    unit = 's'

    def score(self, measurement):
        if measurement.timed_out:
            return self.penalty
        value = self._reduce(measurement.elapsed)
        return self.penalty if value is None else value


class Bandwidth(Objective):
    """Maximise TunIO's perf, in MB/s.

    perf = (1 - alpha) * BW_read + alpha * BW_write

    A workload that only writes has alpha == 1 and reduces to write bandwidth,
    which is every workload currently in this repository.
    """

    name = 'bandwidth'
    sense = SENSE_MAX
    unit = 'MB/s'

    def score(self, measurement):
        if measurement.timed_out:
            return self.penalty
        write = measurement.bw_write
        read = measurement.bw_read
        if write is None and read is None:
            return self.penalty
        alpha = measurement.alpha
        total = 0.0
        if write is not None:
            total += alpha * write
        if read is not None:
            total += (1.0 - alpha) * read
        return total


OBJECTIVES = {
    'walltime': Walltime,
    'bandwidth': Bandwidth,
}


def get_objective(name, reduction='best'):
    try:
        return OBJECTIVES[name](reduction=reduction)
    except KeyError:
        raise ValueError('unknown objective {0!r}; choose from {1}'.format(
            name, ', '.join(sorted(OBJECTIVES))))


# ---------------------------------------------------------------------------
# Scraping bandwidth out of application output
# ---------------------------------------------------------------------------

class OutputScraper:
    """Pull bandwidth and byte counts out of captured application output.

    The pattern may use named groups.  Recognised names:

        bw_write, bw_read           bandwidth, in `unit`
        bandwidth                   treated as bw_write (alpha defaults to 1)
        bytes_written, bytes_read   byte counts, used to compute alpha
        <name>_unit                 unit prefix for <name>, overriding `unit`;
                                    one of "", B, K, M, G, T

    A pattern with no named groups is read as a single bandwidth figure taken
    from group 1, or from group 0 if the pattern has no groups at all.

    When a name matches several times the last match wins.  Benchmarks
    typically print a summary line after per-rank chatter.
    """

    def __init__(self, pattern, unit='MB/s'):
        if unit not in _UNIT_TO_MB:
            raise ValueError('unknown unit {0!r}; choose from {1}'.format(
                unit, ', '.join(sorted(_UNIT_TO_MB))))
        self.pattern = re.compile(pattern, re.MULTILINE)
        self.unit = unit
        self.scale = _UNIT_TO_MB[unit]

    def apply(self, measurement):
        """Fill in bandwidth fields on `measurement` from its output."""
        found = {}
        for match in self.pattern.finditer(measurement.output or ''):
            groups = match.groupdict()
            if groups:
                for key, raw in groups.items():
                    if raw is not None:
                        found[key] = raw
            else:
                try:
                    found['bandwidth'] = match.group(1)
                except IndexError:
                    found['bandwidth'] = match.group(0)

        def number(key):
            raw = found.get(key)
            if raw is None:
                return None
            try:
                return float(raw.replace(',', ''))
            except ValueError:
                return None

        def scale_for(key):
            """Multiplier to MB/s for a value captured under `key`.

            A companion group named `<key>_unit` overrides --perf-unit, so an
            application that scales its own unit is read correctly whatever it
            chose to print this time.
            """
            raw = found.get(key + '_unit')
            if raw is None:
                return self.scale
            prefix = raw.strip().upper()
            if prefix not in _PREFIX_TO_MB:
                raise ValueError(
                    'unit prefix {0!r} scraped for {1} is not one of '
                    '{2}'.format(raw, key,
                                 ', '.join(sorted(_PREFIX_TO_MB))))
            return _PREFIX_TO_MB[prefix]

        write, write_key = number('bw_write'), 'bw_write'
        if write is None:
            write, write_key = number('bandwidth'), 'bandwidth'
        read = number('bw_read')

        if write is not None:
            measurement.bw_write = write * scale_for(write_key)
        if read is not None:
            measurement.bw_read = read * scale_for('bw_read')

        written = number('bytes_written')
        if written is not None:
            measurement.bytes_written = written
        got = number('bytes_read')
        if got is not None:
            measurement.bytes_read = got

        return measurement
