"""Record measurements once, replay them many times.

A trace is a JSON-lines file, one measured configuration per line.  Recording a
sweep on the cluster and replaying it on a laptop is what makes the rest of this
work possible:

  * TunIO's two RL agents need offline pretraining.  Training against the real
    application would cost more than the tuning it is supposed to save.
  * Comparing H5Tuner against TunIO fairly means comparing them on the *same*
    measurements.  Two live runs differ by filesystem noise as much as by
    algorithm.
  * Reproducing a reported result requires the measurements, not just the code.

Entries are keyed by decoded parameter values, not by point indices, so a trace
survives reordering the parameters in space.py.  It does not survive changing
which parameters exist -- that would silently compare different experiments, so
loading refuses it.
"""

import json
import os


def _key(space, point):
    """Canonical key for a configuration: sorted name=value pairs."""
    values = space.decode(point)
    return '|'.join('{0}={1}'.format(name, values[name])
                    for name in sorted(values))


class Entry:
    __slots__ = ('values', 'score', 'measurement', 'wall_cost')

    def __init__(self, values, score, measurement, wall_cost):
        self.values = values
        self.score = score
        self.measurement = measurement
        self.wall_cost = wall_cost


class Recorder:
    """Appends measurements to a JSONL file as they happen.

    Written and flushed per entry.  A campaign that dies at generation 30 still
    leaves 30 generations of usable trace.
    """

    def __init__(self, path, metadata=None):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        self._stream = open(path, 'w')
        self.count = 0
        if metadata:
            self._write({'type': 'metadata', **metadata})

    def _write(self, record):
        self._stream.write(json.dumps(record, sort_keys=True) + '\n')
        self._stream.flush()

    def add(self, space, point, score, measurement, wall_cost):
        self._write({
            'type': 'measurement',
            'space': space.name,
            'values': space.decode(point),
            'score': score,
            'wall_cost': wall_cost,
            'measurement': measurement.as_dict(),
        })
        self.count += 1

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Trace:
    """A loaded trace, queryable by configuration."""

    def __init__(self, entries, metadata=None, parameter_names=None):
        self._entries = entries
        self.metadata = metadata or {}
        self.parameter_names = frozenset(parameter_names or ())

    def __len__(self):
        return len(self._entries)

    def lookup(self, space, point):
        return self._entries.get(_key(space, point))

    def check_compatible(self, space):
        """Raise if `space` does not describe the same knobs as the trace."""
        wanted = frozenset(p.name for p in space.parameters)
        if not self.parameter_names:
            return
        if wanted != self.parameter_names:
            missing = sorted(wanted - self.parameter_names)
            extra = sorted(self.parameter_names - wanted)
            details = []
            if missing:
                details.append('not in trace: ' + ', '.join(missing))
            if extra:
                details.append('only in trace: ' + ', '.join(extra))
            raise ValueError(
                'trace was recorded over a different parameter set; '
                'replaying it would compare different experiments\n  '
                + '\n  '.join(details))

    def scores(self):
        return [entry.score for entry in self._entries.values()]


def load(path):
    """Read a JSONL trace."""
    from .objective import Measurement

    entries = {}
    metadata = {}
    names = set()
    with open(path) as stream:
        for lineno, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError('{0}:{1}: malformed JSON: {2}'.format(
                    path, lineno, error))
            kind = record.get('type')
            if kind == 'metadata':
                metadata = record
                continue
            if kind != 'measurement':
                continue
            values = record['values']
            names.update(values)
            key = '|'.join('{0}={1}'.format(name, values[name])
                           for name in sorted(values))
            entries[key] = Entry(
                values=values,
                score=record['score'],
                measurement=Measurement.from_dict(record.get('measurement',
                                                             {})),
                wall_cost=record.get('wall_cost', 0.0))
    return Trace(entries, metadata=metadata, parameter_names=names)
