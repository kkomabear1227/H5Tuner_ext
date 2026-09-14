"""Framework registry.

Each framework is a published system: a search algorithm plus whatever
components its paper specifies.  Adding one means adding a module here and an
entry in FRAMEWORKS.

Every framework carries a DEFAULTS dict of the settings its paper reports.  The
CLI applies those defaults and lets the user override any of them, so an
ablation ("TunIO but with the heuristic stopper") is a flag rather than a code
change -- which matters, because the TunIO paper's own comparison table is built
out of exactly such ablations.
"""

from .h5tuner import DEFAULTS as H5TUNER_DEFAULTS
from .h5tuner import H5Tuner
from .h5tuner_original import DEFAULTS as H5TUNER_ORIGINAL_DEFAULTS
from .h5tuner_original import H5TunerOriginal
from .tunio import DEFAULTS as TUNIO_DEFAULTS
from .tunio import TunIO

FRAMEWORKS = {
    H5Tuner.name: (H5Tuner, H5TUNER_DEFAULTS),
    H5TunerOriginal.name: (H5TunerOriginal, H5TUNER_ORIGINAL_DEFAULTS),
    TunIO.name: (TunIO, TUNIO_DEFAULTS),
}

NAMES = tuple(sorted(FRAMEWORKS))


def get(name):
    """Return (class, defaults) for a framework name."""
    try:
        return FRAMEWORKS[name]
    except KeyError:
        raise ValueError('unknown framework {0!r}; choose from {1}'.format(
            name, ', '.join(NAMES)))


def defaults(name):
    return dict(get(name)[1])


def describe():
    lines = []
    for name in NAMES:
        cls, settings = FRAMEWORKS[name]
        lines.append('  {0:<18}{1}'.format(name, cls.description))
        lines.append('  {0:<18}space={1} objective={2} stopper={3} '
                     'subset={4}'.format('', settings['space'],
                                         settings['objective'],
                                         settings['stopper'],
                                         settings['subset']))
    return '\n'.join(lines)


__all__ = ['FRAMEWORKS', 'NAMES', 'defaults', 'describe', 'get']
