"""TunIO's Application I/O Discovery component (III-B).

Reduces an application's source to an I/O kernel: the calls that perform I/O,
everything they depend on, and nothing else.  The tuner then evaluates
candidates against the kernel instead of the full application, which is where
the paper's largest reported RoTI gain comes from -- 2.47 to 23.30 once loop
reduction is applied.

    from autotuner import discovery
    kernel = discovery.discover('vpic.c', workdir='/tmp')
    print(discovery.reconstruct(kernel))
"""

from .kernel import DiscoveryError, Kernel, discover, reconstruct
from .marking import DEFAULT_IO_PREFIXES, Marker, Marks
from .reduce import apply_reductions

__all__ = ['DiscoveryError', 'Kernel', 'discover', 'reconstruct',
           'apply_reductions', 'DEFAULT_IO_PREFIXES', 'Marker', 'Marks']
