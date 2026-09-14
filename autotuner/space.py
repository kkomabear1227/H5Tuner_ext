"""The parameter space every framework searches.

A *point* in the space is a tuple of integers, one per parameter, each an index
into that parameter's candidate list.  Index encoding keeps the search
methodology agnostic to whether a parameter is numeric, categorical, or a
string pair: a genetic algorithm mutates an index, and anything that wants
ordinal bounds gets [0, len(values) - 1] for free.

Two named spaces are available:

    paper12   the 12 parameters TunIO tunes.  The primary experiment space.
              All twelve are injected by src/autotuner_hdf5.c as of 2026-08-13.
    minimal   a seven-parameter subset, small enough that a full grid can be
              swept.  Useful for quick runs and for recording a replayable
              trace; not the experiment space.

On the parameter count.  TunIO's III-C names eleven parameters but IV says the
evaluation tunes twelve.  We resolve the discrepancy by counting HDF5's
alignment threshold and alignment value separately, which yields exactly
twelve.  That is a reading, not a fact -- see 07-후속-연구-계획.md #7.  It
has a side benefit: as two parameters they can be mutated independently, which
a single "threshold, alignment" string could not.

On the candidate values.  The paper gives parameter *names* but no value lists,
so the lists below are ours.  Where evo/evolve.py had a list we kept it, which
is why some are not clean powers of two.  Deviations are noted per parameter.
"""

import itertools

# Layer names, used for reporting and for layer-aware subset selection.
LAYER_HDF5 = 'hdf5'
LAYER_MPIIO = 'mpiio'
LAYER_PFS = 'pfs'

KB = 1024
MB = 1024 * KB
GB = 1024 * MB


class Parameter:
    """One tunable knob.

    name        identifier used in logs, traces and framework code
    layer       which I/O stack layer it belongs to
    values      ordered candidate list; a point holds an index into this
    default     value used when the parameter is not being tuned
    element     config.xml element this renders into
    group_pos   position within `element` when several parameters share one
                element (alignment writes "threshold,alignment"); None if the
                parameter owns its element outright
    shim        True if src/autotuner_hdf5.c injects this today
    hdf5_min    minimum HDF5 version required, or None
    note        why the values are what they are
    """

    __slots__ = ('name', 'layer', 'values', 'default', 'element', 'group_pos',
                 'shim', 'hdf5_min', 'note')

    def __init__(self, name, layer, values, default, element, group_pos=None,
                 shim=False, hdf5_min=None, note=''):
        if default not in values:
            raise ValueError(
                '{0}: default {1!r} is not one of the candidate values'.format(
                    name, default))
        self.name = name
        self.layer = layer
        self.values = tuple(values)
        self.default = default
        self.element = element
        self.group_pos = group_pos
        self.shim = shim
        self.hdf5_min = hdf5_min
        self.note = note

    @property
    def default_index(self):
        return self.values.index(self.default)

    def __repr__(self):
        return 'Parameter({0!r}, {1} values)'.format(self.name,
                                                     len(self.values))


# ---------------------------------------------------------------------------
# Parameter definitions
#
# Ordering is by layer (HDF5, MPI-IO, PFS) to match the config.xml sections and
# to make layer-aware reporting readable.  The order is otherwise arbitrary;
# nothing depends on it beyond point encoding, which is stable per Space.
# ---------------------------------------------------------------------------

_HDF5 = [
    Parameter(
        'sieve_buf_size', LAYER_HDF5,
        # evo/evolve.py shipped a single 512MB value.  The original's
        # commented-out list was [65535, 131070, ...] -- 2**16 - 1 and its
        # doublings, which defeats the point of a page-aligned buffer.  A later
        # commented line marked "NEW, NOT FOR IPDPS" fixed it to powers of two;
        # that is the list used here.  HDF5's own default is 64KB.
        values=(64 * KB, 128 * KB, 256 * KB, 512 * KB, 1 * MB, 2 * MB),
        default=64 * KB,
        element='sieve_buf_size', shim=True,
        note='powers of two, from the original\'s "NEW, NOT FOR IPDPS" list'),
    Parameter(
        'alignment_threshold', LAYER_HDF5,
        # Objects at least this large get aligned.  HDF5's default is 1, but 0
        # and 1 both mean "align everything", so only 0 is listed and it
        # doubles as the default.  evo/evolve.py carried both, which made two
        # of its fourteen alignment candidates aliases of each other.
        values=(0, 1 * KB, 4 * KB, 16 * KB, 64 * KB),
        default=0,
        element='alignment', group_pos=0, shim=True,
        note='0 stands in for HDF5\'s default of 1; the two are equivalent'),
    Parameter(
        'alignment', LAYER_HDF5,
        # evo/evolve.py capped this at 256KB while striping_unit started at
        # 1MB, so stripe-boundary alignment -- the main reason to set it on
        # Lustre -- was unreachable.  Extended to 4MB.  1 means "no alignment".
        values=(1, 4 * KB, 64 * KB, 256 * KB, 1 * MB, 4 * MB),
        default=1,
        element='alignment', group_pos=1, shim=True,
        note='extended past striping_unit\'s floor so stripe alignment is '
             'reachable'),
    Parameter(
        'meta_block_size', LAYER_HDF5,
        values=(2 * KB, 8 * KB, 64 * KB, 256 * KB, 1 * MB, 4 * MB),
        default=2 * KB,
        element='meta_block_size', shim=True,
        note='2KB is the HDF5 default'),
    Parameter(
        'chunk_cache', LAYER_HDF5,
        # H5Pset_cache takes four arguments.  We tune only rdcc_nbytes and
        # leave nslots and the preemption policy at HDF5's defaults, so this
        # stays one dimension.  Widening it to the full four is a TODO.
        values=(1 * MB, 4 * MB, 16 * MB, 64 * MB, 256 * MB, 1 * GB),
        default=1 * MB,
        element='chunk_cache', shim=True,
        note='rdcc_nbytes only; nslots and w0 left at HDF5 defaults'),
    Parameter(
        'coll_metadata_write', LAYER_HDF5,
        values=(0, 1), default=0,
        element='coll_metadata_write', shim=True, hdf5_min=(1, 10, 0),
        note='H5Pset_coll_metadata_write, added in HDF5 1.10.0'),
    Parameter(
        'col_meta_ops', LAYER_HDF5,
        values=(0, 1), default=0,
        element='col_meta_ops', shim=True, hdf5_min=(1, 10, 0),
        note='H5Pset_all_coll_metadata_ops, added in HDF5 1.10.0'),
    Parameter(
        'mdc_conf', LAYER_HDF5,
        # H5AC_cache_config_t has twenty-odd fields.  We expose named presets
        # instead; the shim will start from H5Pget_mdc_config and override a
        # few fields per preset.
        values=('default', 'aggressive', 'conservative'),
        default='default',
        element='mdc_conf', shim=True,
        note='named presets, not the full H5AC_cache_config_t struct'),
]

_MPIIO = [
    Parameter(
        'cb_nodes', LAYER_MPIIO,
        values=(1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256),
        default=1,
        element='cb_nodes', shim=True,
        note='from evo/evolve.py unchanged'),
    Parameter(
        'cb_buffer_size', LAYER_MPIIO,
        # evo/evolve.py tied this to striping_unit instead of searching it.
        # These are the values from the original's commented-out list.
        values=(4 * MB, 8 * MB, 16 * MB, 32 * MB, 64 * MB, 128 * MB),
        default=16 * MB,
        element='cb_buffer_size', shim=True,
        note='restored from the original\'s commented-out cb_buf_size list'),
]

_PFS = [
    Parameter(
        'striping_factor', LAYER_PFS,
        # The original's comment says 1 and 4 were skipped as very bad, but 4
        # is in the list.  Kept as-is for fidelity; see PLAN #2.4.
        # -1 means stripe across every available OST.
        values=(4, 8, 16, 24, 32, 48, 64, 96, 128, -1),
        default=4,
        element='striping_factor', shim=True,
        note='from evo/evolve.py unchanged; comment/list mismatch preserved'),
    Parameter(
        'striping_unit', LAYER_PFS,
        values=(1 * MB, 2 * MB, 4 * MB, 8 * MB, 16 * MB, 32 * MB, 64 * MB,
                128 * MB),
        default=1 * MB,
        element='striping_unit', shim=True,
        note='from evo/evolve.py unchanged'),
]

ALL_PARAMETERS = tuple(_HDF5 + _MPIIO + _PFS)


class Space:
    """An ordered collection of parameters plus optional tie constraints.

    A point is a tuple of len(parameters) integers.  `decode` turns a point
    into {parameter name: value}, applying ties.

    ties maps a follower parameter name to a leader parameter name.  The
    follower's own index is ignored and it takes the leader's *value*.  This
    exists for one reason: evo/evolve.py tied cb_buffer_size to striping_unit
    ("Ruth(David's) Suggestion"), removing a dimension.  Only the
    h5tuner-original framework enables it.
    """

    def __init__(self, name, parameters, ties=None):
        self.name = name
        self.parameters = tuple(parameters)
        self.ties = dict(ties or {})
        self._index = {p.name: i for i, p in enumerate(self.parameters)}
        for follower, leader in self.ties.items():
            if follower not in self._index:
                raise ValueError('tie follower {0!r} is not in the space'
                                 .format(follower))
            if leader not in self._index:
                raise ValueError('tie leader {0!r} is not in the space'
                                 .format(leader))

    # -- introspection ------------------------------------------------------

    def __len__(self):
        return len(self.parameters)

    def __iter__(self):
        return iter(self.parameters)

    def index_of(self, name):
        return self._index[name]

    def get(self, name):
        return self.parameters[self._index[name]]

    @property
    def free_names(self):
        """Names that actually carry a degree of freedom.

        A parameter with one candidate value is a constant, and a tie follower
        is not searched.  Both are excluded.
        """
        return tuple(p.name for p in self.parameters
                     if len(p.values) > 1 and p.name not in self.ties)

    def size(self):
        """Number of distinct configurations, counting free parameters only.

        This is a nominal count.  The system may clamp or ignore values, in
        which case several points collapse to one effective configuration --
        see 07-후속-연구-계획.md #8.4.
        """
        total = 1
        for name in self.free_names:
            total *= len(self.get(name).values)
        return total

    def bounds(self):
        """Per-parameter index bounds, for methodologies that want them."""
        return tuple((0, len(p.values) - 1) for p in self.parameters)

    # -- points -------------------------------------------------------------

    def default_point(self):
        return tuple(p.default_index for p in self.parameters)

    def sample(self, rng, mask=None):
        """A uniformly random point.

        mask, when given, is an iterable of parameter names allowed to vary;
        everything else takes its default index.
        """
        allowed = None if mask is None else set(mask)
        return tuple(
            rng.randrange(len(p.values))
            if allowed is None or p.name in allowed
            else p.default_index
            for p in self.parameters)

    def decode(self, point):
        """Turn a point into {parameter name: value}, applying ties."""
        if len(point) != len(self.parameters):
            raise ValueError('point has {0} entries, space has {1}'.format(
                len(point), len(self.parameters)))
        values = {}
        for parameter, index in zip(self.parameters, point):
            values[parameter.name] = parameter.values[index]
        for follower, leader in self.ties.items():
            values[follower] = values[leader]
        return values

    def canonical(self, point):
        """Point with tie followers forced to a fixed index.

        Two points that differ only in a tied parameter's index decode
        identically, so they must share a cache key.
        """
        if not self.ties:
            return tuple(point)
        canonical = list(point)
        for follower in self.ties:
            canonical[self._index[follower]] = 0
        return tuple(canonical)

    def describe(self, point):
        """One-line human-readable rendering of a point."""
        values = self.decode(point)
        return ', '.join('{0}={1}'.format(p.name, values[p.name])
                         for p in self.parameters)

    # -- checks -------------------------------------------------------------

    def unsupported(self, hdf5_version=None):
        """Parameters the current build cannot actually inject.

        Returns a list of (parameter, reason).  Used by the CLI to warn before
        a run silently tunes knobs that never reach the application.
        """
        problems = []
        for parameter in self.parameters:
            if not parameter.shim:
                problems.append((parameter,
                                 'not injected by src/autotuner_hdf5.c'))
            elif (parameter.hdf5_min is not None
                    and hdf5_version is not None
                    and tuple(hdf5_version) < parameter.hdf5_min):
                problems.append((parameter, 'needs HDF5 {0}'.format(
                    '.'.join(str(n) for n in parameter.hdf5_min))))
        return problems


# ---------------------------------------------------------------------------
# Named spaces
# ---------------------------------------------------------------------------

PAPER12 = Space('paper12', ALL_PARAMETERS)

# A small space for quick runs and for grids cheap enough to sweep exhaustively.
# Until 2026-08-13 this was "whatever the C shim could inject", computed as a
# filter on `shim`.  The shim now injects all twelve, so that filter would make
# this identical to paper12; the seven are listed explicitly instead.
_MINIMAL_NAMES = ('sieve_buf_size', 'alignment_threshold', 'alignment',
                  'cb_nodes', 'cb_buffer_size', 'striping_factor',
                  'striping_unit')

MINIMAL = Space('minimal', [p for p in ALL_PARAMETERS
                            if p.name in _MINIMAL_NAMES])

# The historical space: the five genes evo/evolve.py declared, with
# cb_buffer_size tied to striping_unit and sieve_buf_size pinned to 512MB.
# Reproduced so h5tuner-original searches exactly what it used to.
_ORIGINAL_SIEVE = Parameter(
    'sieve_buf_size', LAYER_HDF5, values=(512 * MB,), default=512 * MB,
    element='sieve_buf_size', shim=True,
    note='single value, as shipped in evo/evolve.py -- a constant, not a gene')

# The original kept threshold and alignment in one gene as a "threshold,
# alignment" string, and listed fourteen hand-picked pairs rather than the full
# cross product -- roughly the pairs where alignment exceeds threshold.  Kept
# verbatim so h5tuner-original searches 13,440 configurations, exactly what it
# used to.  As one gene these two cannot be mutated independently, which is one
# reason paper12 splits them.
_ORIGINAL_ALIGN = Parameter(
    'alignment', LAYER_HDF5,
    values=('1,1',
            '0,4096', '0,16384', '0,65536', '0,262144',
            '1024,4096', '1024,16384', '1024,65536', '1024,262144',
            '4096,16384', '4096,65536', '4096,262144',
            '16384,65536', '16384,262144'),
    default='1,1',
    element='alignment', shim=True,
    note='the original fourteen "threshold,alignment" pairs, capped at 256KB')

ORIGINAL = Space(
    'original',
    [_ORIGINAL_SIEVE, _ORIGINAL_ALIGN,
     PAPER12.get('cb_nodes'), PAPER12.get('cb_buffer_size'),
     PAPER12.get('striping_factor'), PAPER12.get('striping_unit')],
    ties={'cb_buffer_size': 'striping_unit'})

SPACES = {
    'paper12': PAPER12,
    'minimal': MINIMAL,
    'original': ORIGINAL,
}


def get_space(name):
    try:
        return SPACES[name]
    except KeyError:
        raise ValueError('unknown parameter space {0!r}; choose from {1}'
                         .format(name, ', '.join(sorted(SPACES))))


def coarsen(space, levels):
    """A copy of `space` with each parameter reduced to `levels` values.

    Sweeping a full grid is the only way to record a trace that a later search
    can replay without ever missing: replay has no way to invent a measurement
    it was not given.  A full grid over paper12 is 4.5x10^8 points, so the
    values have to be thinned first.

    Levels are spread evenly across each parameter's list, always keeping the
    first, the last, and the default.  A parameter with fewer values than
    `levels` is left alone.
    """
    if levels < 2:
        raise ValueError('levels must be at least 2')

    reduced = []
    for parameter in space.parameters:
        count = len(parameter.values)
        if count <= levels:
            reduced.append(parameter)
            continue
        step = (count - 1) / float(levels - 1)
        chosen = sorted({int(round(i * step)) for i in range(levels)})
        chosen = sorted(set(chosen) | {0, count - 1,
                                       parameter.default_index})
        reduced.append(Parameter(
            name=parameter.name, layer=parameter.layer,
            values=tuple(parameter.values[i] for i in chosen),
            default=parameter.default, element=parameter.element,
            group_pos=parameter.group_pos, shim=parameter.shim,
            hdf5_min=parameter.hdf5_min,
            note='{0} (thinned to {1} levels)'.format(parameter.note,
                                                      len(chosen))))
    return Space('{0}-L{1}'.format(space.name, levels), reduced,
                 ties=space.ties)


def grid(space, mask=None):
    """Every point in the space, for exhaustive sweeps.

    Tied and single-valued parameters are pinned so the same effective
    configuration is not emitted twice.  Check space.size() before calling
    this on paper12.
    """
    allowed = set(space.free_names if mask is None else mask)
    ranges = []
    for parameter in space.parameters:
        if parameter.name in allowed and len(parameter.values) > 1:
            ranges.append(range(len(parameter.values)))
        else:
            ranges.append((parameter.default_index,))
    return itertools.product(*ranges)
