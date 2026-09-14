"""Adapters for the applications this tuner drives.

A bare --app-cmd is enough when the application needs no input file and prints
its bandwidth in a format you can hand to --perf-regex.  h5bench needs more
than that, so it gets an adapter: something that writes h5bench's own config
file, builds the command line, and knows the shape of its output.

What is deliberately NOT used here is the `h5bench` Python runner.  It sets up
its own environment for the benchmark process (src/h5bench.py passes
env=self.vol_environment to Popen) and applies its own Lustre striping from the
"file-system" section of its JSON.  Both collide with what this tuner is doing:
the first can drop the LD_PRELOAD that injects our parameters, and the second
overwrites two of the parameters being tuned.  The individual pattern binaries
take a config file and an output path, so we call them directly.
"""

import os

# h5bench's own parser, commons/h5bench_util.c:
#
#   CFG_DELIMS is "=\n", so a line is KEY=VALUE with no spaces around the "=".
#   A line starting with '#' is skipped, but a BLANK line makes read_config()
#   return -1 -- strtok finds no token and the parser gives up.
#   _parse_val() only strips a trailing "#comment"; it does not trim spaces,
#   and MEM_PATTERN/FILE_PATTERN/READ_OPTION are compared against the raw
#   value, so "CONTIG " with a trailing space silently becomes PATTERN_INVALID.
#
# Everything below is written to respect all three.
CONFIG_FILENAME = 'h5bench.cfg'
DEFAULT_OUTPUT = 'h5bench.h5'

# commons/h5bench_util.h: PARTICLE_SIZE = 7 * sizeof(float) + sizeof(int)
PARTICLE_BYTES = 32


class Application:
    """Base class."""

    name = None
    perf_regex = None
    perf_unit = 'MB/s'

    def prepare(self, rundir):
        """Write whatever input files the application needs."""

    def preamble(self):
        """Shell lines to run before the application, inside its directory."""
        return []

    def command(self, mpi_cmd):
        raise NotImplementedError

    def describe(self):
        return ''


class H5BenchPattern(Application):
    """h5bench's VPIC-IO style write/read patterns.

    Bandwidth is scraped from the line h5bench prints last:

        SYNC Raw write rate: 1.621 GB/s

    Note the unit.  h5bench formats every rate through format_human_readable(),
    which divides by 1024 until the number fits and then prints the matching
    prefix, so the same benchmark reports MB/s on one machine and GB/s on a
    faster one.  A campaign that assumed a fixed unit would record a 1024-fold
    error exactly when a candidate got fast, which inverts the search.  The
    regex therefore captures the prefix and the scraper converts.

    Raw vs Observed: `Raw write rate` divides the bytes by the time spent in
    the write calls, `Observed write rate` divides by the whole loop including
    the emulated compute.  This adapter reports Raw and sets the emulated
    compute time to zero, which makes the two nearly identical and removes the
    choice from the experiment's description.  Compute time in the denominator
    would dilute exactly the effect being tuned.
    """

    OPERATIONS = {
        'write': 'h5bench_write',
        'read': 'h5bench_read',
    }

    def __init__(self, operation='write', binary=None, particles='2M',  # noqa: E501
                 timesteps=4, mem_pattern='CONTIG', file_pattern='CONTIG',
                 collective_data=True, output=DEFAULT_OUTPUT, extra=None,
                 csv=False):
        if operation not in self.OPERATIONS:
            raise ValueError('unknown h5bench operation {0!r}; choose from '
                             '{1}'.format(operation,
                                          ', '.join(sorted(self.OPERATIONS))))
        self.operation = operation
        self.binary = binary or self.OPERATIONS[operation]
        self.particles = str(particles)
        self.timesteps = int(timesteps)
        self.mem_pattern = mem_pattern
        self.file_pattern = file_pattern
        self.collective_data = collective_data
        self.output = output
        self.extra = dict(extra or {})
        self.csv = csv

        self.name = 'h5bench-{0}'.format(operation)
        verb = operation
        self.perf_regex = (
            r'Raw {0} rate:\s(?P<bw_{0}>[0-9.]+)\s(?P<bw_{0}_unit>[KMGT ])B/s'
            .format(verb))

    # -- input file ---------------------------------------------------------

    def settings(self):
        """The KEY=VALUE pairs, in the order they are written.

        COLLECTIVE_METADATA is fixed to NO and left to the tuner.  It maps onto
        H5Pset_coll_metadata_write, which is one of the twelve parameters being
        searched, and the shim rewrites the fapl inside its H5Fcreate hook --
        after h5bench has finished configuring it.  Setting it here would not
        change the run, only make the config file lie about it.

        COLLECTIVE_DATA is a dataset transfer property, which the shim does not
        touch, so it stays a property of the workload and is fixed for the
        whole campaign.
        """
        values = [
            ('IO_OPERATION', self.operation.upper()),
            ('MEM_PATTERN', self.mem_pattern),
            ('FILE_PATTERN', self.file_pattern),
            # The particle count comes from the dimensions, NOT from
            # NUM_PARTICLES.  read_config() overwrites num_particles with
            # dim_1 * dim_2 * dim_3 for every write operation, and the
            # defaults are 1 x 1 x 1 -- so a config that sets only
            # NUM_PARTICLES writes one particle per rank.  That is not a
            # hypothetical: it is what the first Nurion run did, 4 KB total
            # where 512 MB was intended, and the bandwidth came out in KB/s.
            ('NUM_DIMS', '1'),
            ('DIM_1', str(self.particle_count())),
            ('DIM_2', '1'),
            ('DIM_3', '1'),
            ('TIMESTEPS', str(self.timesteps)),
            # Raw and Observed rates coincide when this is zero.
            ('EMULATED_COMPUTE_TIME_PER_TIMESTEP', '0'),
            ('DELAYED_CLOSE_TIMESTEPS', '0'),
            ('COLLECTIVE_DATA', 'YES' if self.collective_data else 'NO'),
            ('COLLECTIVE_METADATA', 'NO'),
            ('COMPRESS', 'NO'),
        ]
        if self.csv:
            values.append(('CSV_FILE', 'h5bench.csv'))
        for key in sorted(self.extra):
            values = [(k, v) for k, v in values if k != key]
            values.append((key, str(self.extra[key])))
        return values

    def render_config(self):
        # No blank lines: read_config() treats one as a parse failure.
        lines = ['# generated by autotuner; do not add blank lines',
                 '# h5bench read_config() stops at the first line with no "="']
        for key, value in self.settings():
            lines.append('{0}={1}'.format(key, value))
        return '\n'.join(lines) + '\n'

    def prepare(self, rundir):
        path = os.path.join(rundir, CONFIG_FILENAME)
        with open(path, 'w') as stream:
            stream.write(self.render_config())
        return path

    # -- running ------------------------------------------------------------

    def preamble(self):
        """Remove the previous output file.

        Lustre applies striping when a file is created, so a candidate that
        reuses an existing file inherits the previous candidate's stripe
        layout and two of the twelve parameters stop having any effect.
        """
        if self.operation == 'read':
            return []
        return ['rm -f {0}'.format(self.output)]

    def command(self, mpi_cmd='mpirun'):
        parts = [mpi_cmd, self.binary, CONFIG_FILENAME, self.output]
        return ' '.join(part for part in parts if part)

    def particle_count(self):
        """Particles per rank, as a plain integer.

        The suffix is expanded here rather than passed through, because
        h5bench's parse_unit() splits the number from its unit on a SPACE:
        "512K" parses as 512 with no unit and is silently a thousandfold too
        small, while "512 K" works.  Writing the integer sidesteps the
        distinction entirely -- and it is not hypothetical, it is what the
        first real run did.
        """
        text = self.particles.strip().upper().replace(' ', '')
        scale = 1
        # h5bench_util.h: K_VAL/M_VAL/G_VAL are the 1024-based values.
        for suffix, factor in (('K', 1024), ('M', 1024 ** 2),
                               ('G', 1024 ** 3), ('T', 1024 ** 4)):
            if text.endswith(suffix):
                scale, text = factor, text[:-1]
                break
        try:
            return int(float(text)) * scale
        except ValueError:
            raise ValueError(
                'could not read a particle count from {0!r}'.format(
                    self.particles))

    def bytes_per_rank(self):
        """Bytes one rank writes across the campaign, or None if unparsable."""
        try:
            count = self.particle_count()
        except ValueError:
            return None
        return count * PARTICLE_BYTES * self.timesteps

    def describe(self, ranks=None):
        per_rank = self.bytes_per_rank()
        if per_rank is None:
            return self.name
        text = '{0}: {1} particles x {2} timesteps = {3:.1f} MiB/rank'.format(
            self.name, self.particles, self.timesteps,
            per_rank / (1024.0 * 1024.0))
        if ranks:
            text += ', {0:.1f} GiB total over {1} ranks'.format(
                per_rank * ranks / (1024.0 ** 3), ranks)
        return text


class MACSio(Application):
    """MACSio, driven directly rather than through the h5bench runner.

    This is the application the paper uses to evaluate I/O Discovery (IV-A),
    and on this machine it is already built as part of h5bench.

    Why it is a better tuning target than h5bench's write pattern: it exercises
    more of the twelve parameters.  h5bench write with COMPRESS=NO produces
    contiguous datasets, so `chunk_cache` does nothing, and its contiguous
    collective writes leave `sieve_buf_size` idle.  MACSio in SIF mode single-
    chunks its datasets and writes a mesh with substantial metadata, so the
    chunk cache, the sieve buffer, and all four metadata parameters are live.

    Note what is deliberately NOT passed.  MACSio's HDF5 plugin accepts
    --sieve_buf_size, --meta_block_size and --alignment itself.  Those are
    three of the parameters being tuned, and leaving them off the command line
    is what lets the shim own them.  Setting them here would have the plugin
    configure the fapl and the shim overwrite it moments later, which works but
    makes the command line a misleading record of the experiment.
    """

    name = 'macsio'
    # No regex: MACSio's own performance summary is not scraped.  Darshan
    # measures the I/O directly, which is both what the paper does and the only
    # option once I/O discovery strips an application's print statements.
    perf_regex = None

    def __init__(self, binary=None, part_size='16Mi', num_dumps=5,
                 avg_num_parts=1, part_dim=3, file_mode='SIF', file_count=1,
                 filebase='macsio', json_lib_dir=None, extra=None):
        self.binary = binary or 'macsio'
        self.part_size = str(part_size)
        self.num_dumps = int(num_dumps)
        self.avg_num_parts = avg_num_parts
        self.part_dim = int(part_dim)
        self.file_mode = file_mode
        self.file_count = int(file_count)
        self.filebase = filebase
        # MACSio built inside h5bench links libjson-cwx from the build tree,
        # which is not on the default library path.
        self.json_lib_dir = json_lib_dir
        self.extra = list(extra or [])

    def preamble(self):
        lines = []
        if self.json_lib_dir:
            lines.append('export LD_LIBRARY_PATH={0}:$LD_LIBRARY_PATH'.format(
                self.json_lib_dir))
        # Lustre applies striping at creation, so a stale file would hand the
        # next candidate the previous one's layout.
        lines.append('rm -f {0}*.h5'.format(self.filebase))
        return lines

    def command(self, mpi_cmd='mpirun'):
        parts = [mpi_cmd, self.binary,
                 '--interface', 'hdf5',
                 '--parallel_file_mode', self.file_mode, str(self.file_count),
                 '--part_size', self.part_size,
                 '--avg_num_parts', str(self.avg_num_parts),
                 '--part_dim', str(self.part_dim),
                 '--num_dumps', str(self.num_dumps),
                 '--filebase', self.filebase,
                 '--fileext', 'h5']
        parts.extend(self.extra)
        return ' '.join(part for part in parts if part)

    def _part_bytes(self):
        text = self.part_size.strip()
        scale = 1
        for suffix, factor in (('Ki', 1024), ('Mi', 1024 ** 2),
                               ('Gi', 1024 ** 3), ('K', 1000), ('M', 1000 ** 2),
                               ('G', 1000 ** 3)):
            if text.endswith(suffix):
                scale, text = factor, text[:-len(suffix)]
                break
        try:
            return int(float(text)) * scale
        except ValueError:
            return None

    def describe(self, ranks=None):
        per_part = self._part_bytes()
        if per_part is None:
            return self.name
        per_rank = per_part * float(self.avg_num_parts) * self.num_dumps
        text = ('macsio: {0}/part x {1} part(s)/rank x {2} dumps = '
                '{3:.1f} MiB/rank, {4} mode'.format(
                    self.part_size, self.avg_num_parts, self.num_dumps,
                    per_rank / (1024.0 ** 2), self.file_mode))
        if ranks:
            text += ', {0:.1f} GiB total over {1} ranks'.format(
                per_rank * ranks / (1024.0 ** 3), ranks)
        return text



class Exerciser(Application):
    """h5bench's exerciser kernel.

    Why this one is worth having next to the write pattern.  The exerciser is a
    single self-contained source file -- HDF5, MPI and libm are its only
    dependencies -- so it is the one benchmark besides the patterns that builds
    under the Intel toolchain without the CMake external fetches that stall
    (AMReX, openPMD, MACSio's json-cwx).  It uses the *synchronous* HDF5 API,
    so the shim's original H5Fcreate/H5Dcreate/H5Dwrite hooks fire; the async
    hooks added for h5bench_write are not needed here.

    And it varies the access pattern from the command line.  `--numdims` and
    `--dimranks` reshape the hyperslab, which moves the optimum: the 2026-03
    characterisation found the best Lustre stripe to be 4 MiB x 4 for a 1D
    array, 64 MiB x 16 for 2D independent, and 1 MiB x 1 for 3D collective
    (04-누리온-환경.md 7g).  That is a distribution shift available inside
    one binary, which is what a transfer experiment needs.

    Bandwidth.  The kernel prints a five-row table of moments over its ten
    internal iterations; `RawWrBDWTH` is bytes over the time inside H5Dwrite,
    MPI_Reduce'd with MPI_MIN across ranks, so each iteration's figure is the
    aggregate rate set by the slowest rank.  We scrape the `Avg` row, the mean
    of those ten.  Note the unit: the kernel divides by 1048576, so the number
    is MiB/s, not MB/s -- 4.9% above the MB/s value.  Ratios are unaffected,
    which is what the objective uses, but absolute figures are not comparable
    with the patterns' `Raw write rate` without converting.

    Two traps are encoded below rather than left to the caller.

    `--numdims` must precede `--minels`, `--dimranks` and `--bufmult`; the
    kernel checks and calls exit(-1) if it does not, because each of those
    reads exactly numDims integers off the argument list.

    `--metacoll` is not passed.  It makes the kernel call
    H5Pset_coll_metadata_write and H5Pset_all_coll_metadata_ops itself, and
    both are parameters being tuned.  The shim applies its values when
    H5Fcreate is intercepted, which is after the kernel has configured the
    fapl, so the shim would win -- but the command line would then be a
    misleading record of the experiment.  Same reasoning as MACSio above.
    """

    name = 'exerciser'
    # Avg row, fourth column: Metric, Bufsize, H5DWrite, RawWrBDWTH.
    # No unit suffix in the output, so no unit capture group is needed; the
    # kernel always prints MiB/s.
    perf_regex = (r'^Avg\s+(?P<bufsize>\d+)\s+[0-9.]+\s+'
                  r'(?P<bw_write>[0-9.]+)')
    perf_unit = 'MB/s'

    # The kernel writes doubles.
    ELEMENT_BYTES = 8
    # NUM_ITERATIONS in h5bench_exerciser.c.
    ITERATIONS = 10
    # sprintf(testFileName, "hdf5TestFile-%d", rand()) into the working
    # directory, seeded from the clock, so the name is neither fixed nor
    # configurable.
    OUTPUT_GLOB = 'hdf5TestFile-*'

    def __init__(self, binary=None, numdims=1, minels=None, dimranks=None,
                 nsizes=1, bufmult=None, indepio=False, usechunked=False,
                 derivedtype=False, addattr=False, keepfile=False,
                 extra=None):
        self.binary = binary or 'h5bench_exerciser'
        self.numdims = int(numdims)
        if not 1 <= self.numdims <= 4:
            raise ValueError('exerciser supports 1 to 4 dimensions, not '
                             '{0}'.format(self.numdims))
        self.minels = self._per_dim(minels, 'minels', default=16777216)
        self.dimranks = self._per_dim(dimranks, 'dimranks', default=None)
        self.bufmult = self._per_dim(bufmult, 'bufmult', default=1)
        self.nsizes = int(nsizes)
        self.indepio = bool(indepio)
        self.usechunked = bool(usechunked)
        self.derivedtype = bool(derivedtype)
        self.addattr = bool(addattr)
        # Off by default: a campaign of hundreds of candidates would otherwise
        # leave one file per run on /scratch.
        self.keepfile = bool(keepfile)
        self.extra = list(extra or [])

    def _per_dim(self, value, label, default):
        """Expand a scalar to one value per dimension, or check a list."""
        if value is None:
            if default is None:
                return None
            value = default
        if isinstance(value, (list, tuple)):
            values = [int(item) for item in value]
            if len(values) != self.numdims:
                raise ValueError(
                    '--{0} needs exactly {1} value(s) for numdims={1}, got '
                    '{2}'.format(label, self.numdims, len(values)))
            return values
        return [int(value)] * self.numdims

    def preamble(self):
        # Lustre fixes a file's layout when it is created, so a leftover from
        # a candidate that crashed before unlink() would hand the next one the
        # previous stripe.
        return ['rm -f {0}'.format(self.OUTPUT_GLOB)]

    def command(self, mpi_cmd='mpirun'):
        if self.dimranks is None:
            raise ValueError('exerciser needs --exerciser-dimranks; their '
                             'product must equal the rank count')
        # --numdims first: the kernel exits(-1) if the per-dimension flags are
        # parsed before it knows how many values to read.
        parts = [mpi_cmd, self.binary, '--numdims', str(self.numdims)]
        parts.extend(['--minels'] + [str(n) for n in self.minels])
        parts.extend(['--nsizes', str(self.nsizes)])
        parts.extend(['--bufmult'] + [str(n) for n in self.bufmult])
        parts.extend(['--dimranks'] + [str(n) for n in self.dimranks])
        if self.indepio:
            parts.append('--indepio')
        if self.usechunked:
            parts.append('--usechunked')
        if self.derivedtype:
            parts.append('--derivedtype')
        if self.addattr:
            parts.append('--addattr')
        if self.keepfile:
            parts.append('--keepfile')
        parts.extend(self.extra)
        return ' '.join(part for part in parts if part)

    def ranks(self):
        """Rank count the kernel expects: the product of --dimranks."""
        if self.dimranks is None:
            return None
        total = 1
        for value in self.dimranks:
            total *= value
        return total

    def _bytes_per_rank(self):
        elements = 1
        for count in self.minels:
            elements *= count
        return elements * self.ELEMENT_BYTES

    def describe(self, ranks=None):
        per_rank = self._bytes_per_rank()
        expected = self.ranks()
        text = ('exerciser: {0}D, {1} MiB/rank x {2} iterations, dimranks '
                '{3}, {4} I/O'.format(
                    self.numdims, per_rank / (1024.0 ** 2), self.ITERATIONS,
                    'x'.join(str(n) for n in self.dimranks or []),
                    'independent' if self.indepio else 'collective'))
        if expected:
            total = per_rank * expected * self.ITERATIONS
            text += ', {0:.1f} GiB total over {1} ranks'.format(
                total / (1024.0 ** 3), expected)
        if ranks and expected and ranks != expected:
            text += ('  [WARNING: launcher gives {0} ranks but --dimranks '
                     'multiplies to {1}]'.format(ranks, expected))
        return text

def h5bench_write(**kwargs):
    return H5BenchPattern(operation='write', **kwargs)


def h5bench_read(**kwargs):
    return H5BenchPattern(operation='read', **kwargs)


def macsio(**kwargs):
    return MACSio(**kwargs)


def exerciser(**kwargs):
    return Exerciser(**kwargs)


APPS = {
    'h5bench-write': h5bench_write,
    'h5bench-read': h5bench_read,
    'macsio': macsio,
    'exerciser': exerciser,
}
