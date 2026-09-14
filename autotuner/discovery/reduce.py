"""The two optional reduction steps from TunIO III-B.

    "Once the kept lines have been discovered for all I/O calls, the code is
    reconstructed with those lines and reduced using a technique such as loop
    reduction or I/O path switching.  These exist to further improve the
    runtime of the application at the cost of the accuracy of its I/O kernel,
    and they are optional to apply (a null reduction step could be used
    instead)."

Both trade accuracy for speed, in different directions, and the paper is clear
about which:

    loop reduction      "trade some information about hardware locality and
                        caching"
    path switching      "loses some accuracy due to not tuning for the target
                        storage device"

Loop reduction is where the paper's headline number comes from.  Peak RoTI goes
from 2.47 on the full application to 2.87 on the kernel -- and to 23.30 once
loops run 1% of their iterations.  Nearly all of the gain is here, not in the
AST work.

A note on the scaling factor.  The paper multiplies "scalable metrics" by the
reduction to predict the original loop, and reports bytes-written accurate to
0.19%.  This tuner does not need that correction: its objective is bandwidth,
a ratio, and both numerator and denominator shrink together -- the paper
measures the reported bandwidths as 97.10% accurate under 1% loop reduction.
The write share alpha is a ratio too.  So no rescaling is applied here, and
what would need it is absolute byte counts, which nothing in the objective
reads.
"""

import re

# Named so a reader of a generated kernel can tell what put them there, and so
# they cannot collide with application identifiers.
COUNTER_PREFIX = '__at_loop_'
GUARD_COMMENT = '/* loop reduction: TunIO III-B */'


class Reduction:
    """What a reduction did, for the report."""

    def __init__(self, name, detail):
        self.name = name
        self.detail = detail

    def __str__(self):
        return '{0}: {1}'.format(self.name, self.detail)


def _leading_space(line):
    return line[:len(line) - len(line.lstrip())]


def reduce_loops(kernel, source, keep_iterations):
    """Bound every loop that encloses I/O to `keep_iterations` passes.

    The paper specifies a percentage of the original iteration count.  A
    percentage cannot be applied here: the bound is almost always a runtime
    variable -- h5bench takes TIMESTEPS from its config file, so `t <
    timesteps` has no compile-time value to take 1% of.  An absolute cap is
    used instead, injected as a guard at the top of the body rather than by
    rewriting the loop condition, which would mean parsing arbitrary C
    expressions.

    The effect matches the paper's intent for the case it cares about: a
    100-timestep loop capped at 1 iteration is the 1% reduction it describes.
    Where they differ is a loop whose trip count is smaller than the cap, which
    is then untouched -- and the paper notes the same limitation, that "whenever
    the loop iterations are too small to reduce ... loop reduction will not be
    able to do anything".
    """
    if not kernel.marks.io_loops or keep_iterations is None:
        return source, None

    lines = source.splitlines()
    # The reconstructed text has different line numbers from the formatted
    # source the marks refer to, so brace lines are matched by their content in
    # order rather than by number.
    brace_lines = sorted(brace for brace, _header, _kind in kernel.marks.io_loops)
    wanted = len(brace_lines)

    declarations = []
    out = []
    injected = 0
    original_numbers = _map_to_original(kernel, source)

    for index, line in enumerate(lines):
        out.append(line)
        number = original_numbers.get(index)
        if number is not None and number in brace_lines:
            counter = '{0}{1}'.format(COUNTER_PREFIX, injected)
            declarations.append('static long {0} = 0;'.format(counter))
            indent = _leading_space(line) + '    '
            out.append('{0}if ({1}++ >= {2}) break;  {3}'.format(
                indent, counter, keep_iterations, GUARD_COMMENT))
            injected += 1

    if not injected:
        return source, None

    text = '\n'.join(out) + '\n'
    text = _insert_declarations(text, declarations)
    detail = ('{0} of {1} I/O loop(s) capped at {2} iteration(s)'.format(
        injected, wanted, keep_iterations))
    if injected < wanted:
        detail += ('; {0} could not be located in the reconstructed source'
                   .format(wanted - injected))
    return text, Reduction('loop-reduction', detail)


def _map_to_original(kernel, reconstructed):
    """Map reconstructed line indices back to formatted-source line numbers.

    reconstruct() emits kept lines in order and a comment for each dropped run,
    so walking both in step recovers the correspondence.  Doing it this way
    keeps reconstruct() free of bookkeeping that only the reducer needs.
    """
    kept = sorted(kernel.marks.lines)
    mapping = {}
    position = 0
    for index, line in enumerate(reconstructed.splitlines()):
        if line.lstrip().startswith('/* --- ') and 'removed by I/O' in line:
            continue
        if position < len(kept):
            mapping[index] = kept[position]
            position += 1
    return mapping


def _insert_declarations(text, declarations):
    """Put the counters after the last preprocessor line at the top."""
    lines = text.splitlines()
    insert_at = 0
    for index, line in enumerate(lines):
        if line.strip().startswith('#'):
            insert_at = index + 1
        elif line.strip() and insert_at:
            break
    block = ['', '/* loop reduction counters, added by I/O discovery */']
    block.extend(declarations)
    block.append('')
    return '\n'.join(lines[:insert_at] + block + lines[insert_at:]) + '\n'


def switch_io_paths(kernel, source, memory_path):
    """Rewrite the paths I/O calls open so they land in memory.

    "prepends every path written or read with a path to memory (e.g.,
    /dev/shm or tmpfs) so that calls are not actually performed to slow disks"

    Only literals that appeared as arguments to I/O calls are touched, so a
    format string or a dataset name inside the file is left alone.  An absolute
    path has its leading slash dropped before joining, otherwise the prefix
    would be discarded by the filesystem.
    """
    if not memory_path or not kernel.marks.path_literals:
        return source, None

    wanted = {}
    for _line, _column, spelling in kernel.marks.path_literals:
        text = spelling.strip()
        if not (text.startswith('"') and text.endswith('"')):
            continue
        inner = text[1:-1]
        # Dataset and group names inside an HDF5 file are also string
        # arguments; they are not filesystem paths and must not be rewritten.
        # A filesystem path here is one that names a file, i.e. has a suffix.
        if '.' not in inner.rsplit('/', 1)[-1]:
            continue
        wanted[text] = '"{0}/{1}"'.format(memory_path.rstrip('/'),
                                          inner.lstrip('/'))

    if not wanted:
        return source, None

    rewritten = 0
    out = source
    for original, replacement in wanted.items():
        count = out.count(original)
        if count:
            out = out.replace(original, replacement)
            rewritten += count

    if not rewritten:
        return source, None
    return out, Reduction(
        'io-path-switching',
        '{0} path literal(s) redirected to {1}'.format(rewritten, memory_path))


def apply_reductions(kernel, source, keep_iterations=None, memory_path=None):
    """Run the requested reductions in order and return (source, notes)."""
    notes = []
    if keep_iterations is not None:
        source, note = reduce_loops(kernel, source, keep_iterations)
        if note:
            notes.append(note)
    if memory_path:
        source, note = switch_io_paths(kernel, source, memory_path)
        if note:
            notes.append(note)
    return source, notes
