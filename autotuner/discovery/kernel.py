"""Turn a source file into an I/O kernel.

TunIO III-B end to end: format, parse, mark, reconstruct, reduce.

    "To reduce source code to a kernel, the Application I/O Discovery component
    generates an Abstract Syntax Tree (AST), finds and marks I/O calls and
    related code in a marking loop, reconstructs the kernel from kept lines,
    and reduces the kernel by transforming those lines, ultimately outputting
    an I/O kernel to the tuner."

The safety property the paper claims is the one worth preserving: "There is no
chance of an increase in the application runtime, as the number of instructions
run will be the same or less."  Every step here only ever removes lines or
lowers a loop bound, so a kernel cannot be slower than its original.  And when
the result will not compile, the caller is told to fall back to the full
application -- also the paper's behaviour: "if the I/O kernel of the
application causes an error, TunIO will revert to using the full application."
"""

import os
import shutil
import subprocess
import sys

from . import marking

# "a custom clang-format preprocessing step which avoids line breaking with a
# 200-character column limit while placing curly braces on distinct lines and
# splitting multi-statement lines" (III-B).
CLANG_FORMAT_STYLE = (
    '{BasedOnStyle: LLVM, ColumnLimit: 200, BreakBeforeBraces: Allman, '
    'AllowShortIfStatementsOnASingleLine: false, '
    'AllowShortLoopsOnASingleLine: false, '
    'AllowShortFunctionsOnASingleLine: None, '
    'AllowShortBlocksOnASingleLine: false, '
    'AllowShortCaseLabelsOnASingleLine: false}'
)

DEFAULT_CLANG_ARGS = ('-x', 'c')

# Cached result of probing the host toolchain; see _system_include_args.
_SYSTEM_ARGS = None


def _system_include_args():
    """Include paths libclang needs but does not know about.

    The libclang shipped by pip is a bare library with no resource directory,
    so it cannot find even stddef.h -- and a missing stddef.h is a parse error,
    which by the rule above aborts discovery.  The host clang knows where its
    builtin headers live, and on macOS the SDK has to be named explicitly too.

    Probing once and caching is deliberate: discover() may be called per source
    file, and shelling out to clang twice per file is pure overhead.
    """
    global _SYSTEM_ARGS
    if _SYSTEM_ARGS is not None:
        return _SYSTEM_ARGS

    args = []
    clang = shutil.which('clang')
    if clang:
        try:
            resource = subprocess.check_output(
                [clang, '-print-resource-dir'],
                stderr=subprocess.DEVNULL).decode().strip()
            builtin = os.path.join(resource, 'include')
            if os.path.isdir(builtin):
                args += ['-isystem', builtin]
        except (subprocess.CalledProcessError, OSError):
            pass

    if sys.platform == 'darwin':
        try:
            sdk = subprocess.check_output(
                ['xcrun', '--show-sdk-path'],
                stderr=subprocess.DEVNULL).decode().strip()
            if os.path.isdir(sdk):
                args += ['-isysroot', sdk]
        except (subprocess.CalledProcessError, OSError):
            pass

    _SYSTEM_ARGS = args
    return args


class DiscoveryError(Exception):
    """The kernel could not be produced; the caller should use the full app."""


class Kernel:
    """The result of reducing one source file."""

    def __init__(self, source, path, marks, total_lines, formatted,
                 reductions=()):
        self.source = source
        self.path = path
        self.marks = marks
        self.total_lines = total_lines
        self.formatted = formatted
        self.reductions = list(reductions)

    @property
    def kept_lines(self):
        return len(self.marks)

    @property
    def io_calls(self):
        return self.marks.io_calls

    def report(self):
        kept = self.kept_lines
        total = self.total_lines
        share = (100.0 * kept / total) if total else 0.0
        lines = ['{0}: {1} of {2} lines kept ({3:.1f}%), {4} I/O calls'.format(
            os.path.basename(self.path), kept, total, share,
            len(self.io_calls))]
        if not self.formatted:
            lines.append(
                'WARNING: clang-format was not run. The paper formats the '
                'source first so that one line holds one statement; without '
                'it a line carrying both I/O and compute keeps the compute '
                'too, and the kernel is larger than it needs to be.')
        for note in self.reductions:
            lines.append('reduction: ' + note)
        return '\n'.join(lines)


def _run_clang_format(text, binary):
    process = subprocess.Popen(
        [binary, '--assume-filename=kernel.c',
         '--style=' + CLANG_FORMAT_STYLE],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = process.communicate(text.encode('utf-8'))
    if process.returncode != 0:
        raise DiscoveryError('clang-format failed: {0}'.format(
            err.decode('utf-8', 'replace').strip()))
    return out.decode('utf-8', 'replace')


def _load_cindex():
    """Import clang.cindex, with a message that says what to install."""
    try:
        import clang.cindex as cindex
    except ImportError as error:
        raise DiscoveryError(
            'I/O discovery needs the clang python bindings: '
            'pip install libclang  ({0})'.format(error))
    return cindex


def _find_clang_format():
    for name in ('clang-format',):
        found = shutil.which(name)
        if found:
            return found
    for candidate in (
            '/Library/Developer/CommandLineTools/usr/bin/clang-format',
            '/usr/bin/clang-format', '/opt/homebrew/bin/clang-format'):
        if os.path.exists(candidate):
            return candidate
    return None


def discover(path, workdir, io_prefixes=None, clang_args=(),
             keep_regions=(), use_clang_format=True,
             allow_parse_errors=False):
    """Reduce `path` to an I/O kernel and return a Kernel.

    The formatted source is written next to the kernel so that reported line
    numbers refer to something on disk.  Without that, a marking bug is
    untraceable: the numbers would point into a buffer that no longer exists.
    """
    cindex = _load_cindex()

    with open(path) as stream:
        original = stream.read()

    formatted = False
    text = original
    if use_clang_format:
        binary = _find_clang_format()
        if binary:
            text = _run_clang_format(original, binary)
            formatted = True

    base = os.path.splitext(os.path.basename(path))[0]
    formatted_path = os.path.join(workdir, base + '.formatted.c')
    with open(formatted_path, 'w') as stream:
        stream.write(text)

    args = (list(DEFAULT_CLANG_ARGS) + _system_include_args()
            + list(clang_args))
    index = cindex.Index.create()
    try:
        unit = index.parse(formatted_path, args=args)
    except Exception as error:                      # noqa: BLE001
        raise DiscoveryError('clang could not parse {0}: {1}'.format(
            path, error))

    # Parse errors are fatal by default, and that is not conservatism for its
    # own sake.  clang cannot build an AST node for a call whose types it does
    # not know: `hid_t f = H5Fcreate(...)` vanishes from the tree completely
    # when hid_t is undeclared -- not as an unresolved node, but as nothing at
    # all.  A run with a missing hdf5.h therefore reports "no I/O calls found"
    # on a file full of them, or worse, finds some and silently misses others.
    # Failing here is what makes the component trustworthy.
    fatal = [d for d in unit.diagnostics
             if d.severity >= cindex.Diagnostic.Error]
    if fatal and not allow_parse_errors:
        detail = '; '.join(d.spelling for d in fatal[:3])
        raise DiscoveryError(
            '{0} clang error(s) parsing {1}: {2}\n'
            '  I/O calls whose types are unknown do not appear in the AST at '
            'all, so discovery cannot see them. Pass the include paths with '
            '--clang-arg -I<dir> (on the cluster, -I$HDF5ROOT/include; for '
            'testing, -Itest/stubs). Use --allow-parse-errors to override, '
            'but the resulting kernel is not trustworthy.'.format(
                len(fatal), os.path.basename(path), detail))

    marker = marking.Marker(cindex, unit, formatted_path,
                            io_prefixes=io_prefixes,
                            keep_regions=keep_regions)
    marks = marker.run()

    if not marks.io_calls:
        raise DiscoveryError(
            'no I/O calls found in {0}. Expected calls beginning with {1}; '
            'if the application uses a different I/O library, pass its '
            'prefix.'.format(path, ', '.join(
                io_prefixes or marking.DEFAULT_IO_PREFIXES)))

    kernel = Kernel(source=text, path=formatted_path, marks=marks,
                    total_lines=len(text.splitlines()), formatted=formatted)
    if fatal:
        kernel.reductions.append(
            '{0} clang error(s) during parse were overridden; the kernel may '
            'be missing I/O calls.'.format(len(fatal)))
    return kernel


def reconstruct(kernel):
    """Emit the kernel source from the kept lines.

    Unkept lines are replaced by nothing rather than being deleted, in the
    sense that the output carries a marker comment where a run of lines was
    dropped.  Line numbers no longer match the original either way, and a
    reader of the kernel needs to see that something was removed.
    """
    out = []
    dropped = 0
    for number, line in enumerate(kernel.source.splitlines(), 1):
        if number in kernel.marks:
            if dropped:
                out.append('/* --- {0} line(s) removed by I/O discovery '
                           '--- */'.format(dropped))
                dropped = 0
            out.append(line)
        elif line.strip():
            dropped += 1
    if dropped:
        out.append('  /* --- {0} line(s) removed by I/O discovery '
                   '--- */'.format(dropped))
    return '\n'.join(out) + '\n'
