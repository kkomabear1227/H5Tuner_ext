"""Find the lines of a source file that its I/O depends on.

This is TunIO III-B, the Application I/O Discovery component.  The paper's
description is specific enough to follow directly, so this module follows it
rather than inventing an equivalent:

    "the marking loop traverses the AST, finds I/O calls, and marks them to
    keep.  TunIO continues by marking related source code elements as
    dependents of the kept lines."

Four dependent rules, quoted from III-B:

    function call   its arguments and the left-hand side of its assignment
                    (dep = foo(dep, dep, ...))
    conditional     the boolean statement (if (dep) {body})
    loop            its initialization, update, and condition statements
                    (for (dep; dep; dep) {body})
    assignment      its left-hand side (dep = rhs)

Plus two closures:

    "Whenever a variable is marked, a backward traversal must be applied to
    mark all assignments associated with that variable (e.g., var = dep)."

    "TunIO uses the AST to find and mark the contextual parent of each
    dependent ... the marking loop will continue until it reaches the source
    code's top-level."

Why lines rather than statements, also the paper's choice: "Clang provides a
large amount of nuance which makes it difficult to isolate statements, and
therefore TunIO uses lines as a substitute."  A line is a coarse unit, so the
kernel keeps a little more than it strictly needs.  That direction is safe --
the paper notes the kernel can only be the same size or smaller than the
original, never larger.

Block structure is preserved, not just the marked lines: keeping a `for` header
without its closing brace would produce a kernel that does not compile.  This is
why the paper runs clang-format first with braces on their own lines.
"""

import os

# Prefix that identifies an I/O call.  HDF5 in the reference implementation:
# "In the prototype implementation of TunIO, the I/O calls are HDF5 calls."
DEFAULT_IO_PREFIXES = ('H5',)

# Calls that are I/O by name but carry no data and cost nothing to keep out.
# Not in the paper; dropping them shrinks the kernel without changing its I/O.
# Kept conservative: anything that opens, creates, writes, reads, flushes or
# closes stays.
IGNORED_SUFFIXES = ()

# Nodes whose children must NOT be pulled in automatically.
#
# This is the difference between a kernel and a copy of the application.  The
# marking loop reaches a block by marking the parent of something inside it --
# that is how a `for` header and its braces survive.  But a block is not a
# dependency of its own contents' siblings: if marking a compound statement
# also marked every statement in it, one H5Dwrite deep in main() would drag in
# the whole function, and from there the whole file.  The paper's four rules
# name specific children (arguments, conditions, left-hand sides); nothing in
# them says a block depends on its statements.
NO_AUTO_CHILDREN = frozenset((
    'COMPOUND_STMT',
    'FUNCTION_DECL',
    'TRANSLATION_UNIT',
))


class Marks:
    """Which lines survive, and why.

    Keeping the reason is not decoration.  A kernel that comes out wrong is
    debugged by asking which I/O call dragged in a line, and that question is
    unanswerable after the fact otherwise.
    """

    def __init__(self):
        self.lines = set()
        self.reasons = {}
        self.io_calls = []
        # Loops that enclose an I/O call, as (opening-brace line, header line,
        # kind).  Loop reduction needs the brace line to inject a bound after.
        self.io_loops = []
        # String literals passed to I/O calls, as (line, column, text).  I/O
        # path switching rewrites these.
        self.path_literals = []

    def keep(self, line, reason):
        if line is None or line < 1:
            return False
        fresh = line not in self.lines
        self.lines.add(line)
        self.reasons.setdefault(line, reason)
        return fresh

    def keep_range(self, start, end, reason):
        if start is None or end is None:
            return
        for line in range(start, end + 1):
            self.keep(line, reason)

    def __contains__(self, line):
        return line in self.lines

    def __len__(self):
        return len(self.lines)


def _kind(cursor):
    return cursor.kind.name


def _line_span(cursor):
    extent = cursor.extent
    if extent is None or extent.start.file is None:
        return None, None
    return extent.start.line, extent.end.line


def _in_file(cursor, path):
    """True when the cursor belongs to the file being reduced.

    Headers are pulled in by the parse and must not be marked: the kernel is a
    rewrite of one translation unit's source, not of everything it includes.
    """
    location = cursor.location
    if location is None or location.file is None:
        return False
    return os.path.realpath(location.file.name) == path


class Marker:
    """Runs the paper's marking loop over one translation unit."""

    def __init__(self, cindex, translation_unit, path, io_prefixes=None,
                 keep_regions=()):
        self.ci = cindex
        self.tu = translation_unit
        self.path = os.path.realpath(path)
        self.io_prefixes = tuple(io_prefixes or DEFAULT_IO_PREFIXES)
        # "Options may include manually indicated keep regions" (III-E).
        self.keep_regions = list(keep_regions)

        self.marks = Marks()
        # Functions called from the I/O path that contain no I/O themselves.
        # These are kept entire rather than reduced: see _full_keep below.
        self._full_keep = set()
        self._parent = {}
        self._assignments = {}
        self._children = {}
        self._queued = set()
        self._index()

    # -- indexing -----------------------------------------------------------

    def _index(self):
        """Build the parent map and the variable-to-assignments map.

        Both are needed before marking starts.  clang gives semantic and
        lexical parents but not the syntactic one the paper means by
        "contextual parent", so it is recorded during a single walk.
        """
        stack = [(self.tu.cursor, None)]
        while stack:
            cursor, parent = stack.pop()
            key = self._key(cursor)
            if parent is not None:
                self._parent[key] = parent
            children = list(cursor.get_children())
            self._children[key] = children
            for child in children:
                stack.append((child, cursor))

            if _in_file(cursor, self.path):
                self._index_assignment(cursor)

    @staticmethod
    def _key(cursor):
        return (cursor.hash, cursor.extent.start.offset,
                cursor.extent.end.offset)

    def _index_assignment(self, cursor):
        """Record `cursor` as writing to a variable, if it does.

        Covers `x = ...`, `x += ...`, and `int x = ...`.  The variable is keyed
        by its declaration cursor so that two locals of the same name in
        different scopes stay apart.
        """
        kind = _kind(cursor)
        target = None

        if kind == 'VAR_DECL':
            target = cursor
        elif kind in ('BINARY_OPERATOR', 'COMPOUND_ASSIGNMENT_OPERATOR'):
            if kind == 'BINARY_OPERATOR' and not self._is_assignment(cursor):
                return
            children = list(cursor.get_children())
            if children:
                target = self._referenced_decl(children[0])

        if target is None:
            return
        usr = target.get_usr() or target.spelling
        self._assignments.setdefault(usr, []).append(cursor)

    @staticmethod
    def _is_assignment(cursor):
        """True when a BINARY_OPERATOR is a plain `=`.

        clang's python bindings do not expose the operator, so the tokens are
        inspected.  The first `=` that is not part of ==, !=, <=, >= separates
        the operands.
        """
        depth = 0
        for token in cursor.get_tokens():
            spelling = token.spelling
            if spelling in ('(', '[' ):
                depth += 1
            elif spelling in (')', ']'):
                depth -= 1
            elif depth == 0 and spelling == '=':
                return True
            elif depth == 0 and spelling in ('==', '!=', '<=', '>=', '&&',
                                             '||', '?'):
                return False
        return False

    def _referenced_decl(self, cursor):
        """Walk down to the variable a expression ultimately refers to."""
        seen = 0
        while cursor is not None and seen < 32:
            kind = _kind(cursor)
            if kind == 'DECL_REF_EXPR':
                return cursor.referenced
            if kind == 'MEMBER_REF_EXPR':
                return cursor.referenced
            children = list(cursor.get_children())
            if not children:
                return None
            # Array subscripts and casts wrap the base on the left.
            cursor = children[0]
            seen += 1
        return None

    # -- the loop -----------------------------------------------------------

    def run(self):
        """Mark every line the I/O depends on and return the Marks."""
        worklist = []

        for cursor in self._walk():
            if self._is_io_call(cursor):
                self.marks.io_calls.append(
                    (cursor.spelling, cursor.location.line))
                self._enqueue(worklist, cursor, 'io-call')

        # Manual keep regions are marked outright and also seeded, so that
        # whatever they reference is pulled in like an I/O call's dependents.
        for start, end in self.keep_regions:
            self.marks.keep_range(start, end, 'keep-region')
            for cursor in self._walk():
                line = cursor.location.line if cursor.location else None
                if line is not None and start <= line <= end:
                    self._enqueue(worklist, cursor, 'keep-region')

        while worklist:
            cursor = worklist.pop()
            self._mark_self(cursor)
            for dependent in self._dependents(cursor):
                self._enqueue(worklist, dependent, 'dependent')
            parent = self._contextual_parent(cursor)
            if parent is not None:
                self._enqueue(worklist, parent, 'parent')

        self._keep_called_definitions()
        self._keep_preprocessor()
        self._collect_reduction_targets()
        return self.marks

    def _collect_reduction_targets(self):
        """Record what the optional reduction steps need to know.

        Gathered here rather than in the reducer because it needs the AST, and
        the reducer works on reconstructed text.
        """
        seen_loops = set()
        for cursor in self._walk():
            if self._is_io_call(cursor):
                for loop in self._enclosing_loops(cursor):
                    body = self._body_of(loop)
                    if body is None:
                        continue
                    header, _ = _line_span(loop)
                    brace, _ = _line_span(body)
                    if brace in seen_loops:
                        continue
                    seen_loops.add(brace)
                    self.marks.io_loops.append((brace, header, _kind(loop)))

                for literal in self._string_arguments(cursor):
                    location = literal.location
                    self.marks.path_literals.append(
                        (location.line, location.column, literal.spelling))

    def _enclosing_loops(self, cursor):
        """Every loop between this cursor and the top level, innermost first."""
        out = []
        parent = self._parent.get(self._key(cursor))
        while parent is not None and _kind(parent) != 'TRANSLATION_UNIT':
            if _kind(parent) in ('FOR_STMT', 'WHILE_STMT', 'DO_STMT'):
                out.append(parent)
            parent = self._parent.get(self._key(parent))
        return out

    def _string_arguments(self, cursor):
        """String literals among a call's arguments, however nested."""
        out = []
        for argument in cursor.get_arguments():
            stack = [argument]
            while stack:
                node = stack.pop()
                if _kind(node) == 'STRING_LITERAL':
                    out.append(node)
                    continue
                stack.extend(node.get_children())
        return out

    def _keep_called_definitions(self):
        """Keep the whole body of helper functions the kernel still calls.

        A call that survives needs its callee to exist, or the kernel does not
        link.  Reducing the callee recursively would give a smaller kernel, but
        it would also change what the callee computes, and the data a kernel
        writes is not always irrelevant -- with compression enabled it decides
        the bandwidth.  So a called helper is kept entire.

        The paper does not address this; its own future work lists "simulating
        necessary compute" as open. Keeping the body is the conservative
        reading, and it preserves the paper's guarantee that a kernel runs no
        more instructions than the original.
        """
        for cursor in list(self._full_keep):
            start, end = _line_span(cursor)
            self.marks.keep_range(start, end, 'called-definition')

    def _keep_preprocessor(self):
        """Keep every preprocessor line.

        #include and #define have no cursors in the AST -- clang has already
        consumed them by the time the tree exists -- so no amount of marking
        reaches them, and a kernel without its includes cannot compile.  They
        are also free to keep: a directive runs no instructions.
        """
        try:
            with open(self.path) as stream:
                lines = stream.read().splitlines()
        except (IOError, OSError):
            return
        continued = False
        for number, text in enumerate(lines, 1):
            stripped = text.strip()
            if continued or stripped.startswith('#'):
                self.marks.keep(number, 'preprocessor')
                continued = stripped.endswith('\\')
            else:
                continued = False

    def _walk(self):
        stack = [self.tu.cursor]
        while stack:
            cursor = stack.pop()
            if _in_file(cursor, self.path):
                yield cursor
            stack.extend(self._children.get(self._key(cursor), ()))

    def _enqueue(self, worklist, cursor, reason):
        if cursor is None or not _in_file(cursor, self.path):
            return
        key = self._key(cursor)
        if key in self._queued:
            return
        self._queued.add(key)
        worklist.append(cursor)
        start, _end = _line_span(cursor)
        if start is not None:
            self.marks.reasons.setdefault(start, reason)

    def _is_io_call(self, cursor):
        if _kind(cursor) != 'CALL_EXPR':
            return False
        name = cursor.spelling or ''
        if not name.startswith(self.io_prefixes):
            return False
        return not name.endswith(IGNORED_SUFFIXES) if IGNORED_SUFFIXES else True

    # -- marking one cursor -------------------------------------------------

    def _mark_self(self, cursor):
        """Mark the lines this cursor occupies.

        A compound statement is the exception: marking its whole extent would
        keep every line of a function body.  Only its braces are kept, so that
        a kept statement inside it stays syntactically enclosed.
        """
        kind = _kind(cursor)
        start, end = _line_span(cursor)
        if start is None:
            return

        if kind in ('COMPOUND_STMT',):
            self.marks.keep(start, 'block-open')
            self.marks.keep(end, 'block-close')
            return

        if kind in ('FOR_STMT', 'WHILE_STMT', 'DO_STMT', 'IF_STMT',
                    'SWITCH_STMT', 'FUNCTION_DECL'):
            # Header only; the body is reached through its own marking.
            body = self._body_of(cursor)
            if body is not None:
                body_start, body_end = _line_span(body)
                self.marks.keep_range(start, body_start, 'header')
                self.marks.keep(body_end, 'block-close')
                return

        self.marks.keep_range(start, end, kind.lower())

    def _body_of(self, cursor):
        for child in self._children.get(self._key(cursor), ()):
            if _kind(child) == 'COMPOUND_STMT':
                return child
        return None

    # -- the four dependent rules ------------------------------------------

    def _dependents(self, cursor):
        """The paper's dependent rules, plus the variable backward traversal."""
        kind = _kind(cursor)
        children = self._children.get(self._key(cursor), [])
        out = []

        if kind == 'CALL_EXPR':
            # "its arguments and the left-hand side of its assignment"
            out.extend(cursor.get_arguments())
            out.extend(self._assignment_target_of(cursor))
            self._note_callee(cursor)

        elif kind in ('IF_STMT', 'WHILE_STMT', 'SWITCH_STMT', 'DO_STMT'):
            # "the boolean statement"
            if children:
                out.append(children[0])

        elif kind == 'FOR_STMT':
            # "its initialization, update, and condition statements" -- every
            # child except the body.
            body = self._body_of(cursor)
            out.extend(child for child in children if child is not body)

        elif kind in ('BINARY_OPERATOR', 'COMPOUND_ASSIGNMENT_OPERATOR'):
            # "its left-hand side"
            if children and (kind != 'BINARY_OPERATOR'
                             or self._is_assignment(cursor)):
                out.append(children[0])
            elif kind == 'BINARY_OPERATOR':
                # Not an assignment: it is an expression the caller needs
                # whole, so both sides are dependents.
                out.extend(children)

        elif kind == 'VAR_DECL':
            out.extend(children)

        elif kind in NO_AUTO_CHILDREN:
            # Marked for context only.  Its statements reach the kept set on
            # their own if they matter.
            pass

        else:
            # Any other expression: descend, because the variable references
            # the rules above are written in terms of are nested inside casts,
            # parentheses, subscripts and arithmetic.
            out.extend(children)

        # "Whenever a variable is marked, a backward traversal must be applied
        # to mark all assignments associated with that variable."
        out.extend(self._writes_to_referenced(cursor))
        return out

    def _note_callee(self, cursor):
        """Remember a callee defined in this file so its body is kept."""
        callee = cursor.referenced
        if callee is None or not _in_file(callee, self.path):
            return
        if _kind(callee) != 'FUNCTION_DECL':
            return
        # A definition has a body; a bare prototype does not and needs nothing.
        if self._body_of(callee) is None:
            return
        # The function holding the I/O is reduced, not kept whole.  Only
        # helpers reached from it are.
        if callee.spelling == 'main':
            return
        self._full_keep.add(callee)

    def _assignment_target_of(self, cursor):
        """The left-hand side, when this cursor is the RHS of an assignment."""
        parent = self._parent.get(self._key(cursor))
        while parent is not None and _kind(parent) in (
                'UNEXPOSED_EXPR', 'PAREN_EXPR', 'CSTYLE_CAST_EXPR',
                'IMPLICIT_CAST_EXPR'):
            parent = self._parent.get(self._key(parent))
        if parent is None:
            return []
        kind = _kind(parent)
        if kind == 'VAR_DECL':
            return [parent]
        if kind in ('BINARY_OPERATOR', 'COMPOUND_ASSIGNMENT_OPERATOR'):
            children = list(parent.get_children())
            if children:
                return [parent, children[0]]
        return []

    def _writes_to_referenced(self, cursor):
        """Every assignment to the variable this cursor refers to."""
        if _kind(cursor) != 'DECL_REF_EXPR':
            return []
        declaration = cursor.referenced
        if declaration is None:
            return []
        usr = declaration.get_usr() or declaration.spelling
        out = list(self._assignments.get(usr, ()))
        if _in_file(declaration, self.path):
            out.append(declaration)
        return out

    # -- parents ------------------------------------------------------------

    def _contextual_parent(self, cursor):
        """The enclosing syntactic construct, or None at top level.

        "the contextual parent of a for loop body statement is the loop
        header".  Compound statements are transparent here -- the interesting
        parent is the loop or conditional that owns them -- but they are still
        marked so their braces survive.
        """
        parent = self._parent.get(self._key(cursor))
        while parent is not None and _kind(parent) in ('UNEXPOSED_EXPR',
                                                       'PAREN_EXPR'):
            parent = self._parent.get(self._key(parent))
        if parent is None:
            return None
        if _kind(parent) == 'TRANSLATION_UNIT':
            return None
        return parent
