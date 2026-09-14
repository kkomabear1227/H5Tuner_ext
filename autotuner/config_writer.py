"""Render a candidate configuration as H5Tuner's config.xml.

The shim reads a bare relative "config.xml" from the process working directory
(src/autotuner_hdf5.c:200), so the file has to land in the directory the
application inherits.

Writes are atomic.  The tuner rewrites this file once per candidate -- three
thousand times in a default campaign -- and the dynamic shim does not check the
result of fopen before handing the stream to the XML parser
(src/autotuner_hdf5.c:207).  A rank that reads the file mid-write is a real
race, so we write to a temporary file in the same directory and rename over the
target.
"""

import os
import tempfile
from xml.dom.minidom import Document

from . import space as space_module

CONFIG_FILENAME = 'config.xml'

# config.xml groups elements under per-layer sections.  The shim ignores the
# nesting -- every injector searches the whole document by element name
# (src/autotuner_hdf5.c:83) -- so this is documentation for human readers.
_SECTION = {
    space_module.LAYER_HDF5: 'High_Level_IO_Library',
    space_module.LAYER_MPIIO: 'Middleware_Layer',
    space_module.LAYER_PFS: 'Parallel_File_System',
}

_SECTION_ORDER = (space_module.LAYER_HDF5, space_module.LAYER_MPIIO,
                  space_module.LAYER_PFS)


def _element_text(space, values, element):
    """Text for one config.xml element.

    Several parameters can share an element: HDF5 alignment writes
    "threshold,alignment" from two parameters.  They are joined in group_pos
    order.
    """
    members = [p for p in space.parameters if p.element == element]
    if len(members) == 1 and members[0].group_pos is None:
        return str(values[members[0].name])
    members.sort(key=lambda p: (p.group_pos if p.group_pos is not None else 0))
    return ','.join(str(values[p.name]) for p in members)


def build(space, point):
    """Render a point as a config.xml document string."""
    values = space.decode(point)

    document = Document()
    root = document.createElement('Parameters')
    document.appendChild(root)

    # Group elements by layer, preserving parameter declaration order within a
    # layer and emitting each shared element exactly once.
    by_layer = {}
    for parameter in space.parameters:
        by_layer.setdefault(parameter.layer, [])
        if parameter.element not in by_layer[parameter.layer]:
            by_layer[parameter.layer].append(parameter.element)

    for layer in _SECTION_ORDER:
        elements = by_layer.get(layer)
        if not elements:
            continue
        section = document.createElement(_SECTION[layer])
        root.appendChild(section)
        for element in elements:
            node = document.createElement(element)
            node.appendChild(
                document.createTextNode(_element_text(space, values, element)))
            section.appendChild(node)

    return document.toprettyxml(indent='  ')


def write(space, point, directory, filename=CONFIG_FILENAME):
    """Write config.xml atomically into `directory`; return its path."""
    text = build(space, point)
    target = os.path.join(directory, filename)

    handle, temporary = tempfile.mkstemp(
        prefix='.{0}.'.format(filename), dir=directory, text=True)
    try:
        with os.fdopen(handle, 'w') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    return target
