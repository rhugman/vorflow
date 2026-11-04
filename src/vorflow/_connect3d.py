"""DISU connections between the cells of a layered stack.

Pure array functions. A link joins model nodes ``n`` and ``m`` once; the
exporters write both directions. Link arrays:

* ``n``, ``m``: zero-based model nodes.
* ``ihc``: 0 vertical, 1 horizontal.
* ``cl_n``, ``cl_m``: distance from each node's centre to the shared face.
* ``hwva``: MODFLOW 6 meaning: face width for horizontal links, area for vertical.
* ``height``: vertical extent of a horizontal face (0 for vertical links).
* ``angle``: ANGLDEGX from ``n`` towards ``m``, degrees in [0, 360).
"""

import numpy as np
import pandas as pd

from ._columns import ACTIVE, vertical_pairs

LINK_KEYS = ("n", "m", "ihc", "cl_n", "cl_m", "hwva", "height", "angle")


def empty_links():
    links = {key: np.zeros(0, dtype=float) for key in LINK_KEYS}
    for key in ("n", "m", "ihc"):
        links[key] = np.zeros(0, dtype=np.int64)
    return links


def concat_links(*parts):
    parts = [part for part in parts if len(part["n"])]
    if not parts:
        return empty_links()
    return {key: np.concatenate([part[key] for part in parts]) for key in LINK_KEYS}


def face_angles(faces):
    """ANGLDEGX of each face normal, from ``cell_id_1`` towards ``cell_id_2``."""
    normal_y = faces["normal_y"].to_numpy(dtype=float)
    normal_x = faces["normal_x"].to_numpy(dtype=float)
    return pd.Series(np.degrees(np.arctan2(normal_y, normal_x)) % 360.0, index=faces.index)


def vertical_links(status, nodemap, top_l, bot_l, area):
    """IHC=0 links between active cells of one column, through pass-through runs."""
    upper, lower, cells = vertical_pairs(status)
    return {
        "n": nodemap[upper, cells], "m": nodemap[lower, cells],
        "ihc": np.zeros(len(cells), dtype=np.int64),
        "cl_n": (top_l[upper, cells] - bot_l[upper, cells]) / 2,
        "cl_m": (top_l[lower, cells] - bot_l[lower, cells]) / 2,
        "hwva": area[cells].astype(float),
        "height": np.zeros(len(cells)),
        "angle": np.zeros(len(cells)),
    }


def layered_links(status, nodemap, faces, top_l, bot_l):
    """IHC=1 links between active neighbours in the same layer (DISV connectivity)."""
    i = faces["cell_id_1"].to_numpy(dtype=np.int64)
    j = faces["cell_id_2"].to_numpy(dtype=np.int64)
    nlay = status.shape[0]
    layer = np.repeat(np.arange(nlay), len(i))
    i, j = np.tile(i, nlay), np.tile(j, nlay)
    keep = (status[layer, i] == ACTIVE) & (status[layer, j] == ACTIVE)
    count = int(keep.sum())
    tile = lambda column: np.tile(faces[column].to_numpy(dtype=float), nlay)[keep]  # noqa: E731
    return {
        "n": nodemap[layer[keep], i[keep]], "m": nodemap[layer[keep], j[keep]],
        "ihc": np.ones(count, dtype=np.int64),
        "cl_n": tile("distance_1"), "cl_m": tile("distance_2"),
        "hwva": tile("shared_length"),
        "height": np.zeros(count),
        "angle": np.tile(face_angles(faces).to_numpy(dtype=float), nlay)[keep],
    }
