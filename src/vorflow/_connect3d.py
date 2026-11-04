"""DISU connections between the cells of a layered stack.

Pure array functions. A link joins model nodes ``n`` and ``m`` once; the
exporters write both directions. Link arrays:

* ``n``, ``m``: zero-based model nodes.
* ``ihc``: 0 vertical, 1 horizontal, 2 vertically staggered horizontal.
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


def _ragged(starts, counts):
    """Owner and flat index of ``counts[k]`` consecutive items from ``starts[k]``, for every ``k``."""
    counts = np.asarray(counts, dtype=np.int64)
    owner = np.repeat(np.arange(len(counts)), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    return owner, np.asarray(starts, dtype=np.int64)[owner] + np.arange(counts.sum()) - first


def _containing(point_face, z, cell_face, cell_top, cell_bot):
    """For each point ``(point_face, z)``, the index of the cell of that face containing ``z``, or -1.

    The cells of one face must not overlap. Points are merged into the cells
    sorted from the top down; the last cell top at or above a point is the
    only cell that can contain it.
    """
    n_cells = len(cell_face)
    kind = np.r_[np.zeros(n_cells, dtype=np.int8), np.ones(len(point_face), dtype=np.int8)]
    face = np.r_[cell_face, point_face]
    order = np.lexsort((kind, -np.r_[cell_top, z], face))
    kind, face = kind[order], face[order]
    last_top = np.maximum.accumulate(np.where(kind == 0, np.arange(len(order)), -1))
    is_point = kind == 1
    point = order[is_point] - n_cells
    candidate = last_top[is_point]
    cell = order[np.maximum(candidate, 0)]
    valid = (candidate >= 0) & (face[np.maximum(candidate, 0)] == face[is_point])
    valid &= cell_bot[np.where(valid, cell, 0)] < z[point]
    found = np.full(len(point_face), -1, dtype=np.int64)
    found[point[valid]] = cell[valid]
    return found


def staggered_links(status, nodemap, top_l, bot_l, faces, centres, sliver_abs_tol=0.0, cl12="plan"):
    """Full connectivity: IHC=2 links between active cells whose shared faces overlap vertically.

    Across each plan face the interface elevations of the two columns are
    merged. Every piece between consecutive elevations that lies inside an
    active cell on both sides is one link, with that piece's height. This is
    the connectivity of Langevin's ``disv2disu.py`` (Provost et al. 2025),
    found in one sorted pass for all faces instead of layer by layer.

    With the defaults every piece is kept, which keeps each cell's face areas
    equal to its geometry. Optionally, pieces thinner than ``sliver_abs_tol``
    are merged: each run goes half to the piece above and half to the piece
    below, which keeps the face area but moves it between cell pairs, so the
    result is no longer exact.
    ``hwva`` is scaled because MODFLOW 6 recomputes the overlap from the cell
    elevations.

    Returns ``(links, diagnostics)``.
    """
    ncpl = status.shape[1]
    active = status == ACTIVE
    # Active cells column by column, each column from the top down.
    cc_cell, cc_layer = np.nonzero(active.T)
    cc_top, cc_bot = top_l[cc_layer, cc_cell], bot_l[cc_layer, cc_cell]
    counts = np.bincount(cc_cell, minlength=ncpl)
    starts = np.cumsum(counts) - counts

    ci = faces["cell_id_1"].to_numpy(dtype=np.int64)
    cj = faces["cell_id_2"].to_numpy(dtype=np.int64)
    face_i, cells_i = _ragged(starts[ci], counts[ci])
    face_j, cells_j = _ragged(starts[cj], counts[cj])

    # Merged interface elevations of each face, from the top down, without repeats.
    bp_face = np.concatenate([face_i, face_i, face_j, face_j])
    bp_z = np.concatenate([cc_top[cells_i], cc_bot[cells_i], cc_top[cells_j], cc_bot[cells_j]])
    order = np.lexsort((-bp_z, bp_face))
    bp_face, bp_z = bp_face[order], bp_z[order]
    new = np.ones(len(bp_z), dtype=bool)
    new[1:] = (bp_face[1:] != bp_face[:-1]) | (bp_z[1:] != bp_z[:-1])
    bp_face, bp_z = bp_face[new], bp_z[new]
    same_face = bp_face[1:] == bp_face[:-1]
    piece_face = bp_face[:-1][same_face]
    z_hi, z_lo = bp_z[:-1][same_face], bp_z[1:][same_face]
    middle = z_lo + (z_hi - z_lo) / 2
    a = _containing(piece_face, middle, face_i, cc_top[cells_i], cc_bot[cells_i])
    b = _containing(piece_face, middle, face_j, cc_top[cells_j], cc_bot[cells_j])
    both = (a >= 0) & (b >= 0)
    piece_face, z_hi, z_lo = piece_face[both], z_hi[both], z_lo[both]
    cell_a, cell_b = cells_i[a[both]], cells_j[b[both]]
    # Elevations a roundoff apart split one overlap into two pieces with the
    # same cells: join consecutive pieces that link the same pair.
    repeat = np.zeros(len(piece_face), dtype=bool)
    repeat[1:] = ((piece_face[1:] == piece_face[:-1]) & (cell_a[1:] == cell_a[:-1])
                  & (cell_b[1:] == cell_b[:-1]) & (z_hi[1:] == z_lo[:-1]))
    if repeat.any():
        first = np.flatnonzero(~repeat)
        last = np.r_[first[1:] - 1, len(repeat) - 1]
        piece_face, cell_a, cell_b = piece_face[first], cell_a[first], cell_b[first]
        z_hi, z_lo = z_hi[first], z_lo[last]
    height = z_hi - z_lo

    d1 = faces["distance_1"].to_numpy(dtype=float)[piece_face]
    d2 = faces["distance_2"].to_numpy(dtype=float)[piece_face]
    merged, sliver = _merge_slivers(piece_face, z_hi, z_lo, height, sliver_abs_tol)
    keep = ~sliver
    piece_face, cell_a, cell_b = piece_face[keep], cell_a[keep], cell_b[keep]
    geometric, merged, d1, d2 = height[keep], merged[keep], d1[keep], d2[keep]

    length = faces["shared_length"].to_numpy(dtype=float)[piece_face]
    if cl12 == "connector":
        plan = np.linalg.norm(centres[cc_cell[cell_a]] - centres[cc_cell[cell_b]], axis=1)
        dz = (cc_top[cell_a] + cc_bot[cell_a] - cc_top[cell_b] - cc_bot[cell_b]) / 2
        span = np.hypot(plan, dz)
        cl_n, cl_m = span * d1 / (d1 + d2), span * d2 / (d1 + d2)
    else:
        cl_n, cl_m = d1, d2
    links = {
        "n": nodemap[cc_layer[cell_a], cc_cell[cell_a]],
        "m": nodemap[cc_layer[cell_b], cc_cell[cell_b]],
        "ihc": np.full(len(cell_a), 2, dtype=np.int64),
        "cl_n": cl_n, "cl_m": cl_m,
        "hwva": length * merged / geometric,
        "height": merged,
        "angle": face_angles(faces).to_numpy(dtype=float)[piece_face],
    }
    diagnostics = {
        "slivers_merged": int(sliver.sum()),
        "largest_sliver": float(height[sliver].max()) if sliver.any() else 0.0,
    }
    return links, diagnostics


def _merge_slivers(face, z_hi, z_lo, height, tolerance):
    """Give each run of sliver pieces half to the piece above and half to the piece below.

    Pieces are ordered by face and from the top down. A run of slivers takes
    its neighbours among the pieces it touches on the same face, and goes
    wholly to one side when only one neighbour exists. A run that touches no
    other piece is all the contact there is, and is kept.

    Returns ``(merged_height, is_sliver)``.
    """
    count = len(face)
    merged = height.astype(float).copy()
    small = height < tolerance
    if not small.any():
        return merged, small
    # touches_above[k]: piece k starts where piece k-1 ends, on the same face.
    touches_above = np.zeros(count, dtype=bool)
    touches_above[1:] = (face[1:] == face[:-1]) & (z_lo[:-1] == z_hi[1:])
    touches_below = np.r_[touches_above[1:], False]
    first = np.flatnonzero(small & ~(np.r_[False, small[:-1]] & touches_above))
    last = np.flatnonzero(small & ~(np.r_[small[1:], False] & touches_below))
    run = np.cumsum(np.isin(np.arange(count), first)) * small
    total = np.bincount(run, weights=height, minlength=len(first) + 1)[1:]
    has_above = touches_above[first]
    has_below = touches_below[last]
    both = has_above & has_below
    share_above = np.where(both, total / 2, total) * has_above
    share_below = np.where(both, total / 2, total) * has_below
    np.add.at(merged, first[has_above] - 1, share_above[has_above])
    np.add.at(merged, last[has_below] + 1, share_below[has_below])
    lonely = ~has_above & ~has_below
    is_sliver = small & ~np.isin(run, np.flatnonzero(lonely) + 1)
    return merged, is_sliver
