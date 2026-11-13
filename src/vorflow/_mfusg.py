"""MODFLOW-USG DISU serialization; never use these arrays for MF6 DISU.

Geometry and layer preparation are shared by DISU and original 3D GSF output.
FloPy is deliberately absent: only the tessellator's package wrapper needs it.
"""

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shapely


logger = logging.getLogger(__name__)

# A cell centre may lie this fraction of sqrt(cell area) outside its polygon.
# Barrier sites and boundary generators sit on a domain or barrier edge, and
# clipping or cutting that edge leaves them up to ~1e-8 x the cell size
# outside (absolute 1e-14..1e-7 m), well past coordinate roundoff. A centre
# that is really outside its cell is off by a sizeable fraction of it.
CENTRE_OUTSIDE_FRACTION = 1e-6
# A face recovered from roundoff (boundaries within the tolerance of each other
# but meeting only at points) counts when longer than this fraction of
# sqrt(cell area); at a corner contact the recovered piece is about the
# tolerance long.
RECOVERED_FACE_FRACTION = 1e-6
METADATA_KEYS = {"node_id", "cell_map", "connectivity"}
PACKAGE_KEYS = {
    "nodes", "nlay", "njag", "ivsd", "idsymrd", "nodelay",
    "top", "bot", "area", "iac", "ja", "cl12", "fahl",
}


@dataclass
class _LayeredGrid:
    planar: dict
    top: np.ndarray
    bot: np.ndarray
    area: np.ndarray
    cell_map: pd.DataFrame
    faces: pd.DataFrame

    @property
    def nlay(self):
        return self.bot.shape[0]

    @property
    def ncpl(self):
        return self.planar["ncpl"]


def _layer_arrays(top, botm, nlay, ncpl):
    if isinstance(nlay, (bool, np.bool_)) or not isinstance(nlay, (int, np.integer)) or nlay < 1:
        raise ValueError("nlay must be a positive integer.")
    top = np.asarray(top, dtype=float)
    if top.ndim == 0:
        top = np.full(ncpl, top.item())
    elif top.shape != (ncpl,):
        raise ValueError(f"top must be scalar or have shape ({ncpl},).")
    bot = np.asarray(botm, dtype=float)
    if bot.ndim == 0:
        bot = np.full((nlay, ncpl), bot.item())
    elif bot.shape == (nlay,):
        bot = np.broadcast_to(bot[:, None], (nlay, ncpl)).copy()
    elif bot.shape != (nlay, ncpl):
        raise ValueError(f"botm must be scalar, shape ({nlay},), or shape ({nlay}, {ncpl}).")
    layer_top = np.vstack([top[None, :], bot[:-1]])
    if not np.isfinite(layer_top).all() or not np.isfinite(bot).all():
        raise ValueError("Layer elevations must be finite.")
    with np.errstate(over="ignore"):
        thickness = layer_top - bot
    if not np.isfinite(thickness).all():
        raise ValueError("Layer thicknesses must be finite.")
    bad = np.argwhere(layer_top <= bot)
    if len(bad):
        layer, cell = bad[0]
        raise ValueError(
            f"Non-positive thickness at layer {layer}, cell {cell}; "
            "pinch-outs and vertical pass-through are not supported."
        )
    return layer_top.copy(), bot.copy()


def _line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry] if geometry.length > 0 else []
    if hasattr(geometry, "geoms"):
        return [line for part in geometry.geoms for line in _line_parts(part)]
    return []


def _straight_face(shared, centres, pair, tolerance):
    """Combine contiguous collinear pieces, rejecting bent/disconnected faces."""
    parts = _line_parts(shared)
    if not parts:
        return None  # Corner-only contact.
    longest = max(parts, key=lambda part: part.length)
    xy = np.asarray(longest.coords)[:, :2]
    origin = xy[0]
    direction = xy[-1] - origin
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError(f"Unsupported shared face between cells {pair}.")
    direction /= norm
    normal = np.array([-direction[1], direction[0]])
    intervals = []
    for part in parts:
        offsets = np.asarray(part.coords)[:, :2] - origin
        if np.max(np.abs(offsets @ normal)) > tolerance:
            raise ValueError(
                f"Noncollinear shared face between cells {pair}: the cells meet along a "
                "bent line, which breaks the straight, perpendicular face that CVFD "
                "connections assume (MF6 DISV would silently use only its first edge). "
                "In vorflow grids this comes from cells cut along a barrier; check the "
                "barrier warnings from generate()."
            )
        projected = offsets @ direction
        intervals.append((projected.min(), projected.max()))
    intervals.sort()
    end = intervals[0][1]
    for start, stop in intervals[1:]:
        if start > end + tolerance:
            raise ValueError(f"Disconnected shared face between cells {pair}.")
        end = max(end, stop)
    signed = (centres - origin) @ normal
    if signed[0] * signed[1] >= 0 or not np.isfinite(signed).all():
        raise ValueError(f"Centres must lie on opposite sides of the shared face for cells {pair}.")
    delta = centres[1] - centres[0]
    angle = np.degrees(np.arccos(np.clip(abs(delta @ normal) / np.linalg.norm(delta), 0, 1)))
    return {
        "cell_id_1": pair[0], "cell_id_2": pair[1],
        "shared_length": sum(part.length for part in parts),
        "distance_1": abs(signed[0]), "distance_2": abs(signed[1]),
        "ortho_error": float(angle),
    }


def _near_length_bound(geoms, first, second, distance):
    """Per pair, an upper bound of the length of ``first``'s boundary within ``distance`` of ``second``'s.

    The points within ``distance`` of a segment t lie in a rectangle: along t,
    from ``distance`` before its start to ``distance`` past its end; across
    it, within ``distance`` of it. The bound adds up the length of each
    segment of the first boundary inside the rectangle of each segment of the
    second.
    """
    xy, owner = shapely.get_coordinates(geoms, return_index=True)
    start, end = xy[:-1], xy[1:]
    keep = (owner[:-1] == owner[1:]) & np.any(start != end, axis=1)
    start, end, owner = start[keep], end[keep], owner[:-1][keep]
    count = np.bincount(owner, minlength=len(geoms))
    offset = np.cumsum(count) - count
    length = np.hypot(*(end - start).T)
    (x, y), (ux, uy) = start.T, ((end - start) / length[:, None]).T
    per_pair = count[first] * count[second]
    bound = np.zeros(len(first))
    # Blocks of about a million segment pairs keep memory bounded on large grids.
    for block in np.array_split(np.arange(len(first)), 1 + int(per_pair.sum()) // 2**20):
        pair = np.repeat(block, per_pair[block])
        k = np.arange(len(pair)) - np.repeat(np.cumsum(per_pair[block]) - per_pair[block], per_pair[block])
        s = offset[first[pair]] + k // count[second[pair]]
        t = offset[second[pair]] + k % count[second[pair]]
        dx, dy, vx, vy = x[s] - x[t], y[s] - y[t], ux[t], uy[t]
        low, high = np.zeros(len(pair)), length[s]
        # The positions along s whose coordinate along t, then across it, is within [lower, upper].
        along = (dx * vx + dy * vy, ux[s] * vx + uy[s] * vy, -distance, length[t] + distance)
        across = (dy * vx - dx * vy, uy[s] * vx - ux[s] * vy, -distance, distance)
        for at, rate, lower, upper in (along, across):
            with np.errstate(divide="ignore", invalid="ignore"):
                a, b = (lower - at) / rate, (upper - at) / rate
            parallel = rate == 0
            a = np.where(parallel, np.where((lower <= at) & (at <= upper), -np.inf, np.inf), a)
            b = np.where(parallel, np.inf, b)
            low, high = np.maximum(low, np.minimum(a, b)), np.minimum(high, np.maximum(a, b))
        bound += np.bincount(pair, np.maximum(high - low, 0), minlength=len(first))
    return bound


def _prepare_grid(gdf, top, botm, nlay):
    # Local import avoids a cycle with the thin public wrappers in utils.
    from .utils import grid_to_disv

    required = {"geometry", "x", "y"}
    if not required.issubset(gdf.columns):
        raise ValueError(f"Grid missing required columns: {sorted(required.difference(gdf.columns))}.")
    if gdf.empty:
        raise ValueError("MFUSG export needs a non-empty grid.")
    geoms = gdf.geometry.to_numpy()
    for i, geom in enumerate(geoms):
        if geom is None or geom.geom_type != "Polygon" or geom.is_empty or not geom.is_valid:
            raise ValueError(f"Cell {i} must be a valid non-empty single polygon.")
        if geom.interiors:
            raise ValueError(f"Cell {i} has holes, which GSF cannot represent.")
    centres = gdf[["x", "y"]].to_numpy(dtype=float)
    if not np.isfinite(centres).all():
        raise ValueError("Cell centres must be finite.")
    coords = shapely.get_coordinates(geoms)
    tolerance = 32 * np.finfo(float).eps * max(1.0, float(np.abs(coords).max()))
    # Accept edge generators just outside their cell without moving them.
    centre_tolerance = np.maximum(tolerance, CENTRE_OUTSIDE_FRACTION * np.sqrt(shapely.area(geoms)))
    outside = ~shapely.dwithin(geoms, shapely.points(centres), centre_tolerance)
    if outside.any():
        raise ValueError(f"Centre of cell {int(np.flatnonzero(outside)[0])} lies outside its polygon.")
    layer_top, bot = _layer_arrays(top, botm, nlay, len(gdf))
    # Reuse conforming rings, not MF6 solver semantics. Limit insertion to
    # coordinate roundoff: a looser map-scale tolerance can bend very short
    # faces, making GSF footprints disagree with the numerical geometry.
    planar = grid_to_disv(gdf, tolerance=tolerance)
    tree = shapely.STRtree(geoms)
    # Within the tolerance, not just intersecting: roundoff can leave a hair's gap
    # between neighbours, e.g. along hanging vertices.
    left, right = tree.query(geoms, predicate="dwithin", distance=tolerance)
    boundaries = shapely.boundary(geoms)
    areas = shapely.area(geoms)
    pairs = sorted(zip(left[left < right], right[left < right]))
    # Upper bounds of the part of one boundary within the tolerance of the other
    # (below), from the segments alone; twice the tolerance covers the band's
    # own roundoff.
    bounds = _near_length_bound(geoms, *np.array(pairs, dtype=int).reshape(-1, 2).T, 2 * tolerance)
    records = []
    for (i, j), bound in zip(pairs, bounds):
        overlap = geoms[i].intersection(geoms[j])
        # Faces recovered from roundoff are checked with twice the tolerance:
        # pieces of one straight face can lie up to it on both sides of it.
        recovered = overlap.area > 0
        if recovered:
            # Only forgive slivers contained within BOTH boundary roundoff
            # bands. An area-only cutoff could accept small but deep overlaps.
            for geom in (geoms[i], geoms[j]):
                if not overlap.difference(geom.boundary.buffer(tolerance)).is_empty:
                    raise ValueError(f"Overlapping cells {i} and {j}.")
            # Recover the common face as well as accepting the sliver; otherwise
            # exact boundary intersection could silently lose this connection.
            shared = shapely.intersection(
                geoms[i].boundary, geoms[j].boundary, grid_size=tolerance,
            )
            if shared.is_empty:
                raise ValueError(f"Cannot resolve roundoff overlap between cells {i} and {j}.")
        else:
            shared = geoms[i].boundary.intersection(geoms[j].boundary)
        # Roundoff can keep two boundaries from meeting along all of their face,
        # even snapped: along hanging vertices a hair off the neighbour's straight
        # face (as on a splice outline) they meet over part of it, only at points,
        # or not at all. The part of one boundary within the tolerance of the other
        # is the whole face; at a corner contact it is only about the tolerance long.
        # Its band is costly and most pairs' intersection is already the whole
        # face: look for it only where its bound exceeds the face found by more
        # than half the margin (the other half absorbs roundoff in the lengths).
        found = sum(p.length for p in _line_parts(shared))
        margin = RECOVERED_FACE_FRACTION * np.sqrt(min(areas[i], areas[j]))
        if bound - found > margin / 2:
            near = shapely.intersection(boundaries[i], shapely.buffer(boundaries[j], tolerance))
            if sum(p.length for p in _line_parts(near)) - found > margin:
                shared, recovered = near, True
        record = _straight_face(shared, centres[[i, j]], (int(i), int(j)),
                                2 * tolerance if recovered else tolerance)
        if record is not None:
            records.append(record)
    faces = pd.DataFrame(records, columns=[
        "cell_id_1", "cell_id_2", "shared_length", "distance_1", "distance_2", "ortho_error",
    ])
    if len(faces):
        logger.info("MFUSG export: %s horizontal faces; maximum orthogonality error %.6g degrees",
                    len(faces), faces["ortho_error"].max())
    ncpl = len(gdf)
    mapping = pd.DataFrame({
        "model_node": np.arange(nlay * ncpl),
        "layer": np.repeat(np.arange(nlay), ncpl),
        "cell_id": np.tile(np.arange(ncpl), nlay),
        "orig_index": np.tile(gdf.index.to_numpy(), nlay),
        "node_id": np.tile(planar["node_id"], nlay),
    })
    if "zone_id" in gdf:
        mapping["zone_id"] = np.tile(gdf["zone_id"].to_numpy(), nlay)
    return _LayeredGrid(planar, layer_top, bot, shapely.area(geoms), mapping, faces)


def _disu_arrays(grid):
    """Full-matrix USG CL12 stores each node's own half-connection length."""
    ncpl, nlay = grid.ncpl, grid.nlay
    nodes = ncpl * nlay
    rows = [[] for _ in range(nodes)]
    thickness = grid.top - grid.bot
    for face in grid.faces.itertuples(index=False):
        i, j = int(face.cell_id_1), int(face.cell_id_2)
        for layer in range(nlay):
            n, m = layer * ncpl + i, layer * ncpl + j
            area = face.shared_length * 0.5 * (thickness[layer, i] + thickness[layer, j])
            rows[n].append((m, face.distance_1, area))
            rows[m].append((n, face.distance_2, area))
    for layer in range(nlay - 1):
        for cell in range(ncpl):
            n, m = layer * ncpl + cell, (layer + 1) * ncpl + cell
            rows[n].append((m, thickness[layer, cell] / 2, grid.area[cell]))
            rows[m].append((n, thickness[layer + 1, cell] / 2, grid.area[cell]))
    iac = np.array([len(row) + 1 for row in rows], dtype=np.int32)
    ja, cl12, fahl = [], [], []
    for node, row in enumerate(rows):
        ja.append(node)
        cl12.append(0.0)
        fahl.append(0.0)
        for neighbour, distance, area in sorted(row):
            ja.append(neighbour)
            cl12.append(distance)
            fahl.append(area)
    return {
        "nodes": nodes, "nlay": nlay, "njag": int(iac.sum()),
        "ivsd": -1, "idsymrd": 0,
        "nodelay": np.full(nlay, ncpl, dtype=np.int32),
        "top": grid.top.ravel().copy(), "bot": grid.bot.ravel().copy(),
        "area": grid.area.copy(), "iac": iac,
        "ja": np.array(ja, dtype=np.int32),
        "cl12": np.array(cl12, dtype=float), "fahl": np.array(fahl, dtype=float),
        "node_id": grid.cell_map["node_id"].to_numpy(copy=True),
        "cell_map": grid.cell_map.copy(deep=True),
        "connectivity": grid.faces.copy(deep=True),
    }


def _write_gsf(grid, filepath):
    """Original 3D GSF: top and bottom vertex rings, all IDs one-based."""
    xy = np.asarray(grid.planar["vertices"])[:, 1:]
    vertices, vertex_ids, cells = [], {}, []
    for node in range(grid.ncpl * grid.nlay):
        layer, cell = divmod(node, grid.ncpl)
        ring = grid.planar["cell2d"][cell]
        ids = []
        for elevation in (grid.top[layer, cell], grid.bot[layer, cell]):
            for vertex in ring[4:]:
                xyz = (*xy[vertex], elevation)
                if xyz not in vertex_ids:
                    vertex_ids[xyz] = len(vertices) + 1
                    vertices.append(xyz)
                ids.append(vertex_ids[xyz])
        # Midpoint avoids top+bottom overflow for otherwise finite elevations.
        z = grid.bot[layer, cell] + (grid.top[layer, cell] - grid.bot[layer, cell]) / 2
        cells.append((node + 1, ring[1], ring[2], z, layer + 1, ids))
    path = Path(filepath)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"UNSTRUCTURED\n{len(cells)} {grid.nlay} 1 1\n{len(vertices)}\n")
        for xyz in vertices:
            stream.write(" ".join(format(float(value), ".17g") for value in xyz) + "\n")
        for node, x, y, z, layer, ids in cells:
            stream.write(
                f"{node} {x:.17g} {y:.17g} {z:.17g} {layer} {len(ids)} "
                + " ".join(map(str, ids)) + "\n"
            )
    logger.info("Saved original 3D GSF to %s", path)
    return path
