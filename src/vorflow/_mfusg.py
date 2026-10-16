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
METADATA_KEYS = {"node_id", "cell_map", "connectivity"}


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
            raise ValueError(f"Noncollinear shared face between cells {pair}.")
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
    for i, (geom, centre) in enumerate(zip(geoms, centres)):
        # Clipping a slanted edge can leave a boundary generator a few ulps
        # outside its polygon. Accept roundoff without moving the generator.
        if not shapely.dwithin(geom, shapely.Point(centre), tolerance):
            raise ValueError(f"Centre of cell {i} lies outside its polygon.")
    layer_top, bot = _layer_arrays(top, botm, nlay, len(gdf))
    # Reuse conforming rings, not MF6 solver semantics. Limit insertion to
    # coordinate roundoff: a looser map-scale tolerance can bend very short
    # faces, making GSF footprints disagree with the numerical geometry.
    planar = grid_to_disv(gdf, tolerance=tolerance)
    tree = shapely.STRtree(geoms)
    left, right = tree.query(geoms, predicate="intersects")
    records = []
    for i, j in sorted(zip(left[left < right], right[left < right])):
        overlap = geoms[i].intersection(geoms[j])
        if overlap.area > 0:
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
        record = _straight_face(shared, centres[[i, j]], (int(i), int(j)), tolerance)
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
