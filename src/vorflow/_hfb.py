"""Barrier connections of a built DISU grid and their HFB package data.

A barrier is drawn as a trace: one line for every layer, or one per layer
(``None`` where a layer has no barrier). Each layer's cells must have faces on
its trace, as ``add_line(..., is_barrier=True)`` meshes them. A horizontal
connection is cut when the face between its two cells lies on the trace, so
that faces meeting at a junction or a corner are cut as well. Wherever the
trace crosses a layer's cells it must run on cut faces.

This module returns FloPy constructor arguments without importing FloPy. Node
numbers are zero-based.
"""

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

_LINEAR = {shapely.GeometryType.LINESTRING, shapely.GeometryType.LINEARRING,
           shapely.GeometryType.MULTILINESTRING}


def _traces(trace, nlay):
    """One validated linear geometry (or None) per layer."""
    if trace is None or isinstance(trace, BaseGeometry):
        traces = [trace] * nlay
    else:
        try:
            traces = list(trace)
        except TypeError as error:
            raise ValueError("trace must be a line geometry or one per layer.") from error
        if len(traces) != nlay:
            raise ValueError(f"trace needs one line geometry (or None) per layer: {nlay}, got {len(traces)}.")
    for t in traces:
        if t is not None and (not isinstance(t, BaseGeometry) or shapely.get_type_id(t) not in _LINEAR
                              or t.is_empty):
            raise ValueError("Each trace must be a non-empty LineString, LinearRing or MultiLineString, or None.")
    return traces


def _same(a, b):
    return a is b or (a is not None and b is not None and a.equals(b))


def _cells(grid):
    """Cell table, plan centre, layer and polygon of every active node, in node order."""
    cells = grid.cellmap[grid.cellmap.node >= 0].sort_values("node")
    polygons = np.empty(len(cells), dtype=object)
    which, icpl = cells.grid.to_numpy(dtype=np.int64), cells.icpl.to_numpy(dtype=np.int64)
    for g in np.unique(which):
        polygons[which == g] = grid._grids[g].geometry.to_numpy()[icpl[which == g]]
    return cells, cells[["x", "y"]].to_numpy(dtype=float), cells.layer.to_numpy(), polygons


def _faces_on(trace, links, n, m, xy, polygons, tolerance):
    """The ``links`` whose shared face lies on ``trace``, and those faces."""
    tree = shapely.STRtree(shapely.linestrings(np.stack((xy[n[links]], xy[m[links]]), axis=1)))
    found = links[tree.query(trace, predicate="dwithin", distance=tolerance)]
    faces = shapely.intersection(shapely.boundary(polygons[n[found]]), shapely.boundary(polygons[m[found]]))
    on = (shapely.length(faces) > tolerance) & shapely.covers(trace.buffer(tolerance), faces)
    return found[on], faces[on]


def _require_faces(trace, k, faces, cells, tolerance):
    """Raise unless ``trace`` runs on ``faces`` wherever it crosses ``cells`` (one layer's cells near it)."""
    if not len(cells):
        return
    area = shapely.union_all(cells)
    open_ = shapely.difference(shapely.intersection(trace, area), shapely.boundary(area).buffer(tolerance))
    if len(faces):
        open_ = shapely.difference(open_, shapely.union_all(faces).buffer(tolerance))
    if shapely.length(open_) > 10 * tolerance:
        x, y = shapely.get_coordinates(open_)[0]
        raise ValueError(
            f"The barrier trace of layer {k} runs {shapely.length(open_):.3g} off the cell faces, starting near "
            f"({x:.6g}, {y:.6g}). Mesh each layer with its trace, add_line(..., is_barrier=True), and pass the "
            "lines the faces follow (MeshGenerator.barrier_lines).")


def barrier_faces(grid, trace):
    """Find the connections a barrier cuts; see ``DisuGrid.barrier_faces``."""
    grid._require_build()
    traces = _traces(trace, grid.nlay)
    links = grid.links
    n, m = np.asarray(links["n"], dtype=np.int64), np.asarray(links["m"], dtype=np.int64)
    ihc = np.asarray(links["ihc"])
    _, xy, layer, polygons = _cells(grid)
    tolerance = 1e-7 * max(float(np.ptp(xy, axis=0).max()), 1.0)
    cell_tree = shapely.STRtree(polygons)

    ta = np.array([traces[k] for k in layer[n]], dtype=object)
    tb = np.array([traces[k] for k in layer[m]], dtype=object)
    same = np.array([_same(a, b) or a is None or b is None for a, b in zip(ta, tb)])
    if not same.all():
        raise ValueError("Traces that differ between connected layers (a dipping fault) are not supported.")

    hit = np.zeros(len(n), dtype=bool)
    face = np.empty(len(n), dtype=object)
    horizontal = np.flatnonzero(ihc != 0)
    for t in {id(t): t for t in traces if t is not None}.values():
        own = [k for k, other in enumerate(traces) if _same(other, t)]
        candidates = horizontal[np.isin(layer[n[horizontal]], own) | np.isin(layer[m[horizontal]], own)]
        found, faces = _faces_on(t, candidates, n, m, xy, polygons, tolerance)
        hit[found], face[found] = True, faces
    for k, t in enumerate(traces):
        if t is not None:
            near = cell_tree.query(t, predicate="intersects")
            mine = hit & ((layer[n] == k) | (layer[m] == k))
            _require_faces(t, k, face[mine], polygons[near[layer[near] == k]], tolerance)
    walls = np.flatnonzero(hit)
    return BarrierFaces(walls, n[walls], m[walls])


@dataclass
class BarrierFaces:
    """Connections cut by a barrier, with MF6 and MODFLOW-USG HFB package data.

    ``links`` indexes ``DisuGrid.links``; ``n`` and ``m`` are the zero-based
    nodes on either side.
    """

    links: np.ndarray
    n: np.ndarray
    m: np.ndarray

    def __len__(self):
        return len(self.links)

    @property
    def pairs(self):
        """``(n, m)`` node pairs, one row per cut connection."""
        return np.column_stack((self.n, self.m))

    def _hydchr(self, hydchr):
        try:
            values = np.broadcast_to(np.asarray(hydchr, dtype=float), (len(self),))
        except (TypeError, ValueError) as error:
            raise ValueError("hydchr must be a number or have one value per barrier connection.") from error
        if not np.isfinite(values).all():
            raise ValueError("hydchr must be finite.")
        return values

    def get_gridprops_mf6_hfb(self, hydchr):
        """Return ``ModflowGwfhfb`` kwargs for one stress period, or None when empty.

        ``hydchr`` is the barrier conductivity divided by its width (>= 0; a
        negative value is MODFLOW's conductance multiplier), one number or one
        per connection.
        """
        values = self._hydchr(hydchr)
        if not len(self):
            return None
        return {
            "maxhfb": len(self),
            "stress_period_data": {0: [((int(a),), (int(b),), float(h)) for a, b, h in zip(self.n, self.m, values)]},
        }

    def get_gridprops_mfusg_hfb(self, hydchr):
        """Return ``flopy.modflow.ModflowHfb`` kwargs for a MODFLOW-USG model, or None when empty."""
        values = self._hydchr(hydchr)
        if not len(self):
            return None
        return {
            "nhfbnp": len(self),
            "hfb_data": [[int(a), int(b), float(h)] for a, b, h in zip(self.n, self.m, values)],
        }
