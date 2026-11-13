"""Barrier connections of a built DISU grid and their HFB package data.

A barrier is drawn as a trace: one line for every layer, or one per layer
(``None`` where a layer has no barrier). Each layer's cells must have faces on
its trace, as ``add_line(..., is_barrier=True)`` meshes them. A dipping fault
has a different trace in each layer and becomes a staircase: a vertical wall
on each layer's trace and a horizontal tread between the traces of
consecutive layers. A connection is cut when it crosses that staircase:

* horizontal, between layers with the same trace: the face between the two
  cells lies on the trace, so that faces meeting at a junction or a corner
  are cut as well. Wherever the trace crosses a layer's cells it must run on
  cut faces;
* between layers with different traces (treads, and staggered links between
  dipping layers): the centres lie on opposite sides of their own layers'
  traces and the plan segment between them reaches the zone joining the two
  traces, closed at the fault tips.

``factor`` is ``|n_fault · n_face|`` for the plane through consecutive
traces, each at its layer's mid-depth, so that with flow normal to the fault
the staircase has the plane's resistance.
This module returns FloPy constructor arguments without importing FloPy. Node
numbers are zero-based.
"""

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from .disu import _coordinate_tolerance

_LINEAR = {shapely.GeometryType.LINESTRING, shapely.GeometryType.LINEARRING,
           shapely.GeometryType.MULTILINESTRING}


def _is_line(t):
    return t is not None and shapely.get_type_id(t) == shapely.GeometryType.LINESTRING


def _chord(line):
    """End-to-end vector of a line, in plan."""
    c = np.asarray(line.coords, dtype=float)
    return c[-1, :2] - c[0, :2]


def _traces(trace, nlay):
    """One validated linear geometry (or None) per layer.

    LineStrings are given one common direction, each following the one above
    it, so that "left" means the same side of the fault in every layer.
    """
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
    above = None
    for k, t in enumerate(traces):
        if not _is_line(t):
            above = None if t is not None else above
            continue
        if above is not None and t is not above and np.dot(_chord(above), _chord(t)) < 0:
            traces[k] = t = LineString(list(t.coords)[::-1])
        above = t
    return traces


def _same(a, b):
    return a is b or (a is not None and b is not None and a.equals(b))


class _Step:
    """The tread between the different traces of two layers: the polygon joining them."""

    def __init__(self, a, b, la, lb):
        if not (_is_line(a) and _is_line(b)):
            raise ValueError("Traces that differ between layers (a dipping fault) must each be one LineString; "
                             "give one fault per barrier_faces call.")
        # The traces already run the same way (_traces). They must also overlap along the strike:
        # traces placed end to end are not joined by any fault surface.
        start, chord = np.asarray(a.coords[0], dtype=float)[:2], _chord(a)
        length = np.hypot(*chord)
        along = sorted(np.dot(np.asarray(p, dtype=float)[:2] - start, chord / length) for p in (b.coords[0], b.coords[-1]))
        if min(length, along[1]) - max(0.0, along[0]) <= 0:
            raise ValueError(f"The traces of layers {la} and {lb} do not overlap along the strike, "
                             "so no fault surface joins them.")
        self.a, self.b = a, b
        self.polygon = shapely.make_valid(Polygon(list(a.coords) + list(b.coords)[::-1]))
        self.zone = shapely.union_all([self.polygon, a, b])


def _extended(line, length):
    """``line`` lengthened by ``length`` along its first and last segments; other geometries unchanged."""
    if not _is_line(line):
        return line
    c = np.asarray(line.coords, dtype=float)
    start, end = c[0] - c[1], c[-1] - c[-2]
    if not (np.linalg.norm(start) and np.linalg.norm(end)):
        return line
    start, end = start / np.linalg.norm(start), end / np.linalg.norm(end)
    return LineString(np.vstack((c[0] + length * start, c, c[-1] + length * end)))


def _side(line, xy):
    """+1 or -1 for points left or right of an oriented line (0 on it)."""
    points = shapely.points(xy)
    d = shapely.line_locate_point(line, points)
    step = 1e-6 * line.length
    before = shapely.get_coordinates(shapely.line_interpolate_point(line, np.maximum(d - step, 0)))
    after = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(d + step, line.length)))
    near = shapely.get_coordinates(shapely.line_interpolate_point(line, d))
    t, p = after - before, xy - near
    return np.sign(t[:, 0] * p[:, 1] - t[:, 1] * p[:, 0])


def _sides(line, xy, tolerance):
    """+1 or -1 for points left or right of an oriented line; points on it count as left."""
    on = shapely.distance(line, shapely.points(xy)) <= tolerance
    return np.where(on, 1.0, _side(line, xy))


def _cells(grid):
    """Cell table, plan centre, layer and polygon of every active node, in node order."""
    cells = grid.cellmap[grid.cellmap.node >= 0].sort_values("node")
    polygons = np.empty(len(cells), dtype=object)
    which, icpl = cells.grid.to_numpy(dtype=np.int64), cells.icpl.to_numpy(dtype=np.int64)
    for g in np.unique(which):
        polygons[which == g] = grid._grids[g].geometry.to_numpy()[icpl[which == g]]
    return cells, cells[["x", "y"]].to_numpy(dtype=float), cells.layer.to_numpy(), polygons


def _shared_faces(a, b, roundoff):
    """Faces between polygons ``a`` and ``b``: the part of each ``a`` boundary within ``roundoff`` of ``b``'s.

    As in ``_planar_faces``, which found the grid's faces: where hanging
    vertices lie a roundoff off the neighbour's face, the exact intersection
    of the boundaries loses the face, or part of it.
    """
    return shapely.intersection(shapely.boundary(a), shapely.buffer(shapely.boundary(b), roundoff))


def _faces_on(trace, links, n, m, xy, polygons, tolerance, roundoff):
    """The ``links`` whose shared face lies on ``trace``, and those faces."""
    tree = shapely.STRtree(shapely.linestrings(np.stack((xy[n[links]], xy[m[links]]), axis=1)))
    found = links[tree.query(trace, predicate="dwithin", distance=tolerance)]
    faces = _shared_faces(polygons[n[found]], polygons[m[found]], roundoff)
    on = (shapely.length(faces) > tolerance) & shapely.covers(trace.buffer(tolerance), faces)
    return found[on], faces[on]


def _require_faces(trace, k, faces, cells, tolerance, roundoff):
    """Raise unless ``trace`` runs on ``faces`` wherever it crosses ``cells`` (one layer's cells near it)."""
    if not len(cells):
        return
    # Grown by the roundoff, so that a hair's gap between neighbours is not taken for the edge of the cells.
    area = shapely.union_all(shapely.buffer(cells, roundoff))
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
    cells, xy, layer, polygons = _cells(grid)
    thickness = (cells.top - cells.bot).to_numpy(dtype=float)
    middle = ((cells.top + cells.bot) / 2).to_numpy(dtype=float)
    tolerance = 1e-7 * max(float(np.ptp(xy, axis=0).max()), 1.0)
    roundoff = _coordinate_tolerance(grid._grids)
    cell_tree = shapely.STRtree(polygons)

    def depth(points, lay):
        """Mid-depth of layer ``lay`` at plan points: mean centre elevation of its cells there (NaN outside)."""
        at, cell = cell_tree.query(points, predicate="dwithin", distance=tolerance)
        at, cell = at[layer[cell] == lay], cell[layer[cell] == lay]
        count = np.bincount(at, minlength=len(points))
        return np.bincount(at, weights=middle[cell], minlength=len(points)) / np.where(count, count, np.nan)

    reach = float(np.ptp(xy, axis=0).max())

    def dip(points, a, la, b, lb, fallback):
        """Horizontal and vertical offsets from trace ``a`` (layer la) to trace ``b`` (layer lb) near each point.

        ``b`` is extended along its end segments, so that near the end of a
        trace the offset is still measured across the strike.
        """
        pa = shapely.line_interpolate_point(a, shapely.line_locate_point(a, points))
        b = _extended(b, reach)
        pb = shapely.line_interpolate_point(b, shapely.line_locate_point(b, pa))
        dv = np.abs(depth(pa, la) - depth(pb, lb))
        return shapely.distance(pa, pb), np.where(np.isfinite(dv), dv, fallback)

    ta = np.array([traces[k] for k in layer[n]], dtype=object)
    tb = np.array([traces[k] for k in layer[m]], dtype=object)
    same = np.array([_same(a, b) or a is None or b is None for a, b in zip(ta, tb)])
    hit = np.zeros(len(n), dtype=bool)
    factor = np.ones(len(n))

    # Walls: horizontal links between layers that share the trace (or where one has none), whose face lies on it.
    face = np.empty(len(n), dtype=object)
    horizontal = np.flatnonzero((ihc != 0) & same)
    for t in {id(t): t for t in traces if t is not None}.values():
        own = [k for k, other in enumerate(traces) if _same(other, t)]
        candidates = horizontal[np.isin(layer[n[horizontal]], own) | np.isin(layer[m[horizontal]], own)]
        found, faces = _faces_on(t, candidates, n, m, xy, polygons, tolerance, roundoff)
        hit[found], face[found] = True, faces
    walls = np.flatnonzero(hit)
    if len(walls):
        lay = np.where([t is not None for t in ta[walls]], layer[n[walls]], layer[m[walls]])
        mid = shapely.line_interpolate_point(face[walls], 0.5, normalized=True)
        for k in np.unique(lay):
            w = lay == k
            near = [j for j in (k - 1, k + 1) if 0 <= j < grid.nlay and traces[j] is not None]
            if near:
                fallback = (thickness[n[walls[w]]] + thickness[m[walls[w]]]) / 2
                offsets = [dip(mid[w], traces[k], k, traces[j], j, fallback) for j in near]
                dh, dv = np.mean([o[0] for o in offsets], axis=0), np.mean([o[1] for o in offsets], axis=0)
                factor[walls[w]] = dv / np.hypot(dh, dv)

    # Steps: links between layers whose traces differ.
    steps = {}
    for link in np.flatnonzero(~same):
        key = (layer[n[link]], layer[m[link]])
        if key not in steps:
            steps[key] = _Step(ta[link], tb[link], *key)
    for (la, lb), step in steps.items():
        pair = np.flatnonzero(~same & (layer[n] == la) & (layer[m] == lb))
        dv = (thickness[n[pair]] + thickness[m[pair]]) / 2
        connector = shapely.linestrings(np.stack((xy[n[pair]], xy[m[pair]]), axis=1))
        cut = shapely.dwithin(connector, step.zone, tolerance) & (
            _sides(step.a, xy[n[pair]], tolerance) != _sides(step.b, xy[m[pair]], tolerance))
        vertical = ihc[pair] == 0
        if (cut & vertical).any():
            # Treads: the dip is measured at a point inside the two cells' plan contact.
            v = pair[cut & vertical]
            contact = shapely.point_on_surface(shapely.intersection(polygons[n[v]], polygons[m[v]]))
            dh, d = dip(contact, step.a, la, step.b, lb, dv[cut & vertical])
            hit[v] = True
            factor[v] = dh / np.hypot(dh, d)
        if (cut & ~vertical).any():
            # Staggered horizontal links between dipping layers.
            s = pair[cut & ~vertical]
            dh, d = dip(shapely.points((xy[n[s]] + xy[m[s]]) / 2), step.a, la, step.b, lb, dv[cut & ~vertical])
            hit[s] = True
            factor[s] = d / np.hypot(dh, d)
    # Wherever a trace crosses its layer's cells it must run on cut faces (walls or staggered links).
    staggered = np.flatnonzero(hit & ~same & (ihc != 0))
    face[staggered] = _shared_faces(polygons[n[staggered]], polygons[m[staggered]], roundoff)
    for k, t in enumerate(traces):
        if t is not None:
            near = cell_tree.query(t, predicate="dwithin", distance=roundoff)
            mine = hit & (ihc != 0) & ((layer[n] == k) | (layer[m] == k))
            _require_faces(t, k, face[mine], polygons[near[layer[near] == k]], tolerance, roundoff)
    found = np.flatnonzero(hit)
    cell = grid._node_top - grid._node_bot
    return BarrierFaces(found, n[found], m[found], factor[found], ihc[found] == 0,
                        (cell[n[found]] + cell[m[found]]) / 2)


@dataclass
class BarrierFaces:
    """Connections cut by a barrier, with MF6 and MODFLOW-USG HFB package data.

    ``links`` indexes ``DisuGrid.links``; ``n`` and ``m`` are the zero-based
    nodes on either side. ``vertical`` marks the treads of a dipping fault.
    ``factor`` multiplies the hydraulic characteristic of each face: 1 on a
    vertical fault, and ``|n_fault · n_face|`` on a dipping one, with the dip taken
    from the offset between consecutive traces, each at its layer's mid-depth.
    ``thickness`` is the mean thickness of the two cells of each connection.
    """

    links: np.ndarray
    n: np.ndarray
    m: np.ndarray
    factor: np.ndarray
    vertical: np.ndarray
    thickness: np.ndarray

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
        if np.any(values < 0) and np.any(self.factor != 1):
            raise ValueError(
                "A conductance multiplier (negative hydchr) cannot follow a dipping fault: "
                f"{int(np.sum(self.factor != 1))} of {len(self)} connections have a factor other than 1. "
                "Give the hydraulic characteristic (>= 0), or write your own rows from faces.pairs.")
        return np.where(values < 0, values, values * self.factor)

    def get_gridprops_mf6_hfb(self, hydchr):
        """Return ``ModflowGwfhfb`` kwargs for one stress period, or None when empty.

        ``hydchr`` is the barrier conductivity divided by its width (>= 0; a
        negative value is MODFLOW's conductance multiplier), one number or one
        per connection; it is multiplied by ``factor``. Vertical barrier
        connections need MF6 6.7 or later.
        """
        values = self._hydchr(hydchr)
        if not len(self):
            return None
        return {
            "maxhfb": len(self),
            "stress_period_data": {0: [((int(a),), (int(b),), float(h)) for a, b, h in zip(self.n, self.m, values)]},
        }

    def get_gridprops_mfusg_hfb(self, hydchr):
        """Return ``flopy.modflow.ModflowHfb`` kwargs for a MODFLOW-USG model, or None when empty.

        MODFLOW-USG (and USG-Transport, whose HFB is the hydraulic flow
        barrier) accepts barriers on vertical connections, but multiplies
        their hydraulic characteristic by the mean thickness of the two cells
        as well as by the plan area. For vertical connections ``hydchr`` is
        therefore divided by that thickness, so that the barrier conductance is
        ``hydchr * area`` as in MODFLOW 6. This is exact for confined layers;
        in convertible layers MODFLOW-USG uses the saturated thickness instead.
        """
        values = self._hydchr(hydchr)
        if not len(self):
            return None
        values = np.where(self.vertical, values / self.thickness, values)
        return {
            "nhfbnp": len(self),
            "hfb_data": [[int(a), int(b), float(h)] for a, b, h in zip(self.n, self.m, values)],
        }
