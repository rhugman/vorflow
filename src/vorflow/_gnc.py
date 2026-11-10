"""Ghost-node corrections (GNC) for built DISU grids.

MODFLOW computes the flow between two cells from their two heads, which is exact only when the line joining the cell
centres is normal to their shared face. For every connection where it is not, a ghost point is placed so that the
line joining it to the other end is normal to the face, and its head is interpolated from neighbouring cells. The
weights are non-negative, leave a positive weight on the host cell, use at most five contributors (the
MODFLOW-USG limit) and reproduce any linear head field exactly. They depend on geometry only.
"""

import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from shapely.geometry import Point

from ._gnc_export import GncCorrections

_MAX_CONTRIBUTORS = 5
_HOST_FLOOR = 1e-8   # smallest weight left on the host cell


@dataclass
class _Geometry:
    xyz: np.ndarray
    polygons: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    volume: np.ndarray
    adjacency: list
    normals: np.ndarray
    tolerance: float


def _geometry(grid):
    grid._require_build()
    if grid.options.cl12 != "plan":
        raise ValueError("GNC needs cl12='plan' (perpendicular distances to the face).")
    col = grid._node_cell
    top, bottom = grid._node_top, grid._node_bot
    xyz = np.column_stack((grid._columns["x"].to_numpy(dtype=float)[col],
                           grid._columns["y"].to_numpy(dtype=float)[col], (top + bottom) / 2))
    polygons = grid._columns.geometry.to_numpy()[col]
    links = grid.links
    n, m, ihc = (np.asarray(links[key]) for key in ("n", "m", "ihc"))
    cl_n, cl_m = np.asarray(links["cl_n"], float), np.asarray(links["cl_m"], float)
    angle = np.radians(np.asarray(links["angle"], float))
    normals = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(n))))
    vertical = ihc == 0
    normals[vertical] = 0.0
    normals[vertical, 2] = np.sign(xyz[m[vertical], 2] - xyz[n[vertical], 2])
    adjacency = [[] for _ in range(grid.nodes)]
    for a, b in zip(n, m):
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    adjacency = [sorted(set(nb)) for nb in adjacency]
    return _Geometry(xyz, polygons, top, bottom, (top - bottom) * grid._area[col], adjacency, normals,
                     1e-9 * max(float(np.ptp(xyz, axis=0).max()), 1.0)), (n, m, ihc, cl_n, cl_m)


def estimate_gnc(grid):
    """Ghost-node corrections for every connection whose centre line is not normal to its face.

    See :meth:`vorflow.disu.DisuGrid.estimate_gnc`.
    """
    g, (links_n, links_m, ihc, cl_n, cl_m) = _geometry(grid)
    counts, records, unsupported = Counter(), [], []

    def in_prism(node, target):
        if not g.bottom[node] - g.tolerance <= target[2] <= g.top[node] + g.tolerance:
            return False
        return g.polygons[node].buffer(g.tolerance).covers(Point(*target[:2]))

    def endpoint(n, m, target, link, outside=False, rings=1):
        """Record for a ghost of host n at target; (None, None) when the target is n's centre."""
        offset = target - g.xyz[n]
        if np.linalg.norm(offset) <= g.tolerance:
            return None, None
        if not outside and not in_prism(n, target):
            return None, "ghost outside its cell"
        support = set(g.adjacency[n])
        if rings == 2:
            for j in g.adjacency[n]:
                support.update(g.adjacency[j])
            support.discard(n)
        js = np.array(sorted(support), dtype=int)
        if not len(js):
            return None, "no neighbouring cells"
        vectors = g.xyz[js] - g.xyz[n]
        scale = np.maximum(np.maximum(np.abs(vectors).max(axis=0), np.abs(offset)), g.tolerance)
        distance = np.linalg.norm(vectors, axis=1)
        # Non-negative weights that reproduce the offset (so any linear field), nearest cells preferred.
        fit = linprog(distance / max(distance.max(), g.tolerance) + 1e-6,
                      A_eq=vectors.T / scale[:, None], b_eq=offset / scale,
                      A_ub=np.ones((1, len(js))), b_ub=[1 - _HOST_FLOOR], bounds=(0, None), method="highs")
        if not fit.success:
            return None, "no interpolation from neighbouring cells"
        keep = fit.x > 1e-11
        weights, js = fit.x[keep], js[keep]
        if not 1 <= len(js) <= _MAX_CONTRIBUTORS:
            return None, "more than five contributing cells"
        if weights.sum() > 1 - _HOST_FLOOR + 1e-12:   # the solver's feasibility tolerance can reach a sum of 1
            return None, "no weight left on the host cell"
        if outside and not any(in_prism(j, target) for j in [n, *js]):
            return None, "ghost outside the cells used to interpolate it"
        if np.linalg.norm(weights @ vectors[keep] - offset) > g.tolerance:
            return None, "interpolation residual"
        return dict(n=int(n), m=int(m), link=int(link), target=target.tolist(), contributors=js.tolist(),
                    weights=weights.tolist(), kind="vertical" if ihc[link] == 0 else "horizontal"), None

    def correct(a, b, delta, normal, ca, cb, link, outside, rings):
        """Records for one connection: one ghost (either end, the larger cell first), else one at each end."""
        ends = [(a, b, delta), (b, a, -delta)]
        if g.volume[b] > g.volume[a]:
            ends.reverse()
        for n, m, offset in ends:
            record, reason = endpoint(n, m, g.xyz[n] + offset, link, outside, rings)
            if reason is None:
                return [record] if record is not None else [], None
        # Two ghosts, at ca and cb either side of the face centre along the normal.
        if ihc[link] == 0:
            contact = g.polygons[a].intersection(g.polygons[b])
            if contact.is_empty or contact.area <= 0:
                return [], "no shared area"
            xy = contact.centroid.coords[0]
            targets = (np.r_[xy, g.xyz[a, 2]], np.r_[xy, g.xyz[b, 2]])
        else:
            # Roundoff at the hanging vertices of a splice: find the face within a tolerance band.
            contact = g.polygons[a].boundary.intersection(g.polygons[b].buffer(g.tolerance))
            lower, upper = max(g.bottom[a], g.bottom[b]), min(g.top[a], g.top[b])
            if contact.is_empty or contact.length <= g.tolerance or upper <= lower:
                return [], "no shared face"
            face = np.r_[contact.centroid.coords[0], (lower + upper) / 2]
            targets = (face - ca * normal, face + cb * normal)
        ra, ea = endpoint(a, b, targets[0], link, outside, rings)
        rb, eb = endpoint(b, a, targets[1], link, outside, rings)
        if ea or eb:
            return [], ea or eb
        return [r for r in (ra, rb) if r is not None], None

    for link, (a, b) in enumerate(zip(links_n, links_m)):
        a, b = int(a), int(b)
        normal, distance = g.normals[link], cl_n[link] + cl_m[link]
        delta = g.xyz[b] - g.xyz[a] - distance * normal
        if np.linalg.norm(delta) <= g.tolerance:
            counts["aligned"] += 1
            continue
        if abs(delta @ normal) > g.tolerance:
            reason = "centre distance does not match the connection length"
        else:
            # Ghosts inside their own cell first; then a ghost may leave its cell if it stays inside the cells
            # used to interpolate it; then a second ring of neighbours. A connection left uncorrected next to
            # corrected ones can be less accurate than no correction at all.
            for outside, rings in ((False, 1), (True, 1), (True, 2)):
                added, reason = correct(a, b, delta, normal, cl_n[link], cl_m[link], link, outside, rings)
                if reason is None:
                    records.extend(added)
                    counts["corrected"] += 1
                    counts["two_ghosts" if len(added) == 2 else "one_ghost"] += 1
                    break
            if reason is None:
                continue
        unsupported.append(dict(link=link, n=a, m=b, reason=reason))
        counts["unsupported"] += 1
    if unsupported:
        warnings.warn(f"{len(unsupported)} connections could not be corrected and keep the standard conductance; "
                      "see .unsupported.", RuntimeWarning, stacklevel=3)
    return GncCorrections(records=records, unsupported=unsupported, diagnostics=dict(counts), nodes=grid.nodes)
