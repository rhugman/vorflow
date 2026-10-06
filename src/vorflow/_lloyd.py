"""Density-weighted Lloyd relaxation of Voronoi generators.

Pure numerics (NumPy, SciPy, Shapely), no Gmsh calls; ``VoronoiTessellator``
applies it to the mesh nodes before building the Voronoi grid.

Methodology
-----------
*Why.* The generators are Gmsh triangle-mesh nodes, so a cell's generator is
not its centroid. Lloyd's algorithm moves each generator to the centroid of
its cell and repeats; the fixed point is a centroidal Voronoi tessellation
(CVT). Only free nodes move (interior surface nodes); boundary nodes, nodes
of embedded points and lines, and barrier straddle pairs stay where they are.

*Density.* Plain (unweighted) centroids equalise cell sizes and so destroy
the mesh grading. In a 2D CVT with density rho the local generator spacing
scales as rho^(-1/4), so rho = h(x)^-4 keeps the spacing at the target mesh
size h(x). h is interpolated from per-node sizes supplied by the caller
(``size_interpolator``).

*Smoothed sizes.* Gmsh's per-node sizes (mean incident edge length) jitter by
7-12% (sd of log h) between neighbouring nodes once the intended grading is
removed, which is +-30-46% in rho. At growth factor 1.05 that is twice the
real size step between neighbours, and Lloyd converges to a CVT of the noise
(rhugman/vorflow#31). ``smooth_log_sizes`` removes most of it with a few
damped Jacobi passes over the mesh graph in log space: each pass moves a free
node's log size halfway to the mean of its neighbours', while fixed nodes
(boundaries, embedded points and lines, hex-ring seeds) keep their size and
anchor the grading. On the #31 test model 5 passes cut the jitter 3-4x and
gave the same regularity as the exact size field; from 10-20 passes on the
smoothing starts to flatten steep grading.

*Centroids.* For a cell that lies inside the domain the weighted centroid is
integrated over a fan of triangles from the generator (a Voronoi cell is
convex and contains its generator): sum(w_k x_k) / sum(w_k) over quadrature
points x_k, w_k = (area share)_k * rho(x_k). A degree-2 rule per triangle
is used, not the one-point centroid rule (see FAN_QUADRATURE_POINTS). A cell
that reaches outside the domain uses the plain centroid of the cell clipped
to the domain; these are few and next to fixed boundary nodes.

*Safety.* A proposed move is kept only if the new point stays strictly inside
the node's owner polygon (its original zone) and the segment from the old to
the new point does not cross a constraint line (embedded or barrier lines).
A rejected node stays where it is for that iteration.

*Convergence.* Each step is damped (x + damping * (c - x)), but the stopping
test uses the undamped residual |c - x| / h(x) of the accepted nodes, so a
given tolerance means the same distance from the centroids for any damping.
"""
from __future__ import annotations

import numpy as np
import shapely
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Voronoi, cKDTree

# Far ghost generators sit this many bbox extents outside the nodes, as in
# VoronoiTessellator.generate(), so every real node has a finite region.
GHOST_EXTENT_FACTOR = 10.0
# Degree-2 (3-point Gauss) rule on each fan triangle: barycentric weights on
# (generator, vertex, next vertex) and the share of the triangle area. rho =
# h**-4 varies ~40% across a cell where h changes 20% over one cell; the
# one-point centroid rule then biases the centroids enough to coarsen a
# graded mesh by ~30% in cell area near a refined point after 40 iterations,
# while this rule matches an 8x8 subdivision of each triangle.
FAN_QUADRATURE_POINTS = np.array([
    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
])
FAN_QUADRATURE_WEIGHTS = np.array([1.0, 1.0, 1.0]) / 3.0


def size_interpolator(nodes, sizes):
    """Return f(xy (m, 2)) -> (m,) mesh size, linear inside the node hull and nearest-node outside."""
    nodes = np.asarray(nodes, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"nodes must have shape (n, 2). Got {nodes.shape}.")
    if sizes.shape != (len(nodes),):
        raise ValueError(f"sizes must have shape ({len(nodes)},). Got {sizes.shape}.")
    if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
        raise ValueError("sizes must be positive finite numbers.")
    # Qhull (inside LinearNDInterpolator) loses precision on coordinates with
    # a large offset (e.g. UTM), so interpolate about the bbox centre.
    origin = (nodes.min(axis=0) + nodes.max(axis=0)) / 2.0
    linear = LinearNDInterpolator(nodes - origin, sizes)
    tree = cKDTree(nodes - origin)

    def size_fn(xy):
        """Mesh size at each row of ``xy``."""
        local = np.asarray(xy, dtype=float).reshape(-1, 2) - origin
        values = np.asarray(linear(local), dtype=float).reshape(-1)
        outside = np.isnan(values)
        if outside.any():
            _, nearest = tree.query(local[outside])
            values[outside] = sizes[nearest]
        return values

    return size_fn


def smooth_log_sizes(sizes, free, edges, passes):
    """Damped Jacobi smoothing of log(sizes) over the graph ``edges``; returns new sizes.

    ``edges`` is an (m, 2) array of node positions. Each pass sets
    log h_i = (log h_i + mean of log h over i's neighbours) / 2 for every
    ``free`` node that has a neighbour; the other nodes are returned
    bit-identical, and ``passes == 0`` returns an unchanged copy.
    """
    sizes = np.asarray(sizes, dtype=float)
    free = np.asarray(free, dtype=bool)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if sizes.ndim != 1 or free.shape != sizes.shape:
        raise ValueError(f"sizes and free must have the same 1D shape. Got {sizes.shape} and {free.shape}.")
    if len(edges) and (edges.min() < 0 or edges.max() >= len(sizes)):
        raise ValueError(f"edges must hold node positions in [0, {len(sizes)}).")
    if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
        raise ValueError("sizes must be positive finite numbers.")
    validate_size_smoothing(passes)
    result = sizes.copy()
    ends = np.concatenate([edges[:, 0], edges[:, 1]])
    others = np.concatenate([edges[:, 1], edges[:, 0]])
    degree = np.bincount(ends, minlength=len(sizes))
    moving = np.flatnonzero(free & (degree > 0))
    if passes == 0 or len(moving) == 0:
        return result
    log_h = np.log(sizes)
    for _ in range(passes):
        neighbour_mean = np.bincount(ends, weights=log_h[others], minlength=len(sizes))[moving] / degree[moving]
        log_h[moving] = 0.5 * log_h[moving] + 0.5 * neighbour_mean
    result[moving] = np.exp(log_h[moving])
    return result


def _ghost_points(nodes):
    """Four far ghost generators around the bbox of ``nodes``."""
    minx, miny = nodes.min(axis=0)
    maxx, maxy = nodes.max(axis=0)
    buffer = max(maxx - minx, maxy - miny, 1.0) * GHOST_EXTENT_FACTOR
    return np.array([
        [minx - buffer, miny - buffer],
        [maxx + buffer, miny - buffer],
        [maxx + buffer, maxy + buffer],
        [minx - buffer, maxy + buffer],
    ])


def _cell_vertices(nodes, idx):
    """Voronoi vertices of the cells of ``nodes[idx]``, sorted by angle around each generator.

    Coordinates are relative to the bbox centre ``origin`` (returned). Returns
    (origin, vertices (k, 2), owner (k,) position in ``idx``, finite (len(idx),)
    bool). Cells with an unbounded or empty region get no vertices.
    """
    origin = (nodes.min(axis=0) + nodes.max(axis=0)) / 2.0
    local = nodes - origin
    vor = Voronoi(np.vstack([local, _ghost_points(local)]))
    regions = [vor.regions[r] for r in vor.point_region[idx]]
    finite = np.array([len(r) >= 3 and -1 not in r for r in regions], dtype=bool)
    kept = [r for r, ok in zip(regions, finite) if ok]
    counts = np.array([len(r) for r in kept], dtype=int)
    if counts.sum() == 0:
        return origin, np.empty((0, 2)), np.empty(0, dtype=int), finite
    owner = np.repeat(np.flatnonzero(finite), counts)
    vertices = vor.vertices[np.concatenate(kept)]
    rel = vertices - local[idx][owner]
    angle = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.lexsort((angle, owner))
    return origin, vertices[order], owner[order], finite


def _cell_polygons(vertices, owner, origin, count):
    """Shapely polygons (in original coordinates) of ``count`` cells; None for cells without vertices."""
    polys = np.full(count, None, dtype=object)
    if len(owner) == 0:
        return polys
    cells, ring_index = np.unique(owner, return_inverse=True)
    rings = shapely.linearrings(vertices + origin, indices=ring_index)
    polys[cells] = shapely.polygons(rings)
    return polys


def _fan_centroids(vertices, owner, generators, size_fn, origin,
                   points=FAN_QUADRATURE_POINTS, weights=FAN_QUADRATURE_WEIGHTS):
    """Density-weighted centroids of convex cells by fan triangulation from their generators.

    ``vertices`` are sorted by angle within each ``owner`` group; ``generators``
    holds one row per owner value (all in coordinates relative to ``origin``).
    Each fan triangle is integrated with the barycentric rule (``points``,
    ``weights``).
    """
    count = len(generators)
    starts = np.r_[0, np.flatnonzero(np.diff(owner)) + 1]
    last = np.r_[starts[1:], len(owner)] - 1
    nxt = np.arange(len(owner)) + 1
    nxt[last] = starts
    g = generators[owner]
    a = vertices
    b = vertices[nxt]
    area = 0.5 * np.abs((a[:, 0] - g[:, 0]) * (b[:, 1] - g[:, 1]) - (a[:, 1] - g[:, 1]) * (b[:, 0] - g[:, 0]))
    # (quadrature point, triangle, xy)
    sample = points[:, 0, None, None] * g + points[:, 1, None, None] * a + points[:, 2, None, None] * b
    h = np.asarray(size_fn(sample.reshape(-1, 2) + origin), dtype=float).reshape(len(points), -1)
    # Only ratios of the weights matter; scale by the median first so h**-4
    # neither overflows nor underflows for any CRS unit.
    h = h / np.median(h)
    mass = weights[:, None] * area * h ** -4.0
    total = np.bincount(owner, weights=mass.sum(axis=0), minlength=count)
    cx = np.bincount(owner, weights=(mass * sample[:, :, 0]).sum(axis=0), minlength=count)
    cy = np.bincount(owner, weights=(mass * sample[:, :, 1]).sum(axis=0), minlength=count)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.column_stack([cx / total, cy / total])
    degenerate = ~(total > 0)
    result[degenerate] = generators[degenerate]
    return result


def weighted_centroids(nodes, idx, size_fn, domain):
    """Density-weighted (rho = size_fn**-4) centroids of the Voronoi cells of ``nodes[idx]``.

    ``nodes`` (n, 2) holds every generator (fixed and free). A cell covered by
    ``domain`` (Polygon or MultiPolygon) gets its weighted centroid; any other
    cell the plain centroid of its part inside the domain, or its generator if
    that part is empty (or its region is unbounded). Returns (len(idx), 2).
    """
    nodes = np.asarray(nodes, dtype=float)
    idx = np.asarray(idx, dtype=int).reshape(-1)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"nodes must have shape (n, 2). Got {nodes.shape}.")
    result = nodes[idx].copy()
    if len(idx) == 0:
        return result
    origin, vertices, owner, finite = _cell_vertices(nodes, idx)
    polys = _cell_polygons(vertices, owner, origin, len(idx))
    shapely.prepare(domain)
    # covered_by on the whole cell (not only its vertices) because the domain
    # may be non-convex: a convex cell can have every vertex inside an
    # L-shaped domain and still cut across its re-entrant corner.
    interior = np.zeros(len(idx), dtype=bool)
    interior[finite] = shapely.covered_by(polys[finite], domain)

    inner = np.flatnonzero(interior)
    if len(inner):
        keep = interior[owner]
        position = np.full(len(idx), -1, dtype=int)
        position[inner] = np.arange(len(inner))
        local_gen = nodes[idx[inner]] - origin
        local_c = _fan_centroids(vertices[keep], position[owner[keep]], local_gen, size_fn, origin)
        result[inner] = local_c + origin

    edge = np.flatnonzero(finite & ~interior)
    if len(edge):
        clipped = shapely.intersection(polys[edge], domain)
        empty = shapely.is_empty(clipped)
        centres = shapely.get_coordinates(shapely.centroid(clipped[~empty]))
        result[edge[~empty]] = centres
    return result


def _object_array(geometries):
    """1D object array of a sequence of Shapely geometries (or None entries)."""
    items = list(geometries)
    out = np.empty(len(items), dtype=object)
    out[:] = items
    return out


def accept_moves(old, new, owner_polys, constraint_lines):
    """Bool mask (len(old),): True where moving ``old[i]`` to ``new[i]`` is safe.

    A move is safe when the new point lies strictly inside ``owner_polys[i]``
    (a None entry means no owner constraint) and the segment old -> new does
    not intersect ``constraint_lines`` (one Shapely geometry, or None). A
    point that does not move is always accepted.
    """
    old = np.asarray(old, dtype=float)
    new = np.asarray(new, dtype=float)
    if old.shape != new.shape or old.ndim != 2 or old.shape[1] != 2:
        raise ValueError(f"old and new must both have shape (m, 2). Got {old.shape} and {new.shape}.")
    owners = _object_array(owner_polys)
    if len(owners) != len(old):
        raise ValueError(f"owner_polys must have one entry per point ({len(old)}). Got {len(owners)}.")
    ok = np.ones(len(old), dtype=bool)
    moved = np.flatnonzero(np.any(old != new, axis=1))
    if len(moved) == 0:
        return ok

    has_owner = moved[np.array([g is not None for g in owners[moved]], dtype=bool)]
    if len(has_owner):
        shapely.prepare(owners[has_owner])
        ok[has_owner] = shapely.contains_xy(owners[has_owner], new[has_owner, 0], new[has_owner, 1])

    check = moved[ok[moved]]
    if constraint_lines is not None and not shapely.is_empty(constraint_lines) and len(check):
        coords = np.stack([old[check], new[check]], axis=1).reshape(-1, 2)
        segments = shapely.linestrings(coords, indices=np.repeat(np.arange(len(check)), 2))
        shapely.prepare(constraint_lines)
        ok[check] = ~shapely.intersects(constraint_lines, segments)
    return ok


def _is_real_number(value) -> bool:
    """True for an int, float or NumPy number that is not a bool and not NaN."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return not np.isnan(value)


def _is_count(value) -> bool:
    """True for a non-negative int or NumPy integer that is not a bool."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return False
    return value >= 0


def validate_size_smoothing(size_smoothing, prefix=""):
    """Raise ValueError unless size_smoothing is an int >= 0 (bools and 2.0 are rejected)."""
    if not _is_count(size_smoothing):
        raise ValueError(f"{prefix}size_smoothing must be a non-negative integer. Got {size_smoothing!r}.")


def validate_settings(iterations, damping, tolerance, prefix="", size_smoothing=0):
    """Raise ValueError unless iterations and size_smoothing are ints >= 0, damping a number in (0, 1] and tolerance a number >= 0.

    Bools and NaN are rejected; iterations and size_smoothing must be integer
    types (2.0 is rejected), damping and tolerance may be any int or float.
    ``prefix`` is prepended to the setting names in the messages (e.g.
    ``"lloyd_"``).
    """
    if not _is_count(iterations):
        raise ValueError(f"{prefix}iterations must be a non-negative integer. Got {iterations!r}.")
    if not _is_real_number(damping) or not (0.0 < damping <= 1.0):
        raise ValueError(f"{prefix}damping must be a number in (0, 1]. Got {damping!r}.")
    if not _is_real_number(tolerance) or not (tolerance >= 0.0):
        raise ValueError(f"{prefix}tolerance must be a non-negative number. Got {tolerance!r}.")
    validate_size_smoothing(size_smoothing, prefix=prefix)


def _validate_relax_args(nodes, free, owner_polys, iterations, damping, tolerance):
    """Raise ValueError for inconsistent relax() arguments."""
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"nodes must have shape (n, 2). Got {nodes.shape}.")
    if free.shape != (len(nodes),):
        raise ValueError(f"free must be a bool mask of shape ({len(nodes)},). Got {free.shape}.")
    if len(owner_polys) != len(nodes):
        raise ValueError(f"owner_polys must have one entry per node ({len(nodes)}). Got {len(owner_polys)}.")
    validate_settings(iterations, damping, tolerance)


def relax(nodes, free, size_fn, domain, owner_polys, constraint_lines,
          iterations, damping=1.0, tolerance=1e-3):
    """Damped, density-weighted Lloyd relaxation of the free nodes; fixed nodes are returned bit-identical.

    Each iteration moves every free node ``x`` towards its weighted centroid
    ``c`` (``x + damping * (c - x)``), keeping only moves ``accept_moves``
    allows. ``owner_polys`` has one entry per node (len(nodes)); only the
    entries of free nodes are read, so fixed ones may be None.

    The stopping test uses the residual |c - x| / h(x) at the start of an
    iteration, not the damped step, so it does not depend on ``damping``.
    Only nodes whose move was accepted count: a rejected node does not move,
    and would otherwise keep its residual and block convergence. Relaxation
    stops after the iteration in which the largest such residual drops below
    ``tolerance`` (if every move is rejected the residual is 0 and it stops,
    since the next iteration would be identical).

    Returns (new_nodes, report) with report keys ``iterations`` (run),
    ``max_rel_shift`` (largest residual |c - x| / h(x) over the accepted
    nodes of the last iteration; with damping 1 this is the largest step) and
    ``rejected`` (total moves rejected over all iterations).
    """
    nodes = np.asarray(nodes, dtype=float)
    free = np.asarray(free)
    _validate_relax_args(nodes, free, owner_polys, iterations, damping, tolerance)
    free = free.astype(bool)
    current = nodes.copy()
    report = {"iterations": 0, "max_rel_shift": 0.0, "rejected": 0}
    free_idx = np.flatnonzero(free)
    if iterations == 0 or len(free_idx) == 0:
        return current, report

    owners = _object_array(owner_polys[i] for i in free_idx)
    for step in range(iterations):
        before = current[free_idx]
        centroids = weighted_centroids(current, free_idx, size_fn, domain)
        proposal = before + damping * (centroids - before)
        ok = accept_moves(before, proposal, owners, constraint_lines)
        current[free_idx[ok]] = proposal[ok]
        residual = np.where(ok, np.hypot(*(centroids - before).T), 0.0)
        max_rel_shift = float(np.max(residual / size_fn(before)))
        report = {
            "iterations": step + 1,
            "max_rel_shift": max_rel_shift,
            "rejected": report["rejected"] + int((~ok).sum()),
        }
        if max_rel_shift < tolerance:
            break
    return current, report
