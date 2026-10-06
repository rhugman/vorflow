"""Tests for the density-weighted Lloyd relaxation: vorflow._lloyd (no Gmsh) and its pipeline use."""
import math

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box

from vorflow import _lloyd
from vorflow.blueprint import ConceptualMesh
from vorflow.engine import MeshGenerator, _node_edges_from_elements, _node_sizes_from_elements
from vorflow.tessellator import VoronoiTessellator
from vorflow.utils import calculate_mesh_quality


def _square_lattice(n, spacing=1.0, offset=(0.0, 0.0)):
    """(n*n, 2) lattice nodes and the bool mask of nodes off the lattice border."""
    x, y = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
    nodes = np.column_stack([x.ravel(), y.ravel()]) + np.asarray(offset)
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    interior = ((i > 0) & (i < n - 1) & (j > 0) & (j < n - 1)).ravel()
    return nodes, interior


def _hex_lattice(n, spacing=1.0):
    """(n*n, 2) hexagonal lattice nodes."""
    rows = []
    for j in range(n):
        shift = 0.5 * spacing * (j % 2)
        for i in range(n):
            rows.append((i * spacing + shift, j * spacing * np.sqrt(3.0) / 2.0))
    return np.array(rows)


def _constant(value):
    """Size function returning ``value`` everywhere."""
    return lambda xy: np.full(len(np.asarray(xy).reshape(-1, 2)), float(value))


def _voronoi_cells(points, domain):
    """Plain Shapely Voronoi cell (clipped to ``domain``) of each point, in point order."""
    cells = shapely.voronoi_polygons(MultiPoint(points), extend_to=domain)
    polys = np.array(list(cells.geoms), dtype=object)
    point_idx, poly_idx = shapely.STRtree(polys).query(shapely.points(points), predicate='within')
    ordered = np.empty(len(points), dtype=object)
    ordered[point_idx] = polys[poly_idx]
    return shapely.intersection(ordered, domain)


def _reflect(point, a, b):
    """Reflection of ``point`` across the line through ``a`` and ``b``."""
    p, a, b = (np.asarray(v, dtype=float) for v in (point, a, b))
    d = (b - a) / np.linalg.norm(b - a)
    foot = a + np.dot(p - a, d) * d
    return 2.0 * foot - p


# --- size_interpolator ------------------------------------------------------


def test_size_interpolator_exact_at_nodes_and_finite_outside_hull():
    rng = np.random.default_rng(1)
    nodes = rng.uniform(0.0, 10.0, (50, 2)) + (5.0e5, 6.0e6)
    sizes = rng.uniform(0.5, 2.0, 50)
    size_fn = _lloyd.size_interpolator(nodes, sizes)

    np.testing.assert_allclose(size_fn(nodes), sizes, rtol=1e-9)
    far = np.array([[5.0e5 - 100.0, 6.0e6 - 100.0], [5.0e5 + 50.0, 6.0e6 + 5.0]])
    values = size_fn(far)
    assert np.all(np.isfinite(values))
    nearest = np.argmin(np.linalg.norm(nodes[None, :, :] - far[:, None, :], axis=2), axis=1)
    np.testing.assert_array_equal(values, sizes[nearest])


# --- weighted_centroids -----------------------------------------------------


def test_uniform_square_lattice_is_a_fixed_point():
    nodes, _ = _square_lattice(10)
    domain = box(-0.5, -0.5, 9.5, 9.5)
    centroids = _lloyd.weighted_centroids(nodes, np.arange(len(nodes)), _constant(1.0), domain)
    np.testing.assert_allclose(centroids, nodes, atol=1e-9)


def test_uniform_hex_lattice_interior_cells_are_a_fixed_point():
    nodes = _hex_lattice(12)
    domain = box(-1.0, -1.0, 13.0, 11.0)
    cells = _voronoi_cells(nodes, domain)
    # Hexagonal cells away from the lattice edge (area of a unit hex cell).
    interior = np.flatnonzero(np.isclose(shapely.area(cells), np.sqrt(3.0) / 2.0))
    assert len(interior) > 50
    centroids = _lloyd.weighted_centroids(nodes, interior, _constant(1.0), domain)
    np.testing.assert_allclose(centroids, nodes[interior], atol=1e-9)


def test_lattice_fixed_point_with_large_coordinate_offset():
    nodes, _ = _square_lattice(10, spacing=10.0, offset=(5.0e5, 6.0e6))
    domain = box(5.0e5 - 5.0, 6.0e6 - 5.0, 5.0e5 + 95.0, 6.0e6 + 95.0)
    centroids = _lloyd.weighted_centroids(nodes, np.arange(len(nodes)), _constant(10.0), domain)
    np.testing.assert_allclose(centroids, nodes, rtol=0.0, atol=1e-6)


def test_uniform_size_matches_plain_centroid():
    rng = np.random.default_rng(2)
    nodes, interior = _square_lattice(12)
    nodes = nodes + rng.uniform(-0.3, 0.3, nodes.shape)
    domain = box(-1.0, -1.0, 12.0, 12.0)
    idx = np.flatnonzero(interior)
    expected = shapely.get_coordinates(shapely.centroid(_voronoi_cells(nodes, domain)[idx]))
    centroids = _lloyd.weighted_centroids(nodes, idx, _constant(3.0), domain)
    np.testing.assert_allclose(centroids, expected, atol=1e-9)


def test_size_gradient_pulls_centroid_to_the_dense_side():
    rng = np.random.default_rng(3)
    nodes, interior = _square_lattice(12)
    nodes = nodes + rng.uniform(-0.3, 0.3, nodes.shape)
    domain = box(-1.0, -1.0, 12.0, 12.0)
    idx = np.flatnonzero(interior)
    plain = _lloyd.weighted_centroids(nodes, idx, _constant(1.0), domain)

    def decreasing_in_x(xy):
        return 2.0 - 0.1 * np.asarray(xy)[:, 0]

    weighted = _lloyd.weighted_centroids(nodes, idx, decreasing_in_x, domain)
    assert np.all(weighted[:, 0] > plain[:, 0])
    np.testing.assert_allclose(weighted[:, 1], plain[:, 1], atol=0.05)


def test_cell_across_reentrant_corner_uses_clipped_centroid():
    # L-shaped domain; the cell of g is the triangle T, built by reflecting g
    # across T's edges. Every vertex of T is inside the L, but its long edge
    # cuts across the notch, so a vertex-only interior test would be wrong.
    domain = box(0.0, 0.0, 10.0, 10.0).difference(box(5.0, 5.0, 10.0, 10.0))
    triangle = [(4.5, 7.0), (7.0, 4.5), (3.0, 3.0)]
    g = (4.5, 4.5)
    assert Polygon(triangle).contains(Point(g))
    assert all(domain.contains(Point(v)) for v in triangle)
    mirrors = [_reflect(g, triangle[k], triangle[(k + 1) % 3]) for k in range(3)]
    nodes = np.vstack([g, *mirrors])

    centroid = _lloyd.weighted_centroids(nodes, np.array([0]), _constant(1.0), domain)[0]
    clipped = Polygon(triangle).intersection(domain)
    np.testing.assert_allclose(centroid, np.array(clipped.centroid.coords[0]), atol=1e-9)
    assert domain.contains(Point(centroid))
    assert not np.allclose(centroid, np.array(Polygon(triangle).centroid.coords[0]))


def test_boundary_cell_on_l_shaped_domain_is_clipped():
    domain = box(0.0, 0.0, 10.0, 10.0).difference(box(5.0, 5.0, 10.0, 10.0))
    nodes, _ = _square_lattice(10, offset=(0.7, 0.7))
    nodes = nodes[[domain.contains(Point(p)) for p in nodes]]
    corner = int(np.argmin(np.linalg.norm(nodes - (4.7, 4.7), axis=1)))
    cell = _voronoi_cells(nodes, domain)[corner]
    centroid = _lloyd.weighted_centroids(nodes, np.array([corner]), _constant(1.0), domain)[0]
    np.testing.assert_allclose(centroid, np.array(cell.centroid.coords[0]), atol=1e-9)
    assert domain.contains(Point(centroid))


def test_cell_outside_domain_returns_generator():
    nodes, _ = _square_lattice(5)
    domain = box(-0.5, -0.5, 1.5, 1.5)
    far = np.array([24])  # node (4, 4): its cell does not meet the domain
    centroid = _lloyd.weighted_centroids(nodes, far, _constant(1.0), domain)
    np.testing.assert_array_equal(centroid, nodes[far])


# --- accept_moves -----------------------------------------------------------


def test_accept_moves():
    zone = box(0.0, 0.0, 10.0, 10.0)
    line = LineString([(5.0, 0.0), (5.0, 10.0)])
    old = np.array([[2.0, 2.0], [9.0, 5.0], [4.5, 5.0], [3.0, 3.0], [7.0, 7.0]])
    new = np.array([[2.5, 2.2], [11.0, 5.0], [5.5, 5.0], [3.0, 3.0], [7.2, 7.1]])
    owners = [zone, zone, zone, zone, None]
    ok = _lloyd.accept_moves(old, new, owners, line)
    # ordinary, leaves zone, crosses line, no move, no owner constraint
    np.testing.assert_array_equal(ok, [True, False, False, True, True])
    ok = _lloyd.accept_moves(old, new, owners, None)
    np.testing.assert_array_equal(ok, [True, False, True, True, True])


def test_accept_moves_rejects_landing_on_the_owner_boundary():
    zone = box(0.0, 0.0, 10.0, 10.0)
    ok = _lloyd.accept_moves(np.array([[9.0, 5.0]]), np.array([[10.0, 5.0]]), [zone], None)
    assert not ok[0]


# --- relax ------------------------------------------------------------------


def _jittered_lattice(n, jitter, seed):
    """Square lattice with free interior nodes jittered by up to ``jitter``."""
    rng = np.random.default_rng(seed)
    nodes, free = _square_lattice(n)
    nodes[free] += rng.uniform(-jitter, jitter, (free.sum(), 2))
    return nodes, free


def _mean_drift(nodes, idx, domain):
    """Mean |generator - plain centroid| / sqrt(cell area) over ``nodes[idx]``."""
    cells = _voronoi_cells(nodes, domain)[idx]
    centroids = shapely.get_coordinates(shapely.centroid(cells))
    return float(np.mean(np.linalg.norm(nodes[idx] - centroids, axis=1) / np.sqrt(shapely.area(cells))))


def test_relax_keeps_fixed_nodes_and_reduces_drift():
    nodes, free = _jittered_lattice(15, 0.3, seed=4)
    domain = box(0.0, 0.0, 14.0, 14.0)
    original = nodes.copy()
    owners = [domain if f else None for f in free]
    out, report = _lloyd.relax(nodes, free, _constant(1.0), domain, owners, None, iterations=30)

    np.testing.assert_array_equal(nodes, original)  # input not mutated
    assert out[~free].tobytes() == original[~free].tobytes()
    idx = np.flatnonzero(free)
    before, after = _mean_drift(original, idx, domain), _mean_drift(out, idx, domain)
    assert after < 0.25 * before
    assert 1 <= report["iterations"] <= 30
    assert report["rejected"] == 0
    assert set(report) == {"iterations", "max_rel_shift", "rejected"}


def _graded_rings(h0, slope, radius):
    """Concentric rings of nodes with spacing h = h0 + slope * r; returns (nodes, free, outer ring polygon)."""
    rows, r = [(0.0, 0.0)], h0
    while r < radius:
        count = max(6, int(round(2.0 * np.pi * r / (h0 + slope * r))))
        angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + 0.3 * r
        rows += list(zip(r * np.cos(angle), r * np.sin(angle)))
        last = count
        r += h0 + slope * r
    nodes = np.array(rows)
    free = np.ones(len(nodes), dtype=bool)
    free[-last:] = False
    domain = Polygon(nodes[-last:])
    return nodes, free, domain


def _mean_area_near_origin(nodes, domain, radius):
    """Mean Voronoi cell area of the nodes within ``radius`` of the origin."""
    near = np.flatnonzero(np.linalg.norm(nodes, axis=1) < radius)
    return float(np.mean(shapely.area(_voronoi_cells(nodes, domain)[near])))


def test_relax_preserves_grading_with_density_weights():
    nodes, free, domain = _graded_rings(h0=1.0, slope=0.2, radius=30.0)
    owners = [domain] * len(nodes)
    sizes = 1.0 + 0.2 * np.linalg.norm(nodes, axis=1)
    size_fn = _lloyd.size_interpolator(nodes, sizes)
    weighted, _ = _lloyd.relax(nodes, free, size_fn, domain, owners, None, iterations=30)
    unweighted, _ = _lloyd.relax(nodes, free, _constant(1.0), domain, owners, None, iterations=30)

    start = _mean_area_near_origin(nodes, domain, 3.0)
    kept = _mean_area_near_origin(weighted, domain, 3.0)
    lost = _mean_area_near_origin(unweighted, domain, 3.0)
    # Measured: start 1.58, weighted ~1.7, unweighted ~4.7.
    assert kept < 1.2 * start
    assert lost > 2.0 * kept


def test_relax_stops_early_on_centroidal_lattice():
    nodes, free = _square_lattice(10)
    domain = box(0.0, 0.0, 9.0, 9.0)
    out, report = _lloyd.relax(nodes, free, _constant(1.0), domain, [domain] * len(nodes), None,
                               iterations=10)
    assert report["iterations"] < 10
    assert report["max_rel_shift"] < 1e-3
    np.testing.assert_allclose(out, nodes, atol=1e-9)


def _max_residual(nodes, idx, domain):
    """Largest |plain centroid - generator| over ``nodes[idx]`` (unit sizes)."""
    centroids = _lloyd.weighted_centroids(nodes, idx, _constant(1.0), domain)
    return float(np.max(np.linalg.norm(centroids - nodes[idx], axis=1)))


@pytest.mark.parametrize("damping", [1.0, 0.1])
def test_relax_stopping_test_is_damping_independent(damping):
    # A test on the damped step would stop damping 0.1 at a residual up to
    # 10x the tolerance (~0.09 here).
    nodes, free = _jittered_lattice(10, 0.3, seed=7)
    domain = box(0.0, 0.0, 9.0, 9.0)
    idx = np.flatnonzero(free)
    tolerance = 1e-2
    assert _max_residual(nodes, idx, domain) > 10 * tolerance
    out, report = _lloyd.relax(nodes, free, _constant(1.0), domain, [domain] * len(nodes), None,
                               iterations=1000, damping=damping, tolerance=tolerance)
    assert report["iterations"] < 1000
    assert report["max_rel_shift"] < tolerance
    assert _max_residual(out, idx, domain) < tolerance


def test_relax_rejected_nodes_do_not_block_convergence():
    # One free node owns a tiny polygon around itself, so every move it
    # proposes is rejected; the rest of the lattice must still converge.
    nodes, free = _jittered_lattice(10, 0.3, seed=8)
    domain = box(0.0, 0.0, 9.0, 9.0)
    stuck = int(np.flatnonzero(free)[0])
    owners = [domain] * len(nodes)
    owners[stuck] = Point(nodes[stuck]).buffer(1e-9)
    out, report = _lloyd.relax(nodes, free, _constant(1.0), domain, owners, None,
                               iterations=200, tolerance=1e-2)
    assert report["iterations"] < 200
    assert report["rejected"] >= report["iterations"]
    assert out[stuck].tobytes() == nodes[stuck].tobytes()


def test_relax_rejects_moves_across_constraint_line():
    nodes, free = _jittered_lattice(12, 0.3, seed=5)
    domain = box(0.0, 0.0, 11.0, 11.0)
    line = LineString([(5.5, -1.0), (5.5, 12.0)])
    side = nodes[:, 0] < 5.5
    owners = [domain] * len(nodes)
    out, report = _lloyd.relax(nodes, free, _constant(1.0), domain, owners, line, iterations=10)
    np.testing.assert_array_equal(out[:, 0] < 5.5, side)


def test_relax_zero_iterations_or_no_free_nodes_returns_copy():
    nodes, free = _jittered_lattice(6, 0.2, seed=6)
    domain = box(0.0, 0.0, 5.0, 5.0)
    owners = [domain] * len(nodes)
    for mask, iterations in ((free, 0), (np.zeros_like(free), 5)):
        out, report = _lloyd.relax(nodes, mask, _constant(1.0), domain, owners, None, iterations)
        assert out is not nodes
        np.testing.assert_array_equal(out, nodes)
        assert report == {"iterations": 0, "max_rel_shift": 0.0, "rejected": 0}


@pytest.mark.parametrize("kwargs", [
    {"damping": 0.0},
    {"damping": 1.5},
    {"iterations": -1},
    {"iterations": 2.0},
    {"tolerance": -1.0},
])
def test_relax_rejects_invalid_arguments(kwargs):
    nodes, free = _square_lattice(4)
    domain = box(0.0, 0.0, 3.0, 3.0)
    args = {"iterations": 3, **kwargs}
    with pytest.raises(ValueError):
        _lloyd.relax(nodes, free, _constant(1.0), domain, [domain] * len(nodes), None, **args)


@pytest.mark.parametrize("iterations, damping, tolerance", [
    (0, 1.0, 0.0),
    (5, 0.5, 1e-3),
    (np.int64(3), np.float32(0.2), np.float64(1e-4)),
    (2, 1, 0),
])
def test_validate_settings_accepts(iterations, damping, tolerance):
    _lloyd.validate_settings(iterations, damping, tolerance)


@pytest.mark.parametrize("iterations, damping, tolerance", [
    (-1, 1.0, 1e-3),
    (2.0, 1.0, 1e-3),
    (True, 1.0, 1e-3),
    ("3", 1.0, 1e-3),
    (3, 0.0, 1e-3),
    (3, 1.5, 1e-3),
    (3, True, 1e-3),
    (3, float("nan"), 1e-3),
    (3, "0.5", 1e-3),
    (3, None, 1e-3),
    (3, 1.0, -1e-3),
    (3, 1.0, float("nan")),
    (3, 1.0, False),
    (3, 1.0, "0"),
    (3, 1.0, None),
])
def test_validate_settings_rejects(iterations, damping, tolerance):
    with pytest.raises(ValueError):
        _lloyd.validate_settings(iterations, damping, tolerance)


@pytest.mark.parametrize("size_smoothing", [0, 5, np.int64(3)])
def test_validate_settings_accepts_size_smoothing(size_smoothing):
    _lloyd.validate_settings(3, 1.0, 1e-3, size_smoothing=size_smoothing)


@pytest.mark.parametrize("size_smoothing", [-1, 2.0, True, "5", None, float("nan")])
def test_validate_settings_rejects_size_smoothing(size_smoothing):
    with pytest.raises(ValueError, match="lloyd_size_smoothing"):
        _lloyd.validate_settings(3, 1.0, 1e-3, prefix="lloyd_", size_smoothing=size_smoothing)


def test_relax_rejects_mismatched_owner_polys():
    nodes, free = _square_lattice(4)
    domain = box(0.0, 0.0, 3.0, 3.0)
    with pytest.raises(ValueError):
        _lloyd.relax(nodes, free, _constant(1.0), domain, [domain], None, iterations=1)


# --- smooth_log_sizes -------------------------------------------------------


def _lattice_edges(n):
    """(m, 2) positions of the 4-neighbour edges of an n x n ``_square_lattice``."""
    index = np.arange(n * n).reshape(n, n)
    return np.vstack([
        np.column_stack([index[:, :-1].ravel(), index[:, 1:].ravel()]),
        np.column_stack([index[:-1, :].ravel(), index[1:, :].ravel()]),
    ])


def _edge_jitter(log_h, log_true, edges):
    """sd over edges of the neighbour difference of log_h - log_true (noise after removing the grading)."""
    residual = log_h - log_true
    return np.std(residual[edges[:, 0]] - residual[edges[:, 1]])


def test_smooth_log_sizes_keeps_fixed_nodes_bit_identical():
    n = 8
    nodes, free = _square_lattice(n)
    rng = np.random.default_rng(0)
    sizes = np.exp(rng.normal(0.0, 0.3, len(nodes))) * 3.7
    smoothed = _lloyd.smooth_log_sizes(sizes, free, _lattice_edges(n), passes=5)
    np.testing.assert_array_equal(smoothed[~free], sizes[~free])
    assert np.all(smoothed[free] != sizes[free])
    assert np.all(np.isfinite(smoothed)) and np.all(smoothed > 0)


def test_smooth_log_sizes_leaves_free_nodes_without_neighbours():
    sizes = np.array([1.0, 4.0, 2.5])
    smoothed = _lloyd.smooth_log_sizes(sizes, np.array([True, True, True]), np.array([[0, 1]]), passes=3)
    np.testing.assert_allclose(smoothed[:2], [2.0, 2.0])
    assert smoothed[2] == 2.5


def test_smooth_log_sizes_reduces_neighbour_jitter_and_keeps_grading():
    # Sizes graded by a factor 1.2 per lattice spacing, with 10% (sd of log h)
    # node-to-node noise like Gmsh's; the border carries the exact sizes.
    n = 30
    nodes, free = _square_lattice(n)
    edges = _lattice_edges(n)
    log_true = np.log(5.0) + np.log(1.2) * nodes[:, 0]
    rng = np.random.default_rng(1)
    noisy = np.where(free, log_true + rng.normal(0.0, 0.1, len(nodes)), log_true)
    smoothed = np.log(_lloyd.smooth_log_sizes(np.exp(noisy), free, edges, passes=5))

    before = _edge_jitter(noisy, log_true, edges)
    after = _edge_jitter(smoothed, log_true, edges)
    assert after < before / 3.0
    assert np.abs(smoothed - log_true)[free].mean() < np.abs(noisy - log_true)[free].mean() / 2.0
    # A linear log size is a fixed point of the averaging, so the grading stays.
    slope = np.polyfit(nodes[free, 0], smoothed[free], 1)[0]
    assert slope == pytest.approx(np.log(1.2), rel=0.02)


def test_smooth_log_sizes_zero_passes_returns_unchanged_copy():
    nodes, free = _square_lattice(5)
    sizes = np.linspace(1.0, 2.0, len(nodes))
    smoothed = _lloyd.smooth_log_sizes(sizes, free, _lattice_edges(5), passes=0)
    np.testing.assert_array_equal(smoothed, sizes)
    assert smoothed is not sizes


@pytest.mark.parametrize("sizes, free, edges, passes", [
    (np.ones(3), np.ones(2, dtype=bool), [[0, 1]], 1),
    (np.ones(3), np.ones(3, dtype=bool), [[0, 3]], 1),
    (np.ones(3), np.ones(3, dtype=bool), [[-1, 0]], 1),
    (np.array([1.0, 0.0, 1.0]), np.ones(3, dtype=bool), [[0, 1]], 1),
    (np.array([1.0, np.nan, 1.0]), np.ones(3, dtype=bool), [[0, 1]], 1),
    (np.ones(3), np.ones(3, dtype=bool), [[0, 1]], -1),
    (np.ones(3), np.ones(3, dtype=bool), [[0, 1]], 2.0),
    (np.ones(3), np.ones(3, dtype=bool), [[0, 1]], True),
])
def test_smooth_log_sizes_rejects_invalid_arguments(sizes, free, edges, passes):
    with pytest.raises(ValueError):
        _lloyd.smooth_log_sizes(sizes, free, edges, passes)


# --- MeshGenerator inputs ---------------------------------------------------


def test_node_sizes_from_elements_mean_incident_edge_length():
    # Unit square split into two triangles plus a 2 x 1 quad on its right;
    # node 9 is in no element.
    element_data = {
        "blocks": [
            {"connectivity": np.array([[1, 2, 3], [1, 3, 4]])},
            {"connectivity": np.array([[2, 5, 6, 3]])},
        ],
        "node_tags": np.array([1, 2, 3, 4, 5, 6, 9, 3]),
        "node_xy": np.array([[0, 0], [1, 0], [1, 1], [0, 1], [3, 0], [3, 1], [7, 7], [1, 1]], dtype=float),
    }
    sizes = _node_sizes_from_elements(element_data, np.array([1, 2, 5, 9], dtype=np.uint64), fallback=4.0)
    diag = math.sqrt(2.0)
    # Edge 1-3 is shared by both triangles and counted once.
    np.testing.assert_allclose(sizes, [(1 + 1 + diag) / 3, (1 + 1 + 2) / 3, (2 + 1) / 2, 4.0])


def test_node_edges_from_elements_are_positions_of_unique_edges():
    element_data = {
        "blocks": [
            {"connectivity": np.array([[1, 2, 3], [1, 3, 4]])},
            {"connectivity": np.array([[2, 5, 6, 3]])},
        ],
        "node_tags": np.array([1, 2, 3, 4, 5, 6]),
        "node_xy": np.zeros((6, 2)),
    }
    # Node 6 is not a domain node, so edges 5-6 and 6-3 are dropped.
    node_tags = np.array([5, 3, 1, 9, 2, 4], dtype=np.uint64)
    edges = _node_edges_from_elements(element_data, node_tags)
    as_tags = {tuple(sorted(pair)) for pair in node_tags[edges].astype(int).tolist()}
    assert as_tags == {(1, 2), (2, 3), (1, 3), (3, 4), (1, 4), (2, 5)}
    assert len(edges) == len(as_tags)
    empty = _node_edges_from_elements({"blocks": [], "node_tags": [], "node_xy": np.zeros((0, 2))}, [1, 2])
    assert empty.shape == (0, 2)


def test_node_sizes_from_elements_without_elements_uses_fallback():
    element_data = {"blocks": [], "node_tags": np.array([1]), "node_xy": np.zeros((1, 2))}
    np.testing.assert_array_equal(_node_sizes_from_elements(element_data, [1, 2], fallback=3.0), [3.0, 3.0])


# --- VoronoiTessellator(lloyd_iterations=...) --------------------------------


class _FakeMeshGenerator:
    """Mesh generator stand-in without node_is_free / node_sizes / node_edges (a 6 x 6 lattice)."""

    def __init__(self, zones_gdf):
        x, y = np.meshgrid(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 6))
        self.nodes = np.column_stack([x.ravel(), y.ravel()])
        self.node_tags = np.arange(1, len(self.nodes) + 1)
        self.zones_gdf = zones_gdf


def _unit_square_mesh():
    """ConceptualMesh with one unit-square zone, cleaned."""
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 1, 1), zone_id=1)
    cm.generate()
    return cm


def test_lloyd_needs_mesh_generator_node_classification():
    cm = _unit_square_mesh()
    fake = _FakeMeshGenerator(cm.clean_polygons)
    with pytest.raises(ValueError, match="node_is_free"):
        VoronoiTessellator(fake, cm, lloyd_iterations=3).generate()

    fake.node_is_free = np.ones(3, dtype=bool)
    fake.node_sizes = np.ones(3)
    with pytest.raises(ValueError, match="aligned"):
        VoronoiTessellator(fake, cm, lloyd_iterations=3).generate()


def test_lloyd_uses_inputs_captured_with_the_nodes():
    cm = _unit_square_mesh()
    fake = _FakeMeshGenerator(cm.clean_polygons)
    on_border = (fake.nodes == 0.0).any(axis=1) | (fake.nodes == 1.0).any(axis=1)
    fake.node_is_free = ~on_border
    fake.node_sizes = np.linspace(0.15, 0.25, len(fake.nodes))
    fake.node_edges = _lattice_edges(6)
    reference = VoronoiTessellator(fake, cm, lloyd_iterations=3)
    expected = reference.generate()

    tess = VoronoiTessellator(fake, cm, lloyd_iterations=3)
    # A re-run mesh generator replaces these; the tessellator keeps the
    # arrays that belong to the nodes it captured.
    fake.node_is_free = None
    fake.node_sizes = np.ones(3)
    fake.node_edges = np.array([[0, 1]])
    grid = tess.generate()
    assert tess.lloyd_report == reference.lloyd_report
    assert tess.lloyd_report["n_free"] == 16
    np.testing.assert_array_equal(grid[["x", "y", "lloyd_shift"]], expected[["x", "y", "lloyd_shift"]])


def _fake_with_lloyd_inputs(cm):
    """6 x 6 lattice fake with border nodes fixed, noisy sizes and its lattice edges."""
    fake = _FakeMeshGenerator(cm.clean_polygons)
    on_border = (fake.nodes == 0.0).any(axis=1) | (fake.nodes == 1.0).any(axis=1)
    fake.node_is_free = ~on_border
    fake.node_sizes = 0.2 * np.exp(np.random.default_rng(2).normal(0.0, 0.1, len(fake.nodes)))
    fake.node_edges = _lattice_edges(6)
    return fake


def _density_sizes(monkeypatch, tess):
    """Run ``tess.generate()`` and return the per-node sizes its Lloyd density was built from."""
    captured = []
    original = _lloyd.size_interpolator

    def spy(nodes, sizes):
        captured.append(np.array(sizes, copy=True))
        return original(nodes, sizes)

    monkeypatch.setattr(_lloyd, "size_interpolator", spy)
    tess.generate()
    assert len(captured) == 1
    return captured[0]


def test_lloyd_size_smoothing_zero_uses_node_sizes_unchanged(monkeypatch):
    cm = _unit_square_mesh()
    fake = _fake_with_lloyd_inputs(cm)
    fake.node_edges = None   # not needed without smoothing
    sizes = _density_sizes(monkeypatch, VoronoiTessellator(fake, cm, lloyd_iterations=2, lloyd_size_smoothing=0))
    np.testing.assert_array_equal(sizes, fake.node_sizes)


def test_lloyd_smooths_free_node_sizes_only(monkeypatch):
    cm = _unit_square_mesh()
    fake = _fake_with_lloyd_inputs(cm)
    tess = VoronoiTessellator(fake, cm, lloyd_iterations=2)
    assert tess.lloyd_size_smoothing == 5
    sizes = _density_sizes(monkeypatch, tess)
    expected = _lloyd.smooth_log_sizes(fake.node_sizes, fake.node_is_free, fake.node_edges, 5)
    np.testing.assert_array_equal(sizes, expected)
    np.testing.assert_array_equal(sizes[~fake.node_is_free], fake.node_sizes[~fake.node_is_free])
    assert np.all(sizes[fake.node_is_free] != fake.node_sizes[fake.node_is_free])


def test_lloyd_size_smoothing_needs_node_edges():
    cm = _unit_square_mesh()
    fake = _fake_with_lloyd_inputs(cm)
    fake.node_edges = None
    with pytest.raises(ValueError, match="node_edges"):
        VoronoiTessellator(fake, cm, lloyd_iterations=2).generate()
    fake.node_edges = np.array([[0, len(fake.nodes)]])
    with pytest.raises(ValueError, match="node_edges"):
        VoronoiTessellator(fake, cm, lloyd_iterations=2).generate()


def test_lloyd_off_ignores_missing_node_classification():
    cm = _unit_square_mesh()
    grid = VoronoiTessellator(_FakeMeshGenerator(cm.clean_polygons), cm).generate()
    assert len(grid) == 36
    assert "lloyd_shift" not in grid.columns


@pytest.mark.parametrize("kwargs", [
    {"lloyd_iterations": -1},
    {"lloyd_iterations": 2.0},
    {"lloyd_iterations": True},
    {"lloyd_damping": 0.0},
    {"lloyd_damping": 1.5},
    {"lloyd_damping": float("nan")},
    {"lloyd_tolerance": -1e-3},
    {"lloyd_tolerance": float("nan")},
    {"lloyd_size_smoothing": -1},
    {"lloyd_size_smoothing": 5.0},
    {"lloyd_size_smoothing": False},
    {"lloyd_size_smoothing": None},
])
def test_lloyd_rejects_invalid_kwargs(kwargs):
    cm = _unit_square_mesh()
    with pytest.raises(ValueError, match="lloyd_"):
        VoronoiTessellator(_FakeMeshGenerator(cm.clean_polygons), cm, **kwargs)


# Issue-like case: two refined wells in a graded mesh with an inner zone.
WELLS = {"w1": Point(160.0, 140.0), "w2": Point(330.0, 360.0)}
DOMAIN = box(0.0, 0.0, 500.0, 500.0)
INNER_ZONE = Polygon([(60, 260), (440, 230), (470, 470), (90, 440)])


def _well_mesh(barrier=None, quad_buffer_line=None):
    """(cm, mg) for the issue-like case, optionally with a barrier or a quad-buffered line."""
    cm = ConceptualMesh()
    cm.add_polygon(DOMAIN, zone_id=1)
    cm.add_polygon(INNER_ZONE, zone_id=2, z_order=1)
    cm.add_point(WELLS["w1"], point_id="w1", resolution=2.0, growth_factor=1.2)
    cm.add_point(WELLS["w2"], point_id="w2", resolution=3.0, growth_factor=1.2)
    if barrier is not None:
        cm.add_line(barrier, line_id="fault", resolution=10.0, is_barrier=True)
    if quad_buffer_line is not None:
        cm.add_line(quad_buffer_line, line_id="river", resolution=10.0, quad_buffer=True)
    mg = MeshGenerator(background_lc=25.0, verbosity=0)
    mg.generate(*cm.generate())
    return cm, mg


@pytest.fixture(scope="module")
def well_case():
    """The issue-like case tessellated without and with 20 Lloyd passes."""
    cm, mg = _well_mesh()
    baseline = VoronoiTessellator(mg, cm).generate()
    tess = VoronoiTessellator(mg, cm, lloyd_iterations=20)
    relaxed = tess.generate()
    return cm, mg, baseline, relaxed, tess


def _interior_drift(grid):
    """drift_ratio of the cells that do not touch the domain boundary."""
    quality = calculate_mesh_quality(grid)
    interior = ~quality.geometry.intersects(DOMAIN.boundary)
    return quality.loc[interior, "drift_ratio"]


def _by_node(grid, tags):
    """Rows of ``grid`` for the mesh nodes ``tags``, indexed by node_id."""
    return grid[grid["node_id"].isin(tags)].set_index("node_id").loc[tags]


@pytest.mark.slow
def test_mesh_generator_lloyd_inputs_are_aligned(well_case):
    _, mg, _, _, _ = well_case
    assert mg.node_is_free.shape == (len(mg.nodes),)
    assert mg.node_sizes.shape == (len(mg.nodes),)
    assert mg.node_is_free.dtype == bool
    assert 0 < mg.node_is_free.sum() < len(mg.nodes)
    assert np.all(mg.node_sizes > 0)
    assert mg.buffer_footprints is None
    # node_edges are the unique mesh edges between domain nodes; their mean
    # length at each node is node_sizes.
    edges = mg.node_edges
    assert edges.ndim == 2 and edges.shape[1] == 2 and len(edges) > len(mg.nodes)
    assert edges.min() >= 0 and edges.max() < len(mg.nodes)
    assert len(np.unique(np.sort(edges, axis=1), axis=0)) == len(edges)
    length = np.hypot(*(mg.nodes[edges[:, 0]] - mg.nodes[edges[:, 1]]).T)
    ends = edges.ravel()
    mean = np.bincount(ends, np.repeat(length, 2), len(mg.nodes)) / np.bincount(ends, minlength=len(mg.nodes))
    np.testing.assert_allclose(mean, mg.node_sizes, rtol=1e-12)
    # Wells and domain/zone boundary nodes are fixed.
    for well in WELLS.values():
        at_well = np.hypot(*(mg.nodes - [well.x, well.y]).T) < 1e-9
        assert at_well.sum() == 1 and not mg.node_is_free[at_well].any()
    on_boundary = shapely.dwithin(
        shapely.union(DOMAIN.boundary, INNER_ZONE.boundary), shapely.points(mg.nodes), 1e-6
    )
    assert not (mg.node_is_free & on_boundary).any()


@pytest.mark.slow
def test_lloyd_size_smoothing_reduces_mesh_size_jitter(well_case):
    _, mg, _, _, _ = well_case
    edges = mg.node_edges
    # The intended grading: min over wells of res + (1.2 - 1) * distance, capped at background_lc.
    spec = np.full(len(mg.nodes), 25.0)
    for well, res in [(WELLS["w1"], 2.0), (WELLS["w2"], 3.0)]:
        spec = np.minimum(spec, res + 0.2 * np.hypot(*(mg.nodes - [well.x, well.y]).T))
    log_spec = np.log(spec)
    free = mg.node_is_free
    inner = free[edges[:, 0]] & free[edges[:, 1]]
    smoothed = _lloyd.smooth_log_sizes(mg.node_sizes, free, edges, 5)
    before = _edge_jitter(np.log(mg.node_sizes), log_spec, edges[inner])
    after = _edge_jitter(np.log(smoothed), log_spec, edges[inner])
    assert after < before / 2.0


@pytest.mark.slow
def test_lloyd_size_smoothing_zero_output_independent_of_node_edges(well_case):
    cm, mg, _, _, _ = well_case
    reference = VoronoiTessellator(mg, cm, lloyd_iterations=3, lloyd_size_smoothing=0)
    expected = reference.generate()
    tess = VoronoiTessellator(mg, cm, lloyd_iterations=3, lloyd_size_smoothing=0)
    tess.node_edges = None
    grid = tess.generate()
    assert tess.lloyd_report == reference.lloyd_report
    np.testing.assert_array_equal(grid[["x", "y", "lloyd_shift"]], expected[["x", "y", "lloyd_shift"]])
    smoothed = VoronoiTessellator(mg, cm, lloyd_iterations=3).generate()
    assert not np.array_equal(smoothed[["x", "y"]].to_numpy(), expected[["x", "y"]].to_numpy())


@pytest.mark.slow
def test_lloyd_reduces_interior_drift(well_case):
    _, _, baseline, relaxed, tess = well_case
    before, after = _interior_drift(baseline), _interior_drift(relaxed)
    assert after.median() < before.median()
    assert after.quantile(0.95) < before.quantile(0.95)
    assert tess.lloyd_report["iterations"] >= 1
    assert tess.lloyd_report["n_free"] > 0


@pytest.mark.slow
def test_lloyd_keeps_cells_fixed_nodes_and_zones(well_case):
    cm, mg, baseline, relaxed, _ = well_case
    assert len(relaxed) == len(baseline)
    assert set(relaxed["node_id"]) == set(baseline["node_id"])

    tags = list(mg.node_tags)
    rows = _by_node(relaxed, tags)
    fixed = ~mg.node_is_free
    np.testing.assert_array_equal(rows["x"].to_numpy()[fixed], mg.nodes[fixed, 0])
    np.testing.assert_array_equal(rows["y"].to_numpy()[fixed], mg.nodes[fixed, 1])
    assert np.all(rows["lloyd_shift"].to_numpy()[fixed] == 0.0)
    shift = rows["lloyd_shift"].to_numpy()[~fixed]
    assert (shift > 0).mean() > 0.9
    np.testing.assert_allclose(
        rows["lloyd_shift"], np.hypot(rows["x"] - mg.nodes[:, 0], rows["y"] - mg.nodes[:, 1])
    )

    # Every generator stays in its zone, so zone counts do not change.
    assert relaxed["zone_id"].value_counts().to_dict() == baseline["zone_id"].value_counts().to_dict()
    inner = relaxed[relaxed["zone_id"] == 2]
    assert shapely.covered_by(shapely.points(inner[["x", "y"]].to_numpy()), INNER_ZONE).all()


@pytest.mark.slow
def test_lloyd_keeps_refined_well_cells(well_case):
    _, _, baseline, relaxed, _ = well_case
    for well in WELLS.values():
        cell0 = baseline[baseline.contains(well)].iloc[0]
        cell1 = relaxed[relaxed.contains(well)].iloc[0]
        assert (cell1["x"], cell1["y"]) == (well.x, well.y)
        assert cell1.geometry.area == pytest.approx(cell0.geometry.area, rel=0.25)


@pytest.mark.slow
def test_lloyd_off_output_unchanged(well_case):
    cm, mg, baseline, _, _ = well_case
    assert "lloyd_shift" not in baseline.columns
    tess = VoronoiTessellator(mg, cm, lloyd_iterations=0)
    grid = tess.generate()
    assert tess.lloyd_report is None
    rows = _by_node(grid, list(mg.node_tags))
    np.testing.assert_array_equal(rows[["x", "y"]].to_numpy(), mg.nodes)
    assert grid.geometry.geom_equals_exact(baseline.geometry, tolerance=0.0).all()


@pytest.mark.slow
def test_lloyd_with_inset_mirror(well_case):
    cm, mg, _, _, _ = well_case
    baseline = VoronoiTessellator(mg, cm, boundary_centering="inset_mirror").generate()
    relaxed = VoronoiTessellator(mg, cm, boundary_centering="inset_mirror", lloyd_iterations=20).generate()
    assert len(relaxed) == len(baseline)
    assert relaxed["boundary_centered"].sum() == baseline["boundary_centered"].sum()
    assert relaxed.geometry.is_valid.all()
    assert relaxed.geometry.area.sum() == pytest.approx(DOMAIN.area, rel=1e-9)
    assert _interior_drift(relaxed).median() < _interior_drift(baseline).median()


def _side(xy, line):
    """Sign of the cross product of each point against the straight ``line``."""
    (x0, y0), (x1, y1) = line.coords
    return np.sign((x1 - x0) * (xy[:, 1] - y0) - (y1 - y0) * (xy[:, 0] - x0))


@pytest.mark.slow
def test_lloyd_generators_stay_on_their_side_of_a_barrier():
    barrier = LineString([(0.0, 60.0), (500.0, 220.0)])
    cm, mg = _well_mesh(barrier=barrier)
    baseline = VoronoiTessellator(mg, cm).generate()
    relaxed = VoronoiTessellator(mg, cm, lloyd_iterations=20).generate()

    tags = list(mg.node_tags)
    side0 = _side(_by_node(baseline, tags)[["x", "y"]].to_numpy(), barrier)
    rows = _by_node(relaxed, tags)
    np.testing.assert_array_equal(_side(rows[["x", "y"]].to_numpy(), barrier), side0)
    assert (rows["lloyd_shift"] > 0).any()
    # Barrier mirrors are no mesh nodes.
    mirrors = relaxed[~relaxed["node_id"].isin(tags)]
    assert mirrors["lloyd_shift"].isna().all()
    # No cell straddles the barrier.
    pieces = shapely.difference(relaxed.geometry.to_numpy(), barrier.buffer(1e-6))
    assert (shapely.get_num_geometries(pieces) == 1).all()


@pytest.mark.slow
def test_lloyd_keeps_free_nodes_out_of_quad_buffer_strips():
    cm, mg = _well_mesh(quad_buffer_line=LineString([(20.0, 60.0), (480.0, 200.0)]))
    assert mg.buffer_footprints is not None
    relaxed = VoronoiTessellator(mg, cm, lloyd_iterations=20).generate()
    free_tags = list(np.asarray(mg.node_tags)[mg.node_is_free])
    rows = _by_node(relaxed, free_tags)
    assert (rows["lloyd_shift"] > 0).any()
    points = shapely.points(rows[["x", "y"]].to_numpy())
    assert not shapely.intersects(mg.buffer_footprints, points).any()


@pytest.mark.slow
def test_lloyd_keeps_hex_ring_cell_regular():
    point, r = Point(101.3, 98.7), 2.0
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 200, 200), zone_id="domain")
    cm.add_polygon(box(60, 60, 140, 140), zone_id="zone", resolution=10, z_order=1)
    cm.add_point(point, point_id="well", resolution=r, growth_factor=1.2, hex_ring=True)
    mg = MeshGenerator(background_lc=20, verbosity=0)
    mg.generate(*cm.generate())
    grid = VoronoiTessellator(mg, cm, lloyd_iterations=20).generate()

    cell = grid[grid.contains(point)].iloc[0]
    vertices = np.array(cell.geometry.exterior.coords)[:-1]
    assert len(vertices) == 6
    radii = np.hypot(vertices[:, 0] - point.x, vertices[:, 1] - point.y)
    np.testing.assert_allclose(radii, r / math.sqrt(3), rtol=1e-6)
    assert (cell["x"], cell["y"]) == (point.x, point.y)
    assert cell["lloyd_shift"] == 0.0
