"""Existing-face approximations stay explicit and never invent adjacency."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator
from vorflow.utils import grid_barrier_report


def cells(n=4):
    return gpd.GeoDataFrame(
        {
            "node_id": np.arange(n * n),
            "x": np.tile(np.arange(n) + 0.5, n),
            "y": np.repeat(np.arange(n) + 0.5, n),
        },
        geometry=[box(x, y, x + 1, y + 1) for y in range(n) for x in range(n)],
    )


def report(points):
    lines = gpd.GeoDataFrame(
        {"is_barrier": [True], "line_id": ["fault"]}, geometry=[LineString(points)]
    )
    return grid_barrier_report(cells(), lines, approximate=True)


def test_closed_imperfect_ring_has_a_real_closed_cell_partition_route():
    result = report([(1.2, 1.2), (2.8, 1.2), (2.8, 2.8), (1.2, 2.8), (1.2, 1.2)])
    assert result["faces"].empty
    route = result["approximate_faces"]
    assert len(route) == 8 and route.usable.all()
    assert route.max_offset.max() == pytest.approx(np.sqrt(0.08))
    assert result["approximate_summary"].iloc[0].single_cycle
    for pair in route.itertuples():
        shared = (
            cells()
            .geometry.iloc[pair.left_cell]
            .boundary.intersection(cells().geometry.iloc[pair.right_cell].boundary)
        )
        assert pair.geometry.equals(shared)


def test_reversing_ring_swaps_sides_without_changing_selected_faces():
    points = [(1.2, 1.2), (2.8, 1.2), (2.8, 2.8), (1.2, 2.8), (1.2, 1.2)]
    a = report(points)["approximate_faces"]
    b = report(points[::-1])["approximate_faces"]
    assert set(zip(a.left_cell, a.right_cell)) == set(zip(b.right_cell, b.left_cell))


def test_open_imperfect_line_selects_adjacent_faces_with_offset():
    result = report([(1.8, 0), (1.8, 4)])
    route = result["approximate_faces"]
    assert len(route) == 4
    assert route.max_offset.tolist() == pytest.approx([0.2] * 4)
    assert result["approximate_summary"].iloc[0].components == 1
    assert not result["approximate_summary"].iloc[0].single_cycle


def test_approximation_is_opt_in_and_does_not_change_exact_report():
    lines = gpd.GeoDataFrame(
        {"is_barrier": [True]}, geometry=[LineString([(2, 0), (2, 4)])]
    )
    original = grid_barrier_report(cells(), lines)
    extended = grid_barrier_report(cells(), lines, approximate=True)
    assert "approximate_faces" not in original
    assert extended["faces"].equals(original["faces"])
    assert len(extended["approximate_faces"]) == 4


def test_centres_on_target_are_flagged_and_excluded_from_approximation():
    result = report([(1.5, 0), (1.5, 4)])
    assert result["approximate_faces"].empty
    assert result["approximate_summary"].iloc[0].boundary_centre_ties == 4


def two_cell_report(points, offset=(0.0, 0.0)):
    """A real shared face whose generator connection passes through a bend."""
    x, y = offset
    grid = gpd.GeoDataFrame(
        {"node_id": [8, 27], "x": [x + 0.5, x + 1.5], "y": [y + 0.5, y + 0.5]},
        geometry=[box(x, y, x + 1, y + 1), box(x + 1, y, x + 2, y + 1)],
    )
    lines = gpd.GeoDataFrame(
        {"is_barrier": [True]}, geometry=[LineString(np.asarray(points) + offset)]
    )
    return grid_barrier_report(grid, lines, approximate=True)


@pytest.mark.parametrize("turn", [0.0, 7.2, -7.2, 90.0, -90.0, 150.0, -150.0, 179.9])
@pytest.mark.parametrize("offset", [(0.0, 0.0), (1e7, 6e6)])
def test_single_proper_crossing_at_bend_keeps_real_pair_and_reverses_sides(
    turn, offset
):
    angle = np.radians(turn / 2)
    corner = np.array([1.0, 0.5])
    incoming = np.array([np.sin(angle), np.cos(angle)])
    outgoing = np.array([-np.sin(angle), np.cos(angle)])
    points = [corner - 0.4 * incoming, corner, corner + 0.4 * outgoing]
    forward = two_cell_report(points, offset)
    reverse = two_cell_report(points[::-1], offset)
    pair = forward["approximate_faces"].iloc[0]
    assert (pair.left_cell, pair.right_cell) == (0, 1)
    assert (pair.left_node_id, pair.right_node_id) == (8, 27)
    assert pair.usable and pair.length == pytest.approx(1)
    expected_face = LineString(
        [(offset[0] + 1, offset[1]), (offset[0] + 1, offset[1] + 1)]
    )
    assert pair.geometry.hausdorff_distance(expected_face) <= forward["tolerance"]
    assert list(
        zip(
            reverse["approximate_faces"].left_cell,
            reverse["approximate_faces"].right_cell,
        )
    ) == [(1, 0)]
    summary = forward["approximate_summary"].iloc[0]
    assert summary.pairs == 1 and summary.components == 1
    assert summary.ambiguous_connectors == 0


@pytest.mark.parametrize(
    "points",
    [
        [(0.0, 1.0), (1.0, 0.5), (2.0, 1.0)],  # Tangent touch at the vertex.
        [(1.0, -1.0), (1.0, 0.75), (1.25, 0.75), (1.25, -1.0)],  # Two crossings.
        [(1.0, 0.5), (1.0, 1.0)],  # Touching an open barrier endpoint.
    ],
)
def test_corner_tracking_does_not_turn_touches_or_multiple_hits_into_barriers(points):
    assert two_cell_report(points)["approximate_faces"].empty
    assert two_cell_report(points[::-1])["approximate_faces"].empty


def test_repeated_bend_coordinate_does_not_hide_a_proper_crossing():
    result = two_cell_report([(1.0, 0.0), (1.0, 0.5), (1.0, 0.5), (1.1, 1.0)])
    assert len(result["approximate_faces"]) == 1
    assert result["approximate_summary"].iloc[0].ambiguous_connectors == 0


def test_closed_area_ties_are_opt_in_reversible_and_leave_exact_geometry_unchanged():
    grid = cells(6)
    points = [(0.5, 0.5), (4.5, 0.5), (4.5, 4.5), (0.5, 4.5), (0.5, 0.5)]
    lines = gpd.GeoDataFrame({"is_barrier": [True]}, geometry=[LineString(points)])
    before = grid.copy(deep=True)
    strict = grid_barrier_report(grid, lines, approximate=True)
    result = grid_barrier_report(grid, lines, approximate=True, closed_ties="majority")
    reverse = grid_barrier_report(
        grid,
        lines.set_geometry([LineString(points[::-1])]),
        approximate=True,
        closed_ties="majority",
    )
    assert grid.equals(before)
    for name in ["faces", "crossings", "unmatched", "summary"]:
        assert result[name].equals(strict[name])
    assert strict["approximate_faces"].empty
    summary = result["approximate_summary"].iloc[0]
    assert summary.boundary_centre_ties == summary.resolved_centre_ties == 16
    assert summary.unresolved_centre_ties == 0
    assert summary.balanced_centre_ties == 12
    assert summary.single_cycle and summary.usable_single_cycle
    assert summary.target_max_offset == pytest.approx(np.sqrt(0.5))
    ties = result["approximate_ties"]
    assert ties.resolved.all() and not ties.assigned_inside.any()
    assert set(ties.reason) == {"majority_area", "balanced_outside"}
    forward_pairs, reverse_pairs = (
        result["approximate_faces"],
        reverse["approximate_faces"],
    )
    assert set(zip(forward_pairs.left_cell, forward_pairs.right_cell)) == set(
        zip(reverse_pairs.right_cell, reverse_pairs.left_cell)
    )


def test_closed_tie_option_does_not_resolve_open_line_ties():
    grid = cells()
    lines = gpd.GeoDataFrame(
        {"is_barrier": [True]}, geometry=[LineString([(1.5, 0), (1.5, 4)])]
    )
    result = grid_barrier_report(grid, lines, approximate=True, closed_ties="majority")
    assert result["approximate_faces"].empty
    assert not result["approximate_ties"].resolved.any()
    assert result["approximate_summary"].iloc[0].unresolved_centre_ties == 4


@pytest.mark.parametrize("value", ["nearest", None, True, []])
def test_invalid_closed_tie_policy_is_actionable(value):
    lines = gpd.GeoDataFrame(
        {"is_barrier": [True]}, geometry=[LineString([(2, 0), (2, 4)])]
    )
    with pytest.raises(ValueError, match="closed_ties"):
        grid_barrier_report(cells(), lines, approximate=True, closed_ties=value)


def test_source_report_checks_actual_square_buffer_all_sides_and_meshing_mode():
    cm = ConceptualMesh()
    domain = box(0, 0, 96, 96)
    cm.add_polygon(domain, 1, resolution=16)
    cm.add_line(
        LineString(box(32, 32, 64, 64).exterior.coords),
        "closed",
        8,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
        barrier_policy="preserve",
        barrier_mirrors=False,
        barrier_merge="restore",
        simplify_tolerance=0,
        snap_to_polygons=False,
        densify=False,
    )
    mg = MeshGenerator(background_lc=16, conceptual_mesh=cm, verbosity=0).generate()
    source = mg.source_report.iloc[0]
    assert source.complete
    assert source.requested_area == pytest.approx(1024)
    assert source.actual_area == pytest.approx(1024)
    assert source.quads == 16 and source.triangles == 0
    assert source.transfinite_surfaces == 0 and source.recombine_surfaces == 1
    assert source.quad_area_fraction == pytest.approx(1)
    assert source.target_outside_buffer_length == pytest.approx(0)
    vt = VoronoiTessellator(mg, cm)
    vt.generate()
    assert vt.n_barrier_mirrors == 0
    assert vt.barrier_report()["crossings"].empty
