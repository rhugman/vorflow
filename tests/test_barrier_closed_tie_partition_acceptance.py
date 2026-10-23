"""Independent cell-partition tests for an explicitly approximated closed ring."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

from vorflow.utils import grid_barrier_report


def tied_square(*, asymmetric=False):
    polygons = []
    for y in range(9):
        for x in range(9):
            left = 3.1 if asymmetric and x == 3 else x
            right = 3.1 if asymmetric and x == 2 else x + 1
            polygons.append(box(left, y, right, y + 1))
    grid = gpd.GeoDataFrame(
        {
            "node_id": np.arange(100, 181),
            "x": np.tile(np.arange(9) + 0.5, 9),
            "y": np.repeat(np.arange(9) + 0.5, 9),
        },
        geometry=polygons,
    )
    points = [(2.5, 2.5), (6.5, 2.5), (6.5, 6.5), (2.5, 6.5), (2.5, 2.5)]
    return grid, points


def report(grid, points, *, mode="exclude"):
    lines = gpd.GeoDataFrame(
        {"line_id": ["closed"], "is_barrier": [True]}, geometry=[LineString(points)]
    )
    return grid_barrier_report(grid, lines, approximate=True, closed_ties=mode)


def assert_actual_faces(grid, route, tolerance):
    for row in route.itertuples():
        shared = grid.geometry.iloc[row.left_cell].boundary.intersection(
            grid.geometry.iloc[row.right_cell].boundary
        )
        assert row.geometry.hausdorff_distance(shared) <= 2 * tolerance
        assert row.length == pytest.approx(shared.length, abs=2 * tolerance)


def test_default_still_excludes_centre_ties_and_majority_is_explicit():
    grid, points = tied_square()
    strict = report(grid, points)
    resolved = report(grid, points, mode="majority")
    assert strict["approximate_faces"].empty
    assert len(strict["approximate_ties"]) == 16
    assert not strict["approximate_ties"].resolved.any()
    assert strict["approximate_ties"].assigned_inside.isna().all()
    summary = resolved["approximate_summary"].iloc[0]
    assert summary.single_cycle and summary.usable_single_cycle
    assert len(resolved["approximate_faces"]) == 12
    assert resolved["approximate_faces"].usable.all()
    assert_actual_faces(grid, resolved["approximate_faces"], resolved["tolerance"])


def test_balanced_cells_use_the_declared_outside_convention():
    grid, points = tied_square()
    result = report(grid, points, mode="majority")
    ties = result["approximate_ties"]
    balanced = ties[ties.reason == "balanced_outside"]
    assert len(balanced) == 12
    np.testing.assert_allclose(balanced.inside_fraction, 0.5)
    assert not balanced.assigned_inside.any()
    assert not ties.assigned_inside.any()


def test_majority_assignment_uses_actual_polygon_area_and_retains_traceability():
    grid, points = tied_square(asymmetric=True)
    result = report(grid, points, mode="majority")
    ties = result["approximate_ties"]
    assigned = ties[ties.assigned_inside]
    assert assigned.cell_id.tolist() == [29, 38, 47]
    assert assigned.node_id.tolist() == [129, 138, 147]
    np.testing.assert_allclose(assigned.inside_fraction, 6 / 11)
    summary = result["approximate_summary"].iloc[0]
    assert summary.single_cycle and summary.usable_single_cycle
    assert len(result["approximate_faces"]) == 14
    assert result["approximate_faces"].max_offset.max() >= 0.49
    assert_actual_faces(grid, result["approximate_faces"], result["tolerance"])


def test_reversing_the_ring_swaps_sides_but_preserves_tie_assignments():
    grid, points = tied_square(asymmetric=True)
    forward = report(grid, points, mode="majority")
    reverse = report(grid, points[::-1], mode="majority")
    a, b = forward["approximate_faces"], reverse["approximate_faces"]
    assert set(zip(a.left_cell, a.right_cell)) == set(zip(b.right_cell, b.left_cell))
    left = forward["approximate_ties"].set_index("node_id")
    right = reverse["approximate_ties"].set_index("node_id")
    assert left.assigned_inside.equals(right.assigned_inside)
    np.testing.assert_allclose(left.inside_fraction, right.inside_fraction)


def test_tie_resolution_never_changes_grid_or_exact_target_tracking():
    grid, points = tied_square(asymmetric=True)
    before = grid.copy(deep=True)
    strict = report(grid, points)
    resolved = report(grid, points, mode="majority")
    assert grid.equals(before)
    for key in ["faces", "crossings", "unmatched", "summary"]:
        assert strict[key].equals(resolved[key])
    assert strict["faces"].empty
    assert resolved["summary"].matched_length.iloc[0] == 0
    assert resolved["summary"].unmatched_length.iloc[0] == pytest.approx(16)
