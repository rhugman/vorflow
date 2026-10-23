"""Solver-free grid diagnostics retain defects instead of raising on cells."""

import builtins

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon, box

from vorflow._quality import grid_quality_report


def _grid(polygons=None, centres=None, **kwargs):
    polygons = [box(0, 0, 1, 1), box(1, 0, 2, 1)] if polygons is None else polygons
    if centres is None:
        centres = [
            p.representative_point().coords[0]
            if p is not None and not p.is_empty
            else (0, 0)
            for p in polygons
        ]
    return gpd.GeoDataFrame(
        {
            "node_id": np.arange(10, 10 + len(polygons)),
            "x": [p[0] for p in centres],
            "y": [p[1] for p in centres],
        },
        geometry=polygons,
        **kwargs,
    )


def test_two_cells_have_exact_geometry_and_mapping():
    grid = _grid(centres=[(0.25, 0.5), (1.5, 0.5)])
    grid.index = ["same", "same"]  # Original row labels need not be unique IDs.
    report = grid_quality_report(grid, cell_size=1, domain=box(0, 0, 2, 1))
    cells, faces, summary = report["cells"], report["faces"], report["summary"]
    assert summary["hard_valid"]
    assert summary["warning_counts"] == {}
    assert summary["domain_gap_area"] == 0 and summary["outside_domain_area"] == 0
    assert cells.orig_index.tolist() == ["same", "same"]
    assert cells.node_id.tolist() == [10, 11]
    np.testing.assert_allclose(cells.area, [1, 1])
    np.testing.assert_allclose(cells.compactness, [np.pi / 4] * 2)
    assert cells.convex.all() and cells.centre_inside.all()
    face = faces.iloc[0]
    assert (face.cell_id_1, face.cell_id_2) == (0, 1)
    assert face.shared_length == 1
    assert (face.distance_1, face.distance_2) == (0.75, 0.5)
    assert face.ortho_error == 0 and face.intersection_fraction == 0.5
    assert face.midpoint_skew == 0 and not face.connector_misses_face


def test_corner_contacts_are_not_connections():
    report = grid_quality_report(_grid([box(0, 0, 1, 1), box(1, 1, 2, 2)]))
    assert report["faces"].empty
    assert report["summary"]["hard_valid"]
    assert report["summary"]["corner_contacts"] == 1


def test_hanging_vertices_keep_full_shared_face():
    tall = Polygon([(0, 0), (1, 0), (1, 0.5), (1, 2), (0, 2)])
    report = grid_quality_report(_grid([tall, box(1, 0, 2, 2)]))
    face = report["faces"].iloc[0]
    assert report["summary"]["hard_valid"]
    assert face.shared_length == 2
    assert len(shapely.get_coordinates(face.geometry)) >= 3
    assert face.geometry.length == 2


def test_bent_face_is_not_replaced_by_longest_segment():
    left = Polygon([(0, 0), (1, 0), (1.5, 1), (1, 2), (0, 2)])
    right = Polygon([(1, 0), (2, 0), (2, 2), (1, 2), (1.5, 1)])
    report = grid_quality_report(_grid([left, right], [(0.5, 1), (1.8, 1)]))
    face = report["faces"].iloc[0]
    assert not report["summary"]["hard_valid"]
    assert face.bent and face.reasons == ("bent_face",)
    assert face.shared_length == pytest.approx(2 * np.hypot(0.5, 1))
    assert face.geometry.length == pytest.approx(face.shared_length)
    assert np.isnan(face.distance_1)


def test_disconnected_shared_face_is_explicit():
    # Two separated contacts between valid, connected, non-overlapping cells.
    first = Polygon(
        [(0, 0), (1, 0), (1, 1), (0.5, 1), (0.5, 2), (1, 2), (1, 3), (0, 3)]
    )
    second = box(1, 0, 2, 3)
    report = grid_quality_report(_grid([first, second], [(0.25, 1.5), (1.5, 1.5)]))
    face = report["faces"].iloc[0]
    assert face.shared_length == 2
    assert face.disconnected and "disconnected_face" in face.reasons
    assert not report["summary"]["hard_valid"]


@pytest.mark.parametrize("offset", [0, 5_000_000])
def test_material_overlap_detected_even_when_boundaries_only_cross_at_points(offset):
    report = grid_quality_report(
        _grid([box(offset, 0, offset + 2, 2), box(offset + 1, 1, offset + 3, 3)])
    )
    assert not report["summary"]["hard_valid"]
    assert report["summary"]["overlapping_pairs"] == 1
    assert report["faces"].iloc[0].overlap_area == 1
    assert "material_overlap" in report["faces"].iloc[0].reasons
    assert all("material_overlap" in reasons for reasons in report["cells"].reasons)


def test_contained_cell_overlap_is_not_missed():
    report = grid_quality_report(_grid([box(0, 0, 3, 3), box(1, 1, 2, 2)]))
    assert report["summary"]["overlapping_pairs"] == 1
    assert "material_overlap" in report["faces"].iloc[0].reasons
    assert not report["summary"]["hard_valid"]


@pytest.mark.parametrize("offset", [0, 5_000_000])
def test_roundoff_overlap_recovers_face_without_modifying_cells(offset):
    epsilon = np.spacing(float(offset + 1))
    grid = _grid(
        [box(offset, 0, offset + 1, 1), box(offset + 1 - 2 * epsilon, 0, offset + 2, 1)]
    )
    before = grid.geometry.to_wkb().tolist()
    report = grid_quality_report(grid)
    assert report["summary"]["hard_valid"]
    assert report["summary"]["overlapping_pairs"] == 0
    assert len(report["faces"]) == 1
    assert report["faces"].iloc[0].roundoff_overlap
    assert report["faces"].iloc[0].shared_length == pytest.approx(1)
    assert grid.geometry.to_wkb().tolist() == before


@pytest.mark.parametrize("offset", [0, 5_000_000])
def test_roundoff_gap_cannot_silently_hide_missing_connection(offset):
    epsilon = np.spacing(float(offset + 1))
    grid = _grid(
        [box(offset, 0, offset + 1, 1), box(offset + 1 + 2 * epsilon, 0, offset + 2, 1)]
    )
    before = grid.geometry.to_wkb().tolist()
    report = grid_quality_report(grid, domain=box(offset, 0, offset + 2, 1))
    assert not report["summary"]["hard_valid"]
    assert report["summary"]["roundoff_gap_pairs"] == 1
    assert report["summary"]["domain_gap_material_area"] == 0
    face = report["faces"].iloc[0]
    assert face.roundoff_gap and face.gap_distance > 0
    assert face.shared_length == 0 and face.geometry.is_empty
    assert face.near_contact_length == pytest.approx(1)
    assert face.reasons == ("unresolved_roundoff_gap",)
    assert grid.geometry.to_wkb().tolist() == before


def test_nearly_touching_corners_are_not_reported_as_missing_faces():
    epsilon = np.spacing(1.0)
    report = grid_quality_report(
        _grid([box(0, 0, 1, 1), box(1 + epsilon, 1 + epsilon, 2, 2)])
    )
    assert report["summary"]["hard_valid"]
    assert report["summary"]["roundoff_gap_pairs"] == 0
    assert report["faces"].empty


def test_small_deep_overlap_is_not_forgiven_by_area_cutoff():
    first = Polygon(
        [(0, 0), (1, 0), (1, 0.499999), (2, 0.5), (1, 0.500001), (1, 1), (0, 1)]
    )
    report = grid_quality_report(_grid([first, box(1, 0, 2, 1)]))
    assert not report["summary"]["hard_valid"]
    assert report["summary"]["overlapping_pairs"] == 1


def test_outside_nonfinite_duplicate_and_invalid_cells_are_reported():
    bow_tie = Polygon([(3, 0), (4, 1), (3, 1), (4, 0), (3, 0)])
    with_hole = Polygon(
        [(5, 0), (7, 0), (7, 2), (5, 2)],
        [[(5.5, 0.5), (6.5, 0.5), (6.5, 1.5), (5.5, 1.5)]],
    )
    grid = _grid(
        [box(0, 0, 1, 1), box(1, 0, 2, 1), bow_tie, with_hole, Point(8, 0), None],
        [(-0.1, 0.5), (np.inf, 0.5), (3.5, 0.5), (5.25, 0.25), (8, 0), (0, 0)],
    )
    grid.loc[1, "node_id"] = grid.loc[0, "node_id"]
    report = grid_quality_report(grid)
    reasons = report["cells"].reasons.tolist()
    assert "centre_outside" in reasons[0] and "duplicate_node_id" in reasons[0]
    assert "nonfinite_centre" in reasons[1]
    assert "invalid_geometry" in reasons[2]
    assert "cell_has_holes" in reasons[3]
    assert "nonpolygon_geometry" in reasons[4]
    assert "missing_or_empty_geometry" in reasons[5]
    assert not report["summary"]["hard_valid"]
    assert report["summary"]["hard_invalid_cells"] == 6


def test_face_distance_and_side_failures_are_hard_errors():
    report = grid_quality_report(_grid(centres=[(1, 0.5), (1.5, 0.5)]))
    assert "nonpositive_face_distance" in report["faces"].iloc[0].reasons
    assert "centres_not_opposite" in report["faces"].iloc[0].reasons
    assert not report["summary"]["hard_valid"]


def test_connector_missing_finite_face_is_a_warning_with_fraction():
    polygons = [box(0, 0, 1, 3), box(1, 2, 2, 3)]
    report = grid_quality_report(_grid(polygons, [(0.1, 0.1), (1.9, 2.1)]))
    face = report["faces"].iloc[0]
    assert report["summary"]["hard_valid"]
    assert face.connector_misses_face and face.midpoint_skew > 0.5
    assert not 0 <= face.intersection_fraction <= 1
    assert "connector_misses_face" in face.warnings
    assert "nonorthogonal_face" in face.warnings


def test_cell_quality_thresholds_do_not_claim_hard_invalid_geometry():
    report = grid_quality_report(_grid([box(0, 0, 0.01, 1)]), cell_size=1)
    assert report["summary"]["hard_valid"]
    assert (
        report["cells"].iloc[0].small_area and report["cells"].iloc[0].low_compactness
    )
    assert report["summary"]["warning_counts"] == {
        "small_area": 1,
        "low_compactness": 1,
    }


def test_nonconvex_cell_is_a_warning_even_for_small_deep_notch():
    cell = Polygon(
        [
            (0, 0),
            (1, 0),
            (1, 0.499999999999999),
            (0.5, 0.5),
            (1, 0.500000000000001),
            (1, 1),
            (0, 1),
        ]
    )
    report = grid_quality_report(_grid([cell], [(0.25, 0.5)]))
    assert report["summary"]["hard_valid"]
    assert not report["cells"].iloc[0].convex
    assert "nonconvex_cell" in report["cells"].iloc[0].warnings


def test_domain_gaps_and_outside_cells_are_explicit():
    report = grid_quality_report(
        _grid([box(0, 0, 1, 1), box(2, 0, 3, 1)]), domain=box(0, 0, 2.5, 1)
    )
    summary = report["summary"]
    assert not summary["hard_valid"]
    assert summary["domain_gap_area"] == 1
    assert summary["outside_domain_area"] == 0.5
    assert set(summary["domain_reasons"]) == {"domain_gap", "outside_domain"}


def test_projected_coordinate_roundoff_centres_are_accepted():
    offset = 5_000_000
    epsilon = np.spacing(float(offset + 1))
    report = grid_quality_report(
        _grid(
            [box(offset, offset, offset + 1, offset + 1)],
            [(offset - 2 * epsilon, offset + 0.5)],
            crs="EPSG:32750",
        )
    )
    assert report["summary"]["hard_valid"]
    assert report["cells"].crs == "EPSG:32750"
    assert report["summary"]["tolerance"] == 32 * np.finfo(float).eps * (offset + 1)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"cell_size": 0}, "cell_size"),
        ({"cell_size": True}, "cell_size"),
        ({"tolerance": np.nan}, "tolerance"),
        ({"domain": LineString([(0, 0), (1, 1)])}, "domain"),
    ],
)
def test_invalid_options_are_useful_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        grid_quality_report(_grid(), **kwargs)


def test_empty_missing_centres_and_geographic_crs_are_useful_errors():
    with pytest.raises(ValueError, match="non-empty"):
        grid_quality_report(_grid([]))
    with pytest.raises(ValueError, match="numerical-centre"):
        grid_quality_report(_grid().drop(columns="x"))
    with pytest.raises(ValueError, match="reproject"):
        grid_quality_report(_grid(crs="EPSG:4326"))
    with pytest.raises(ValueError, match="CRS must match"):
        grid_quality_report(_grid(crs="EPSG:32750"), domain=_grid(crs="EPSG:28350"))
    with pytest.raises(ValueError, match="active polygon geometry"):
        grid_quality_report(gpd.GeoDataFrame({"x": [0], "y": [0]}))


def test_quality_does_not_import_flopy(monkeypatch):
    original = builtins.__import__

    def without_flopy(name, *args, **kwargs):
        if name == "flopy" or name.startswith("flopy."):
            raise AssertionError("quality diagnostics must not import FloPy")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_flopy)
    assert grid_quality_report(_grid())["summary"]["hard_valid"]
