import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box

from vorflow.utils import (
    build_connectivity,
    calculate_mesh_quality,
    cells_on,
    check_geometry_resolution,
    faces_on,
    resample_geometry,
    summarize_quality,
)


@pytest.fixture
def paired_polygons():
    poly_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])

    gdf = gpd.GeoDataFrame(
        {
            "geometry": [poly_a, poly_b],
            "x": [0.5, 1.5],
            "y": [0.5, 0.5],
        }
    )
    return gdf


def test_calculate_mesh_quality_computes_expected_scalar_metrics(paired_polygons):
    quality = calculate_mesh_quality(paired_polygons, calc_ortho=True)

    assert "area" in quality.columns
    assert "compactness" in quality.columns
    assert "ortho_error" in quality.columns

    assert quality["area"].iloc[0] == pytest.approx(1.0)
    assert quality["perimeter"].iloc[0] == pytest.approx(4.0)
    assert quality["compactness"].iloc[0] == pytest.approx(np.pi / 4, rel=1e-5)
    assert quality["drift_ratio"].iloc[0] == pytest.approx(0.0, abs=1e-8)
    assert quality["ortho_error"].iloc[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "line",
    [
        LineString([(-1, 0.5), (3, 0.5)]),
        MultiLineString([[(-1, 0.5), (1, 0.5)], [(1, 0.5), (3, 0.5)]]),
    ],
)
def test_cells_on_line_returns_cell_areas_and_intersection_lengths(paired_polygons, line):
    cells = cells_on(paired_polygons, line)

    assert cells["cell_area"].tolist() == pytest.approx([1.0, 1.0])
    assert cells["length"].tolist() == pytest.approx([1.0, 1.0])


def test_cells_on_point_returns_containing_cell_and_zero_length(paired_polygons):
    cells = cells_on(paired_polygons, Point(0.5, 0.5))

    assert len(cells) == 1
    assert cells["cell_area"].tolist() == pytest.approx([1.0])
    assert cells["length"].tolist() == pytest.approx([0.0])


def test_cells_on_point_on_shared_edge_returns_both_touching_cells(paired_polygons):
    cells = cells_on(paired_polygons, Point(1.0, 0.5))

    assert len(cells) == 2


@pytest.mark.parametrize("labels", [[10, 30], ["west", "east"], ["same", "same"]])
@pytest.mark.parametrize(
    ("geometry", "positions"),
    [
        (Point(10, 10), []),
        (Point(1.5, 0.5), [1]),
        (Point(1, 0.5), [0, 1]),
        (LineString([(-1, 0.5), (3, 0.5)]), [0, 1]),
    ],
)
def test_cells_on_returns_positions_and_preserves_indices_and_metadata(
    paired_polygons, labels, geometry, positions
):
    grid = paired_polygons.set_crs("EPSG:32650")
    grid.index = pd.Index(labels, name="source_cell")
    grid["node_id"] = [101, 205]
    grid["zone"] = ["left", "right"]
    before = grid.copy(deep=True)

    cells = cells_on(grid, geometry)

    assert isinstance(cells, gpd.GeoDataFrame)
    assert cells.geometry.name == "geometry"
    assert cells.crs == grid.crs
    assert cells["cell_id"].tolist() == positions
    pd.testing.assert_index_equal(cells.index, grid.index.take(positions))
    pd.testing.assert_frame_equal(cells[grid.columns], grid.iloc[positions])
    pd.testing.assert_frame_equal(grid, before)
    assert {"cell_id", "cell_area", "length"}.issubset(cells.columns)


def test_cells_on_point_at_grid_corner_returns_every_touching_cell():
    grid = gpd.GeoDataFrame(
        {"geometry": [box(x, y, x + 1, y + 1) for y in range(2) for x in range(2)]},
        index=[9, 9, 2, 5],
    )

    cells = cells_on(grid, Point(1, 1))

    assert cells["cell_id"].tolist() == [0, 1, 2, 3]
    assert cells["length"].tolist() == [0.0] * 4


def test_cells_on_empty_grid_retains_geodataframe_schema(paired_polygons):
    grid = paired_polygons.iloc[:0].copy()

    cells = cells_on(grid, Point(0.5, 0.5))

    assert isinstance(cells, gpd.GeoDataFrame)
    assert cells.empty
    assert cells["cell_id"].dtype == np.dtype(np.intp)
    assert list(cells.columns) == [*grid.columns, "cell_area", "length", "cell_id"]


def test_cells_on_rejects_unsupported_geometry(paired_polygons):
    with pytest.raises(TypeError, match="Point, LineString, or MultiLineString"):
        cells_on(paired_polygons, Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))


def test_faces_on_returns_shared_face_length_and_connected_cell_ids(paired_polygons):
    connectivity = build_connectivity(paired_polygons, center="centroid")
    faces = faces_on(connectivity, LineString([(1, -1), (1, 2)]))

    assert len(faces) == 1
    assert faces[["cell_id_1", "cell_id_2"]].iloc[0].tolist() == [0, 1]
    assert faces["face_length"].tolist() == pytest.approx([1.0])


def test_faces_on_tolerance_selects_nearby_shared_faces(paired_polygons):
    connectivity = build_connectivity(paired_polygons, center="centroid")
    faces = faces_on(
        connectivity,
        LineString([(0.99, -1), (0.99, 2)]),
        tolerance=0.02,
    )

    assert len(faces) == 1
    assert faces["face_length"].tolist() == pytest.approx([1.0])


def test_faces_on_zero_tolerance_matches_exact_face_including_endpoints(paired_polygons):
    connectivity = build_connectivity(paired_polygons)

    faces = faces_on(connectivity, LineString([(1, 0), (1, 1)]), tolerance=0)
    shifted = faces_on(
        connectivity, LineString([(1 + 1e-12, 0), (1 + 1e-12, 1)]), tolerance=0
    )

    assert len(faces) == 1
    assert faces.geometry.name == "shared_edge"
    assert faces["face_length"].tolist() == pytest.approx([1.0])
    assert shifted.empty


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, -np.inf, True, False, np.bool_(True)])
def test_faces_on_rejects_nonfinite_and_boolean_tolerance(paired_polygons, tolerance):
    connectivity = build_connectivity(paired_polygons)

    with pytest.raises(ValueError, match="tolerance must be non-negative and finite"):
        faces_on(connectivity, LineString([(1, 0), (1, 1)]), tolerance=tolerance)


def _multipart_connectivity(kind):
    if kind == "bent":
        polygons = [
            Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 3), (0, 3)]),
            box(1, 1, 3, 3),
        ]
    else:
        polygons = [
            MultiPolygon([box(0, 0, 1, 1), box(0, 2, 1, 4)]),
            MultiPolygon([box(1, 0, 2, 1), box(1, 2, 2, 5)]),
        ]
    grid = gpd.GeoDataFrame(
        {"geometry": polygons, "node_id": [50, 70], "x": [0.5, 2], "y": [2, 2]},
        index=["source", "source"],
        crs="EPSG:32650",
    )
    connectivity = build_connectivity(grid)
    connectivity.index = pd.Index(["interface"], name="source_face")
    connectivity["material"] = "fault"
    return connectivity


@pytest.mark.parametrize("kind", ["bent", "disconnected"])
def test_faces_on_requires_complete_interface_and_returns_all_segments(kind):
    connectivity = _multipart_connectivity(kind)
    before = connectivity.copy(deep=True)
    row = connectivity.iloc[0]
    full = row["full_shared_edge"]
    representative = row["shared_edge"]

    assert full.length == pytest.approx(3.0)
    assert representative.length == pytest.approx(2.0)
    assert faces_on(connectivity, representative, tolerance=0).empty
    assert faces_on(connectivity, representative, tolerance=0.01).empty

    faces = faces_on(connectivity, full, tolerance=0)

    assert len(faces) == 1
    assert faces.geometry.name == "shared_edge"
    assert faces.geometry.iloc[0].equals(full)
    assert faces["face_length"].tolist() == pytest.approx([3.0])
    assert faces.crs == connectivity.crs
    pd.testing.assert_index_equal(faces.index, connectivity.index)
    assert faces["material"].tolist() == ["fault"]
    assert faces[["cell_id_1", "cell_id_2"]].iloc[0].tolist() == [0, 1]
    assert faces[["orig_index_1", "orig_index_2"]].iloc[0].tolist() == ["source"] * 2
    for metric in ("angle", "ortho_error", "skewness"):
        assert faces.iloc[0][metric] == pytest.approx(row[metric], nan_ok=True)
    pd.testing.assert_frame_equal(connectivity, before)


def test_faces_on_legacy_report_falls_back_to_supplied_shared_edge():
    connectivity = _multipart_connectivity("disconnected").drop(columns="full_shared_edge")
    representative = connectivity.geometry.iloc[0]

    faces = faces_on(connectivity, representative, tolerance=0)

    assert len(faces) == 1
    assert faces.geometry.iloc[0].equals(representative)
    assert faces["face_length"].tolist() == pytest.approx([2.0])


def test_faces_on_selects_shared_edge_even_when_another_geometry_is_active(paired_polygons):
    connectivity = build_connectivity(paired_polygons).set_geometry("connector")

    faces = faces_on(connectivity, LineString([(1, 0), (1, 1)]), tolerance=0)

    assert len(faces) == 1
    assert faces.geometry.name == "shared_edge"
    assert faces["face_length"].tolist() == pytest.approx([1.0])


def test_faces_on_no_matches_retains_empty_geodataframe_schema(paired_polygons):
    connectivity = build_connectivity(paired_polygons)
    faces = faces_on(connectivity, LineString([(10, 0), (10, 1)]), tolerance=0)

    assert isinstance(faces, gpd.GeoDataFrame)
    assert faces.empty
    assert list(faces.columns) == [*connectivity.columns, "face_length"]
    assert faces.geometry.name == "shared_edge"


def test_faces_on_empty_connectivity_retains_complete_face_schema(paired_polygons):
    connectivity = build_connectivity(paired_polygons.iloc[:0])

    faces = faces_on(connectivity, LineString([(1, 0), (1, 1)]), tolerance=0)

    assert faces.empty
    assert "full_shared_edge" in faces.columns
    assert "face_length" in faces.columns
    assert faces.geometry.name == "shared_edge"


def test_faces_on_multiple_matches_preserve_row_ids_and_duplicate_indices():
    grid = gpd.GeoDataFrame(
        {
            "x": [0.5, 1.5, 0.5, 1.5],
            "y": [0.5, 0.5, 1.5, 1.5],
            "geometry": [box(x, y, x + 1, y + 1) for y in range(2) for x in range(2)],
        },
        index=["cell"] * 4,
    )
    connectivity = build_connectivity(grid)
    connectivity.index = ["face"] * len(connectivity)

    faces = faces_on(connectivity, LineString([(1, 0), (1, 2)]), tolerance=0)

    assert len(faces) == 2
    assert faces.index.tolist() == ["face", "face"]
    assert set(map(tuple, faces[["cell_id_1", "cell_id_2"]].to_numpy())) == {(0, 1), (2, 3)}
    assert faces["face_length"].tolist() == pytest.approx([1.0, 1.0])


def test_faces_on_rejects_negative_tolerance(paired_polygons):
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        faces_on(paired_polygons, LineString([(0, 0), (1, 1)]), tolerance=-1)


def test_faces_on_rejects_non_line_geometry(paired_polygons):
    connectivity = build_connectivity(paired_polygons, center="centroid")
    with pytest.raises(TypeError, match="LineString or MultiLineString"):
        faces_on(connectivity, Point(0.5, 0.5))


def test_faces_on_requires_connectivity_geodataframe(paired_polygons):
    with pytest.raises(ValueError, match="connectivity GeoDataFrame"):
        faces_on(paired_polygons, LineString([(0, 0), (1, 1)]))


class TestCheckGeometryResolution:
    def test_reports_segment_statistics_for_square(self):
        square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        gdf = gpd.GeoDataFrame({"geometry": [square]})
        stats = check_geometry_resolution(gdf)
        assert stats["count"] == 4
        assert stats["min"] == pytest.approx(2.0)
        assert stats["max"] == pytest.approx(2.0)
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["median"] == pytest.approx(2.0)

    def test_mixed_geometries_include_lines_and_skip_points(self):
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        line = LineString([(0, 0), (3, 4)])  # single 5.0 segment
        gdf = gpd.GeoDataFrame({"geometry": [square, line, Point(9, 9)]})
        stats = check_geometry_resolution(gdf)
        assert stats["count"] == 5
        assert stats["max"] == pytest.approx(5.0)

    def test_no_segments_warns_and_returns_empty_stats(self):
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]})
        with pytest.warns(UserWarning, match="no valid segments"):
            stats = check_geometry_resolution(gdf)
        assert stats["count"] == 0
        assert np.isnan(stats["min"]) and np.isnan(stats["mean"])


class TestResampleGeometry:
    def test_linestring_vertices_evenly_spaced(self):
        line = LineString([(0, 0), (10, 0)])
        out = resample_geometry(line, target_spacing=2.5)
        coords = np.asarray(out.coords)
        assert len(coords) == 5  # 4 segments of 2.5
        spacing = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        assert spacing == pytest.approx([2.5] * 4)
        assert out.length == pytest.approx(line.length)

    def test_polygon_preserves_shape_and_holes(self):
        outer = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]],
        )
        out = resample_geometry(outer, target_spacing=1.0)
        assert out.is_valid
        assert len(out.interiors) == 1
        assert out.area == pytest.approx(outer.area, rel=1e-6)
        seg = np.linalg.norm(
            np.diff(np.asarray(out.exterior.coords), axis=0), axis=1
        )
        assert seg.max() <= 1.0 + 1e-9

    def test_point_and_empty_geometries_pass_through(self):
        pt = Point(1, 2)
        assert resample_geometry(pt, 1.0) is pt
        empty = LineString()
        assert resample_geometry(empty, 1.0).is_empty


class TestSummarizeQuality:
    def test_report_lists_cell_counts(self, paired_polygons, caplog):
        quality = calculate_mesh_quality(paired_polygons, calc_ortho=True)
        vlog = logging.getLogger("vorflow")
        old_propagate = vlog.propagate
        vlog.propagate = True  # let caplog's root handler see the records
        try:
            with caplog.at_level(logging.INFO, logger="vorflow.utils"):
                summarize_quality(quality)
        finally:
            vlog.propagate = old_propagate
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "Mesh Quality Report" in messages
        assert "Total Cells: 2" in messages
