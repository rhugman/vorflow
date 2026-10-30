import logging

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from vorflow.blueprint import ConceptualMesh
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
        MultiLineString([[(-1, 0.5), (1, 0.5)], [(1, 0.5), (3, 0.5)]])

    ],
)
def test_cells_on_line_returns_cell_areas_and_intersection_lengths(paired_polygons, line):
    cells = cells_on(paired_polygons, line)

    # length of line
    assert cells["cell_area"].tolist() == pytest.approx([1.0, 1.0])
    assert cells["line_length"].tolist() == pytest.approx([1.0, 1.0])


def test_cells_on_point_returns_containing_cell_and_zero_length(paired_polygons):
    cells = cells_on(paired_polygons, Point(0.5, 0.5))

    assert len(cells) == 1
    assert cells["cell_area"].tolist() == pytest.approx([1.0])
    assert cells["line_length"].tolist() == pytest.approx([0.0])


def test_cells_on_point_on_shared_edge_returns_single_cell(paired_polygons):
    cells = cells_on(paired_polygons, Point(1.0, 0.5))

    assert len(cells) == 1


def test_cells_on_rejects_unsupported_geometry(paired_polygons):
    with pytest.raises(ValueError, match="Point, LineString, or MultiLineString"):
        cells_on(paired_polygons, Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]))


def test_cells_on_accepts_features_from_conceptual_mesh(paired_polygons):
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]), zone_id="domain")
    cm.add_line(
        LineString([(0.25, 0.5), (1.75, 0.5)]),
        line_id="river",
        resolution=1.0,
    )
    cm.add_point(Point(0.5, 0.5), point_id="well", resolution=1.0)
    cm.generate()

    features = cm.features()
    cells = cells_on(paired_polygons, features)

    assert set(cells["feature_id"]) == {"river", "well"}
    assert cells.groupby("feature_id")["feature_type"].first().to_dict() == {
        "river": "line",
        "well": "point",
    }


def test_cells_on_infers_feature_metadata_from_geometry_and_index(paired_polygons):
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]), zone_id="domain")
    cm.add_line(
        LineString([(0.25, 0.5), (1.75, 0.5)]),
        line_id="river",
        resolution=1.0,
    )
    cm.add_point(Point(0.5, 0.5), point_id="well", resolution=1.0)
    cm.generate()

    features = cm.features().drop(columns=["feature_type", "feature_id"])
    expected_types = {
        index: geometry.geom_type
        for index, geometry in features.geometry.items()
        if geometry.geom_type in {"LineString", "Point"}
    }

    cells = cells_on(paired_polygons, features)

    assert cells.groupby("feature_id")["feature_type"].first().to_dict() == expected_types


def test_cells_on_rejects_geometry_outside_grid(paired_polygons):
    with pytest.raises(ValueError, match="the geometry is outside the grid"):
        cells_on(paired_polygons, Point(999,999))


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


def test_faces_on_rejects_negative_tolerance(paired_polygons):
    with pytest.raises(ValueError, match="tolerance must be greater than zero."):
        faces_on(paired_polygons, LineString([(0, 0), (1, 1)]), tolerance=-1)


def test_faces_on_rejects_non_line_geometry(paired_polygons):
    connectivity = build_connectivity(paired_polygons, center="centroid")
    with pytest.raises(ValueError, match="LineString or MultiLineString"):
        faces_on(connectivity, Point(0.5, 0.5))


def test_faces_on_requires_connectivity_geodataframe(paired_polygons):
    with pytest.raises(ValueError, match="connectivity GeoDataFrame"):
        faces_on(paired_polygons, LineString([(0, 0), (1, 1)]))


@pytest.fixture
def grid_3x2():
    """Six unit squares, row positions 0-2 along y=0..1 and 3-5 along y=1..2.

    The index labels (10..60) differ from the row positions so tests catch
    label/position mix-ups. Generators sit at the cell centroids.
    """
    polys = [
        Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
        for j in range(2)
        for i in range(3)
    ]
    return gpd.GeoDataFrame(
        {
            "geometry": polys,
            "x": [p.centroid.x for p in polys],
            "y": [p.centroid.y for p in polys],
        },
        index=[10, 20, 30, 40, 50, 60],
    )


def test_cells_on_line_returns_row_positions_not_index_labels(grid_3x2):
    cells = cells_on(grid_3x2, LineString([(-1, 0.5), (4, 0.5)]))

    assert cells["cell_id"].tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    "point, nearer_position, nearer_xy",
    [
        # On the face shared by positions 4 and 5; position 4's generator is nearer.
        (Point(2.0, 1.4), 4, (1.8, 1.5)),
        # On the vertex shared by positions 1, 2, 4 and 5; position 2's generator is nearer.
        (Point(2.0, 1.0), 2, (2.2, 0.8)),
    ],
)
def test_cells_on_point_on_cell_boundary_returns_nearest_generator_cell(
    grid_3x2, point, nearer_position, nearer_xy
):
    grid = grid_3x2.copy()
    label = grid.index[nearer_position]
    grid.loc[label, ["x", "y"]] = nearer_xy

    cells = cells_on(grid, point)

    assert np.ravel(cells["cell_id"]).tolist() == [nearer_position]


def test_cells_on_point_outside_grid_raises_value_error(grid_3x2):
    with pytest.raises(ValueError):
        cells_on(grid_3x2, Point(10.0, 10.0))


def test_cells_on_line_leaves_out_cells_it_only_touches(grid_3x2):
    # Runs through cells 0 and 1 and ends on the corner shared with 3, 4 and 5.
    cells = cells_on(grid_3x2, LineString([(0.5, 0.5), (2.0, 1.0)]))

    assert cells["cell_id"].tolist() == [0, 1]
    assert (cells["line_length"] > 0).all()


def test_cells_on_features_does_not_modify_the_input(grid_3x2):
    features = gpd.GeoDataFrame(geometry=[LineString([(0.5, 0.5), (2.5, 0.5)])])

    cells_on(grid_3x2, features)

    assert list(features.columns) == ["geometry"]


def test_cells_on_features_skips_polygons_and_features_outside_the_grid_with_a_warning(grid_3x2):
    features = gpd.GeoDataFrame(
        {"feature_id": ["well", "far", "zone"]},
        geometry=[Point(0.5, 0.5), Point(10.0, 10.0), Polygon([(0, 0), (1, 0), (1, 1)])],
    )

    with pytest.warns(UserWarning) as record:
        cells = cells_on(grid_3x2, features)

    assert cells["feature_id"].tolist() == ["well"]
    messages = " ".join(str(w.message) for w in record)
    assert "'far' is outside the grid" in messages
    assert "'zone' is not supported" in messages


def test_cells_on_features_without_matches_returns_an_empty_frame_with_columns(grid_3x2):
    features = gpd.GeoDataFrame({"feature_id": []}, geometry=[])

    cells = cells_on(grid_3x2, features)

    assert cells.empty
    assert {"cell_id", "line_length", "cell_area", "feature_type", "feature_id"} <= set(cells.columns)


def test_faces_on_default_excludes_faces_touching_line_at_an_endpoint(grid_3x2):
    connectivity = build_connectivity(grid_3x2, center="centroid")
    # Runs along the faces between the rows; the faces between neighbours in
    # the same row only touch it at one end.
    faces = faces_on(connectivity, LineString([(0, 1), (3, 1)]))

    pairs = sorted(zip(faces["cell_id_1"], faces["cell_id_2"]))
    assert pairs == [(0, 3), (1, 4), (2, 5)]


def test_faces_on_uses_shared_edge_when_another_geometry_is_active(paired_polygons):
    connectivity = build_connectivity(paired_polygons, center="centroid")
    faces = faces_on(connectivity.set_geometry("connector"), LineString([(1, -1), (1, 2)]))

    assert len(faces) == 1
    assert faces["face_length"].tolist() == pytest.approx([1.0])


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
