import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

import vorflow.engine as engine_module
from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator
from vorflow.utils import build_connectivity, calculate_mesh_quality, calculate_orthogonality


TRIANGULAR_QUALITY_COLUMNS = [
    "element_tag",
    "element_type",
    "element_name",
    "is_triangle",
    "minSICN",
    "minDetJac",
    "maxDetJac",
    "minSJ",
    "minSIGE",
    "gamma",
    "innerRadius",
    "outerRadius",
    "minIsotropy",
    "angleShape",
    "minEdge",
    "maxEdge",
]

ELEMENT_GRID_COLUMNS = [
    "element_tag",
    "element_type",
    "element_name",
    "is_triangle",
    "is_quad",
    "node_tags",
    "centroid_x",
    "centroid_y",
    "geometry",
    "zone_id",
    "z_order",
]



@pytest.fixture
def paired_polygons():
    poly_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    return gpd.GeoDataFrame(
        {
            "node_id": [10, 20],
            "x": [0.5, 1.5],
            "y": [0.5, 0.5],
            "geometry": [poly_a, poly_b],
        }
    )


def test_build_connectivity_reports_generator_based_pair_metrics(paired_polygons):
    connectivity = build_connectivity(paired_polygons)

    assert len(connectivity) == 1
    row = connectivity.iloc[0]
    assert row["cell_id_1"] != row["cell_id_2"]
    assert {row["node_id_1"], row["node_id_2"]} == {10, 20}
    assert row["center_mode"] == "generator"
    assert row["shared_edge"].length == pytest.approx(1.0)
    assert row["full_shared_edge"].equals(row["shared_edge"])
    assert row["connector"].length == pytest.approx(1.0)
    assert row["angle"] == pytest.approx(90.0, abs=1e-6)
    assert row["ortho_error"] == pytest.approx(0.0, abs=1e-6)
    assert row["skewness"] == pytest.approx(0.0, abs=1e-12)


def test_build_connectivity_centroid_mode_works_without_generator_columns(paired_polygons):
    gdf = paired_polygons.drop(columns=["x", "y"])

    connectivity = build_connectivity(gdf, center="centroid")

    assert len(connectivity) == 1
    row = connectivity.iloc[0]
    assert row["center_mode"] == "centroid"
    assert row["angle"] == pytest.approx(90.0, abs=1e-6)
    assert row["ortho_error"] == pytest.approx(0.0, abs=1e-6)
    assert row["skewness"] == pytest.approx(0.0, abs=1e-12)


def test_build_connectivity_generator_mode_still_requires_generator_columns(paired_polygons):
    gdf = paired_polygons.drop(columns=["x"])

    with pytest.raises(ValueError, match="center='generator'"):
        build_connectivity(gdf)


def test_centroid_mode_exposes_asymmetric_cell_center_error():
    left = Polygon([(0, 0), (1, 0), (1, 2), (0, 2)])
    right = Polygon([(1, 0), (3, 0), (3, 1), (1, 1)])
    gdf = gpd.GeoDataFrame(
        {
            "node_id": [1, 2],
            "x": [0.5, 2.0],
            "y": [0.5, 0.5],
            "geometry": [left, right],
        }
    )

    generator_report = build_connectivity(gdf)
    centroid_report = build_connectivity(gdf, center="centroid")
    centroid_quality = calculate_mesh_quality(
        gdf.drop(columns=["x", "y"]),
        calc_ortho=True,
        calc_skewness=True,
        connectivity=centroid_report,
    )

    assert generator_report.iloc[0]["ortho_error"] == pytest.approx(0.0, abs=1e-6)
    assert generator_report.iloc[0]["skewness"] == pytest.approx(0.0, abs=1e-12)
    assert centroid_report.iloc[0]["center_mode"] == "centroid"
    assert centroid_report.iloc[0]["ortho_error"] > 0.0
    # Centroids (0.5, 1) and (2, 0.5) meet the face x=1 at y=5/6; the face
    # midpoint is y=0.5 and the face length is 1.
    assert centroid_report.iloc[0]["skewness"] == pytest.approx(1.0 / 3.0)
    assert centroid_quality["skewness"].tolist() == pytest.approx([1.0 / 3.0, 1.0 / 3.0])
    assert "ortho_error" in centroid_quality.columns


def test_build_connectivity_uses_unique_row_ids_when_index_is_duplicated(paired_polygons):
    duplicated_index = paired_polygons.copy()
    duplicated_index.index = [7, 7]

    connectivity = build_connectivity(duplicated_index)
    quality = calculate_mesh_quality(
        duplicated_index,
        calc_ortho=True,
        calc_skewness=True,
        connectivity=connectivity,
    )

    assert connectivity[["cell_id_1", "cell_id_2"]].iloc[0].tolist() == [0, 1]
    assert connectivity[["orig_index_1", "orig_index_2"]].iloc[0].tolist() == [7, 7]
    assert quality["ortho_error"].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert quality["skewness"].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_calculate_mesh_quality_validates_supplied_connectivity_schema(paired_polygons):
    connectivity = build_connectivity(paired_polygons).drop(columns=["skewness"])

    with pytest.raises(ValueError, match="connectivity missing required columns"):
        calculate_mesh_quality(
            paired_polygons,
            calc_skewness=True,
            connectivity=connectivity,
        )


def test_build_connectivity_skips_point_only_touching_cells():
    gdf = gpd.GeoDataFrame(
        {
            "x": [0.5, 1.5],
            "y": [0.5, 1.5],
            "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)],
        }
    )

    connectivity = build_connectivity(gdf)
    assert connectivity.empty
    assert "full_shared_edge" in connectivity.columns


def test_build_connectivity_uses_longest_multilinestring_shared_boundary():
    left = MultiPolygon([box(0, 0, 1, 1), box(0, 2, 1, 4)])
    right = MultiPolygon([box(1, 0, 2, 1), box(1, 2, 2, 5)])
    gdf = gpd.GeoDataFrame(
        {
            "x": [0.5, 1.5],
            "y": [3.0, 3.0],
            "geometry": [left, right],
        }
    )

    connectivity = build_connectivity(gdf)

    assert len(connectivity) == 1
    assert connectivity.iloc[0]["shared_edge"].length == pytest.approx(2.0)
    assert connectivity.iloc[0]["full_shared_edge"].equals(left.intersection(right))
    assert connectivity.iloc[0]["full_shared_edge"].length == pytest.approx(3.0)


def test_build_connectivity_preserves_bent_face_without_changing_representative_metrics():
    left = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 3), (0, 3)])
    right = box(1, 1, 3, 3)
    grid = gpd.GeoDataFrame(
        {"x": [0.5, 2.0], "y": [2.0, 2.0], "geometry": [left, right]}
    )

    connectivity = build_connectivity(grid)

    row = connectivity.iloc[0]
    assert row["full_shared_edge"].equals(left.intersection(right))
    assert row["full_shared_edge"].length == pytest.approx(3.0)
    assert row["shared_edge"].length == pytest.approx(2.0)
    assert row["angle"] == pytest.approx(90.0)
    assert row["ortho_error"] == pytest.approx(0.0)
    assert row["skewness"] == pytest.approx(0.0)


def test_build_connectivity_full_face_excludes_isolated_point_contact():
    left = MultiPolygon([box(0, 0, 1, 1), box(0, 2, 1, 3)])
    right = MultiPolygon([box(1, 0, 2, 1), box(1, 3, 2, 4)])
    grid = gpd.GeoDataFrame(
        {"x": [0.5, 1.5], "y": [0.5, 0.5], "geometry": [left, right]}
    )

    connectivity = build_connectivity(grid)

    assert left.intersection(right).geom_type == "GeometryCollection"
    assert len(connectivity) == 1
    row = connectivity.iloc[0]
    assert row["full_shared_edge"].equals(LineString([(1, 0), (1, 1)]))
    assert row["full_shared_edge"].equals(row["shared_edge"])


def test_build_connectivity_skewness_uses_face_line_when_connector_misses_face():
    gdf = gpd.GeoDataFrame(
        {
            "x": [0.5, 1.5],
            "y": [2.0, 2.0],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        }
    )

    connectivity = build_connectivity(gdf)

    # Connector y=2 meets the face line x=1 at (1, 2); midpoint (1, 0.5).
    assert len(connectivity) == 1
    assert connectivity.iloc[0]["skewness"] == pytest.approx(1.5)


def test_build_connectivity_skips_rows_without_geometry():
    gdf = gpd.GeoDataFrame(
        {
            "node_id": [10, 20, 30, 40],
            "x": [0.5, 1.5, 0.0, 0.0],
            "y": [0.5, 0.5, 0.0, 0.0],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), None, Polygon()],
        },
        index=[5, 3, 9, 1],
    )

    for center in ("generator", "centroid"):
        connectivity = build_connectivity(gdf, center=center)

        assert len(connectivity) == 1
        row = connectivity.iloc[0]
        assert [row["cell_id_1"], row["cell_id_2"]] == [0, 1]
        assert [row["orig_index_1"], row["orig_index_2"]] == [5, 3]
        assert [row["node_id_1"], row["node_id_2"]] == [10, 20]
        assert row["center_1"].equals(Point(0.5, 0.5))
        assert row["connector"].equals(LineString([(0.5, 0.5), (1.5, 0.5)]))
        assert row["shared_edge"].length == pytest.approx(1.0)
        assert row["angle"] == pytest.approx(90.0)


def test_build_connectivity_infers_node_id_dtype_from_object_column(paired_polygons):
    # inset_mirror grids carry uint64 tags in an object column (ghost cells add -1).
    grid = paired_polygons.copy()
    grid["node_id"] = pd.Series([np.uint64(5), np.uint64(7)], dtype=object)
    grid.index = pd.Index([np.uint64(3), np.uint64(4)], dtype=object)

    connectivity = build_connectivity(grid)

    assert connectivity["node_id_1"].dtype == np.uint64
    assert connectivity["node_id_2"].dtype == np.uint64
    assert connectivity["orig_index_1"].dtype == np.uint64


def test_calculate_orthogonality_regression_value_unchanged(paired_polygons):
    result = calculate_orthogonality(paired_polygons)
    expected = pd.Series([0.0, 0.0], index=paired_polygons.index)

    pd.testing.assert_series_equal(result, expected)


def _reference_orthogonality_02c82b6(gdf):
    """Return per-cell orthogonality error as computed by utils.py at commit 02c82b6."""
    df = gdf.copy()
    if not df.index.is_unique:
        df = df.reset_index(drop=True)
    neighbors = gpd.sjoin(df, df, how="inner", predicate="touches")
    pairs = neighbors[neighbors.index < neighbors.index_right].copy()
    if pairs.empty:
        return pd.Series(0.0, index=gdf.index)
    gx = df.loc[pairs.index_right, "x"].values - df.loc[pairs.index, "x"].values
    gy = df.loc[pairs.index_right, "y"].values - df.loc[pairs.index, "y"].values
    geoms1 = df.loc[pairs.index, "geometry"].values
    geoms2 = df.loc[pairs.index_right, "geometry"].values
    errors = []
    for i in range(len(pairs)):
        inter = geoms1[i].intersection(geoms2[i])
        if inter.is_empty or inter.geom_type not in ["LineString", "MultiLineString"]:
            errors.append(np.nan)
            continue
        if inter.geom_type == "MultiLineString":
            if not inter.geoms:
                errors.append(np.nan)
                continue
            edge = max(inter.geoms, key=lambda x: x.length)
        else:
            edge = inter
        coords = list(edge.coords)
        if len(coords) < 2:
            errors.append(np.nan)
            continue
        nx = -(coords[-1][1] - coords[0][1])
        ny = coords[-1][0] - coords[0][0]
        mag_g = np.sqrt(gx[i] ** 2 + gy[i] ** 2)
        mag_n = np.sqrt(nx**2 + ny**2)
        if mag_g == 0 or mag_n == 0:
            errors.append(np.nan)
            continue
        cos_theta = min(1.0, max(0.0, abs(gx[i] * nx + gy[i] * ny) / (mag_g * mag_n)))
        errors.append(np.degrees(np.arccos(cos_theta)))
    pairs["ortho_error"] = errors
    s1 = pairs["ortho_error"].groupby(pairs.index).max()
    s2 = pairs["ortho_error"].groupby(pairs.index_right).max()
    combined = pd.concat([s1, s2], axis=1).max(axis=1)
    return combined.reindex(gdf.index).fillna(0.0)


def _generated_voronoi_grid():
    """Return a refined vorflow Voronoi grid with a line and a point feature."""
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 20, 12), zone_id=1, resolution=2.0)
    cm.add_line(LineString([(3, 3), (16, 9)]), "river", resolution=0.6)
    cm.add_point(Point(15, 3), "well", resolution=0.4)
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=2.0, verbosity=0)
    assert mesher.generate(clean_polys, clean_lines, clean_points)
    return VoronoiTessellator(mesher, cm, clip_to_boundary=True).generate()


def _square_lattice_voronoi(n=4):
    """Return the clipped Voronoi grid of an n x n unit square lattice: unit boxes."""
    centers = np.arange(n) + 0.5
    points = np.array([(x, y) for y in centers for x in centers])
    return gpd.GeoDataFrame(
        {"x": points[:, 0], "y": points[:, 1]},
        geometry=[box(x - 0.5, y - 0.5, x + 0.5, y + 0.5) for x, y in points],
    )


@pytest.mark.parametrize("center", ["generator", "centroid"])
def test_square_lattice_voronoi_has_zero_face_skewness(center):
    grid = _square_lattice_voronoi()

    connectivity = build_connectivity(grid, center=center)
    quality = calculate_mesh_quality(
        grid, calc_skewness=True, connectivity=connectivity
    )

    assert len(connectivity) == 2 * 4 * 3
    assert connectivity["skewness"].to_numpy() == pytest.approx(0.0, abs=1e-9)
    assert quality["skewness"].to_numpy() == pytest.approx(0.0, abs=1e-9)


def test_generator_skewness_detects_off_center_voronoi_face():
    # The shared face x=0.5 is the exact perpendicular bisector of the two
    # generators, but spans y in [-0.2, 0.8]: the connector (y=0) meets it
    # 0.3 below the face midpoint (y=0.3), and the face length is 1.
    gdf = gpd.GeoDataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "geometry": [box(-0.5, -0.2, 0.5, 0.8), box(0.5, -0.2, 1.5, 0.8)],
        }
    )

    connectivity = build_connectivity(gdf)
    quality = calculate_mesh_quality(gdf, calc_ortho=True, calc_skewness=True)

    assert connectivity.iloc[0]["ortho_error"] == pytest.approx(0.0, abs=1e-6)
    assert connectivity.iloc[0]["skewness"] == pytest.approx(0.3)
    assert quality["skewness"].tolist() == pytest.approx([0.3, 0.3])


def test_skewness_of_oblique_connector_matches_analytic_value():
    # Generators (0, 0) and (2, 1): connector y = x / 2 meets the face line
    # x = 1 at y = 0.5; the face runs from y = -1 to y = 1 (midpoint y = 0).
    gdf = gpd.GeoDataFrame(
        {
            "x": [0.0, 2.0],
            "y": [0.0, 1.0],
            "geometry": [box(-1, -1, 1, 1), box(1, -1, 3, 1)],
        }
    )

    connectivity = build_connectivity(gdf)

    assert connectivity.iloc[0]["skewness"] == pytest.approx(0.5 / 2.0)


def test_skewness_is_nan_without_warning_when_connector_parallels_face():
    gdf = gpd.GeoDataFrame(
        {
            "x": [1.0, 1.0],
            "y": [0.2, 0.8],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        connectivity = build_connectivity(gdf)
        quality = calculate_mesh_quality(
            gdf, calc_skewness=True, connectivity=connectivity
        )

    assert np.isnan(connectivity.iloc[0]["skewness"])
    assert quality["skewness"].isna().all()


def test_generated_grid_generator_skewness_is_not_constant():
    grid = _generated_voronoi_grid()

    connectivity = build_connectivity(grid, center="generator")
    quality = calculate_mesh_quality(grid, calc_skewness=True)
    skewness = connectivity["skewness"].to_numpy()

    assert np.isfinite(skewness).all()
    assert (skewness >= 0.0).all()
    assert skewness.std() > 0.01
    assert np.median(skewness) > 0.01
    assert quality["skewness"].max() > 0.1
    assert quality["skewness"].nunique() > len(quality) // 2


def test_calculate_orthogonality_matches_02c82b6_on_generated_grid():
    grid = _generated_voronoi_grid()
    # Replacing generators by centroids makes the orthogonality non-trivial.
    centroid_grid = grid.assign(x=grid.geometry.centroid.x, y=grid.geometry.centroid.y)

    for gdf in (grid, centroid_grid):
        reference = _reference_orthogonality_02c82b6(gdf)
        via_connectivity = calculate_mesh_quality(gdf, calc_ortho=True, calc_skewness=True)

        pd.testing.assert_series_equal(calculate_orthogonality(gdf), reference)
        assert via_connectivity["ortho_error"].to_numpy() == pytest.approx(
            reference.to_numpy(), abs=1e-5
        )
    assert reference.max() > 1.0


def test_calculate_mesh_quality_skewness_is_opt_in(paired_polygons):
    default_quality = calculate_mesh_quality(paired_polygons)

    assert "ortho_error" not in default_quality.columns
    assert "skewness" not in default_quality.columns

    connectivity = build_connectivity(paired_polygons)
    skew_quality = calculate_mesh_quality(
        paired_polygons,
        calc_ortho=True,
        calc_skewness=True,
        connectivity=connectivity,
    )

    assert "ortho_error" in skew_quality.columns
    assert "skewness" in skew_quality.columns
    assert skew_quality["ortho_error"].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert skew_quality["skewness"].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_get_triangular_quality_requires_generation():
    mesher = MeshGenerator(background_lc=2.0, verbosity=0)

    with pytest.raises(RuntimeError, match="Call MeshGenerator.generate"):
        mesher.get_triangular_quality()

    with pytest.raises(RuntimeError, match="Call MeshGenerator.generate"):
        mesher.get_element_grid()


def test_collect_triangular_quality_preserves_unavailable_metric_columns(monkeypatch):
    mesher = MeshGenerator(background_lc=2.0, verbosity=0)
    monkeypatch.setattr(
        mesher,
        "_get_2d_elements",
        lambda _surface_tags: ([2], [np.array([101])], [np.array([1, 2, 3])]),
    )
    monkeypatch.setattr(
        engine_module.gmsh.model.mesh,
        "getElementProperties",
        lambda _element_type: ("Triangle 3", 2, 1, 3, [], 3),
    )

    unavailable = {"minDetJac", "maxDetJac", "innerRadius"}

    def get_element_qualities(_tags, measure):
        if measure in unavailable:
            raise Exception(f"Unknown quality name '{measure}'")
        return np.array([0.5])

    monkeypatch.setattr(
        engine_module.gmsh.model.mesh,
        "getElementQualities",
        get_element_qualities,
    )

    quality = mesher._collect_triangular_quality()

    assert list(quality.columns) == TRIANGULAR_QUALITY_COLUMNS
    assert quality[list(unavailable)].isna().all().all()
    assert quality["minSICN"].tolist() == [0.5]


def test_collect_triangular_quality_propagates_unrelated_gmsh_errors(monkeypatch):
    mesher = MeshGenerator(background_lc=2.0, verbosity=0)
    monkeypatch.setattr(
        mesher,
        "_get_2d_elements",
        lambda _surface_tags: ([2], [np.array([101])], [np.array([1, 2, 3])]),
    )
    monkeypatch.setattr(
        engine_module.gmsh.model.mesh,
        "getElementProperties",
        lambda _element_type: ("Triangle 3", 2, 1, 3, [], 3),
    )

    def raise_invalid_model_error(_tags, _measure):
        raise Exception("Invalid element tag in current model")

    monkeypatch.setattr(
        engine_module.gmsh.model.mesh,
        "getElementQualities",
        raise_invalid_model_error,
    )

    with pytest.raises(Exception, match="Invalid element tag"):
        mesher._collect_triangular_quality()


def test_get_triangular_quality_returns_cached_gmsh_metrics_after_generate():
    cm = ConceptualMesh()
    cm.add_polygon(
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        zone_id=1,
        resolution=2.0,
        dist_max=4.0,
    )
    clean_polys, clean_lines, clean_points = cm.generate()

    mesher = MeshGenerator(
        background_lc=2.0,
        verbosity=0,
        smoothing_steps=0,
        optimization_cycles=0,
    )
    assert mesher.generate(clean_polys, clean_lines, clean_points)

    quality = mesher.get_triangular_quality()

    assert list(quality.columns) == TRIANGULAR_QUALITY_COLUMNS
    assert not quality.empty
    assert quality["element_tag"].is_unique
    assert quality["is_triangle"].all()
    always_supported = quality[["minSICN", "minSJ", "minSIGE", "gamma"]]
    assert np.isfinite(always_supported.to_numpy()).all()
    optional_metrics = quality.drop(
        columns=[
            "element_tag",
            "element_type",
            "element_name",
            "is_triangle",
            *always_supported.columns,
        ]
    )
    for _, values in optional_metrics.items():
        assert values.isna().all() or np.isfinite(values.to_numpy()).all()
    assert quality["gamma"].between(0.0, 1.0).all()
    assert quality["minSICN"].between(-1.0, 1.0).all()

    quality_copy = mesher.get_triangular_quality()
    quality_copy.loc[quality_copy.index[0], "gamma"] = -999.0
    assert mesher.get_triangular_quality()["gamma"].iloc[0] != -999.0


def test_get_element_grid_returns_cached_gmsh_element_polygons_after_generate():
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        zone_id="domain",
        resolution=2.0,
        dist_max=4.0,
    )
    clean_polys, clean_lines, clean_points = cm.generate()

    mesher = MeshGenerator(
        background_lc=2.0,
        verbosity=0,
        smoothing_steps=0,
        optimization_cycles=0,
    )
    assert mesher.generate(clean_polys, clean_lines, clean_points)

    element_grid = mesher.get_element_grid()
    quality = mesher.get_triangular_quality()

    assert list(element_grid.columns) == ELEMENT_GRID_COLUMNS
    assert not element_grid.empty
    assert len(element_grid) == len(quality)
    assert element_grid["element_tag"].is_unique
    assert element_grid["element_tag"].tolist() == sorted(element_grid["element_tag"].tolist())
    assert element_grid["is_triangle"].all()
    assert not element_grid["is_quad"].any()
    assert element_grid.geometry.is_valid.all()
    assert (element_grid.geometry.area > 0).all()
    assert element_grid["zone_id"].eq("domain").all()
    assert "x" not in element_grid.columns
    assert "y" not in element_grid.columns

    triangle_grid = mesher.get_element_grid("triangles")
    quad_grid = mesher.get_element_grid("quads")

    assert len(triangle_grid) == len(element_grid)
    assert quad_grid.empty

    element_grid_copy = mesher.get_element_grid()
    element_grid_copy.loc[element_grid_copy.index[0], "element_tag"] = -999
    assert mesher.get_element_grid()["element_tag"].iloc[0] != -999


def test_element_grid_supports_centroid_connectivity_without_generator_columns():
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(
        Polygon([(0, 0), (6, 0), (6, 4), (0, 4)]),
        zone_id=1,
        resolution=2.0,
        dist_max=4.0,
    )
    clean_polys, clean_lines, clean_points = cm.generate()

    mesher = MeshGenerator(
        background_lc=2.0,
        verbosity=0,
        smoothing_steps=0,
        optimization_cycles=0,
    )
    assert mesher.generate(clean_polys, clean_lines, clean_points)

    element_grid = mesher.get_element_grid()
    connectivity = build_connectivity(element_grid, center="centroid")
    quality = calculate_mesh_quality(
        element_grid,
        calc_ortho=True,
        calc_skewness=True,
        connectivity=connectivity,
    )

    assert not connectivity.empty
    assert connectivity["center_mode"].eq("centroid").all()
    assert "ortho_error" in quality.columns
    assert "skewness" in quality.columns
    assert "drift_ratio" not in quality.columns


def test_get_element_grid_validates_filter():
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        zone_id=1,
        resolution=2.0,
    )
    clean_polys, clean_lines, clean_points = cm.generate()

    mesher = MeshGenerator(background_lc=2.0, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    assert mesher.generate(clean_polys, clean_lines, clean_points)

    with pytest.raises(ValueError, match="element_filter"):
        mesher.get_element_grid("hexes")


def test_element_grid_and_quality_exclude_field_only_surfaces():
    # embed=False polygons become standalone meshed surfaces in gmsh; their
    # elements must not leak into the element grid or quality report.
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), zone_id="domain", resolution=2.0)
    cm.add_polygon(Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), zone_id="overlay", embed=False, resolution=2.0)
    clean_polys, clean_lines, clean_points = cm.generate()

    mesher = MeshGenerator(background_lc=2.0, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    assert mesher.generate(clean_polys, clean_lines, clean_points)

    element_grid = mesher.get_element_grid()
    quality = mesher.get_triangular_quality()

    # If the overlay's standalone mesh leaked in, the total element area would
    # be ~domain + overlay (16 + 4) instead of the domain alone.
    total_area = float(element_grid.geometry.area.sum())
    assert abs(total_area - 16.0) < 1e-6
    assert len(quality) == len(element_grid)


def _fake_element_grid(centroid):
    geometry = [Polygon([(centroid[0] - 0.1, centroid[1] - 0.1),
                         (centroid[0] + 0.1, centroid[1] - 0.1),
                         (centroid[0], centroid[1] + 0.1)])]
    return gpd.GeoDataFrame(
        {
            "element_tag": [1],
            "centroid_x": [centroid[0]],
            "centroid_y": [centroid[1]],
        },
        geometry=geometry,
        crs=None,
    )


def test_zone_assignment_tie_break_is_deterministic():
    from vorflow.engine import _assign_zones_to_elements

    zone_a = box(0, 0, 1, 1)
    zone_b = box(1, 0, 2, 1)
    on_border = (1.0, 0.5)

    # With z_order, the higher z_order wins regardless of zone row order.
    for order in (["a", "b"], ["b", "a"]):
        zones = gpd.GeoDataFrame(
            {
                "zone_id": order,
                "z_order": [2 if z == "b" else 1 for z in order],
                "geometry": [zone_b if z == "b" else zone_a for z in order],
            },
            geometry="geometry",
            crs=None,
        )
        result = _assign_zones_to_elements(_fake_element_grid(on_border), zones)
        assert result.loc[0, "zone_id"] == "b"

    # Without z_order, the earliest zone row wins, reproducibly.
    zones_ab = gpd.GeoDataFrame(
        {"zone_id": ["a", "b"], "geometry": [zone_a, zone_b]},
        geometry="geometry",
        crs=None,
    )
    zones_ba = gpd.GeoDataFrame(
        {"zone_id": ["b", "a"], "geometry": [zone_b, zone_a]},
        geometry="geometry",
        crs=None,
    )
    for zones, expected in ((zones_ab, "a"), (zones_ba, "b")):
        results = {
            _assign_zones_to_elements(_fake_element_grid(on_border), zones).loc[0, "zone_id"]
            for _ in range(3)
        }
        assert results == {expected}
