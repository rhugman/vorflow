import warnings

import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

import vorflow.blueprint as blueprint_module
from vorflow.blueprint import ConceptualMesh
from vorflow.fields import ExponentialField, GeometricGrowthField, ThresholdField


def test_resolve_overlaps_respects_z_order():
    cm = ConceptualMesh()

    outer = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    inner = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])

    cm.add_polygon(outer, zone_id=1, z_order=0)
    cm.add_polygon(inner, zone_id=2, z_order=1)

    clean_polys, _, _ = cm.generate()

    assert len(clean_polys) == 2
    assert clean_polys.geometry.is_valid.all()

    resolved_union = unary_union(clean_polys.geometry)
    expected_union = unary_union([outer, inner])
    assert pytest.approx(expected_union.area, rel=1e-6) == resolved_union.area


def test_resolve_overlaps_keeps_first_added_polygon_for_equal_z_order():
    cm = ConceptualMesh()
    for zone_id in range(20):
        cm.add_polygon(
            Polygon([(100, 0), (101, 0), (101, 1), (100, 1)]),
            zone_id=f"background-{zone_id}",
            z_order=-1,
        )
        cm.add_polygon(
            Polygon([(zone_id, 0), (zone_id + 2, 0), (zone_id + 2, 1), (zone_id, 1)]),
            zone_id=zone_id,
            z_order=0,
        )

    clean_polys, _, _ = cm.generate()

    containing_zone = clean_polys[clean_polys.geometry.contains(Point(3.5, 0.5))]
    assert containing_zone["zone_id"].tolist() == [2]


def test_generate_preserves_registered_raw_features():
    cm = ConceptualMesh(connectivity_tolerance=0.1)
    domain = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    field_only = Polygon([(8, 8), (12, 8), (12, 12), (8, 12)])
    crossing_line = LineString([(-1, 5), (5, 5)])
    near_corner = Point(0.05, 0.05)
    outside_point = Point(20, 20)
    cm.add_polygon(domain, zone_id="domain")
    cm.add_polygon(field_only, zone_id="field", embed=False)
    cm.add_line(crossing_line, line_id="line", resolution=1, densify=False)
    cm.add_point(near_corner, point_id="near", resolution=1)
    cm.add_point(outside_point, point_id="outside", resolution=1)

    cm.generate()

    assert [feature["zone_id"] for feature in cm.raw_polygons] == ["domain", "field"]
    assert cm.raw_polygons[0]["geometry"].equals(domain)
    assert cm.raw_polygons[1]["geometry"].equals(field_only)
    assert cm.raw_lines[0]["geometry"].equals(crossing_line)
    assert [feature["point_id"] for feature in cm.raw_points] == ["near", "outside"]
    assert cm.raw_points[0]["geometry"].equals(near_corner)
    assert cm.raw_points[1]["geometry"].equals(outside_point)


def test_generate_repeated_calls_return_identical_clean_frames_with_field_only_polygon():
    cm = ConceptualMesh(connectivity_tolerance=0.1)
    field = GeometricGrowthField(growth_factor=1.2)
    cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id="domain")
    cm.add_polygon(
        Polygon([(8, 8), (12, 8), (12, 12), (8, 12)]),
        zone_id="field",
        fields=[field],
        embed=False,
    )
    cm.add_line(
        LineString([(-1, 5), (5, 5)]),
        line_id="line",
        resolution=1,
        densify=False,
    )
    cm.add_point(Point(20, 20), point_id="outside", resolution=1)

    first = tuple(frame.copy(deep=True) for frame in cm.generate())
    second = cm.generate()

    assert first[0]["zone_id"].tolist() == ["domain", "field"]
    assert first[0].iloc[1]["fields"] == [field]
    for first_frame, second_frame in zip(first, second):
        assert_geodataframe_equal(first_frame, second_frame)


def test_generate_preserves_raw_inputs_when_processing_fails():
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id="domain")
    cm.add_polygon(Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]), zone_id="field", embed=False)
    original_line = LineString([(1, 1), (2, 1.01), (3, 1)])
    cm.add_line(
        original_line,
        line_id="noisy",
        resolution=1,
        simplify_tolerance=0.1,
    )

    with pytest.raises(ValueError, match="connectivity_tolerance"):
        cm.generate(connectivity_tolerance=True)

    assert [feature["zone_id"] for feature in cm.raw_polygons] == ["domain", "field"]
    assert cm.raw_lines[0]["geometry"].equals(original_line)


def test_growth_factor_must_exceed_one():
    cm = ConceptualMesh()
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(ValueError, match="growth_factor"):
        cm.add_polygon(square, zone_id=1, resolution=0.5, growth_factor=1.0)
    with pytest.raises(ValueError, match="growth_factor"):
        cm.add_line(LineString([(0, 0), (1, 0)]), line_id="l", resolution=0.5, growth_factor=0.9)
    with pytest.raises(ValueError, match="growth_factor"):
        cm.add_point(Point(0, 0), point_id="p", resolution=0.5, growth_factor=True)


@pytest.mark.parametrize("growth_factor", [float("nan"), float("inf"), -float("inf")])
def test_growth_factor_must_be_finite(growth_factor):
    cm = ConceptualMesh(crs=None)
    with pytest.raises(ValueError, match="finite"):
        cm.add_point(
            Point(0, 0),
            point_id="p",
            resolution=0.5,
            growth_factor=growth_factor,
        )


def test_growth_factor_defaults_to_none_and_is_stored():
    cm = ConceptualMesh()
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    cm.add_polygon(square, zone_id=1, resolution=0.5)
    cm.add_polygon(square, zone_id=2, resolution=0.5, growth_factor=1.3)
    assert cm.raw_polygons[0]["growth_factor"] is None
    assert cm.raw_polygons[1]["growth_factor"] == 1.3


def test_lines_and_points_snap_to_polygons():
    cm = ConceptualMesh(connectivity_tolerance=1.0)

    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    cm.add_polygon(square, zone_id=1)

    line = LineString([(-0.5, 1.0), (0.5, 1.0)])
    point = Point(-0.0005, 0.0)

    cm.add_line(line, line_id="river", resolution=0.1)
    cm.add_point(point, point_id="well", resolution=0.1)

    cm.generate()

    assert not cm.clean_lines.empty
    assert not cm.clean_points.empty

    boundary = cm.clean_polygons.iloc[0].geometry.boundary
    snapped_line = cm.clean_lines.iloc[0].geometry
    snapped_point = cm.clean_points.iloc[0].geometry

    tolerance = 1e-3
    assert snapped_line.distance(boundary) <= tolerance
    assert snapped_point.distance(boundary) <= tolerance


def test_line_snapping_defaults_to_enabled_and_can_be_disabled():
    cm = ConceptualMesh(connectivity_tolerance=0.25)
    cm.add_polygon(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), zone_id=1)
    cm.add_line(LineString([(0.1, 0.1), (1, 1)]), line_id="default", resolution=0.1, densify=False)
    cm.add_line(
        LineString([(0.1, 1.9), (1, 1.5)]),
        line_id="disabled",
        resolution=0.1,
        snap_to_polygons=False,
        densify=False,
    )

    _, clean_lines, _ = cm.generate()

    default_line = clean_lines.loc[clean_lines["line_id"] == "default", "geometry"].iloc[0]
    disabled_line = clean_lines.loc[clean_lines["line_id"] == "disabled", "geometry"].iloc[0]
    assert default_line.coords[0] == (0.0, 0.0)
    assert disabled_line.coords[0] == (0.1, 1.9)


def test_points_outside_domain_are_removed_after_connectivity():
    cm = ConceptualMesh(connectivity_tolerance=0.01)

    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    cm.add_polygon(square, zone_id=1)
    cm.add_line(LineString([(0, 1), (2, 1)]), line_id="river", resolution=0.5, densify=False)
    cm.add_point(Point(-0.05, 1), point_id="outside_well", resolution=0.05)

    _, _, clean_points = cm.generate()

    assert clean_points.empty


def test_points_snapped_to_domain_boundary_are_kept():
    cm = ConceptualMesh(connectivity_tolerance=0.1)

    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    cm.add_polygon(square, zone_id=1)
    cm.add_line(LineString([(0, 1), (2, 1)]), line_id="river", resolution=0.5, densify=False)
    cm.add_point(Point(-0.05, 1), point_id="snapped_well", resolution=0.05)

    _, _, clean_points = cm.generate()

    assert len(clean_points) == 1
    assert clean_points.iloc[0].geometry.equals(Point(0, 1))


def test_lines_are_clipped_to_domain_after_connectivity():
    cm = ConceptualMesh(connectivity_tolerance=0.0)

    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    cm.add_polygon(square, zone_id=1)
    cm.add_line(LineString([(-1, 1), (3, 1)]), line_id="crossing_line", resolution=0.5, densify=False)

    _, clean_lines, _ = cm.generate()

    assert len(clean_lines) == 1
    assert clean_lines.iloc[0].geometry.equals(LineString([(0, 1), (2, 1)]))


def test_generate_uses_constructor_connectivity_tolerance(monkeypatch):
    cm = ConceptualMesh(connectivity_tolerance=0.25)
    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

    cm.add_polygon(square, zone_id=1)
    cm.add_line(LineString([(0, 0), (1, 0)]), line_id="river", resolution=0.1, densify=False)
    cm.add_point(Point(0.1, 0.1), point_id="well", resolution=0.1)

    recorded_tolerances = []

    def fake_snap(geometry, reference_geometry, tolerance):
        recorded_tolerances.append(tolerance)
        return geometry

    monkeypatch.setattr(blueprint_module, "snap", fake_snap)

    cm.generate()

    assert recorded_tolerances == [pytest.approx(0.25), pytest.approx(0.25)]


def test_generate_can_override_connectivity_tolerance(monkeypatch):
    cm = ConceptualMesh(connectivity_tolerance=1.0)
    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

    cm.add_polygon(square, zone_id=1)
    cm.add_line(LineString([(0, 0), (1, 0)]), line_id="river", resolution=0.1, densify=False)
    cm.add_point(Point(0.1, 0.1), point_id="well", resolution=0.1)

    recorded_tolerances = []

    def fake_snap(geometry, reference_geometry, tolerance):
        recorded_tolerances.append(tolerance)
        return geometry

    monkeypatch.setattr(blueprint_module, "snap", fake_snap)

    cm.generate(connectivity_tolerance=0.05)

    assert recorded_tolerances == [pytest.approx(0.05), pytest.approx(0.05)]

def test_polygon_simplification():
    """Test that polygons are simplified when tolerance is provided."""
    cm = ConceptualMesh()
    
    # Create a "noisy" square with a tiny bump on the top edge
    # (0,1) -> (0.5, 1.001) -> (1,1)
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0.5, 1.001), (0, 1)])
    
    # Add with a tolerance larger than the noise (0.001)
    cm.add_polygon(poly, zone_id=1, simplify_tolerance=0.01)
    
    clean_polys, _, _ = cm.generate()
    
    simplified_geom = clean_polys.iloc[0].geometry
    
    # The original polygon has 5 vertices + closing = 6 points in exterior ring
    # The simplified one should remove the bump, leaving 4 corners + closing = 5 points
    assert len(simplified_geom.exterior.coords) == 5
    assert len(simplified_geom.exterior.coords) < len(poly.exterior.coords)

def test_line_simplification():
    """Test that lines are simplified when tolerance is provided."""
    cm = ConceptualMesh()
    # Noisy line: straight but with a midpoint slightly off
    line = LineString([(0, 0), (0.5, 0.001), (1, 0)])
    
    cm.add_line(line, line_id="noisy_line", resolution=0.1, simplify_tolerance=0.01, densify=False)
    
    _, clean_lines, _ = cm.generate()
    
    simplified_line = clean_lines.iloc[0].geometry
    # Should be simplified to just start and end points
    assert len(simplified_line.coords) == 2

def test_point_deduplication():
    """Test that close points are merged and the finest resolution is kept."""
    cm = ConceptualMesh()
    p1 = Point(0, 0)
    p2 = Point(0.0001, 0) # Very close to p1
    
    # Case 1: No simplification (default) -> Should keep both
    cm.add_point(p1, "p1", resolution=1.0)
    cm.add_point(p2, "p2", resolution=0.5) 
    
    _, _, clean_points = cm.generate()
    assert len(clean_points) == 2
    
    # Case 2: With simplification -> Should merge
    cm2 = ConceptualMesh()
    # p2 has finer resolution (0.5), so it should be the one kept
    cm2.add_point(p1, "p1", resolution=1.0, simplify_tolerance=0.01)
    cm2.add_point(p2, "p2", resolution=0.5, simplify_tolerance=0.01)
    
    _, _, clean_points_merged = cm2.generate()
    
    assert len(clean_points_merged) == 1
    
    # Verify we kept the point with the finer resolution (0.5)
    kept_point = clean_points_merged.iloc[0]
    assert kept_point['lc'] == 0.5
    assert kept_point['point_id'] == "p2"

def _wavy_neighbours():
    """Two polygons sharing a wavy edge that simplification would flatten."""
    xs = np.linspace(0, 10, 101)
    wave = [(x, 5 + 0.3 * np.sin(3 * x)) for x in xs]
    lower = Polygon([(0, 0), (10, 0)] + wave[::-1])
    upper = Polygon(wave + [(10, 10), (0, 10)])
    return lower, upper


@pytest.mark.parametrize("upper_tol", [None, 0.5])
def test_polygon_simplification_keeps_shared_edges_gap_free(upper_tol):
    lower, upper = _wavy_neighbours()
    cm = ConceptualMesh()
    cm.add_polygon(lower, zone_id=1, resolution=1, simplify_tolerance=0.5)
    cm.add_polygon(upper, zone_id=2, resolution=1, z_order=1, simplify_tolerance=upper_tol)
    clean_polys, _, _ = cm.generate()

    domain = unary_union([lower, upper])
    covered = unary_union(list(clean_polys.geometry))
    assert domain.difference(covered).area < 1e-9


def test_polygon_simplification_still_simplifies_free_edges():
    # The wavy top edge is shared with nobody, so it is simplified; the
    # straight shared bottom edge is kept.
    xs = np.linspace(0, 10, 101)
    wavy_top = [(x, 10 + 0.01 * np.sin(3 * x)) for x in xs]
    top = Polygon([(0, 5), (10, 5)] + wavy_top[::-1])
    bottom = Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])
    cm = ConceptualMesh()
    cm.add_polygon(bottom, zone_id=1)
    cm.add_polygon(top, zone_id=2, simplify_tolerance=0.1)
    clean_polys, _, _ = cm.generate()

    simplified = clean_polys.loc[clean_polys["zone_id"] == 2].geometry.iloc[0]
    assert len(simplified.exterior.coords) < 10
    assert unary_union([top, bottom]).difference(unary_union(list(clean_polys.geometry))).area < 0.2


@pytest.mark.parametrize("coarse_tol, fine_tol", [(0.01, None), (None, 0.01), (0.01, 0.01)])
def test_point_deduplication_does_not_depend_on_which_point_has_the_tolerance(coarse_tol, fine_tol):
    cm = ConceptualMesh()
    cm.add_point(Point(0, 0), "coarse", resolution=1.0, simplify_tolerance=coarse_tol)
    cm.add_point(Point(0.0001, 0), "fine", resolution=0.5, simplify_tolerance=fine_tol)
    _, _, clean_points = cm.generate()
    assert clean_points["point_id"].tolist() == ["fine"]


def test_point_order_is_preserved_without_deduplication():
    cm = ConceptualMesh()
    for point_id, x, lc in [("coarse", 0, 5.0), ("fine", 10, 1.0), ("unset", 20, None)]:
        cm.add_point(Point(x, 0), point_id, resolution=lc)
    _, _, clean_points = cm.generate()
    assert clean_points["point_id"].tolist() == ["coarse", "fine", "unset"]


def test_line_densification_options():
    """Test the three modes of line densification: False, True, and float."""
    cm = ConceptualMesh()
    # A line of length 10
    line = LineString([(0, 0), (10, 0)])
    
    # 1. densify=False: Should NOT add vertices
    cm.add_line(line, "no_densify", resolution=1.0, densify=False)
    
    # 2. densify=True (default): Should use resolution (1.0) -> ~10 segments
    cm.add_line(line, "default_densify", resolution=1.0, densify=True)
    
    # 3. densify=5.0: Should use custom spacing (5.0) -> ~2 segments
    cm.add_line(line, "custom_densify", resolution=1.0, densify=5.0)
    
    _, clean_lines, _ = cm.generate()
    
    # Check 1: No densification
    l1 = clean_lines[clean_lines['line_id'] == "no_densify"].iloc[0].geometry
    assert len(l1.coords) == 2 # Just start and end
    
    # Check 2: Default densification (lc=1.0)
    l2 = clean_lines[clean_lines['line_id'] == "default_densify"].iloc[0].geometry
    # Should have 11 points (10 segments)
    assert len(l2.coords) == 11 
    
    # Check 3: Custom densification (val=5.0)
    l3 = clean_lines[clean_lines['line_id'] == "custom_densify"].iloc[0].geometry
    # Should have roughly 3 points (2 segments)
    assert len(l3.coords) == 3

@pytest.mark.parametrize("bool_tol", [True, False])
def test_simplify_tolerance_bool_is_rejected(bool_tol):
    cm = ConceptualMesh()

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(ValueError):
        cm.add_polygon(poly, zone_id=1, simplify_tolerance=bool_tol)

    line = LineString([(0, 0), (1, 0)])
    with pytest.raises(ValueError):
        cm.add_line(line, line_id="l1", resolution=0.1, simplify_tolerance=bool_tol, densify=False)

    pt = Point(0, 0)
    with pytest.raises(ValueError):
        cm.add_point(pt, point_id="p1", resolution=0.1, simplify_tolerance=bool_tol)


@pytest.mark.parametrize("bad_tolerance", [True, False, -1])
def test_connectivity_tolerance_validation(bad_tolerance):
    with pytest.raises(ValueError):
        ConceptualMesh(connectivity_tolerance=bad_tolerance)


class TestCrsHandling:
    def test_geographic_crs_warns(self):
        with pytest.warns(UserWarning, match="geographic"):
            ConceptualMesh(crs="EPSG:4326")

    def test_projected_crs_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ConceptualMesh(crs="EPSG:32618")

    def test_default_crs_is_none_and_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cm = ConceptualMesh()
        assert cm.crs is None

    def test_none_crs_propagates_to_outputs(self):
        cm = ConceptualMesh()
        cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id=1)
        clean_polys, _, _ = cm.generate()
        assert clean_polys.crs is None

    def test_explicit_projected_crs_propagates_to_outputs(self):
        cm = ConceptualMesh(crs="EPSG:32618")
        cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id=1)
        clean_polys, _, _ = cm.generate()
        assert clean_polys.crs is not None
        assert clean_polys.crs.to_epsg() == 32618


@pytest.mark.parametrize("method", ["add_line", "add_polygon"])
def test_quad_buffer_requires_embedded_feature(method):
    cm = ConceptualMesh()
    geometry = Polygon([(0, 0), (4, 0), (4, 4)]) if method == "add_polygon" else LineString([(0, 0), (4, 0)])
    identifier = {"zone_id": 1} if method == "add_polygon" else {"line_id": "l"}
    with pytest.raises(ValueError, match="quad_buffer=True requires embed=True"):
        getattr(cm, method)(geometry, resolution=1.0, quad_buffer=True, embed=False, **identifier)


# --- hex_ring ---------------------------------------------------------------

def _hex_ring_mesh():
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]), zone_id="domain")
    return cm


@pytest.mark.parametrize("value", [1, "yes", None])
def test_hex_ring_rejects_non_bool(value):
    cm = _hex_ring_mesh()
    with pytest.raises(ValueError, match="hex_ring"):
        cm.add_point(Point(50, 50), point_id="well", resolution=2, hex_ring=value)


def test_hex_ring_requires_embed_and_resolution():
    cm = _hex_ring_mesh()
    with pytest.raises(ValueError, match="embed=True"):
        cm.add_point(Point(50, 50), point_id="well", resolution=2, embed=False, hex_ring=True)
    with pytest.raises(ValueError, match="positive resolution"):
        cm.add_point(Point(50, 50), point_id="well", resolution=None, hex_ring=True)


def test_hex_ring_seeds_form_regular_hexagon():
    cm = _hex_ring_mesh()
    cm.add_point(Point(50.3, 49.7), point_id="well", resolution=2, hex_ring=True)
    _, _, clean_points = cm.generate()

    seeds = clean_points.iloc[0]["ring_seeds"]
    assert isinstance(seeds, list) and len(seeds) == 6
    offsets = np.array(seeds) - np.array([50.3, 49.7])
    np.testing.assert_allclose(np.hypot(offsets[:, 0], offsets[:, 1]), 2.0)
    angles = np.degrees(np.arctan2(offsets[:, 1], offsets[:, 0])) % 360
    np.testing.assert_allclose(angles, [30, 90, 150, 210, 270, 330])


def test_hex_ring_false_has_no_seeds():
    cm = _hex_ring_mesh()
    cm.add_point(Point(50, 50), point_id="well", resolution=2)
    _, _, clean_points = cm.generate()
    assert not clean_points.iloc[0]["hex_ring"]
    assert clean_points.iloc[0]["ring_seeds"] is None


@pytest.mark.parametrize(
    "add_obstacle",
    [
        lambda cm: cm.add_line(LineString([(53, 0), (53, 100)]), line_id="river", resolution=2),
        lambda cm: cm.add_point(Point(52, 50), point_id="other", resolution=2),
        lambda cm: cm.add_line(
            LineString([(58, 0), (58, 100)]), line_id="strip", resolution=10, quad_buffer=True
        ),
    ],
    ids=["line", "point", "quad-buffer band"],
)
def test_hex_ring_dropped_near_other_feature(add_obstacle):
    cm = _hex_ring_mesh()
    cm.add_point(Point(50, 50), point_id="well", resolution=2, hex_ring=True)
    add_obstacle(cm)
    with pytest.warns(UserWarning, match="'well'.*hex_ring ignored"):
        _, _, clean_points = cm.generate()
    well = clean_points[clean_points["point_id"] == "well"].iloc[0]
    assert well["ring_seeds"] is None


def test_hex_ring_dropped_near_domain_boundary():
    cm = _hex_ring_mesh()
    cm.add_point(Point(3, 50), point_id="edge_well", resolution=2, hex_ring=True)
    with pytest.warns(UserWarning, match="'edge_well'.*polygon 'domain' boundary"):
        _, _, clean_points = cm.generate()
    assert clean_points.iloc[0]["ring_seeds"] is None


def test_hex_ring_ignores_field_only_line():
    cm = _hex_ring_mesh()
    cm.add_point(Point(50, 50), point_id="well", resolution=2, hex_ring=True)
    cm.add_line(LineString([(51, 0), (51, 100)]), line_id="field", resolution=2, embed=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, clean_points = cm.generate()
    assert len(clean_points.iloc[0]["ring_seeds"]) == 6


def test_empty_points_frame_has_hex_ring_columns():
    _, _, clean_points = _hex_ring_mesh().generate()
    assert clean_points.empty
    assert {"hex_ring", "ring_seeds"} <= set(clean_points.columns)


def test_hex_ring_seeds_do_not_leak_into_raw_points():
    cm = _hex_ring_mesh()
    cm.add_point(Point(50, 50), point_id="well", resolution=2, hex_ring=True)
    cm.generate()
    assert "ring_seeds" not in cm.raw_points[0]
    assert cm.raw_points[0]["hex_ring"] is True


def _ring_point_in(polygon_resolution=None):
    """A 100 x 100 domain at ``polygon_resolution`` with an r=4 hex_ring well at (50, 50)."""
    cm = ConceptualMesh()
    cm.add_polygon(
        Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
        zone_id="domain",
        resolution=polygon_resolution,
    )
    cm.add_point(Point(50, 50), point_id="well", resolution=4, hex_ring=True)
    return cm


@pytest.mark.parametrize(
    "add_source, label",
    [
        (lambda cm: None, "polygon 'domain'"),
        (
            lambda cm: cm.add_polygon(
                Polygon([(45, 0), (55, 0), (55, 100), (45, 100)]),
                zone_id="field", resolution=1, embed=False,
            ),
            "polygon 'field'",
        ),
        (
            lambda cm: cm.add_line(LineString([(60, 0), (60, 100)]), line_id="fine", resolution=0.5),
            "line 'fine'",
        ),
        (
            lambda cm: cm.add_line(
                LineString([(60, 0), (60, 100)]), line_id="ramp", resolution=0.5,
                fields=[ThresholdField(size_min=0.5, dist_min=0, dist_max=50, size_max=20)],
            ),
            "line 'ramp'",
        ),
    ],
    ids=["finer enclosing zone", "finer field-only polygon", "fine line beyond clearance",
         "explicit threshold field"],
)
def test_hex_ring_dropped_when_size_field_finer_than_ring(add_source, label):
    cm = _ring_point_in(polygon_resolution=1.0 if "domain" in label else None)
    add_source(cm)
    with pytest.warns(UserWarning, match=f"'well': {label} sets a mesh size.*hex_ring ignored"):
        _, _, clean_points = cm.generate()
    assert clean_points.iloc[0]["ring_seeds"] is None


def test_hex_ring_kept_when_size_fields_are_coarse_enough():
    cm = _ring_point_in(polygon_resolution=10)
    # A fine line 40 away grows to 0.5 + 0.2 * 36.5 ~ 7.8 >= 0.9 * 4 at the ring.
    cm.add_line(LineString([(90, 0), (90, 100)]), line_id="far", resolution=0.5)
    # Unmodelled explicit fields are left to the engine's post-mesh check.
    cm.add_point(Point(50, 60), point_id="probe", resolution=1, embed=False,
                 fields=[ExponentialField(size_min=0.5, decay_length=50)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, clean_points = cm.generate()
    well = clean_points[clean_points["point_id"] == "well"].iloc[0]
    assert len(well["ring_seeds"]) == 6


def test_hex_ring_point_merged_by_simplification_warns():
    cm = _hex_ring_mesh()
    cm.add_point(Point(50, 50), point_id="well", resolution=2, hex_ring=True, simplify_tolerance=1)
    cm.add_point(Point(50.5, 50), point_id="finer", resolution=1)
    with pytest.warns(UserWarning, match="'well' was merged into point 'finer'.*hex_ring is dropped"):
        _, _, clean_points = cm.generate()
    assert list(clean_points["point_id"]) == ["finer"]


def test_hex_ring_dropped_inside_finer_field_only_polygon():
    # The engine holds a field-only polygon at its resolution throughout its
    # interior, so a ring 16 from the polygon edge still sees a size of 1.
    cm = _ring_point_in()
    cm.add_polygon(
        Polygon([(30, 30), (70, 30), (70, 70), (30, 70)]),
        zone_id="field", resolution=1, embed=False,
    )
    with pytest.warns(UserWarning, match="'well': polygon 'field' sets a mesh size of ~1 .*hex_ring ignored"):
        _, _, clean_points = cm.generate()
    assert clean_points.iloc[0]["ring_seeds"] is None


def test_hex_ring_kept_inside_field_only_polygon_at_ring_resolution():
    cm = _ring_point_in()
    cm.add_polygon(
        Polygon([(30, 30), (70, 30), (70, 70), (30, 70)]),
        zone_id="field", resolution=4, embed=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, clean_points = cm.generate()
    well = clean_points[clean_points["point_id"] == "well"].iloc[0]
    assert len(well["ring_seeds"]) == 6


def test_raise_error_features_export_before_generate():
    cm = ConceptualMesh()

    with pytest.raises(ValueError, match=r"Run ``\.generate\(\)`` before exporting features"):
            cm.features()


def test_features_export():
    cm = ConceptualMesh()
    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    cm.add_polygon(square, zone_id=1)

    line = LineString([(-0.5, 1.0), (0.5, 1.0)])
    point = Point(-0.0005, 0.0)

    cm.add_line(line, line_id="river", resolution=0.1)
    cm.add_point(point, point_id="well", resolution=0.1)
    cm.add_point(point, point_id="well-false", resolution=0.1, embed=False)

    cm.generate()

    feat = cm.features()

    assert len(feat) == 4
    assert {'feature_type', 'feature_id'}.issubset(feat.columns)
    assert {'line', 'polygon', 'point'}.issubset(feat['feature_type'])
    assert {1, 'river', 'well'}.issubset(feat['feature_id'])
    assert not feat[feat['feature_id']=="well-false"]["embed"].values
    assert sum(feat["embed"]) == 3