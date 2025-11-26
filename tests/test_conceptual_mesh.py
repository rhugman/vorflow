import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from vorflow.blueprint import ConceptualMesh


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


def test_lines_and_points_snap_to_polygons():
    cm = ConceptualMesh()

    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    cm.add_polygon(square, zone_id=1)

    line = LineString([(-0.5, 1.0), (0.5, 1.0)])
    point = Point(-0.0005, 0.5)

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
