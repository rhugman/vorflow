import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from vorflow.blueprint import ConceptualMesh
from vorflow.tessellator import VoronoiTessellator


class FakeMeshGenerator:
    def __init__(self, clean_polygons, spacing=0.35):
        domain = unary_union(clean_polygons.geometry)
        minx, miny, maxx, maxy = domain.bounds

        nx = max(3, int(np.ceil((maxx - minx) / spacing)) + 1)
        ny = max(3, int(np.ceil((maxy - miny) / spacing)) + 1)

        xs = np.linspace(0.0, 1.0, nx)
        ys = np.linspace(0.0, 1.0, ny)

        points = []
        for x_norm in xs:
            for y_norm in ys:
                x = minx + x_norm * max(0.0, maxx - minx)
                y = miny + y_norm * max(0.0, maxy - miny)
                candidate = Point(x, y)
                if domain.contains(candidate) or domain.touches(candidate):
                    points.append([x, y])

        points_np = np.array(points)
        points_np = (
            np.unique(points_np, axis=0)
            if points_np.size != 0
            else np.empty((0, 2))
        )

        if points_np.shape[0] < 3:
            center = Point((minx + maxx) / 2.0, (miny + maxy) / 2.0)
            offsets = [[-spacing, 0], [spacing, 0], [0, spacing]]
            points_np = np.array([[center.x + dx, center.y + dy] for dx, dy in offsets])

        self.nodes = points_np
        self.node_tags = list(range(len(points_np)))
        self.zones_gdf = clean_polygons


class EmptyMeshGenerator:
    def __init__(self):
        self.nodes = np.empty((0, 2))
        self.node_tags = []
        self.zones_gdf = gpd.GeoDataFrame()


def _build_simple_conceptual_mesh():
    cm = ConceptualMesh(crs="EPSG:3857")
    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    cm.add_polygon(square, zone_id=99, border_density=0.5)
    return cm


def test_full_pipeline_assigns_zones_and_covers_domain():
    cm = _build_simple_conceptual_mesh()
    clean_polys, clean_lines, clean_points = cm.generate()

    mesh_gen = FakeMeshGenerator(clean_polys)
    tessellator = VoronoiTessellator(mesh_gen, cm, clip_to_boundary=True)
    final_grid = tessellator.generate()

    assert not final_grid.empty
    assert set(final_grid["zone_id"]) == {99}
    assert set(final_grid["node_id"]) == set(mesh_gen.node_tags)
    assert "centroid_x" in final_grid.columns
    assert "centroid_y" in final_grid.columns

    domain_area = unary_union(clean_polys.geometry).area
    grid_area = unary_union(final_grid.geometry).area
    assert pytest.approx(domain_area, rel=1e-2) == grid_area


def test_pipeline_reports_empty_when_no_domain():
    cm = ConceptualMesh()
    tessellator = VoronoiTessellator(EmptyMeshGenerator(), cm, clip_to_boundary=True)
    grid = tessellator.generate()

    assert grid.empty
