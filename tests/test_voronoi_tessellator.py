import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon

from vorflow.blueprint import ConceptualMesh
from vorflow.tessellator import VoronoiTessellator


class DummyMeshGenerator:
    def __init__(self, nodes, tags, zones_gdf):
        self.nodes = nodes
        self.node_tags = tags
        self.zones_gdf = zones_gdf


def _build_conceptual_mesh():
    cm = ConceptualMesh()
    box = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    cm.add_polygon(box, zone_id=42)
    return cm


def test_voronoi_clips_to_domain_and_assigns_zones():
    cm = _build_conceptual_mesh()
    clean_polys, _, _ = cm.generate()

    nodes = np.array(
        [
            [0.2, 0.2],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.8, 0.8],
        ]
    )
    tags = [1, 2, 3, 4]

    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=clean_polys)

    tessellator = VoronoiTessellator(mesh_gen, cm, clip_to_boundary=True)
    grid = tessellator.generate()

    assert not grid.empty
    assert grid["zone_id"].nunique() == 1
    assert grid["zone_id"].iloc[0] == 42

    domain = clean_polys.iloc[0].geometry
    assert grid.geometry.apply(lambda cell: cell.intersects(domain)).all()

    assert set(grid["node_id"]) == set(tags)
    assert "centroid_x" in grid.columns
    assert "centroid_y" in grid.columns
