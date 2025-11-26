import pytest
import geopandas as gpd
from shapely.geometry import Polygon, LineString
import gmsh

from vorflow.blueprint import ConceptualMesh
from vorflow.engine import MeshGenerator
from vorflow.tessellator import VoronoiTessellator

@pytest.fixture(autouse=True)
def ensure_gmsh_finalized():
    """Ensure gmsh is finalized before and after each test to prevent state leakage."""
    if gmsh.is_initialized():
        gmsh.finalize()
    yield
    if gmsh.is_initialized():
        gmsh.finalize()

def test_gmsh_integration_simple_square():
    """
    Test the full pipeline with a simple square domain using the real Gmsh engine.
    """
    # 1. Setup Conceptual Model
    cm = ConceptualMesh(crs="EPSG:3857")
    # 10x10 square
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    # Zone ID 1, resolution 2.0 (coarse mesh for speed)
    cm.add_polygon(square, zone_id=1, resolution=2.0)
    
    clean_polys, clean_lines, clean_points = cm.generate()
    
    # 2. Generate Mesh using Gmsh
    mg = MeshGenerator(background_lc=2.0, verbosity=1)
    
    success = mg.generate(clean_polys, clean_lines, clean_points)
    assert success
    
    assert mg.nodes is not None
    assert len(mg.nodes) > 0
    
    # 3. Tessellate
    vt = VoronoiTessellator(mg, cm, clip_to_boundary=True)
    grid = vt.generate()
    
    # 4. Assertions
    assert not grid.empty
    assert "zone_id" in grid.columns
    assert grid.iloc[0]["zone_id"] == 1
    
    # Check area coverage
    total_area = grid.geometry.area.sum()
    expected_area = square.area
    # Should be very close as we clip to the exact boundary
    assert pytest.approx(total_area, rel=0.01) == expected_area

def test_gmsh_integration_with_internal_line():
    """
    Test that internal lines (constraints) are respected by the mesher.
    """
    cm = ConceptualMesh(crs="EPSG:3857")
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    cm.add_polygon(square, zone_id=1, resolution=5.0)
    
    # Diagonal line with finer resolution
    line = LineString([(1, 1), (9, 9)])
    cm.add_line(line, line_id="fault", resolution=1.0)
    
    clean_polys, clean_lines, clean_points = cm.generate()
    
    mg = MeshGenerator(background_lc=5.0, verbosity=1)
    
    success = mg.generate(clean_polys, clean_lines, clean_points)
    assert success
    
    vt = VoronoiTessellator(mg, cm, clip_to_boundary=True)
    grid = vt.generate()
    
    assert not grid.empty
    
    # We expect significantly more cells than a 5.0 resolution square would imply (approx 4 cells)
    # because of the 1.0 resolution line.
    assert len(grid) > 10
