# vorflow

Voronoi mesh generation for MODFLOW 6 using Gmsh and Geopandas.

`vorflow` is a Python package for creating 2D unstructured Voronoi cell meshes for groundwater modeling, particularly for MODFLOW 6. It leverages the power of `Gmsh` for robust triangular meshing and `Shapely`/`Geopandas` for geometric operations.

The process is designed to translate a conceptual model—defined by geometric features like polygons, lines, and points—into a high-quality Voronoi grid suitable for numerical simulation.

## Core Components

The library is built around three main classes that work in sequence:

1.  **`ConceptualMesh`**: A blueprinting tool to define the model domain and its features. You can add polygons (e.g., model boundary, refinement zones), lines (rivers, faults), and points (wells) and specify the desired mesh density and refinement behavior for each.

2.  **`MeshGenerator`**: This is the engine that generates a triangular mesh based on the blueprint from `ConceptualMesh`. It uses `Gmsh` as its backend to create a quality-conforming Delaunay triangulation.

3.  **`VoronoiTessellator`**: This class takes the triangular mesh from `MeshGenerator` and computes its dual: the Voronoi diagram. The result is a grid of polygonal cells. It includes logic to clip the grid to the domain boundary and enforce barrier features by cutting through cells.

## Workflow

The typical workflow follows these steps:

1.  **Define Geometry**: Create `shapely` objects for your model features (domain boundary, rivers, wells, etc.).
2.  **Create a Blueprint**: Instantiate `ConceptualMesh` and add your geometries, specifying parameters like mesh resolution, refinement distances, and feature types (e.g., barriers).
3.  **Generate Mesh**: Instantiate `MeshGenerator` and call its `generate()` method with the processed geometries from the blueprint. This produces a triangular mesh.
4.  **Tessellate to Voronoi**: Instantiate `VoronoiTessellator` with the generated mesh and the blueprint. Calling its `generate()` method produces the final `GeoDataFrame` of Voronoi cells.
5.  **Export**: The resulting `GeoDataFrame` can be easily saved to a shapefile or other formats.

## Installation

The package dependencies are listed in `pyproject.toml`. You can install them using pip:

```bash
pip install numpy pandas geopandas shapely scipy gmsh matplotlib
```

To install `vorflow` itself, you can install it in editable mode from the root of the repository:

```bash
pip install -e .
```

## Basic Usage

Here is a simple example of how to generate a grid:

```python
from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator
from shapely.geometry import box, Point, LineString

# 1. Define conceptual model features
domain = box(0, 0, 200, 200)
well_point = Point(25, 25)
fault_line = LineString([(100, 0), (100, 150)])

# 2. Create a blueprint
blueprint = ConceptualMesh(crs="EPSG:3857")
blueprint.add_polygon(domain, zone_id=1)
blueprint.add_point(well_point, point_id="Well-A", resolution=2, dist_max=300)
blueprint.add_line(fault_line, line_id="Fault-1", resolution=1, is_barrier=True)

clean_polys, clean_lines, clean_pts = blueprint.generate()

# 3. Generate the triangular mesh
mesher = MeshGenerator(background_lc=100)
mesher.generate(clean_polys, clean_lines, clean_pts)

# 4. Convert to Voronoi grid
tessellator = VoronoiTessellator(mesher, blueprint, clip_to_boundary=True)
grid_gdf = tessellator.generate()

# 5. Save the output
grid_gdf.to_file("mf6_grid.shp")

print("Grid generation complete.")
```
