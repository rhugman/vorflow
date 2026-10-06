<p align="center">
  <img src="https://raw.githubusercontent.com/rhugman/vorflow/main/docs/images/vorflow-banner-network.png"
       alt="vorflow Voronoi grid: cells refine along a stream network, around wells and inside a circular zone, with a hole cut into the zone"
       width="100%">
</p>

# vorflow

[![PyPI](https://img.shields.io/pypi/v/vorflow)](https://pypi.org/project/vorflow/)
[![Python](https://img.shields.io/pypi/pyversions/vorflow)](https://pypi.org/project/vorflow/)
[![Tests](https://github.com/rhugman/vorflow/actions/workflows/python-app.yml/badge.svg)](https://github.com/rhugman/vorflow/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rhugman/vorflow/blob/main/LICENSE)

Voronoi mesh generation for MODFLOW 6 using Gmsh and GeoPandas.

`vorflow` is a Python package for creating 2D unstructured Voronoi cell meshes for groundwater modeling, particularly for MODFLOW 6. It leverages the power of `Gmsh` for robust triangular meshing and `Shapely`/`GeoPandas` for geometric operations.

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

`vorflow` requires Python 3.10 or newer. Install it from PyPI:

```bash
pip install vorflow
```

The examples and notebooks also need Matplotlib, which the `examples` extra
installs:

```bash
pip install "vorflow[examples]"
```

On Linux, the `gmsh` wheel from PyPI needs the system GLU library (for example
`sudo apt-get install libglu1-mesa` on Debian/Ubuntu). Alternatively, install
the geospatial stack and Gmsh from conda-forge first, then `pip install vorflow`
into that environment.

To try the latest unreleased changes, install from GitHub:

```bash
pip install "git+https://github.com/rhugman/vorflow.git"
```

Release candidates are published to TestPyPI before each release. To test one
(dependencies still come from PyPI):

```bash
pip install --pre --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ vorflow
```

### Development installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/rhugman/vorflow.git
cd vorflow
pip install -e ".[dev]"
```

For plotting examples and notebooks without all development tools:

```bash
pip install -e ".[examples]"
```

Alternatively, create the Conda development environment from
[`etc/environment.yml`](https://github.com/rhugman/vorflow/blob/main/etc/environment.yml),
which installs the package in editable mode with the `dev` extra:

```bash
micromamba env create -f etc/environment.yml
```

## Basic Usage

Here is a simple example of how to generate a non-empty Voronoi grid:

The complete runnable version is
[examples/basic_usage.py](https://github.com/rhugman/vorflow/blob/main/examples/basic_usage.py).

```python
from shapely.geometry import LineString, Point, box

from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator

domain = box(0, 0, 200, 200)
well_point = Point(25, 25)
fault_line = LineString([(100, 0), (100, 150)])

blueprint = ConceptualMesh(crs="EPSG:3857")
blueprint.add_polygon(domain, zone_id=1)
blueprint.add_point(
    well_point,
    point_id="Well-A",
    resolution=2,
    growth_factor=1.2,
)
blueprint.add_line(
    fault_line,
    line_id="Fault-1",
    resolution=1,
    is_barrier=True,
)

clean_polys, clean_lines, clean_pts = blueprint.generate()

mesher = MeshGenerator(background_lc=100)
mesher.generate(clean_polys, clean_lines, clean_pts)

tessellator = VoronoiTessellator(mesher, blueprint, clip_to_boundary=True)
grid_gdf = tessellator.generate()
if grid_gdf.empty:
    raise RuntimeError("Basic Usage generated an empty Voronoi grid")
```

### Optional file export

GeoPandas writes formats such as Shapefile and GeoPackage through an I/O engine
such as Pyogrio or Fiona. Install one of those engines before calling:

```python
grid_gdf.to_file("mf6_grid.gpkg", driver="GPKG")
```

### Mesh gradation

Feature resolutions use `GeometricGrowthField` by default. Its
`growth_factor` is an upper target for neighboring characteristic edge-length
growth, not cell area growth and not an exact guarantee for every generated
neighbor pair. The default `growth_factor=1.2` uses the transparent spatial law

```text
h(d) = feature_lc + (growth_factor - 1) * d.
```

For the continuous-metric convention, pass an explicit
`GeometricGrowthField(growth_model="continuous_metric")`; this uses the gentler
gradient `log(growth_factor)`. In normal `MeshGenerator` use, the global
background field caps either result at `background_lc`.

A polygon added with `embed=False` is a refinement region: the mesh is held
at its `resolution` throughout its interior and grows away from its boundary
as for an embedded polygon, but the polygon adds no mesh edges and is not a
zone.

```python
blueprint.add_polygon(refine_area, zone_id="refine", resolution=2, embed=False)
```

> **Coordinate systems:** always work in a *projected* CRS (e.g. UTM or a
> national grid) so mesh sizes are in real length units (meters/feet).
> Geographic coordinates (lat/lon degrees, e.g. EPSG:4326) produce
> physically meaningless MODFLOW grids — reproject your data first with
> `GeoDataFrame.to_crs()`.

### Centred point cells

A point feature is always the generator of its own cell, but the cell is
usually an irregular polygon and the point is not its centroid. With
`hex_ring=True`, six fixed nodes are placed at radius `resolution` around the
point, so its cell is a regular hexagon centred on it (apothem
`resolution / 2`):

```python
blueprint.add_point(well_point, point_id="Well-A", resolution=2, hex_ring=True)
```

The ring adds about 40 cells per point (`growth_factor=1.2`). It is dropped,
with a `UserWarning`, when a polygon boundary, line or other point lies closer
than 2 x `resolution`, or when a finer size field reaches the ring (below
0.9 x `resolution`, e.g. inside a zone with a finer resolution than the
point's). After meshing, `MeshGenerator.diagnostics["hex_rings"]` records
whether each ring came out intact.

### Lloyd relaxation

`lloyd_iterations` moves the free generators (interior nodes of the zones)
towards the centroids of their cells before the grid is built. The centroids
are weighted by the local mesh size (density h^-4), so the grading is kept;
boundary, zone-edge, point and line nodes stay fixed. h comes from Gmsh's
per-node sizes, smoothed over the mesh edges first (`lloyd_size_smoothing`,
5 passes by default; 0 uses the raw sizes) because their node-to-node noise
would otherwise set the cell shapes.

```python
tessellator = VoronoiTessellator(mesher, blueprint, lloyd_iterations=100)
grid_gdf = tessellator.generate()
print(tessellator.lloyd_report)  # passes run, last residual, rejected moves
```

Lloyd is off by default. It is mainly for models that use cell centroids as
cell centres, or for visually more regular cells. Use 100 or more passes
together with `hex_ring=True` on refined points, and avoid about 20 passes:
a partly relaxed grid has the most very short faces. On a 2 km model with
four refined wells, 100 passes lower the p95 centroid-to-centroid
`ortho_error` from 4.1 to 1.4 degrees, for several times the runtime of
meshing and tessellating once. The grid is then no longer the exact dual of
`MeshGenerator.get_element_grid()`. The mesh generator must have run before
the tessellator is constructed.

Head accuracy is set mainly by `growth_factor`, not by cell shape. In a
steady radial-flow (Thiem) test
([#31](https://github.com/rhugman/vorflow/issues/31)), halving
`growth_factor - 1` roughly doubled the cell count and cut the head RMSE about
threefold, while 100 Lloyd passes improved it by at most about 5%. Lower
`growth_factor` (or the resolution) where accuracy matters.

## Examples

The [examples/](https://github.com/rhugman/vorflow/tree/main/examples)
folder contains runnable scripts and notebooks
covering field-based refinement, mesh quality diagnostics, structured quad
buffers, active-domain workflows, and triangular element-grid export.
[examples/point_centring_demo.ipynb](https://github.com/rhugman/vorflow/blob/main/examples/point_centring_demo.ipynb)
compares Gmsh smoothing, `hex_ring` and `lloyd_iterations` for centring cells
around wells.

## Roadmap

See [ROADMAP.md](https://github.com/rhugman/vorflow/blob/main/ROADMAP.md)
for planned and completed milestones.

## License

MIT — see [LICENSE](https://github.com/rhugman/vorflow/blob/main/LICENSE).
