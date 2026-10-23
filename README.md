<p align="center">
  <img src="https://raw.githubusercontent.com/rhugman/vorflow/main/docs/images/vorflow-banner-network.png"
       alt="vorflow Voronoi grid: cells refine along a stream network, around wells and inside a circular zone, with a hole cut into the zone"
       width="100%">
</p>

# vorflow

[![PyPI](https://img.shields.io/pypi/v/vorflow)](https://pypi.org/project/vorflow/)
[![Python](https://img.shields.io/pypi/pyversions/vorflow)](https://pypi.org/project/vorflow/)
[![Tests](https://github.com/rhugman/vorflow/actions/workflows/python-app.yml/badge.svg)](https://github.com/rhugman/vorflow/actions/workflows/python-app.yml)
[![Docs](https://github.com/rhugman/vorflow/actions/workflows/docs.yml/badge.svg)](https://rhugman.github.io/vorflow/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rhugman/vorflow/blob/main/LICENSE)

Voronoi mesh generation for MODFLOW 6 using Gmsh and GeoPandas.

Documentation, including the API reference and executed example notebooks:
<https://rhugman.github.io/vorflow/>

`vorflow` is a Python package for creating 2D unstructured Voronoi cell meshes for groundwater modeling, particularly for MODFLOW 6. It leverages the power of `Gmsh` for robust triangular meshing and `Shapely`/`GeoPandas` for geometric operations.

The process is designed to translate a conceptual model—defined by geometric features like polygons, lines, and points—into a high-quality Voronoi grid suitable for numerical simulation.

<!-- docs:overview-start -->

## Core Components

The library is built around three main classes that work in sequence:

1.  **`ConceptualMesh`**: A blueprinting tool to define the model domain and its features. You can add polygons (e.g., model boundary, refinement zones), lines (rivers, faults), and points (wells) and specify the desired mesh density and refinement behavior for each.

2.  **`MeshGenerator`**: This is the engine that generates a triangular mesh based on the blueprint from `ConceptualMesh`. It uses `Gmsh` as its backend to create a quality-conforming Delaunay triangulation.

3.  **`VoronoiTessellator`**: This class takes the triangular mesh from `MeshGenerator` and computes its dual: the Voronoi diagram. The result is a grid of polygonal cells. It includes logic to clip the grid to the domain boundary and enforce barrier features by cutting through cells.

## Workflow

The typical workflow follows these steps:

1.  **Define Geometry**: Create `shapely` objects for your model features (domain boundary, rivers, wells, etc.).
2.  **Create a Blueprint**: Instantiate `ConceptualMesh` and add your geometries, specifying parameters like mesh resolution, refinement distances, and feature types (e.g., barriers).
3.  **Generate Mesh**: Instantiate `MeshGenerator` with the blueprint (`conceptual_mesh=blueprint`) and call its `generate()` method. It cleans the blueprint's features and produces a triangular mesh.
4.  **Tessellate to Voronoi**: Instantiate `VoronoiTessellator` with the mesh generator. Calling its `generate()` method produces the final `GeoDataFrame` of Voronoi cells, also kept as `tessellator.grid`.
5.  **Export**: The resulting `GeoDataFrame` can be easily saved to a shapefile or other formats.

<!-- docs:installation-start -->

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

The `plot()` methods need Matplotlib, from the `plot` extra; the `examples`
extra installs it for the example scripts and notebooks:

```bash
pip install -e ".[plot]"
```

Alternatively, create the Conda development environment from
[`etc/environment.yml`](https://github.com/rhugman/vorflow/blob/main/etc/environment.yml),
which installs the package in editable mode with the `dev` extra:

```bash
micromamba env create -f etc/environment.yml
```

<!-- docs:usage-start -->

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

mesher = MeshGenerator(background_lc=100, conceptual_mesh=blueprint)
mesher.generate()

tessellator = VoronoiTessellator(mesher, clip_to_boundary=True)
grid_gdf = tessellator.generate()
if grid_gdf.empty:
    raise RuntimeError("Basic Usage generated an empty Voronoi grid")
```

`mesher.generate()` runs `blueprint.generate()` itself. To inspect or edit the
cleaned features before meshing, call it yourself and pass the three
GeoDataFrames on; keep their index, which links refinement fields back to
their features:

```python
clean_polys, clean_lines, clean_pts = blueprint.generate()
mesher = MeshGenerator(background_lc=100)
mesher.generate(clean_polys, clean_lines, clean_pts)
tessellator = VoronoiTessellator(mesher, blueprint)
```

### Plotting

Each stage has a `plot()` method (Matplotlib, from the `plot` extra). Each
takes an optional `ax` and returns the axes, so the stages can be overlaid:

```python
blueprint.plot()                    # registered features; plot(clean=True) after generate()
mesher.plot()                       # triangular mesh; column="zone_id" to colour it
tessellator.plot(column="zone_id")  # Voronoi cells; show_generators=True adds x/y
```

Keyword arguments go to `GeoDataFrame.plot`, and the returned axes take
further layers:

```python
import matplotlib.pyplot as plt

ax = tessellator.plot(column=None, facecolor="whitesmoke", edgecolor="0.4")
ax.plot(*fault_line.xy, color="tab:red", label="Fault-1 (barrier)")
ax.plot(well_point.x, well_point.y, "o", color="tab:blue", label="Well-A")
ax.legend()
plt.show()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/rhugman/vorflow/main/docs/images/basic_usage.png"
       alt="Basic Usage grid: cells refine towards the fault barrier at x = 100 and around the well at (25, 25)"
       width="60%">
</p>

### MODFLOW 6 export

`to_disv()` returns the grid as MODFLOW 6 DISV arrays (`ncpl`, `nvert`,
`vertices`, `cell2d`), zero-based as flopy takes them. Cell centres are the
generator points, vertices are listed clockwise and shared between neighbours,
and `node_id` maps each `icell2d` back to its grid row:

```python
disv = tessellator.to_disv()
```

With flopy installed, `to_modflowgwfdisv()` adds the DISV package to a model
directly; further keywords go to `flopy.mf6.ModflowGwfdisv`:

```python
import flopy

sim = flopy.mf6.MFSimulation(sim_ws="model")
gwf = flopy.mf6.ModflowGwf(sim, modelname="vorflow")
tessellator.to_modflowgwfdisv(gwf, top=10.0, botm=[0.0, -10.0], nlay=2)
```

### MODFLOW-USG DISU and GSF export

MODFLOW-USG and MODFLOW 6 both have a package called **DISU**, but their
input formats differ. `to_mfusg_disu()` explicitly targets **MODFLOW-USG**;
its arrays must not be passed to `flopy.mf6.ModflowGwfdisu`.

```python
grid = tessellator.to_mfusg_disu(top=10.0, botm=[0.0, -10.0], nlay=2)
```

The result contains `MfUsgDisU` grid inputs, including zero-based `ja`,
`iac`, `cl12`, `fahl`, `ivsd=-1` and `idsymrd=0`. `fahl` is interface
**area**, including thickness for horizontal connections; MF6's `hwva`
uses horizontal **width** instead. The flags for vertical connections also
differ between engines. Geometry helpers can be shared; solver inputs cannot
be interchanged.

As with DISV, final `x`/`y` centres and grid row order are preserved. Layers
repeat the same horizontal mesh, with nodes numbered by layer then cell.
`node_id`, `cell_map` and `connectivity` are metadata, **not** FloPy package
arguments: `cell_map` relates each zero-based model node to its layer, input
row and original index/IDs; `connectivity` reports face lengths, distances
and orthogonality errors. Review these diagnostics before modelling; the
exporter does not generate ghost-node corrections.

With FloPy installed (`pip install "vorflow[examples]"`), the package wrapper
excludes metadata automatically:

```python
import flopy

model = flopy.mfusg.MfUsg(modelname="model", model_ws="model", structured=False)
disu = tessellator.to_mfusgdisu(model, top=10.0, botm=[0.0, -10.0], nlay=2)
disu.write_file()  # Or model.write_input() after adding the other packages.
tessellator.to_gsf("model/model.gsf", top=10.0, botm=[0.0, -10.0], nlay=2)
```

Use the same elevations for both exports. GSF stores original 3D cell
geometry with one-based IDs and XYZ vertices; it is not a solver input or
the particle-tracking-specific GSF variant. FloPy can read it using
`UnstructuredGrid.from_gridspec()`. The DISU/GSF pair is intended for external
mesh import into Groundwater Vistas, but has not been manually verified in
Vistas. GSF does not retain CRS metadata, so keep your projected CRS separately.

`top` accepts a scalar or `(ncpl,)`; `botm` accepts a scalar, `(nlay,)`, or
`(nlay, ncpl)`. All cells must have finite, positive thickness. Pinch-outs,
pass-through cells, confining beds and different meshes per layer are not
supported. Grid arrays cannot be overridden through the FloPy wrapper;
time settings and units can be passed as extra keywords. Barrier geometry
does not assign hydraulic resistance automatically.

DISU and GSF currently require contiguous, straight shared faces. A barrier
that changes direction inside a cell can create a bent face and prevent both
exports; disconnected shared faces are also rejected. Effective connection
distances for bent faces are not yet implemented. Centre and overlap checks
allow coordinate roundoff while preserving the supplied centres and polygons.

For an existing GeoDataFrame, use `vorflow.utils.grid_to_mfusg_disu()` and
`vorflow.utils.grid_to_gsf()`. Plain arrays and GSF output do not need FloPy.
The [simple MFUSG notebook](examples/basic_example_mfusg.ipynb) runs a
single confined layer between two fixed heads, plots the result and checks
analytical heads and the 500 m³/day flow rate as well as budget balance. From
the repository root, install `python -m pip install -e ".[examples]"` in the
notebook kernel's environment. The example
requires the separate `mfusg` executable on PATH or an `MFUSG_EXE` path;
install it outside the notebook with `get-modflow /path/to/bin --subset mfusg`.

### Optional file export

GeoPandas writes formats such as Shapefile and GeoPackage through an I/O engine
such as Pyogrio or Fiona. Install one of those engines before calling:

```python
tessellator.to_file("mf6_grid.gpkg", driver="GPKG")
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

<!-- docs:usage-end -->

## Examples

The [examples/](https://github.com/rhugman/vorflow/tree/main/examples)
folder contains runnable scripts and notebooks
covering field-based refinement, mesh quality diagnostics, structured quad
buffers, active-domain workflows, and triangular element-grid export.
[examples/point_centring_demo.ipynb](https://github.com/rhugman/vorflow/blob/main/examples/point_centring_demo.ipynb)
compares Gmsh smoothing, `hex_ring` and `lloyd_iterations` for centring cells
around wells.

## Contributing

Bug reports, questions and pull requests are welcome; see
[CONTRIBUTING.md](https://github.com/rhugman/vorflow/blob/main/CONTRIBUTING.md).
Everyone taking part follows the
[Code of Conduct](https://github.com/rhugman/vorflow/blob/main/CODE_OF_CONDUCT.md).

## Citing

If you use `vorflow` in your work, please cite it using the metadata in
[CITATION.cff](https://github.com/rhugman/vorflow/blob/main/CITATION.cff)
(GitHub's "Cite this repository" button formats it).

## Acknowledgements

Thanks to David ([@d-strom](https://github.com/d-strom)) for reporting the
line-junction problem in [#31](https://github.com/rhugman/vorflow/issues/31)
and for [#35](https://github.com/rhugman/vorflow/pull/35), whose parallel-line
guard and junction diagnostics are part of the line-junction handling.

## License

MIT — see [LICENSE](https://github.com/rhugman/vorflow/blob/main/LICENSE).
