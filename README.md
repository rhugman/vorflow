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

3.  **`VoronoiTessellator`**: This class computes a Voronoi diagram from the mesh nodes, clips it to the domain and handles barrier features according to their policy. Original Gmsh connectivity is not necessarily the final Voronoi dual. Existing `add_line` barriers retain historical mirror/cut enforcement; `barrier_policy="preserve"` explicitly retains the generated layout.

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

### Quad barriers and tracking the cells on each side

`vorflow.utils.cells_on(gdf, geometry)` returns selected cells with zero-based
row-position `cell_id` values, full-cell areas and line-intersection lengths.
It always returns a GeoDataFrame, including empty and single-cell results, and
keeps the original index and metadata. Points on a shared boundary select all
touching cells; decide explicitly which cells should receive a model condition.

`faces_on(build_connectivity(gdf), line, tolerance=...)` selects complete shared
faces. Connectivity retains its representative `shared_edge` and metrics for
compatibility, and adds `full_shared_edge` containing all linear segments.
The lookup returns the full geometry and total `face_length`; an incomplete
target match selects no face. Zero tolerance requests exact coverage. Face
selection does not certify numerical suitability: use the quality and barrier
coverage reports before assigning hydraulic resistance. These lookup helpers
build on Erik Toller's [PR #64](https://github.com/rhugman/vorflow/pull/64).

A transfinite quad strip controls the generator arrangement. When a one-row
strip already aligns the Voronoi faces with a straight barrier, cutting adds
nothing. A two-row strip puts generators on the line, so cells can cross it.
Cutting those cells enforces the geometry, but can leave centres on faces,
nonorthogonal connections or bent shared faces. Sharp bends, curves, ends and
trimmed strips can also leave crossings; there is no universal safe angle.

Keep authoring features through `add_line()` and `add_polygon()`. For a line,
choose straddle sites or a quad buffer and make optional repairs explicit:

```python
cm.add_line(
    fault_line, "fault", resolution=5.0, is_barrier=True,
    quad_buffer=False,          # True selects a one-row Gmsh quad buffer.
    barrier_policy="preserve", # Retain source cells without cutting.
    barrier_mirrors=False,      # Independent optional reflected generators.
    simplify_tolerance=0,
)
# To clip instead: barrier_policy="cut", barrier_merge="restore".
# "checked" keeps rejected fragments; "none" disables merging.
```

Polygon quad bands already use `add_polygon(..., quad_buffer=True)`. For an
outline that should guide the mesh without becoming a zone or a hydraulic
barrier, use `add_line(polygon.boundary, ..., quad_buffer=True,
is_barrier=False)`. Positive `straddle_width` also enables straddle meshing on
ordinary lines. Hydraulic properties are assigned separately in the model.

`straddle` places pairs across sampled stations. `quad` meshes a buffer with
Gmsh and uses its nodes as generators. Open strips attempt transfinite rows;
closed rings use complete annular geometry with recombination. This fixes the
previous missing sides on closed square strips. Recombination cannot promise
paired rows around every curve. Inspect `mesh_generator.source_report`: it
counts actual quads/triangles, transfinite/recombine surfaces, missing footprint
area and target length outside the generated buffer. Requested input flags alone
do not establish successful application.

Mirrors add generators and recompute Voronoi cells. Clipping changes polygons.
Clipping is followed by the selected `barrier_merge` policy: `checked` attempts
merges and keeps rejected fragments; `restore` attempts the same merges and then
undoes failed parent cuts; `none` disables merging. Only fragments smaller than
20% of their original cell are candidates. Inspect
`tessellator.barrier_merge_diagnostics` for actual final merges, retained
fragments, restored parent cells and rejection checks.
`barrier_merge="restore"` starts every retry from the pre-cut partition, undoing all
dependent fragment absorption when a parent cut is refused. It retains original
IDs/centres and allows an imperfect barrier instead of leftover fragment cells.
It checks convexity, all affected faces, retained-parent compactness (at least
0.4) and retained-parent area (at least 5% of its original area). These are
conservative geometry checks, not a guarantee of orthogonality or flow accuracy.

```python
quality = tessellator.quality_report(cell_size=5.0, domain=domain)
print(quality["summary"])  # Hard geometry failures separate from quality warnings.
tracking = tessellator.barrier_report(approximate=True)
```

Exact `tracking["faces"]` remains separate from proposed
`tracking["approximate_faces"]`. The latter uses **existing full shared faces**
between inside/outside centres for closed rings, or crossed centre connections
for open lines. Inspect `usable`, `max_offset` and `approximate_summary` topology
before choosing this approximation. Open routes can be disconnected; closed
routes must form the intended single cycle. Centres on the target and ambiguous
crossings are excluded. No hydraulic resistance is assigned automatically.
Changing the target to a cell-face approximation is an explicit modelling choice.

For a closed loop, `barrier_report(approximate=True, closed_ties="majority")`
can assign cells whose centres lie on the target by their inside/outside polygon
area, using outside for balanced areas. Inspect `approximate_ties`, offsets in
both directions, and `usable_single_cycle` before choosing these real-face
connections. The default `closed_ties="exclude"` still leaves these cells
unassigned. Open-line ties are always excluded; neither option changes the mesh
or creates an HFB package.

Run `python scripts/validate_barrier_flow.py --engines mf6` after installing
`.[examples]` and providing the executable through `MF6_EXE` or PATH. Optional
USG benchmarks also require the separate MFUSG export support.
It retains fresh model inputs, heads, budgets and measured errors under `.bin/`.
It compares heads **and total flow** against analytical linear/HFB solutions;
normal termination and budget balance alone are insufficient.

Closed straddle layouts can retain unmatched corners even after conservative
repairs. The exact closed-barrier benchmark refuses those configurations rather
than assigning an incomplete HFB. A quad layout may give complete coverage;
check the report instead of assuming that a requested source layout guarantees it.

The existing `add_line` API retains historical defaults: all barriers, including
quad buffers, receive mirror/cut enforcement and legacy fragment merging.
Missing policy/merge columns in loaded feature frames retain those defaults.
To preserve a quad layout through the older API, opt in explicitly.
Domain-boundary clipping remains independent:

```python
cm.add_line(
    fault_line, line_id="fault", resolution=5.0,
    is_barrier=True, quad_buffer=True, quad_buffer_thickness=1,
    barrier_policy="preserve",  # Explicit opt-in; the compatibility default enforces cuts.
)
```

The per-line policies are:

| Policy | Behaviour |
| --- | --- |
| `"auto"` (default) | Historical mirror/cut fallback on every barrier, including quad buffers. |
| `"preserve"` | No cuts or fragment merges; mirrors off unless explicitly enabled; report unresolved crossings. |
| `"strict"` | Preserve the layout and reject the result if the barrier enters a cell interior. |
| `"cut"` | Allow barrier mirrors and fallback cuts, including for quad barriers. Merging is controlled separately. |

Cutting and fragment merging are independent. For a line using the cut policy:

```python
cm.add_line(
    fault_line, line_id="fault", resolution=5.0, is_barrier=True,
    barrier_policy="cut", barrier_merge="none",  # Cut, but retain all fragments.
)
```

`barrier_merge="legacy"` retains the historical `add_line` default.
Explicit `barrier_merge="checked"` considers the same small fragments
(less than 20% of their source cell area), but accepts a merge only when the
union is convex to coordinate roundoff, retains its numerical centre, and all
affected shared faces are contiguous and straight with positive centre-to-face
distances on opposite sides. It also checks contacts with other retained
fragments. The geometry and centres are not moved to make a proposal pass.
Refused proposals retain their fragments as separate cells. Inspect
`tessellator.barrier_merge_diagnostics` for merged/retained counts and reasons
such as `nonconvex_union`, `bent_face` or `invalid_face_centres`. Reason counts
refer to candidate checks, which can repeat when neighbouring fragments merge.

`barrier_merge="none"` disables merging; `"legacy"` explicitly restores the old
longest-shared-boundary rule. Convexity alone is insufficient: a convex merged
cell can share two bent edges with a concave neighbour. Checked merging does
not repair defects introduced by the cut itself, certify orthogonality or
eliminate retained slivers. Final geometry and flow still need validation.
Merge settings have no effect on a line using `preserve` or `strict`.
`barrier_mirrors=True/False` overrides policy-dependent mirroring independently
of cutting, including mirror-only repair with `barrier_policy="preserve"`.
They control cell-fragment unions; roundoff cleanup of duplicate vertices is
a separate operation.

`tessellator.barrier_diagnostics` gives final crossing counts and zero-based cell
positions for each barrier. Preserved crossings produce a warning; they must
not be interpreted as an enforced hydraulic barrier. To obtain the actual
paired cells along the line, request the optional, solver-independent report:

```python
report = tessellator.barrier_report()
faces = report["faces"]
print(faces[["line_id", "left_cell", "right_cell",
             "left_node_id", "right_node_id", "centres_opposite", "length"]])
print(report["summary"])

# Inspect the barrier sections entering cell interiors or lacking a unique pair.
crossings = report["crossings"]
unmatched = report["unmatched"]

# Unique horizontal pairs for later, engine-specific hydraulic assignment.
import numpy as np
pairs = np.unique(
    np.sort(faces[["left_cell", "right_cell"]].to_numpy(dtype=int), axis=1), axis=0,
)
```

An **unresolved crossing** is a barrier section inside a cell; the polygon can
still be valid. Optional `strict` mode stops on that alignment condition and
does not cut or merge that barrier. Refusing a proposed merge is different:
the merge is skipped and its fragment retained; that alone does not fail grid
generation. The report's summary includes the effective policy and merge mode.

Left/right follow the coordinate direction of each **cleaned line part**;
reversing it swaps sides. Cell IDs are positions in the final grid, matching
the horizontal DISV and USG-export cell order. Original grid indices and node
IDs remain available. Hanging vertices yield multiple pairs, corner-only
contacts yield none, and bent interfaces retain their individual segments.
One pair can have several segment records; deduplicate pairs when constructing
a hydraulic package. Keep `line_id` when different barriers need different
properties, and resolve any overlapping assignments deliberately.

Matching tolerance is in CRS units; it changes the report, not the grid.
For an existing GeoDataFrame, use
`vorflow.utils.grid_barrier_report(gdf, cm.clean_lines, tolerance=...)`.
Inspect unmatched lengths, crossings, `centres_opposite` and orthogonality
before creating model inputs. This report assigns no resistance, expands no
layers and writes no HFB package. MF6 and MODFLOW-USG require their own cell/node
formats and hydraulic parameters; expand horizontal pairs through the model's
layer mapping when needed.

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
