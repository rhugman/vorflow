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

3.  **`VoronoiTessellator`**: This class takes the triangular mesh from `MeshGenerator` and computes its dual: the Voronoi diagram. The result is a grid of polygonal cells. It clips the grid to the domain boundary. Barrier lines are already made of whole cell faces at this stage, because `MeshGenerator` placed the cell generators on either side of them.

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

### Barriers

A line added with `is_barrier=True`, such as a fault, becomes a line of cell
faces, so that a horizontal flow barrier (HFB) can be placed on them. vorflow
places a pair of cell generators across each barrier segment, on circles
drawn around the segment ends. When no other generator lies inside any
circle, each segment is exactly the face between its pair. This is the
empty-circle rule of the UPR method
([Berge et al., 2019](https://doi.org/10.1007/s10596-018-9790-0)).
`MeshGenerator.generate()` places these generators and checks the rule
before Gmsh runs. As a result, each barrier segment is one straight face with
an orthogonality error of 0, also at corners, junctions, crossings and on
finely drawn curves, and the barrier cells need no cutting. This does not
hold for two kinds of barrier, which still go through the tessellator's
mirror generators and cut, so their faces are not always exact: a barrier with
`quad_buffer=True`, and a barrier added with `embed=False`, which only sets
the mesh size.

Before meshing, `ConceptualMesh.generate()` cleans the barrier lines:

- an end within `max(connectivity_tolerance, simplify_tolerance)` of another
  barrier or of the domain boundary is snapped onto it, and a short
  overshoot is trimmed back. A branch no longer than that distance, which
  snaps onto the barrier it starts on, is dropped as a stub. An end is not
  snapped onto a barrier it runs alongside;
- a part of a barrier that lies on the domain boundary (within the same
  distance) is dropped with a warning, because a barrier there separates
  nothing. The end left behind is moved onto the boundary, so a barrier that
  ended on the boundary still does;
- junctions and crossings become vertices shared by both lines, and so does
  a point where a line crosses or touches itself;
- with `simplify_tolerance` above 0, the line is simplified after the
  junctions are joined, so a splay stays on the fault it starts on;
- a vertex closer than 0.1 x `resolution` to its neighbour is dropped as
  digitising noise, with a warning, even when `simplify_tolerance` is 0. A
  vertex is kept where dropping it would leave a fold sharper than
  18 degrees, so the corner of a narrow U-turn stays.

`MeshGenerator.barrier_report` lists each of these changes with its place.
It also lists what the construction changed or gave up while placing the
faces. Each kind raises one `UserWarning` with its count:

- the tip of a spike sharper than `sharp_angle` (18 degrees by default) is
  left inside a few cells without faces (`tips`);
- barrier legs that meet at a junction, or leave the domain boundary or a
  hole, at less than `sharp_angle` share a short stem instead of bounding a
  sliver, and so do fans of legs closer than 29 degrees (`collapsed`). No
  point of a line moves more than 0.1 x `resolution`;
- two nearly parallel barriers closer than about 1.2 x `resolution` share
  one row of cells between them, and so do the close legs of one barrier (a
  U-turn, a thin loop, a lens) and a barrier within about 0.6 x `resolution`
  of the domain boundary (`corridors`). No barrier is moved, and flow along
  the gap stays connected. The cells of the row are at most `resolution`
  and at most 3 x the gap long, so a narrow gap adds cells: about
  `length / (3 x gap)` in the row, and the mesh around it is graded down to
  match. Rows of cells shorter than 0.5 x `resolution` raise one warning
  with their cell count. With `MeshGenerator(gap_field=...)`, a gap where
  `GapField` asks for more than one cell across gets no corridor: each
  barrier gets its own faces, sized by `GapField`, with its rows of cells
  between them (see Narrow gaps). A gap that narrows below the `GapField`
  `min_size` somewhere keeps its one row all along.

Check the report before you build HFB input.

Standard lines, zone edges and quad-buffer strips next to a barrier keep
their nodes: the barrier faces are made shorter there, so the circles do not
reach them. Where such a line crosses a barrier, the crossing takes its
nodes over. If a node of one is removed anyway, `generate()` warns and
`barrier_report["fixed_nodes"]` names the line.

```python
mesher = MeshGenerator(background_lc=100, conceptual_mesh=blueprint,
                       sharp_angle=18, barrier_gradation=0.3)
mesher.generate()
print(mesher.barrier_report["snaps"], mesher.barrier_report["tips"])
```

When the faces would have to be far shorter than `resolution`, `generate()`
raises `vorflow.BarrierMeshError`. The message gives the place, what makes
the faces that short there and what helps for that cause. For example: an
end that nearly meets another barrier is snapped onto it by a
`simplify_tolerance` of at least the gap; a barrier running close to the
boundary is dropped there by the same tolerance; next to the boundary, a
point or a fixed line, the message gives a barrier `resolution` at which
the faces fit. Where barrier legs meet or fold back at angles too narrow for
cells between them (a narrow spike or zigzag, or many legs at one point),
the message gives the `simplify_tolerance` that removes a short fold there,
if it is digitising noise, and otherwise `quad_buffer=True` on the barrier
meshes it with a structured strip of cells instead, whose faces are not
exact.

**Breaking:** a point feature, such as a well, on a barrier or at its end
(closer than about 0.05 x `resolution`) now raises `BarrierMeshError`,
naming the point and the barrier. In 0.3.0 it meshed, with mirrored
generators and cut cells around the point. A cell centre cannot sit on a
barrier face, so move the point off the line, or, where it marks the
barrier's end, end the barrier that far short of it.

Quad buffers on barriers (`quad_buffer=True`) do not use the exact faces
yet. On a straight stretch their cells meet on the barrier, with no node
row on it (`quad_buffer_thickness` counts the cell rows on each side). At
corners, and where the barrier leaves the domain at a slant, they still
rely on mirrored generators and on cutting cells, so they can leave cut
cells and bent faces there. `straddle_width` is deprecated, with a
`FutureWarning`, and its value is ignored, so remove it from barriers. A
non-barrier line that sets it still gets exact faces along it.

[examples/barrier_meshing.ipynb](https://github.com/rhugman/vorflow/blob/main/examples/barrier_meshing.ipynb)
shows each case, the report and these limits.

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
disu = tessellator.to_mfusgdisu(
    model, top=10.0, botm=[0.0, -10.0], nlay=2, gsf_path="model/model.gsf"
)
disu.write_file()  # Or model.write_input() after adding the other packages.
```

`gsf_path` (also accepted by `to_mfusg_disu()`) writes the matching GSF from
the same validated grid. `to_gsf()` writes the GSF on its own; use the same
elevations as for DISU. GSF stores original 3D cell
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

DISU and GSF require contiguous, straight shared faces. Bent faces, such as
those left where a barrier corner is cut inside a cell, and disconnected
shared faces are rejected. Centre and overlap checks allow coordinate
roundoff while preserving the supplied centres and polygons.

For an existing GeoDataFrame, use `vorflow.utils.grid_to_mfusg_disu()` and
`vorflow.utils.grid_to_gsf()`. Plain arrays and GSF output do not need FloPy.
The [simple MFUSG notebook](examples/basic_example_mfusg.ipynb) runs a
single confined layer between two fixed heads, plots the result and checks
analytical heads and the 500 m³/day flow rate as well as budget balance. From
the repository root, install `python -m pip install -e ".[examples]"` in the
notebook kernel's environment. The example
requires the separate `mfusg` executable on PATH or an `MFUSG_EXE` path;
install it outside the notebook with `get-modflow /path/to/bin --subset mfusg`.

### Fully unstructured DISU (full connectivity)

`vorflow.disu` turns a vorflow grid and FloPy-style layer arrays into a fully
unstructured DISU grid for MODFLOW 6 or MODFLOW-USG. Layers can follow dipping or undulating
surfaces: cells whose shared faces overlap vertically are connected even when
they are in different layers ("full connectivity", Provost et al. 2025).

```python
import flopy
from vorflow import disu

d = disu.DisuGrid(grid, top, botm, idomain).build()   # FloPy shapes: (ncpl,), (nlay, ncpl)
print(d.report())
flopy.mf6.ModflowGwfdisu(gwf, **d.get_gridprops_mf6_disu())
flopy.mf6.ModflowGwfnpf(gwf, k=d.to_nodes(k_layers), xt3doptions=True, angle2=0.0, angle3=0.0)
heads = d.to_layers(gwf.output.head().get_data().ravel())  # back to (nlay, ncpl), NaN where removed
```

- **Pinch-outs:** idomain -1 or `min_thickness` marks pass-through cells.
  They are removed, and their interval goes to the active cells above and
  below (`collapse="split"`), so no thickness is lost. idomain 0 is no-flow
  and never passes flow.
- **XT3D:** use it with offset layers, and give NPF `angle2` and `angle3`
  (zeros are fine). Without them MODFLOW 6 XT3D ignores the vertical
  offsets. `connectivity="layered"` gives the DISV connections instead.
- **MODFLOW-USG:** `get_gridprops_mfusg_disu()` for `flopy.mfusg.MfUsgDisU`
  and `write_gsf()`. USG has no XT3D but still gains from the extra
  connections. Between cells of unequal thickness its default conductance is
  too small; `MfUsgLpf(ikcflag=1, ksat=d.get_mfusg_ksat(k))` makes it exact.
- **Finer cells at depth:** pass one grid per layer, e.g.
  `DisuGrid([base, base, pit, pit], top, [b0, b1, b2, b3])` with
  `pit = disu.splice(base, fine, polygon)`: base cells outside the polygon,
  fine cells inside. The outline must be cell faces in both grids
  (`add_line(outline, is_barrier=True)`); its vertices need not match. The
  contact where the grid changes is made gap-free. A change of grid at depth
  suits near-vertical features; for subhorizontal ones use the same grid in
  every layer. The cells on either side of a spliced outline are not mirror
  images of each other, so faces there can be far from orthogonal
  (`report()` gives the largest error).
  `disu.refill(base, polygon, resolution)` avoids that: it keeps the base
  cells outside the polygon and fills the inside with finer Voronoi cells
  that share the outline's vertices, graded by `growth_factor` or `fields`
  as in `add_polygon`. `outline_split=2` makes the cells along the outline
  finer.
- **Dipping contacts at a change of grid:** where fine cells straddle coarse
  ones, the gap-free contact under a refinement is one flat level, which
  moves the bottoms above it; `report()` says by how much.
  `grids, top, botm, idomain, parent = disu.split_contacts(grids, top, botm,
  idomain)` keeps them as given by adding a wedge layer.
- **Sublayers:** `disu.subdivide(top, botm, idomain, nsub=..., where=mask)`
  splits layers everywhere or only in some columns. Its `parent` (and that of
  `split_contacts`) maps the layers back to the original units for
  `layer_view` and `save_cellmap`.
- **Tracking:** `nodemap`, `cellmap`, `to_nodes`, `to_layers`,
  `locate(x, y, z)` and `column(x, y)`. FloPy reads the layer number from the
  IHC diagonal, so `PlotMapView(modelgrid=gwf.modelgrid, layer=k)` works
  directly. `idomain_layers` (1 active, -1 pass-through, 0 no-flow),
  `layer_view` (one 2D map for a layer, also when split into sublayers) and
  `vertex_grid` (a FloPy grid for those maps) cover the rest; MF6 DISU
  accepts IDOMAIN 0/1 only.
- **Layer structure for other tools:** MF6 copies each node's plan polygon
  to the binary grid file, but not its layer. `save_cellmap(path, parent)`
  writes it for tools that work layer by layer in 2D, such as pilot points
  per layer: every layer cell with its node, plan cell, centre, top, bottom
  and mid-elevation, its `idomain` code, and its `unit`, the original layer.
- **Barriers and faults:** mesh the fault into every layer
  (`add_line(fault, is_barrier=True)`), then `faces = d.barrier_faces(fault)`
  returns the connections it cuts, and
  `flopy.mf6.ModflowGwfhfb(gwf, **faces.get_gridprops_mf6_hfb(hydchr))`
  (or `get_gridprops_mfusg_hfb` for `flopy.modflow.ModflowHfb` in USG) makes
  the HFB package. `hydchr` is the barrier conductivity divided by its
  width. Give one trace per layer, with `None`, to limit the barrier to some
  layers. Pass the lines the faces follow, `MeshGenerator.barrier_lines`:
  a trace that runs off the cell faces raises an error.
  For a dipping fault give one trace per layer, at that layer's mid-depth,
  and mesh one grid with all the traces for every layer: the fault becomes a
  staircase of walls and treads (`faces.vertical`), and `faces.factor`
  (|n<sub>fault</sub> · n<sub>face</sub>|, multiplied into `hydchr` by the
  exporters) gives flow normal to the fault the plane's resistance. Flow
  along a dipping fault still meets the staircase. Vertical HFB connections
  need MODFLOW 6.7 or later; for MODFLOW-USG the exporter divides their
  `hydchr` by the mean cell thickness, which USG multiplies it by (exact for
  confined layers).
- **Existing DISV models** (FloPy, LoopFlopy) go in through
  `disu.DisuGrid.from_disv(vertices, cell2d, top, botm, idomain)`.

#### Checking the grid

vorflow builds the grid it is given; whether it is good enough for the model
is the modeller's call. `d.quality()` measures every connection:
`ortho_error_3d` measures the angle between the line joining the two cell
centres and the face normal (`calculate_mesh_quality` gives the plan error of
each plan grid); `cl_ratio` measures how unevenly the face sits between the
centres; `gap` gives the pass-through thickness a vertical connection jumps
over without resistance (with `collapse=None`, or between two changes of
grid). `d.plot_quality(layer, column)` maps one layer's connections and
`d.plot_quality_section(line, column)` draws them in a section; pass part of
the table to draw only those connections.
To screen every layer at once, colour each cell by its worst connection:

```python
q = d.quality()
side = q[q.ihc != 0]                                   # q.ihc == 0: top and bottom faces
both = pd.concat([side.assign(node=side.n), side.assign(node=side.m)])
worst = both.groupby("node").ortho_error_3d.max().reindex(range(d.nodes))
for k, (grid, v) in enumerate(zip(d.grids_per_layer, d.to_layers(worst.to_numpy()))):
    grid.assign(v=v).plot("v", ax=axes[k], vmin=0, vmax=60)
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

### Narrow gaps

Where two features run close together, the mesh between them is only as fine
as their resolutions and the background make it, so a narrow gap can end up
one or two cells across. `GapField` finds the gaps between embedded features
and caps the size there at gap / `cells_across`. It is model-wide, so it goes
on the mesh generator rather than on a feature:

```python
from vorflow import GapField

mesher = MeshGenerator(
    background_lc=40,
    conceptual_mesh=blueprint,
    gap_field=GapField(cells_across=3, min_size=2.0),
)
mesher.generate()
print(mesher.diagnostics["gap_field"])  # field_created, min_gap, n_floor_bound, ...
```

- **What is a gap.** The local distance between embedded polygon
  boundaries, lines and points, measured on an approximate medial axis of
  their linework. Separation is measured along the linework, so a junction or
  corner wider than about 30 degrees is not a gap, while a polygon's own
  narrow neck is. Field-only (`embed=False`) features are never a side of a
  gap. Barrier, straddle and quad-buffered lines count as plain lines (the
  points vorflow places along a barrier do not).
- **Barriers.** Where `GapField` asks for more than one cell across a gap
  between two barriers, or between a barrier and the domain boundary, that
  gap gets no shared row of cells (see Barriers): each barrier gets its own
  faces, no longer than the `GapField` size there, with the rows of cells
  between them. A gap that narrows below `min_size` somewhere keeps its
  shared row all along, as without `GapField`. Elsewhere along a barrier the faces follow the `GapField`
  size too, so the cells along it match Gmsh's. If the faces would have to
  be far shorter than the barrier's `resolution`, `generate()` raises
  `BarrierMeshError` and gives a `min_size` that works.
- **Size.** Across a gap of width w the size is `w / cells_across`, graded
  away from it at `growth_factor` (default 1.2), and Gmsh still takes the
  smallest of this and the model's own fields. With `cells_across=3`, about
  three cells lie wholly between the two walls, plus the cells centred on
  them.
- **No narrow gap, no change.** Only gaps narrower than `cells_across` times
  the local size (the background, or a feature's default size field where it
  is finer) are refined. When there are none, no Gmsh field is added and the
  mesh is identical to one built without `gap_field`.
- **`min_size`** (model units, default 0.1 x `background_lc`) is the floor on
  the size. It is needed because wedges, such as a line meeting the boundary
  at under about 30 degrees, narrow to zero at the tip. Wherever gap /
  `cells_across` is below it, a warning names the place and the cells across
  it gets. The linework is sampled every `min_size / 2`, so a smaller
  `min_size` costs more.
- **Opting features out.** The field cannot tell an oversight from a
  deliberate coarse region. Add `gap_wall=False` to features whose gaps should
  not be refined, such as a boundary-condition line digitised just inside the
  domain edge or a sliver polygon:

  ```python
  blueprint.add_line(river_bc, line_id="river", resolution=40, gap_wall=False)
  ```

  A `gap_wall=False` feature is never a side of a gap; for a polygon, that
  covers its whole boundary, including edges it shares with other polygons.

Refining a gap costs cells. In a 1000 x 600 m domain at `background_lc=40`,
two lines 15 m apart get about three cells across with `GapField(3)`, and the
grid grows from about 550 to 2400 cells. Vertex jitter larger than
`min_size / 4` can read as small notches, which the field refines down to
`min_size`; simplify such input first (`simplify_tolerance`).

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

### Concave cells and particle tracking

Clipping to the domain makes a cell concave wherever the boundary or a hole
has a reflex corner (one that bends into the domain): Gmsh puts a node on
that corner, so its cell is cut into an L or V shape. Flow models handle
these cells, but MODFLOW 6 particle tracking (PRT) can loop forever, with no
error, in a cell where a face cannot be seen from the cell centroid. PRT
splits each cell into triangles that meet at the centroid, and a hidden face
gives one of them a negative area.

`calculate_mesh_quality` flags both cases: `is_concave` and
`star_from_centroid` (False where a face is hidden from the centroid).

```python
from vorflow.utils import calculate_mesh_quality

quality = calculate_mesh_quality(grid_gdf)
print(quality["is_concave"].sum(), (~quality["star_from_centroid"]).sum())
```

For PRT, `split_concave="hidden_face"` cuts only the cells with a hidden
face into convex pieces; `split_concave="all"` cuts every concave cell. Cuts
run from the reflex corner along a diagonal of the cell where one works.
Otherwise the cut follows the corner's angle bisector, and the neighbouring
cell gains a vertex where the cut ends (its shape does not change). Where the
bisector would end on a barrier face, two diagonals to the ends of that face
cut it off instead, so every barrier segment stays one face. The
largest piece holding the generator keeps the cell's `node_id` and zone; the
others get new IDs and the zone that holds most of their area. Pieces usually carry their centroid as `x`/`y`, unlike every
other cell: the generator sits on the corner, at the end of the cut, and a
cell centre there makes XT3D crash. A piece that holds the generator clear of
its boundary keeps it, and a piece with one barrier face takes a point on the
normal to that face through the generator, so barrier faces stay orthogonal.
The mesh node stays in `source_x`/`source_y`,
and a `concave_split` column marks the pieces. A cell that cannot be split
(for example one with a hole, which a DISV cell cannot represent) is left as
it is, with a warning.

```python
tessellator = VoronoiTessellator(mesher, blueprint, split_concave="hidden_face")
grid_gdf = tessellator.generate()
print(tessellator.concave_split_report)  # split cells and their piece IDs
```

The split is off by default because it costs some flow accuracy: the new
faces have a CVFD skewness of about 0.5. In a test with an exact linear head
solution, the head error rose from round-off to about 1e-5. Use it when you
run particle tracking, not as a general improvement.

<!-- docs:usage-end -->

## Examples

The [examples/](https://github.com/rhugman/vorflow/tree/main/examples)
folder contains runnable scripts and notebooks
covering field-based refinement, mesh quality diagnostics, structured quad
buffers, active-domain workflows, and triangular element-grid export.
[examples/point_centring_demo.ipynb](https://github.com/rhugman/vorflow/blob/main/examples/point_centring_demo.ipynb)
compares Gmsh smoothing, `hex_ring` and `lloyd_iterations` for centring cells
around wells.
[examples/barrier_meshing.ipynb](https://github.com/rhugman/vorflow/blob/main/examples/barrier_meshing.ipynb)
shows how barrier faces are placed, what `barrier_report` lists and the
remaining limits.

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

The full-connectivity DISU builder (`vorflow.disu`) follows the algorithm of
C. D. Langevin's USGS `disv2disu.py`, published with Provost, A.M., Bardot,
K., Langevin, C.D. and McCallum, J.L. (2025), *Accurate simulation of flow
through dipping aquifers with MODFLOW 6 using enhanced cell connectivity*,
Groundwater 63(3), 399-408 (<https://doi.org/10.1111/gwat.13459>), and the
problem described by Bardot, K., Lesueur, M., Siade, A.J. and McCallum, J.L.
(2023), *Revisiting MODFLOW's capability to model flow through sedimentary
structures*, Groundwater 61, 663-673 (<https://doi.org/10.1111/gwat.13273>).
Its pass-through of pinched cells follows
[LoopFlopy](https://github.com/kerrybardot/loopflopy) (Bardot et al. 2025,
Environmental Modelling & Software 192, 106557).

Thanks to David ([@d-strom](https://github.com/d-strom)) for reporting the
line-junction problem in [#31](https://github.com/rhugman/vorflow/issues/31)
and for [#35](https://github.com/rhugman/vorflow/pull/35), whose parallel-line
guard and junction diagnostics are part of the line-junction handling.

## License

MIT — see [LICENSE](https://github.com/rhugman/vorflow/blob/main/LICENSE).
