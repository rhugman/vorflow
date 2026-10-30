# Changelog

All notable changes to `vorflow` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ConceptualMesh.features()` returns the cleaned lines, points and polygons
  as one GeoDataFrame with `feature_type`, `feature_id` and `embed`.
- `vorflow.utils.cells_on(grid, geometry)` returns the cells a point or line
  lies on, with `cell_id` (the DISV cell number), `cell_area` and the
  `line_length` inside each cell. It also takes a GeoDataFrame such as
  `ConceptualMesh.features()`. `vorflow.utils.faces_on(connectivity, line)`
  returns the shared faces along a line, e.g. a barrier, with `face_length`,
  for building MF6 inputs such as HFB.
- `ConceptualMesh.check(background_lc=None)` reports input defects that tend
  to produce bad Voronoi cells, before meshing and without Gmsh: short
  segments, close vertices (including a line crossing a polygon edge beside a
  vertex), line ends and points that just miss another feature, sliver
  polygons and narrow gaps. Each is measured against the local mesh size `h`
  as `ratio = distance / h` and graded `"error"` or `"warning"`; one defect
  site gives one row of a GeoDataFrame. Only embedded features are checked,
  on the cleaned features before densification. A barrier end that runs
  alongside another barrier or the domain outline is reported as a narrow
  gap, not a near miss, since the barrier construction gives that gap one
  row of cells. The registered features and earlier `generate()` results are
  left unchanged. See "Checking inputs before meshing" in the user guide.
- The release workflow creates the GitHub Release for each final tag once it
  is on PyPI, with notes from the tag's changelog section and the wheel and
  sdist attached (`scripts/release_notes.py`). A tag without a changelog
  section now fails the build before anything is uploaded.
- `vorflow.BarrierMeshError`, a `ValueError`. `MeshGenerator.generate()`
  raises it, before Gmsh runs, when it cannot make every barrier segment a
  cell face, for example where the faces would have to be far shorter than
  the barrier's `resolution`. The message gives the place (x, y), what
  makes the faces that short there (another barrier part, the boundary, a
  point, a fixed line, closely spaced vertices, a barrier leaving the
  boundary at a narrow angle, or legs meeting at too narrow an angle) and
  what helps for that cause: a `simplify_tolerance`
  that snaps the end or drops the part, the barrier `resolution` at which
  the faces fit next to the boundary, a point or a fixed line (rounded down,
  with a 10% margin, so the value given meshes), the `simplify_tolerance`
  that removes a short fold, measured from the fold itself, or
  `quad_buffer=True` where nothing else does. Where the construction made
  the pieces short (a merged junction), the message says the input is not
  at fault and asks for a report. A geometric failure inside the
  construction is raised as `BarrierMeshError` too, asking for a report. A point feature (or a kept hex-ring seed) on a
  barrier raises it as well (see Changed).
- `MeshGenerator.barrier_report`: a dict that lists, each with its place,
  what the barrier construction changed or gave up. The entries are ends
  snapped or trimmed onto another barrier or the boundary (`snaps`),
  vertices dropped as digitising noise (`noise`), short pieces dropped past
  a junction or collapsed by snapping (`stubs`), sharp spike tips left
  without faces (`tips`), barrier parts dropped because they lie on the
  domain boundary (`on_boundary`), hex rings dropped because a barrier is
  too close (`rings`), nodes of standard lines, zone edges or quad-buffer
  strips that had to be removed (`fixed_nodes`), narrow gaps given one
  shared row of cells (`corridors`, with the gap, how each end closes and
  the cells of the row) and narrow junctions and boundary exits merged over
  a short reach (`collapsed`).
  `polish` reports the Lloyd polish, with the nodes it `moved`.
  `removed_nodes` counts the Gmsh nodes
  removed for the barrier sites, not counting Gmsh's own copies of the
  sites. `MeshGenerator.barrier_lines` holds the lines
  the faces follow, `barrier_circles` the empty circles and `node_is_site`
  marks the nodes placed on them.
- `MeshGenerator(barrier_gradation=0.3, sharp_angle=18,
  barrier_polish_iterations=20)`. `barrier_gradation` limits how fast the
  barrier face length changes along the barrier network. `sharp_angle` is
  the angle, in degrees, below which a spike's tip is left without faces
  and two legs of a junction share a short stem. `barrier_polish_iterations`
  sets the Lloyd steps that move the free mesh nodes within 3 x their local
  size of the barrier circles; nodes further away keep their Gmsh
  positions.
- `ConceptualMesh(keep_close_rings=False)`. With True, a point's `hex_ring`
  is kept next to a barrier, and the barrier faces there are made shorter
  so the ring stays clear of them. By default the ring is dropped with a
  warning, as next to any other embedded line.
- `MeshField.python_size()`: the size a field asks for at a distance from
  its feature, evaluated before Gmsh runs. The barrier construction reads
  it to size the barrier faces. The built-in fields implement it. A custom
  field without it is left out of that estimate, with a `UserWarning`
  naming the field, though Gmsh still applies it.
- The notebook `examples/barrier_meshing.ipynb` shows how barrier faces are
  placed, what `barrier_report` lists and the remaining limits. The README
  has a new "Barriers" section.
- MODFLOW-USG-specific `to_mfusg_disu()` / `grid_to_mfusg_disu()` arrays and
  mapping/orthogonality diagnostics, plus optional `to_mfusgdisu()` FloPy
  package creation. These use USG DISU conventions, not MF6 DISU. Stacked
  layers share one horizontal mesh; invalid geometry, bent or disconnected
  shared faces and nonpositive thickness are rejected. Coordinate roundoff
  at clipped boundary centres and overlapping boundary slivers is accepted
  without moving the input centres or polygons.
- Original 3D GSF geometry export with `to_gsf()` / `grid_to_gsf()`, using
  the same centres, elevations and node numbering as USG DISU. Pass
  `gsf_path` to the DISU exports to write both from one validation pass.
  FloPy and MFUSG executables remain optional.
- A simple executed MODFLOW-USG notebook: one confined layer, two fixed-head
  sides, paired DISU/GSF export, head plot and water-budget checks.
- The Conda development environment includes FloPy and installs both the
  `dev` and `examples` extras in editable mode, so model experiments use the
  checkout selected when updating the environment.
- `GapField(cells_across=3, growth_factor=None, min_size=None)`, passed as
  `MeshGenerator(gap_field=...)`, refines narrow gaps between embedded
  features to about `cells_across` cells. The gap is read off an approximate
  medial axis of the feature linework (shapely + SciPy, `vorflow._gap`), with
  separation measured along the noded linework so junctions are not gaps, and
  a polygon's own narrow neck counts. The size is gap / `cells_across`,
  graded at `growth_factor`, and combined with the model's own fields under
  the usual `Min`. Where no gap is narrower than `cells_across` times the
  local size, no field is created and the mesh is unchanged. `min_size` (in
  model units, default 0.1 x `background_lc`) floors the size, with a warning
  that names each place where it leaves fewer than `cells_across` cells.
  `MeshGenerator.diagnostics["gap_field"]` summarises the estimate. With
  barriers, the barrier faces follow the `GapField` sizes (read from the same
  estimate as the Gmsh field), and a gap where it asks for more than one cell
  across, between two barriers or a barrier and the boundary, gets no shared
  row of cells: each barrier gets its own faces with `cells_across` cells
  between them. A gap that narrows below `min_size` somewhere keeps its
  shared row all along, since the `GapField` rows could not pass there.
  The points vorflow places along a barrier are not gap
  walls. Faces too short to build raise `BarrierMeshError` with a `min_size`
  that works.
- `gap_wall=` on `ConceptualMesh.add_polygon`, `add_line` and `add_point`
  (default True): with `gap_wall=False` the feature is never a side of a gap
  for `GapField`, e.g. a boundary-condition line just inside the domain edge.
- `calculate_mesh_quality` adds `is_concave` (convexity below 1 - 1e-6)
  and `star_from_centroid` (every face visible from the cell centroid).
  MODFLOW 6 particle tracking (PRT) can loop forever, without an error, in a
  cell where a face is hidden from the centroid. `summarize_quality` logs the
  counts.
- `VoronoiTessellator(split_concave=False)`: `"hidden_face"` cuts the
  concave cells with a face hidden from the centroid into convex pieces, and
  `"all"` cuts every concave cell. Cuts run along diagonals from the reflex
  corner (a bisector cut, with the new vertex added to the neighbour, where no
  diagonal works; where the bisector ends on a barrier face, diagonals to the
  ends of that face instead, so each barrier segment stays one face). The largest piece holding the generator keeps `node_id`
  and zone, and the others get new IDs and the zone holding most of their
  area; a cell that cannot be split is left as it is,
  with a warning. Pieces carry their centroid as `x`/`y`, because the
  generator sits at the end of the cut and XT3D crashes on a cell centre
  there, except that a piece holding the generator clear of its boundary
  keeps it and a piece with one barrier face takes a point on that face's
  normal through the generator, so barrier faces stay orthogonal; the mesh
  node stays in `source_x`/`source_y`, and a `concave_split`
  column marks the pieces. `concave_split_report` lists the split cells. Off by default: the new faces
  have a skewness of about 0.5 and add about 1e-5 head error on a linear
  problem, so it is meant for particle tracking.

### Changed

- Next to a barrier, a cell whose generator lies on the edge between zones
  now takes the one of those zones holding most of its area (summed over
  all polygons with that `zone_id`), when it holds at least 0.1 more of the
  cell than the zone of highest `z_order`. The barrier sites bound such a
  cell on one side, so `z_order` alone could give it a zone holding as
  little as 5% of it. Only the cells of the barrier sites and the cells
  touching them are affected. Everywhere else, and in models without
  barriers, `z_order` decides as before. A cell still never takes a zone
  its generator is not in.
- Barriers (`add_line(..., is_barrier=True)`) are now meshed so that every
  barrier segment is exactly one Voronoi cell face. vorflow cuts each
  barrier into segments and draws a circle around each segment end. It
  places a pair of cell generators where the circles of a segment cross,
  one on each side of the line. If no other generator lies inside a circle,
  the segment is the face between that pair (the empty-circle rule of the
  UPR method; Berge, Klemetsdal and Lie, 2019, *Computational Geosciences*
  23:169-188, <https://doi.org/10.1007/s10596-018-9790-0>). These
  generators are placed and checked before Gmsh runs. Gmsh then meshes the
  rest of the domain, and the mesh nodes inside the circles are removed.
  Every barrier face is one straight edge with an orthogonality error of 0.
  On the 19 barrier benchmark cases, cut cells fell from 48 to 0 and faces
  of several edges from 529 to 0. The mean worst compactness of the cells
  next to a barrier rose from 0.59 to 0.76, with about 20% fewer cells near
  the barriers. Meshes of models with barriers change: the node positions
  and counts near every barrier differ from 0.3.0. Meshing a model with
  barriers takes about 1.7 x as long as in 0.3.0 over 23 benchmark cases
  (12.7 s against 7.4 s, 1.5-2.3 x per case) and 2.0 x over 43 held-out
  cases, up to 5.2 x for one case.
- **Breaking:** a point feature (or a kept hex-ring seed) on a barrier or
  at its end, closer than about 0.05 x the barrier's `resolution`, now
  raises `BarrierMeshError` naming the point and the barrier. In 0.3.0 it
  meshed, with mirrored generators and cut cells around it. Move the point
  off the line, or, where it marks the barrier's end, end the barrier that
  far short of it.
- Two nearly parallel barriers closer than about 1.2 x their resolution
  share one row of cells (a corridor), with generators on the midline
  between them; no barrier is moved. Along the gap, the cells are at most
  the resolution and at most 3 x the gap long, so the spacing shrinks
  smoothly as the gap narrows, with no jump at any gap width, and flow
  along the gap stays connected. The 3:1 limit keeps the cells of the row
  at a compactness of about 0.59 (a 3:1 rectangle), and the cells around it
  above 0.6. The same applies to the legs of one barrier that run close
  together: a U-turn, a thin loop, the two strands of a lens. A corridor
  follows a wavy gap, turns round the bend of a U-turn and continues past
  a fork. A well, point or fixed line in the gap keeps the corridor out;
  the faces there are then made short enough for cells on each side. A barrier
  closer than about 0.6 x its resolution to the domain boundary gets the
  same treatment, with the generators on the boundary. Before, such gaps
  were filled with thin cells or cut cells. Rows of cells shorter than
  0.5 x resolution raise one `UserWarning` with their cell count and the
  count the resolution alone would give; the cells of each row are in
  `barrier_report["corridors"]`. A narrow gap costs cells: about
  `length / (3 x gap)` in the row, and about ten times that in all once the
  mesh around it is graded down.
- Barrier legs that meet at a junction (a splay, a T-branch, a crossing),
  or leave the domain boundary or a hole edge, at less than `sharp_angle`
  share a short stem along the bisector and then part at 1.5 x
  `sharp_angle`, so no sliver is left between them (a boundary exit
  slides along the boundary and parts from it at 1.5 x `sharp_angle`, or at
  most 25 degrees more than its own angle, short of the next bend; a corner
  within 0.1 x resolution of the exit counts); fans and stars whose
  neighbouring legs are closer than 29 degrees are merged the same way. No
  point of a line moves more than 0.1 x its resolution. One `UserWarning`
  gives the count; the details are in `barrier_report["collapsed"]`. This
  also changes meshes that worked before, with 10-30% more cells near the
  merged legs (a crossing at 10 degrees went from 568 to 733 cells), and a
  barrier leaving the boundary at 3 degrees now starts up to 1.9 x its
  resolution further along the boundary.
- `ConceptualMesh.generate()` cleans barrier lines before meshing. An end
  within `max(connectivity_tolerance, simplify_tolerance)` of another
  barrier, a barrier vertex or the domain boundary is snapped onto it, and
  a short overshoot past another barrier is trimmed back. Junctions and
  crossings become shared vertices. A vertex closer than 0.1 x
  `resolution` to its neighbour is dropped as digitising noise, with a
  `UserWarning` giving the largest move, even when `simplify_tolerance` is
  0. A vertex is kept where dropping it would leave a fold sharper than
  18 degrees, so the corner of a narrow U-turn stays. A branch that
  snapping collapses onto the barrier it starts on is
  dropped as a stub. An end is not snapped onto a barrier it runs
  alongside. A barrier part lying on the domain boundary is dropped with a
  `UserWarning`, since a barrier there separates nothing, and the end left
  behind is moved onto the boundary so the barrier still ends there; a
  barrier lying wholly on the boundary leaves no row in `clean_lines`. A
  line that crosses or touches itself gets a vertex there. Cleaning a long
  digitised trace takes time linear in its vertex count. Barrier lines are
  no longer densified.
- `simplify_tolerance` on a barrier now simplifies after the junctions are
  joined, keeping junctions, crossings and ends, and uses the fewest
  straight pieces within the tolerance. Before, a per-line Douglas-Peucker
  pass ran first and could pull a splay off the fault it starts on.
- A spike sharper than `sharp_angle` (18 degrees) leaves the last
  0.45 x `resolution` of each leg without faces, so the cells at the tip
  stay regular. The tip is listed in `barrier_report["tips"]`.
  `MeshGenerator.generate()` raises one `UserWarning` with the count of
  spike tips, one for dropped stubs, one for snapped or trimmed ends, one
  for removed nodes of fixed lines, one for merged junctions and one for
  dense corridors; the details are in
  `barrier_report`.
- Next to a barrier, the barrier faces are made shorter so that the nodes of
  standard lines, zone edges and quad-buffer strips stay outside the empty
  circles. Where such a line crosses a barrier, the crossing's own sites
  take over its nodes.
- The tessellator's barrier mirror generators and cell cutting are now a
  safety net for barriers. They found nothing to do on any of the 73
  barrier benchmark cases. If they are ever needed on a barrier,
  `VoronoiTessellator.generate()` warns. `MeshGenerator.get_element_grid()`
  and `get_triangular_quality()` describe Gmsh's mesh before the barrier
  sites are fitted. Quad buffers on barriers
  (`quad_buffer=True`) still use the mirror generators and the cut, so they
  can still leave cut cells and faces of several edges.
- Embedding points (point features, hex-ring seeds, barrier straddle pairs)
  now scales linearly with their number. Each point's owning surface was
  found with one `gmsh.model.isInside` call, whose cost grows with the points
  already in the surface, so the step was quadratic: 2.1 s of a 4.8 s
  `generate()` at 10⁴ points. The owner now comes from the fragmented
  topology (the surface holding the point as an interior vertex), with
  `isInside` kept for points on a boundary or in no surface, and points are
  embedded with one `embed` call per surface. Meshes are unchanged: nodes and
  `node_tags` are identical on all benchmark cases.

### Deprecated

- `add_line(straddle_width=...)` emits a `FutureWarning`, and its value is
  ignored: the barrier faces are now placed automatically, so remove it
  from barriers. A non-barrier line that sets it to any positive value
  still gets exact cell faces along it, as before; `is_barrier=True` gives
  the same faces and marks the line in `barrier_lines`.

### Removed

- The internal straddle-pair code (`vorflow._straddle`: straddle pairs,
  crossing plans and their spacing), replaced by `vorflow._barrier`.

### Fixed

- Barriers no longer leave cut cells or faces of several edges at corners
  ([#67](https://github.com/rhugman/vorflow/issues/67)), short legs between
  corners, finely drawn curves, T-junctions and crossings of barriers, or
  standard lines crossing a barrier next to a corner. A cut cell had no
  mesh node of its own, so its connections were not at right angles to its
  faces. A face of several edges is a bent face shared by two cells.
  MODFLOW 6 DISV takes the connection width from its first edge only, so on
  two benchmark cases the flow through an HFB on these faces was 42-56%
  lower than on a 4 x finer grid. With exact faces it is within 1% of the
  finer grid (24.84 against 24.97 on `real_faults`).
- `boundary_centering="inset_mirror"` no longer leaves uncovered strips (and
  MODFLOW 6 disconnected faces) along irregular boundaries
  ([#71](https://github.com/rhugman/vorflow/issues/71)). At a reflex boundary
  vertex (one bending into the domain) a mirror ghost outside the domain still
  owned part of it, and ghost cells are discarded before clipping. Such ghosts
  are now dropped and their nodes clipped (`boundary_centering` is `"clip"` in
  their row). Every vertex of a polygonised circular hole is reflex, so its
  nodes are all clipped.
- Cells are no longer dropped from the grid, or split in two, where zone
  polygons overhang the domain polygon. The union of the cookie-cut zones,
  which cells are clipped to, kept roundoff slits along their shared edges.
  GEOS could then intersect a cell the domain fully covered into a line,
  leaving a gap of up to 0.2% of the domain area. The same slits made the
  MODFLOW-USG export fail with "Centres must lie on opposite sides of the
  shared face". The clip domain is now closed over gaps narrower than 1e-9
  x its extent, a cell the domain covers is kept whole, and any other cell
  lost with its generator inside or on the domain is named in a warning.
  Separately, a node on a vertex of a polygonised hole could lose its cell
  before clipping: Qhull returned near-coincident cell vertices out of
  order, the ring crossed itself, and the invalid cell was dropped without
  a warning (0.09% of the domain in one case). Such a ring is now replaced
  by its convex hull, and a mesh node left without a cell is warned about.
- A cell clipped around a sharp notch tip no longer passes twice through the
  tip. Clipping computed the tip twice, a roundoff apart, so the cell was two
  lobes joined by a neck of near-zero width: DISV listed one vertex twice,
  `grid_to_mfusg_disu` rejected the grid, and a barrier from the tip made the
  barrier safety net cut the cell. The lobe away from the generator now
  becomes a cell of its own, as other detached parts of clipped cells do.
- A barrier with `quad_buffer=True, quad_buffer_thickness=2` no longer puts
  a row of mesh nodes on the barrier. The cells of that row were cut through
  their generators, with no warning; `to_mfusg_disu()` then failed with an
  error that did not name the barrier, and `to_disv()` wrote faces split
  into two edges. On a barrier `quad_buffer_thickness` is now the number of
  cell rows on each side of the line, so `2` gives a strip three cells wide
  whose middle faces follow the line. Two changes keep those faces on the
  line: the two sides of a strip carry their nodes at matching stations,
  also around corners (each side used to spread its own count, which
  sheared the rows wherever the sides' lengths differed), and Gmsh's
  smoothing and optimization passes no longer move the node rows inside a
  quad buffer off their transfinite positions.
- A quad buffer whose line leaves the domain at a slant now runs to the
  boundary on both sides. One side used to stop short, and on a barrier the
  unstructured wedge left there could put nodes beside the line and leave
  cells it crossed. The overlay with the domain also no longer adds a
  vertex to the strip's end cap on the boundary, which made Gmsh give up
  the structured strip and mesh it unstructured.

## [0.3.0] - 2026-10-15

MODFLOW 6 DISV export (`to_disv()`, `to_modflowgwfdisv()`), `plot()` methods
with legend labels and line colours, `to_file()`, and a documentation site.
`MeshGenerator.generate()` now returns the generator instead of `True`.

### Added

- `VoronoiTessellator.to_disv()` and `vorflow.utils.grid_to_disv()` return
  the grid as MODFLOW 6 DISV arrays (`ncpl`, `nvert`, `vertices`, `cell2d`,
  zero-based as flopy takes them) plus the `node_id` of each `icell2d`
  ([#44](https://github.com/rhugman/vorflow/issues/44)). Cell centres are the
  generator points; vertices are clockwise and shared between neighbours, and
  a vertex lying on another cell's edge is inserted into that edge so
  MODFLOW 6 finds the connection. `to_modflowgwfdisv(gwf, top, botm, nlay=1)`
  builds the flopy package directly; flopy stays optional.
- `VoronoiTessellator.to_file()`: writes the grid with `GeoDataFrame.to_file`,
  passing keywords through.
- `plot()` methods on `ConceptualMesh` (registered features, or the cleaned
  ones with `clean=True`), `MeshGenerator` (the element grid) and
  `VoronoiTessellator` (the cells, optionally with their generators)
  ([#42](https://github.com/rhugman/vorflow/issues/42)). Each takes an
  optional `ax` and returns the axes. Matplotlib stays optional: it is
  imported only when `plot()` is called, and the new `plot` extra installs it.
- `label=` on `ConceptualMesh.add_polygon`, `add_line` and `add_point`:
  `ConceptualMesh.plot()` gives each label one legend entry, so
  `ax.legend()` needs no arguments
  ([#50](https://github.com/rhugman/vorflow/issues/50)). Features sharing a
  label share an entry; unlabelled features get none. Before, all lines (and
  all points) were drawn as one artist, so a hand-written legend could label
  only the first of them.
- `color=` on `ConceptualMesh.add_line`: the line's colour in `plot()`
  (default: barriers red, others blue).
- `examples/plotting_demo.ipynb`: the three `plot()` methods, legend labels
  and layering the conceptual model over the grid.
- `MeshGenerator(conceptual_mesh=...)`: `generate()` called without the clean
  frames runs `conceptual_mesh.generate()` itself, and
  `VoronoiTessellator(mesher)` takes the conceptual mesh from the generator
  ([#42](https://github.com/rhugman/vorflow/issues/42)). Passing the three
  clean frames and the conceptual mesh explicitly still works.
- `VoronoiTessellator.grid`: the grid from the last `generate()`, raising a
  `RuntimeError` before it has run. `final_grid` remains as before.

- Documentation site at <https://rhugman.github.io/vorflow/>
  ([#39](https://github.com/rhugman/vorflow/issues/39)): a Sphinx API
  reference built from the docstrings, the README's user guide, and the
  example notebooks, executed on every build. A `docs` extra installs the
  toolchain; `.github/workflows/docs.yml` builds it on pull requests and
  deploys it to GitHub Pages: `main` at the site root and `develop`
  under `/dev/`. The deprecated keywords of
  `add_polygon`, `add_line` and `add_point` are now listed in their own
  "Deprecated parameters" docstring section.
- `VoronoiTessellator(..., lloyd_size_smoothing=5)` smooths the per-node
  sizes behind the Lloyd density before relaxing
  ([#31](https://github.com/rhugman/vorflow/issues/31)). Gmsh's node sizes
  jitter by 7-12% (sd of log h) between neighbouring nodes once the intended
  grading is removed, i.e. +-30-46% in the density h^-4, and Lloyd converged
  to that noise. Each pass moves the log size of every free node halfway to
  the mean over its mesh-edge neighbours; fixed nodes keep their size and
  anchor the grading. On the #31 model (growth factor 1.2, 100 passes,
  `hex_ring`) this lowers the area-ratio p95 from 1.78 to 1.57, the p95
  `drift_ratio` from 0.105 to 0.070, the share of non-hexagonal cells from
  28.5% to 21.8% and the p95 centroid `ortho_error` from 2.4 to 1.25
  degrees, as well as the exact analytic size field does. 0 keeps the raw
  `node_sizes` and reproduces 0.2.0 bit for bit.
- `VoronoiTessellator(..., min_face_fraction=0.0)`: opt-in removal of very
  short cell faces before the grid is built
  ([#31](https://github.com/rhugman/vorflow/issues/31)). A face is short when
  it is below this fraction of the distance between its two generators; such
  faces come from four generators almost on one circle (typically a 5-7
  valence defect) and are more common after Lloyd. Each such quad is opened
  by moving its free generators slightly, with the same free nodes and move
  checks as Lloyd, for up to 20 passes; `short_face_report` holds the counts.
  On a two-well test model with 30 Lloyd passes, 0.2 takes the faces below
  0.2·sqrt(cell area) from 1.6% to 0% and the maximum face skewness from 5.3
  to 0.58, without changing the cell count or the share of hexagons.
- `MeshGenerator.node_edges`: the unique mesh edges between two of `nodes`,
  as (m, 2) positions into `nodes`, set by `generate()`. `node_sizes` is
  unchanged.

### Changed

- `VoronoiTessellator.export_to_shapefile()` is deprecated in favour of
  `to_file()`.
- `MeshGenerator.generate()` returns the mesh generator instead of `True`
  ([#42](https://github.com/rhugman/vorflow/issues/42)). It is still truthy,
  so `assert mesher.generate(...)` keeps working; code comparing the result
  with `is True` or `== True` needs updating.

- Documentation follow-ups from
  [#39](https://github.com/rhugman/vorflow/issues/39): the constructor
  docstrings of `ConceptualMesh`, `MeshGenerator` and `VoronoiTessellator`
  move to the class, `ThresholdField` and
  `VoronoiTessellator.export_to_shapefile` gain docstrings (so they appear in
  the API reference), the example notebooks log warnings only
  (`set_verbosity(0)`), and Basic Usage shows a plot of the grid.
  `examples/basic_usage.py --plot PATH` saves that figure.
- With `lloyd_iterations > 0` the grid differs from 0.2.0, because the Lloyd
  density now uses the smoothed sizes. Pass `lloyd_size_smoothing=0` for the
  0.2.0 grid. Without `hex_ring`, the cells of refined points can grow by
  up to about 60% towards the size the field asks for, because Gmsh makes
  the first ring of nodes around a point 5-15% finer than that.
- Lloyd guidance in the `lloyd_iterations` docstring, the README and
  `examples/point_centring_demo.ipynb`: Lloyd stays opt-in, mainly for models
  that use cell centroids as cell centres or for visually more regular cells.
  Use 100 or more passes with `hex_ring=True`, and avoid about 20, where the
  share of cells with very short faces peaks. Head accuracy depends mainly on
  `growth_factor`: in a Thiem test, 100 passes improved the head RMSE by at
  most about 5% at 5-10 times the build time.

### Fixed

- Embedded lines that meet at a T-junction or cross no longer put two mesh
  nodes a few metres apart at the junction
  ([#31](https://github.com/rhugman/vorflow/issues/31)). Each line was
  densified from its own ends, and Gmsh then added the junction node wherever
  it fell, 0 to `resolution` from the nearest line vertex. `ConceptualMesh.generate()`
  now, after clipping and before densification: snaps a line end that misses
  another line by less than `line_junction_fraction` x its vertex spacing
  (`densify` if numeric, else `resolution`) onto that line; trims an
  overshoot shorter than that at the crossing; makes each junction a vertex
  of every line through it, with one shared coordinate; and drops line
  vertices closer than that to a junction, so densification starts at the
  junction. Line ends and vertices on polygon boundaries are never moved,
  trimmed or dropped. On a 160 m background with two lines at resolution 30,
  the smallest node spacing goes from 6.0 to 18.4 m (line ending on the
  other), 7.5 to 17.3 m (ending 5 m short), 3.4 to 21.4 m (overshooting by
  5 m) and 2.6 to 17.7 m (crossing; 2.6 to 6.0 m at resolution 10). The new
  `ConceptualMesh(line_junction_fraction=0.5)` sets the threshold; 0 turns
  the step off. It is separate from `connectivity_tolerance`, which is an
  absolute distance; in this step it only sets what counts as touching.
  Barrier and straddle lines, quad-buffer lines and field-only lines are left
  out. An end is not snapped when the gap to the other line points more than 80
  degrees away from the line's own direction, so parallel lines closer than
  the threshold stay apart. `ConceptualMesh.junction_diagnostics` records each
  run's junctions (point and line IDs), snapped ends and trimmed ends. The
  parallel-line guard and the diagnostics follow
  [@d-strom](https://github.com/d-strom)'s
  [#35](https://github.com/rhugman/vorflow/pull/35).

## [0.2.0] - 2026-10-05

Centred point cells (`hex_ring`) and size-weighted Lloyd relaxation
(`lloyd_iterations`), both opt-in. Field-only polygons now refine their
whole interior, and the minimum Shapely version is 2.1.

### Added

- `ConceptualMesh.add_point(..., hex_ring=True)` adds six fixed mesh nodes at
  radius `resolution` around the point (at 30 + k x 60 degrees), so the
  point's Voronoi cell is a regular hexagon centred on it with apothem
  `resolution / 2` ([#31](https://github.com/rhugman/vorflow/issues/31)). On
  a 2 km model with four wells (resolution 5-10 m, `growth_factor=1.2`) the
  well cells' `drift_ratio` goes from 0.04-0.10 to 0 and the neighbour-area
  coefficient of variation from 0.17 to 0.04, for 42 extra cells per well.
  The ring is built after snapping and clipping, and dropped with a
  `UserWarning` when a polygon boundary, an embedded line (including its
  straddle or quad-buffer band) or another embedded point is closer than
  `HEX_RING_CLEARANCE` (2) x `resolution`; when a polygon (embedded or
  field-only, held at its resolution throughout its interior), line or point
  is estimated to set a mesh size below
  `HEX_RING_MIN_SIZE_RATIO` (0.9) x `resolution` at the ring, e.g. a well
  inside a zone with a finer resolution; or when `simplify_tolerance` merges
  the point into another. After meshing, `MeshGenerator.generate()` checks
  that each ring's centre node has exactly six triangles onto its six seeds,
  warns when a size field the estimate does not model (e.g. a
  `background_lc` below the ring radius) split it, and records the outcome
  in `diagnostics['hex_rings']` (`{point_id: bool}`). Requires `embed=True`
  and a positive `resolution`. The point's size field also targets the six
  seeds, so its refined patch is about one radius larger.
- `VoronoiTessellator(..., lloyd_iterations=0, lloyd_damping=1.0,
  lloyd_tolerance=1e-3)` runs a density-weighted Lloyd relaxation of the
  generators before the grid is built
  ([#31](https://github.com/rhugman/vorflow/issues/31),
  [#2](https://github.com/rhugman/vorflow/issues/2)). Centroids are weighted
  by h^-4, with h the local mesh size, so the mesh grading is kept; only
  interior nodes of the embedded polygon surfaces move, and moves that would
  leave the node's zone or cross an embedded line are rejected.
  `lloyd_tolerance` stops the run once the largest remaining distance from
  an accepted node to its centroid, relative to the local mesh size, is
  below it, independent of `lloyd_damping`. The mesh generator must have run
  before the tessellator is constructed. On the same
  model 20 passes lower the interior p95 `drift_ratio` from 0.118 to 0.099
  and the p95 centroid-to-centroid `ortho_error` from 4.1 to 2.7 degrees, and
  take 0.70 s against 0.33 s without. Unweighted Lloyd would grow the
  refined well cells 2-4 times in area. With Lloyd on, the Voronoi grid is
  no longer the exact dual of `MeshGenerator.get_element_grid()`.
- `MeshGenerator.node_is_free`, `node_sizes` (mean incident mesh-edge length
  per node) and `buffer_footprints` (union of the quad-buffer footprints),
  set by `generate()` as inputs to the Lloyd relaxation.
- A `lloyd_shift` grid column (distance each generator moved; 0 for fixed
  nodes, NaN for barrier mirrors, barrier fragments and detached cell parts)
  and `VoronoiTessellator.lloyd_report` when `lloyd_iterations > 0`: passes
  run (`iterations`), the last pass's largest residual |centroid - node| /
  local size over accepted nodes (`max_rel_shift`), moves rejected
  (`rejected`) and free generators (`n_free`).
- `examples/point_centring_demo.ipynb` compares Gmsh smoothing, `hex_ring`
  and `lloyd_iterations` on the model above.

### Changed

- Requires `shapely>=2.1` (was `>=2.0`) for
  `shapely.constrained_delaunay_triangles`, used by the field-only polygon
  interior field below.

### Fixed

- Field-only polygons (`add_polygon(..., embed=False)`) now refine their
  whole interior to `resolution`, as embedded polygons do. Before, their
  interior size was a Gmsh `Constant` field scoped (`SurfacesList`) to the
  polygon's own surface, which is not fragmented into the domain and so
  holds no domain mesh nodes; only the growth from the boundary took effect.
  The interior is now a `PostView` field over a triangulation of the polygon
  (holes excluded) and the rest of the model's bounding box, which applies by
  position. On a 200 x 200 domain (`background_lc=20`) with a 40 x 40 polygon
  at resolution 1, the 10 x 10 core gets 115 nodes, against 116 when embedded
  and 8 before. The `hex_ring` size check now measures field-only polygons as
  areas instead of from their boundary, so a ring inside a finer field-only
  polygon is dropped.
- The `MeshGenerator` docstring described `smoothing_steps` as Lloyd
  smoothing. It sets Gmsh `Mesh.Smoothing` (Laplacian smoothing of the
  triangle mesh), and `optimization_cycles` runs `Relocate2D` + `Laplace2D`
  passes; neither centres Voronoi generators in their cells. Raising them from
  the default 10/2 to 100/10 leaves the p95 centroid `ortho_error` at 4.1-4.2
  degrees.

## [0.1.0] - 2026-09-30

First release on PyPI. There are no changes since 0.1.0rc1.

## [0.1.0rc1] - 2026-09-30

### Fixed

- Kept conceptual-mesh inputs intact across repeated preprocessing calls.
- Made overlapping-zone tie-breaking deterministic.
- Honored the `snap_to_polygons=False` opt-out for line features.
- Preserved integer cell IDs when splitting cells along barrier lines.
- Kept quality reports usable with Gmsh 4.11 by retaining unsupported metrics as `NaN`.
- Restored Shapely 2.0 resampling plus stable lint and minimum-dependency CI.
- Kept barrier straddle points separate from point features with the same
  index; a point's size field no longer leaks onto an unrelated barrier.
- Enforced barriers wherever cells actually straddle them, including quad
  buffers with `quad_buffer_thickness=2` and cells at barrier ends.
- Field-only (`embed=False`) polygons no longer assign zones or change the
  clip domain, in both the Voronoi grid and the element grid.
- Kept `node_id` unique when clipping splits a cell into several parts.
- Cells whose generator sits on a slanted domain edge get the nearest zone
  instead of no `zone_id` (about 3% of cells on a simple pentagon domain).
- Polygon simplification no longer opens gaps along edges shared with
  neighbouring polygons.
- Point deduplication no longer depends on which point of a close pair has
  `simplify_tolerance`, and clean points keep their insertion order.
- Face skewness now reports the standard CVFD measure; the generator-mode
  value was always zero.
- `MeshGenerator(verbosity=...)` no longer changes the package-wide log
  level; the setting applies only while `generate()` runs.
- Custom `MeshField` subclasses with unhashable attributes can be grouped.
- With `heal_shapes=True`, surfaces or curves with identical bounding boxes
  (e.g. two triangles tiling a square) no longer swap feature ownership, which
  gave one zone's refinement to its neighbour. Entities are matched across
  `removeAllDuplicates`/`healShapes` by location within a tolerance, so near
  coincident points also resolve to the nearest survivor.
- Inset-mirror boundary centering skips nodes whose mirror ghost would land
  inside the domain, and is about 20x faster on large meshes.
- A standard line crossing a barrier (or straddle) line now ends exactly on a
  straddle pair placed at the crossing, instead of being trimmed back by the
  barrier corridor with its end nodes at an arbitrary offset from the nearest
  pair. The other pairs are spaced evenly between crossings and barrier ends.
  The pair stays perpendicular to the barrier and the line bends onto it; on
  an oblique crossing the line's nodes nearest the barrier are placed one
  cell apart and mirrored across it, so the barrier cells stay symmetric.
  Within 4 m of the river x fault crossing of the holed 200 x 200 example
  model the worst cell compactness rises from 0.68 to 0.74 and no barrier
  mirror cells are needed (4 before). A line T-junction that ends inside the barrier corridor
  also ends on a pair. Crossings within half a barrier cell of a barrier end
  or of another crossing keep the old trimming. Grids without such
  crossings are unchanged.
- Cells that straddle a barrier now get a mirror generator across the line
  instead of a centroid-centred fragment, so every face stays a Voronoi
  bisector and MODFLOW 6 connections stay orthogonal. On a fault crossed by a
  river (benchmark case F4) the linear-head L2 error drops from 3e-5 to 1e-10.
  The primary cell keeps its `node_id` and `x`/`y`; mirrors get fresh IDs and
  their count is `VoronoiTessellator.n_barrier_mirrors`. A mirror of a
  boundary node near a barrier end can lie just outside the domain. Nodes on
  the line itself still fall back to the post-hoc split.
- A curved barrier no longer leaves sliver cells along it. Straddle pairs'
  Voronoi faces are chords of the curve, and the post-hoc barrier split
  turned every bulge of the curve across a chord into a cell of its own
  (about 80 cells with compactness < 0.3 and areas of 1e-2 to 1e-15 on a
  sine barrier at `resolution=2`). A split piece smaller than 20% of its
  cell (`BARRIER_FRAGMENT_MERGE_FRACTION`) now joins the neighbouring cell on
  its side of the barrier, so the barrier face follows the line. Barrier
  mirrors within 10% of their reflection distance of an existing generator
  (`BARRIER_MIRROR_MERGE_FRACTION`, was 1e-3) are dropped, as are mirrors
  outside the domain whose piece would merge anyway; on curves tighter than
  `lc` both left cells of 1e-3 lc^2. A barrier split also no longer loses a
  piece (a hole in the grid) where the barrier runs along a cell face to
  roundoff.
- Barrier splits no longer leave zero-length edges where the line passes
  through (or within roundoff of) a cell vertex.
- `heal_shapes=True` no longer hangs Gmsh on lines that meet near a thin
  sliver polygon (`cleaning_limitations_demo`, Problem 4). After healing,
  curves and surfaces are matched to the nearest survivor within half their
  own extent, one-to-one, instead of keeping a surviving tag number that
  healing had reused for a different piece. Pieces shorter than the 1e-4
  tolerance no longer take a neighbour's or another line's entity, and pieces
  that healing deleted are pruned. A line fragment whose endpoint lies on a
  surface boundary without being a vertex of that surface (healing can
  duplicate a hole's edges instead of sharing them) is no longer embedded; a
  warning names it and `diagnostics['embedding']['nonconforming_skip']`
  counts it.
- Polygons with holes are no longer built as inverted OCC faces. Gmsh's
  `addPlaneSurface` expects hole loops wound like the exterior, but Shapely
  overlay output (overlap resolution, clipping) winds them the other way, so
  every holed polygon became a face of area shell + holes. `isInside()` was
  wrong on it, and points, lines and barrier straddle points inside it were
  silently left unembedded. Gmsh still meshed them as free entities, so their
  nodes made sliver cells along lines (the comprehensive demo lost 154
  embeds). A line crossing a holed polygon also no longer fails to split it.
  Meshes are unchanged where no clean polygon has a hole; a zone cut out of
  the domain by overlap resolution counts as one (the basic example goes from
  2746 to 2744 nodes).
- `removeAllDuplicates` renumbers curves and surfaces as well as points, and
  reuses freed tags for other entities. The fragment map now matches every
  entry by location after it, not only killed point tags, so renumbered line
  pieces keep their embedding and size fields.
- Points and line fragments that no domain surface contains are now reported
  with a warning, and listed in `diagnostics['embedding']['unmatched_tags']`,
  instead of being skipped silently.
- Barrier and straddle lines that end on a domain or hole boundary at an
  oblique angle no longer put one straddle point of the end pair outside the
  domain, where no surface embedded it and its node was not a triangle
  vertex. The end pair now slides inward along the line until its outer point
  lies on the boundary, so the pair's bisector still runs along the line to
  the boundary. Interior pairs it comes within half a spacing of, and any
  other straddle point outside the domain, are dropped. On benchmark case
  v4_barrier (30 degrees to the grid axes) this removes the two unmatched
  points and both barrier mirrors, and each end has two cells of 10 and 18
  m² instead of about 2 m². Pairs at perpendicular ends are unchanged, as are
  the example notebooks' grids. Below about 24 degrees between line and
  boundary, a boundary node still lies nearer the end than the slid pair, and
  its cell is split with a mirror.
- With pandas 1.5 / GeoPandas 0.13, clipping to the domain no longer puts
  cell geometries on the wrong rows, which gave most cells another
  generator's `node_id` and `x`/`y`. The grid is clipped without `gpd.clip`
  and keeps the Voronoi row order. Barrier fragment merging also works with
  Shapely 2.0, whose `STRtree` made the cell array read-only.
- Merging close cell vertices no longer aborts the whole grid with a GEOS
  `Invalid number of points in LinearRing` error when every vertex of one
  ring (a tiny cell, or a tiny hole in one) falls within the merge
  tolerance. That cell now keeps its unmerged geometry and is counted in the
  "Kept N cells unmerged" warning, as a cell that merging would make invalid
  already was.
- Enforcing a barrier no longer crashes with a GEOS `Invalid number of points
  in LinearRing` error when a piece split off a cell, or a hole in it, is
  smaller than the vertex-snapping tolerance. Such a piece now keeps its
  unsnapped geometry, as a piece that snapping would make invalid already
  did.
- A quad-buffer footprint whose corner touches a zone's outline no longer
  drops the zone surface. The difference left a ring that passed through the
  touch point twice, a few ulp apart; OCC merged the copies and could not
  close the curve loop, so most of the domain went unmeshed with only log
  warnings. Such rings are now rebuilt as a shell with a touching hole, or as
  separate polygons.

### Added

- Voronoi and triangular/mixed-element grid generation for MODFLOW 6 workflows.
- Mesh-quality and connectivity diagnostics.
- Optional boundary inset/mirror points and structured quad buffers.
- Explicit mesh-size growth fields and runnable examples.
- Cross-platform tests and TestPyPI release automation.
- `vorflow.set_verbosity(level, console=False)` routes messages to the
  application's logging configuration instead of vorflow's console handler.

### Changed

- Prepared project metadata, installation documentation, and dependency floors
  for the first public release candidate.
- Features finer than `background_lc` now grade outward with a
  `GeometricGrowthField` (growth factor 1.2) by default. Models that relied on
  the old implicit sizing will generally get more, better-graded cells.
- Progress output uses the `vorflow` logger (stderr) instead of `print()`.
- The mesh-size field setup's index-matching lines are `[DIAG]` output
  (verbosity 2) instead of printing at the default verbosity.
- `ConceptualMesh(crs=...)` defaults to `None` instead of `"EPSG:4326"`;
  geographic CRSs trigger a warning.
- `MeshGenerator.generate()` raises before starting Gmsh if `background_lc`
  is missing or not positive.
- `get_element_grid()` builds element polygons on first use instead of in
  every `generate()` call.
- `quad_buffer=True` requires `embed=True`.
- Barrier straddle offsets use a tangent probe proportional to line length,
  which can move barrier nodes by floating-point amounts.
- Python 3.10 is now the minimum; matplotlib is optional (`examples` extra).
- `VoronoiTessellator(boundary_inset_fraction=...)` now defaults to 0.25
  instead of 0.5. At 0.5, `boundary_centering="inset_mirror"` moved boundary
  nodes past the point where their cells are centred on Gmsh meshes and made
  centroid-to-centroid boundary orthogonality worse than `"clip"` (median
  ortho_error 12.0 vs 6.4 degrees on a 200 x 200 box at `background_lc=20`).
  At 0.25 the median is 1.4 degrees there, and 1.3 vs 6.9 degrees on the
  comprehensive demo model. Pass `boundary_inset_fraction=0.5` for the old
  behaviour.
- `MeshGenerator.generate()` raises `RuntimeError` when an embedded polygon or
  quad-buffer piece wider than a sliver cannot become an OCC surface, instead
  of logging a warning and meshing around the hole.

### Deprecated

- `add_polygon(mesh_refinement=...)` (no effect), `dist_max_out` (use
  `dist_max`), `border_density` (use `densify`; the border grading is kept),
  and `dist_max_in`.
- `dist_min`/`dist_max` on features; use `growth_factor` or explicit `fields`.

[Unreleased]: https://github.com/rhugman/vorflow/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/rhugman/vorflow/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rhugman/vorflow/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rhugman/vorflow/compare/v0.1.0rc1...v0.1.0
[0.1.0rc1]: https://github.com/rhugman/vorflow/tree/v0.1.0rc1
