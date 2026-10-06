# Changelog

All notable changes to `vorflow` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
- `MeshGenerator.node_edges`: the unique mesh edges between two of `nodes`,
  as (m, 2) positions into `nodes`, set by `generate()`. `node_sizes` is
  unchanged.

### Changed

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

[Unreleased]: https://github.com/rhugman/vorflow/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rhugman/vorflow/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rhugman/vorflow/compare/v0.1.0rc1...v0.1.0
[0.1.0rc1]: https://github.com/rhugman/vorflow/tree/v0.1.0rc1
