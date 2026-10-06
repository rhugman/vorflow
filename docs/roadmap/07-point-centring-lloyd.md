# Milestone 7 — Centred point cells and Lloyd relaxation (opt-in)

**Status:** done · **Risk:** medium · **Behavior change:** none by default
**Back to** [ROADMAP.md](../../ROADMAP.md)

## Outcome

Implemented as three changes, prompted by
[#31](https://github.com/rhugman/vorflow/issues/31) (point cells not centred
or symmetric around the point) and
[#2](https://github.com/rhugman/vorflow/issues/2) (Lloyd iterations):

- The `MeshGenerator` docstring now says what `smoothing_steps` and
  `optimization_cycles` do: Gmsh `Mesh.Smoothing` (Laplacian smoothing of the
  triangle mesh) and `Relocate2D` + `Laplace2D` passes. It called the first
  "Lloyd smoothing", which is why #2 was closed.
- `ConceptualMesh.add_point(..., hex_ring=True)` adds six fixed mesh nodes at
  radius `resolution` (angles 30 + k x 60 degrees). The point and two
  adjacent seeds form equilateral triangles, so the point's Voronoi cell is a
  regular hexagon centred on it with apothem `resolution / 2`. The ring is
  built about the point's final position (after snapping and clipping) and
  dropped with a `UserWarning` when:
  - a polygon boundary, an embedded line (plus its straddle or quad-buffer
    half-width) or another embedded point is closer than
    `HEX_RING_CLEARANCE` (2.0) x `resolution`;
  - a polygon (embedded or field-only, held at its resolution throughout
    its interior), line or point is estimated to set a
    mesh size below `HEX_RING_MIN_SIZE_RATIO` (0.9) x `resolution` at the
    ring. With a uniform size, Gmsh kept the ring at 0.85 r and split it at
    0.80 r; 0.9 leaves a margin for the linear size estimate, which models
    the default growth fields and explicit `GeometricGrowthField` /
    `ThresholdField` entries;
  - `simplify_tolerance` merges the point into another.

  After meshing, `MeshGenerator.generate()` checks that each ring's centre
  node has exactly six triangles onto its six seeds. It warns when the ring
  was split by a size field the estimate does not model (other explicit
  fields, a `background_lc` below the ring radius) and records
  `diagnostics['hex_rings'] = {point_id: bool}`. The seeds are recorded
  under the point's feature id, so embedding, node collection and the size
  field treat them like the point.
- `VoronoiTessellator(..., lloyd_iterations=0, lloyd_damping=1.0,
  lloyd_tolerance=1e-3)` relaxes the free generators towards their
  density-weighted cell centroids before the grid is built
  (`src/vorflow/_lloyd.py`). The density is rho = h^-4, with h interpolated
  from `MeshGenerator.node_sizes` (mean incident mesh-edge length), because
  the 2D CVT spacing scales as rho^(-1/4). Each cell's weighted centroid uses
  a 3-point rule on each triangle of a fan from its generator. Only interior
  nodes of embedded polygon surfaces move (`MeshGenerator.node_is_free`); a
  move is rejected if it leaves the node's zone piece (zone minus
  `MeshGenerator.buffer_footprints`) or crosses an embedded line. The grid
  gets a `lloyd_shift` column and the tessellator a `lloyd_report`. The stop
  test compares `lloyd_tolerance` with the residual of the last pass, the
  largest |centroid - node| / h over the accepted nodes
  (`lloyd_report['max_rel_shift']`), so it does not depend on
  `lloyd_damping`. The tessellator reads `node_is_free`, `node_sizes` and
  `buffer_footprints` when it is constructed, so the mesh generator must
  have run first.
- Follow-up (after 0.2.0): Gmsh's `node_sizes` jitter by 7-12% between
  neighbouring nodes, and Lloyd converged to that noise. The density now
  uses sizes smoothed in log space over `MeshGenerator.node_edges`
  (`lloyd_size_smoothing`, default 5 passes, fixed nodes held). The
  measurements below predate this change; see the
  [#31](https://github.com/rhugman/vorflow/issues/31) experiment and
  `examples/point_centring_demo.ipynb` for current numbers.

Unweighted Lloyd was tried first. On a 2 km square with points at
`resolution` 5 and 10, 20 passes halved the interior p95 `drift_ratio`
(0.111 to 0.056) but grew the 5 m point's cell from 23.6 to 68.5 m², so the
weighting is required.

Measurements on a 2 x 2 km domain with an upper zone whose lower edge is
irregular and four wells at `resolution` 5, 10, 7 and 5 m
(`growth_factor=1.2`, `background_lc=100`). Interior cells do not touch the
domain boundary; well values are means over the four wells; `ortho_error` is
the p95 of `build_connectivity(grid, center="centroid")`; runtime is meshing
plus tessellation. The same cases are in
`examples/point_centring_demo.ipynb`.

| Case | Cells | Runtime (s) | Well drift | Well neighbour-area CV | Interior drift median | Interior drift p95 | p95 ortho_error (deg) | Well cell area (m²) |
|------|-------|-------------|------------|------------------------|-----------------------|--------------------|-----------------------|---------------------|
| defaults (smoothing 10/2) | 1673 | 0.33 | 0.077 | 0.171 | 0.060 | 0.118 | 4.12 | 46.6 |
| smoothing 0/0 | 1673 | 0.12 | 0.075 | 0.188 | 0.062 | 0.136 | 5.67 | 49.6 |
| smoothing 100/10 | 1673 | 1.80 | 0.082 | 0.174 | 0.060 | 0.117 | 4.19 | 48.2 |
| `hex_ring` | 1842 | 0.33 | 0.000 | 0.044 | 0.058 | 0.122 | 4.12 | 43.1 |
| Lloyd 5 | 1673 | 0.40 | 0.052 | 0.139 | 0.054 | 0.100 | 2.67 | 48.2 |
| Lloyd 20 | 1673 | 0.70 | 0.046 | 0.134 | 0.053 | 0.099 | 2.70 | 49.4 |
| `hex_ring` + Lloyd 20 | 1842 | 0.77 | 0.000 | 0.036 | 0.053 | 0.104 | 2.73 | 43.1 |

With `hex_ring` each well cell has exactly the area (sqrt(3)/2) x
`resolution`², six neighbours and zero drift. Most of the Lloyd gain comes in
the first five passes; the p95 values then stay within 0.1 degree and 0.001.
Unweighted Lloyd (constant `node_sizes`) on the same mesh grows the well cells
2-4 times in area; weighted Lloyd keeps them within 20% of the default.

Limits:

- `hex_ring` adds about 40 cells per point at `growth_factor=1.2` (42 here),
  and the refined patch grows by about one ring radius because the size
  field also targets the seeds.
- `hex_ring` is for point features only and its radius is tied to
  `resolution`. It is dropped near other features and where a finer size
  field reaches the ring, e.g. for a well inside a zone whose resolution is
  finer than 0.9 x the well's.
- The size estimate is linear and covers only the default growth fields and
  explicit `GeometricGrowthField` / `ThresholdField` entries; other fields
  are caught only after meshing (warning plus `diagnostics['hex_rings']`),
  when the ring has already been meshed through.
- Field-only (`embed=False`) polygons originally refined only from their
  boundary: their interior Constant field was scoped to a surface entity that
  holds no domain mesh nodes. That was fixed separately (a positional
  PostView interior field), and the size estimate now measures them as areas.
- With Lloyd on, the Voronoi grid is no longer the exact dual of
  `MeshGenerator.get_element_grid()`.
- Cells next to fixed nodes improve little. The p95 `ortho_error` tail above
  about 6 degrees is almost all domain-boundary cells (143 of 151 such
  connections at defaults); `boundary_centering="inset_mirror"` addresses
  those. The well's own cell is not centred by Lloyd, because the well node
  is fixed.
- On graded meshes the residual levels off at a few 1e-3 (0.006 after 20
  passes, 0.002 after 40), so the default `lloyd_tolerance` of 1e-3 is
  rarely reached and the run uses every pass.
- Zone edges and embedded lines can reject moves; a rejected node stays put
  for that pass.

Tests:

- `tests/test_conceptual_mesh.py` (`test_hex_ring_*`): argument validation,
  seed geometry, dropping near lines, other points and the domain boundary,
  field-only lines ignored for clearance, the size rule, the deduplication
  warning, empty-frame columns.
- `tests/test_point_tracking.py::TestHexRing`: seeds mapped under the point's
  feature id, the point cell is a regular hexagon end to end, a dropped ring
  still meshes, the post-mesh ring check and `diagnostics['hex_rings']`.
- `tests/test_lloyd.py`: unit tests of `_lloyd` (lattice fixed points,
  density-weighted centroids, clipped boundary cells, move rejection, early
  stop independent of damping, rejected nodes not blocking convergence,
  shared settings validation, `node_sizes` from elements, inputs captured
  with the nodes) and slow end-to-end
  tests (interior drift drops, fixed nodes bit-identical, zones kept, refined
  well cells kept, output unchanged with Lloyd off, `inset_mirror`, barrier
  sides, quad-buffer strips, hex ring kept regular).

The rest of this document is the original plan.

## Goal

Give point features (wells, observation points) centred, symmetric cells on
request, and offer a grid-wide generator-centring option, without changing
default output or adding dependencies.

## Why

A vorflow cell's generator is a Gmsh node. A point feature is always the
generator of its own cell, but nothing makes that generator the cell's
centroid, and the shape of the cell depends on where the frontal-Delaunay
mesher puts the neighbouring nodes. Prototype measurements on a 2 km square
with points at h = 5 and h = 10:

- Gmsh smoothing barely changes centroidality: median drift 0.054 with
  smoothing off, 0.052 with 100 steps.
- A hexagon of six fixed seeds at radius h gives a centred, regular cell:
  drift 0.000, neighbour-area CV 0.02.
- Unweighted Lloyd halves the interior p95 drift (0.111 to 0.056) but grows
  the h = 5 cell from 23.6 to 68.5 m², so Lloyd must be weighted by the size
  field.

## Files to touch

- `src/vorflow/engine.py` — correct the `smoothing_steps` /
  `optimization_cycles` docstring; add hex-ring seeds in
  `_add_point_features`; capture `node_is_free`, `node_sizes` and
  `buffer_footprints` in `generate()`.
- `src/vorflow/blueprint.py` — `add_point(hex_ring=False)`, seed placement
  and the clearance check after snapping and clipping, `ring_seeds` column.
- `src/vorflow/_lloyd.py` (new) — size interpolation, weighted centroids,
  move acceptance, the relaxation loop.
- `src/vorflow/tessellator.py` — `lloyd_iterations`, `lloyd_damping`,
  `lloyd_tolerance`; call the relaxation before boundary centring.
- `tests/test_conceptual_mesh.py`, `tests/test_point_tracking.py`,
  `tests/test_lloyd.py`.
- `examples/point_centring_demo.ipynb`, README, CHANGELOG, this document.

## Detail

### Smoothing documentation

`smoothing_steps` maps to Gmsh `Mesh.Smoothing`, Laplacian smoothing of the
triangle mesh; `optimization_cycles` runs `Relocate2D` + `Laplace2D`. Neither
centres Voronoi generators; point to `VoronoiTessellator(lloyd_iterations=...)`.

### `hex_ring`

- `add_point(..., hex_ring=False)`; reject non-bool values, `embed=False` and
  a missing resolution.
- After snapping and clipping, compute six seeds at radius r = `lc` and
  angles 30 + k x 60 degrees about the final point position.
- Drop the ring with a `UserWarning` naming the point and the reason when a
  domain or polygon boundary, a line (barriers, straddle lines and quad
  buffers included) or another point is closer than
  `HEX_RING_CLEARANCE * r` (`HEX_RING_CLEARANCE = 2.0`).
- Store the seeds in a `ring_seeds` object column. In the engine, add each
  seed as an OCC point recorded under the parent point's feature id, so
  embedding, node collection and field targeting pick it up unchanged. The
  distance field then also measures from the seeds, so the refined patch
  grows by about r.

### Lloyd relaxation weighted by the size field

- While Gmsh is live, record `node_is_free` (nodes of embedded polygon
  surfaces from `getNodes(2, surf, includeBoundary=False)`, which leaves out
  nodes on curves and points; structured-buffer surfaces left out) and
  `node_sizes` (mean incident edge length from the captured element data,
  standing in for the size field, which Gmsh cannot evaluate after shutdown).
- `_lloyd.py`: `size_interpolator` (linear inside the hull, nearest outside),
  `weighted_centroids` (fan triangulation of each convex Voronoi region, rho =
  h^-4; cells reaching outside the domain use the plain centroid of the
  clipped cell), `accept_moves` (new point inside the owner polygon, old-new
  segment crossing no constraint line), `relax` (damped loop with early stop
  on max |shift| / h < tolerance).
- Tessellator: validate the new kwargs; relax between the node-count check
  and boundary centring, so `inset_mirror`, barrier mirrors and far ghosts
  work on the relaxed nodes; raise `ValueError` if the mesh generator has no
  `node_is_free` / `node_sizes`. Add `lloyd_shift` and `lloyd_report`.

### Demo notebook

`examples/point_centring_demo.ipynb`: the issue-like case at defaults, the
smoothing sweep, `hex_ring` (diagram, zooms, dropped-ring warning, pros and
cons), Lloyd (fixed and free nodes, convergence, histograms, `lloyd_shift`
map, grading against unweighted Lloyd, pros and cons), and a summary table
with recommendations. Committed without outputs.

## Verification

- `tests/test_conceptual_mesh.py`: `hex_ring` validation; `ring_seeds` holds
  six points at distance `lc`; the ring is dropped with a warning near a
  line, the boundary or another point; the empty points frame has the new
  columns.
- `tests/test_point_tracking.py`: end to end, the well cell has six vertices
  at r / sqrt(3) and area (sqrt(3)/2) r²; the point feature maps to seven
  Gmsh point tags.
- `tests/test_lloyd.py`: uniform density gives the plain centroid; a density
  gradient moves the centroid to the dense side; fixed nodes never move;
  moves across a barrier or out of the polygon are rejected; a regular hex
  lattice is a fixed point. End to end (slow): interior p95 drift drops, node
  count unchanged, fixed nodes bit-identical, the refined point's cell area
  within 25% of baseline, `zone_id` unchanged, no generator on the wrong side
  of a barrier, `ValueError` with a mesh generator lacking the inputs.
- `ruff check src tests scripts`, `pytest -m "not slow"`, full `pytest`,
  `examples/basic_usage.py`, `examples/field_capabilities_example.py`.
