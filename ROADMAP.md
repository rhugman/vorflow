# vorflow Roadmap - Porting robust capabilities from `gmshflow`

This roadmap captures the agreed plan for bringing the strongest features of the
pre-refactor `gmshflow` library into `vorflow`, while keeping `vorflow`'s current
defaults backward-compatible and adding robust paths as opt-in options.

Each milestone has a self-contained document under [`docs/roadmap/`](docs/roadmap/)
with goals, files to touch, source references, reuse notes, and verification
steps.

## Context

`gmshflow` (`../gmshflow/src/gmshflow.py`, a single module on the gmsh geo
kernel) was the version that existed before the refactor into `vorflow`.
`vorflow` (`src/vorflow/`) is modular and more robust: it runs on the gmsh OCC
kernel with `fragment` / `removeAllDuplicates` / `healShapes`, has a better field
system, and tracks generator points through the pipeline so Voronoi quality can
be computed correctly.

However, a handful of `gmshflow` capabilities are either missing from `vorflow`
or useful enough to port.

## Decisions

- Keep both the lightweight straddle path and the robust structured-quad buffer
  path. Structured buffers are opt-in and do not replace straddle.
- Keep the current Voronoi clipping behavior as the default.
- Do not add a core sliver-filtering or active-domain API for now. Larger
  mesh-envelope / smaller active-domain workflows should select or tag active
  cells externally with GeoPandas after grid generation.
- Add triangular/2D element quality, Voronoi/MODFLOW-facing connectivity reports,
  CVFD skewness, and centroid-based quality diagnostics.
- Avoid new hard dependencies. Skip `topojson`; document external preprocessing
  recipes instead.
- Leave `calculate_orthogonality` backward-compatible. New connectivity reports
  can expose both Voronoi-dual generator connectivity and MODFLOW-facing centroid
  connectivity.

## Out of Scope

- Outer-domain construction (`prepare_mesh_domain`): users can construct convex
  hulls, square extents, or buffered envelopes with Shapely/GeoPandas.
- Topology-preserving simplification with `topojson`: avoid the dependency.
- A general `merge_many_multilinestring_into_one_linestring` helper: add only a
  localized repair if structured buffers actually encounter offset-curve splits.

## Current Objective

No milestone is in progress. Follow-ups to the first release (CI on newer
Python versions, a docs site, a citation/DOI, a conda-forge feedstock) are
listed under Phase 3 in
[06-pypi-publishing.md](docs/roadmap/06-pypi-publishing.md).

## Completed Milestones

| # | Milestone | Status | Doc |
|---|-----------|--------|-----|
| 1 | Quality metrics | Done | [01-quality-metrics.md](docs/roadmap/01-quality-metrics.md) |
| 2 | Active-domain filtering workflow | Done (example) | [02-robust-clipping.md](docs/roadmap/02-robust-clipping.md) |
| 3 | Boundary inset/mirror points | Done | [03-boundary-mirror-points.md](docs/roadmap/03-boundary-mirror-points.md) |
| 4 | Structured-quad transfinite buffer | Done | [04-structured-quad-buffer.md](docs/roadmap/04-structured-quad-buffer.md) |
| 5 | Triangular/mixed element-grid output | Done | [05-triangular-grid-output.md](docs/roadmap/05-triangular-grid-output.md) |
| 6 | First PyPI release | Done (`0.1.0`, 2026-09-30) | [06-pypi-publishing.md](docs/roadmap/06-pypi-publishing.md) |

All six milestones are complete: quality metrics and connectivity reports
(`get_triangular_quality`, `utils.build_connectivity`), the active-domain
workflow example (`examples/active_domain_quality_example.py`), opt-in boundary
inset/mirror points (`VoronoiTessellator(boundary_centering="inset_mirror")`),
opt-in structured quad buffers (`add_polygon`/`add_line` with
`quad_buffer=True`), and the element-grid exporter
(`MeshGenerator.get_element_grid()`, see
`examples/triangular_grid_example.py` and
`examples/structured_buffer_example.py`). `vorflow` 0.1.0 is published on
[PyPI](https://pypi.org/project/vorflow/).

## Summary per Milestone

1. **Quality metrics**: add cached gmsh 2D element quality, generator/centroid
   connectivity reports, skewness, tests, and example diagnostics.
2. **Active-domain filtering workflow**: no core API. Add an example showing how
   to mesh a larger envelope, select active cells externally, and compare quality
   against direct clipping.
3. **Boundary inset/mirror points**: add an opt-in Voronoi construction mode that
   improves boundary-cell center placement and boundary connectivity quality.
   On Gmsh meshes the default inset (0.25 x boundary spacing) lowers the median
   boundary ortho_error from about 6 to under 2 degrees; cells next to sharp
   corners do not improve. See the milestone doc for measurements.
4. **Structured-quad transfinite buffer**: add opt-in robust feature alignment for
   line and polygon features in the OCC pipeline, keeping straddle as the default.
5. **Triangular/mixed element-grid output**: add a separate `MeshGenerator`
   exporter for gmsh element polygons, supporting triangles now and mixed tri/quad
   meshes for structured buffers.

## Verification

- Per milestone: add focused tests under `tests/`.
- End-to-end: keep `examples/field_capabilities_example.py` for field and quality
  diagnostics, and add `examples/active_domain_quality_example.py` for the larger
  mesh-envelope workflow.
- Backward compatibility: with new options disabled, current Voronoi output and
  public APIs remain unchanged.
