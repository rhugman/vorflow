"""vorflow adapter: case -> ConceptualMesh -> MeshGenerator -> VoronoiTessellator.

Spec translation: ``resolution = h_f``, ``growth_factor = g``,
``background_lc = h_max``. The domain polygon uses the background size.
Barrier lines keep ``is_barrier=True`` (vorflow is the only tool with barriers).

vorflow has no DISV writer, so the grid goes through FloPy's ``to_cvfd``.
Written centres are the generator points (``x``, ``y``). Barrier mirror
generators (reflections of nodes whose cell straddles a barrier) are true
generators too; their count is ``info['n_barrier_mirrors']``. Cells still split
by a barrier or exploded after clipping carry their centroid instead; their
count (cells minus mesh nodes minus mirrors) is ``info['n_split_cells']``.
"""

import time
from pathlib import Path

from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator, set_verbosity

from ..case import Case, line_parts
from ..grid import Grid, grid_from_polygons

TOOL = "vorflow"


def build(case: Case, scale: float, ws: Path, tessellator_kwargs: dict | None = None,
          tool: str = TOOL) -> Grid:
    """Generate the vorflow grid for one case at one spec scale.

    ``tessellator_kwargs`` go to ``VoronoiTessellator`` (e.g. Lloyd settings);
    ``tool`` names the grid, so variants are reported as separate tools.
    """
    set_verbosity(0)
    t0 = time.perf_counter()
    blueprint = _blueprint(case, scale)
    clean_polys, clean_lines, clean_pts = blueprint.generate()
    mesher = MeshGenerator(background_lc=case.h_max * scale, verbosity=0)
    mesher.generate(clean_polys, clean_lines, clean_pts)
    tessellator = VoronoiTessellator(mesher, blueprint, clip_to_boundary=True, **(tessellator_kwargs or {}))
    cells = tessellator.generate()
    t1 = time.perf_counter()
    grid = _to_grid(cells, tool)
    grid.timings = {"mesh_s": t1 - t0, "export_s": time.perf_counter() - t1}
    n_mirrors = getattr(tessellator, "n_barrier_mirrors", 0)
    grid.info["n_barrier_mirrors"] = n_mirrors
    grid.info["n_split_cells"] = len(cells) - len(mesher.nodes) - n_mirrors
    return grid


def _blueprint(case: Case, scale: float) -> ConceptualMesh:
    """ConceptualMesh holding the case domain and features."""
    g = case.growth
    blueprint = ConceptualMesh(crs=case.crs)
    blueprint.add_polygon(case.domain, zone_id="domain", growth_factor=g)
    for f in case.features_of("polygon"):
        blueprint.add_polygon(f.geometry, zone_id=f.id, resolution=f.h * scale, z_order=1, growth_factor=g)
    for f in case.features_of("line"):
        for i, part in enumerate(line_parts(f.geometry)):
            blueprint.add_line(part, line_id=f"{f.id}-{i}", resolution=f.h * scale,
                               is_barrier=f.barrier, growth_factor=g)
    for f in case.features_of("point"):
        blueprint.add_point(f.geometry, point_id=f.id, resolution=f.h * scale, growth_factor=g)
    return blueprint


def _to_grid(cells, tool: str = TOOL) -> Grid:
    """Grid from the tessellator output; x, y are the written centres."""
    xy = cells[["x", "y"]].to_numpy()
    return grid_from_polygons(tool, list(cells.geometry), xc=xy, generators=xy)
