"""vorflow with Lloyd relaxation and short-face cleanup (rhugman/vorflow#36, #37).

Same spec translation as ``vorflow_adapter``; the tessellator additionally
runs ``lloyd_iterations=100`` density-weighted Lloyd passes and
``min_face_fraction=0.2``. Written centres are still the (moved) generators,
so the grid stays orthogonal as MF6 reads it. Reported as its own tool so it
can be compared with plain vorflow and VOROGRIDGEN at matched cell counts.
"""

from pathlib import Path

from ..case import Case
from ..grid import Grid
from . import vorflow_adapter

TOOL = "vorflow_lloyd"
TESSELLATOR_KWARGS = {"lloyd_iterations": 100, "min_face_fraction": 0.2}


def build(case: Case, scale: float, ws: Path) -> Grid:
    """Generate the relaxed vorflow grid for one case at one spec scale."""
    grid = vorflow_adapter.build(case, scale, ws, tessellator_kwargs=TESSELLATOR_KWARGS, tool=TOOL)
    grid.info.update(TESSELLATOR_KWARGS)
    return grid
