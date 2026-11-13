"""Fully unstructured MODFLOW grids (DISU) from vorflow meshes.

The workflow follows FloPy conventions: give the 2D grid and the usual layer
arrays (``top``, ``botm``, ``idomain``), call :meth:`DisuGrid.build`, then
unpack :meth:`DisuGrid.get_gridprops_mf6_disu` into ``flopy.mf6.ModflowGwfdisu``,
as with ``flopy.utils.gridgen.Gridgen``. vorflow reads no rasters and writes
no model files.

Full connectivity (vertically staggered horizontal connections between cells
in different layers) follows the algorithm of C. D. Langevin's USGS
``disv2disu.py``, published with

    Provost, A.M., Bardot, K., Langevin, C.D. and McCallum, J.L. (2025).
    Accurate simulation of flow through dipping aquifers with MODFLOW 6 using
    enhanced cell connectivity. Groundwater 63(3), 399-408.
    https://doi.org/10.1111/gwat.13459 (code: https://doi.org/10.5066/P13BNARA)

and motivated by Bardot, K., Lesueur, M., Siade, A.J. and McCallum, J.L.
(2023), Revisiting MODFLOW's capability to model flow through sedimentary
structures, Groundwater 61, 663-673 (https://doi.org/10.1111/gwat.13273).
Vertical pass-through of pinched cells follows LoopFlopy
(https://github.com/kerrybardot/loopflopy; Bardot et al. 2025, Environmental
Modelling & Software 192, 106557). This module is an independent, vectorised
implementation with its own pinch-out and sliver handling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import _columns
from ._columns import ACTIVE, STATUS_NAMES
from ._connect3d import concat_links, layered_links, staggered_links, vertical_links
from ._disu_export import mf6_gridprops

__all__ = ["DisuGrid", "DisuOptions"]

CONNECTIVITY = ("full", "layered")
CL12_MODES = ("plan", "connector")


@dataclass(frozen=True)
class DisuOptions:
    """Validated :class:`DisuGrid` settings; see the class docstring for their meaning."""

    min_thickness: float = 0.0
    collapse: str | None = "split"
    connectivity: str = "full"
    sliver_abs_tol: float = 0.0
    cl12: str = "plan"

    def __post_init__(self):
        for name in ("min_thickness", "sliver_abs_tol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}.")
        if self.collapse not in _columns.COLLAPSE_MODES:
            raise ValueError(f"collapse must be one of {_columns.COLLAPSE_MODES}, got {self.collapse!r}.")
        if self.connectivity not in CONNECTIVITY:
            raise ValueError(f"connectivity must be one of {CONNECTIVITY}, got {self.connectivity!r}.")
        if self.cl12 not in CL12_MODES:
            raise ValueError(f"cl12 must be one of {CL12_MODES}, got {self.cl12!r}.")


class DisuGrid:
    """A DISU grid built from a vorflow 2D grid and FloPy-style layer arrays.

    Args:
        grid: The vorflow grid (GeoDataFrame with ``x``/``y`` generator
            columns), used by every layer.
        top: Model top, scalar or one value per cell.
        botm: Layer bottoms, ``(nlay,)`` or ``(nlay, ncpl)``.
        idomain: ``None`` (all active) or ``(nlay, ncpl)``: >0 active,
            0 no-flow (never passed through), -1 pass-through.
        min_thickness: Cells this thin or thinner are treated as pass-through.
        collapse: What happens to the interval of pass-through cells:
            ``"split"`` (default) shares it between the active cells above and
            below, ``"up"``/``"down"`` gives it to one side, and ``None`` keeps
            the gap, as MODFLOW 6 and LoopFlopy do. With a gap-free column the
            vertical connection lengths are exact and no thickness is lost.
        connectivity: ``"full"`` adds vertically staggered connections between
            offset neighbours (IHC=2); ``"layered"`` reproduces DISV
            connectivity (same-layer neighbours only).
        sliver_abs_tol: Optional merging of small pieces, off by default
            (0). A piece of a side face thinner than ``sliver_abs_tol`` is
            given to the pieces above and below it. Merging gives fewer
            connections but moves face area between cells, so results are no
            longer exact; the default keeps every piece, and MODFLOW handles
            tiny ones fine.
        cl12: ``"plan"`` (perpendicular plan distance to the face, as in the
            paper; exact for XT3D) or ``"connector"``.

    After :meth:`build`, ``nodemap`` is an ``(nlay, ncpl)`` array mapping
    ``(layer, cell)`` to the DISU node (-1 for removed cells), and ``cellmap``
    describes every layer cell.
    """

    def __init__(self, grid, top, botm, idomain=None, *, min_thickness=0.0, collapse="split",
                 connectivity="full", sliver_abs_tol=0.0, cl12="plan"):
        self.options = DisuOptions(
            min_thickness=min_thickness, collapse=collapse, connectivity=connectivity,
            sliver_abs_tol=sliver_abs_tol, cl12=cl12,
        )
        missing = {"geometry", "x", "y"}.difference(getattr(grid, "columns", ()))
        if missing:
            raise ValueError(f"grid missing required columns: {sorted(missing)}.")
        if len(grid) == 0:
            raise ValueError("grid must not be empty.")
        self._grid = grid
        self._layers = _columns.layer_arrays(top, botm, idomain, ncpl=len(grid))
        self.nodemap = None
        self.cellmap = None
        self.links = None

    @property
    def nlay(self):
        return self._layers[1].shape[0]

    @property
    def grid(self):
        """The plan grid used by every layer."""
        return self._grid

    @property
    def nodes(self):
        self._require_build()
        return int((self.nodemap >= 0).sum())

    def build(self):
        """Classify cells, close pass-through gaps and connect the active cells. Returns ``self``."""
        from ._mfusg import _planar_faces

        opts = self.options
        self._planar, self._area, self._faces, _ = _planar_faces(self._grid)
        top_l, bot_l, idomain = self._layers
        self.top_l, self.bot_l = top_l, bot_l
        status = _columns.classify(top_l, bot_l, idomain, opts.min_thickness)
        top_l, bot_l = _columns.collapse(top_l, bot_l, status, opts.collapse)
        active = status == ACTIVE
        nodemap = np.full(status.shape, -1, dtype=np.int64)
        nodemap[active] = np.arange(int(active.sum()))
        self.status = status
        self._node_top, self._node_bot = top_l[active], bot_l[active]
        self._node_layer, self._node_cell = np.nonzero(active)

        self.diagnostics = {"slivers_merged": 0, "largest_sliver": 0.0}
        vertical = vertical_links(status, nodemap, top_l, bot_l, self._area)
        if opts.connectivity == "full":
            centres = self._grid[["x", "y"]].to_numpy(dtype=float)
            horizontal, found = staggered_links(
                status, nodemap, top_l, bot_l, self._faces, centres, sliver_abs_tol=opts.sliver_abs_tol,
                cl12=opts.cl12,
            )
            self.diagnostics.update(found)
        else:
            horizontal = layered_links(status, nodemap, self._faces, top_l, bot_l)
        self.links = concat_links(vertical, horizontal)
        self.diagnostics.update(_offset_ratio(status, top_l, bot_l, self._faces))
        self.diagnostics["max_ortho_error"] = self._connected_ortho_error()
        self.nodemap = nodemap
        self.cellmap = self._cellmap(top_l, bot_l)
        return self

    def _connected_ortho_error(self):
        """Largest plan orthogonality error over the faces that carry a horizontal connection."""
        side = np.asarray(self.links["ihc"]) != 0
        if not side.any() or not len(self._faces):
            return 0.0
        a = self._node_cell[np.asarray(self.links["n"])[side]]
        b = self._node_cell[np.asarray(self.links["m"])[side]]
        f1 = self._faces["cell_id_1"].to_numpy(dtype=np.int64)
        f2 = self._faces["cell_id_2"].to_numpy(dtype=np.int64)
        ncol = int(max(a.max(), b.max(), f1.max(), f2.max())) + 1
        used = np.isin(np.minimum(f1, f2) * ncol + np.maximum(f1, f2), np.minimum(a, b) * ncol + np.maximum(a, b))
        return float(self._faces["ortho_error"].to_numpy()[used].max()) if used.any() else 0.0

    def get_gridprops_mf6_disu(self):
        """Keyword arguments for ``flopy.mf6.ModflowGwfdisu`` (gridgen key names).

        The one-based layer number is written on the IHC diagonal, which
        MODFLOW 6 does not read. IDOMAIN cannot carry tags: MODFLOW 6 DISU
        accepts only 0 and 1 there, so use :attr:`cellmap` to map nodes to
        layers, cells and zones.
        """
        self._require_build()
        return mf6_gridprops(
            self.nodes, self._node_top, self._node_bot, self._area[self._node_cell],
            self._node_layer, self.links, self._planar, self._node_cell,
        )

    def report(self):
        """Plain-text summary of the grid and its connections."""
        self._require_build()
        counts = {STATUS_NAMES[code]: int((self.status == code).sum()) for code in STATUS_NAMES}
        ihc = np.bincount(self.links["ihc"], minlength=3)
        # Thickness each node takes from collapsed pass-through intervals.
        gained = (self._node_top - self._node_bot) - (self.top_l - self.bot_l)[self.status == ACTIVE]
        gained = float(gained.max()) if len(gained) else 0.0
        lines = [
            f"DISU grid: {self.nodes} nodes from {self.nlay} layers x {len(self._grid)} cells",
            f"  cells: {counts['active']} active, {counts['passthrough']} pass-through, "
            f"{counts['noflow']} no-flow",
            f"  thickness given to active cells by pass-through cells: up to {gained:.6g} per cell "
            f"(collapse={self.options.collapse!r})",
            f"  connections: {len(self.links['n'])} ({ihc[0]} vertical, {ihc[1]} horizontal, "
            f"{ihc[2]} staggered)",
        ]
        if len(self._faces):
            lines.append(f"  max plan orthogonality error: {self.diagnostics['max_ortho_error']:.3g} deg "
                         "(faces that carry a connection)")
        diag = self.diagnostics
        if diag["slivers_merged"]:
            lines.append(
                f"  merged pieces: {diag['slivers_merged']} on side faces (largest {diag['largest_sliver']:.3g}); "
                "merged pieces move face area between cells, so results are not exact "
                "(sliver_abs_tol=0 keeps every piece)."
            )
        ratio = diag["max_offset_ratio"]
        lines.append(f"  max offset ratio (offset / thinner cell, same layer): {ratio:.3g}")
        if ratio >= 1:
            joined = ("joined only through other layers" if self.options.connectivity == "full"
                      else "still joined as same-layer neighbours")
            lines.append(
                f"  {diag['disconnected_faces']} same-layer neighbour pairs are offset by a whole cell "
                f"or more and are {joined}; about {int(np.ceil(ratio)) + 1} sublayers would keep "
                "each layer connected (Provost et al. 2025)."
            )
        if ratio > 0:
            if self.options.connectivity == "layered":
                lines.append("  layers are vertically offset: connectivity='full' is recommended.")
            lines.append(
                "  for offset layers use XT3D, and give NPF angle2 and angle3 (zeros are fine): "
                "without them MODFLOW 6 XT3D ignores the vertical offsets."
            )
        return "\n".join(lines)

    def _cellmap(self, top_l, bot_l):
        grid = self._grid
        ncpl = len(grid)
        cellmap = pd.DataFrame({
            "node": self.nodemap.ravel(), "layer": np.repeat(np.arange(self.nlay), ncpl),
            "icpl": np.tile(np.arange(ncpl), self.nlay),
            "node_id": np.tile(grid["node_id"].to_numpy() if "node_id" in grid else np.arange(ncpl), self.nlay),
            "status": self.status.ravel(),
            "x": np.tile(grid["x"].to_numpy(dtype=float), self.nlay),
            "y": np.tile(grid["y"].to_numpy(dtype=float), self.nlay),
            "top": top_l.ravel(), "bot": bot_l.ravel(),
        })
        if "zone_id" in grid:
            cellmap["zone_id"] = np.tile(grid["zone_id"].to_numpy(), self.nlay)
        cellmap["status"] = pd.Categorical.from_codes(
            cellmap["status"].to_numpy(), [STATUS_NAMES[code] for code in range(len(STATUS_NAMES))],
        )
        return cellmap

    def _require_build(self):
        if self.nodemap is None:
            raise RuntimeError("Call build() first.")


def _offset_ratio(status, top_l, bot_l, faces):
    """Largest vertical offset between same-layer neighbours, relative to the thinner cell.

    Provost et al. (2025) show connectivity changes where this ratio reaches 1
    (same-layer neighbours stop touching), 2 and 3.
    """
    i = faces["cell_id_1"].to_numpy(dtype=np.int64)
    j = faces["cell_id_2"].to_numpy(dtype=np.int64)
    both = (status[:, i] == ACTIVE) & (status[:, j] == ACTIVE)
    if not both.any():
        return {"max_offset_ratio": 0.0, "disconnected_faces": 0}
    offset = np.maximum(np.abs(top_l[:, i] - top_l[:, j]), np.abs(bot_l[:, i] - bot_l[:, j]))
    thinner = np.minimum(top_l[:, i] - bot_l[:, i], top_l[:, j] - bot_l[:, j])
    ratio = np.where(both, offset / np.where(both, thinner, 1.0), 0.0)
    return {"max_offset_ratio": float(ratio.max()), "disconnected_faces": int((ratio >= 1).sum())}
