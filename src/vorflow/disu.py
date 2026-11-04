"""Fully unstructured MODFLOW grids (DISU) from vorflow meshes.

The workflow follows FloPy conventions: give the 2D grid and the usual layer
arrays (``top``, ``botm``, ``idomain``), call :meth:`DisuGrid.build`, then
unpack :meth:`DisuGrid.get_gridprops_mf6_disu` into ``flopy.mf6.ModflowGwfdisu``,
as with ``flopy.utils.gridgen.Gridgen``. vorflow reads no rasters and writes
no model files.

Vertical pass-through of pinched cells follows LoopFlopy
(https://github.com/kerrybardot/loopflopy; Bardot et al. 2025, Environmental
Modelling & Software 192, 106557). This module is an independent, vectorised
implementation with its own pinch-out handling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import _columns
from ._columns import ACTIVE, STATUS_NAMES
from ._connect3d import concat_links, layered_links, vertical_links
from ._disu_export import mf6_gridprops

__all__ = ["DisuGrid", "DisuOptions"]

CONNECTIVITY = ("layered",)


@dataclass(frozen=True)
class DisuOptions:
    """Validated :class:`DisuGrid` settings; see the class docstring for their meaning."""

    min_thickness: float = 0.0
    collapse: str | None = "split"
    connectivity: str = "layered"

    def __post_init__(self):
        value = self.min_thickness
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"min_thickness must be a finite number >= 0, got {value!r}.")
        if self.collapse not in _columns.COLLAPSE_MODES:
            raise ValueError(f"collapse must be one of {_columns.COLLAPSE_MODES}, got {self.collapse!r}.")
        if self.connectivity not in CONNECTIVITY:
            raise ValueError(f"connectivity must be one of {CONNECTIVITY}, got {self.connectivity!r}.")


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
        connectivity: ``"layered"`` reproduces DISV connectivity (same-layer
            neighbours only).

    After :meth:`build`, ``nodemap`` is an ``(nlay, ncpl)`` array mapping
    ``(layer, cell)`` to the DISU node (-1 for removed cells), and ``cellmap``
    describes every layer cell.
    """

    def __init__(self, grid, top, botm, idomain=None, *, min_thickness=0.0, collapse="split",
                 connectivity="layered"):
        self.options = DisuOptions(min_thickness=min_thickness, collapse=collapse, connectivity=connectivity)
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

        self.links = concat_links(
            vertical_links(status, nodemap, top_l, bot_l, self._area),
            layered_links(status, nodemap, self._faces, top_l, bot_l),
        )
        self.diagnostics = {"max_ortho_error": self._connected_ortho_error()}
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
        ihc = np.bincount(self.links["ihc"], minlength=2)
        # Thickness each node takes from collapsed pass-through intervals.
        gained = (self._node_top - self._node_bot) - (self.top_l - self.bot_l)[self.status == ACTIVE]
        gained = float(gained.max()) if len(gained) else 0.0
        lines = [
            f"DISU grid: {self.nodes} nodes from {self.nlay} layers x {len(self._grid)} cells",
            f"  cells: {counts['active']} active, {counts['passthrough']} pass-through, "
            f"{counts['noflow']} no-flow",
            f"  thickness given to active cells by pass-through cells: up to {gained:.6g} per cell "
            f"(collapse={self.options.collapse!r})",
            f"  connections: {len(self.links['n'])} ({ihc[0]} vertical, {ihc[1]} horizontal)",
        ]
        if len(self._faces):
            lines.append(f"  max plan orthogonality error: {self.diagnostics['max_ortho_error']:.3g} deg "
                         "(faces that carry a connection)")
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
