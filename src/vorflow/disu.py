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

import warnings
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import _columns
from ._columns import ABSENT, ACTIVE, NOFLOW, PASSTHROUGH, STATUS_NAMES, subdivide
from ._connect3d import concat_links, layered_links, overlay_links, staggered_links, vertical_links
from ._disu_export import mf6_gridprops, mfusg_gridprops, mfusg_ksat, write_gsf

__all__ = ["DisuGrid", "DisuOptions", "refill", "splice", "split_contacts", "subdivide"]

CONNECTIVITY = ("full", "layered")
CL12_MODES = ("plan", "connector")
REFILL_PRIORITIES = ("resolution", "growth")
WEDGES = ("upper", "lower")


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
    """A DISU grid built from vorflow 2D grids and FloPy-style layer arrays.

    Args:
        grids: The vorflow grid (GeoDataFrame with ``x``/``y`` generator
            columns), or a list with one grid per layer. Layers may use
            different grids, e.g. a :func:`splice` that is finer inside an area.
        top: Model top, scalar or one value per cell of the first layer's grid.
            With several grids it may also be a list with one entry per layer:
            ``None``, or the tops of that layer's cells. Where a layer's grid
            has cells that the layer above lacks (a change of grid), the
            contact is made gap-free: cells that overlap across it share one
            elevation, the area-weighted mean of the bottoms above (exactly
            those bottoms when they agree, e.g. for nested refinement). Tops
            given here must match the bottoms above them. To keep dipping
            bottoms there as given, see :func:`split_contacts`.
        botm: Layer bottoms: ``(nlay,)`` or ``(nlay, ncpl)`` for one grid, or a
            list with one value or array per layer.
        idomain: ``None`` (all active), ``(nlay, ncpl)`` or a list per layer:
            >0 active, 0 no-flow (never passed through), -1 pass-through.
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
            given to the pieces above and below it; a plan overlap between
            grids smaller than ``sliver_abs_tol**2`` is added to that cell's
            largest overlap. Merging gives fewer connections but moves face
            area between cells, so results are no longer exact; the default
            keeps every piece, and MODFLOW handles tiny ones fine.
        cl12: ``"plan"`` (perpendicular plan distance to the face, as in the
            paper; exact for XT3D) or ``"connector"``.

    After :meth:`build`, ``nodemap`` maps ``(layer, cell)`` to the DISU node
    (-1 for removed cells): an ``(nlay, ncpl)`` array for one grid, a list of
    per-layer arrays for several. ``cellmap`` describes every layer cell.

    Cells with identical footprints in different grids (as kept by
    :func:`splice`) form one column through all their layers, so full
    connectivity applies across a change of grid. Other cells meet the cells
    above and below them through their plan overlaps.
    """

    def __init__(self, grids, top, botm, idomain=None, *, min_thickness=0.0, collapse="split",
                 connectivity="full", sliver_abs_tol=0.0, cl12="plan"):
        self.options = DisuOptions(
            min_thickness=min_thickness, collapse=collapse, connectivity=connectivity,
            sliver_abs_tol=sliver_abs_tol, cl12=cl12,
        )
        self._grids, self._layer_grid, self._layers = _layer_inputs(grids, top, botm, idomain)
        self.nodemap = None
        self.cellmap = None
        self.links = None

    @classmethod
    def from_disv(cls, vertices, cell2d, top, botm, idomain=None, **options):
        """Build from MODFLOW 6 DISV arrays, e.g. an existing FloPy or LoopFlopy model.

        ``vertices`` is ``[[iv, x, y], ...]`` and ``cell2d`` is
        ``[[icell2d, xc, yc, ncvert, iv1, ...], ...]`` as for
        ``flopy.mf6.ModflowGwfdisv``; ``xc``/``yc`` are kept as the cell centres.
        The other arguments are those of :class:`DisuGrid`.
        """
        xy = {int(row[0]): (float(row[1]), float(row[2])) for row in vertices}
        polygons, xc, yc, ids = [], [], [], []
        for row in cell2d:
            ncvert = int(row[3])
            ring = [xy[int(iv)] for iv in row[4:4 + ncvert]]
            if len(ring) > 1 and ring[0] == ring[-1]:
                ring = ring[:-1]
            polygons.append(shapely.Polygon(ring))
            ids.append(int(row[0]))
            xc.append(float(row[1]))
            yc.append(float(row[2]))
        grid = gpd.GeoDataFrame({"node_id": ids, "x": xc, "y": yc}, geometry=polygons)
        return cls(grid, top, botm, idomain, **options)

    @property
    def nlay(self):
        return len(self._layer_grid)

    @property
    def single_grid(self):
        return len(self._grids) == 1

    @property
    def grid(self):
        """The plan grid when every layer uses the same one."""
        if not self.single_grid:
            raise AttributeError("Layers use different grids; see grids_per_layer.")
        return self._grids[0]

    @property
    def grids_per_layer(self):
        return [self._grids[g] for g in self._layer_grid]

    @property
    def nodes(self):
        self._require_build()
        return int((self._nodemap >= 0).sum())

    def build(self):
        """Classify cells, close pass-through gaps and connect the active cells. Returns ``self``."""
        from ._mfusg import _planar_faces

        opts = self.options
        planar = [_planar_faces(grid) for grid in self._grids]
        tolerance = max(p[3] for p in planar)
        self._prepare_columns(tolerance)
        columns = self._columns
        self._faces = _global_faces([p[2] for p in planar], self._col_of)
        cross = self._cross_faces(tolerance)
        if len(cross):
            faces = pd.concat([self._faces, cross.astype(self._faces.dtypes.to_dict())], ignore_index=True)
            # A column can belong to several grids: one face per column pair, a grid's own face first.
            key = np.sort(faces[["cell_id_1", "cell_id_2"]].to_numpy(dtype=np.int64), axis=1)
            self._faces = faces[~pd.DataFrame(key).duplicated().to_numpy()].reset_index(drop=True)
        from .utils import grid_to_disv

        self._planar = grid_to_disv(columns, tolerance=tolerance)
        top_l, bot_l, idomain = self._lattice()
        self.top_l, self.bot_l = top_l, bot_l
        status = _columns.classify(top_l, bot_l, idomain, opts.min_thickness)
        top_l, bot_l = _columns.collapse(top_l, bot_l, status, opts.collapse)
        self._top_c, self._bot_c = top_l, bot_l
        active = status == ACTIVE
        nodemap = np.full(status.shape, -1, dtype=np.int64)
        nodemap[active] = np.arange(int(active.sum()))
        self.status, self._nodemap = status, nodemap
        self._node_top, self._node_bot = top_l[active], bot_l[active]
        self._node_layer, self._node_cell = np.nonzero(active)

        self.diagnostics = {"slivers_merged": 0, "largest_sliver": 0.0, "plan_slivers_merged": 0,
                            "chained_links": 0, "largest_skipped_run": 0.0, **self._contact}
        overlaps = self._overlaps()
        vertical = concat_links(
            vertical_links(status, nodemap, top_l, bot_l, self._area),
            overlay_links(status, nodemap, top_l, bot_l, overlaps),
            self._chained_links(status, nodemap, top_l, bot_l),
        )
        if opts.connectivity == "full":
            centres = columns[["x", "y"]].to_numpy(dtype=float)
            horizontal, found = staggered_links(
                status, nodemap, top_l, bot_l, self._faces, centres, sliver_abs_tol=opts.sliver_abs_tol,
                cl12=opts.cl12,
            )
            self.diagnostics.update(found)
        else:
            horizontal = layered_links(status, nodemap, self._faces, top_l, bot_l)
        self.links = concat_links(vertical, horizontal)
        self.diagnostics.update(_offset_ratio(status, top_l, bot_l, self._faces))
        self.diagnostics["vertical_offset"] = _vertical_offset(self.links, self._node_top, self._node_bot)
        self.diagnostics["max_ortho_error"] = self._connected_ortho_error()
        self.nodemap = self._public(nodemap)
        self.cellmap = self._cellmap(top_l, bot_l)
        return self

    def _prepare_columns(self, tolerance):
        """Global plan columns, their areas, and the area below which two cells only touch."""
        self._col_of, self._columns = _plan_columns(self._grids)
        geoms = self._columns.geometry.to_numpy()
        self._area = shapely.area(geoms)
        # One threshold decides whether two cells of different grids overlap
        # (vertical link) or only touch (face), so no pair gets both.
        self._touch_area = tolerance * shapely.length(geoms)

    def _connected_ortho_error(self):
        """Largest plan orthogonality error over the faces that carry a horizontal connection.

        Faces between columns of different grids that never share an
        elevation (a coarse cell above a change of grid next to a fine cell
        below it) carry no connection, so they are left out.
        """
        face = self._link_faces()
        face = face[face >= 0]
        return float(self._faces["ortho_error"].to_numpy()[face].max()) if len(face) else 0.0

    def _link_faces(self):
        """Row of the plan face table behind each connection; -1 for vertical connections."""
        found = np.full(len(self.links["n"]), -1, dtype=np.int64)
        side = np.flatnonzero(np.asarray(self.links["ihc"]) != 0)
        if not len(side) or not len(self._faces):
            return found
        a = self._node_cell[np.asarray(self.links["n"])[side]]
        b = self._node_cell[np.asarray(self.links["m"])[side]]
        f1 = self._faces["cell_id_1"].to_numpy(dtype=np.int64)
        f2 = self._faces["cell_id_2"].to_numpy(dtype=np.int64)
        ncol = len(self._columns)
        face_key = np.minimum(f1, f2) * ncol + np.maximum(f1, f2)
        link_key = np.minimum(a, b) * ncol + np.maximum(a, b)
        order = np.argsort(face_key)
        pos = order[np.minimum(np.searchsorted(face_key, link_key, sorter=order), len(order) - 1)]
        found[side] = np.where(face_key[pos] == link_key, pos, -1)
        return found

    def get_gridprops_mf6_disu(self):
        """Keyword arguments for ``flopy.mf6.ModflowGwfdisu`` (gridgen key names).

        The one-based layer number is written on the IHC diagonal, which
        MODFLOW 6 does not read. IDOMAIN cannot carry tags: MODFLOW 6 DISU
        accepts only 0 and 1 there, so use :attr:`cellmap` (or
        :meth:`save_cellmap`) to map nodes to layers, cells and zones.

        Contacts at a change of grid are gap-free, so cells above and below
        meet exactly. If any vertical connection still overlaps in elevation,
        ``vertical_offset_tolerance`` is set to the largest overlap so MODFLOW 6
        accepts it.
        """
        self._require_build()
        props = mf6_gridprops(
            self.nodes, self._node_top, self._node_bot, self._area[self._node_cell],
            self._node_layer, self.links, self._planar, self._node_cell,
        )
        if self.diagnostics["vertical_offset"] > 0:
            props["vertical_offset_tolerance"] = self.diagnostics["vertical_offset"] * (1 + 1e-9)
        return props

    def get_gridprops_mfusg_disu(self):
        """Keyword arguments for ``flopy.mfusg.MfUsgDisU`` (MODFLOW-USG; gridgen key names).

        Uses ``ivsd=1`` with an explicit ``ivc`` so staggered connections stay
        horizontal. ``fahl`` is the interface area: for a horizontal connection
        the face width times its (merged) overlap height. MODFLOW-USG has no
        XT3D; offset layers rely on the two-point formulation.

        As with Gridgen, MODFLOW-USG divides a horizontal ``fahl`` by the mean
        thickness of the two cells and uses the harmonic mean of K times
        thickness, which is exact only when both cells are equally thick. For
        staggered connections between cells of unequal thickness (wedges,
        sublayers, pinch-outs), use ``MfUsgLpf(ikcflag=1, ksat=...)`` with
        :meth:`get_mfusg_ksat`.

        MODFLOW-USG needs at least one node in every layer, so a layer with no
        active cell (for example one pinched out everywhere) raises ValueError.
        """
        self._require_build()
        return mfusg_gridprops(
            self.nodes, self.nlay, self._node_top, self._node_bot, self._area[self._node_cell],
            self._node_layer, self.links,
        )

    def get_mfusg_ksat(self, k, k33=None):
        """Hydraulic conductivity per connection, for ``flopy.mfusg.MfUsgLpf(ikcflag=1, ksat=...)``.

        With ``ikcflag=1`` MODFLOW-USG computes every conductance as
        ``fahl * ksat / (cl1 + cl2)``, so the conductance follows the shared
        face area also between cells of unequal thickness. ``ksat`` is the
        distance-weighted harmonic mean of the two cells' K (``k33`` for
        vertical connections), which gives MODFLOW 6's conductance.

        Use it with confined layers (``laytyp=0``) or convertible layers with
        upstream weighting (``laytyp=4``); both are exact while the cells are
        saturated. With a water table inside cells of unequal thickness,
        ``laytyp=4`` divides the face area by their mean thickness again.
        MODFLOW-USG and USG-Transport do not allow ``laytyp=1`` with
        ``ikcflag=1``; USG-Transport only warns and runs on with wrong
        conductances.

        Parameters
        ----------
        k : float or array_like of shape (nodes,)
            Horizontal hydraulic conductivity per active node.
        k33 : float or array_like of shape (nodes,), optional
            Vertical hydraulic conductivity; ``k`` if omitted.

        Returns
        -------
        numpy.ndarray
            One value per entry of the ``ja`` array from
            :meth:`get_gridprops_mfusg_disu` (the node's own K on the
            diagonal, which MODFLOW-USG does not read).
        """
        self._require_build()
        k = _node_values(k, self.nodes, "k")
        k33 = k if k33 is None else _node_values(k33, self.nodes, "k33")
        return mfusg_ksat(self.nodes, self.links, k, k33)

    def write_gsf(self, filepath):
        """Write the grid specification file (GSF) used by MODFLOW-USG post-processors."""
        self._require_build()
        return write_gsf(filepath, self._planar, self._node_cell, self._node_top, self._node_bot,
                         self._node_layer, self.nlay)

    def to_nodes(self, values):
        """Pick the values of the active cells, in DISU node order.

        ``values`` is shaped like ``botm``: ``(nlay, ncpl)`` for one grid, or a
        list with one array (or scalar) per layer.
        """
        self._require_build()
        lattice = np.full(self._nodemap.shape, np.nan, dtype=object)
        for layer, layer_values in enumerate(self._per_layer(values, "values")):
            lattice[layer, self._col_of[self._layer_grid[layer]]] = layer_values
        picked = lattice[self._nodemap >= 0]
        return np.array(picked.tolist())

    def to_layers(self, values, fill=np.nan):
        """Spread per-node values back to the layers, with ``fill`` for removed cells.

        Returns ``(nlay, ncpl)`` for one grid, or a list of per-layer arrays.
        """
        self._require_build()
        values = np.asarray(values).ravel()
        if values.shape != (self.nodes,):
            raise ValueError(f"values must have {self.nodes} entries, one per node.")
        lattice = np.full(self._nodemap.shape, fill, dtype=np.result_type(values, np.asarray(fill)))
        lattice[self._nodemap >= 0] = values
        return self._public(lattice)

    @property
    def idomain_layers(self):
        """Each layer's cells in MODFLOW 6 idomain codes: 1 active, -1 pass-through, 0 no-flow.

        Shaped like ``nodemap`` (``(nlay, ncpl)`` for one grid, a list per
        layer otherwise), so it plots like a DISV idomain. MODFLOW 6 DISU
        itself only accepts 0 and 1; this map is for tracking and plotting.
        """
        self._require_build()
        codes = np.zeros(len(STATUS_NAMES), dtype=np.int64)
        codes[ACTIVE], codes[PASSTHROUGH] = 1, -1
        return self._public(codes[self.status])

    def layer_view(self, values, layer, parent=None, pick="top", fill="nan"):
        """One value per plan cell for an original layer, for 2D maps.

        Args:
            values: One value per DISU node (e.g. ``heads.ravel()``).
            layer: The original layer, as numbered by ``parent``.
            parent: The original layer of each layer, as returned by
                :func:`subdivide`; ``None`` when no layer was split.
            pick: Which sublayer gives the value in each column: ``"top"``
                or ``"bottom"`` (the highest or lowest active one),
                ``"middle"`` (the active cell at the original layer's
                mid-elevation) or ``"mean"`` (thickness-weighted).
            fill: For columns where the layer has no active cell: ``"nan"``,
                or ``"cover"`` for the active cell that took over its interval
                (see ``collapse``); no-flow cells always stay NaN.

        Returns a ``(ncpl,)`` array on that layer's plan grid
        (``grids_per_layer``); :meth:`vertex_grid` gives a FloPy grid for it.
        """
        self._require_build()
        if pick not in ("top", "bottom", "middle", "mean"):
            raise ValueError(f"pick must be 'top', 'bottom', 'middle' or 'mean', got {pick!r}.")
        if fill not in ("nan", "cover"):
            raise ValueError(f"fill must be 'nan' or 'cover', got {fill!r}.")
        values = np.asarray(values, dtype=float).ravel()
        if values.shape != (self.nodes,):
            raise ValueError(f"values must have {self.nodes} entries, one per node.")
        parent = np.arange(self.nlay) if parent is None else np.asarray(parent)
        if parent.shape != (self.nlay,):
            raise ValueError(f"parent needs one entry per layer ({self.nlay}).")
        sub = np.flatnonzero(parent == layer)
        if not len(sub):
            raise ValueError(f"No layer has parent {layer!r}.")
        grids = {self._layer_grid[k] for k in sub}
        if len(grids) > 1:
            raise ValueError(f"The sublayers of layer {layer} use different plan grids.")
        cols = self._col_of[self._layer_grid[sub[0]]]
        node = self._nodemap[np.ix_(sub, cols)]
        active = node >= 0
        value = np.where(active, values[np.maximum(node, 0)], np.nan)
        top, bot = self._top_c[np.ix_(sub, cols)], self._bot_c[np.ix_(sub, cols)]
        middle = (self.top_l[sub[0], cols] + self.bot_l[sub[-1], cols]) / 2
        any_active = active.any(axis=0)
        first = np.argmax(active, axis=0)
        last = len(sub) - 1 - np.argmax(active[::-1], axis=0)
        ncpl = np.arange(len(cols))
        if pick == "top":
            view = value[first, ncpl]
        elif pick == "bottom":
            view = value[last, ncpl]
        elif pick == "mean":
            thick = np.where(active, top - bot, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                view = np.nansum(value * thick, axis=0) / thick.sum(axis=0)
        else:
            hit = active & (bot <= middle) & (middle <= top)
            view = np.where(hit.any(axis=0), value[np.argmax(hit, axis=0), ncpl], np.nan)
        view = np.where(any_active, view, np.nan)
        if fill == "cover":
            noflow = (self.status[np.ix_(sub, cols)] == NOFLOW).all(axis=0)
            empty = ~any_active & ~noflow
            if empty.any():
                view[empty] = self._covering_value(values, cols[empty], middle[empty])
        return view

    def _covering_value(self, values, cols, z):
        """Value of the active cell in each column whose collapsed interval contains ``z``."""
        out = np.full(len(cols), np.nan)
        for k in range(self.nlay):
            node = self._nodemap[k, cols]
            hit = (node >= 0) & (self._bot_c[k, cols] <= z) & (z <= self._top_c[k, cols]) & np.isnan(out)
            out[hit] = values[node[hit]]
        return out

    def vertex_grid(self, layer):
        """A ``flopy.discretization.VertexGrid`` of one layer's plan grid, for 2D plots.

        Plot :meth:`layer_view`, :meth:`to_layers` or :attr:`idomain_layers`
        results on it with ``flopy.plot.PlotMapView``. Cell centres are the
        generators, as in the DISU grid.
        """
        self._require_build()
        try:
            import flopy
        except ImportError as err:
            raise ImportError("vertex_grid() needs flopy; install it with `pip install flopy`.") from err
        from .utils import grid_to_disv

        grid = self.grids_per_layer[layer]
        planar = grid_to_disv(grid)
        cols = self._col_of[self._layer_grid[layer]]
        idomain = self.idomain_layers[layer]
        return flopy.discretization.VertexGrid(
            vertices=planar["vertices"], cell2d=planar["cell2d"], ncpl=planar["ncpl"], nlay=1,
            top=self.top_l[layer, cols], botm=self.bot_l[layer, cols][None, :],
            idomain=np.where(idomain == 0, 0, 1)[None, :], crs=grid.crs,
        )

    def column(self, x, y):
        """Active DISU nodes at plan point ``(x, y)``, from the top down."""
        self._require_build()
        cols = self._columns_at(np.atleast_1d(float(x)), np.atleast_1d(float(y)))
        nodes = [self._nodemap[k, cols[g][0]] for k, g in enumerate(self._layer_grid) if cols[g][0] >= 0]
        return np.array([n for n in nodes if n >= 0], dtype=np.int64)

    def locate(self, x, y, z):
        """DISU node containing each point ``(x, y, z)``, or -1 outside every active cell.

        Uses the cell elevations after pass-through gaps are closed, so a
        point in a collapsed pass-through cell maps to the active cell that
        took its interval. Accepts scalars or arrays.
        """
        self._require_build()
        x, y, z = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (x, y, z))
        cols = self._columns_at(x, y)
        found = np.full(len(x), -1, dtype=np.int64)
        for k, g in enumerate(self._layer_grid):
            col = cols[g]
            node = np.where(col >= 0, self._nodemap[k, np.maximum(col, 0)], -1)
            hit = (node >= 0) & (found < 0)
            node_safe = np.maximum(node, 0)
            hit &= (self._node_bot[node_safe] <= z) & (z <= self._node_top[node_safe])
            found[hit] = node[hit]
        return found if found.size > 1 else int(found[0])

    def save_cellmap(self, path, parent=None):
        """Write :attr:`cellmap` to CSV, so results can be mapped back without vorflow.

        The DISU arrays give each node's plan polygon (``vertices`` and
        ``cell2d``), and MODFLOW 6 copies it to the binary grid file, but not
        the layer structure. The file adds it for tools that work layer by
        layer in 2D, such as pilot points per layer: besides the columns of
        :attr:`cellmap`, ``z`` is the cell's mid-elevation, ``idomain`` the
        code of :attr:`idomain_layers`, and ``unit`` the original layer.

        Args:
            path: The CSV file to write.
            parent: The original layer of each layer, as returned by
                :func:`subdivide` or :func:`split_contacts` (a wedge layer
                belongs to the layer above or below it); ``None`` when no
                layer was split, and ``unit`` is the layer.
        """
        self._require_build()
        unit = np.arange(self.nlay) if parent is None else np.asarray(parent)
        if unit.shape != (self.nlay,) or not np.issubdtype(unit.dtype, np.integer):
            raise ValueError(f"parent needs one integer per layer ({self.nlay}).")
        codes = np.zeros(len(STATUS_NAMES), dtype=np.int64)
        codes[ACTIVE], codes[PASSTHROUGH] = 1, -1
        cells = self.cellmap
        layer = cells["layer"].to_numpy()
        table = cells.assign(z=(cells["top"] + cells["bot"]) / 2,
                             idomain=codes[self.status[layer, cells["column"].to_numpy()]],
                             unit=unit[layer])
        table.to_csv(path, index=False)
        return path

    def _columns_at(self, x, y):
        """Per grid, the global column containing each plan point (-1 outside)."""
        if not hasattr(self, "_trees"):
            geoms = self._columns.geometry.to_numpy()
            self._trees = [shapely.STRtree(geoms[cols]) for cols in self._col_of]
        points = shapely.points(np.column_stack([x, y]))
        out = []
        for tree, cols in zip(self._trees, self._col_of):
            found = np.full(len(points), -1, dtype=np.int64)
            point, hit = tree.query(points, predicate="intersects")
            found[point[::-1]] = cols[hit[::-1]]     # First hit wins on shared edges.
            out.append(found)
        return out

    def quality(self):
        """Geometric quality of every connection, one row each.

        These are measurements, not pass/fail checks: vorflow builds the grid
        it is given, and whether it is good enough for the model is the
        modeller's judgement. Filter or group the table (for example by
        ``ihc``, or per node with ``groupby``) and map it with
        :meth:`plot_quality`, :meth:`plot_quality_section` or
        :meth:`to_layers`.

        Columns: ``n``, ``m`` and ``ihc`` as in :attr:`links`; ``area``, the
        connection's area (face width times shared height for side faces,
        overlap area for vertical connections); and

        * ``ortho_error_3d``: angle in degrees between the line joining the
          two cell centres and the normal of the shared face. Between centres
          at the same depth it is the plan orthogonality error of the face
          (per plan grid, :func:`vorflow.utils.build_connectivity` and
          :func:`vorflow.utils.calculate_mesh_quality` give that one); it adds
          the tilt between offset cells and, for vertical connections, the
          plan shift between cells of different grids. Without XT3D or ghost
          nodes the flow between two cells is exact only where it is 0.
        * ``cl_ratio``: the longer of the two centre-to-face distances over
          the shorter: the grading across the face (plan distances for side
          faces, half thicknesses for vertical connections). 1 when the face
          is midway between the centres.
        * ``gap``: thickness of the pass-through cells a vertical connection
          skips between two changes of grid; 0 elsewhere.
        """
        self._require_build()
        n, m, ihc = (np.asarray(self.links[key]) for key in ("n", "m", "ihc"))
        top, bot, col = self._node_top, self._node_bot, self._node_cell
        centre = np.column_stack([self._columns["x"].to_numpy(dtype=float)[col],
                                  self._columns["y"].to_numpy(dtype=float)[col], (top + bot) / 2])
        joint = centre[m] - centre[n]
        side = ihc != 0
        face = self._link_faces()
        normal = np.zeros_like(joint)
        normal[~side, 2] = 1.0
        normal[side, 0] = self._faces["normal_x"].to_numpy(dtype=float)[face[side]]
        normal[side, 1] = self._faces["normal_y"].to_numpy(dtype=float)[face[side]]
        cosine = np.abs((joint * normal).sum(axis=1)) / np.linalg.norm(joint, axis=1)
        cl_n, cl_m = np.asarray(self.links["cl_n"], dtype=float), np.asarray(self.links["cl_m"], dtype=float)
        upper = np.where(top[n] >= top[m], n, m)
        lower = np.where(upper == n, m, n)
        return pd.DataFrame({
            "n": n, "m": m, "ihc": ihc, "area": np.asarray(self.links["fahl"], dtype=float),
            "ortho_error_3d": np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))),
            "cl_ratio": np.maximum(cl_n, cl_m) / np.minimum(cl_n, cl_m),
            "gap": np.where(side, 0.0, np.maximum(0.0, bot[upper] - top[lower])),
        })

    def plot_quality(self, layer, column="ortho_error_3d", quality=None, ax=None, colorbar=True, **kwargs):
        """Plan view of one layer's connections, coloured by a column of :meth:`quality`.

        Each connection with an end in ``layer`` is a line between the two
        cell centres (a vertical connection within one cell column is a dot),
        drawn over the layer's plan grid in order of value, so the largest end
        on top. Needs Matplotlib (``pip install vorflow[plot]``).

        Args:
            layer: The layer to draw.
            column: The :meth:`quality` column to colour by.
            quality: The :meth:`quality` table, or some of its rows (for
                example ``q[q.ihc == 0]`` or ``q[q.cl_ratio > 2]``) to draw only
                those; computed when None.
            ax (matplotlib.axes.Axes, optional): Axes to draw on; a new figure
                is created when None.
            colorbar (bool): Add a colour bar.
            **kwargs: ``cmap``, ``norm`` and ``linewidth`` of the connections.

        Returns:
            matplotlib.axes.Axes: The axes drawn on.
        """
        from ._plot import draw_connections, get_ax

        self._require_build()
        if not 0 <= layer < self.nlay:
            raise ValueError(f"layer must be in [0, {self.nlay}), got {layer}.")
        q = self.quality() if quality is None else quality
        q = q[q[column].notna()]
        n, m = q["n"].to_numpy(), q["m"].to_numpy()
        q = q[(self._node_layer[n] == layer) | (self._node_layer[m] == layer)]
        ax = get_ax(ax)
        self.grids_per_layer[layer].boundary.plot(ax=ax, color="0.75", linewidth=0.3)
        xy = self._columns[["x", "y"]].to_numpy(dtype=float)[self._node_cell]
        draw_connections(ax, xy[q["n"].to_numpy()], xy[q["m"].to_numpy()], q[column].to_numpy(dtype=float),
                         colorbar=colorbar, label=column, **kwargs)
        return ax

    def plot_quality_section(self, line, column="ortho_error_3d", quality=None, ax=None, colorbar=True, **kwargs):
        """Vertical section along a plan line, with connections coloured by a column of :meth:`quality`.

        The active cells the line crosses are outlined, each over the stretch
        of the line inside it; each connection between two of them is a line
        between their middles. The horizontal axis is the distance along
        ``line``; with ``ax.set_aspect("equal")`` the drawn angles are true.
        Needs Matplotlib (``pip install vorflow[plot]``).

        Args:
            line: A shapely LineString, or its points, in plan.
            column, quality, ax, colorbar, **kwargs: As for :meth:`plot_quality`.

        Returns:
            matplotlib.axes.Axes: The axes drawn on.
        """
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle

        from ._plot import draw_connections, require_matplotlib

        self._require_build()
        plt = require_matplotlib()
        line = line if isinstance(line, shapely.LineString) else shapely.LineString(line)
        cells = self._section_cells(line)
        q = self.quality() if quality is None else quality
        q = q[q[column].notna() & q["n"].isin(cells.index) & q["m"].isin(cells.index)]
        if ax is None:
            _, ax = plt.subplots()
        top, bot = self._node_top[cells.index], self._node_bot[cells.index]
        boxes = [Rectangle((s0, b), s1 - s0, t - b) for s0, s1, b, t in zip(cells.s0, cells.s1, bot, top)]
        ax.add_collection(PatchCollection(boxes, facecolor="none", edgecolor="0.6", linewidth=0.4))
        middle = np.column_stack([(cells.s0 + cells.s1) / 2, (top + bot) / 2])
        row = pd.Series(np.arange(len(cells)), index=cells.index)
        draw_connections(ax, middle[row[q["n"]].to_numpy()], middle[row[q["m"]].to_numpy()],
                         q[column].to_numpy(dtype=float), colorbar=colorbar, label=column, **kwargs)
        ax.set_xlabel("distance along the line")
        ax.set_ylabel("elevation")
        return ax

    def _section_cells(self, line):
        """Active nodes whose plan cell ``line`` crosses, with the stretch of the line (s0, s1) inside each."""
        rows = []
        for g, grid in enumerate(self._grids):
            geoms = grid.geometry.to_numpy()
            hit = shapely.STRtree(geoms).query(line, predicate="intersects")
            cuts = shapely.intersection(geoms[hit], line)
            for i, cut in zip(hit, cuts):
                if cut.length == 0:
                    continue
                along = shapely.line_locate_point(line, shapely.points(shapely.get_coordinates(cut)))
                for layer in np.flatnonzero(np.asarray(self._layer_grid) == g):
                    node = self._nodemap[layer, self._col_of[g][i]]
                    if node >= 0:
                        rows.append((node, along.min(), along.max()))
        return pd.DataFrame(rows, columns=["node", "s0", "s1"]).drop_duplicates("node").set_index("node")

    def report(self):
        """Plain-text summary of the grid and its connections."""
        self._require_build()
        counts = {STATUS_NAMES[code]: int((self.status == code).sum()) for code in STATUS_NAMES}
        ihc = np.bincount(self.links["ihc"], minlength=3)
        # Thickness each node takes from collapsed pass-through intervals.
        gained = (self._node_top - self._node_bot) - (self.top_l - self.bot_l)[self.status == ACTIVE]
        gained = float(gained.max()) if len(gained) else 0.0
        shape = (f"{self.nlay} layers x {self._grids[0].shape[0]} cells" if self.single_grid
                 else f"{self.nlay} layers on {len(self._grids)} grids")
        lines = [
            f"DISU grid: {self.nodes} nodes from {shape}",
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
        if len(self.links["n"]):
            q = self.quality()
            lines.append(f"  max 3D orthogonality error: {q['ortho_error_3d'].max():.3g} deg; largest cl ratio: "
                         f"{q['cl_ratio'].max():.3g} (every connection: quality())")
        diag = self.diagnostics
        if diag["chained_links"]:
            lines.append(
                f"  {diag['chained_links']} vertical connections cross pass-through cells between two changes "
                f"of grid; their thickness is skipped (largest {diag['largest_skipped_run']:.6g})."
            )
        merged = diag["slivers_merged"] + diag["plan_slivers_merged"]
        if merged:
            lines.append(
                f"  merged pieces: {diag['slivers_merged']} on side faces (largest {diag['largest_sliver']:.3g}), "
                f"{diag['plan_slivers_merged']} plan overlaps; merged pieces move face area between cells, "
                "so results are not exact (sliver_abs_tol=0 keeps every piece)."
            )
        if not self.single_grid:
            if diag["contact_groups"]:
                lines.append(f"  change of grid: {diag['contact_groups']} contact groups made gap-free")
            if diag["max_contact_shift"] > 0:
                lines.append(
                    f"  the contact under a change of grid moved bottoms by up to {diag['max_contact_shift']:.3g}; "
                    "disu.split_contacts keeps them as given."
                )
            if diag["vertical_offset"] > 0:
                lines.append(
                    f"  cells above and below a change of grid overlap by up to {diag['vertical_offset']:.3g}; "
                    "MODFLOW 6 vertical_offset_tolerance is set to accept it."
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

    def _lattice(self):
        """Elevations and idomain over every plan column, NaN where a layer's grid lacks the column."""
        ncol = len(self._columns)
        top_l = np.full((self.nlay, ncol), np.nan)
        bot_l = np.full((self.nlay, ncol), np.nan)
        idomain = np.zeros((self.nlay, ncol), dtype=np.int64)
        self._contact = {"contact_groups": 0, "max_contact_shift": 0.0}
        for layer, (top, bot, ido) in enumerate(self._layers):
            cols = self._col_of[self._layer_grid[layer]]
            bot_l[layer, cols] = bot
            idomain[layer, cols] = ido
            if layer == 0:
                top_l[0, cols] = top
                continue
            above = bot_l[layer - 1, cols]
            new = np.isnan(above)
            layer_top = above.copy()
            if new.any():
                in_layer = np.zeros(ncol, dtype=bool)
                in_layer[cols] = True
                ending = np.flatnonzero(~np.isnan(bot_l[layer - 1]) & ~in_layer)
                given = None if top is None else np.broadcast_to(top, cols.shape)[new]
                start_top, end_bot = self._contact_levels(layer, ending, cols[new], bot_l[layer - 1], given)
                bot_l[layer - 1, ending] = end_bot
                layer_top[new] = start_top
            top_l[layer, cols] = layer_top
        with np.errstate(invalid="ignore"):
            bad = np.argwhere(top_l < bot_l)
        if len(bad):
            layer, col = bad[0]
            raise ValueError(
                f"Negative thickness in layer {layer} (column {col}): each bottom must lie at or "
                "below the bottom of the layer above. At a change of grid the contact is the "
                "area-weighted mean of the bottoms above it; give a flat botm there to choose it."
            )
        return top_l, bot_l, idomain

    def _contact_levels(self, layer, ending, starting, bot_above, given=None):
        """One contact elevation per group of cells that meet across a change of grid.

        ``ending`` columns stop below ``layer - 1`` and ``starting`` columns
        begin in ``layer``. Cells that overlap in plan are joined into groups;
        each group's contact is the area-weighted mean of its upper cells'
        bottoms, so no gap or overlap remains between the two grids. A group
        with a single upper cell (nested refinement), or whose bottoms already
        agree, keeps that elevation exactly. Tops given for ``starting`` cells
        are kept, and must match the bottoms above them.

        Returns ``(top of each starting column, bottom of each ending column)``.
        """
        geoms = self._columns.geometry.to_numpy()
        group_s, group_e, n_groups, _ = _contact_groups(geoms, starting, ending, self._touch_area)
        bottom = bot_above[ending]
        level, lowest, highest = _group_bottoms(group_e, bottom, self._area[ending], n_groups)
        joined = ~np.isnan(level)

        start_top = level[group_s]
        if given is not None:
            scale = np.nanmax(np.abs(np.r_[given, bottom, 1.0]))
            spread = np.where(joined, highest - lowest, 0.0)[group_s]
            mismatch = (np.abs(given - np.where(np.isnan(start_top), given, start_top)) > 1e-9 * scale) | (
                spread > 1e-9 * scale)
            if mismatch.any():
                raise ValueError(
                    f"Layer {layer}: the tops given for cells below the change of grid do not meet the "
                    "bottoms of the cells above them. Leave that layer's top as None to make the contact "
                    "gap-free automatically."
                )
            start_top = np.asarray(given, dtype=float).copy()
        lone = np.isnan(start_top)
        if lone.any():
            start_top[lone] = self._sample_above(layer, starting[lone], bot_above)
        end_bot = level[group_e]
        groups = np.unique(group_s[~lone])
        self._contact["contact_groups"] += int(len(groups))
        if len(ending):
            self._contact["max_contact_shift"] = max(self._contact["max_contact_shift"],
                                                     float(np.abs(end_bot - bottom).max()))
        return start_top, end_bot

    def _sample_above(self, layer, cols, bot_above):
        """Bottom of the layer-above cell that contains each column's centre."""
        return bot_above[self._col_of[self._layer_grid[layer - 1]][self._cell_above(layer, cols)]]

    def _cell_above(self, layer, cols):
        """Index, in the layer above's grid, of the cell that contains each column's centre."""
        above_cols = self._col_of[self._layer_grid[layer - 1]]
        geoms = self._columns.geometry.to_numpy()
        tree = shapely.STRtree(geoms[above_cols])
        centres = shapely.points(self._columns[["x", "y"]].to_numpy(dtype=float)[cols])
        point, hit = tree.query(centres, predicate="intersects")
        found = np.full(len(cols), -1)
        found[point[::-1]] = hit[::-1]           # First hit wins on shared edges.
        if (found < 0).any():
            raise ValueError(
                f"Layer {layer}: a cell centre lies outside the layer above; give its top explicitly."
            )
        return found

    def _cross_faces(self, tolerance):
        """Faces between cells of different grids that touch along an edge without overlapping.

        Inside a refined area, a coarse cell above a change of grid and a fine
        cell below it can be plan neighbours; with offset layers their walls
        overlap vertically and need a staggered connection. As in
        ``_planar_faces``, the part of one boundary within the tolerance of the
        other is the face when roundoff hides part of it.
        """
        from ._mfusg import RECOVERED_FACE_FRACTION, _line_parts, _straight_face

        geoms = self._columns.geometry.to_numpy()
        centres = self._columns[["x", "y"]].to_numpy(dtype=float)
        records = []
        for g1 in range(len(self._grids)):
            for g2 in range(g1 + 1, len(self._grids)):
                a_cols, b_cols = self._col_of[g1], self._col_of[g2]
                a_only = a_cols[~np.isin(a_cols, b_cols)]
                b_only = b_cols[~np.isin(b_cols, a_cols)]
                if not len(a_only) or not len(b_only):
                    continue
                ia, ib = shapely.STRtree(geoms[b_only]).query(geoms[a_only], predicate="dwithin",
                                                               distance=tolerance)
                i, j = a_only[ia], b_only[ib]
                # Overlapping cells meet vertically, not side by side.
                inside = shapely.relate_pattern(geoms[i], geoms[j], "T********")
                area = np.zeros(len(i))
                area[inside] = shapely.area(shapely.intersection(geoms[i[inside]], geoms[j[inside]]))
                i, j = i[area <= self._touch_area[i]], j[area <= self._touch_area[i]]
                for a, b in zip(i.tolist(), j.tolist()):
                    shared = shapely.intersection(geoms[a].boundary, geoms[b].boundary, grid_size=tolerance)
                    near = shapely.intersection(geoms[a].boundary, shapely.buffer(geoms[b].boundary, tolerance))
                    lost = sum(p.length for p in _line_parts(near)) - sum(p.length for p in _line_parts(shared))
                    if lost > RECOVERED_FACE_FRACTION * np.sqrt(min(self._area[a], self._area[b])):
                        shared = near
                    try:
                        record = _straight_face(shared, centres[[a, b]], (a, b), 2 * tolerance)
                    except ValueError:
                        record = _bent_face(shared, centres[[a, b]], (a, b), tolerance)
                    if record is not None:
                        records.append(record)
        columns = ["cell_id_1", "cell_id_2", "shared_length", "distance_1", "distance_2", "ortho_error",
                   "normal_x", "normal_y"]
        faces = pd.DataFrame(records, columns=columns)
        return faces[faces.shared_length > tolerance].reset_index(drop=True)

    def _overlaps(self):
        """Plan overlaps between the columns of grids used in consecutive layers."""
        pairs = {tuple(sorted(p)) for p in zip(self._layer_grid[:-1], self._layer_grid[1:]) if p[0] != p[1]}
        geoms = self._columns.geometry.to_numpy()
        frames = []
        for g1, g2 in sorted(pairs):
            a_cols, b_cols = self._col_of[g1], self._col_of[g2]
            a_only = a_cols[~np.isin(a_cols, b_cols)]
            b_only = b_cols[~np.isin(b_cols, a_cols)]
            if not len(a_only) or not len(b_only):
                continue
            ia, ib = shapely.STRtree(geoms[b_only]).query(geoms[a_only], predicate="intersects")
            area = shapely.area(shapely.intersection(geoms[a_only][ia], geoms[b_only][ib]))
            frames.append(pd.DataFrame({"col_a": a_only[ia], "col_b": b_only[ib], "area": area}))
        if not frames:
            return pd.DataFrame({"col_a": [], "col_b": [], "area": []})
        overlaps = pd.concat(frames, ignore_index=True)
        # Separate grid objects can reuse the same global plan columns.
        # overlay_links reuses a column pair at each layer transition, so
        # collecting that pair twice would create duplicate DISU links.
        pairs = np.sort(overlaps[["col_a", "col_b"]].to_numpy(dtype=np.int64), axis=1)
        overlaps = overlaps.loc[~pd.DataFrame(pairs).duplicated().to_numpy()]
        overlaps = overlaps[overlaps.area > self._touch_area[overlaps.col_a.to_numpy(dtype=int)]]
        overlaps, merged = _merge_plan_slivers(overlaps, self._area, self.options.sliver_abs_tol)
        self.diagnostics["plan_slivers_merged"] = merged
        return overlaps

    def _chained_links(self, status, nodemap, top_l, bot_l):
        """IHC=0 links through pass-through runs that cross more than one change of grid.

        ``overlay_links`` joins the nearest active cells on either side of one
        change of grid. A pass-through run that ends at a further change of
        grid (grids A, B, A with a pass-through B cell) is followed across each
        one, and the active cells at its two ends are joined over the plan
        area they share through it. As in a single column with
        ``collapse=None``, the run's thickness is skipped.
        """
        nlay = status.shape[0]
        present = status != ABSENT
        up, down = _columns.nearest_active(status)
        geoms = self._columns.geometry.to_numpy()
        starting = [np.flatnonzero(~present[s] & present[s + 1]) for s in range(nlay - 1)]
        # Boundary below which a pass-through run starting at a change of grid leaves its column (-1: none).
        exit_at = np.full(status.shape, -1, dtype=np.int64)
        for s, cols in enumerate(starting):
            run = status[s + 1, cols] == PASSTHROUGH
            for t in range(s + 1, nlay - 1):
                exit_at[s + 1, cols[run & (status[t + 1, cols] == ABSENT)]] = t
                run &= status[t + 1, cols] == PASSTHROUGH
        trees = [shapely.STRtree(geoms[cols]) for cols in starting]
        found = {}
        for s in range(nlay - 1):
            bridges = starting[s][exit_at[s + 1, starting[s]] >= 0]
            ending = np.flatnonzero(present[s] & ~present[s + 1] & (up[s] >= 0))
            if not len(bridges) or not len(ending):
                continue
            ia, ib = shapely.STRtree(geoms[bridges]).query(geoms[ending], predicate="intersects")
            stack = [(nodemap[up[s, a], a], shapely.intersection(geoms[a], geoms[b]), exit_at[s + 1, b])
                     for a, b in zip(ending[ia], bridges[ib])]
            while stack:
                node, piece, t = stack.pop()
                if piece.area <= self._touch_area.min():
                    continue
                cols = starting[t][trees[t].query(piece, predicate="intersects")]
                for c, part in zip(cols, shapely.intersection(piece, geoms[cols])):
                    if down[t + 1, c] >= 0:
                        key = (node, nodemap[down[t + 1, c], c])
                        found[key] = found.get(key, 0.0) + part.area
                    elif exit_at[t + 1, c] >= 0:
                        stack.append((node, part, exit_at[t + 1, c]))
        if not found:
            return concat_links()
        n, m = (np.array(v, dtype=np.int64) for v in zip(*found))
        area = np.array(list(found.values()))
        keep = area > self._touch_area[self._node_cell[n]]
        n, m, area = n[keep], m[keep], area[keep]
        self.diagnostics["chained_links"] = len(n)
        self.diagnostics["largest_skipped_run"] = float(np.max(self._node_bot[n] - self._node_top[m], initial=0.0))
        return {
            "n": n, "m": m, "ihc": np.zeros(len(n), dtype=np.int64),
            "cl_n": (self._node_top[n] - self._node_bot[n]) / 2,
            "cl_m": (self._node_top[m] - self._node_bot[m]) / 2,
            "hwva": area, "height": np.zeros(len(n)), "angle": np.zeros(len(n)), "fahl": area,
        }

    def _per_layer(self, values, name):
        """Split values shaped like ``botm`` into one array per layer."""
        if isinstance(values, (list, tuple)):
            if len(values) != self.nlay:
                raise ValueError(f"{name} needs one entry per layer ({self.nlay}).")
            parts = values
        else:
            values = np.asarray(values)
            if not self.single_grid or values.ndim != 2 or values.shape[0] != self.nlay:
                raise ValueError(f"{name} must be shaped like botm.")
            parts = list(values)
        out = []
        for layer, part in enumerate(parts):
            size = len(self._col_of[self._layer_grid[layer]])
            part = np.broadcast_to(np.asarray(part), (size,))
            out.append(part)
        return out

    def _public(self, lattice):
        if self.single_grid:
            return lattice[:, self._col_of[0]]
        return [lattice[layer, self._col_of[g]] for layer, g in enumerate(self._layer_grid)]

    def _cellmap(self, top_l, bot_l):
        frames = []
        for layer, g in enumerate(self._layer_grid):
            cols = self._col_of[g]
            grid = self._grids[g]
            frame = pd.DataFrame({
                "node": self._nodemap[layer, cols], "layer": layer, "icpl": np.arange(len(cols)),
                "grid": g, "column": cols,
                "node_id": grid["node_id"].to_numpy() if "node_id" in grid else np.arange(len(cols)),
                "status": self.status[layer, cols],
                "x": grid["x"].to_numpy(dtype=float), "y": grid["y"].to_numpy(dtype=float),
                "top": top_l[layer, cols], "bot": bot_l[layer, cols],
            })
            if "zone_id" in grid:
                frame["zone_id"] = grid["zone_id"].to_numpy()
            frames.append(frame)
        cellmap = pd.concat(frames, ignore_index=True)
        cellmap["status"] = pd.Categorical.from_codes(
            cellmap["status"].to_numpy(), [STATUS_NAMES[code] for code in range(len(STATUS_NAMES))],
        )
        return cellmap

    def _require_build(self):
        if self.nodemap is None:
            raise RuntimeError("Call build() first.")


def split_contacts(grids, top, botm, idomain=None, *, wedge="upper", min_thickness=0.0, layers=None):
    """Keep the given bottoms at a change of grid by adding a wedge layer under a flat cut.

    :class:`DisuGrid` makes a contact where the plan grid changes gap-free:
    cells that overlap across it share one elevation. Each cell has one top,
    so a dipping contact is a staircase on each grid; under a refinement,
    fine cells straddle the coarse steps, the groups chain together, and the
    whole contact becomes one flat level that moves the bottoms above it
    (``report()`` gives the largest move). ``split_contacts`` instead cuts
    each such group flat and fills the space between the cut and the given
    contact with one new layer:

    - ``wedge="upper"`` puts the wedge on the grid above (the coarse one when
      refining at depth), between each upper bottom and the lowest one. It
      belongs to the layer below. It adds the fewest cells.
    - ``wedge="lower"`` puts it on the grid below, between the highest
      contact and the contact at each lower cell, so it follows the contact
      on the finer grid. It belongs to the layer above. The contact at the
      lower cells is the ``top`` given for that layer, else the bottom of the
      cell above each cell's centre.

    A change of grid at depth suits near-vertical features such as a pit or
    a fault zone. For subhorizontal features (dipping less than about 45
    degrees), use the same plan grid in every layer instead.

    Args:
        grids, top, botm, idomain: The layer inputs of :class:`DisuGrid`.
        wedge: ``"upper"`` or ``"lower"``.
        min_thickness: Wedge pieces this thin or thinner are not made; the
            bottom next to them moves to the cut instead, by at most this much.
        layers: Split only the contacts above these layers; ``None`` for
            every change of grid.

    Returns ``(grids, top, botm, idomain, parent)``, lists per layer ready for
    ``DisuGrid(grids, top, botm, idomain)``, and ``parent``, the input layer
    of each new layer (a wedge's parent is the layer it belongs to). Outside
    the cut groups the wedge has zero thickness and idomain -1; it takes its
    idomain from the cell it belongs to. Contacts whose bottoms already agree
    (nested refinement) are left alone; with nothing to split, the inputs
    come back unchanged. If the layers came from :func:`subdivide`, its
    ``parent[split_parent]`` maps the new layers to the original ones.
    """
    if wedge not in WEDGES:
        raise ValueError(f"wedge must be one of {WEDGES}, got {wedge!r}.")
    if isinstance(min_thickness, bool) or not np.isfinite(min_thickness) or min_thickness < 0:
        raise ValueError(f"min_thickness must be a finite number >= 0, got {min_thickness!r}.")
    d = DisuGrid(grids, top, botm, idomain)
    unchanged = (grids, top, botm, idomain, np.arange(d.nlay))
    if d.single_grid:
        return unchanged
    d._prepare_columns(_coordinate_tolerance(d._grids))
    tops = [None if t is None else t.copy() for t, _, _ in d._layers]
    bots = [b.copy() for _, b, _ in d._layers]
    idos = [i.copy() for _, _, i in d._layers]
    thin = max(min_thickness, _columns.roundoff(*bots, *[t for t in tops if t is not None]))
    wedges, changed = {}, False
    for k in range(1, d.nlay):
        if layers is None or k in layers:
            split, layer = _split_contact(d, k, tops, bots, idos, wedge, thin)
            changed |= split
            if layer is not None:
                wedges[k] = layer
    if not changed:
        return unchanged
    out = []
    for k in range(d.nlay):
        if wedge == "lower" and k in wedges:
            out.append(wedges[k])
        out.append((d._layer_grid[k], tops[k], bots[k], idos[k], k))
        if wedge == "upper" and k + 1 in wedges:
            out.append(wedges[k + 1])
    g, t, b, i, parent = zip(*out)
    return [d._grids[k] for k in g], list(t), list(b), list(i), np.array(parent)


def _split_contact(d, k, tops, bots, idos, wedge, thin):
    """Cut the contact above layer ``k`` flat, updating ``tops``/``bots`` in place.

    Returns ``(changed, wedge)``, where ``wedge`` is ``(grid, top, botm,
    idomain, parent)`` of the new layer, or None when no piece is thicker
    than ``thin``.
    """
    g_up, g_lo = d._layer_grid[k - 1], d._layer_grid[k]
    up_cols, lo_cols = d._col_of[g_up], d._col_of[g_lo]
    ending, starting = ~np.isin(up_cols, lo_cols), ~np.isin(lo_cols, up_cols)
    if g_up == g_lo or not ending.any() or not starting.any():
        return False, None
    geoms = d._columns.geometry.to_numpy()
    group_s, group_e, n_groups, (s_idx, e_idx, overlap) = _contact_groups(
        geoms, lo_cols[starting], up_cols[ending], d._touch_area)
    bottom = bots[k - 1][ending]
    _, lowest, highest = _group_bottoms(group_e, bottom, d._area[up_cols[ending]], n_groups)
    given = None if tops[k] is None else tops[k][starting]

    if wedge == "upper":
        need = highest > lowest
        if not need.any():
            return False, None
        in_e = need[group_e]
        cut = np.where(in_e, lowest[group_e], bottom)
        made = in_e & (bottom - cut > thin)
        bots[k - 1][ending] = np.where(made, bottom, cut)
        if given is not None:
            tops[k][starting] = np.where(need[group_s], lowest[group_s], given)
        if not made.any():
            return True, None
        w_bot = bots[k - 1].copy()
        w_bot[ending] = cut
        # Each wedge cell belongs to the lower cell it overlaps most.
        best = np.zeros(len(bottom), dtype=np.int64)
        order = np.argsort(overlap, kind="stable")
        best[e_idx[order]] = s_idx[order]
        w_ido = np.full(len(up_cols), -1, dtype=np.int64)
        w_ido[np.flatnonzero(ending)[made]] = idos[k][starting][best[made]]
        return True, (g_up, None, w_bot, w_ido, k)

    above = d._cell_above(k, lo_cols[starting])
    contact = given if given is not None else bots[k - 1][above]
    c_lo = np.full(n_groups, np.inf)
    c_hi = np.full(n_groups, -np.inf)
    np.minimum.at(c_lo, group_s, contact)
    np.maximum.at(c_hi, group_s, contact)
    cut_level = np.maximum(highest, c_hi)
    need = np.isfinite(lowest) & (cut_level > np.minimum(lowest, c_lo))
    if not need.any():
        return False, None
    in_s, in_e = need[group_s], need[group_e]
    cut = np.where(in_s, cut_level[group_s], contact)
    floor = np.minimum(contact, cut)
    made = in_s & (cut - floor > thin)
    bots[k - 1][ending] = np.where(in_e, cut_level[group_e], bottom)
    upper_top = tops[0] if k == 1 else (bots[k - 2] if d._layer_grid[k - 2] == g_up else None)
    if upper_top is not None and (bots[k - 1] > upper_top).any():
        raise ValueError(f"Layer {k - 1} is too thin for a lower wedge under it; use wedge='upper'.")
    if not made.any():
        if given is not None:
            tops[k][starting] = cut
        return True, None
    bot_above = np.full(len(d._columns), np.nan)
    bot_above[up_cols] = bots[k - 1]
    w_bot = bot_above[lo_cols]
    w_bot[starting] = np.where(made, floor, cut)
    w_top = None
    if given is not None:
        w_top = w_bot.copy()
        w_top[starting] = cut
    tops[k] = None
    w_ido = np.full(len(lo_cols), -1, dtype=np.int64)
    w_ido[np.flatnonzero(starting)[made]] = idos[k - 1][above[made]]
    return True, (g_lo, w_top, w_bot, w_ido, k - 1)


def _contact_groups(geoms, starting, ending, touch_area):
    """Join the columns that overlap in plan across a change of grid into groups.

    ``starting`` columns begin below the change and ``ending`` columns stop
    above it. Returns ``(group of each starting column, group of each ending
    column, number of groups, overlaps)``, with ``overlaps`` as
    ``(starting index, ending index, area)``.
    """
    n_s, n_e = len(starting), len(ending)
    s_idx = e_idx = np.zeros(0, dtype=np.int64)
    overlap = np.zeros(0)
    if n_e:
        s_idx, e_idx = shapely.STRtree(geoms[ending]).query(geoms[starting], predicate="intersects")
        overlap = shapely.area(shapely.intersection(geoms[starting][s_idx], geoms[ending][e_idx]))
        keep = overlap > touch_area[starting[s_idx]]
        s_idx, e_idx, overlap = s_idx[keep], e_idx[keep], overlap[keep]
    graph = coo_matrix((np.ones(len(s_idx)), (s_idx, n_s + e_idx)), shape=(n_s + n_e, n_s + n_e))
    n_groups, label = connected_components(graph, directed=False)
    return label[:n_s], label[n_s:], n_groups, (s_idx, e_idx, overlap)


def _group_bottoms(group, bottom, weight, n_groups):
    """Per group: area-weighted mean, lowest and highest bottom (NaN, inf and -inf for groups without one).

    The mean is exactly the bottoms when they agree.
    """
    total = np.bincount(group, weights=weight, minlength=n_groups)
    level = np.full(n_groups, np.nan)
    joined = total > 0
    level[joined] = np.bincount(group, weights=weight * bottom, minlength=n_groups)[joined] / total[joined]
    lowest = np.full(n_groups, np.inf)
    highest = np.full(n_groups, -np.inf)
    np.minimum.at(lowest, group, bottom)
    np.maximum.at(highest, group, bottom)
    agree = joined & (lowest == highest)
    level[agree] = lowest[agree]
    return level, lowest, highest


def _coordinate_tolerance(grids):
    """Coordinate roundoff of the grids, as used for their plan faces."""
    coords = np.concatenate([shapely.get_coordinates(g.geometry.to_numpy()) for g in grids])
    return 32 * np.finfo(float).eps * max(1.0, float(np.abs(coords).max()))


def splice(base, fine, polygon):
    """A plan grid with ``base`` cells outside ``polygon`` and ``fine`` cells inside it.

    Both grids must have the polygon outline as cell faces, which vorflow
    gives when the outline is added as a barrier line
    (``add_line(..., is_barrier=True)``) in the blueprint of both. No cell may
    cross the outline. The nodes along the outline need not match: the
    connections there come from the shared face segments (hanging nodes).

    The base cells are kept unchanged, so in :class:`DisuGrid` they continue
    the base grid's columns. Fine cells get new ``node_id`` values after the
    base ones; ``splice_source`` and ``source_node_id`` record their origin.
    """
    for name, grid in (("base", base), ("fine", fine)):
        missing = {"geometry", "x", "y"}.difference(grid.columns)
        if missing:
            raise ValueError(f"{name} grid missing required columns: {sorted(missing)}.")
    polygon = shapely.make_valid(polygon)
    parts = []
    for name, grid, keep_inside in (("base", base, False), ("fine", fine, True)):
        geoms = grid.geometry.to_numpy()
        area = shapely.area(geoms)
        inside = shapely.area(shapely.intersection(geoms, polygon))
        tolerance = 1e-9 * area
        crossing = (inside > tolerance) & (area - inside > tolerance)
        if crossing.any():
            raise ValueError(
                f"{int(crossing.sum())} {name} cells cross the splice outline; add the outline as a "
                "barrier line (is_barrier=True) so it becomes cell faces."
            )
        selected = grid[(inside > tolerance) == keep_inside].copy()
        selected["splice_source"] = name
        selected["source_node_id"] = (selected["node_id"].to_numpy() if "node_id" in selected
                                      else np.flatnonzero((inside > tolerance) == keep_inside))
        parts.append(selected)
    outside, inside = parts
    inside_area, polygon_area = inside.geometry.area.sum(), polygon.intersection(
        shapely.union_all(base.geometry.to_numpy())).area
    if abs(inside_area - polygon_area) > 1e-9 * max(polygon_area, 1.0):
        raise ValueError(
            f"Fine cells cover {inside_area:.6g} of the {polygon_area:.6g} splice area; the fine grid "
            "must cover the whole area inside the outline."
        )
    start = int(outside["node_id"].max()) + 1 if "node_id" in outside and len(outside) else 0
    inside["node_id"] = np.arange(start, start + len(inside))
    if "node_id" not in outside:
        outside["node_id"] = outside["source_node_id"]
    return gpd.GeoDataFrame(pd.concat([outside, inside], ignore_index=True), crs=base.crs)


def refill(base, polygon, resolution, *, growth_factor=None, fields=None, outline_split=1,
           priority="resolution"):
    """Re-mesh the inside of ``polygon`` with finer Voronoi cells that share the outline's vertices.

    The outline must be made of cell faces in ``base``, which vorflow gives
    when it is added as a barrier line (``add_line(..., is_barrier=True)``).
    Cells outside stay exactly as in ``base``, so their columns continue
    through every layer; the new cells inside end at the same outline
    vertices, so every outline face is shared one to one (no hanging
    vertices). Use the result for the layers that need the finer cells, as
    with :func:`splice`.

    Args:
        base: The vorflow grid to start from.
        polygon: The area to re-mesh: one polygon without holes.
        resolution: Target cell size inside.
        growth_factor: Cell-to-cell growth ratio (> 1) of the default
            :class:`~vorflow.fields.GeometricGrowthField`, as in
            ``add_polygon``: cells grow from ``resolution`` in the core to
            the outline spacing at the outline. Defaults to 1.2. Ignored when
            ``fields`` is given.
        fields: List of size fields (``MeshField`` objects), as in
            ``add_polygon``, applied outwards from a core of ``resolution``
            cells. The core is inset from the outline by the distance at
            which the fields reach 99 % of the outline spacing (at most half
            the polygon's inradius), so each field must give its size with
            ``python_size`` (the built-in growth fields do).
        outline_split: Split every outline face into this many faces on the
            inside, so the cells along the outline start finer (e.g. 2 for
            half the outline spacing). The base vertices are still shared;
            the extra ones are hanging vertices on the outside cells' faces,
            as in a quadtree grid.
        priority: What gives way when the fields cannot grow from
            ``resolution`` to the outline spacing within half the inradius.
            ``"resolution"`` (default) keeps the core at ``resolution`` and
            warns, since the cells then jump in size at the outline;
            ``"growth"`` makes the core cells larger, just enough for a
            smooth grading.

    Returns the spliced grid (base cells outside, new cells inside, with
    ``splice_source`` and ``source_node_id`` as in :func:`splice`). Its
    ``attrs["refill"]`` holds the number of new cells, the outline spacing,
    the core cell size, the size the fields reach at the outline and the
    largest area ratio between neighbouring new cells.
    """
    from . import _refill
    from .blueprint import _validate_growth_factor
    from .fields import DEFAULT_GROWTH_FACTOR, GeometricGrowthField, MeshField

    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError(f"resolution must be a positive number, got {resolution!r}.")
    growth_factor = _validate_growth_factor(growth_factor)
    if fields is None:
        fields = [GeometricGrowthField(growth_factor or DEFAULT_GROWTH_FACTOR)]
    fields = list(fields)
    if not fields or not all(isinstance(f, MeshField) for f in fields):
        raise ValueError("fields must be a non-empty list of MeshField objects.")
    if isinstance(outline_split, bool) or int(outline_split) != outline_split or outline_split < 1:
        raise ValueError(f"outline_split must be a positive integer, got {outline_split!r}.")
    if priority not in REFILL_PRIORITIES:
        raise ValueError(f"priority must be one of {REFILL_PRIORITIES}, got {priority!r}.")
    polygon = shapely.make_valid(polygon)
    if polygon.geom_type != "Polygon" or len(polygon.interiors):
        raise ValueError("refill takes one polygon without holes; call it once per area.")
    ring = polygon.exterior
    geoms = base.geometry.to_numpy()
    area = shapely.area(geoms)
    inside = shapely.area(shapely.intersection(geoms, polygon))
    crossing = (inside > 1e-9 * area) & (area - inside > 1e-9 * area)
    if crossing.any():
        raise ValueError(
            f"{int(crossing.sum())} base cells cross the refill outline; add the outline as a barrier "
            "line (is_barrier=True) so it becomes cell faces."
        )
    tolerance = 1e-7 * ring.length
    touching = geoms[shapely.dwithin(geoms, ring, tolerance)]
    base_nodes = _refill.outline_nodes(touching, ring, tolerance)
    nodes = _refill.subdivide_outline(base_nodes, int(outline_split))
    sites_in, sites_out, centres, radii = _refill.mirrored_sites(nodes, polygon)
    segment = np.linalg.norm(np.roll(nodes, -1, axis=0) - nodes, axis=1)
    spacing = float(np.median(segment))
    core_size, inset, size_at_outline = _refill_core(polygon, spacing, float(resolution), fields, priority)
    candidates = _refill_mesh_nodes(polygon, spacing, core_size, inset, fields)
    fill = _refill.fill_nodes(candidates, centres, radii, np.vstack([sites_in, sites_out]),
                              min(float(resolution), spacing))
    generators = np.vstack([sites_in, fill, sites_out])
    n = len(nodes)
    own_site = [(i, (i + 1) % n) for i in range(n)]
    own = own_site + [(-1, -1)] * len(fill) + own_site
    _refill.check_circles(generators, centres, radii, own)
    keep = np.arange(n + len(fill))
    cells = _refill.voronoi_cells(generators, polygon, keep, nodes, tolerance)
    new = gpd.GeoDataFrame({"x": generators[keep, 0], "y": generators[keep, 1]}, geometry=cells, crs=base.crs)
    new_nodes = _refill.outline_nodes(cells, ring, tolerance)
    shared = _shared_points(base_nodes, new_nodes, tolerance)
    if shared < len(base_nodes):
        raise ValueError(
            f"refill: only {shared} of the {len(base_nodes)} outline vertices are shared; please report this."
        )
    result = splice(base, new, polygon)
    result.attrs["refill"] = {
        "cells": len(new), "outline_vertices": len(base_nodes), "outline_spacing": spacing,
        "core_size": core_size, "size_at_outline": size_at_outline,
        "max_neighbour_area_ratio": _max_neighbour_ratio(cells),
    }
    return result


def _shared_points(a, b, tolerance):
    """How many points of ``a`` have a point of ``b`` within ``tolerance``."""
    from scipy.spatial import cKDTree

    distance, _ = cKDTree(b).query(a)
    return int((distance <= tolerance).sum())


def _field_size(fields, distance, spacing, core_size):
    """Size the fields (their minimum) ask for at ``distance`` from a core of ``core_size`` cells."""
    sizes = [f.python_size(np.asarray(distance), spacing, core_size) for f in fields]
    if any(s is None for s in sizes):
        raise ValueError("refill needs fields that give their size with python_size().")
    return float(np.min(sizes))


def _refill_core(polygon, spacing, resolution, fields, priority):
    """Core cell size, its inset from the outline, and the size the fields reach at the outline.

    The inset is where the fields reach 99 % of ``spacing``, at most half the
    polygon's inradius. If they cannot reach it there, ``"growth"`` raises
    the core size by bisection until they do; ``"resolution"`` keeps it and
    warns.
    """
    if resolution >= spacing:
        return resolution, 0.0, spacing
    cap = 0.5 * shapely.length(shapely.maximum_inscribed_circle(polygon))
    target = 0.99 * spacing
    core_size = resolution
    reached = _field_size(fields, cap, spacing, resolution)
    if reached < target:
        if priority == "growth" and _field_size(fields, cap, spacing, spacing) >= target:
            lo, hi = resolution, spacing
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if _field_size(fields, cap, spacing, mid) < target else (lo, mid)
            core_size = hi
        else:
            warnings.warn(
                f"refill: growing from {resolution:.3g}, the fields reach only {reached:.3g} at the outline, "
                f"whose spacing is {spacing:.3g}, so cells jump in size there. Use priority='growth' for "
                "larger core cells, a larger growth_factor, or outline_split for a finer outline.",
                UserWarning, stacklevel=3,
            )
    inset = _refill_inset(fields, spacing, core_size, cap)
    return core_size, inset, min(_field_size(fields, inset, spacing, core_size), spacing)


def _refill_inset(fields, spacing, core_size, cap):
    """Smallest distance at which ``fields`` reach 99 % of ``spacing``, at most ``cap``."""
    target = 0.99 * spacing
    if _field_size(fields, cap, spacing, core_size) < target:
        return cap
    lo, hi = 0.0, cap
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _field_size(fields, mid, spacing, core_size) < target else (lo, mid)
    return hi


def _refill_mesh_nodes(polygon, spacing, core_size, inset, fields):
    """Gmsh nodes filling ``polygon``: ``spacing`` at the outline, graded by ``fields`` to ``core_size`` inside."""
    from ._log import verbosity_scope
    from .blueprint import ConceptualMesh
    from .engine import MeshGenerator

    with verbosity_scope(0):
        cm = ConceptualMesh()
        cm.add_polygon(polygon, zone_id=1, resolution=max(spacing, core_size))
        if core_size < spacing:
            core = polygon.buffer(-inset, join_style="mitre")
            if not core.is_empty:
                cm.add_polygon(core, zone_id=2, resolution=core_size, embed=False, fields=fields)
        mg = MeshGenerator(background_lc=max(spacing, core_size), conceptual_mesh=cm, verbosity=0)
        mg.generate()
    return np.asarray(mg.nodes, dtype=float)[:, :2]


def _max_neighbour_ratio(cells):
    """Largest area ratio between two cells that share a face."""
    cells = np.asarray(cells)
    area = shapely.area(cells)
    i, j = shapely.STRtree(cells).query(cells, predicate="touches")
    pair = i < j
    i, j = i[pair], j[pair]
    shared = shapely.length(shapely.intersection(cells[i], cells[j]))
    i, j = i[shared > 0], j[shared > 0]
    if not len(i):
        return 1.0
    return float((np.maximum(area[i], area[j]) / np.minimum(area[i], area[j])).max())


def _layer_inputs(grids, top, botm, idomain):
    """Normalize grids and layer arrays into ``(unique grids, grid per layer, per-layer arrays)``."""
    if isinstance(grids, (list, tuple)):
        if not grids:
            raise ValueError("grids must not be empty.")
        unique, layer_grid = [], []
        for grid in grids:
            index = next((i for i, g in enumerate(unique) if g is grid), None)
            if index is None:
                unique.append(grid)
                index = len(unique) - 1
            layer_grid.append(index)
    else:
        unique, layer_grid = [grids], None
    for grid in unique:
        missing = {"geometry", "x", "y"}.difference(getattr(grid, "columns", ()))
        if missing:
            raise ValueError(f"grid missing required columns: {sorted(missing)}.")
        if len(grid) == 0:
            raise ValueError("grid must not be empty.")

    if len(unique) == 1:
        top_l, bot_l, ido = _columns.layer_arrays(top, botm, idomain, ncpl=len(unique[0]))
        nlay = bot_l.shape[0]
        if layer_grid is not None and len(layer_grid) != nlay:
            raise ValueError(f"grids lists {len(layer_grid)} layers but botm has {nlay}.")
        layers = [(top_l[0] if k == 0 else None, bot_l[k], ido[k]) for k in range(nlay)]
        return unique, [0] * nlay, layers

    if not isinstance(botm, (list, tuple)):
        raise ValueError("With different grids per layer, botm must be a list with one entry per layer.")
    nlay = len(botm)
    if layer_grid is None:
        layer_grid = [0] * nlay
    if len(layer_grid) != nlay:
        raise ValueError(f"grids lists {len(layer_grid)} layers but botm has {nlay}.")
    tops = top if isinstance(top, (list, tuple)) else [top] + [None] * (nlay - 1)
    idos = idomain if isinstance(idomain, (list, tuple)) else [idomain] * nlay
    if len(tops) != nlay or len(idos) != nlay:
        raise ValueError(f"Per-layer top and idomain lists need {nlay} entries.")
    if tops[0] is None:
        raise ValueError("The top of the first layer is required.")
    layers = []
    for k in range(nlay):
        size = len(unique[layer_grid[k]])
        bot = np.broadcast_to(np.asarray(botm[k], dtype=float), (size,)).copy()
        layer_top = None if tops[k] is None else np.broadcast_to(np.asarray(tops[k], dtype=float), (size,)).copy()
        ido = np.ones(size, dtype=np.int64) if idos[k] is None else np.broadcast_to(
            np.asarray(idos[k]), (size,)).astype(np.int64)
        if not np.isfinite(bot).all() or (layer_top is not None and not np.isfinite(layer_top).all()):
            raise ValueError(f"Layer {k} elevations must be finite.")
        layers.append((layer_top, bot, ido))
    return unique, layer_grid, layers


def _plan_columns(grids):
    """Global plan columns: cells with the same generator and footprint in several grids are one column."""
    rows, col_of, index = [], [], {}
    for g, grid in enumerate(grids):
        geoms = grid.geometry.to_numpy()
        xy = grid[["x", "y"]].to_numpy(dtype=float)
        ids = np.empty(len(grid), dtype=np.int64)
        for i, (geom, (x, y)) in enumerate(zip(geoms, xy)):
            match = next((c for c in index.get((x, y), ()) if rows[c][0].equals(geom)), None)
            if match is None:
                match = len(rows)
                rows.append((geom, x, y, g, i))
                index.setdefault((x, y), []).append(match)
            ids[i] = match
        col_of.append(ids)
    geom, x, y, g, i = zip(*rows)
    columns = gpd.GeoDataFrame({"x": x, "y": y, "grid": g, "icpl": i, "node_id": np.arange(len(rows))},
                               geometry=list(geom))
    return col_of, columns


def _global_faces(face_tables, col_of):
    """Plan faces of every grid, renumbered to global columns, each column pair once."""
    frames = []
    for faces, cols in zip(face_tables, col_of):
        faces = faces.copy()
        faces["cell_id_1"] = cols[faces["cell_id_1"].to_numpy(dtype=np.int64)]
        faces["cell_id_2"] = cols[faces["cell_id_2"].to_numpy(dtype=np.int64)]
        frames.append(faces)
    faces = pd.concat(frames, ignore_index=True)
    key = np.sort(faces[["cell_id_1", "cell_id_2"]].to_numpy(), axis=1)
    keep = ~pd.DataFrame(key).duplicated().to_numpy()
    return faces[keep].reset_index(drop=True)


def _bent_face(shared, centres, pair, tolerance):
    """One face for a contact that is not a single straight segment: total length, mean normal.

    None when the contact is no longer than ``tolerance`` (``_cross_faces`` drops those
    anyway) or its piece normals cancel, as they can for pieces a few 1e-11 m long.
    """
    from ._mfusg import _line_parts

    parts = _line_parts(shared)
    if not parts:
        return None
    lengths = np.array([part.length for part in parts])
    if lengths.sum() <= tolerance:
        return None
    vectors = np.array([np.subtract(part.coords[-1], part.coords[0])[:2] for part in parts])
    normals = np.column_stack([-vectors[:, 1], vectors[:, 0]]) / lengths[:, None]
    towards = centres[1] - centres[0]
    normals *= np.sign(normals @ towards)[:, None]
    normal = (normals * lengths[:, None]).sum(axis=0)
    if not np.linalg.norm(normal) > 0:
        return None
    normal /= np.linalg.norm(normal)
    midpoints = np.array([part.interpolate(0.5, normalized=True).coords[0][:2] for part in parts])
    face_point = (midpoints * lengths[:, None]).sum(axis=0) / lengths.sum()
    d1, d2 = (face_point - centres[0]) @ normal, (centres[1] - face_point) @ normal
    angle = np.degrees(np.arccos(np.clip(abs(towards @ normal) / np.linalg.norm(towards), 0, 1)))
    return {"cell_id_1": pair[0], "cell_id_2": pair[1], "shared_length": float(lengths.sum()),
            "distance_1": float(abs(d1)), "distance_2": float(abs(d2)), "ortho_error": float(angle),
            "normal_x": float(normal[0]), "normal_y": float(normal[1])}


def _node_values(values, nodes, name):
    """Positive finite values, one per active node (a scalar applies to all)."""
    try:
        values = np.broadcast_to(np.asarray(values, dtype=float), (nodes,)).copy()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number or have shape ({nodes},).") from error
    if not (np.isfinite(values).all() and (values > 0).all()):
        raise ValueError(f"{name} must hold positive finite values.")
    return values


def _merge_plan_slivers(overlaps, area, sliver_abs_tol=0.0):
    """Add overlaps smaller than ``sliver_abs_tol**2`` to the smaller cell's largest overlap."""
    if overlaps.empty or sliver_abs_tol == 0:
        return overlaps.reset_index(drop=True), 0
    overlaps = overlaps.reset_index(drop=True)
    a = overlaps.col_a.to_numpy(dtype=np.int64)
    b = overlaps.col_b.to_numpy(dtype=np.int64)
    small_cell = np.where(area[a] <= area[b], a, b)
    tiny = overlaps.area.to_numpy() < sliver_abs_tol ** 2
    if not tiny.any():
        return overlaps, 0
    merged_area = overlaps.area.to_numpy(dtype=float).copy()
    kept = ~tiny
    for row in np.flatnonzero(tiny):
        cell = small_cell[row]
        candidates = np.flatnonzero(kept & ((a == cell) | (b == cell)))
        if not len(candidates):
            kept[row] = True                    # The only overlap of that cell: keep it.
            continue
        target = candidates[np.argmax(merged_area[candidates])]
        merged_area[target] += merged_area[row]
    overlaps = overlaps.assign(area=merged_area)[kept].reset_index(drop=True)
    return overlaps, int((~kept).sum())


def _vertical_offset(links, top, bot):
    """Largest elevation overlap between the upper and lower cells of a vertical connection."""
    vertical = links["ihc"] == 0
    if not vertical.any():
        return 0.0
    n, m = links["n"][vertical], links["m"][vertical]
    upper = np.where(top[n] >= top[m], n, m)
    lower = np.where(upper == n, m, n)
    return float(max(0.0, (top[lower] - bot[upper]).max()))


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
    with np.errstate(invalid="ignore"):
        offset = np.maximum(np.abs(top_l[:, i] - top_l[:, j]), np.abs(bot_l[:, i] - bot_l[:, j]))
        thinner = np.minimum(top_l[:, i] - bot_l[:, i], top_l[:, j] - bot_l[:, j])
    ratio = np.where(both, offset / np.where(both, thinner, 1.0), 0.0)
    return {"max_offset_ratio": float(ratio.max()), "disconnected_faces": int((ratio >= 1).sum())}
