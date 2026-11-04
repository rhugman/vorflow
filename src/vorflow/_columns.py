"""Cell columns for DISU: layer arrays, pass-through, gap-free collapse, sublayers.

Pure array functions in the flopy layer convention: ``top`` is ``(ncpl,)``,
``botm`` and ``idomain`` are ``(nlay, ncpl)``. The upper surface of a layer is
the bottom of the layer above. No geometry and no flopy here.

``idomain`` follows MODFLOW 6: values above zero are active, zero is a
no-flow cell that never passes flow, and -1 is a vertical pass-through cell.
A cell no thicker than ``min_thickness`` (or roundoff) is treated as
pass-through as well.

When layers use different plan grids, the arrays cover every plan column of
every grid; a column missing from a layer's grid is ``ABSENT`` there (NaN
elevations). Like no-flow cells, absent cells end a column.
"""

import numpy as np

NOFLOW, ACTIVE, PASSTHROUGH, ABSENT = 0, 1, 2, 3
STATUS_NAMES = {NOFLOW: "noflow", ACTIVE: "active", PASSTHROUGH: "passthrough", ABSENT: "absent"}
COLLAPSE_MODES = ("split", "up", "down", None)
ROUNDOFF = 1e-9         # relative to the largest elevation: thinner cells have zero thickness


def roundoff(*elevations):
    """Thickness that counts as zero: ``ROUNDOFF`` times the largest elevation (at least 1)."""
    scale = max([1.0] + [float(np.nanmax(np.abs(e), initial=0.0)) for e in elevations])
    return ROUNDOFF * scale


def layer_arrays(top, botm, idomain=None, ncpl=None):
    """Return ``(top_l, bot_l, idomain)``, each ``(nlay, ncpl)``, from flopy-style inputs.

    ``top`` is scalar or ``(ncpl,)``; ``botm`` is ``(nlay,)`` (one value per
    layer) or ``(nlay, ncpl)``; ``idomain`` is ``None`` (all active), scalar or
    ``(nlay, ncpl)``. ``ncpl`` sizes scalar and per-layer inputs; by default it
    comes from ``top`` or ``botm``. Thickness may be zero, which marks a
    pinched cell, but not negative.
    """
    bot = np.asarray(botm, dtype=float)
    top = np.asarray(top, dtype=float)
    if ncpl is None:
        ncpl = top.size if top.ndim == 1 else (bot.shape[1] if bot.ndim == 2 else 1)
    if bot.ndim == 1:
        bot = np.broadcast_to(bot[:, None], (bot.size, ncpl)).copy()
    if bot.ndim != 2 or bot.shape[0] < 1 or bot.shape[1] < 1:
        raise ValueError("botm must have shape (nlay,) or (nlay, ncpl).")
    if bot.shape[1] != ncpl:
        raise ValueError(f"botm must have {ncpl} cells per layer, got {bot.shape[1]}.")
    nlay = bot.shape[0]
    if top.ndim == 0:
        top = np.full(ncpl, top.item())
    if top.shape != (ncpl,):
        raise ValueError(f"top must be scalar or have shape ({ncpl},).")
    top_l = np.vstack([top[None, :], bot[:-1]])
    if not (np.isfinite(top_l).all() and np.isfinite(bot).all()):
        raise ValueError("Layer elevations must be finite.")
    bad = np.argwhere(top_l < bot)
    if len(bad):
        layer, cell = bad[0]
        raise ValueError(
            f"Negative thickness at layer {layer}, cell {cell}: each bottom must lie "
            "at or below the bottom of the layer above."
        )
    if idomain is None:
        idomain = np.ones((nlay, ncpl), dtype=np.int64)
    else:
        idomain = np.asarray(idomain)
        if idomain.ndim == 0:
            idomain = np.full((nlay, ncpl), idomain.item())
        if idomain.shape != (nlay, ncpl):
            raise ValueError(f"idomain must have shape ({nlay}, {ncpl}).")
        if not np.issubdtype(idomain.dtype, np.integer):
            if not np.array_equal(idomain, np.round(idomain)):
                raise ValueError("idomain must hold integers.")
        idomain = idomain.astype(np.int64)
    return top_l, bot.copy(), idomain


def classify(top_l, bot_l, idomain, min_thickness=0.0):
    """Status per cell: ``NOFLOW`` (idomain 0), ``PASSTHROUGH`` (idomain < 0 or thin),
    ``ABSENT`` (NaN elevations) or ``ACTIVE``.

    A cell is thin at or below ``min_thickness``, or below roundoff
    (``ROUNDOFF`` times the largest elevation), which MODFLOW rejects as zero.
    """
    thickness = top_l - bot_l
    status = np.full(idomain.shape, ACTIVE, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        thin = thickness <= max(min_thickness, roundoff(top_l, bot_l))
    status[(idomain < 0) | thin] = PASSTHROUGH
    status[idomain == 0] = NOFLOW
    status[np.isnan(thickness)] = ABSENT
    return status


def vertical_pairs(status):
    """``(upper_layer, lower_layer, cell)`` for each pair of active cells linked through pass-through cells.

    A no-flow or absent cell breaks the column: flow never passes through it.
    """
    nlay, ncpl = status.shape
    above = np.full(ncpl, -1)
    upper, lower, cells = [], [], []
    for layer in range(nlay):
        active = status[layer] == ACTIVE
        linked = active & (above >= 0)
        upper.append(above[linked])
        lower.append(np.full(linked.sum(), layer))
        cells.append(np.flatnonzero(linked))
        above = np.where(active, layer, above)
        above = np.where((status[layer] == NOFLOW) | (status[layer] == ABSENT), -1, above)
    return (np.concatenate(upper).astype(np.int64), np.concatenate(lower).astype(np.int64),
            np.concatenate(cells).astype(np.int64))


def collapse(top_l, bot_l, status, mode="split"):
    """Give each run of pass-through cells to the active cells next to it, so columns stay gap-free.

    A run between two active cells is shared at its midpoint (``"split"``) or
    given to the cell above (``"up"``) or below (``"down"``). A run with an
    active cell on one side only (next to a no-flow cell or the top or bottom
    of the column) goes to that side. ``mode=None`` keeps every gap, as in
    MODFLOW 6 and LoopFlopy, so the vertical connection skips the run's thickness.

    Returns new ``(top_l, bot_l)``; only active cells change, and only by growing.
    """
    if mode not in COLLAPSE_MODES:
        raise ValueError(f"collapse must be one of {COLLAPSE_MODES}, got {mode!r}.")
    top_l, bot_l = top_l.copy(), bot_l.copy()
    if mode is None:
        return top_l, bot_l
    nlay, ncpl = status.shape
    active = status == ACTIVE
    passthrough = status == PASSTHROUGH
    absent = status == ABSENT
    # Where a column ends at an absent cell, the run ends at the neighbouring cell's face.
    stop_top = np.where(absent, np.vstack([bot_l[:1], bot_l[:-1]]), top_l)
    stop_bot = np.where(absent, np.vstack([top_l[1:], top_l[-1:]]), bot_l)
    # Extend each active cell down over the pass-through cells below it.
    floor = bot_l[-1].copy()            # Bottom of the run below; column bottom at first.
    below_is_active = np.zeros(ncpl, dtype=bool)
    reach_down = bot_l.copy()
    lower_neighbour = np.zeros((nlay, ncpl), dtype=bool)
    for layer in range(nlay - 1, -1, -1):
        reach_down[layer] = np.where(active[layer], floor, reach_down[layer])
        lower_neighbour[layer] = active[layer] & below_is_active
        # Moving up: a non-pass-through cell starts a new run for the cells above it.
        stops = ~passthrough[layer]
        floor = np.where(stops, stop_top[layer], floor)
        below_is_active = np.where(stops, active[layer], below_is_active)
    ceiling = top_l[0].copy()
    above_is_active = np.zeros(ncpl, dtype=bool)
    reach_up = top_l.copy()
    upper_neighbour = np.zeros((nlay, ncpl), dtype=bool)
    for layer in range(nlay):
        reach_up[layer] = np.where(active[layer], ceiling, reach_up[layer])
        upper_neighbour[layer] = active[layer] & above_is_active
        stops = ~passthrough[layer]
        ceiling = np.where(stops, stop_bot[layer], ceiling)
        above_is_active = np.where(stops, active[layer], above_is_active)
    # An active cell takes its whole run when no active cell shares it.
    new_bot = np.where(active & ~lower_neighbour, reach_down, bot_l)
    new_top = np.where(active & ~upper_neighbour, reach_up, top_l)
    # Runs between two active cells: the gap is [reach_down of the upper cell, bot of the upper cell].
    upper, lower, cells = vertical_pairs(status)
    gap_top, gap_bot = bot_l[upper, cells], top_l[lower, cells]
    if mode == "split":
        cut = gap_bot + (gap_top - gap_bot) / 2
    elif mode == "up":
        cut = gap_bot
    else:
        cut = gap_top
    new_bot[upper, cells] = cut
    new_top[lower, cells] = cut
    return new_top, new_bot


def nearest_active(status):
    """``(up, down)``: per cell, the layer of the nearest active cell at or above (below) it.

    Only pass-through cells may lie in between; -1 where there is none.
    """
    nlay, ncpl = status.shape
    up = np.full((nlay, ncpl), -1, dtype=np.int64)
    down = np.full((nlay, ncpl), -1, dtype=np.int64)
    last = np.full(ncpl, -1, dtype=np.int64)
    for layer in range(nlay):
        last = np.where(status[layer] == ACTIVE, layer, np.where(status[layer] == PASSTHROUGH, last, -1))
        up[layer] = last
    last = np.full(ncpl, -1, dtype=np.int64)
    for layer in range(nlay - 1, -1, -1):
        last = np.where(status[layer] == ACTIVE, layer, np.where(status[layer] == PASSTHROUGH, last, -1))
        down[layer] = last
    return up, down


def subdivide(top, botm, idomain=None, nsub=None, fractions=None, where=None):
    """Split layers into sublayers, everywhere or only in some columns.

    Args:
        top, botm, idomain: flopy-style layer arrays (see :func:`layer_arrays`).
        nsub: Sublayers per layer, as an int or one int per layer. Mutually
            exclusive with ``fractions``.
        fractions: One sequence per layer (or ``None`` to keep that layer
            whole) of positive thickness fractions, top to bottom.
        where: Columns to split: ``None`` (all), a ``(ncpl,)`` boolean mask,
            or one mask (or ``None``) per layer.

    Returns:
        ``(botm, idomain, parent)``: the new ``(nlay_new, ncpl)`` bottoms and
        idomain, and ``parent``, the original layer of each new layer.

    Outside ``where``, the parent cell keeps its whole interval in its first
    sublayer; the other sublayers there have zero thickness and idomain -1, so
    they pass flow through and are removed from the DISU grid.

    The arrays describe one plan grid used by every layer. For a different
    grid per layer, subdivide each grid's layers separately or use
    :func:`split_contacts`.
    """
    top_l, bot_l, idomain = layer_arrays(top, botm, idomain)
    nlay, ncpl = bot_l.shape
    if (nsub is None) == (fractions is None):
        raise ValueError("Give exactly one of nsub or fractions.")
    if fractions is None:
        counts = np.broadcast_to(np.asarray(nsub), (nlay,))
        if not np.issubdtype(counts.dtype, np.integer) or (counts < 1).any():
            raise ValueError("nsub must hold positive integers.")
        parts = [np.full(int(n), 1.0 / int(n)) for n in counts]
    else:
        if len(fractions) != nlay:
            raise ValueError(f"fractions needs one entry per layer ({nlay}).")
        parts = []
        for layer, f in enumerate(fractions):
            f = np.array([1.0] if f is None else f, dtype=float)
            if f.ndim != 1 or len(f) < 1 or (f <= 0).any() or not np.isfinite(f).all():
                raise ValueError(f"fractions for layer {layer} must be positive numbers.")
            parts.append(f / f.sum())
    masks = _column_masks(where, nlay, ncpl)
    new_bot, new_idomain, parent = [], [], []
    for layer in range(nlay):
        thickness = top_l[layer] - bot_l[layer]
        edges = np.cumsum(parts[layer])
        edges[-1] = 1.0                 # Exact parent bottom despite roundoff.
        for i, edge in enumerate(edges):
            split = top_l[layer] - edge * thickness
            # Unsplit columns: the first slot keeps the whole cell, later slots are empty.
            new_bot.append(np.where(masks[layer], split, bot_l[layer]))
            empty = ~masks[layer] & (i > 0) & (idomain[layer] != 0)
            new_idomain.append(np.where(empty, -1, idomain[layer]))
            parent.append(layer)
    return np.array(new_bot), np.array(new_idomain, dtype=np.int64), np.array(parent)


def _column_masks(where, nlay, ncpl):
    if where is None:
        return [np.ones(ncpl, dtype=bool)] * nlay
    if isinstance(where, (list, tuple)):
        if len(where) != nlay:
            raise ValueError(f"where needs one mask per layer ({nlay}).")
        return [_column_masks(w, 1, ncpl)[0] for w in where]
    mask = np.asarray(where)
    if mask.shape != (ncpl,) or mask.dtype != bool:
        raise ValueError(f"where must be a boolean mask of shape ({ncpl},).")
    return [mask] * nlay
