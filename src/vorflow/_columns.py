"""Cell columns for DISU: layer arrays, pass-through and gap-free collapse.

Pure array functions in the flopy layer convention: ``top`` is ``(ncpl,)``,
``botm`` and ``idomain`` are ``(nlay, ncpl)``. The upper surface of a layer is
the bottom of the layer above. No geometry and no flopy here.

``idomain`` follows MODFLOW 6: values above zero are active, zero is a
no-flow cell that never passes flow, and -1 is a vertical pass-through cell.
A cell no thicker than ``min_thickness`` (or roundoff) is treated as
pass-through as well.
"""

import numpy as np

NOFLOW, ACTIVE, PASSTHROUGH = 0, 1, 2
STATUS_NAMES = {NOFLOW: "noflow", ACTIVE: "active", PASSTHROUGH: "passthrough"}
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
    """Status per cell: ``NOFLOW`` (idomain 0), ``PASSTHROUGH`` (idomain < 0 or thin) or ``ACTIVE``.

    A cell is thin at or below ``min_thickness``, or below roundoff
    (``ROUNDOFF`` times the largest elevation), which MODFLOW rejects as zero.
    """
    thickness = top_l - bot_l
    status = np.full(idomain.shape, ACTIVE, dtype=np.int8)
    status[(idomain < 0) | (thickness <= max(min_thickness, roundoff(top_l, bot_l)))] = PASSTHROUGH
    status[idomain == 0] = NOFLOW
    return status


def vertical_pairs(status):
    """``(upper_layer, lower_layer, cell)`` for each pair of active cells linked through pass-through cells.

    A no-flow cell breaks the column: flow never passes through it.
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
        above = np.where(status[layer] == NOFLOW, -1, above)
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
        floor = np.where(stops, top_l[layer], floor)
        below_is_active = np.where(stops, active[layer], below_is_active)
    ceiling = top_l[0].copy()
    above_is_active = np.zeros(ncpl, dtype=bool)
    reach_up = top_l.copy()
    upper_neighbour = np.zeros((nlay, ncpl), dtype=bool)
    for layer in range(nlay):
        reach_up[layer] = np.where(active[layer], ceiling, reach_up[layer])
        upper_neighbour[layer] = active[layer] & above_is_active
        stops = ~passthrough[layer]
        ceiling = np.where(stops, bot_l[layer], ceiling)
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
