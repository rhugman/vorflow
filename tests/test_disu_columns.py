"""Column core of the DISU builder: layer arrays, pass-through and collapse."""

import numpy as np
import pytest

from vorflow._columns import (
    ACTIVE, NOFLOW, PASSTHROUGH, classify, collapse, layer_arrays, vertical_pairs,
)


def _column(bottoms, idomain=None, top=10.0):
    """One-column layer arrays: ``bottoms`` top to bottom."""
    botm = np.asarray(bottoms, dtype=float)[:, None]
    ido = None if idomain is None else np.asarray(idomain)[:, None]
    return layer_arrays(np.array([top]), botm, ido)


def test_layer_arrays_shapes_and_defaults():
    top_l, bot_l, idomain = layer_arrays(10.0, [[5.0, 4.0], [0.0, 0.0]])
    np.testing.assert_array_equal(top_l, [[10, 10], [5, 4]])
    np.testing.assert_array_equal(bot_l, [[5, 4], [0, 0]])
    np.testing.assert_array_equal(idomain, np.ones((2, 2)))
    top_l, bot_l, _ = layer_arrays([10.0, 9.0], [5.0, 0.0])
    np.testing.assert_array_equal(bot_l, [[5, 5], [0, 0]])


@pytest.mark.parametrize("top, botm, idomain, match", [
    (10.0, [[5.0], [6.0]], None, "Negative thickness at layer 1"),
    ([10.0, 9.0], [[5.0]], None, "2 cells per layer"),
    (10.0, [[5.0]], [[1], [1]], "idomain must have shape"),
    (10.0, [[5.0]], [[0.5]], "integers"),
    (np.nan, [[5.0]], None, "finite"),
])
def test_layer_arrays_rejects_bad_input(top, botm, idomain, match):
    with pytest.raises(ValueError, match=match):
        layer_arrays(top, botm, idomain)


def test_classify_noflow_passthrough_and_thin_cells():
    top_l, bot_l, idomain = _column([8, 7.9, 6, 5, 0], idomain=[1, 1, -1, 0, 1])
    np.testing.assert_array_equal(
        classify(top_l, bot_l, idomain, min_thickness=0.5)[:, 0],
        [ACTIVE, PASSTHROUGH, PASSTHROUGH, NOFLOW, ACTIVE],
    )
    # Zero thickness is pass-through even without a minimum thickness.
    top_l, bot_l, idomain = _column([8, 8, 0])
    np.testing.assert_array_equal(classify(top_l, bot_l, idomain)[:, 0], [ACTIVE, PASSTHROUGH, ACTIVE])


def test_classify_treats_roundoff_thickness_as_zero():
    # MODFLOW 6 stops on a cell a few ulps thick ("thickness <= 0"), so it passes flow through.
    top_l, bot_l, idomain = _column([50.0, 50.0 - 1e-14, 40.0], top=60.0)
    np.testing.assert_array_equal(classify(top_l, bot_l, idomain)[:, 0], [ACTIVE, PASSTHROUGH, ACTIVE])
    # The threshold follows the elevations (1e-9 of the largest): 1 um is still a cell.
    top_l, bot_l, idomain = _column([0.0, -1e-6, -1.0], top=1e-3)
    np.testing.assert_array_equal(classify(top_l, bot_l, idomain)[:, 0], [ACTIVE, ACTIVE, ACTIVE])


def test_vertical_pairs_pass_through_but_stop_at_noflow():
    status = np.array([[ACTIVE], [PASSTHROUGH], [ACTIVE], [NOFLOW], [PASSTHROUGH], [ACTIVE]])
    upper, lower, cells = vertical_pairs(status)
    assert list(zip(upper, lower, cells)) == [(0, 2, 0)]


@pytest.mark.parametrize("mode, expected_cut", [("split", 7.5), ("up", 7.0), ("down", 8.0)])
def test_collapse_shares_a_run_between_two_active_cells(mode, expected_cut):
    top_l, bot_l, idomain = _column([8, 7, 0], idomain=[1, -1, 1])
    status = classify(top_l, bot_l, idomain)
    new_top, new_bot = collapse(top_l, bot_l, status, mode)
    assert new_bot[0, 0] == expected_cut
    assert new_top[2, 0] == expected_cut
    # Untouched: column top and bottom, and the pass-through cell itself.
    assert new_top[0, 0] == 10 and new_bot[2, 0] == 0
    assert (new_top[1, 0], new_bot[1, 0]) == (8, 7)


def test_collapse_none_keeps_the_gap():
    top_l, bot_l, idomain = _column([8, 7, 0], idomain=[1, -1, 1])
    new_top, new_bot = collapse(top_l, bot_l, classify(top_l, bot_l, idomain), None)
    np.testing.assert_array_equal(new_top, top_l)
    np.testing.assert_array_equal(new_bot, bot_l)


def test_collapse_one_sided_runs_go_to_the_only_active_neighbour():
    # Pass-through at the column top, next to no-flow, and at the column bottom.
    top_l, bot_l, idomain = _column([9, 8, 6, 5, 4, 2, 1, 0], idomain=[-1, 1, -1, 0, -1, 1, -1, -1])
    status = classify(top_l, bot_l, idomain)
    new_top, new_bot = collapse(top_l, bot_l, status, "down")
    assert (new_top[1, 0], new_bot[1, 0]) == (10, 6)     # Takes the top run and the run above no-flow.
    assert (new_top[5, 0], new_bot[5, 0]) == (5, 0)      # Takes the run below no-flow and the bottom run.


def test_collapse_conserves_column_thickness():
    rng = np.random.default_rng(1)
    nlay, ncpl = 6, 50
    bot = np.sort(rng.uniform(-50, 9, (nlay, ncpl)), axis=0)[::-1]
    idomain = rng.choice([1, 1, 1, -1], size=(nlay, ncpl))
    idomain[0] = 1
    top_l, bot_l, idomain = layer_arrays(np.full(ncpl, 10.0), bot, idomain)
    status = classify(top_l, bot_l, idomain, min_thickness=1.0)
    for mode in ("split", "up", "down"):
        new_top, new_bot = collapse(top_l, bot_l, status, mode)
        active = status == ACTIVE
        thickness = np.where(active, new_top - new_bot, 0).sum(axis=0)
        has_active = active.any(axis=0)
        np.testing.assert_allclose(thickness[has_active], (top_l[0] - bot_l[-1])[has_active])
        # Active cells stack without gaps or overlaps.
        for cell in np.flatnonzero(has_active):
            layers = np.flatnonzero(active[:, cell])
            np.testing.assert_array_equal(new_bot[layers[:-1], cell], new_top[layers[1:], cell])
