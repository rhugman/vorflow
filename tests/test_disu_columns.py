"""Column core of the DISU builder: layer arrays, pass-through, collapse and sublayers."""

import numpy as np
import pytest

from vorflow._columns import (
    ABSENT, ACTIVE, NOFLOW, PASSTHROUGH, classify, collapse, layer_arrays, nearest_active, subdivide,
    vertical_pairs,
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


def test_nan_elevations_are_absent_and_end_a_column():
    # Layer 1 lacks this column (its grid has no such cell): NaN elevations.
    top_l = np.array([[10.0], [8.0], [np.nan], [4.0]])
    bot_l = np.array([[8.0], [6.0], [np.nan], [0.0]])
    status = classify(top_l, bot_l, np.ones((4, 1), dtype=int))
    np.testing.assert_array_equal(status[:, 0], [ACTIVE, ACTIVE, ABSENT, ACTIVE])
    upper, lower, _ = vertical_pairs(status)
    assert list(zip(upper, lower)) == [(0, 1)]                  # No link across the absent cell.


def test_a_pass_through_run_ending_at_an_absent_cell_stops_at_its_face():
    top_l = np.array([[10.0], [8.0], [np.nan]])
    bot_l = np.array([[8.0], [6.0], [np.nan]])
    status = np.array([[ACTIVE], [PASSTHROUGH], [ABSENT]])
    new_top, new_bot = collapse(top_l, bot_l, status, "split")
    assert (new_top[0, 0], new_bot[0, 0]) == (10.0, 6.0)        # Takes the run down to the absent cell.


def test_nearest_active_looks_through_pass_through_cells_only():
    status = np.array([[ACTIVE], [PASSTHROUGH], [PASSTHROUGH], [NOFLOW], [PASSTHROUGH], [ACTIVE]])
    up, down = nearest_active(status)
    np.testing.assert_array_equal(up[:, 0], [0, 0, 0, -1, -1, 5])
    np.testing.assert_array_equal(down[:, 0], [0, -1, -1, -1, 5, 5])


def test_subdivide_uniform_and_fractions():
    botm, idomain, parent = subdivide(10.0, [[4.0, 6.0], [0.0, 0.0]], nsub=[3, 1])
    np.testing.assert_allclose(botm[:, 0], [8, 6, 4, 0])
    np.testing.assert_allclose(botm[:, 1], [26 / 3, 22 / 3, 6, 0])
    np.testing.assert_array_equal(parent, [0, 0, 0, 1])
    botm, _, parent = subdivide(10.0, [[0.0]], fractions=[[1, 3]])
    np.testing.assert_allclose(botm[:, 0], [7.5, 0])


def test_subdivide_only_where_requested():
    where = np.array([True, False])
    botm, idomain, parent = subdivide(10.0, [[4.0, 4.0]], [[1, 1]], nsub=2, where=where)
    np.testing.assert_allclose(botm, [[7, 4], [4, 4]])
    np.testing.assert_array_equal(idomain, [[1, 1], [1, -1]])
    np.testing.assert_array_equal(parent, [0, 0])
    # The empty slot outside the area is pass-through, so the parent cell stays whole.
    top_l, bot_l, idomain = layer_arrays(10.0, botm, idomain)
    status = classify(top_l, bot_l, idomain)
    assert status[1, 1] == PASSTHROUGH and (top_l[0, 1], bot_l[0, 1]) == (10, 4)


@pytest.mark.parametrize("kwargs, match", [
    ({}, "exactly one"),
    ({"nsub": 2, "fractions": [[1]]}, "exactly one"),
    ({"nsub": 0}, "positive"),
    ({"fractions": [[1, -1]]}, "positive"),
    ({"fractions": [[1], [1]]}, "one entry per layer"),
    ({"nsub": 2, "where": np.array([1])}, "boolean mask"),
])
def test_subdivide_rejects_bad_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        subdivide(10.0, [[0.0]], **kwargs)
