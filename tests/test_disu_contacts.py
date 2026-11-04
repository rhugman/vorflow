"""disu.split_contacts: a wedge layer under a flat cut keeps the given bottoms at a change of grid."""

import functools

import numpy as np
import pandas as pd
import pytest

from vorflow import disu

from test_disu_grid import assert_cells_close
from test_disu_multigrid import PIT, VORONOI_PIT, _boxes, _contact_is_gap_free, _straddling_pit, voronoi_pit


def _straddling_model():
    """Layers 0-1 on 1 m boxes dipping 10 %, layers 2-3 on a pit whose cells straddle the coarse ones."""
    base = _boxes(4, 1.0)
    pit = _straddling_pit(base)
    slope = 6.0 - 0.1 * base.x.to_numpy()
    contact = 4.0 - 0.1 * pit.x.to_numpy()          # The layer 1 bottom, at the pit cells' centres.
    botm = [slope, slope - 2, np.full(len(pit), 1.0), np.full(len(pit), 0.0)]
    return base, pit, [base, base, pit, pit], botm, contact


def _layer(d, layer):
    return d.cellmap[d.cellmap.layer == layer].reset_index(drop=True)


def test_upper_wedge_keeps_the_bottoms_above_and_belongs_to_the_layer_below():
    base, pit, grids, botm, _ = _straddling_model()
    grids2, top2, botm2, idomain2, parent = disu.split_contacts(grids, 10.0, botm)
    np.testing.assert_array_equal(parent, [0, 1, 2, 2, 3])
    assert grids2[2] is base and grids2[3] is pit
    d = disu.DisuGrid(grids2, top2, botm2, idomain2).build()
    assert d.diagnostics["max_contact_shift"] == 0
    np.testing.assert_array_equal(_layer(d, 1).bot, botm[1])
    # The four coarse pit cells end at 3.85 and 3.75; the wedge fills the 0.1 m under the higher two.
    wedge = _layer(d, 2)
    made = wedge[wedge.node >= 0]
    assert len(made) == 2
    np.testing.assert_allclose(made.top - made.bot, 0.1)
    np.testing.assert_array_equal(made.bot, 3.75)
    _contact_is_gap_free(d, 2, 3)
    assert_cells_close(d)


@pytest.mark.parametrize("wedge, unit", [("upper", [0, 1, 2, 2, 3]), ("lower", [0, 1, 1, 2, 3])])
def test_the_cell_map_file_puts_the_wedge_in_the_unit_it_belongs_to(tmp_path, wedge, unit):
    base, pit, grids, botm, contact = _straddling_model()
    top = [10.0, None, contact, None] if wedge == "lower" else 10.0
    grids2, top2, botm2, idomain2, parent = disu.split_contacts(grids, top, botm, wedge=wedge)
    d = disu.DisuGrid(grids2, top2, botm2, idomain2).build()
    cells = pd.read_csv(d.save_cellmap(tmp_path / "cells.csv", parent))
    np.testing.assert_array_equal(cells.groupby("layer").unit.first(), unit)
    # The wedge (layer 2) keeps its own plan grid: its unit spans two grids.
    other = 3 if wedge == "upper" else 1
    assert set(cells[cells.layer == 2].grid) != set(cells[cells.layer == other].grid)


def test_lower_wedge_follows_the_contact_given_on_the_fine_grid():
    base, pit, grids, botm, contact = _straddling_model()
    top = [10.0, None, contact, None]
    grids2, top2, botm2, idomain2, parent = disu.split_contacts(grids, top, botm, wedge="lower")
    np.testing.assert_array_equal(parent, [0, 1, 1, 2, 3])
    assert grids2[2] is pit
    d = disu.DisuGrid(grids2, top2, botm2, idomain2).build()
    assert d.diagnostics["max_contact_shift"] == 0
    # The cut is the highest contact (3.8625); below it the given contact is kept on the fine cells.
    fine = (pit.splice_source == "fine").to_numpy()
    below = _layer(d, 3)
    np.testing.assert_array_equal(below.top[fine], contact[fine])
    np.testing.assert_allclose(_layer(d, 2).top[fine], 3.8625)
    assert (_layer(d, 2).node[fine] >= 0).sum() == 6           # The column at x = 1.375 has no wedge.
    _contact_is_gap_free(d, 1, 2)
    assert_cells_close(d)


def test_min_thickness_moves_the_bottom_instead_by_at_most_that():
    base, pit, grids, botm, contact = _straddling_model()
    # Upper wedge: the 0.1 m pieces are not made, so the higher bottoms move down to the cut.
    grids2, top2, botm2, idomain2, parent = disu.split_contacts(grids, 10.0, botm, min_thickness=0.2)
    np.testing.assert_array_equal(parent, [0, 1, 2, 3])
    moved = np.abs(botm2[1] - botm[1])
    assert 0 < moved.max() <= 0.2
    assert disu.DisuGrid(grids2, top2, botm2, idomain2).build().diagnostics["max_contact_shift"] == 0
    # Lower wedge: the 0.0625 m pieces are dropped, the 0.125 m ones kept.
    out = disu.split_contacts(grids, [10.0, None, contact, None], botm, wedge="lower", min_thickness=0.1)
    d = disu.DisuGrid(*out[:4]).build()
    fine = (pit.splice_source == "fine").to_numpy()
    assert (_layer(d, 2).node[fine] >= 0).sum() == 3
    assert 0 < np.abs(_layer(d, 3).top[fine] - contact[fine]).max() <= 0.1


def test_wedges_take_the_idomain_of_the_layer_they_belong_to():
    base, pit, grids, botm, contact = _straddling_model()
    idomain = [np.ones(len(base), int), 20 + np.arange(len(base)), 40 + np.arange(len(pit)),
               np.ones(len(pit), int)]
    # Upper wedge: from the fine cell each coarse cell overlaps most (its corner cell).
    _, _, _, idomain2, _ = disu.split_contacts(grids, 10.0, botm, idomain)
    xb, yb = base.x.to_numpy(), base.y.to_numpy()
    xp, yp = pit.x.to_numpy(), pit.y.to_numpy()
    for i in np.flatnonzero(np.isclose(xb, 1.5) & (yb > 1) & (yb < 3)):
        corner = np.flatnonzero(np.isclose(xp, 1.375) & np.isclose(np.abs(yp - yb[i]), 0.125))[0]
        assert idomain2[2][i] == 40 + corner
    assert (idomain2[2][~np.isclose(xb, 1.5) | (yb < 1) | (yb > 3)] == -1).all()
    # Lower wedge: from the coarse cell above each fine cell's centre.
    _, _, _, idomain2, _ = disu.split_contacts(grids, [10.0, None, contact, None], botm, idomain, wedge="lower")
    j = np.flatnonzero(np.isclose(xp, 2.625) & np.isclose(yp, 1.375))[0]
    i = np.flatnonzero(np.isclose(xb, 2.5) & np.isclose(yb, 1.5))[0]
    assert idomain2[2][j] == 20 + i
    assert idomain2[2][np.isclose(xp, 1.375)].tolist() == [-1] * 3


def test_nothing_to_split_returns_the_inputs():
    base = _boxes(4, 1.0)
    nested = disu.splice(base, _boxes(8, 0.5, start_id=500), PIT)
    slope = 6.0 - 0.1 * base.x.to_numpy()
    botm = [slope, slope - 2, np.full(len(nested), 1.0)]
    grids = [base, base, nested]
    out = disu.split_contacts(grids, 10.0, botm)
    assert out[0] is grids and out[2] is botm
    np.testing.assert_array_equal(out[4], [0, 1, 2])
    # One grid, or contacts left out with layers=.
    assert disu.split_contacts(base, 10.0, [6.0, 4.0])[2] == [6.0, 4.0]
    _, _, grids, botm, _ = _straddling_model()
    assert disu.split_contacts(grids, 10.0, botm, layers=[1, 3])[2] is botm


def test_bad_options_raise():
    _, _, grids, botm, _ = _straddling_model()
    with pytest.raises(ValueError, match="wedge"):
        disu.split_contacts(grids, 10.0, botm, wedge="middle")
    with pytest.raises(ValueError, match="min_thickness"):
        disu.split_contacts(grids, 10.0, botm, min_thickness=-1)


def test_report_points_to_split_contacts():
    _, _, grids, botm, _ = _straddling_model()
    assert "split_contacts" in disu.DisuGrid(grids, 10.0, botm).build().report()


@functools.lru_cache(maxsize=None)
def _voronoi_pit_at_3_percent():
    """The notebook pit, 3 % dip: layer 1 (index 1) bottom is the contact, refilled grid below it."""
    _, base = voronoi_pit()
    pit = disu.refill(base, VORONOI_PIT, 12)
    xb, xp = base.x.to_numpy(), pit.x.to_numpy()
    grids = [base, base, pit, pit]
    top = [100 - 0.03 * xb, None, 60 - 0.03 * xp, None]
    botm = [80 - 0.03 * xb, 60 - 0.03 * xb, 50 - 0.03 * xp, 40 - 0.03 * xp]
    return base, pit, grids, top, botm


@pytest.mark.slow
@pytest.mark.parametrize("wedge", ["upper", "lower"])
def test_voronoi_pit_keeps_the_dipping_contact(wedge):
    # Properties only: cell counts and the size of the shift depend on the mesher.
    base, pit, grids, top, botm = _voronoi_pit_at_3_percent()
    today = disu.DisuGrid(grids, top[0], botm).build()
    assert today.diagnostics["max_contact_shift"] > 0
    out = disu.split_contacts(grids, top if wedge == "lower" else top[0], botm, wedge=wedge)
    d = disu.DisuGrid(*out[:4]).build()
    assert d.diagnostics["max_contact_shift"] == 0
    # The added nodes are exactly the wedge layer's nodes.
    assert d.nodes > today.nodes
    assert (_layer(d, 2).node >= 0).sum() == d.nodes - today.nodes
    assert_cells_close(d)
    # The given contact is kept, to roundoff where a piece is a few ulps thin.
    if wedge == "upper":
        np.testing.assert_allclose(_layer(d, 1).bot, botm[1], rtol=1e-12)
        _contact_is_gap_free(d, 2, 3)
    else:
        np.testing.assert_allclose(_layer(d, 3).top, top[2], rtol=1e-12)
        _contact_is_gap_free(d, 1, 2)
