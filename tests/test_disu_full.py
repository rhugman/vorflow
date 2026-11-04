"""Full connectivity (IHC=2) by interface merging, sliver merging and face-area closure."""

import numpy as np
import pytest

from vorflow import disu

from test_disu_grid import _entries, _grid, assert_cells_close


def _pairs(d):
    links = d.links
    return {frozenset((int(n), int(m))): (int(ihc), h) for n, m, ihc, h in
            zip(links["n"], links["m"], links["ihc"], links["height"]) if ihc}


def _two_columns(top, botm_a, botm_b, **kwargs):
    grid = _grid(nx=2, ny=1)
    botm = np.column_stack([botm_a, botm_b])
    return disu.DisuGrid(grid, top, botm, **kwargs).build()


@pytest.mark.parametrize("kwargs, match", [
    ({"connectivity": "staggered"}, "connectivity"),
    ({"sliver_abs_tol": -1}, "sliver_abs_tol"),
    ({"cl12": "centre"}, "cl12"),
])
def test_full_connectivity_options_are_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        disu.DisuGrid(_grid(), 10, [0], **kwargs)


def test_without_offsets_full_matches_layered_connectivity():
    grid = _grid(nx=3, ny=2)
    full = disu.DisuGrid(grid, 10.0, [6.0, 2.0]).build()
    layered = disu.DisuGrid(grid, 10.0, [6.0, 2.0], connectivity="layered").build()
    assert set(_pairs(full)) == set(_pairs(layered))
    assert set(full.links["ihc"]) == {0, 2}
    np.testing.assert_allclose(np.sort(full.links["hwva"]), np.sort(layered.links["hwva"]))


def test_dipping_offset_links_each_overlap_once():
    # Column A: 20 | 12 | 4; column B: 16 | 8 | 0 (top 20 in both, B's first layer thicker on top).
    d = _two_columns(np.array([20.0, 20.0]), [12, 4], [16, 0])
    # Pieces of the shared wall: 20-16 A1|B1, 16-12 A1|B2, 12-4 A2|B2.
    a1, b1, a2, b2 = d.nodemap[0, 0], d.nodemap[0, 1], d.nodemap[1, 0], d.nodemap[1, 1]
    pairs = _pairs(d)
    assert set(pairs) == {frozenset((a1, b1)), frozenset((a1, b2)), frozenset((a2, b2))}
    assert pairs[frozenset((a1, b1))] == (2, 4)
    assert pairs[frozenset((a1, b2))] == (2, 4)
    assert pairs[frozenset((a2, b2))] == (2, 8)
    props = d.get_gridprops_mf6_disu()
    entries = _entries(props)
    assert entries[a1, b2] == (2, 0.5, 1.0, 0.0)     # HWVA is the face width; MF6 applies the overlap.
    assert entries[b2, a1] == (2, 0.5, 1.0, 180.0)


def test_by_default_every_piece_is_kept():
    top = np.array([20.0, 20.0])
    d = _two_columns(top, [10.000, 0], [10.002, 0])
    a1, b2 = d.nodemap[0, 0], d.nodemap[1, 1]
    pairs = _pairs(d)
    assert len(pairs) == 3
    assert pairs[frozenset((a1, b2))][1] == pytest.approx(0.002)
    assert d.diagnostics["slivers_merged"] == 0
    assert "merged pieces" not in d.report()


def test_opt_in_sliver_merge_keeps_face_area_and_is_reported():
    top = np.array([20.0, 20.0])
    d = _two_columns(top, [10.000, 0], [10.002, 0], sliver_abs_tol=0.003)
    a1, b1, a2, b2 = d.nodemap[0, 0], d.nodemap[0, 1], d.nodemap[1, 0], d.nodemap[1, 1]
    pairs = _pairs(d)
    assert set(pairs) == {frozenset((a1, b1)), frozenset((a2, b2))}
    assert pairs[frozenset((a1, b1))][1] == pytest.approx(9.999)
    assert pairs[frozenset((a2, b2))][1] == pytest.approx(10.001)
    assert sum(h for _, h in pairs.values()) == pytest.approx(20.0)
    # MF6 recomputes the overlap (9.998) from the elevations, so HWVA carries the merged height.
    entries = _entries(d.get_gridprops_mf6_disu())
    assert entries[a1, b1][2] == pytest.approx(9.999 / 9.998)
    assert d.diagnostics["slivers_merged"] == 1
    assert d.diagnostics["largest_sliver"] == pytest.approx(0.002)
    assert "not exact" in d.report()
    # 2 mm is below 3 mm but not below 1 mm.
    assert len(_pairs(_two_columns(top, [10.000, 0], [10.002, 0], sliver_abs_tol=0.001))) == 3


def _brute_force(d):
    """Every pair of active cells in neighbouring columns whose vertical intervals overlap."""
    cm = d.cellmap[d.cellmap["node"] >= 0]
    by_cell = {cell: rows for cell, rows in cm.groupby("icpl")}
    expected = set()
    for i, j in zip(d._faces["cell_id_1"], d._faces["cell_id_2"]):
        for a in by_cell.get(i, cm.iloc[:0]).itertuples():
            for b in by_cell.get(j, cm.iloc[:0]).itertuples():
                if min(a.top, b.top) - max(a.bot, b.bot) > 0:
                    expected.add(frozenset((a.node, b.node)))
    return expected


@pytest.mark.parametrize("collapse", ["split", None])
def test_links_match_brute_force_overlaps_on_random_layers(collapse):
    rng = np.random.default_rng(7)
    grid = _grid(nx=5, ny=4, widths=[1, 2, 1, 3, 1])
    ncpl = len(grid)
    botm = np.sort(rng.uniform(-30, 9, (6, ncpl)), axis=0)[::-1]
    idomain = rng.choice([1, 1, 1, 1, -1, 0], size=botm.shape)
    d = disu.DisuGrid(grid, 10.0, botm, idomain, collapse=collapse, min_thickness=0.2).build()
    assert set(_pairs(d)) == _brute_force(d)
    assert_cells_close(d)
    # Each horizontal link's height is the vertical overlap of its two cells.
    top = d.cellmap.set_index("node")["top"]
    bot = d.cellmap.set_index("node")["bot"]
    for (n, m), (_, h) in ((tuple(k), v) for k, v in _pairs(d).items()):
        assert h == pytest.approx(min(top[n], top[m]) - max(bot[n], bot[m]))


def test_report_flags_layers_offset_by_a_whole_cell():
    # Column B sits one whole cell (8 m) lower: its layer 1 faces column A's layer 2.
    d = _two_columns(np.array([20.0, 12.0]), [12, 4], [4, -4])
    assert d.diagnostics["max_offset_ratio"] == pytest.approx(1.0)
    a2, b1 = d.nodemap[1, 0], d.nodemap[0, 1]
    assert set(_pairs(d)) == {frozenset((a2, b1))}
    text = d.report()
    assert "joined only through other layers" in text and "2 sublayers" in text
    assert "angle2 and angle3" in text


def _dipping_aquifer(dip, connectivity):
    """Provost et al. (2025) style cross-section: nine 1 m layers rising at ``dip`` degrees."""
    grid = _grid(nx=11, ny=1)
    rise = np.tan(np.radians(dip)) * grid.x.to_numpy()
    surfaces = rise[None, :] + np.arange(6, -4, -1)[:, None]
    return disu.DisuGrid(grid, surfaces[0], surfaces[1:], connectivity=connectivity).build()


@pytest.mark.parametrize("dip", [10, 30, 60])
def test_dipping_aquifer_cells_close(dip):
    """Every overlapping wall piece is connected and adds up to the cell geometry (what XT3D needs)."""
    d = _dipping_aquifer(dip, "full")
    assert_cells_close(d)
    assert len(_pairs(d)) > len(_pairs(_dipping_aquifer(dip, "layered")))


def test_one_ulp_offset_does_not_duplicate_a_connection():
    """A piece one ulp thick has a midpoint on a boundary; it must not repeat its neighbour's pair."""
    top = np.array([20.0, 20.0])
    d = _two_columns(top, [10.0, 0], [np.nextafter(10.0, 20.0), 0])
    props = d.get_gridprops_mf6_disu()                    # Raises on a duplicate connection.
    assert props["nja"] == len(props["ja"])
    assert sum(h for _, h in _pairs(d).values()) == pytest.approx(20.0)
