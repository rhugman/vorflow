"""MODFLOW-USG output of DisuGrid: IVSD/IVC, FAHL from overlaps and GSF."""

import numpy as np
import pytest

from vorflow import disu
from vorflow.utils import grid_to_mfusg_disu

from test_disu_full import _dipping_aquifer, _two_columns
from test_disu_grid import _grid, _rows


def test_disu5_marks_vertical_connections_explicitly():
    d = _two_columns(np.array([20.0, 20.0]), [12, 4], [16, 0])
    props = d.get_gridprops_mfusg_disu()
    assert (props["ivsd"], props["idsymrd"], props["nlay"]) == (1, 0, 2)
    np.testing.assert_array_equal(props["nodelay"], [2, 2])
    a1, b1, a2, b2 = d.nodemap[0, 0], d.nodemap[0, 1], d.nodemap[1, 0], d.nodemap[1, 1]
    rows = {n: dict(zip(ja[1:], zip(ivc[1:], fahl[1:], cl[1:]))) for n, (ja, ivc, fahl, cl) in
            enumerate(zip(*(_rows(props, k) for k in ("ja", "ivc", "fahl", "cl12"))))}
    assert rows[a1][a2] == (1, 1.0, 4.0)            # Vertical: IVC=1, plan area, half thickness.
    assert rows[a1][b2] == (0, 4.0, 0.5)            # Staggered stays horizontal: width x overlap.
    assert rows[a1][b1] == (0, 4.0, 0.5)
    assert rows[a2][b2] == (0, 8.0, 0.5)
    assert all(ivc[0] == 0 for ivc in _rows(props, "ivc"))


def test_disu5_counts_nodes_per_layer_after_removing_cells():
    grid = _grid(nx=3, ny=1)
    d = disu.DisuGrid(grid, 10.0, [[6, 6, 6], [4, 4, 4], [0, 0, 0]], [[1, 1, 1], [1, -1, 0], [1, 1, 1]]).build()
    np.testing.assert_array_equal(d.get_gridprops_mfusg_disu()["nodelay"], [3, 1, 3])
    empty = disu.DisuGrid(grid, 10.0, [[6, 6, 6], [6, 6, 6]]).build()
    with pytest.raises(ValueError, match="Layer 1 has no active cells"):
        empty.get_gridprops_mfusg_disu()


def test_layered_disu5_matches_the_original_usg_export():
    grid = _grid(nx=3, ny=2, widths=[1, 2, 1])
    legacy = grid_to_mfusg_disu(grid, 10.0, [6.0, 0.0], nlay=2)
    new = disu.DisuGrid(grid, 10.0, [6.0, 0.0], connectivity="layered").build().get_gridprops_mfusg_disu()
    for key in ("iac", "ja", "cl12", "fahl", "top", "bot", "nodelay"):
        np.testing.assert_allclose(new[key], legacy[key], err_msg=key)
    # The original export keeps one layer of areas and lets USG infer vertical connections.
    assert legacy["ivsd"] == -1 and new["ivsd"] == 1
    np.testing.assert_allclose(new["area"], np.tile(legacy["area"], 2))


def test_gsf_lists_each_node_with_its_own_elevations(tmp_path):
    d = _two_columns(np.array([20.0, 20.0]), [12, 4], [16, 0])
    path = d.write_gsf(tmp_path / "grid.gsf")
    lines = path.read_text().splitlines()
    assert lines[0] == "UNSTRUCTURED" and lines[1] == "4 2 1 1"
    cells = lines[3 + int(lines[2]):]
    assert len(cells) == 4
    # Node 2 is column A, layer 2 (12 to 4): centre elevation 8, layer 2.
    node, x, y, z, layer, nverts = cells[2].split()[:6]
    assert (int(node), float(z), int(layer), int(nverts)) == (3, 8.0, 2, 8)


def test_disu5_side_areas_cover_the_walls_of_a_dipping_aquifer():
    """Each aquifer cell's two walls are wholly connected: FAHL adds up to width x thickness."""
    d = _dipping_aquifer(30, "full")
    props = d.get_gridprops_mfusg_disu()
    side = np.array([fahl[1:][ivc[1:] == 0].sum() for fahl, ivc in zip(_rows(props, "fahl"), _rows(props, "ivc"))])
    below = np.array([fahl[1:][ivc[1:] == 1].sum() for fahl, ivc in zip(_rows(props, "fahl"), _rows(props, "ivc"))])
    thickness = np.asarray(props["top"]) - np.asarray(props["bot"])
    cells = d.cellmap[d.cellmap.node >= 0]
    layer, x = cells.layer.to_numpy(), cells.x.to_numpy()
    inner = (layer >= 3) & (layer <= 5) & (x > 1) & (x < 10)        # Aquifer cells away from the ends.
    np.testing.assert_allclose(side[inner], 2 * 1.0 * thickness[inner])
    np.testing.assert_allclose(below[inner], 2 * 1.0)             # 1 m2 above and below.


@pytest.mark.parametrize("right, overlap", [((9.0, 11.0), 1.0), ((9.0, 10.0), 1.0), ((9.5, 10.0), 0.5),
                                            ((5.0, 10.0), 5.0)])
def test_ksat_gives_the_overlap_conductance_with_ikcflag_1(right, overlap):
    # Cells [0, 10] and `right`, 1 m wide, centres 1 m apart, K = 2. With ikcflag=1, MODFLOW-USG
    # computes fahl * ksat / (cl1 + cl2); without it, fahl / mean thickness * harmonic(K b) / (cl1 + cl2).
    bottom, top = right
    d = disu.DisuGrid(_grid(nx=2, ny=1), [10.0, top], [[0.0, bottom]]).build()
    props = d.get_gridprops_mfusg_disu()
    ksat = d.get_mfusg_ksat(2.0)
    assert ksat.shape == props["ja"].shape
    i = int(np.flatnonzero(props["ja"] == 1)[0])          # Row 0, entry for node 1.
    fahl, cl = props["fahl"][i], props["cl12"][i] + props["cl12"][props["iac"][0] + 1]
    np.testing.assert_allclose(fahl * ksat[i] / cl, 2.0 * overlap)
    b = np.array([10.0, top - bottom])
    default = fahl / b.mean() * 2.0 * (2.0 * b[0] * 2.0 * b[1] / (2.0 * b[0] + 2.0 * b[1])) / cl
    assert default < 2.0 * overlap


def test_ksat_is_the_distance_weighted_harmonic_mean():
    d = disu.DisuGrid(_grid(nx=2, ny=1, widths=[1.0, 3.0]), 10.0, [6.0, 0.0]).build()
    props = d.get_gridprops_mfusg_disu()
    k = np.array([1.0, 4.0, 1.0, 4.0])
    ksat = dict(zip(zip(np.repeat(np.arange(d.nodes), props["iac"]), props["ja"]), d.get_mfusg_ksat(k, k33=k / 10)))
    a, b, below = d.nodemap[0, 0], d.nodemap[0, 1], d.nodemap[1, 0]
    assert ksat[a, a] == 1.0 and ksat[b, b] == 4.0
    np.testing.assert_allclose(ksat[a, b], 2.0 / (0.5 / 1.0 + 1.5 / 4.0))   # cl 0.5 and 1.5
    np.testing.assert_allclose(ksat[a, below], 0.1)                          # vertical: k33
    with pytest.raises(ValueError, match="k must hold positive"):
        d.get_mfusg_ksat([1.0, -1.0, 1.0, 1.0])
