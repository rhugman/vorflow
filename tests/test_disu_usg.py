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
