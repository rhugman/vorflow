"""Tracking for DisuGrid: values between nodes and layers, point lookup and the cell map file."""

import numpy as np
import pandas as pd
import pytest

from vorflow import disu

from test_disu_full import _two_columns
from test_disu_grid import _grid


def _three_columns(idomain=None, **options):
    grid = _grid(nx=3, ny=1)
    botm = [[7.0, 7.0, 7.0], [3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]
    return disu.DisuGrid(grid, 10.0, botm, idomain, **options).build()


def test_to_nodes_to_layers_round_trip():
    grid = _grid(nx=2, ny=1)
    d = disu.DisuGrid(grid, 10.0, [[6, 6], [4, 4], [0, 0]], [[1, 1], [-1, 1], [1, 1]],
                      connectivity="layered").build()
    k = np.arange(6).reshape(3, 2) + 1
    np.testing.assert_array_equal(d.to_nodes(k), [1, 2, 4, 5, 6])
    np.testing.assert_array_equal(d.to_nodes([10.0, 20.0, 30.0]), [10, 10, 20, 30, 30])
    back = d.to_layers(d.to_nodes(k).astype(float))
    np.testing.assert_array_equal(np.isnan(back), [[0, 0], [1, 0], [0, 0]])
    np.testing.assert_array_equal(np.nan_to_num(back), np.where(np.isnan(back), 0, k))
    # MF6 DISU accepts IDOMAIN 0/1 only, so no tags are written there.
    assert "idomain" not in d.get_gridprops_mf6_disu()
    with pytest.raises(ValueError, match="one entry per layer"):
        d.to_nodes([1.0, 2.0])
    with pytest.raises(ValueError, match="one per node"):
        d.to_layers(np.zeros(3))


def test_locate_and_column_follow_the_collapsed_cells():
    # Column A: 20 | 12 | 4; column B: 20 | 16 | 0, with B's first layer passed through.
    grid = _grid(nx=2, ny=1)
    d = disu.DisuGrid(grid, 20.0, [[12, 16], [4, 0]], [[1, -1], [1, 1]]).build()
    a1, a2, b2 = d.nodemap[0, 0], d.nodemap[1, 0], d.nodemap[1, 1]
    assert d.locate(0.5, 0.5, 15.0) == a1
    assert d.locate(0.5, 0.5, 5.0) == a2
    assert d.locate(1.5, 0.5, 18.0) == b2            # B's pass-through interval went to the cell below.
    assert d.locate(0.5, 0.5, 2.0) == -1             # Below the model.
    assert d.locate(5.0, 0.5, 10.0) == -1            # Outside the grid.
    np.testing.assert_array_equal(d.locate([0.5, 1.5], [0.5, 0.5], [15.0, 1.0]), [a1, b2])
    np.testing.assert_array_equal(d.column(0.5, 0.5), [a1, a2])
    np.testing.assert_array_equal(d.column(1.5, 0.5), [b2])
    assert len(d.column(5.0, 0.5)) == 0
    with pytest.raises(RuntimeError, match="build"):
        disu.DisuGrid(grid, 10.0, [0.0]).locate(0, 0, 0)


def test_cellmap_csv_round_trip(tmp_path):
    d = _two_columns(np.array([20.0, 20.0]), [12, 4], [16, 0])
    path = d.save_cellmap(tmp_path / "cells.csv")
    back = pd.read_csv(path)
    assert list(back.columns[:3]) == ["node", "layer", "icpl"]
    np.testing.assert_array_equal(back.node, d.cellmap.node)
    assert list(back.status) == list(d.cellmap.status.astype(str))


def test_cellmap_csv_gives_each_layer_cell_its_node_unit_and_idomain(tmp_path):
    # Layer 1 split into two sublayers in the middle column only (the second slot is empty elsewhere).
    grid = _grid(nx=3, ny=1)
    top = np.full(3, 10.0)
    botm = [[6.0, 6.0, 6.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]
    idomain = [[1, 1, 1], [1, 1, 0], [-1, 1, 0]]
    parent = np.array([0, 1, 1])
    d = disu.DisuGrid(grid, top, botm, idomain).build()
    back = pd.read_csv(d.save_cellmap(tmp_path / "cells.csv", parent))
    np.testing.assert_array_equal(back.unit, parent[back.layer])
    np.testing.assert_array_equal(back.idomain, np.ravel(d.idomain_layers))
    # Active rows in node order carry the exported node centres and mid-elevations.
    props = d.get_gridprops_mf6_disu()
    nodes = back[back.node >= 0].sort_values("node")
    np.testing.assert_array_equal(nodes.node, np.arange(d.nodes))
    np.testing.assert_allclose(nodes[["x", "y"]], [row[1:3] for row in props["cell2d"]])
    np.testing.assert_allclose(nodes.z, (props["top"] + props["bot"]) / 2)
    assert (nodes.idomain == 1).all()
    np.testing.assert_array_equal(pd.read_csv(d.save_cellmap(tmp_path / "plain.csv")).unit, back.layer)
    with pytest.raises(ValueError, match="parent"):
        d.save_cellmap(tmp_path / "bad.csv", parent[:-1])


def test_idomain_layers_maps_status_to_mf6_codes():
    d = _three_columns([[1, 1, 1], [-1, 1, 0], [1, 1, 1]])
    np.testing.assert_array_equal(d.idomain_layers, [[1, 1, 1], [-1, 1, 0], [1, 1, 1]])
    thin = disu.DisuGrid(_grid(nx=2, ny=1), 10.0, [[7.0, 7.0], [6.95, 3.0]], min_thickness=0.1).build()
    np.testing.assert_array_equal(thin.idomain_layers, [[1, 1], [-1, 1]])


def test_layer_view_picks_a_sublayer_where_a_layer_was_split():
    # Layer 1 split into three equal sublayers in the middle column only (the other slots are empty).
    grid = _grid(nx=3, ny=1)
    top = np.full(3, 10.0)
    botm = [[6.0] * 3, [0.0, 4.0, 0.0], [0.0, 2.0, 0.0], [0.0] * 3]
    idomain = [[1, 1, 1], [1, 1, 1], [-1, 1, -1], [-1, 1, -1]]
    parent = np.array([0, 1, 1, 1])
    d = disu.DisuGrid(grid, top, botm, idomain).build()
    values = np.arange(d.nodes, dtype=float)
    n = d.nodemap
    # Columns 0 and 2 keep one cell (first sublayer); column 1 has three equal sublayers.
    expected = {
        "top": [n[1, 0], n[1, 1], n[1, 2]],
        "bottom": [n[1, 0], n[3, 1], n[1, 2]],
        "middle": [n[1, 0], n[2, 1], n[1, 2]],
        "mean": [n[1, 0], (n[1, 1] + n[2, 1] + n[3, 1]) / 3, n[1, 2]],
    }
    for pick, want in expected.items():
        np.testing.assert_allclose(d.layer_view(values, 1, parent, pick=pick), want, err_msg=pick)
    np.testing.assert_allclose(d.layer_view(values, 0, parent), n[0])
    with pytest.raises(ValueError, match="No layer has parent"):
        d.layer_view(values, 5, parent)


@pytest.mark.parametrize("collapse, source", [("down", 2), ("up", 0), (None, None)])
def test_layer_view_cover_shows_the_cell_that_took_a_pass_through_interval(collapse, source):
    d = _three_columns([[1, 1, 1], [-1, 1, 0], [1, 1, 1]], collapse=collapse)
    values = np.arange(d.nodes, dtype=float)
    plain = d.layer_view(values, 1)
    covered = d.layer_view(values, 1, fill="cover")
    assert np.isnan(plain[0]) and np.isnan(plain[2])
    assert np.isnan(covered[2])                                   # No-flow stays empty.
    if source is None:
        assert np.isnan(covered[0])                               # The gap is kept: nothing covers it.
    else:
        assert covered[0] == d.nodemap[source, 0]
    assert covered[1] == plain[1] == d.nodemap[1, 1]


def test_vertex_grid_is_the_layer_plan_grid_for_flopy():
    pytest.importorskip("flopy")
    d = _three_columns([[1, 1, 1], [-1, 1, 0], [1, 1, 1]])
    vg = d.vertex_grid(1)
    assert vg.ncpl == 3 and vg.nlay == 1
    np.testing.assert_allclose(vg.xcellcenters, d.grid.x)
    np.testing.assert_array_equal(np.asarray(vg.idomain).ravel(), [1, 1, 0])


def test_flopy_reads_the_layers_from_the_disu_grid():
    """FloPy layers a DISU grid by the IHC diagonal, which vorflow sets to the layer number."""
    flopy = pytest.importorskip("flopy", minversion="3.10")
    d = _three_columns([[1, 1, 1], [-1, 1, 0], [1, 1, 1]])
    sim = flopy.mf6.MFSimulation()
    gwf = flopy.mf6.ModflowGwf(sim)
    flopy.mf6.ModflowGwfdisu(gwf, **d.get_gridprops_mf6_disu())
    np.testing.assert_array_equal(gwf.modelgrid.ncpl, [3, 1, 3])
