"""Tracking for DisuGrid: values between nodes and layers, and point lookup."""

import numpy as np
import pytest

from vorflow import disu

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


def test_idomain_layers_maps_status_to_mf6_codes():
    d = _three_columns([[1, 1, 1], [-1, 1, 0], [1, 1, 1]])
    np.testing.assert_array_equal(d.idomain_layers, [[1, 1, 1], [-1, 1, 0], [1, 1, 1]])
    thin = disu.DisuGrid(_grid(nx=2, ny=1), 10.0, [[7.0, 7.0], [6.95, 3.0]], min_thickness=0.1).build()
    np.testing.assert_array_equal(thin.idomain_layers, [[1, 1], [-1, 1]])
