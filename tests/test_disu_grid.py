"""DisuGrid on one plan grid: layered connections, pass-through cells and the MF6 DISU arrays.

Also holds the geometry checks shared by the DISU tests. The MODFLOW runs
live outside the test suite, in local benchmarks.
"""

from collections import defaultdict

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import box

from vorflow import disu


def _grid(nx=3, ny=2, widths=None):
    widths = np.ones(nx) if widths is None else np.asarray(widths, dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    polygons = [box(edges[i], j, edges[i + 1], j + 1) for j in range(ny) for i in range(nx)]
    return gpd.GeoDataFrame({
        "node_id": np.arange(100, 100 + len(polygons)),
        "zone_id": 1,
        "x": [p.centroid.x for p in polygons], "y": [p.centroid.y for p in polygons],
    }, geometry=polygons)


def _rows(props, key):
    return np.split(np.asarray(props[key]), np.cumsum(props["iac"])[:-1])


def _entries(props):
    """{(n, m): (ihc, cl12, hwva, angldegx)} for every off-diagonal entry."""
    result = {}
    for n, (ja, ihc, cl, hw, ang) in enumerate(zip(*(_rows(props, k) for k in
                                                     ("ja", "ihc", "cl12", "hwva", "angldegx")))):
        for m, *values in zip(ja[1:], ihc[1:], cl[1:], hw[1:], ang[1:]):
            result[n, int(m)] = tuple(values)
    return result



def assert_cells_close(d):
    """Each cell's connections add up to its own geometry, as MODFLOW's cell-by-cell balance needs.

    Side faces: the heights of a cell's connections across a plan face add
    up to its overlap with the active cells of the neighbour column. Top and
    bottom: the vertical connections cover the whole area or nothing, and
    (with ``collapse``) the cells they join meet.
    """
    top, bot, col = d._node_top, d._node_bot, d._node_cell
    n, m, ihc = (np.asarray(d.links[key]) for key in ("n", "m", "ihc"))
    hwva, height = np.asarray(d.links["hwva"]), np.asarray(d.links["height"])
    scale = max(1.0, float(np.abs(np.r_[top, bot]).max()))
    vertical = ihc == 0
    if d.options.collapse is not None:
        np.testing.assert_allclose(top[m[vertical]], bot[n[vertical]], rtol=0, atol=1e-9 * scale)
    area = d._area[col]
    for side in (n, m):
        covered = np.bincount(side[vertical], weights=hwva[vertical], minlength=d.nodes)
        hit = covered > 0
        np.testing.assert_allclose(covered[hit], area[hit], rtol=1e-9)
    got = defaultdict(float)
    for a, b, h in zip(n[~vertical], m[~vertical], height[~vertical]):
        got[a, col[b]] += h
        got[b, col[a]] += h
    nodes_of = defaultdict(list)
    for node, c in enumerate(col):
        nodes_of[c].append(node)
    for i, j in zip(d._faces["cell_id_1"], d._faces["cell_id_2"]):
        for own, other in ((i, j), (j, i)):
            for a in nodes_of[own]:
                expected = sum(max(0.0, min(top[a], top[b]) - max(bot[a], bot[b])) for b in nodes_of[other])
                assert got[a, other] == pytest.approx(expected, rel=1e-9, abs=1e-9 * scale), (a, other)


def test_options_are_validated_eagerly():
    grid = _grid()
    with pytest.raises(ValueError, match="collapse"):
        disu.DisuGrid(grid, 10, [0], collapse="sideways")
    with pytest.raises(ValueError, match="min_thickness"):
        disu.DisuGrid(grid, 10, [0], min_thickness=-1)
    with pytest.raises(ValueError, match="6 cells per layer"):
        disu.DisuGrid(grid, 10, [[0, 0]])
    with pytest.raises(RuntimeError, match="build"):
        disu.DisuGrid(grid, 10, [0], connectivity="layered").get_gridprops_mf6_disu()


def test_layered_csr_structure_and_symmetry():
    grid = _grid()
    d = disu.DisuGrid(grid, 10.0, [5.0, 0.0], connectivity="layered").build()
    props = d.get_gridprops_mf6_disu()
    assert props["nodes"] == 12 and len(props["iac"]) == 12
    assert props["nja"] == props["iac"].sum() == len(props["ja"])
    for node, (ja, ihc) in enumerate(zip(_rows(props, "ja"), _rows(props, "ihc"))):
        assert ja[0] == node and ihc[0] == node // 6 + 1     # Diagonal first; layer on the IHC diagonal.
        assert list(ja[1:]) == sorted(ja[1:])
    entries = _entries(props)
    for (n, m), (ihc, cl, hwva, angle) in entries.items():
        ihc_r, cl_r, hwva_r, angle_r = entries[m, n]
        assert ihc == ihc_r and hwva == hwva_r
        if ihc:
            assert (angle - angle_r) % 360 == pytest.approx(180)
    # Cell 0 (x=0.5, y=0.5): east neighbour at 0 degrees, north at 90, below it node 6.
    assert entries[0, 1] == (1, 0.5, 1.0, 0.0)
    assert entries[0, 3] == (1, 0.5, 1.0, 90.0)
    assert entries[0, 6] == (0, 2.5, 1.0, 0.0)
    assert entries[6, 0] == (0, 2.5, 1.0, 0.0)          # Vertical links carry no angle either way.
    assert len(props["cell2d"]) == 12 and props["cell2d"][7][0] == 7
    np.testing.assert_array_equal(props["top"][:6], 10)
    np.testing.assert_array_equal(props["bot"][6:], 0)


def test_pass_through_and_noflow_cells_are_removed_and_tracked():
    grid = _grid(nx=2, ny=1)
    idomain = [[1, 1], [-1, 0], [1, 1]]
    d = disu.DisuGrid(grid, 10.0, [[6, 6], [4, 4], [0, 0]], idomain, connectivity="layered").build()
    np.testing.assert_array_equal(d.nodemap, [[0, 1], [-1, -1], [2, 3]])
    assert list(d.cellmap["status"]) == ["active", "active", "passthrough", "noflow", "active", "active"]
    assert list(d.cellmap["node_id"]) == [100, 101] * 3
    props = d.get_gridprops_mf6_disu()
    entries = _entries(props)
    # Pass-through links node 0 to node 2; no-flow breaks the second column.
    assert (0, 2) in entries and (1, 3) not in entries
    # collapse="split" closes the 2 m gap at its midpoint.
    assert props["bot"][0] == 5 and props["top"][2] == 5
    assert entries[0, 2][1] == 2.5 and entries[2, 0][1] == 2.5
    # Second column keeps its gap next to no-flow only on the no-flow side: both actives take their run.
    assert props["bot"][1] == 6 and props["top"][3] == 4
    assert "by pass-through cells: up to 1 per cell" in d.report()


def test_roundoff_thin_cells_are_passed_through():
    grid = _grid(nx=2, ny=1)
    d = disu.DisuGrid(grid, 60.0, [[50.0, 50.0], [50.0 - 7e-15, 45.0], [40.0, 40.0]]).build()
    np.testing.assert_array_equal(d.nodemap, [[0, 1], [-1, 2], [3, 4]])
    assert (d.get_gridprops_mf6_disu()["top"] > d.get_gridprops_mf6_disu()["bot"]).all()


def test_layered_disu_has_the_disv_connections():
    """Same-layer faces and columns through pass-through cells, with the DISV geometry (collapse=None)."""
    grid = _grid(nx=4, ny=3, widths=[1, 2, 1.5, 3])
    ncpl = len(grid)
    rng = np.random.default_rng(3)
    botm = np.array([6.0, 3.0, 0.0])[:, None] + rng.uniform(-0.5, 0.5, (3, ncpl))
    idomain = np.ones((3, ncpl), dtype=int)
    idomain[1, 5] = -1
    d = disu.DisuGrid(grid, 10.0, botm, idomain, connectivity="layered", collapse=None).build()
    entries = _entries(d.get_gridprops_mf6_disu())
    geoms, xy = grid.geometry.to_numpy(), grid[["x", "y"]].to_numpy()
    top_l = np.vstack([np.full(ncpl, 10.0), botm[:-1]])
    expected = {}
    for i in range(ncpl):
        for j in range(ncpl):
            face = geoms[i].intersection(geoms[j])
            if i == j or face.length == 0:
                continue
            angle = np.degrees(np.arctan2(*(xy[j] - xy[i])[::-1])) % 360
            for k in np.flatnonzero((idomain[:, i] > 0) & (idomain[:, j] > 0)):
                expected[d.nodemap[k, i], d.nodemap[k, j]] = (
                    1, shapely.Point(xy[i]).distance(face), face.length, angle)
        active = np.flatnonzero(idomain[:, i] > 0)
        for k_up, k_down in zip(active[:-1], active[1:]):          # Pass-through cells are skipped.
            n, m = d.nodemap[k_up, i], d.nodemap[k_down, i]
            expected[n, m] = (0, (top_l[k_up, i] - botm[k_up, i]) / 2, geoms[i].area, 0.0)
            expected[m, n] = (0, (top_l[k_down, i] - botm[k_down, i]) / 2, geoms[i].area, 0.0)
    assert set(entries) == set(expected)
    for key, value in expected.items():
        np.testing.assert_allclose(entries[key], value, atol=1e-12, err_msg=str(key))
