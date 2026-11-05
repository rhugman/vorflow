"""DisuGrid.quality(): one row of geometric measurements per connection."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from vorflow import disu
from vorflow.utils import build_connectivity

from test_disu_full import _two_columns
from test_disu_grid import _grid
from test_disu_multigrid import _boxes, _straddling_pit, grids  # noqa: F401  (fixture)


def test_aligned_layers_are_ideal():
    d = disu.DisuGrid(_grid(nx=3, ny=2), 10.0, [6.0, 2.0]).build()
    q = d.quality()
    assert len(q) == len(d.links["n"])
    np.testing.assert_array_equal(q.n, d.links["n"])
    np.testing.assert_allclose(q.ortho_error_3d, 0.0, atol=1e-9)
    np.testing.assert_allclose(q.cl_ratio, 1.0)
    np.testing.assert_array_equal(q.gap, 0.0)


def test_offset_connections_measure_their_tilt_area_and_grading():
    # Column A: 20 | 12 | 4; column B: 20 | 16 | 0; cells 1 m wide.
    d = _two_columns(np.array([20.0, 20.0]), [12, 4], [16, 0])
    q = d.quality().set_index(["n", "m"])
    a1, b1, a2, b2 = d.nodemap[0, 0], d.nodemap[0, 1], d.nodemap[1, 0], d.nodemap[1, 1]
    # Centres: A1 z 16, B1 z 18, A2 z 8, B2 z 8; plan distance 1.
    assert q.loc[(a1, b1)].ortho_error_3d == pytest.approx(np.degrees(np.arctan(2.0)))
    assert q.loc[(a1, b2)].ortho_error_3d == pytest.approx(np.degrees(np.arctan(8.0)))
    assert q.loc[(a2, b2)].ortho_error_3d == pytest.approx(0.0, abs=1e-9)
    assert q.loc[(a1, b2)].area == pytest.approx(4.0)           # 1 m wide face, 4 m shared
    assert q.loc[(a1, a2)].area == pytest.approx(1.0)           # the 1 m x 1 m plan cell
    assert q.loc[(a1, b2)].cl_ratio == pytest.approx(1.0)       # the face is midway in plan
    assert q.loc[(b1, b2)].cl_ratio == pytest.approx(16 / 4)    # half thicknesses 2 and 8


def test_between_centres_at_one_depth_the_3d_error_is_the_plan_error():
    # Two parallelograms sheared by 0.5: their shared face leans atan(0.5) off the line between centres.
    cells = [Polygon([(i, 0), (i + 1, 0), (i + 1.5, 1), (i + 0.5, 1)]) for i in range(2)]
    grid = gpd.GeoDataFrame({"node_id": [100, 101], "zone_id": 1, "x": [c.centroid.x for c in cells],
                             "y": [c.centroid.y for c in cells]}, geometry=cells)
    q = disu.DisuGrid(grid, 10.0, [6.0, 2.0]).build().quality()
    side = q[q.ihc != 0].ortho_error_3d
    np.testing.assert_allclose(side, np.degrees(np.arctan(0.5)))
    np.testing.assert_allclose(side, build_connectivity(grid, center="generator").ortho_error.iloc[0])


def test_cl_ratio_measures_plan_grading():
    d = disu.DisuGrid(_grid(nx=2, ny=1, widths=[1.0, 3.0]), 10.0, [0.0]).build()
    q = d.quality()
    assert q[q.ihc != 0].cl_ratio.iloc[0] == pytest.approx(3.0)


def test_a_change_of_grid_tilts_vertical_connections():
    base = _boxes(4, 1.0)
    pit = _straddling_pit(base)
    slope = 6.0 - 0.1 * base.x.to_numpy()
    d = disu.DisuGrid([base, base, pit, pit], 10.0, [slope, slope - 2, np.full(len(pit), 1.0),
                                                     np.full(len(pit), 0.0)]).build()
    q = d.quality()
    cm = d.cellmap[d.cellmap.node >= 0].sort_values("node")
    layer = cm.layer.to_numpy()
    xyz = np.c_[cm.x, cm.y, (cm.top + cm.bot) / 2]
    # Vertical connections from a coarse cell to the fine cells under it lean by their plan shift.
    vertical = q[(q.ihc == 0) & (layer[q.n] == 1) & (layer[q.m] == 2)]
    plan = np.hypot(*(xyz[vertical.m, :2] - xyz[vertical.n, :2]).T)
    dz = np.abs(xyz[vertical.m, 2] - xyz[vertical.n, 2])
    np.testing.assert_allclose(vertical.ortho_error_3d, np.degrees(np.arctan2(plan, dz)))
    assert (vertical.ortho_error_3d > 0).any()


def test_without_collapse_the_gap_is_the_pass_through_thickness():
    idomain = [np.ones(2, int), -np.ones(2, int), np.ones(2, int)]
    q = disu.DisuGrid(_grid(nx=2, ny=1), 10.0, [6.0, 4.0, 0.0], idomain, collapse=None).build().quality()
    np.testing.assert_allclose(q[q.ihc == 0].gap, 2.0)
    q = disu.DisuGrid(_grid(nx=2, ny=1), 10.0, [6.0, 4.0, 0.0], idomain).build().quality()
    np.testing.assert_allclose(q.gap, 0.0)                  # collapse="split" hands the 2 m to the neighbours


def test_a_chained_connection_reports_the_skipped_thickness(grids):  # noqa: F811
    base, pit = grids
    idomain = np.ones(len(pit), dtype=int)
    idomain[np.flatnonzero(pit.splice_source.to_numpy() == "fine")[0]] = -1
    d = disu.DisuGrid([base, pit, base], 10.0, [np.full(len(base), 6.0), np.full(len(pit), 4.0),
                                                np.full(len(base), 0.0)], [None, idomain, None]).build()
    q = d.quality()
    assert (q.gap > 0).sum() == d.diagnostics["chained_links"] == 1
    assert q.gap.max() == pytest.approx(d.diagnostics["largest_skipped_run"])


def test_quality_needs_a_built_grid_and_is_in_the_report():
    with pytest.raises(RuntimeError):
        disu.DisuGrid(_grid(), 10.0, [0.0]).quality()
    d = _two_columns(np.array([20.0, 20.0]), [12, 4], [16, 0])
    assert "max 3D orthogonality error: 82.9 deg; largest cl ratio: 4" in d.report()
