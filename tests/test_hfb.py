"""Barrier connections of a built DISU grid and their HFB data."""

import geopandas as gpd
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, Point, box

from vorflow import disu

FAULT = LineString([(2, -1), (2, 5)])


def boxes(n, size=1.0):
    polygons = [box(i * size, j * size, (i + 1) * size, (j + 1) * size) for j in range(n) for i in range(n)]
    return gpd.GeoDataFrame({"node_id": np.arange(len(polygons)),
                             "x": [p.centroid.x for p in polygons],
                             "y": [p.centroid.y for p in polygons]}, geometry=polygons)


def spliced():
    """Three 4 x 4 layers over three layers refined to 0.5 m in [1, 3] x [1, 3]."""
    base = boxes(4)
    pit = disu.splice(base, boxes(8, 0.5), box(1, 1, 3, 3))
    return disu.DisuGrid([base] * 3 + [pit] * 3, 10, [8, 6, 4, 2, 0, -2]).build()


def dipping():
    """4 x 4 columns, three layers dipping along x, so links across x = 2 are staggered."""
    grid = boxes(4)
    x = grid.x.to_numpy()
    return disu.DisuGrid(grid, 10.0 - 0.6 * x, [7.0 - 0.6 * x, 4.0 - 0.6 * x, 1.0 - 0.6 * x]).build()


def cells(d):
    c = d.cellmap[d.cellmap.node >= 0].sort_values("node")
    return np.column_stack((c.x, c.y, (c.top + c.bot) / 2)), c.layer.to_numpy()


def crossing(d, x0=2.0):
    """Horizontal links whose two centres lie on opposite sides of x = x0."""
    xyz, _ = cells(d)
    n, m = d.links["n"], d.links["m"]
    return np.flatnonzero((d.links["ihc"] != 0) & ((xyz[n, 0] - x0) * (xyz[m, 0] - x0) < 0))


def sealed(d, faces, side, within=None):
    """With the cut connections removed, no cell on one side reaches the other (moving only
    through the cells ``within``, all by default)."""
    keep = np.ones(len(d.links["n"]), dtype=bool)
    keep[faces.links] = False
    if within is not None:
        keep &= within[d.links["n"]] & within[d.links["m"]]
        side = side & within, ~side & within
    else:
        side = side, ~side
    n, m = d.links["n"][keep], d.links["m"][keep]
    _, label = connected_components(coo_matrix((np.ones(len(n)), (n, m)), shape=(d.nodes,) * 2), directed=False)
    return not set(label[side[0]]) & set(label[side[1]])


def test_vertical_trace_finds_the_faces_on_it_in_every_layer():
    d = spliced()
    faces = d.barrier_faces(FAULT)
    _, layer = cells(d)
    np.testing.assert_array_equal(np.sort(faces.links), crossing(d))
    np.testing.assert_array_equal(np.bincount(layer[faces.n]), [4, 4, 4, 6, 6, 6])
    assert len(faces) == 30
    np.testing.assert_array_equal(faces.pairs, np.column_stack((faces.n, faces.m)))
    xyz, _ = cells(d)
    assert sealed(d, faces, xyz[:, 0] > 2)


def test_one_trace_per_layer_limits_the_barrier_to_those_layers():
    d = spliced()
    faces = d.barrier_faces([None] * 3 + [FAULT] * 3)
    _, layer = cells(d)
    assert len(faces) == 18 and set(layer[faces.n]) == {3, 4, 5}


def test_staggered_links_across_a_dipping_stack_are_cut():
    d = dipping()
    faces = d.barrier_faces(FAULT)
    _, layer = cells(d)
    np.testing.assert_array_equal(np.sort(faces.links), crossing(d))
    assert (layer[faces.n] != layer[faces.m]).any()


def test_a_closed_trace_cuts_the_faces_around_it():
    d = disu.DisuGrid(boxes(6), 1.0, [0.0]).build()
    ring = box(2, 2, 4, 4).exterior
    faces = d.barrier_faces(ring)
    xyz, _ = cells(d)
    inside = (abs(xyz[:, 0] - 3) < 1) & (abs(xyz[:, 1] - 3) < 1)
    assert len(faces) == 8 and sealed(d, faces, inside)


def test_a_trace_off_the_faces_is_rejected():
    d = disu.DisuGrid(boxes(6), 1.0, [0.0]).build()
    with pytest.raises(ValueError, match="is_barrier=True"):
        d.barrier_faces(LineString([(-0.1, 0.35), (6.1, 4.9)]))


def test_a_trace_ending_inside_a_face_is_rejected():
    d = disu.DisuGrid(boxes(6), 1.0, [0.0]).build()
    with pytest.raises(ValueError, match=r"runs 0.5 off the cell faces, starting near \(3, 2"):
        d.barrier_faces(LineString([(3, -1), (3, 2.5)]))


@pytest.mark.parametrize("east", [3, -1])
def test_faces_meeting_at_a_junction_are_all_cut(east):
    # A T (east = 3) or a crossing (east = -1) of barriers on the faces x = 3 and y = 3. The cells beside the
    # junction are as far from one line as from the other; every face on either line is still cut.
    d = disu.DisuGrid(boxes(6), 1.0, [0.0]).build()
    faces = d.barrier_faces(LineString([(3, -1), (3, 7)]).union(LineString([(east, 3), (7, 3)])))
    xyz, _ = cells(d)
    a, b = xyz[d.links["n"], :2], xyz[d.links["m"], :2]
    across_x = (a[:, 0] - 3) * (b[:, 0] - 3) < 0
    across_y = ((a[:, 1] - 3) * (b[:, 1] - 3) < 0) & (a[:, 0] > east)
    np.testing.assert_array_equal(np.sort(faces.links), np.flatnonzero(across_x | across_y))
    # The lines split the grid into three (T) or four (crossing) regions, none reaching another.
    region = np.where(xyz[:, 0] < 3, 0 if east == 3 else 2 * (xyz[:, 1] > 3), 1 + 2 * (xyz[:, 1] > 3))
    keep = np.ones(len(d.links["n"]), dtype=bool)
    keep[faces.links] = False
    n, m = d.links["n"][keep], d.links["m"][keep]
    count, label = connected_components(coo_matrix((np.ones(len(n)), (n, m)), shape=(d.nodes,) * 2), directed=False)
    assert count == len(set(region)) == (3 if east == 3 else 4)
    assert all(len(set(label[region == r])) == 1 for r in set(region))


def test_hfb_package_data():
    faces = spliced().barrier_faces(FAULT)
    mf6 = faces.get_gridprops_mf6_hfb(0.5)
    assert mf6["maxhfb"] == 30 and len(mf6["stress_period_data"][0]) == 30
    assert mf6["stress_period_data"][0][0] == ((int(faces.n[0]),), (int(faces.m[0]),), 0.5)
    hydchr = np.linspace(0.0, 1.0, 30)
    usg = faces.get_gridprops_mfusg_hfb(hydchr)
    assert usg["nhfbnp"] == 30
    assert usg["hfb_data"][-1] == [int(faces.n[-1]), int(faces.m[-1]), 1.0]
    assert faces.get_gridprops_mf6_hfb(-0.1)["stress_period_data"][0][0][2] == -0.1


@pytest.mark.parametrize("hydchr", [np.nan, np.inf, np.ones(3), "low"])
def test_invalid_hydchr_is_rejected(hydchr):
    with pytest.raises(ValueError, match="hydchr"):
        spliced().barrier_faces(FAULT).get_gridprops_mf6_hfb(hydchr)


def test_trace_outside_the_grid_gives_no_package():
    faces = spliced().barrier_faces(LineString([(10, 0), (10, 4)]))
    assert len(faces) == 0
    assert faces.get_gridprops_mf6_hfb(0.0) is None and faces.get_gridprops_mfusg_hfb(0.0) is None


@pytest.mark.parametrize("trace", [box(0, 0, 1, 1), Point(1, 1), LineString(), [FAULT] * 5, 3.0])
def test_invalid_trace_is_rejected(trace):
    with pytest.raises(ValueError, match="trace"):
        spliced().barrier_faces(trace)


@pytest.mark.slow
def test_a_fault_meshed_as_a_barrier_is_cut_on_its_faces():
    from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator

    fault = LineString([(70, 0), (110, 120)])
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 200, 120), zone_id=1, resolution=10)
    cm.add_line(fault, line_id="fault", resolution=6, is_barrier=True)
    mg = MeshGenerator(background_lc=10, conceptual_mesh=cm, verbosity=0)
    mg.generate()
    grid = VoronoiTessellator(mg, cm).generate()
    d = disu.DisuGrid(grid, 2.0, [1.0, 0.0]).build()
    faces = d.barrier_faces(fault)
    xyz, _ = cells(d)
    west = (110 - 70) * xyz[:, 1] - 120 * (xyz[:, 0] - 70) > 0
    assert len(faces) and sealed(d, faces, west)
    np.testing.assert_array_equal(np.sort(faces.links), np.flatnonzero(west[d.links["n"]] != west[d.links["m"]]))
