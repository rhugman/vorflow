"""Barrier connections of a built DISU grid and their HFB data."""

import geopandas as gpd
import numpy as np
import pytest
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, Point, Polygon, box

from vorflow import _hfb, disu

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


def hanging(off):
    """Two fine cells beside a coarse one; their shared vertex lies ``off`` the coarse face, as on a splice outline."""
    polygons = [Polygon([(990, 0), (1000, 0), (1000 - off, 5), (990, 5)]),
                Polygon([(990, 5), (1000 - off, 5), (1000, 10), (990, 10)]), box(1000, 0, 1010, 10)]
    grid = gpd.GeoDataFrame({"node_id": np.arange(3), "x": [995, 995, 1005], "y": [2.5, 7.5, 5]}, geometry=polygons)
    return disu.DisuGrid(grid, 10.0, [0.0]).build()


@pytest.mark.parametrize("off", [0.0, 1e-12, -1e-12])
def test_faces_a_roundoff_off_their_neighbour_are_cut(off):
    # A roundoff outside the coarse face, the boundaries meet only at points; inside it, the cells overlap.
    d = hanging(off)
    faces = d.barrier_faces(LineString([(1000, 0), (1000, 10)]))
    np.testing.assert_array_equal(faces.pairs, [[0, 2], [1, 2]])
    assert faces.get_gridprops_mf6_hfb(0.1)["stress_period_data"][0] == [((0,), (2,), 0.1), ((1,), (2,), 0.1)]
    polygons = d._grids[0].geometry.to_numpy()
    np.testing.assert_allclose(shapely.length(_hfb._shared_faces(polygons[:2], polygons[2],
                                                                 disu._coordinate_tolerance(d._grids))), 5)


@pytest.mark.parametrize("off", [0.0, 1e-12, -1e-12])
@pytest.mark.parametrize("ends, length", [((-1, 2.5), 2.5), ((2.5, 7.5), 5)])
def test_a_trace_ending_inside_roundoff_faces_is_rejected(off, ends, length):
    # The hair's gap between the cells (off > 0) must not pass for the edge of the grid.
    with pytest.raises(ValueError, match=f"runs {length} off the cell faces"):
        hanging(off).barrier_faces(LineString([(1000, ends[0]), (1000, ends[1])]))


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


def section(ncol=12, nlay=6):
    """One row of unit columns over unit layers: a vertical x-z section."""
    polygons = [box(i, 0, i + 1, 1) for i in range(ncol)]
    grid = gpd.GeoDataFrame({"node_id": np.arange(ncol), "x": [p.centroid.x for p in polygons],
                             "y": [0.5] * ncol}, geometry=polygons)
    return disu.DisuGrid(grid, float(nlay), [float(nlay - 1 - k) for k in range(nlay)]).build()


def turn(xy, angle):
    """Points turned anticlockwise by ``angle`` degrees about the origin."""
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    return np.asarray(xy, dtype=float) @ np.array([[c, s], [-s, c]])


def rotated(n, angle):
    """``boxes(n)`` turned by ``angle`` degrees about the origin, so that its faces run obliquely."""
    grid = boxes(n)
    grid = grid.set_geometry(grid.geometry.rotate(angle, origin=(0, 0)))
    grid["x"], grid["y"] = turn(np.column_stack((grid.x, grid.y)), angle).T
    return grid


def two_point_flows(d, faces, heads, gradient, hydchr):
    """Exact flows and two-point flows with HFB series conductance (K = 1); hydchr per face or one value."""
    xyz, _ = cells(d)
    links = d.links
    n, m = links["n"], links["m"]
    area = np.asarray(links["fahl"], dtype=float)
    conductance = area / (links["cl_n"] + links["cl_m"])
    barrier = np.full(len(n), np.inf)
    barrier[faces.links] = hydchr
    conductance = 1 / (1 / conductance + 1 / (barrier * area))
    normal = np.column_stack((np.cos(np.radians(links["angle"])), np.sin(np.radians(links["angle"])), np.zeros(len(n))))
    vertical = links["ihc"] == 0
    normal[vertical] = 0.0
    normal[vertical, 2] = np.sign(xyz[m[vertical], 2] - xyz[n[vertical], 2])
    return -(normal @ gradient) * area, conductance * (heads[n] - heads[m])


def test_dipping_fault_has_walls_and_treads_with_the_plane_resistance():
    d = section()
    # 45 degrees: each layer's trace one column further than the layer above.
    faces = d.barrier_faces([LineString([(3 + k, -1), (3 + k, 2)]) for k in range(6)])
    xyz, _ = cells(d)
    far = xyz[:, 0] + xyz[:, 2] > 3 + 5.5
    n, m = d.links["n"], d.links["m"]
    np.testing.assert_array_equal(np.sort(faces.links), np.flatnonzero(far[n] != far[m]))
    assert len(faces) == 11 and faces.vertical.sum() == 5
    np.testing.assert_allclose(faces.factor, np.sqrt(0.5))
    # Flow normal to the fault: with the factor every connection carries the exact flow.
    normal, q, hydchr = np.array([1.0, 0.0, 1.0]) / np.sqrt(2), 0.7, 0.4
    heads = 50 - q * xyz @ normal - far * q / hydchr
    exact, with_factor = two_point_flows(d, faces, heads, -q * normal, hydchr * faces.factor)
    _, without = two_point_flows(d, faces, heads, -q * normal, hydchr)
    assert np.abs(with_factor - exact).max() < 1e-12
    assert np.abs(without - exact).max() > 0.1


def stacked(name):
    if name == "uneven offsets":
        xs = [2, 3, 3, 5]
        d = disu.DisuGrid(boxes(6), 4.0, [3.0, 2.0, 1.0, 0.0]).build()
        return d, [LineString([(x, -1), (x, 7)]) for x in xs], lambda xy, k: xy[:, 0] > np.array(xs)[k]
    if name == "oblique strike":
        d = disu.DisuGrid(rotated(8, 30), 3.0, [2.0, 1.0, 0.0]).build()
        return (d, [LineString(turn([(2 + k, -1), (2 + k, 9)], 30)) for k in range(3)],
                lambda xy, k: turn(xy, -30)[:, 0] > 2 + k)
    grid = boxes(8)
    x = grid.x.to_numpy()
    d = disu.DisuGrid(grid, 12 - 0.3 * x, [10 - 0.3 * x - 2 * k for k in range(4)]).build()
    return d, [LineString([(3 + k, -1), (3 + k, 9)]) for k in range(4)], lambda xy, k: xy[:, 0] > 3 + k


@pytest.mark.parametrize("name", ["uneven offsets", "oblique strike", "dipping layers"])
def test_stacked_traces_cut_every_connection_that_changes_side(name):
    d, traces, right = stacked(name)
    faces = d.barrier_faces(traces)
    xyz, layer = cells(d)
    side = right(xyz[:, :2], layer)
    n, m = d.links["n"], d.links["m"]
    np.testing.assert_array_equal(np.sort(faces.links), np.flatnonzero(side[n] != side[m]))
    assert faces.vertical.any() and np.all((faces.factor > 0) & (faces.factor < 1))


@pytest.mark.parametrize("slope", [0.6, -0.6, 0.0])
@pytest.mark.parametrize("partial", [0, 1])
@pytest.mark.parametrize("flip", [False, True])
def test_traces_of_different_length_are_paired_end_to_end(slope, partial, flip):
    # A fault that dies out along the strike in one layer: x = 3 in layer 0, x = 4 in layer 1,
    # one of them only for y in [6, 11]. Where both exist, every change of side is cut, and nothing else.
    grid = boxes(10)
    x = grid.x.to_numpy()
    d = disu.DisuGrid(grid, 10 - slope * x, [7 - slope * x, 4 - slope * x]).build()
    traces = [LineString([(3 + k, 6 if k == partial else -1), (3 + k, 11)]) for k in range(2)]
    if flip:
        traces[partial] = LineString(traces[partial].coords[::-1])
    faces = d.barrier_faces(traces)
    xyz, layer = cells(d)
    side = xyz[:, 0] > 3 + layer
    n, m = d.links["n"], d.links["m"]
    both = (xyz[n, 1] > 6.5) & (xyz[m, 1] > 6.5)
    cut = np.zeros(len(n), dtype=bool)
    cut[faces.links] = True
    assert not (both & (side[n] != side[m]) & ~cut).any()
    assert not (both & (side[n] == side[m]) & cut).any()


def test_a_reversed_trace_gives_the_same_barrier():
    d, traces, _ = stacked("uneven offsets")
    forward = d.barrier_faces(traces)
    backward = d.barrier_faces([traces[0], LineString(traces[1].coords[::-1]), traces[2],
                                LineString(traces[3].coords[::-1])])
    np.testing.assert_array_equal(backward.pairs, forward.pairs)
    np.testing.assert_allclose(backward.factor, forward.factor)


def test_traces_placed_end_to_end_are_rejected():
    # The closest ends of these traces are the inner ones: pairing them would reverse one trace.
    d = disu.DisuGrid(boxes(10), 3.0, [2.0, 1.0]).build()
    with pytest.raises(ValueError, match="do not overlap along the strike"):
        d.barrier_faces([LineString([(3, 5), (3, -1)]), LineString([(4, 6), (4, 11)])])


def test_uneven_offsets_give_the_local_dip():
    d, traces, _ = stacked("uneven offsets")
    faces = d.barrier_faces(traces)
    _, layer = cells(d)
    walls, treads = ~faces.vertical & (layer[faces.n] == layer[faces.m]), faces.vertical
    # Walls in layer 3 see an offset of 2 (factor 1/sqrt(5)); treads from layer 2 to 3 the same offset.
    np.testing.assert_allclose(faces.factor[walls & (layer[faces.n] == 3)], 1 / np.sqrt(5))
    np.testing.assert_allclose(faces.factor[treads & (layer[faces.n] == 2)], 2 / np.sqrt(5))
    np.testing.assert_allclose(faces.factor[walls & (layer[faces.n] == 1)], 1 / np.sqrt(1.25))


def test_dip_is_measured_across_the_strike_up_to_the_grid_edge():
    # Oblique traces that end on the grid edges, 1 m apart across the strike, through 1 m layers: 45 degrees.
    d = disu.DisuGrid(rotated(8, 30), 3.0, [2.0, 1.0, 0.0]).build()
    faces = d.barrier_faces([LineString(turn([(2 + k, 0), (2 + k, 8)], 30)) for k in range(3)])
    assert faces.vertical.any()
    np.testing.assert_allclose(faces.factor, np.sqrt(0.5), atol=1e-9)


def test_fault_tips_close_the_step_polygon():
    d, traces, right = stacked("uneven offsets")
    faces = d.barrier_faces([LineString([(t.coords[0][0], 1), (t.coords[0][0], 5)]) for t in traces])
    xyz, layer = cells(d)
    assert len(faces) and np.all((xyz[faces.pairs, 1] > 1) & (xyz[faces.pairs, 1] < 5))
    assert faces.vertical.any()
    # Between the tips the fault seals the rows y = 1.5 to 4.5; around the tips water can pass.
    between = (xyz[:, 1] > 1) & (xyz[:, 1] < 5)
    side = right(xyz[:, :2], layer)
    assert sealed(d, faces, side, within=between) and not sealed(d, faces, side)


def test_usg_export_divides_vertical_hydchr_by_the_mean_thickness():
    # MODFLOW-USG and USG-Transport multiply a vertical barrier's hydchr by the mean cell thickness
    # (gwf2hfb7u1.f); dividing by it beforehand gives hydchr * area, as in MODFLOW 6.
    d = section()
    faces = d.barrier_faces([LineString([(3 + k, -1), (3 + k, 2)]) for k in range(6)])
    mf6 = faces.get_gridprops_mf6_hfb(0.1)
    usg = faces.get_gridprops_mfusg_hfb(0.1)
    assert mf6["maxhfb"] == usg["nhfbnp"] == 11 and faces.vertical.any()
    hydchr6 = np.array([h for _, _, h in mf6["stress_period_data"][0]])
    hydchr_usg = np.array([h for _, _, h in usg["hfb_data"]])
    thickness = d._node_top - d._node_bot
    mean = (thickness[faces.n] + thickness[faces.m]) / 2
    np.testing.assert_allclose(hydchr_usg, np.where(faces.vertical, hydchr6 / mean, hydchr6))


def test_a_multiplier_cannot_follow_a_dipping_fault():
    d = section()
    faces = d.barrier_faces([LineString([(3 + k, -1), (3 + k, 2)]) for k in range(6)])
    with pytest.raises(ValueError, match="11 of 11 connections"):
        faces.get_gridprops_mf6_hfb(-0.5)
    vertical = d.barrier_faces(LineString([(3, -1), (3, 2)]))
    assert vertical.get_gridprops_mf6_hfb(-0.5)["stress_period_data"][0][0][2] == -0.5


def test_flopy_hfb_packages_write_the_barrier_nodes_one_based(tmp_path):
    flopy = pytest.importorskip("flopy")
    d = section()
    faces = d.barrier_faces([LineString([(3 + k, -1), (3 + k, 2)]) for k in range(6)])
    expected = {(a + 1, b + 1) for a, b in zip(faces.n, faces.m)}

    def pairs(path):
        rows = [line.split() for line in path.read_text().splitlines()]
        # node1 node2 hydchr; the USG header line also has three numbers, all integers
        return {(int(r[0]), int(r[1])) for r in rows if len(r) == 3 and r[0].isdigit() and not r[2].isdigit()}

    sim = flopy.mf6.MFSimulation(sim_ws=str(tmp_path / "mf6"))
    flopy.mf6.ModflowTdis(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="m")
    flopy.mf6.ModflowGwfdisu(gwf, **d.get_gridprops_mf6_disu())
    flopy.mf6.ModflowGwfhfb(gwf, **faces.get_gridprops_mf6_hfb(0.1))
    sim.write_simulation(silent=True)
    assert pairs(tmp_path / "mf6" / "m.hfb") == expected

    usg = flopy.mfusg.MfUsg(modelname="u", model_ws=str(tmp_path / "usg"), structured=False)
    flopy.mfusg.MfUsgDisU(usg, **d.get_gridprops_mfusg_disu())
    flopy.modflow.ModflowHfb(usg, **faces.get_gridprops_mfusg_hfb(0.1))
    usg.write_input()
    assert pairs(tmp_path / "usg" / "u.hfb") == expected


def test_traces_that_differ_must_be_single_lines():
    two = LineString([(3, -1), (3, 2)]).union(LineString([(8, -1), (8, 2)]))
    with pytest.raises(ValueError, match="one LineString"):
        section(nlay=2).barrier_faces([two, LineString([(4, -1), (4, 2)])])


def test_dipping_fault_through_dipping_layers_cuts_staggered_links():
    # Layers 0.7 thick slope 0.3 along x, so traces one column apart make a 45 degree fault plane
    # through each layer's mid-depth; links across the fault include staggered ones.
    grid = boxes(12)
    x = grid.x.to_numpy()
    top = 12 - 0.3 * x
    d = disu.DisuGrid(grid, top, [top - 0.7 * (k + 1) for k in range(6)]).build()
    faces = d.barrier_faces([LineString([(3 + k, -1), (3 + k, 13)]) for k in range(6)])
    xyz, layer = cells(d)
    plane = 3 + 12 - 0.3 * 3 - 0.35          # x + z at the top layer's mid-depth on x = 3
    far = xyz[:, 0] + xyz[:, 2] > plane
    n, m = d.links["n"], d.links["m"]
    np.testing.assert_array_equal(np.sort(faces.links), np.flatnonzero(far[n] != far[m]))
    assert (layer[faces.n] != layer[faces.m]).sum() > faces.vertical.sum()   # staggered links are cut too
    np.testing.assert_allclose(faces.factor, np.sqrt(0.5))


@pytest.mark.slow
def test_one_mesh_with_every_trace_gives_an_exact_dipping_barrier():
    # A 45 degree fault through four 10 m layers. Every layer uses one mesh holding all four traces,
    # so each tread joins two stacked cells and two-point flows are exact everywhere.
    from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator

    normal = np.array([120.0, -40.0]) / np.hypot(120.0, 40.0)
    traces = [LineString([(70 + k * 10 / normal[0], 0), (110 + k * 10 / normal[0], 120)]) for k in range(4)]
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 200, 120), zone_id=1, resolution=10)
    for k, trace in enumerate(traces):
        cm.add_line(trace, line_id=f"fault-{k}", resolution=6, is_barrier=True)
    mg = MeshGenerator(background_lc=10, conceptual_mesh=cm, verbosity=0)
    mg.generate()
    grid = VoronoiTessellator(mg, cm).generate()
    d = disu.DisuGrid(grid, 40.0, [30.0, 20.0, 10.0, 0.0]).build()

    faces = d.barrier_faces(traces)
    xyz, _ = cells(d)
    beyond = (xyz[:, :2] - [70, 0]) @ normal - (35 - xyz[:, 2]) > 0
    n, m = d.links["n"], d.links["m"]
    np.testing.assert_array_equal(np.sort(faces.links), np.flatnonzero(beyond[n] != beyond[m]))
    gradient, hydchr = -0.01 * np.r_[normal, 1.0] / np.sqrt(2), 0.01
    heads = 50 + xyz @ gradient - beyond * 0.01 / hydchr
    exact, flows = two_point_flows(d, faces, heads, gradient, hydchr * faces.factor)
    np.testing.assert_allclose(flows, exact, rtol=0, atol=1e-12 * np.abs(exact).max())
