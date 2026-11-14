"""Ghost-node corrections: geometry and package input only (no MODFLOW runs)."""

import warnings

import geopandas as gpd
import numpy as np
import pytest
import shapely
from scipy.optimize import linprog
from shapely.geometry import MultiPoint, Point, Polygon, box

from vorflow._gnc import _weights
from vorflow.disu import DisuGrid, splice


def plan(polygons):
    return gpd.GeoDataFrame({"node_id": np.arange(len(polygons)), "x": [p.centroid.x for p in polygons],
                             "y": [p.centroid.y for p in polygons]}, geometry=polygons)


def squares(n, size=1.0):
    return plan([box(x * size / n, y * size / n, (x + 1) * size / n, (y + 1) * size / n)
                 for y in range(n) for x in range(n)])


def refined_grid(**options):
    """Two coarse cells beside eight fine ones: plan offsets at the change of resolution."""
    polygons = [box(-2, 0, 0, 2), box(-2, -2, 0, 0)] + [box(x, y, x + 1, y + 1) for y in range(-2, 2) for x in range(2)]
    return DisuGrid(plan(polygons), 1.0, [0.0], **options).build()


def sandwich():
    """A 3 x 3 layer between two 2 x 2 layers: every vertical connection leans."""
    return DisuGrid([squares(2), squares(3), squares(2)], 3.0, [2.0, 1.0, 0.0]).build()


def boundary_grid():
    """Dipping layers on a Voronoi mesh clipped at the domain edge, where ghosts cannot all stay in their cell."""
    xy = np.array([(0, 0), (2, 0), (4, 0), (4, 2), (4, 4), (2, 4), (0, 4), (0, 2), (0.6, 0.8), (1.7, 0.6),
                   (3.2, 0.7), (0.8, 2.8), (2.1, 2.2), (3.4, 3.1)])
    domain = box(0, 0, 4, 4)
    cells = shapely.voronoi_polygons(MultiPoint(xy), extend_to=domain, ordered=True).geoms
    grid = plan([p.intersection(domain) for p in cells])
    grid["x"], grid["y"] = xy.T
    dip = 0.6 * xy[:, 0]
    return DisuGrid(grid, 10 - dip, [7 - dip, 4 - dip, 1 - dip]).build()


def centres(d):
    col = d._node_cell
    return np.column_stack((d._columns["x"].to_numpy()[col], d._columns["y"].to_numpy()[col],
                            (d._node_top + d._node_bot) / 2))


def ghost_head(record, heads):
    return heads[record["n"]] + np.asarray(record["weights"]) @ (heads[record["contributors"]] - heads[record["n"]])


@pytest.mark.parametrize("make", [refined_grid, sandwich, boundary_grid])
def test_ghost_heads_reproduce_linear_heads_with_bounded_weights(make):
    d = make()
    gnc = d.estimate_gnc()
    assert gnc.records and not gnc.unsupported
    xyz = centres(d)
    for gradient in [*np.eye(3), [0.71, -0.39, 0.2]]:
        heads = 31.0 + xyz @ gradient
        for r in gnc.records:
            np.testing.assert_allclose(ghost_head(r, heads), 31.0 + np.asarray(r["target"]) @ gradient, atol=1e-10)
    for r in gnc.records:
        assert 1 <= len(r["contributors"]) <= 5 and r["n"] not in r["contributors"]
        assert min(r["weights"]) >= 0 and 1 - sum(r["weights"]) >= 1e-8 - 1e-12


def test_corrected_flows_are_exact_for_a_linear_field_even_at_clipped_cells():
    d = boundary_grid()
    gnc = d.estimate_gnc()
    assert not gnc.unsupported
    xyz = centres(d)
    gradient = np.array([0.2, -0.1, 0.7])
    heads = 31 + xyz @ gradient
    links = d.links
    a, b, area = np.asarray(links["n"]), np.asarray(links["m"]), np.asarray(links["fahl"], float)
    conductance = area / (np.asarray(links["cl_n"]) + np.asarray(links["cl_m"]))
    flow = conductance * (heads[b] - heads[a])        # standard two-point flow from a to b, K = 1
    for r in gnc.records:                             # each ghost replaces its host's head in that flow
        sign = -1 if r["n"] == a[r["link"]] else 1
        flow[r["link"]] += sign * conductance[r["link"]] * (ghost_head(r, heads) - heads[r["n"]])
    angle = np.radians(np.asarray(links["angle"], float))
    normal = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(a))))
    vertical = np.asarray(links["ihc"]) == 0
    normal[vertical] = np.c_[np.zeros((vertical.sum(), 2)), np.sign(xyz[b[vertical], 2] - xyz[a[vertical], 2])]
    np.testing.assert_allclose(flow, area * (normal @ gradient), atol=1e-8)


def test_ghosts_may_leave_their_cell_only_inside_the_cells_that_interpolate_them():
    d = boundary_grid()
    gnc = d.estimate_gnc()
    polygons = d._columns.geometry.to_numpy()[d._node_cell]

    def holds(j, target):
        return (polygons[j].buffer(1e-8).covers(Point(*target[:2]))
                and d._node_bot[j] - 1e-8 <= target[2] <= d._node_top[j] + 1e-8)
    outside = [r for r in gnc.records if not holds(r["n"], r["target"])]
    assert outside, "fixture should need ghosts outside their own cell"
    for r in outside:
        assert any(holds(j, r["target"]) for j in r["contributors"])


def test_second_ring_of_neighbours_is_used_when_the_first_cannot_interpolate():
    d = sandwich()
    gnc = d.estimate_gnc()
    assert gnc.corrected_connections == 32 and not gnc.unsupported
    neighbours = [set() for _ in range(d.nodes)]
    for a, b in zip(d.links["n"], d.links["m"]):
        neighbours[int(a)].add(int(b))
        neighbours[int(b)].add(int(a))
    assert any(set(r["contributors"]) - neighbours[r["n"]] for r in gnc.records)


def test_weight_solver_agrees_with_a_general_lp_solver():
    rng = np.random.default_rng(7)
    cap = 1 - 1e-8
    solved = 0
    for trial in range(300):
        n = int(rng.integers(1, 13))
        rows = rng.normal(size=(3, n))
        if trial % 4 == 0:
            rows[2] = 0.0                                   # one layer: no vertical offsets to reproduce
        # Half the right-hand sides come from valid weights; the others are mostly out of reach.
        rhs = rows @ rng.uniform(0, 1.5 / n, n) if trial % 2 else rng.normal(size=3) * (rows.any(axis=1))
        cost = rng.uniform(0.1, 1.0, n)
        fit = linprog(cost, A_eq=rows, b_eq=rhs, A_ub=np.ones((1, n)), b_ub=[cap], bounds=(0, None), method="highs")
        w = _weights(cost, rows, rhs, cap)
        assert (w is not None) == fit.success
        if w is not None:
            solved += 1
            assert w.min() >= 0 and w.sum() <= cap + 1e-12
            np.testing.assert_allclose(rows @ w, rhs, atol=1e-9)
            assert cost @ w == pytest.approx(fit.fun, abs=1e-8)
    assert 100 < solved < 250, "the trials should mix problems with and without a solution"


def test_aligned_grid_needs_no_correction():
    d = DisuGrid(squares(3), 2.0, [1.0, 0.0]).build()
    gnc = d.estimate_gnc()
    assert not gnc.records and not gnc.unsupported
    assert gnc.get_gridprops_mf6_gnc() is None and gnc.get_gridprops_mfusg_gnc() is None
    assert gnc.write_mfusg_gnc("unused.gnc") is None


def test_roundoff_at_a_spliced_hanging_vertex_keeps_the_two_ghost_correction():
    base = plan([box(x, y, x + 1, y + 1) for y in range(4) for x in range(4)])
    fine = plan([box(x / 2, y / 2, (x + 1) / 2, (y + 1) / 2) for y in range(8) for x in range(8)])
    grid = splice(base, fine, box(1, 1, 3, 3))
    dip = 0.6 * grid.x.to_numpy()
    d = DisuGrid(grid, 10 - dip, [7 - dip, 4 - dip, 1 - dip]).build()
    before = d.estimate_gnc()
    per_link = {}
    for r in before.records:
        per_link.setdefault(r["link"], []).append(r)
    col = d._node_cell
    source = grid.splice_source.to_numpy()
    pair = next((rs for rs in per_link.values() if len(rs) == 2 and source[col[rs[0]["n"]]] != source[col[rs[0]["m"]]]
                 and d.links["ihc"][rs[0]["link"]] != 0), None)
    assert pair is not None, "fixture needs a two-ghost correction across the splice outline"
    a, b = col[pair[0]["n"]], col[pair[0]["m"]]
    polygon, other = d._columns.geometry.iloc[a], d._columns.geometry.iloc[b]
    midpoint = polygon.boundary.intersection(other.boundary).centroid
    xy = np.array(midpoint.coords[0])
    inward = np.array(polygon.centroid.coords[0]) - xy
    ring = list(polygon.exterior.coords)
    for i, (start, end) in enumerate(zip(ring, ring[1:])):
        if shapely.LineString([start, end]).distance(midpoint) < 1e-14:
            ring.insert(i + 1, tuple(xy + 5e-14 * inward / np.linalg.norm(inward)))
            break
    # Independently meshed splice outlines can differ by roundoff at hanging vertices.
    d._columns.at[d._columns.index[a], "geometry"] = Polygon(ring)
    assert d._columns.geometry.iloc[a].boundary.intersection(other.boundary).length < 1e-10
    after = d.estimate_gnc()
    link = pair[0]["link"]
    assert not any(u["link"] == link for u in after.unsupported)
    assert sum(r["link"] == link for r in after.records) == 2


def test_mf6_input_pads_contributors_without_changing_ghost_heads():
    d = boundary_grid()
    gnc = d.estimate_gnc()
    props = gnc.get_gridprops_mf6_gnc()
    width = props["numalphaj"]
    assert props["numgnc"] == gnc.record_count == len(props["gncdata"]) and not props["explicit"]
    assert width == max(len(r["weights"]) for r in gnc.records)
    heads = np.random.default_rng(3).normal(size=d.nodes)
    for row, record in zip(props["gncdata"], gnc.records):
        n, m = row[0][0], row[1][0]
        js = [cell[0] for cell in row[2:2 + width]]
        ws = np.asarray(row[2 + width:])
        assert (n, m) == (record["n"], record["m"]) and len(ws) == width
        np.testing.assert_allclose(heads[n] + ws @ (heads[js] - heads[n]), ghost_head(record, heads), atol=1e-12)
    assert gnc.get_gridprops_mf6_gnc(explicit=True)["explicit"]


def test_mfusg_input_and_file_are_one_based_with_full_precision(tmp_path):
    d = sandwich()
    gnc = d.estimate_gnc()
    props = gnc.get_gridprops_mfusg_gnc()
    assert props["numgnc"] == gnc.record_count and props["isymgncn"] == 1
    path = gnc.write_mfusg_gnc(tmp_path / "m.gnc", explicit=False)
    lines = path.read_text().splitlines()
    width = props["numalphaj"]
    assert lines[1].split() == ["0", "0", str(gnc.record_count), str(width), "0", "0", "0"]
    for line, row in zip(lines[2:], props["gncdata"]):
        values = line.split()
        assert [int(v) for v in values[:2 + width]] == [int(v) + 1 for v in row[:2 + width]]
        np.testing.assert_array_equal([float(v) for v in values[2 + width:]], row[2 + width:])
    assert len(lines) == 2 + gnc.record_count


def test_plan_distances_and_a_built_grid_are_required():
    with pytest.raises(ValueError, match="cl12"):
        refined_grid(cl12="connector").estimate_gnc()
    with pytest.raises((ValueError, RuntimeError), match="build"):
        DisuGrid(plan([box(0, 0, 1, 1)]), 1.0, [0.0]).estimate_gnc()


def test_no_warning_when_every_connection_is_corrected():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gnc = refined_grid().estimate_gnc()
    assert gnc.corrected_connections <= gnc.record_count
