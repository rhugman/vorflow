"""Different plan grids per layer: splice, global columns, overlay links and gap-free contacts."""

import functools

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from vorflow import disu
from vorflow.disu import _merge_plan_slivers

from test_disu_grid import assert_cells_close


def _boxes(n, size, start_id=0):
    polygons = [box(i * size, j * size, (i + 1) * size, (j + 1) * size) for j in range(n) for i in range(n)]
    return gpd.GeoDataFrame({
        "node_id": np.arange(start_id, start_id + len(polygons)),
        "x": [p.centroid.x for p in polygons], "y": [p.centroid.y for p in polygons],
    }, geometry=polygons)


PIT = box(1, 1, 3, 3)


@pytest.fixture
def grids():
    base = _boxes(4, 1.0)                    # 16 cells of 1 m
    fine = _boxes(8, 0.5, start_id=500)      # 64 cells of 0.5 m
    return base, disu.splice(base, fine, PIT)


def test_splice_keeps_base_outside_and_fine_inside(grids):
    base, pit = grids
    assert len(pit) == 12 + 16
    outside = pit[pit.splice_source == "base"]
    assert list(outside.node_id) == list(outside.source_node_id)
    assert all(a.equals(b) for a, b in zip(outside.geometry, base.set_index("node_id").loc[outside.node_id].geometry))
    inside = pit[pit.splice_source == "fine"]
    assert inside.node_id.min() == 16 and inside.node_id.is_unique
    assert inside.geometry.area.sum() == pytest.approx(PIT.area)


def test_splice_numbers_fine_cells_after_base_cells_without_node_id():
    base, fine = _boxes(4, 1.0).drop(columns="node_id"), _boxes(8, 0.5).drop(columns="node_id")
    pit = disu.splice(base, fine, PIT)
    assert pit.node_id.is_unique
    outside, inside = pit[pit.splice_source == "base"], pit[pit.splice_source == "fine"]
    assert list(outside.node_id) == list(outside.source_node_id)
    assert inside.node_id.min() == outside.node_id.max() + 1


def test_splice_rejects_cells_crossing_the_outline():
    base, fine = _boxes(4, 1.0), _boxes(8, 0.5)
    with pytest.raises(ValueError, match="cross the splice outline"):
        disu.splice(base, fine, box(0.5, 0.5, 3, 3))
    with pytest.raises(ValueError, match="cross the splice outline"):
        disu.splice(base, fine, box(1, 1, 2.75, 2.75))


def _pit_model(grids, botm_value=None, **kwargs):
    base, pit = grids
    nb, np_ = len(base), len(pit)
    botm = botm_value or [np.full(nb, 6.0), np.full(nb, 4.0), np.full(np_, 2.0), np.full(np_, 0.0)]
    return disu.DisuGrid([base, base, pit, pit], 10.0, botm, **kwargs).build()


def test_repeated_footprints_in_distinct_grids_do_not_duplicate_connections():
    grids = [_boxes(2, .5), _boxes(3, 1/3), _boxes(2, .5, start_id=100)]
    d = disu.DisuGrid(grids, 3., [2., 1., 0.]).build()
    links = pd.DataFrame(d.links)
    pairs = np.sort(links[["n", "m"]].to_numpy(), axis=1)
    assert len(np.unique(pairs, axis=0)) == len(pairs)
    cells = d.cellmap[d.cellmap.node >= 0].sort_values("node")
    layer = cells.layer.to_numpy()
    for upper in [0, 1]:
        use = (links.ihc == 0) & (layer[links.n] == upper) & (layer[links.m] == upper+1)
        assert links.loc[use, "fahl"].sum() == pytest.approx(1.)
    for props in [d.get_gridprops_mf6_disu(), d.get_gridprops_mfusg_disu()]:
        assert len(props["ja"]) == d.nodes + 2*len(links)


def test_columns_and_links_across_a_change_of_grid(grids):
    base, pit = grids
    d = _pit_model(grids)
    assert d.nodes == 16 * 2 + 28 * 2
    assert [len(m) for m in d.nodemap] == [16, 16, 28, 28]
    links = pd.DataFrame(d.links)
    layer_of = np.repeat(np.arange(4), [16, 16, 28, 28])
    # Overlay links join the 4 coarse pit cells of layer 1 to the 16 fine cells of layer 2.
    overlay = links[(links.ihc == 0) & (layer_of[links.n] == 1) & (layer_of[links.m] == 2)]
    shared = overlay.groupby("n").hwva.sum()
    assert len(overlay) == 12 + 16                         # 12 exterior columns continue, 16 overlaps.
    np.testing.assert_allclose(shared, 1.0)                # Area conserved for every upper cell.
    # On the ring, each exterior face of 1 m meets two fine faces of 0.5 m.
    ring = links[(links.ihc > 0) & (layer_of[links.n] == 2) & (layer_of[links.m] == 2)]
    assert sorted(set(np.round(ring.hwva, 9))) == [0.5, 1.0]
    assert d.diagnostics["vertical_offset"] == 0
    assert "on 2 grids" in d.report()


def test_nested_refinement_keeps_each_coarse_bottom(grids):
    base, pit = grids
    slope = 6.0 - 0.1 * base.x.to_numpy()
    d = _pit_model(grids, [slope, slope - 2, np.full(len(pit), 1.0), np.full(len(pit), 0.0)])
    cm = d.cellmap
    fine_l2 = cm[(cm.layer == 2) & (cm.grid == 1) & (cm.node_id >= 16)]
    # A fine cell centred at x=1.25 lies in the coarse cell centred at x=1.5: bottom 6 - 0.15 - 2.
    first = fine_l2[np.isclose(fine_l2.x, 1.25)].iloc[0]
    assert first.top == pytest.approx(3.85)
    # Each coarse cell is its own group, so nothing moves.
    assert d.diagnostics["contact_groups"] == 4
    assert d.diagnostics["max_contact_shift"] == 0.0
    np.testing.assert_array_equal(cm[(cm.layer == 1)].bot.to_numpy(), slope - 2)


def _straddling_pit(base):
    """Fine cells inside PIT whose middle row and column straddle the coarse edges at 2."""
    edges = [1.0, 1.75, 2.25, 3.0]
    polygons = [box(edges[i], edges[j], edges[i + 1], edges[j + 1]) for j in range(3) for i in range(3)]
    fine = gpd.GeoDataFrame({"node_id": np.arange(900, 909), "x": [p.centroid.x for p in polygons],
                             "y": [p.centroid.y for p in polygons]}, geometry=polygons)
    return disu.splice(base, fine, PIT)


def _contact_is_gap_free(d, upper, lower):
    """Every cell starting in ``lower`` meets the bottoms of all cells it overlaps in ``upper``."""
    cm = d.cellmap
    above = cm[cm.layer == upper].reset_index(drop=True)
    below = cm[cm.layer == lower].reset_index(drop=True)
    ga, gb = d.grids_per_layer[upper].geometry.to_numpy(), d.grids_per_layer[lower].geometry.to_numpy()
    for i, cell in enumerate(gb):
        hits = [k for k, g in enumerate(ga) if g.intersection(cell).area > 1e-9]
        assert all(above.bot[k] == below.top[i] for k in hits)


def test_a_straddling_refinement_gets_one_gap_free_contact():
    base = _boxes(4, 1.0)
    pit = _straddling_pit(base)
    slope = 6.0 - 0.1 * base.x.to_numpy()
    d = disu.DisuGrid([base, base, pit, pit], 10.0, [slope, slope - 2, np.full(len(pit), 1.0),
                                                     np.full(len(pit), 0.0)]).build()
    _contact_is_gap_free(d, 1, 2)
    # The four coarse pit cells are linked by the straddling cells: one group at their mean bottom.
    inside = d.cellmap[(d.cellmap.layer == 1) & d.cellmap.x.between(1, 3) & d.cellmap.y.between(1, 3)]
    np.testing.assert_allclose(inside.bot, 6.0 - 0.1 * 2.0 - 2)
    assert d.diagnostics["contact_groups"] == 1
    assert d.diagnostics["max_contact_shift"] == pytest.approx(0.05)
    assert d.diagnostics["vertical_offset"] == 0
    assert "1 contact groups made gap-free" in d.report()
    # Columns outside the pit continue unchanged.
    outside = d.cellmap[(d.cellmap.layer == 1) & ~(d.cellmap.x.between(1, 3) & d.cellmap.y.between(1, 3))]
    np.testing.assert_allclose(outside.bot, 6.0 - 0.1 * outside.x - 2)


def test_given_tops_must_meet_the_bottoms_above():
    base = _boxes(4, 1.0)
    pit = _straddling_pit(base)
    slope = 6.0 - 0.1 * base.x.to_numpy()
    botm = [slope, slope - 2, np.full(len(pit), 1.0), np.full(len(pit), 0.0)]
    with pytest.raises(ValueError, match="do not meet the bottoms"):
        disu.DisuGrid([base, base, pit, pit], [10.0, None, np.full(len(pit), 3.8), None], botm).build()
    flat = [np.full(len(base), 6.0), np.full(len(base), 4.0), np.full(len(pit), 1.0), np.full(len(pit), 0.0)]
    d = disu.DisuGrid([base, base, pit, pit], [10.0, None, np.full(len(pit), 4.0), None], flat).build()
    _contact_is_gap_free(d, 1, 2)


def test_lists_shaped_like_botm_round_trip(grids):
    d = _pit_model(grids)
    k = [np.full(16, 1.0), np.full(16, 2.0), np.full(28, 3.0), np.full(28, 4.0)]
    nodes = d.to_nodes(k)
    assert nodes.shape == (88,) and list(np.unique(nodes)) == [1, 2, 3, 4]
    back = d.to_layers(nodes)
    assert [len(b) for b in back] == [16, 16, 28, 28]
    np.testing.assert_array_equal(back[2], 3.0)
    with pytest.raises(ValueError, match="one entry per layer"):
        d.to_nodes(k[:3])


def test_plan_slivers_join_the_largest_overlap_of_the_smaller_cell():
    area = np.array([10.0, 4.0, 5.0, 0.5])
    overlaps = pd.DataFrame({"col_a": [0, 0, 3], "col_b": [1, 2, 1], "area": [0.1, 5.0, 0.3]})
    # Overlaps are compared with sliver_abs_tol squared: 0.1 is below 0.16, 0.3 is not.
    merged, count = _merge_plan_slivers(overlaps, area, 0.4)
    assert count == 1
    # It joins the largest other overlap of cell 1, the smaller cell; total area is kept.
    np.testing.assert_allclose(sorted(merged.area), [0.4, 5.0])
    assert _merge_plan_slivers(overlaps, area, 0.3)[1] == 0      # 0.1 > 0.09
    assert _merge_plan_slivers(overlaps, area)[1] == 0
    # A sliver that is a cell's only overlap stays.
    alone = pd.DataFrame({"col_a": [0], "col_b": [1], "area": [0.01]})
    assert _merge_plan_slivers(alone, area, 0.4)[1] == 0


@pytest.mark.parametrize("dip", [0.0, 0.2])
def test_pit_refined_at_depth_cells_close(grids, dip):
    """Hanging-node ring faces, overlay links and cross-grid faces cover every cell's geometry."""
    base, pit = grids
    tilt_b, tilt_p = dip * base.x.to_numpy(), dip * pit.x.to_numpy()
    botm = [8.0 + tilt_b, 6.0 + tilt_b, 4.0 + tilt_b, 2.0 + tilt_p, 0.0 + tilt_p, -2.0 + tilt_p]
    d = disu.DisuGrid([base, base, base, pit, pit, pit], 10.0 + tilt_b, botm).build()
    assert d.diagnostics["vertical_offset"] == 0
    assert_cells_close(d)
    _contact_is_gap_free(d, 2, 3)


def _assert_no_vertical_holes(d):
    """No part of a cell's bottom is left unconnected above an active cell (only pass-through between)."""
    import shapely

    n, m, ihc, hwva = (np.asarray(d.links[key]) for key in ("n", "m", "ihc", "hwva"))
    vertical = ihc == 0
    geoms, col = d._columns.geometry.to_numpy(), d._node_cell
    covered = np.bincount(n[vertical], weights=hwva[vertical], minlength=d.nodes)
    for a in np.flatnonzero(np.abs(covered - d._area[col]) > 1e-9):
        hole = geoms[col[a]].difference(shapely.union_all(geoms[col[m[vertical & (n == a)]]]))
        lower = np.flatnonzero(d._node_top <= d._node_bot[a] + 1e-9)
        assert not (shapely.area(shapely.intersection(hole, geoms[col[lower]])) > 1e-9).any(), a


def test_a_pass_through_cell_between_two_changes_of_grid_keeps_the_column_connected(grids):
    """Grids base, pit, base: the coarse cells above and below a pass-through fine cell meet through it."""
    base, pit = grids
    idomain = np.ones(len(pit), dtype=int)
    hole = np.flatnonzero(pit.splice_source.to_numpy() == "fine")[0]
    idomain[hole] = -1
    d = disu.DisuGrid([base, pit, base], 10.0, [np.full(len(base), 6.0), np.full(len(pit), 4.0),
                                                np.full(len(base), 0.0)], [None, idomain, None]).build()
    assert d.diagnostics["chained_links"] == 1
    assert d.diagnostics["largest_skipped_run"] == pytest.approx(2.0)
    assert "1 vertical connections cross pass-through cells" in d.report()
    n, m, hwva = (np.asarray(d.links[key]) for key in ("n", "m", "hwva"))
    chained = (d._node_layer[n] == 0) & (d._node_layer[m] == 2)
    assert hwva[chained] == pytest.approx([0.25])
    assert d._node_cell[n[chained]] == d._node_cell[m[chained]]
    assert_cells_close(d)
    _assert_no_vertical_holes(d)


@pytest.mark.parametrize("layout", ["ABAB", "ABBA", "ABCA"])
def test_pass_through_runs_across_several_changes_of_grid_leave_no_holes(grids, layout):
    base, pit = grids
    finer = disu.splice(base, _boxes(16, 0.25, start_id=900), PIT)
    pick = {"A": base, "B": pit, "C": finer}
    rng = np.random.default_rng(7)
    for _ in range(8):
        layers = [pick[c] for c in layout]
        tilt = rng.uniform(0, 0.5)
        botm = [20 - 3 * (k + 1) + tilt * g.x.to_numpy() - rng.uniform(0, 2, len(g)) * (rng.random(len(g)) > 0.3)
                for k, g in enumerate(layers)]
        for k in range(1, len(layers)):
            if layers[k] is layers[k - 1]:
                botm[k] = np.minimum(botm[k], botm[k - 1])
        # Pass-through cells in the inner layers only: in the top or bottom layer they leave a real hole.
        idomain = [rng.choice([-1, 1, 1, 1], len(g)) if 0 < k < len(layers) - 1 else np.ones(len(g), dtype=int)
                   for k, g in enumerate(layers)]
        collapse = str(rng.choice(["split", "up", "down"]))
        d = disu.DisuGrid(layers, 20 + tilt * base.x.to_numpy(), botm, idomain, collapse=collapse).build()
        assert_cells_close(d)
        _assert_no_vertical_holes(d)


VORONOI_DOMAIN, VORONOI_PIT = box(0, 0, 1000, 600), box(400, 200, 600, 400)


@functools.lru_cache(maxsize=None)
def voronoi_pit(pit_resolution=None):
    """The notebook's Voronoi model: 60 m cells, the pit outline as a barrier, optionally 12 m inside.

    Returns ``(tessellator, grid)``.
    """
    from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator

    cm = ConceptualMesh()
    cm.add_polygon(VORONOI_DOMAIN, zone_id=1, resolution=60)
    cm.add_line(LineString(VORONOI_PIT.exterior.coords), line_id="pit", resolution=30, is_barrier=True)
    if pit_resolution:
        cm.add_polygon(VORONOI_PIT, zone_id=2, resolution=pit_resolution, embed=False)
    mg = MeshGenerator(background_lc=60, conceptual_mesh=cm, verbosity=0)
    mg.generate()
    vt = VoronoiTessellator(mg, cm)
    return vt, vt.generate()


def six_layers(grids, dip):
    """Six layers dipping at ``dip`` on the Voronoi pit domain; grids is one per layer."""
    tilt = [-dip * g.x.to_numpy() for g in grids]
    botm = [level + t for level, t in zip((80.0, 60.0, 50.0, 40.0, 30.0, 20.0), tilt)]
    return disu.DisuGrid(grids, 100.0 + tilt[0], botm).build()


@pytest.mark.slow
@pytest.mark.parametrize("dip", [0.0, 0.03])
def test_voronoi_pit_refined_at_depth_cells_close(dip):
    """Independent fine Voronoi below, hanging outline vertices, straddling cells."""
    _, base = voronoi_pit()
    _, fine = voronoi_pit(12)
    d = six_layers([base] * 3 + [disu.splice(base, fine, VORONOI_PIT)] * 3, dip)
    assert d.diagnostics["vertical_offset"] == 0
    assert_cells_close(d)


def test_a_face_one_grid_already_has_is_not_added_again_between_two_others():
    # A cell left of x = 2 is the same in the layer 1 and 2 grids, a cell right of it
    # in the layer 2 and 3 grids. Comparing the layer 1 and 3 grids, the two meet as
    # if they were cells of different grids; the layer 2 grid already joins them.
    base = _boxes(4, 1.0)
    fine = _boxes(8, 0.5, start_id=500)
    east, west = disu.splice(base, fine, box(2, 0, 4, 4)), disu.splice(base, fine, box(0, 0, 2, 4))
    d = disu.DisuGrid([east, base, west], 3.0, [2.0, 1.0, 0.0]).build()
    pairs = np.sort(np.column_stack((d.links["n"], d.links["m"])), axis=1)
    assert len(np.unique(pairs, axis=0)) == len(pairs)
    layer = d.cellmap[d.cellmap.node >= 0].sort_values("node").layer.to_numpy()
    middle = (layer[d.links["n"]] == 1) & (layer[d.links["m"]] == 1)
    assert middle.sum() == 24                     # the 4 x 4 grid's own faces, once each
    for props in (d.get_gridprops_mf6_disu(), d.get_gridprops_mfusg_disu()):
        assert len(props["ja"]) == d.nodes + 2 * len(pairs)
