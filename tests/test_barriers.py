"""Barrier policy and solver-independent tracking on deterministic small grids."""

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, Polygon, box

from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator
from vorflow._barriers import barrier_policy
from vorflow.utils import grid_barrier_report


def grid():
    return gpd.GeoDataFrame(
        {'x': [.5, 1.5], 'y': [.5, .5],
         'node_id': np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64)},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], index=['same', 'same'],
    )


def barriers(points=((1, 0), (1, 1)), **kwargs):
    return gpd.GeoDataFrame(
        {'line_id': ['fault'], 'is_barrier': [True], **{k: [v] for k, v in kwargs.items()}},
        geometry=[LineString(points)],
    )


def tessellator(points=((.25, 0), (.25, 1)), **options):
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 2, 1), zone_id=1)
    cm.add_line(LineString(points), 'fault', resolution=1, is_barrier=True,
                snap_to_polygons=False, densify=False, **options)
    cm.generate()
    mg = SimpleNamespace(nodes=np.empty((0, 2)), node_tags=[], zones_gdf=cm.clean_polygons)
    return VoronoiTessellator(mg, cm)


@pytest.mark.parametrize('quad,policy,expected', [
    (False, 'auto', 'cut'), (True, 'auto', 'cut'),
    (True, 'cut', 'cut'), (False, 'preserve', 'preserve'), (True, 'strict', 'strict'),
])
def test_feature_policy_survives_conceptual_cleanup(quad, policy, expected):
    vt = tessellator(quad_buffer=quad, barrier_policy=policy)
    assert barrier_policy(vt.cm.clean_lines.iloc[0]) == expected
    assert len(vt._barrier_lines()) == (expected == 'cut')


@pytest.mark.parametrize('value', [True, None, 'skip', 1, []])
def test_invalid_authored_policy_is_rejected(value):
    with pytest.raises(ValueError, match='barrier_policy'):
        tessellator(barrier_policy=value)


def test_legacy_feature_rows_keep_historical_enforcement_on_quads():
    assert barrier_policy({'quad_buffer': True, 'barrier_policy': np.nan}) == 'cut'
    assert barrier_policy({}) == 'cut'


@pytest.mark.parametrize('policy', ['preserve', 'strict'])
def test_quad_policy_does_not_cut_or_move_cells(policy):
    vt = tessellator(quad_buffer=True, barrier_policy=policy)
    before = grid().reset_index(drop=True)
    after = vt._enforce_barriers(before)
    assert after.equals(before)


def test_quad_explicit_cut_restores_splitting_and_ids():
    vt = tessellator(quad_buffer=True, barrier_policy='cut')
    before = grid().reset_index(drop=True)
    before['node_id'] = [1, 2]
    after = vt._enforce_barriers(before)
    assert len(after) == 3
    assert set(after.node_id) == {1, 2, 3}
    assert after.geometry.area.sum() == pytest.approx(2)
    assert after[after.node_id == 1].geometry.iloc[0].area == pytest.approx(.75)


def test_strict_diagnostics_are_available_after_rejection():
    vt = tessellator(quad_buffer=True, barrier_policy='strict')
    vt.final_grid = grid()
    with pytest.raises(ValueError, match='Strict barrier policy.*fault'):
        vt._check_barriers()
    assert vt.barrier_diagnostics.iloc[0].cell_ids == [0]
    assert vt.barrier_report()['crossings'].cell_id.tolist() == [0]


def test_directed_pairs_preserve_positions_duplicate_indices_and_large_node_ids():
    report = grid_barrier_report(grid(), barriers())
    face = report['faces'].iloc[0]
    assert (face.left_cell, face.right_cell) == (0, 1)
    assert (face.left_index, face.right_index) == ('same', 'same')
    assert int(face.left_node_id) == 2**64 - 2
    assert int(face.right_node_id) == 2**64 - 1
    assert face.centres_opposite
    assert face.length == pytest.approx(1)
    assert list(face.geometry.coords) == [(1, 0), (1, 1)]
    assert report['crossings'].empty and report['unmatched'].empty
    assert report['summary'].iloc[0].matched_length == pytest.approx(1)


def test_reversing_barrier_exchanges_left_and_right():
    face = grid_barrier_report(grid(), barriers([(1, 1), (1, 0)]))['faces'].iloc[0]
    assert (face.left_cell, face.right_cell) == (1, 0)
    assert list(face.geometry.coords) == [(1, 1), (1, 0)]


def test_hanging_vertices_match_multiple_pairs_without_dropping_segments():
    cells = gpd.GeoDataFrame({'x': [.5, 1.5, 1.5], 'y': [.5, .2, .7]},
                            geometry=[box(0, 0, 1, 1), box(1, 0, 2, .4), box(1, .4, 2, 1)])
    faces = grid_barrier_report(cells, barriers())['faces']
    assert list(zip(faces.left_cell, faces.right_cell)) == [(0, 1), (0, 2)]
    assert faces.length.tolist() == pytest.approx([.4, .6])


def test_internal_barrier_end_is_reported_inside_a_cell():
    report = grid_barrier_report(grid(), barriers([(.5, .25), (.5, .75)]))
    assert report['faces'].empty
    assert report['crossings'].cell_id.tolist() == [0]
    assert report['crossings'].length.sum() == pytest.approx(.5)
    assert report['unmatched'].length.sum() == pytest.approx(.5)


def test_corner_contact_and_outside_sections_do_not_create_pairs():
    report = grid_barrier_report(grid(), barriers([(0, 2), (1, 1)]))
    assert report['faces'].empty and report['crossings'].empty
    report = grid_barrier_report(grid(), barriers([(1, -.5), (1, 1.5)]))
    assert report['faces'].length.sum() == pytest.approx(1)
    assert report['unmatched'].length.sum() == pytest.approx(1)


def test_bent_shared_interface_keeps_each_barrier_segment():
    points = [(1, 0), (.9, .5), (1, 1)]
    cells = gpd.GeoDataFrame({'x': [.5, 1.5], 'y': [.5, .5]}, geometry=[
        Polygon([(0, 0), *points, (0, 1)]),
        Polygon([*points, (2, 1), (2, 0)]),
    ])
    report = grid_barrier_report(cells, barriers(points))
    assert report['faces'].segment_id.tolist() == [0, 1]
    assert report['faces'].length.sum() == pytest.approx(LineString(points).length)
    assert report['unmatched'].empty and report['crossings'].empty


def test_multipart_direction_and_identity_are_retained():
    lines = barriers()
    lines.geometry = [MultiLineString([[(1, 0), (1, .4)], [(1, 1), (1, .6)]])]
    faces = grid_barrier_report(grid(), lines)['faces']
    assert faces.part_id.tolist() == [0, 1]
    assert faces.left_cell.tolist() == [0, 1]
    assert faces.length.sum() == pytest.approx(.8)


def test_crossing_barriers_keep_separate_ids():
    cells = gpd.GeoDataFrame({'x': [.5, 1.5, .5, 1.5], 'y': [.5, .5, 1.5, 1.5]},
                            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1),
                                      box(0, 1, 1, 2), box(1, 1, 2, 2)])
    lines = gpd.GeoDataFrame({'line_id': ['vertical', 'horizontal'], 'is_barrier': [True, True]},
                            geometry=[LineString([(1, 0), (1, 2)]), LineString([(0, 1), (2, 1)])])
    report = grid_barrier_report(cells, lines)
    assert len(report['faces']) == 4
    assert report['faces'].groupby('line_id').length.sum().to_dict() == {'horizontal': 2, 'vertical': 2}
    assert report['unmatched'].empty and report['crossings'].empty


def test_duplicate_barrier_indices_still_have_separate_positions():
    lines = gpd.GeoDataFrame({'line_id': ['face', 'inside'], 'is_barrier': [True, True]},
                            geometry=[LineString([(1, 0), (1, 1)]),
                                      LineString([(.5, 0), (.5, 1)])], index=['same', 'same'])
    report = grid_barrier_report(grid(), lines)
    assert report['faces'].barrier_index.tolist() == [0]
    assert report['crossings'].barrier_index.tolist() == [1]
    assert report['summary'].crossing_cells.tolist() == [0, 1]


def test_tracking_at_projected_coordinates_keeps_orientation():
    from shapely.affinity import translate
    cells = grid()
    cells.geometry = [translate(p, xoff=700000, yoff=6000000) for p in cells.geometry]
    cells['x'] += 700000
    cells['y'] += 6000000
    lines = barriers([(700001, 6000000), (700001, 6000001)])
    report = grid_barrier_report(cells, lines)
    assert report['faces'].left_cell.tolist() == [0]
    assert report['faces'].length.sum() == pytest.approx(1)
    assert report['crossings'].empty


def test_local_polygon_sides_work_when_a_generator_is_on_the_face():
    cells = grid()
    cells.iloc[0, cells.columns.get_loc('x')] = 1
    face = grid_barrier_report(cells, barriers())['faces'].iloc[0]
    assert (face.left_cell, face.right_cell) == (0, 1)
    assert not face.centres_opposite


def test_nearby_line_requires_explicit_matching_tolerance():
    lines = barriers([(1 + 1e-5, 0), (1 + 1e-5, 1)])
    assert grid_barrier_report(grid(), lines)['faces'].empty
    assert len(grid_barrier_report(grid(), lines, tolerance=1e-4)['faces']) == 1


def test_ambiguous_duplicate_cells_do_not_become_hydraulic_pairs():
    cells = gpd.GeoDataFrame({'x': [.5, .5, 1.5], 'y': [.5, .5, .5]},
                            geometry=[box(0, 0, 1, 1), box(0, 0, 1, 1), box(1, 0, 2, 1)])
    report = grid_barrier_report(cells, barriers())
    assert report['faces'].empty
    assert report['unmatched'].length.sum() == pytest.approx(1)


@pytest.mark.parametrize('bad', [-1, np.nan, np.inf, True])
def test_tracking_rejects_invalid_tolerance(bad):
    with pytest.raises(ValueError, match='tolerance'):
        grid_barrier_report(grid(), barriers(), tolerance=bad)


def test_tracking_validates_centres_geometry_and_crs():
    with pytest.raises(ValueError, match='finite generator'):
        grid_barrier_report(grid().drop(columns='x'), barriers())
    cells = grid()
    cells['x'] = np.nan
    with pytest.raises(ValueError, match='finite generator'):
        grid_barrier_report(cells, barriers())
    cells = grid().set_crs(32618)
    with pytest.raises(ValueError, match='CRS'):
        grid_barrier_report(cells, barriers().set_crs(32619))


def test_empty_tracking_and_nonbarrier_lines_have_stable_tables():
    lines = barriers()
    lines['is_barrier'] = False
    report = grid_barrier_report(grid(), lines)
    assert all(report[k].empty for k in ('faces', 'crossings', 'unmatched', 'summary'))
    report = grid_barrier_report(grid().iloc[:0], barriers())
    assert report['faces'].empty
    assert report['summary'].unmatched_length.iloc[0] == pytest.approx(1)


def test_tracking_does_not_require_flopy(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'flopy', None)
    vt = tessellator(points=((1, 0), (1, 1)), quad_buffer=True)
    vt.final_grid = grid()
    assert len(vt.barrier_report()['faces']) == 1


def test_mixed_policies_only_force_the_requested_line():
    vt = tessellator(quad_buffer=True, barrier_policy='preserve')
    vt.cm.add_line(LineString([(1.25, 0), (1.25, 1)]), 'other', resolution=1,
                   is_barrier=True, densify=False, snap_to_polygons=False)
    vt.cm.generate()
    assert len(vt._barrier_lines()) == 1
    before = grid().reset_index(drop=True)
    before['node_id'] = [1, 2]
    after = vt._enforce_barriers(before)
    assert after[after.node_id == 1].geometry.iloc[0].equals(before.geometry.iloc[0])
    assert len(after) == 3


@pytest.mark.slow
@pytest.mark.parametrize('thickness', [1, 2])
def test_real_transfinite_barrier_preserves_layout_and_reports_crossings(thickness):
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 10, 4), zone_id=1, resolution=1)
    cm.add_line(LineString([(0, 2), (10, 2)]), 'fault', resolution=1,
                is_barrier=True, quad_buffer=True, quad_buffer_thickness=thickness,
                barrier_policy='preserve')
    mg = MeshGenerator(background_lc=1.5, conceptual_mesh=cm, verbosity=0)
    mg.generate()
    vt = VoronoiTessellator(mg, cm)
    before = []
    enforce = vt._enforce_barriers

    def capture(cells):
        before.append(cells.copy(deep=True))
        return enforce(cells)

    vt._enforce_barriers = capture
    result = vt.generate()
    assert vt.n_barrier_mirrors == 0
    assert len(result) == len(before[0])
    assert result.node_id.tolist() == before[0].node_id.tolist()
    assert result.geometry.area.sum() == pytest.approx(40)
    assert result[['x', 'y']].equals(before[0][['x', 'y']])
    report = vt.barrier_report()
    if thickness == 1:
        assert report['crossings'].empty
        assert report['faces'].length.sum() == pytest.approx(10)
    else:
        assert len(report['crossings']) > 0
        assert report['summary'].unmatched_length.iloc[0] > 8
        cm.clean_lines['barrier_policy'] = 'strict'
        with pytest.raises(ValueError, match='Strict barrier policy'):
            VoronoiTessellator(mg, cm).generate()
