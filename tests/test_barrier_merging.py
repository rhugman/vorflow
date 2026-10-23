"""Cutting and merging are separate operations with separate acceptance checks."""

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.affinity import translate
from shapely.geometry import LineString, Polygon, box

from vorflow import ConceptualMesh, VoronoiTessellator
from vorflow._barriers import barrier_merge
from vorflow.tessellator import _checked_fragment_union


def make_case(points=((.9, 0), (.9, 1)), mode='checked', policy='cut'):
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 2, 1), zone_id=1)
    cm.add_line(LineString(points), 'fault', resolution=1, is_barrier=True,
                barrier_policy=policy, barrier_merge=mode, densify=False, snap_to_polygons=False)
    cm.generate()
    mesh = SimpleNamespace(nodes=np.empty((0, 2)), node_tags=[], zones_gdf=cm.clean_polygons)
    cells = gpd.GeoDataFrame({'node_id': [1, 2], 'x': [.5, 1.5], 'y': [.5, .5]},
                            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)])
    return VoronoiTessellator(mesh, cm), cells


@pytest.mark.parametrize('bad', [False, True, None, 0, 'convex', []])
def test_invalid_merge_mode_is_rejected(bad):
    with pytest.raises(ValueError, match='barrier_merge'):
        make_case(mode=bad)


@pytest.mark.parametrize('mode', ['none', 'checked', 'legacy'])
def test_merge_setting_survives_cleanup(mode):
    vt, _ = make_case(mode=mode)
    assert barrier_merge(vt.cm.clean_lines.iloc[0]) == mode
    assert barrier_merge({}) == 'legacy'
    assert barrier_merge({'barrier_merge': np.nan}) == 'legacy'


def test_no_merge_cuts_but_keeps_fragment_area_identity_and_centre():
    vt, cells = make_case(mode='none')
    result = vt._enforce_barriers(cells)
    assert result.node_id.tolist() == [2, 1, 3]
    fragment = result[result.node_id == 3].iloc[0]
    assert fragment.geometry.area == pytest.approx(.1)
    assert fragment.x == pytest.approx(.95)
    assert result.geometry.area.sum() == pytest.approx(2)
    stats = vt.barrier_merge_diagnostics.iloc[0]
    assert (stats.fragments, stats.merged, stats.retained) == (1, 0, 1)
    assert stats.rejected_candidates == {}


def test_checked_accepts_a_convex_union_with_straight_opposite_side_faces():
    vt, cells = make_case()
    result = vt._enforce_barriers(cells)
    assert set(result.node_id) == {1, 2}
    assert result.set_index('node_id').geometry.area.to_dict() == pytest.approx({1: .9, 2: 1.1})
    assert result.set_index('node_id').loc[2, 'x'] == 1.5
    assert result.geometry.is_valid.all()
    stats = vt.barrier_merge_diagnostics.iloc[0]
    assert (stats.fragments, stats.merged, stats.retained) == (1, 1, 0)


def test_preserve_ignores_the_merge_setting_because_no_cuts_are_made():
    vt, cells = make_case(mode='legacy', policy='preserve')
    assert vt._enforce_barriers(cells).equals(cells)
    assert vt.barrier_merge_diagnostics.empty


def test_convex_merged_polygon_is_insufficient_when_neighbour_is_concave():
    points = [(1, 0), (.9, .5), (1, 1)]
    legacy, cells = make_case(points, mode='legacy')
    old = legacy._enforce_barriers(cells)
    merged = old[old.node_id == 2].geometry.iloc[0]
    assert merged.equals(merged.convex_hull)  # The proposed convexity-only guard would pass.
    checked, _ = make_case(points)
    result = checked._enforce_barriers(cells)
    assert len(result) == 3
    assert result[result.node_id == 2].geometry.iloc[0].equals(cells.geometry.iloc[1])
    assert result.geometry.area.sum() == pytest.approx(2)
    assert checked.barrier_merge_diagnostics.iloc[0].rejected_candidates['bent_face'] >= 1


def reason(merged, others=(), *, centre=(.5, .5), other_centres=(), fragments=()):
    geoms = np.asarray([merged, *others], dtype=object)
    centres = np.asarray([centre, *other_centres])
    return _checked_fragment_union(0, merged, geoms, centres,
                                   shapely.STRtree(geoms.copy()), fragments)


def test_nonconvex_candidate_is_refused_without_moving_vertices():
    merged = Polygon([(0, 0), (1, 0), (1, 1), (.5, .7), (0, 1)])
    original = np.asarray(merged.exterior.coords).copy()
    assert reason(merged) == 'nonconvex_union'
    np.testing.assert_array_equal(np.asarray(merged.exterior.coords), original)


def test_candidate_centre_must_remain_in_the_union():
    assert reason(box(0, 0, 1, 1), centre=(2, .5)) == 'centre_outside_union'


def test_disconnected_shared_intervals_are_refused():
    neighbour = Polygon([(1, 0), (2, 0), (2, 1), (1, 1),
                         (1, .75), (1.5, .75), (1.5, .25), (1, .25)])
    assert reason(box(0, 0, 1, 1), [neighbour], other_centres=[(1.7, .5)]) == 'disconnected_face'


def test_bent_contacts_to_other_retained_fragments_are_checked_too():
    neighbour = Polygon([(1, 0), (2, 0), (2, 2), (0, 2), (0, 1), (1, 1)])
    assert reason(box(0, 0, 1, 1)) is None
    assert reason(box(0, 0, 1, 1), fragments=[(neighbour, neighbour.area, None)]) == 'bent_face'


def test_centre_on_shared_face_is_not_a_positive_connection_distance():
    assert reason(box(0, 0, 1, 1), [box(1, 0, 2, 1)],
                  other_centres=[(1, .5)]) == 'invalid_face_centres'


def test_real_overlap_is_refused():
    assert reason(box(0, 0, 1, 1), [box(.8, 0, 2, 1)],
                  other_centres=[(1.5, .5)]) == 'overlap'


def test_corner_contact_does_not_become_a_connection():
    assert reason(box(0, 0, 1, 1), [box(1, 1, 2, 2)], other_centres=[(1.5, 1.5)]) is None


def test_nonorthogonality_is_not_claimed_to_be_fixed_by_the_merge_guard():
    assert reason(box(0, 0, 1, 1), [box(1, 0, 2, 1)],
                  centre=(.5, .2), other_centres=[(1.5, .8)]) is None


def test_guard_accepts_projected_coordinates_without_modifying_them():
    first = translate(box(0, 0, 1, 1), xoff=700000, yoff=6000000)
    second = translate(box(1, 0, 2, 1), xoff=700000, yoff=6000000)
    assert reason(first, [second], centre=(700000.5, 6000000.5),
                  other_centres=[(700001.5, 6000000.5)]) is None


def test_roundoff_sized_concavity_does_not_trigger_material_geometry_repair():
    merged = Polygon([(0, 0), (1, 0), (1, 1), (.5, 1 - 1e-15), (0, 1)])
    assert reason(merged) is None


def test_failed_geometry_check_refuses_merge_and_keeps_fragment(monkeypatch):
    import vorflow.tessellator as module

    def fail(*args):
        raise shapely.errors.GEOSException('synthetic geometry check failure')

    monkeypatch.setattr(module, '_checked_fragment_union', fail)
    vt, cells = make_case()
    result = vt._enforce_barriers(cells)
    assert len(result) == 3
    assert result.geometry.area.sum() == pytest.approx(2)
    assert vt.barrier_merge_diagnostics.iloc[0].rejected_candidates['geometry_check_failed'] >= 1
