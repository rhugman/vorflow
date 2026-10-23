"""Independent behavioral acceptance of explicit add_line barrier controls."""

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, box

from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator
from vorflow._barriers import barrier_mirrors, barrier_policy
from vorflow.buffer import plan_line_strip
from vorflow.tessellator import _restore_barrier_cuts
from vorflow.utils import grid_barrier_report


def conceptual(
    *,
    quad_buffer=False,
    thickness=1,
    policy="preserve",
    mirrors=False,
    merge="restore",
    **kwargs,
):
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 2, 2), zone_id=1)
    cm.add_line(
        LineString([(0.9, 0), (0.9, 2)]),
        "barrier",
        resolution=1,
        is_barrier=True,
        quad_buffer=quad_buffer,
        quad_buffer_thickness=thickness,
        barrier_policy=policy,
        barrier_mirrors=mirrors,
        barrier_merge=merge,
        densify=False,
        snap_to_polygons=False,
        **kwargs,
    )
    cm.generate()
    return cm


def synthetic_grid():
    return gpd.GeoDataFrame(
        {
            "node_id": [11, 12, 13, 14],
            "x": [0.4, 1.5, 0.4, 1.5],
            "y": [0.5, 0.5, 1.5, 1.5],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)],
        index=["duplicate"] * 4,
    )


def tessellator(cm):
    mg = SimpleNamespace(
        nodes=np.empty((0, 2)), node_tags=[], zones_gdf=cm.clean_polygons
    )
    return VoronoiTessellator(mg, cm)


@pytest.mark.parametrize("quad_buffer,thickness", [(False, 1), (True, 1), (True, 2)])
@pytest.mark.parametrize(
    "policy,mirrors",
    [("preserve", False), ("preserve", True), ("cut", False), ("cut", True)],
)
@pytest.mark.parametrize("merge", ["none", "checked", "restore"])
def test_explicit_add_line_controls_survive_cleanup(
    quad_buffer, thickness, policy, mirrors, merge
):
    cm = conceptual(
        quad_buffer=quad_buffer,
        thickness=thickness,
        policy=policy,
        mirrors=mirrors,
        merge=merge,
    )
    row = cm.clean_lines.iloc[0]
    assert row.quad_buffer == quad_buffer
    assert row.quad_buffer_thickness == thickness
    assert barrier_mirrors(row) == mirrors
    assert barrier_policy(row) == policy
    assert row.barrier_merge == merge


@pytest.mark.parametrize("mirrors", [False, True])
def test_preserve_policy_keeps_the_input_partition_exactly(mirrors):
    vt = tessellator(conceptual(policy="preserve", mirrors=mirrors))
    before = synthetic_grid()
    after = vt._enforce_barriers(before)
    assert after.equals(before)
    assert vt.barrier_merge_diagnostics.empty


def test_mirror_only_really_proposes_generators_but_never_cuts_polygons():
    vt = tessellator(conceptual(policy="preserve", mirrors=True))
    before = synthetic_grid().reset_index(drop=True)
    mirrors = vt._barrier_mirror_points(
        before, before[["x", "y"]].to_numpy(), box(0, 0, 2, 2)
    )
    assert len(mirrors) > 0
    assert vt._enforce_barriers(before).equals(before)


def test_clip_only_really_cuts_but_does_not_propose_mirror_generators():
    vt = tessellator(conceptual(policy="cut", mirrors=False, merge="none"))
    before = synthetic_grid().reset_index(drop=True)
    mirrors = vt._barrier_mirror_points(
        before, before[["x", "y"]].to_numpy(), box(0, 0, 2, 2)
    )
    assert mirrors.shape == (0, 2)
    result = vt._enforce_barriers(before)
    assert len(result) == 6
    assert shapely.union_all(result.geometry).equals(box(0, 0, 2, 2))
    assert result.geometry.area.sum() == pytest.approx(4)
    assert result.node_id.is_unique


def test_restore_commits_safe_cuts_without_changing_row_identity_or_centres():
    before = synthetic_grid()
    barrier = LineString([(0.9, 0), (0.9, 2)])
    after, stats = _restore_barrier_cuts(before, barrier, [barrier])
    assert stats["restored"] == 0
    assert stats["merged"] == 2
    assert after.node_id.tolist() == before.node_id.tolist()
    np.testing.assert_array_equal(after[["x", "y"]], before[["x", "y"]])
    assert after.geometry.area.tolist() == pytest.approx([0.9, 1.1, 0.9, 1.1])
    assert shapely.union_all(after.geometry).equals(box(0, 0, 2, 2))


def test_restore_retry_undoes_an_accepted_neighbour_merge_from_restored_parent(
    monkeypatch,
):
    import vorflow.tessellator as module

    original = module._checked_fragment_union
    rejected_once = False

    def reject_first_primary(position, merged, *args):
        nonlocal rejected_once
        # The receiving unions have already been accepted when the primary
        # check runs. Reject one parent while the other can remain committed.
        if position == 0 and merged.area < 1 and not rejected_once:
            rejected_once = True
            return "synthetic_retained_parent_failure"
        return original(position, merged, *args)

    monkeypatch.setattr(module, "_checked_fragment_union", reject_first_primary)
    before = synthetic_grid()
    barrier = LineString([(0.9, 0), (0.9, 2)])
    after, stats = _restore_barrier_cuts(before, barrier, [barrier])
    assert rejected_once
    assert stats["restored_node_ids"] == [11]
    # Both the restored parent AND its former receiving neighbour revert.
    assert after.geometry.iloc[0].equals(before.geometry.iloc[0])
    assert after.geometry.iloc[1].equals(before.geometry.iloc[1])
    assert after.geometry.iloc[2].equals(box(0, 1, 0.9, 2))
    assert after.geometry.iloc[3].equals(box(0.9, 1, 2, 2))
    assert after.node_id.tolist() == before.node_id.tolist()
    np.testing.assert_array_equal(after[["x", "y"]], before[["x", "y"]])
    assert after.geometry.area.sum() == pytest.approx(4)
    assert shapely.union_all(after.geometry).area == pytest.approx(4)


@pytest.mark.parametrize("winding", [1, -1])
@pytest.mark.parametrize("start", [0, 1, 2, 3])
def test_closed_quad_band_covers_all_sides_independently_of_ring_start_and_winding(
    start, winding
):
    xy = [(16, 16), (48, 16), (48, 48), (16, 48)][::winding]
    xy = xy[start:] + xy[:start]
    line = LineString([*xy, xy[0]])
    parts = plan_line_strip(line, 8, 1, box(0, 0, 64, 64), "closed")
    assert len(parts) == 1
    part = parts[0]
    expected = box(12, 12, 52, 52).difference(box(20, 20, 44, 44))
    assert part.strip.equals(expected)
    assert part.strip.area == pytest.approx(1024)
    assert part.corners == []
    assert part.side_lines == ()
    for point in [(16, 32), (48, 32), (32, 16), (32, 48)]:
        assert part.strip.covers(Point(point))


def test_approximate_face_reports_interior_departure_from_a_bent_target():
    # Both face endpoints lie on the target. Endpoints alone would incorrectly
    # report zero displacement while the middle of the face misses it by 0.707m.
    grid = gpd.GeoDataFrame(
        {"x": [0.5, 0.5], "y": [-1, 1.5], "node_id": [7, 8]},
        geometry=[box(0, -2, 2, 0), box(0, 0, 2, 2)],
    )
    target = LineString([(0, 0), (1, 1), (2, 0)])
    barriers = gpd.GeoDataFrame(
        {"is_barrier": [True], "line_id": ["V"]}, geometry=[target]
    )
    report = grid_barrier_report(grid, barriers, approximate=True)
    face = report["approximate_faces"].iloc[0]
    actual = Point(1, 0).distance(target)
    assert face.usable
    assert face.max_offset == pytest.approx(actual)
    assert face.offset_upper_bound >= actual
    assert face.offset_upper_bound - face.max_offset < 0.01


@pytest.mark.slow
@pytest.mark.parametrize("quad_buffer,thickness", [(False, 1), (True, 1), (True, 2)])
@pytest.mark.parametrize("closed", [False, True])
def test_explicit_preserve_keeps_generated_source_layout(
    quad_buffer, thickness, closed
):
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 64, 64), zone_id=1, resolution=8)
    xy = (
        [(16, 16), (48, 16), (48, 48), (16, 48), (16, 16)]
        if closed
        else [(0, 32), (64, 32)]
    )
    cm.add_line(
        LineString(xy),
        "feature",
        resolution=8,
        is_barrier=True,
        quad_buffer=quad_buffer,
        quad_buffer_thickness=thickness,
        barrier_policy="preserve",
        barrier_mirrors=False,
        barrier_merge="restore",
        simplify_tolerance=0,
    )
    mg = MeshGenerator(background_lc=8, conceptual_mesh=cm, verbosity=0)
    mg.generate()
    vt = VoronoiTessellator(mg, cm)
    captured = []
    original = vt._enforce_barriers

    def capture(before):
        captured.append(before.copy(deep=True))
        return original(before)

    vt._enforce_barriers = capture
    final = vt.generate()
    assert vt.n_barrier_mirrors == 0
    assert vt.barrier_merge_diagnostics.empty
    assert final.node_id.tolist() == captured[0].node_id.tolist()
    np.testing.assert_array_equal(final[["x", "y"]], captured[0][["x", "y"]])
    assert all(
        a.hausdorff_distance(b) < 1e-8
        for a, b in zip(final.geometry, captured[0].geometry)
    )
    assert final.node_id.is_unique
    assert final.geometry.is_valid.all()
    assert final.geometry.area.sum() == pytest.approx(4096, abs=1e-7)
    assert shapely.union_all(final.geometry).area == pytest.approx(4096, abs=1e-7)
    if quad_buffer and thickness == 1 and closed:
        summary = vt.barrier_report()["summary"].iloc[0]
        assert summary.matched_length == pytest.approx(128)
        assert summary.crossing_cells == 0
