"""Preserve committed 3532a3 add_line defaults alongside explicit new choices.

The bent two-cell fixture was independently executed through the committed
_enforce_barriers and fragment-merging functions. Both source layouts produced
the same two polygons below, including their bent shared face. Keeping this
behavior is compatibility, not an endorsement of its numerical quality.
"""

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon, box

from vorflow import ConceptualMesh, VoronoiTessellator
from vorflow._barriers import barrier_merge, barrier_mirrors, barrier_policy
from vorflow.buffer import plan_line_strip
from vorflow.utils import build_connectivity, grid_quality_report


def old_api_case(quad=False, **options):
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 2, 1), 1)
    cm.add_line(
        LineString([(1, 0), (0.9, 0.5), (1, 1)]),
        "bend",
        1,
        is_barrier=True,
        quad_buffer=quad,
        densify=False,
        snap_to_polygons=False,
        **options,
    )
    cm.generate()
    mesh = SimpleNamespace(
        nodes=np.empty((0, 2)), node_tags=[], zones_gdf=cm.clean_polygons
    )
    cells = gpd.GeoDataFrame(
        {"node_id": [1, 2], "x": [0.5, 1.5], "y": [0.5, 0.5]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
    )
    return VoronoiTessellator(mesh, cm), cells


@pytest.mark.parametrize("quad", [False, True])
def test_existing_add_line_keeps_all_barrier_mirrors_cuts_and_legacy_merge(quad):
    vt, cells = old_api_case(quad)
    row = vt.cm.clean_lines.iloc[0]
    assert barrier_policy(row) == "cut"
    assert barrier_mirrors(row)
    assert barrier_merge(row) == "legacy"
    assert len(vt._barrier_lines()) == 1
    result = vt._enforce_barriers(cells)
    # Exact output geometry and row identity from committed 3532a3.
    assert result.node_id.tolist() == [2, 1]
    expected_right = Polygon([(2, 1), (2, 0), (1, 0), (0.9, 0.5), (1, 1)])
    expected_left = Polygon([(1, 0), (0, 0), (0, 1), (1, 1), (0.9, 0.5)])
    assert result.geometry.iloc[0].equals(expected_right)
    assert result.geometry.iloc[1].equals(expected_left)
    assert result.geometry.area.tolist() == pytest.approx([1.05, 0.95])
    np.testing.assert_array_equal(result[["x", "y"]], [[1.5, 0.5], [0.5, 0.5]])


@pytest.mark.parametrize("quad", [False, True])
@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_old_serialized_missing_values_resolve_to_historical_behavior(quad, missing):
    row = {
        "quad_buffer": quad,
        "barrier_policy": missing,
        "barrier_merge": missing,
        "barrier_mirrors": missing,
    }
    assert barrier_policy(row) == "cut"
    assert barrier_mirrors(row)
    assert barrier_merge(row) == "legacy"


@pytest.mark.parametrize("quad", [False, True])
def test_old_serialized_rows_without_new_columns_keep_bent_merge_behavior(quad):
    vt, cells = old_api_case(quad)
    vt.cm.clean_lines = vt.cm.clean_lines.drop(
        columns=["barrier_policy", "barrier_merge", "barrier_mirrors"]
    )
    assert len(vt._barrier_lines()) == 1
    result = vt._enforce_barriers(cells)
    assert result.node_id.tolist() == [2, 1]
    assert result.geometry.area.tolist() == pytest.approx([1.05, 0.95])


def test_new_checked_behavior_remains_available_as_an_explicit_option():
    vt, cells = old_api_case(True, barrier_merge="checked")
    result = vt._enforce_barriers(cells)
    assert len(result) == 3
    assert result.geometry.area.sum() == pytest.approx(2)
    assert vt.barrier_merge_diagnostics.iloc[0].rejected_candidates["bent_face"] > 0


@pytest.mark.parametrize("quad_buffer,thickness", [(False, 1), (True, 1), (True, 2)])
def test_existing_add_line_accepts_explicit_preserve_controls(quad_buffer, thickness):
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 32, 32), 1)
    cm.add_line(
        LineString([(0, 16), (32, 16)]),
        "explicit",
        4,
        is_barrier=True,
        quad_buffer=quad_buffer,
        quad_buffer_thickness=thickness,
        barrier_policy="preserve",
        barrier_mirrors=False,
        barrier_merge="restore",
        simplify_tolerance=0,
    )
    cm.generate()
    row = cm.clean_lines.iloc[0]
    assert barrier_policy(row) == "preserve"
    assert not barrier_mirrors(row)
    assert barrier_merge(row) == "restore"
    assert row.quad_buffer == quad_buffer
    assert row.quad_buffer_thickness == thickness
    assert row.simplify_tolerance == 0


def test_old_add_line_buffer_simplification_default_is_unchanged():
    vt, _ = old_api_case(True)
    assert vt.cm.raw_lines[0]["simplify_tolerance"] is None
    # Legacy None means 1.5h in buffer planning; explicit zero is opt-in.
    line = LineString([(0, 0), (16, 1), (32, 0)])
    domain = box(-8, -8, 40, 16)
    old_default = plan_line_strip(line, 4, 1, domain, "old")
    explicit_old = plan_line_strip(line, 4, 1, domain, "old", simplify_tolerance=6)
    exact = plan_line_strip(line, 4, 1, domain, "new", simplify_tolerance=0)
    assert old_default[0].strip.equals(explicit_old[0].strip)
    assert not old_default[0].strip.equals(exact[0].strip)


@pytest.mark.parametrize(
    "centres,expected_skew,expected_ortho",
    [
        ([(0.2, 0.9), (1.8, 0.9)], 0.4, 0),
        ([(0.2, 0.2), (1.8, 0.8)], 0, np.degrees(np.arctan(0.6 / 1.6))),
    ],
)
def test_new_face_report_preserves_existing_skewness_separately_from_angle(
    centres, expected_skew, expected_ortho
):
    cells = gpd.GeoDataFrame(
        {"x": [c[0] for c in centres], "y": [c[1] for c in centres]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
    )
    old = build_connectivity(cells, center="generator").iloc[0]
    new = grid_quality_report(cells)["faces"].iloc[0]
    assert new.midpoint_skew == pytest.approx(old.skewness)
    assert new.midpoint_skew == pytest.approx(expected_skew)
    assert new.ortho_error == pytest.approx(expected_ortho)
