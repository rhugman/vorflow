"""
Tests for line embedding in the presence of polygons.

Regression tests for the bug where lines were not properly embedded into
the triangular mesh when polygons were also present. Root causes:
  1. getEntitiesInBoundingBox requires containment, not intersection — tiny
     entity bboxes could never contain large domain surfaces.
  2. Boundary lines (created by fragment when a line crosses a polygon edge)
     were re-embedded into their own surface, corrupting the mesh.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon, LineString, Point, box

from vorflow import buffer
from vorflow.blueprint import ConceptualMesh
from vorflow._straddle import (
    KINK_INNER_MARGIN,
    _straddle_distances,
    _straddle_pair,
    _unit_tangent,
    kink_fill_points,
    kink_offset,
    plan_barrier_crossings,
    straddle_kinks,
)
from vorflow.engine import MeshGenerator
from vorflow.tessellator import VoronoiTessellator



def _nodes_near_line(nodes, line, tolerance):
    """Count mesh nodes that lie within `tolerance` of a LineString."""
    count = 0
    for x, y in nodes:
        pt = Point(x, y)
        if line.distance(pt) < tolerance:
            count += 1
    return count


class TestLineEmbeddingWithPolygons:
    """
    Core regression: a line crossing through a domain that also contains
    interior polygons must produce mesh nodes along the full line path,
    not just at polygon boundaries.
    """

    def test_line_embedded_with_polygon_present(self):
        """
        A line crossing through a domain with an interior polygon must
        produce nodes along the line — the defining symptom of the bug
        was that zero nodes appeared along interior line segments.
        """
        cm = ConceptualMesh(crs="EPSG:3857")

        # Large domain
        domain = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
        cm.add_polygon(domain, zone_id=1, resolution=10.0, z_order=0)

        # Small interior square that the line crosses through
        square = Polygon([(40, 15), (60, 15), (60, 35), (40, 35)])
        cm.add_polygon(square, zone_id=2, resolution=5.0, z_order=1)

        # Horizontal line crossing through the square
        line = LineString([(10, 25), (90, 25)])
        cm.add_line(line, line_id="crossing_line", resolution=3.0)

        clean_polys, clean_lines, clean_points = cm.generate()

        mg = MeshGenerator(background_lc=10.0, verbosity=0)
        success = mg.generate(clean_polys, clean_lines, clean_points)
        assert success, "Mesh generation failed"

        # Count nodes near the line — before the fix this was ~0
        nodes_on_line = _nodes_near_line(mg.nodes, line, tolerance=1.0)

        # With resolution=3.0 on an 80-unit line, we expect ~25+ nodes
        assert nodes_on_line >= 10, (
            f"Only {nodes_on_line} nodes found near line — "
            f"line embedding likely broken (expected >= 10)"
        )

    def test_line_embedded_without_polygon(self):
        """
        Baseline: line embedding works correctly without interior polygons.
        This should always pass — it's the control case.
        """
        cm = ConceptualMesh(crs="EPSG:3857")

        domain = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
        cm.add_polygon(domain, zone_id=1, resolution=10.0, z_order=0)

        line = LineString([(10, 25), (90, 25)])
        cm.add_line(line, line_id="simple_line", resolution=3.0)

        clean_polys, clean_lines, clean_points = cm.generate()

        mg = MeshGenerator(background_lc=10.0, verbosity=0)
        success = mg.generate(clean_polys, clean_lines, clean_points)
        assert success

        nodes_on_line = _nodes_near_line(mg.nodes, line, tolerance=1.0)
        assert nodes_on_line >= 10, (
            f"Only {nodes_on_line} nodes near line in no-polygon case"
        )

    def test_line_node_count_comparable_with_and_without_polygons(self):
        """
        The number of nodes near a line should be roughly similar whether
        or not an interior polygon exists. The original bug caused a near-
        total loss of line nodes when polygons were present.
        """
        line = LineString([(10, 25), (90, 25)])

        # Case A: without polygon
        cm_a = ConceptualMesh(crs="EPSG:3857")
        domain = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
        cm_a.add_polygon(domain, zone_id=1, resolution=10.0, z_order=0)
        cm_a.add_line(line, line_id="line_a", resolution=3.0)
        polys_a, lines_a, pts_a = cm_a.generate()
        mg_a = MeshGenerator(background_lc=10.0, verbosity=0)
        mg_a.generate(polys_a, lines_a, pts_a)
        nodes_a = _nodes_near_line(mg_a.nodes, line, tolerance=1.0)

        # Case B: with polygon
        cm_b = ConceptualMesh(crs="EPSG:3857")
        cm_b.add_polygon(domain, zone_id=1, resolution=10.0, z_order=0)
        square = Polygon([(40, 15), (60, 15), (60, 35), (40, 35)])
        cm_b.add_polygon(square, zone_id=2, resolution=5.0, z_order=1)
        cm_b.add_line(line, line_id="line_b", resolution=3.0)
        polys_b, lines_b, pts_b = cm_b.generate()
        mg_b = MeshGenerator(background_lc=10.0, verbosity=0)
        mg_b.generate(polys_b, lines_b, pts_b)
        nodes_b = _nodes_near_line(mg_b.nodes, line, tolerance=1.0)

        # Case B should have at least 50% of Case A's nodes
        # (it may have more due to the finer polygon resolution)
        ratio = nodes_b / max(nodes_a, 1)
        assert ratio >= 0.5, (
            f"With-polygon case has {nodes_b} nodes vs {nodes_a} without — "
            f"ratio {ratio:.2f} < 0.5, embedding likely broken"
        )

    def test_line_crossing_multiple_polygons(self):
        """
        A line that crosses through multiple interior polygons must still
        produce nodes along its full length.
        """
        cm = ConceptualMesh(crs="EPSG:3857")

        domain = Polygon([(0, 0), (200, 0), (200, 50), (0, 50)])
        cm.add_polygon(domain, zone_id=1, resolution=15.0, z_order=0)

        # Three squares along the line path
        for i, x_start in enumerate([30, 80, 140]):
            sq = Polygon([
                (x_start, 15), (x_start + 20, 15),
                (x_start + 20, 35), (x_start, 35)
            ])
            cm.add_polygon(sq, zone_id=10 + i, resolution=5.0, z_order=1)

        line = LineString([(10, 25), (190, 25)])
        cm.add_line(line, line_id="multi_cross", resolution=5.0)

        clean_polys, clean_lines, clean_points = cm.generate()

        mg = MeshGenerator(background_lc=15.0, verbosity=0)
        success = mg.generate(clean_polys, clean_lines, clean_points)
        assert success

        nodes_on_line = _nodes_near_line(mg.nodes, line, tolerance=1.5)
        # 180-unit line at resolution 5 → ~36 segments → ~30+ nodes expected
        assert nodes_on_line >= 15, (
            f"Only {nodes_on_line} nodes on line crossing 3 polygons"
        )

    def test_mesh_area_conservation_with_embedded_line(self):
        """
        Total mesh area must match the domain area, ensuring no
        garbage triangles extend outside the domain.
        """
        cm = ConceptualMesh(crs="EPSG:3857")

        domain = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
        cm.add_polygon(domain, zone_id=1, resolution=10.0, z_order=0)

        square = Polygon([(40, 15), (60, 15), (60, 35), (40, 35)])
        cm.add_polygon(square, zone_id=2, resolution=5.0, z_order=1)

        line = LineString([(10, 25), (90, 25)])
        cm.add_line(line, line_id="area_test", resolution=3.0)

        clean_polys, clean_lines, clean_points = cm.generate()

        mg = MeshGenerator(background_lc=10.0, verbosity=0)
        success = mg.generate(clean_polys, clean_lines, clean_points)
        assert success

        vt = VoronoiTessellator(mg, cm, clip_to_boundary=True)
        grid = vt.generate()

        total_area = grid.geometry.area.sum()
        expected_area = domain.area  # 5000
        assert pytest.approx(total_area, rel=0.02) == expected_area, (
            f"Area mismatch: {total_area:.1f} vs {expected_area:.1f} — "
            f"possible out-of-domain triangles"
        )


class TestUnitTangent:
    """Regression tests for the straddle-point tangent probe.

    The old implementation used fixed absolute steps (0.01/0.001 CRS units),
    which blended directions across corners of short lines and degenerated
    on lines shorter than the step.
    """

    def test_straight_line_tangent(self):
        line = LineString([(0, 0), (10, 0)])
        probe = line.length * 1e-4
        for d in (0.0, 5.0, 10.0):
            dx, dy = _unit_tangent(line, d, probe)
            assert (dx, dy) == pytest.approx((1.0, 0.0), abs=1e-9)

    def test_bent_line_respects_local_direction(self):
        # L-shape with legs much shorter than the old 0.01 fixed probe:
        # tangent at the start must follow the first leg, at the end the
        # second leg -- not the corner-cutting chord.
        line = LineString([(0, 0), (0.005, 0), (0.005, 0.005)])
        probe = line.length * 1e-4
        dx, dy = _unit_tangent(line, 0.0, probe)
        assert (dx, dy) == pytest.approx((1.0, 0.0), abs=1e-6)
        dx, dy = _unit_tangent(line, line.length, probe)
        assert (dx, dy) == pytest.approx((0.0, 1.0), abs=1e-6)

    def test_tangent_is_unit_length_everywhere(self):
        line = LineString([(0, 0), (3, 4), (10, 4)])
        probe = line.length * 1e-4
        for frac in (0.0, 0.2, 0.5, 0.8, 1.0):
            dx, dy = _unit_tangent(line, line.length * frac, probe)
            assert dx * dx + dy * dy == pytest.approx(1.0, abs=1e-12)

    def test_degenerate_line_returns_unit_vector(self):
        line = LineString([(2, 2), (2, 2)])
        dx, dy = _unit_tangent(line, 0.0, 1e-12)
        assert dx * dx + dy * dy == pytest.approx(1.0)


class TestStraddleDistances:
    """Straddle pairs stay inside the domain where a barrier ends on its boundary."""

    LC, EPS = 2.0, 0.4

    def _pairs(self, line, domain):
        probe = line.length * 1e-4
        distances = _straddle_distances(line, self.LC, self.EPS, probe, domain, self.EPS * 1e-6)
        return distances, [_straddle_pair(line, d, self.EPS, probe) for d in distances]

    def test_perpendicular_end_pairs_are_unchanged(self):
        line = LineString([(10, 0), (10, 10)])
        distances, _ = self._pairs(line, box(0, 0, 20, 10))
        assert distances == pytest.approx(np.linspace(0, 10, 6))

    def test_without_domain_pairs_include_both_endpoints(self):
        line = LineString([(10, 0), (4, 10)])
        distances, _ = self._pairs(line, None)
        assert distances[0] == 0.0 and distances[-1] == pytest.approx(line.length)

    def test_oblique_end_pairs_slide_until_both_points_are_inside(self):
        domain = box(0, 0, 20, 10)
        line = LineString([(12.887, 0), (7.113, 10)])  # 60 degrees to the boundary
        distances, pairs = self._pairs(line, domain)
        # The slide is epsilon / tan(60 deg) at each end.
        slide = self.EPS / np.tan(np.radians(60))
        assert distances[0] == pytest.approx(slide, rel=1e-3)
        assert line.length - distances[-1] == pytest.approx(slide, rel=1e-3)
        for pair in (pairs[0], pairs[-1]):
            points = [Point(xy) for xy in pair]
            assert all(domain.buffer(1e-9).covers(p) for p in points)
            # One point lands on the boundary; the pair's bisector stays on the line.
            assert min(domain.exterior.distance(p) for p in points) < 1e-9
            assert line.distance(Point(np.mean(pair, axis=0))) < 1e-9

    def test_anchors_are_kept_and_each_gap_is_split_evenly(self):
        line = LineString([(10, 0), (10, 10)])
        probe = line.length * 1e-4
        distances = _straddle_distances(line, self.LC, self.EPS, probe, box(0, 0, 20, 10),
                                        self.EPS * 1e-6, anchors=[3.3])
        # 3.3 in two steps, the remaining 6.7 in four.
        assert distances == pytest.approx([0, 1.65, 3.3, 4.975, 6.65, 8.325, 10])


def _crossing_plan(line, barrier=LineString([(50, 0), (50, 100)])):
    """plan_barrier_crossings for a lc-2 line and a lc-2 barrier in a 100 x 100 domain."""
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 100, 100), zone_id=1)
    cm.add_line(barrier, line_id="barrier", resolution=2.0, is_barrier=True)
    cm.add_line(line, line_id="line", resolution=2.0)
    polys, lines, _ = cm.generate()
    corridors = buffer.protected_corridors(polys, lines, 10.0)
    plan = plan_barrier_crossings(
        lines, 10.0, buffer.domain_union_geometry(polys), buffer.barrier_zone(corridors),
        lambda b_idx: None,
    )
    return plan, lines


class TestBarrierCrossingPlan:
    """A line crossing a barrier gets a straddle pair at the crossing and ends on it."""

    def test_perpendicular_crossing_line_ends_on_the_anchor_pair(self):
        plan, _ = _crossing_plan(LineString([(0, 50.7), (100, 50.7)]))
        assert list(plan.fixed) == [0]
        (distance, pair), = plan.fixed[0].items()
        assert distance == pytest.approx(50.7)
        assert sorted(pair) == [pytest.approx((49.6, 50.7)), pytest.approx((50.4, 50.7))]
        ends = sorted(xy for part in plan.line_parts[1] for xy in (part.coords[0], part.coords[-1])
                      if abs(xy[0] - 50) < 1)
        assert ends == sorted(pair)
        # The next node along the line is one cell away from the pair point.
        for part in plan.line_parts[1]:
            coords = list(part.coords)
            end, nxt = (coords[-1], coords[-2]) if abs(coords[-1][0] - 50) < 1 else (coords[0], coords[1])
            assert np.hypot(end[0] - nxt[0], end[1] - nxt[1]) == pytest.approx(2.0)

    def test_oblique_crossing_mirrors_the_line_nodes_near_the_barrier(self):
        angle = np.radians(30)
        direction = np.array([np.sin(angle), np.cos(angle)])
        plan, _ = _crossing_plan(LineString([(50, 50) - 60 * direction, (50, 50) + 60 * direction]))
        fixed = plan.fixed[0]
        anchor = fixed[min(fixed, key=lambda d: abs(d - 50))]
        # The anchor pair stays perpendicular to the barrier.
        assert sorted(anchor) == [pytest.approx((49.6, 50)), pytest.approx((50.4, 50))]
        mirrored = [pair for d, pair in fixed.items() if abs(d - 50) > 1e-9]
        assert len(mirrored) >= 2
        line_nodes = {xy for part in plan.line_parts[1] for xy in part.coords}
        for (x0, y0), (x1, y1) in mirrored:
            # Each is a line node and its reflection across the barrier.
            assert y0 == pytest.approx(y1) and x0 + x1 == pytest.approx(100.0)
            assert ((x0, y0) in line_nodes) != ((x1, y1) in line_nodes)

    def test_crossing_near_a_barrier_end_is_trimmed_as_before(self):
        plan, _ = _crossing_plan(LineString([(0, 20.5), (100, 20.5)]),
                                 barrier=LineString([(50, 20), (50, 100)]))
        assert plan.fixed == {} and plan.line_parts == {}

    def test_lines_without_crossings_are_left_out_of_the_plan(self):
        plan, _ = _crossing_plan(LineString([(0, 50), (40, 50)]))
        assert plan.fixed == {} and plan.line_parts == {}


def _densified(coords, step=1.0):
    """``coords`` as a LineString with extra collinear vertices about ``step`` apart."""
    return LineString(LineString(coords).segmentize(step).coords)


def test_straddle_kinks_finds_turns_and_ignores_collinear_vertices():
    line = _densified([(0, 0), (20, 0), (20, 20)])
    kinks = straddle_kinks(line, lc=4.0, epsilon=0.8)
    assert len(kinks) == 1
    d, h, slope = kinks[0]
    assert d == pytest.approx(20.0) and h == pytest.approx(2.0)
    assert slope == pytest.approx(np.tan(np.radians(45)) / KINK_INNER_MARGIN)


def test_straddle_kinks_drops_kinks_with_a_short_leg():
    # A 1 m jog in a 4 m-resolution line keeps the evenly spaced pairs.
    line = LineString([(0, 0), (20, 0), (20, 1), (40, 1)])
    assert straddle_kinks(line, lc=4.0, epsilon=0.8) == []


def test_kink_pairs_and_fill_points_are_equidistant_from_the_vertex():
    # The vertex must be a Voronoi vertex of the barrier pairs: its nearest
    # generators are the two kink pairs and the fill points, all at one radius.
    line = _densified([(0, 0), (20, 0), (30, 25)])
    lc, eps = 4.0, 0.8
    kinks = straddle_kinks(line, lc, eps)
    probe = 1e-4 * line.length
    distances = _straddle_distances(line, lc, eps, probe, None, eps * 1e-6, kinks=kinks)
    (d, h, _), = kinks
    assert not any(d - h < x < d + h for x in distances)
    assert np.diff(distances).max() <= lc + 1e-9
    pairs = [xy for x in distances for xy in _straddle_pair(line, x, kink_offset(x, kinks, eps), probe)]
    fill = kink_fill_points(line, kinks, eps)
    assert fill
    radius = np.hypot(*(np.array(pairs + fill) - (20, 0)).T)
    nearest = np.sort(radius)[:4 + len(fill)]
    np.testing.assert_allclose(nearest, nearest[0], rtol=1e-9)
    assert np.sort(radius)[4 + len(fill)] > nearest[0] * 1.01


def test_sharp_kink_pairs_use_a_smaller_offset():
    # Interior angle 20 degrees: full-offset inner points of the two legs
    # would be nearer each other than their own partners.
    end = 20 * np.array([np.cos(np.radians(20)), np.sin(np.radians(20))])
    line = _densified([(20, 0), (0, 0), tuple(end)])
    kinks = straddle_kinks(line, lc=4.0, epsilon=0.8)
    (d, h, _), = kinks
    expected = h * np.tan(np.radians(10)) / KINK_INNER_MARGIN
    assert kink_offset(d - h, kinks, 0.8) == pytest.approx(expected)
    assert kink_offset(d + h, kinks, 0.8) == pytest.approx(expected)
    assert kink_offset(d + h + 4.0, kinks, 0.8) == 0.8
