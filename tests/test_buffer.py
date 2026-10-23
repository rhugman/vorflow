import warnings

import gmsh
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import split

from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator
from vorflow.buffer import plan_line_strip, plan_polygon_band, plan_quad_buffers

pytestmark = pytest.mark.slow  # gmsh-heavy end-to-end tests


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("start", [0, 1, 2, 3])
def test_closed_line_buffer_is_complete_annulus(reverse, start):
    # An exterior offset_curve of a closed line can omit two sides at its
    # start point. A closed feature must use polygon buffering, without fake
    # end caps, so every side receives the requested footprint.
    corners = [(134, 134), (166, 134), (166, 166), (134, 166)]
    if reverse:
        corners.reverse()
    corners = corners[start:] + corners[:start]
    line = LineString(corners + [corners[0]])
    parts = plan_line_strip(line, 8.0, 1, box(0, 0, 300, 300), "loop")

    assert len(parts) == 1
    part = parts[0]
    expected = box(130, 130, 170, 170).difference(box(138, 138, 162, 162))
    assert part.strip.is_valid
    assert part.strip.symmetric_difference(expected).area == pytest.approx(0.0)
    assert part.strip.area == pytest.approx(1024.0)
    assert len(part.strip.interiors) == 1
    assert part.corners == []
    assert part.side_lines == ()


@pytest.mark.parametrize("thickness", [1, 2])
def test_closed_line_buffer_matches_polygon_band(thickness):
    # This also exercises a non-rectangular loop and preserves the domain's
    # interior: converting the outline to a band does not create a new zone.
    polygon = Polygon([(20, 20), (84, 20), (84, 84), (52, 100), (20, 84)])
    line = LineString(polygon.exterior.coords)
    part = plan_line_strip(line, 8.0, thickness, None, "loop")[0]
    expected = plan_polygon_band(polygon, 8.0, thickness, "polygon")
    assert part.strip.symmetric_difference(expected).area == pytest.approx(0.0)


def test_closed_line_buffer_rejects_self_intersection():
    line = LineString([(0, 0), (32, 32), (0, 32), (32, 0), (0, 0)])
    with pytest.warns(UserWarning, match="must form a simple polygon ring"):
        assert plan_line_strip(line, 8.0, 1, None, "bow-tie") == []


def test_explicit_zero_buffer_simplification_preserves_gentle_turn():
    line = LineString([(0, 0), (16, 2), (32, 0)])
    exact = plan_line_strip(line, 8, 1, None, "gentle", simplify_tolerance=0)[0]
    legacy = plan_line_strip(line, 8, 1, None, "gentle")[0]
    explicit_legacy = plan_line_strip(line, 8, 1, None, "gentle", simplify_tolerance=12)[0]
    expected = line.buffer(4, cap_style=2, join_style=2, mitre_limit=5)
    assert exact.strip.symmetric_difference(expected).area == pytest.approx(0, abs=1e-12)
    assert legacy.strip.symmetric_difference(exact.strip).area > 0
    assert legacy.strip.equals(explicit_legacy.strip)


def test_explicit_zero_closed_buffer_preserves_rounded_boundary():
    polygon = Point(40, 40).buffer(32, quad_segs=8)
    line = LineString(polygon.exterior.coords)
    exact = plan_line_strip(line, 8, 1, None, "rounded", simplify_tolerance=0)[0]
    expected = polygon.buffer(4, join_style=2).difference(polygon.buffer(-4, join_style=2))
    assert exact.strip.symmetric_difference(expected).area == pytest.approx(0, abs=1e-10)
    legacy = plan_line_strip(line, 8, 1, None, "rounded")[0]
    assert legacy.strip.symmetric_difference(exact.strip).area > 1


@pytest.mark.parametrize("closed", [False, True])
def test_quad_buffer_planning_propagates_feature_simplification_zero(closed):
    line = (LineString(Point(40, 40).buffer(32, quad_segs=8).exterior.coords)
            if closed else LineString([(8, 24), (40, 28), (72, 24)]))
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(-8, -8, 88, 88), zone_id=1, resolution=8)
    cm.add_line(line, line_id="exact", resolution=8, quad_buffer=True,
                simplify_tolerance=0, densify=False)
    polygons, lines, _ = cm.generate()
    plans = plan_quad_buffers(polygons, lines, 8, box(-8, -8, 88, 88))
    expected = plan_line_strip(line, 8, 1, None, "expected", simplify_tolerance=0)[0].strip
    assert plans[('line', 0)].footprint.symmetric_difference(expected).area == pytest.approx(0)


@pytest.mark.parametrize("reverse", [False, True])
def test_closed_line_quad_buffer_mesh_covers_every_side_without_cutting(reverse):
    corners = [(16, 16), (48, 16), (48, 48), (16, 48)]
    if reverse:
        corners.reverse()
    loop = LineString(corners + [corners[0]])
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 64, 64), zone_id="domain", resolution=8.0)
    cm.add_line(
        loop,
        line_id="closed-barrier",
        resolution=8.0,
        is_barrier=True,
        barrier_policy="preserve",
        quad_buffer=True,
    )
    mesher = MeshGenerator(
        background_lc=8.0, verbosity=0, smoothing_steps=0, optimization_cycles=0
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert mesher.generate(*cm.generate())
    assert not [w for w in caught if "transfinite structure" in str(w.message)]
    elements = mesher.get_element_grid()
    quads = mesher.get_element_grid("quads")
    expected = box(12, 12, 52, 52).difference(box(20, 20, 44, 44))
    quad_footprint = shapely.union_all(quads.geometry.to_numpy())
    assert quad_footprint.symmetric_difference(expected).area == pytest.approx(0.0)
    assert elements.geometry.area.sum() == pytest.approx(64.0**2)
    assert elements["zone_id"].eq("domain").all()
    assert len(quads) == 16

    tessellator = VoronoiTessellator(mesher, cm, clip_to_boundary=True)
    grid = tessellator.generate()
    report = tessellator.barrier_report()
    assert report["crossings"].empty
    assert report["unmatched"].empty
    assert grid.geometry.area.sum() == pytest.approx(64.0**2)



def _capture_crossings(mesher, monkeypatch):
    """Record the crossings generate() hands to _setup_fields."""
    captured = []
    setup_fields = mesher._setup_fields

    def spy(*args, crossings=(), **kwargs):
        captured.extend(crossings)
        return setup_fields(*args, crossings=crossings, **kwargs)

    monkeypatch.setattr(mesher, "_setup_fields", spy)
    return captured


def _generate_line_buffer_mesh(*, thickness=1, add_crossing_line=False, return_context=False,
                               barrier_policy='auto'):
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(
        box(0, 0, 10, 4),
        zone_id="domain",
        resolution=1.0,
        densify=True,
    )
    cm.add_line(
        LineString([(0, 2), (10, 2)]),
        line_id="buffered-line",
        resolution=1.0,
        is_barrier=True,
        barrier_policy=barrier_policy,
        quad_buffer=True,
        quad_buffer_thickness=thickness,
        dist_min=0.5,
        dist_max=2.0,
    )
    if add_crossing_line:
        cm.add_line(
            LineString([(5, 0), (5, 4)]),
            line_id="crossing-line",
            resolution=0.5,
            is_barrier=False,
            dist_min=0.5,
            dist_max=1.5,
        )

    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(
        background_lc=1.5,
        verbosity=0,
        smoothing_steps=0,
        optimization_cycles=0,
    )
    assert mesher.generate(clean_polys, clean_lines, clean_points)
    # A successful run must actually produce mesh nodes covering the domain.
    assert mesher.nodes is not None and len(mesher.nodes) > 0
    element_area = mesher.get_element_grid().geometry.area.sum()
    domain_area = clean_polys.geometry.area.sum()
    assert element_area == pytest.approx(domain_area, rel=1e-2)
    if return_context:
        return mesher, cm
    return mesher


def test_line_structured_quad_buffer_produces_mixed_element_grid():
    mesher = _generate_line_buffer_mesh(thickness=1)
    element_grid = mesher.get_element_grid()
    quads = mesher.get_element_grid("quads")
    triangles = mesher.get_element_grid("triangles")

    assert not element_grid.empty
    assert not quads.empty
    assert not triangles.empty
    assert len(quads) + len(triangles) == len(element_grid)
    assert quads["is_quad"].all()
    assert quads["centroid_y"].between(1.45, 2.55).all()
    # Quads inside the domain must receive a valid zone assignment.
    assert quads["zone_id"].notna().all()
    assert quads["zone_id"].eq("domain").all()
    # A thickness-1 strip on a length-10 line at lc=1 is a single structured
    # row of ~10 quads, not an arbitrary recombination.
    assert 7 <= len(quads) <= 14


def test_line_structured_quad_buffer_thickness_controls_quad_band_area():
    thin = _generate_line_buffer_mesh(thickness=1).get_element_grid("quads")
    thick = _generate_line_buffer_mesh(thickness=2).get_element_grid("quads")

    assert not thin.empty
    assert not thick.empty
    assert thick.geometry.area.sum() > thin.geometry.area.sum()


def test_structured_quad_buffer_protects_crossing_feature_corridor():
    baseline_quads = _generate_line_buffer_mesh(thickness=1).get_element_grid("quads")
    if gmsh.is_initialized():
        gmsh.finalize()

    mesher = _generate_line_buffer_mesh(thickness=1, add_crossing_line=True)
    quads = mesher.get_element_grid("quads")
    triangles = mesher.get_element_grid("triangles")

    assert not quads.empty
    # The crossing standard line is trimmed out of the strip corridor, so the
    # strip interior stays quad-dominated. (The crossing line's size field may
    # still grade into the strip and leave a couple of split triangles, but no
    # feature nodes are injected.)
    strip_interior = LineString([(0, 2), (10, 2)]).buffer(0.4, cap_style=2)
    strip_quads = quads[quads.geometry.centroid.within(strip_interior)]
    strip_triangles = triangles[triangles.geometry.centroid.within(strip_interior)]
    assert len(strip_quads) >= 8
    assert len(strip_triangles) <= max(2, len(strip_quads) // 4)
    # The strip structure is preserved despite the crossing feature.
    assert abs(len(quads) - len(baseline_quads)) <= max(2, int(0.25 * len(baseline_quads)))


def test_curved_line_buffer_produces_structured_quads():
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 10, 6), zone_id="domain", resolution=1.0, densify=True)
    line = LineString([(1, 2), (5, 4), (9, 2)])
    cm.add_line(
        line,
        line_id="curved",
        resolution=1.0,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
        dist_min=0.5,
        dist_max=2.0,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert mesher.generate(clean_polys, clean_lines, clean_points)
    degradations = [
        w for w in caught if "transfinite" in str(w.message) or "recombine-only" in str(w.message)
    ]
    assert not degradations, [str(w.message) for w in degradations]

    quads = mesher.get_element_grid("quads")
    assert not quads.empty
    assert quads["is_quad"].all()
    # A full structured row of quads hugs the whole curved line.
    assert quads.geometry.centroid.apply(lambda p: line.distance(p) < 0.75).all()
    assert len(quads) >= round(line.length) - 3


def test_crossing_quad_buffers_are_protected_and_warn():
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 10, 4), zone_id="domain", resolution=1.0, densify=True)
    cm.add_line(
        LineString([(0, 2), (10, 2)]),
        line_id="horizontal",
        resolution=1.0,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    cm.add_line(
        LineString([(5, 0), (5, 4)]),
        line_id="vertical",
        resolution=1.0,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)

    with pytest.warns(UserWarning, match="crosses a higher-priority protected feature"):
        assert mesher.generate(clean_polys, clean_lines, clean_points)

    quads = mesher.get_element_grid("quads")
    assert not quads.empty
    # Equal z_order/lc/thickness: the horizontal line was added first, so it
    # wins the tie-break and stays continuous THROUGH the crossing (one-sided
    # trimming) while the vertical one is trimmed there.
    winner_at_crossing = quads[
        (quads["centroid_y"].sub(2).abs() < 0.5) & (quads["centroid_x"].sub(5).abs() < 0.6)
    ]
    assert not winner_at_crossing.empty
    horizontal_arm = quads[
        (quads["centroid_y"].sub(2).abs() < 0.5) & (quads["centroid_x"].sub(5).abs() > 1)
    ]
    vertical_arm = quads[
        (quads["centroid_x"].sub(5).abs() < 0.5) & (quads["centroid_y"].sub(2).abs() > 1)
    ]
    assert not horizontal_arm.empty
    assert not vertical_arm.empty


def test_transfinite_survives_fragmentation():
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 10, 4), zone_id="domain", resolution=1.0, densify=True)
    line = LineString([(1, 2), (9, 2)])
    cm.add_line(
        line,
        line_id="buffered",
        resolution=1.0,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    cm.add_line(
        LineString([(5, 0), (5, 4)]),
        line_id="crossing-standard",
        resolution=0.5,
        is_barrier=False,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert mesher.generate(clean_polys, clean_lines, clean_points)
    degradations = [
        w for w in caught if "transfinite" in str(w.message) or "recombine-only" in str(w.message)
    ]
    assert not degradations, [str(w.message) for w in degradations]

    quads = mesher.get_element_grid("quads")
    assert not quads.empty
    # Structured rows: near-uniform quad areas and a count matching length/lc.
    areas = quads.geometry.area
    assert areas.std() / areas.mean() < 0.35
    assert round(line.length) - 2 <= len(quads) <= round(line.length) + 4


def _straddling_cell_count(grid, line, sliver_fraction=1e-6):
    """Count cells the line splits into at least two non-sliver pieces."""
    count = 0
    for cell in grid.geometry[grid.geometry.intersects(line)]:
        pieces = [p for p in split(cell, line).geoms if p.area > sliver_fraction * cell.area]
        count += len(pieces) > 1
    return count


def test_structured_quad_buffer_explicit_cut_only_splits_straddling_cells():
    mesher, cm = _generate_line_buffer_mesh(thickness=1, return_context=True, barrier_policy='cut')
    line = cm.clean_lines.iloc[0].geometry

    tessellator = VoronoiTessellator(mesher, cm, clip_to_boundary=True)
    grid = tessellator.generate()

    # Buffer cells already have faces on the line and are not re-cut. Only
    # the end caps, where the strip meets the domain boundary and nodes land
    # on the line, straddle it and are split.
    fragments = grid[~grid["node_id"].isin(mesher.node_tags)]
    assert set(mesher.node_tags).issubset(set(grid["node_id"]))
    assert grid["node_id"].is_unique
    assert len(fragments) <= 4
    assert (fragments.geometry.distance(line.boundary) < 1.0).all()
    assert _straddling_cell_count(grid, line) == 0


def test_structured_quad_buffer_thickness2_explicit_cut_is_enforced():
    """Explicit cutting retains the legacy enforcement on a centreline node row."""
    mesher, cm = _generate_line_buffer_mesh(thickness=2, return_context=True, barrier_policy='cut')
    line = cm.clean_lines.iloc[0].geometry

    grid = VoronoiTessellator(mesher, cm, clip_to_boundary=True).generate()

    # The node row on the line gives ~one straddling cell per lc of line;
    # each is split in two along the barrier.
    fragments = grid[~grid["node_id"].isin(mesher.node_tags)]
    assert len(fragments) >= 0.5 * line.length
    assert (fragments.geometry.distance(line) < 1e-9).all()
    assert _straddling_cell_count(grid, line) == 0
    assert grid["node_id"].is_unique
    assert grid.geometry.area.sum() == pytest.approx(40.0)


def test_polygon_structured_quad_buffer_produces_quads():
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(
        box(0, 0, 10, 6),
        zone_id="domain",
        resolution=1.5,
        densify=True,
    )
    cm.add_polygon(
        Polygon([(3, 2), (7, 2), (7, 4), (3, 4)]),
        zone_id="inner",
        resolution=1.0,
        z_order=1,
        densify=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()

    mesher = MeshGenerator(
        background_lc=1.5,
        verbosity=0,
        smoothing_steps=0,
        optimization_cycles=0,
    )
    # Polygon bands are recombine-only by design (annulus), which must not be
    # reported as a degradation.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert mesher.generate(clean_polys, clean_lines, clean_points)
    degradations = [
        w for w in caught if "transfinite" in str(w.message) or "recombine-only" in str(w.message)
    ]
    assert not degradations, [str(w.message) for w in degradations]

    quads = mesher.get_element_grid("quads")
    assert not quads.empty


def test_node_extraction_covers_all_regions_split_by_buffer_strip():
    # A full-width strip crossing an inner zone splits the domain into pieces
    # OCC's fragment map can drop (regression: all generators above the strip
    # were lost, producing giant merged-looking Voronoi cells).
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 7), zone_id="domain", resolution=1.5, densify=True)
    cm.add_polygon(
        Polygon([(4, 2), (8, 2), (8, 5), (4, 5)]),
        zone_id="inner-zone",
        resolution=1.0,
        z_order=1,
        densify=True,
    )
    cm.add_line(
        LineString([(0, 3.5), (12, 3.5)]),
        line_id="fault-buffer",
        resolution=1.0,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    assert mesher.generate(clean_polys, clean_lines, clean_points)

    element_grid = mesher.get_element_grid()
    corner_xy = {
        (round(x, 6), round(y, 6))
        for geom in element_grid.geometry
        for x, y in geom.exterior.coords[:-1]
    }
    node_xy = {(round(float(x), 6), round(float(y), 6)) for x, y in mesher.nodes}
    missing = corner_xy - node_xy
    assert not missing, f"{len(missing)} gmsh mesh nodes missing from Voronoi generators"


def test_strip_crossing_embedded_zone_stays_transfinite():
    # Regression: a strip split by an embedded zone boundary degraded to
    # recombined triangles. Pieces are now re-cornered from the offset side
    # lines and kept transfinite, and densified ring vertices falling inside
    # the strip are pushed onto its boundary so the caps stay single curves.
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 7), zone_id="domain", resolution=1.5, densify=True)
    cm.add_polygon(
        Polygon([(4, 2), (8, 2), (8, 5), (4, 5)]),
        zone_id="inner",
        resolution=1.0,
        z_order=1,
        densify=True,
    )
    cm.add_line(
        LineString([(0, 3.5), (12, 3.5)]),
        line_id="fault",
        resolution=1.0,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert mesher.generate(clean_polys, clean_lines, clean_points)
    degradations = [
        w for w in caught if "transfinite" in str(w.message) or "recombine-only" in str(w.message)
    ]
    assert not degradations, [str(w.message) for w in degradations]

    quads = mesher.get_element_grid("quads")
    triangles = mesher.get_element_grid("triangles")
    strip = LineString([(0, 3.5), (12, 3.5)]).buffer(0.4, cap_style=2)
    assert triangles[triangles.geometry.centroid.within(strip)].empty
    # One structured row of ~12 quads along the 12-length fault.
    assert 10 <= len(quads) <= 16


def test_size_field_attached_to_quad_buffer_line_refines_halo():
    # Regression: distance-growth fields on quad_buffer lines were silently
    # dropped because the strip surfaces were not listed as embedded surfaces.
    # An explicit geometric-growth field must still grade the halo outward from the
    # strip (now also the default behavior -- see test_default_field_*).
    from vorflow.fields import GeometricGrowthField

    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 20, 12), zone_id="domain", resolution=4.0, densify=True)
    cm.add_line(
        LineString([(4, 6), (16, 6)]),
        line_id="buffered",
        resolution=0.5,
        quad_buffer=True,
        quad_buffer_thickness=1,
        fields=[GeometricGrowthField(growth_factor=1.2)],
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=4.0, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    assert mesher.generate(clean_polys, clean_lines, clean_points)

    grid = mesher.get_element_grid()
    cent = grid.geometry.centroid
    # The halo grades element size outward from the strip: elements just off the
    # strip are smaller than those out near the coarse domain edge. If the field
    # were dropped (the regression), sizes would be uniform.
    near = grid[cent.y.sub(6).abs().between(1.0, 3.0)]
    far = grid[cent.y.sub(6).abs() > 4.5]
    assert not near.empty and not far.empty
    assert near.geometry.area.mean() < far.geometry.area.mean()


def _generate_polygon_buffer_voronoi(thickness):
    zone = Polygon([(3, 3), (9, 3), (9, 7), (3, 7)])
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 10), zone_id="domain", resolution=1.0, densify=False)
    cm.add_polygon(
        zone,
        zone_id="inner",
        resolution=1.0,
        z_order=1,
        densify=True,
        quad_buffer=True,
        quad_buffer_thickness=thickness,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.0, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    assert mesher.generate(clean_polys, clean_lines, clean_points)
    grid = VoronoiTessellator(mesher, cm, clip_to_boundary=True).generate()
    return zone, grid


def test_polygon_quad_buffer_thickness1_faces_follow_boundary():
    # The zone outline never becomes mesh edges: a single quad row straddles it
    # (nodes at +/- lc/2), so the Voronoi faces trace the shapefile shape and
    # no cell straddles the outline.
    zone, grid = _generate_polygon_buffer_voronoi(1)
    centroids = grid.geometry.centroid
    assert (centroids.distance(zone.exterior) < 0.05).sum() == 0
    inside_frac = grid.geometry.intersection(zone).area / grid.geometry.area
    crossing = ((inside_frac > 0.03) & (inside_frac < 0.97)).sum()
    assert crossing == 0


def test_polygon_quad_buffer_thickness2_centers_cells_on_boundary():
    # Two quad rows put a node row on the outline, giving a ring of ~square
    # (lc x lc) Voronoi cells centered on the shape.
    zone, grid = _generate_polygon_buffer_voronoi(2)
    centroids = grid.geometry.centroid
    ring = grid[centroids.distance(zone.exterior) < 0.05]
    perimeter = zone.exterior.length
    assert 0.65 * perimeter <= len(ring) <= 1.35 * perimeter
    assert abs(ring.geometry.area.mean() - 1.0) < 0.15


def test_line_quad_buffer_thickness2_produces_square_cells_on_line():
    line = LineString([(1, 5), (11, 5)])
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 10), zone_id="domain", resolution=1.0, densify=False)
    cm.add_line(
        line,
        line_id="drain",
        resolution=1.0,
        quad_buffer=True,
        quad_buffer_thickness=2,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.0, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    assert mesher.generate(clean_polys, clean_lines, clean_points)
    grid = VoronoiTessellator(mesher, cm, clip_to_boundary=True).generate()

    on_line = grid[grid.geometry.centroid.distance(line) < 0.05]
    assert round(line.length) - 2 <= len(on_line) <= round(line.length) + 3
    areas = on_line.geometry.area
    assert areas.std() / areas.mean() < 0.2


def test_narrow_zone_quad_buffer_falls_back_with_warning():
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 10), zone_id="domain", resolution=1.0, densify=False)
    cm.add_polygon(
        Polygon([(3, 4.7), (9, 4.7), (9, 5.3), (3, 5.3)]),
        zone_id="sliver",
        resolution=1.0,
        z_order=1,
        densify=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.0, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    with pytest.warns(UserWarning, match="too narrow for a quad_buffer band"):
        assert mesher.generate(clean_polys, clean_lines, clean_points)
    assert not mesher.get_element_grid().empty


def test_buffer_surfaces_do_not_double_mesh_crossing_configuration():
    # Regression: OCC fragment silently refused to cut overlapping faces when a
    # trimmed line strip crossed a trimmed polygon band, leaving the domain and
    # zone surfaces triangulated on top of the strip quads (duplicate nodes,
    # junk Voronoi cells). Embedded polygon surfaces are now built disjoint
    # from every buffer footprint, so the elements tile the domain exactly.
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 10), zone_id="domain", resolution=1.0, densify=False)
    cm.add_polygon(
        Polygon([(3, 3), (9, 3), (9, 7), (3, 7)]),
        zone_id="inner",
        resolution=1.0,
        z_order=1,
        densify=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    cm.add_line(
        LineString([(0, 5), (12, 5)]),
        line_id="fault",
        resolution=1.0,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    # The inner zone has z_order=1, so it wins and stays continuous; the fault
    # line (z_order=0) is the lower-priority feature trimmed at the crossing.
    with pytest.warns(UserWarning, match="crosses a higher-priority protected feature"):
        assert mesher.generate(clean_polys, clean_lines, clean_points)

    element_grid = mesher.get_element_grid()
    assert abs(element_grid.geometry.area.sum() - 120.0) < 0.01


def test_quad_buffer_thickness_is_validated():
    cm = ConceptualMesh(crs=None)

    with pytest.raises(ValueError, match="quad_buffer_thickness"):
        cm.add_line(
            LineString([(0, 0), (1, 0)]),
            line_id="bad",
            resolution=1.0,
            quad_buffer=True,
            quad_buffer_thickness=3,
        )

    with pytest.raises(ValueError, match="quad_buffer_thickness"):
        cm.add_polygon(
            box(0, 0, 1, 1),
            zone_id="bad",
            resolution=1.0,
            quad_buffer=True,
            quad_buffer_thickness=0,
        )


def _crossing_line_and_band(line_lc=1.0, poly_z_order=0):
    """A horizontal quad-buffered line crossing a quad-buffered polygon band.

    The domain is given z_order=-1 so the inner zone (z_order=poly_z_order,
    defaulting to 0) survives overlap resolution while keeping a z_order equal to
    the line's default 0 -- letting the finer-lc tie-break decide the winner.
    """
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 10), zone_id="domain", resolution=1.0, z_order=-1, densify=True)
    cm.add_polygon(
        Polygon([(4, 3), (8, 3), (8, 7), (4, 7)]),
        zone_id="inner",
        resolution=1.0,
        z_order=poly_z_order,
        densify=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    cm.add_line(
        LineString([(0, 5), (12, 5)]),
        line_id="fault",
        resolution=line_lc,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    return cm, mesher, (clean_polys, clean_lines, clean_points)


def test_crossing_priority_finer_lc_wins():
    # The line (lc=0.5) is finer than the band (lc=1.0), so it wins the priority
    # and stays continuous; the band is the one trimmed at the crossing.
    _, mesher, clean = _crossing_line_and_band(line_lc=0.5)
    with pytest.warns(UserWarning, match=r"polygon feature \d+ crosses a higher-priority"):
        assert mesher.generate(*clean)
    quads = mesher.get_element_grid("quads")
    # The finer line strip runs continuously through the band region at y=5.
    line_row = quads[quads["centroid_y"].sub(5).abs() < 0.4]
    assert not line_row[line_row["centroid_x"].sub(6).abs() < 0.6].empty


def test_crossing_priority_z_order_override():
    # Same geometry, but z_order=10 on the polygon flips the winner: now the
    # band stays continuous and the line is trimmed.
    _, mesher, clean = _crossing_line_and_band(line_lc=1.0, poly_z_order=10)
    with pytest.warns(UserWarning, match=r"line feature \d+ crosses a higher-priority"):
        assert mesher.generate(*clean)
    # Disjoint tiling is preserved regardless of which feature wins.
    assert abs(mesher.get_element_grid().geometry.area.sum() - 120.0) < 0.01


def test_winner_strip_stays_transfinite_through_crossing():
    # Two equal quad buffers cross; the first-added (winning) horizontal strip
    # must keep its transfinite structure -- only the trimmed vertical loser may
    # degrade to recombine-only.
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 8), zone_id="domain", resolution=1.0, densify=True)
    # Interior winner (does not touch the domain boundary, so any degradation
    # would come from the crossing, not boundary end caps).
    cm.add_line(
        LineString([(1, 4), (11, 4)]),
        line_id="winner",
        resolution=1.0,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    cm.add_line(
        LineString([(6, 0), (6, 8)]),
        line_id="loser",
        resolution=1.0,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert mesher.generate(clean_polys, clean_lines, clean_points)
    winner_degraded = [
        w for w in caught
        if ("transfinite" in str(w.message) or "recombine-only" in str(w.message))
        and "('line', 0)" in str(w.message)
    ]
    assert not winner_degraded, [str(w.message) for w in winner_degraded]
    quads = mesher.get_element_grid("quads")
    winner_row = quads[quads["centroid_y"].sub(4).abs() < 0.5]
    # Continuous structured row through the crossing at x=6.
    assert not winner_row[winner_row["centroid_x"].sub(6).abs() < 0.6].empty


def test_partial_crossing_t_junction_records_refinement(monkeypatch):
    # A vertical buffer that terminates ON a horizontal buffer (a T-junction)
    # must mesh cleanly, conserve area, and register a crossing refinement disk.
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 8), zone_id="domain", resolution=1.0, densify=True)
    cm.add_line(
        LineString([(0, 4), (12, 4)]),
        line_id="trunk",
        resolution=1.0,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    cm.add_line(
        LineString([(6, 0), (6, 4)]),
        line_id="branch",
        resolution=1.0,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    crossings = _capture_crossings(mesher, monkeypatch)
    assert mesher.generate(clean_polys, clean_lines, clean_points)
    assert abs(mesher.get_element_grid().geometry.area.sum() - 96.0) < 0.01
    assert len(crossings) >= 1


def test_tangential_overlap_drops_sliver():
    # A loser strip overlapping the winner's corridor at a shallow offset leaves
    # a thin remnant that must be dropped (not meshed as a sliver surface).
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 8), zone_id="domain", resolution=1.0, densify=True)
    cm.add_line(
        LineString([(0, 4), (12, 4)]),
        line_id="winner",
        resolution=1.0,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    cm.add_line(
        LineString([(0, 4.8), (12, 4.8)]),
        line_id="loser",
        resolution=1.0,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    with pytest.warns(UserWarning, match=r"dropped \d+ sliver piece"):
        assert mesher.generate(clean_polys, clean_lines, clean_points)


def test_crossing_refinement_limits_size_jump(monkeypatch):
    # With a coarse background, the trimmed gap would fill with large triangles
    # next to the dense strip rows. The Ball refinement field pins it to the
    # feature size, so elements near the crossing stay close to lc, not
    # background_lc.
    lc = 1.0
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 10), zone_id="domain", resolution=lc, densify=True)
    cm.add_polygon(
        Polygon([(4, 3), (8, 3), (8, 7), (4, 7)]),
        zone_id="inner",
        resolution=lc,
        z_order=1,
        densify=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    cm.add_line(
        LineString([(0, 5), (12, 5)]),
        line_id="fault",
        resolution=lc,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=3.0, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    crossings = _capture_crossings(mesher, monkeypatch)
    assert mesher.generate(clean_polys, clean_lines, clean_points)
    assert len(crossings) >= 1

    grid = mesher.get_element_grid()
    centroids = grid.geometry.centroid
    # Around the left crossing point (4, 5), elements stay near lc rather than
    # jumping to the background size (area (3.0)**2 = 9).
    near = grid[(centroids.x.sub(4).abs() < 1.5) & (centroids.y.sub(5).abs() < 1.5)]
    assert not near.empty
    assert near.geometry.area.max() < (1.6 * lc) ** 2


def _blob(rx, ry, harmonics, n=160):
    """Ellipse with sinusoidal radius perturbations (the README splash domain)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones_like(t)
    for k, amp, phase in harmonics:
        r += amp * np.sin(k * t + phase)
    return Polygon(np.c_[rx * r * np.cos(t), ry * r * np.sin(t)])


# half_span 1300: the clipped river's strip ends with a corner on the domain
# outline, so domain minus footprint is a ring pinched at that corner. 1700:
# the strip crosses the whole domain and splits it in two.
@pytest.mark.parametrize("half_span", [1300, 1700])
def test_quad_buffer_touching_the_domain_outline_keeps_the_domain_surface(half_span):
    domain = _blob(1500, 480, [(2, 0.06, 0.4), (3, 0.05, 1.3), (5, 0.03, 2.1), (7, 0.015, 0.2)])
    x = np.linspace(-half_span, half_span, 300)
    river = LineString(np.c_[x, 120 * np.sin(x / 330) - 0.12 * x - 60])
    cm = ConceptualMesh()
    cm.add_polygon(domain, zone_id=1, resolution=110)
    cm.add_line(river, line_id="river", resolution=16, quad_buffer=True, quad_buffer_thickness=2)
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=110, verbosity=0)
    assert mesher.generate(clean_polys, clean_lines, clean_points)

    assert not mesher.get_element_grid("quads").empty
    grid = VoronoiTessellator(mesher, cm, clip_to_boundary=True).generate()
    # A dropped domain surface left ~500 cells around the river alone.
    assert len(grid) > 2000
    assert grid.geometry.area.sum() == pytest.approx(domain.area, rel=1e-6)
