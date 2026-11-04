"""disu.refill: finer cells inside an outline that share the outline's vertices with the base grid."""

import pytest
import shapely
from shapely.geometry import Polygon, box

from vorflow import disu
from vorflow._mfusg import _planar_faces
from vorflow._refill import outline_nodes
from vorflow.fields import ExponentialField, MeshField

from test_disu_grid import assert_cells_close
from test_disu_multigrid import PIT, VORONOI_PIT, _boxes, six_layers, voronoi_pit


def _outline_vertices(grid, polygon, inside):
    geoms = grid.geometry.to_numpy()
    in_poly = shapely.area(shapely.intersection(geoms, polygon)) > 1e-9 * shapely.area(geoms)
    cells = geoms[in_poly == inside]
    nodes = outline_nodes(cells[shapely.dwithin(cells, polygon.exterior, 1e-9)], polygon.exterior, 1e-7)
    return {(round(x, 6), round(y, 6)) for x, y in nodes}


def test_refill_shares_every_outline_vertex_and_keeps_the_outside():
    base = _boxes(4, 1.0)
    grid = disu.refill(base, PIT, 0.25)
    outside = grid[grid.splice_source == "base"]
    # Base cells outside the outline are unchanged, with their node_ids.
    kept = base.set_index("node_id").loc[outside.node_id]
    assert all(a.equals(b) for a, b in zip(outside.geometry, kept.geometry))
    assert len(outside) == 12
    inside = grid[grid.splice_source == "fine"]
    assert inside.geometry.area.sum() == pytest.approx(PIT.area)
    assert len(inside) > 4 * 4 and grid.attrs["refill"]["cells"] == len(inside)
    # The new cells end exactly at the base vertices on the outline: one face per base face.
    assert _outline_vertices(grid, PIT, True) == _outline_vertices(base, PIT, False)
    assert grid.attrs["refill"]["outline_vertices"] == 8


def test_outline_split_add_vertices_between_the_shared_ones():
    base = _boxes(4, 1.0)
    grid = disu.refill(base, PIT, 0.25, outline_split=2)
    inner, outer = _outline_vertices(grid, PIT, True), _outline_vertices(base, PIT, False)
    assert outer < inner and len(inner) == 2 * len(outer)
    assert grid.attrs["refill"]["outline_spacing"] == pytest.approx(0.5)


def test_priority_chooses_between_resolution_and_smooth_growth():
    base = _boxes(4, 1.0)
    # From 0.25, growth 1.2 reaches only 0.35 within half the inradius (0.5): the default warns.
    with pytest.warns(UserWarning, match="priority='growth'"):
        sharp = disu.refill(base, PIT, 0.25).attrs["refill"]
    assert sharp["core_size"] == 0.25 and sharp["size_at_outline"] == pytest.approx(0.35)
    smooth = disu.refill(base, PIT, 0.25, priority="growth").attrs["refill"]
    assert smooth["core_size"] == pytest.approx(0.99 - 0.2 * 0.5)
    assert smooth["size_at_outline"] == pytest.approx(0.99)


def test_refill_reports_the_size_jump_between_neighbours():
    base = _boxes(4, 1.0)
    gentle = disu.refill(base, PIT, 0.25, growth_factor=1.2).attrs["refill"]["max_neighbour_area_ratio"]
    assert 1.0 <= gentle < 4.0


def test_refill_takes_size_fields_like_add_polygon():
    base = _boxes(4, 1.0)
    grid = disu.refill(base, PIT, 0.25, fields=[ExponentialField(size_min=0.25, decay_length=0.3)])
    assert _outline_vertices(grid, PIT, True) == _outline_vertices(base, PIT, False)
    assert grid.attrs["refill"]["cells"] > 4 * 4


class _NoSizeField(MeshField):
    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        return None


def test_refill_rejects_bad_input():
    base = _boxes(4, 1.0)
    with pytest.raises(ValueError, match="cross the refill outline"):
        disu.refill(base, box(0.5, 0.5, 3, 3), 0.25)
    with pytest.raises(ValueError, match="without holes"):
        disu.refill(base, Polygon(PIT.exterior.coords, [box(1.5, 1.5, 2.5, 2.5).exterior.coords]), 0.25)
    with pytest.raises(ValueError, match="growth_factor"):
        disu.refill(base, PIT, 0.25, growth_factor=1.0)
    with pytest.raises(ValueError, match="MeshField"):
        disu.refill(base, PIT, 0.25, fields=[1.2])
    with pytest.raises(ValueError, match="python_size"):
        disu.refill(base, PIT, 0.25, fields=[_NoSizeField()])
    with pytest.raises(ValueError, match="outline_split"):
        disu.refill(base, PIT, 0.25, outline_split=0)
    with pytest.raises(ValueError, match="resolution"):
        disu.refill(base, PIT, -1.0)
    with pytest.raises(ValueError, match="priority"):
        disu.refill(base, PIT, 0.25, priority="smooth")


@pytest.mark.slow
def test_refill_of_the_voronoi_pit_shares_all_outline_vertices():
    _, base = voronoi_pit()
    grid = disu.refill(base, VORONOI_PIT, 12)
    assert _outline_vertices(grid, VORONOI_PIT, True) == _outline_vertices(base, VORONOI_PIT, False)
    info = grid.attrs["refill"]
    assert info["outline_vertices"] == 28 and info["max_neighbour_area_ratio"] < 4
    # Every face, the outline included, is perpendicular to the line between its two generators.
    faces = _planar_faces(grid)[2]
    assert faces["ortho_error"].max() < 1e-4          # degrees; roundoff only
    assert faces["shared_length"].min() > 1e-3        # no roundoff-length faces on the outline
    # Refined at depth: faces between a coarse cell above and a fine cell below
    # are skewed but carry no connection, so the report's angle stays at roundoff.
    d = disu.DisuGrid([base, base, grid, grid], 100.0, [80.0, 60.0, 50.0, 40.0]).build()
    assert d._faces["ortho_error"].max() > 1
    assert d.diagnostics["max_ortho_error"] < 1e-4


@pytest.mark.slow
def test_growth_priority_grades_the_voronoi_pit_more_smoothly():
    _, base = voronoi_pit()
    with pytest.warns(UserWarning, match="outline_split"):
        sharp = disu.refill(base, VORONOI_PIT, 12, growth_factor=1.1).attrs["refill"]
    smooth = disu.refill(base, VORONOI_PIT, 12, growth_factor=1.1, priority="growth").attrs["refill"]
    assert sharp["core_size"] == 12 and smooth["core_size"] > 12
    assert smooth["max_neighbour_area_ratio"] < sharp["max_neighbour_area_ratio"]


@pytest.mark.slow
@pytest.mark.parametrize("outline_split", [1, 2])
@pytest.mark.parametrize("dip", [0.0, 0.03])
def test_voronoi_pit_refilled_at_depth_cells_close(dip, outline_split):
    _, base = voronoi_pit()
    grid = disu.refill(base, VORONOI_PIT, 12, outline_split=outline_split)
    d = six_layers([base] * 3 + [grid] * 3, dip)
    assert d.diagnostics["vertical_offset"] == 0
    assert_cells_close(d)


@pytest.mark.slow
def test_refill_of_an_irregular_outline_covers_it_exactly():
    """The mirrored generator pairs are cocircular; GEOS's Voronoi broke on this quarry, Qhull does not."""
    from shapely.geometry import LineString

    from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator

    quarry = Polygon([(250, 320), (520, 290), (560, 560), (300, 610), (230, 470)])
    cm = ConceptualMesh()
    cm.add_polygon(box(-150, -100, 2100, 1300), zone_id=1, resolution=80)
    cm.add_line(LineString(quarry.exterior.coords), line_id="quarry", resolution=25, is_barrier=True)
    mg = MeshGenerator(background_lc=80, conceptual_mesh=cm, verbosity=0)
    mg.generate()
    base = VoronoiTessellator(mg, cm).generate()
    grid = disu.refill(base, quarry, 12)
    fine = grid[grid.splice_source == "fine"].geometry
    assert (fine.geom_type == "Polygon").all()
    assert fine.area.sum() == pytest.approx(quarry.area, rel=1e-9)
    assert shapely.union_all(fine.to_numpy()).area == pytest.approx(quarry.area, rel=1e-9)
