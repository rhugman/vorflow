import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from vorflow.blueprint import ConceptualMesh
from vorflow.tessellator import VoronoiTessellator
from vorflow.utils import build_connectivity, grid_to_disv

from test_pipeline import FakeMeshGenerator


def _grid(polygons, centres=None):
    centres = centres if centres is not None else [p.centroid.coords[0] for p in polygons]
    return gpd.GeoDataFrame(
        {
            "node_id": np.arange(10, 10 + len(polygons)),
            "x": [c[0] for c in centres],
            "y": [c[1] for c in centres],
        },
        geometry=polygons,
    )


def _cell_rings(disv):
    xy = {iv: (x, y) for iv, x, y in disv["vertices"]}
    return [[xy[iv] for iv in cell[4:4 + cell[3]]] for cell in disv["cell2d"]]


def _signed_area(ring):
    x, y = np.asarray(ring).T
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def test_two_squares_share_vertices_and_list_them_clockwise():
    disv = grid_to_disv(_grid([box(0, 0, 1, 1), box(1, 0, 2, 1)]))

    assert disv["ncpl"] == 2
    assert disv["nvert"] == 6
    assert [cell[3] for cell in disv["cell2d"]] == [4, 4]
    shared = set(disv["cell2d"][0][4:]) & set(disv["cell2d"][1][4:])
    assert len(shared) == 2
    assert all(_signed_area(ring) < 0 for ring in _cell_rings(disv))
    np.testing.assert_array_equal(disv["node_id"], [10, 11])


def test_counter_clockwise_input_is_reversed():
    ccw = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert ccw.exterior.is_ccw
    disv = grid_to_disv(_grid([ccw]))
    assert _signed_area(_cell_rings(disv)[0]) < 0


def test_small_clockwise_cell_at_utm_coordinates_stays_clockwise():
    x, y = 500000., 5000000.
    polygon = Polygon([(x, y), (x, y + .01), (x + .01, y + .01), (x + .01, y)])
    disv = grid_to_disv(_grid([polygon]))
    ring = np.asarray(_cell_rings(disv)[0])
    assert _signed_area(ring - ring[0]) < 0


def test_mixed_orientations_share_vertices_and_all_list_clockwise():
    ccw = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    cw = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    hanging = Polygon([(0, 1), (2, 1), (2, 2), (0, 2)])
    disv = grid_to_disv(_grid([cw, ccw, hanging]))

    rings = _cell_rings(disv)
    assert all(_signed_area(ring) < 0 for ring in rings)
    assert (1.0, 1.0) in rings[2]
    assert disv["nvert"] == 8


def test_cell_centres_are_generator_points_not_centroids():
    disv = grid_to_disv(_grid([box(0, 0, 1, 1)], centres=[(0.2, 0.3)]))
    assert disv["cell2d"][0][1:3] == [0.2, 0.3]


def test_hanging_vertex_is_inserted_into_neighbouring_edge():
    # The tall cell's right edge (1,0)-(1,2) passes through (1,1), a corner of
    # both right-hand cells; without it MODFLOW 6 would not connect them.
    disv = grid_to_disv(_grid([box(0, 0, 1, 2), box(1, 0, 2, 1), box(1, 1, 2, 2)]))

    tall = _cell_rings(disv)[0]
    assert len(tall) == 5
    assert (1.0, 1.0) in tall
    assert _signed_area(tall) < 0
    assert Polygon(tall).equals(box(0, 0, 1, 2))


def test_hanging_vertex_insertion_matches_flopy_neighbours():
    pytest.importorskip("flopy")
    from flopy.discretization import VertexGrid

    grid = _grid([box(0, 0, 1, 2), box(1, 0, 2, 1), box(1, 1, 2, 2)])
    disv = grid_to_disv(grid)
    vertex_grid = VertexGrid(vertices=disv["vertices"], cell2d=disv["cell2d"], nlay=1)
    neighbours = vertex_grid.neighbors(method="rook")
    assert sorted(neighbours[0]) == [1, 2]


@pytest.mark.parametrize(
    "geometry, match",
    [
        (box(0, 0, 4, 4).difference(box(1, 1, 2, 2)), "holes"),
        (box(0, 0, 1, 1).union(box(2, 0, 3, 1)), "single polygons"),
    ],
)
def test_rejects_cells_disv_cannot_represent(geometry, match):
    with pytest.raises(ValueError, match=match):
        grid_to_disv(_grid([geometry], centres=[(0.5, 0.5)]))


def test_rejects_missing_columns_and_empty_grid():
    with pytest.raises(ValueError, match="x"):
        grid_to_disv(gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]))
    with pytest.raises(ValueError, match="non-empty"):
        grid_to_disv(_grid([]))


def _tessellated(barrier=False):
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (4, 0), (4, 3), (0, 3)]), zone_id=1, densify=0.5)
    if barrier:
        cm.add_line(LineString([(0.3, 0.1), (3.7, 2.9)]), line_id="fault", resolution=1, is_barrier=True)
    clean_polys, _, _ = cm.generate()
    tessellator = VoronoiTessellator(FakeMeshGenerator(clean_polys, spacing=0.45), cm)
    tessellator.generate()
    return tessellator


@pytest.mark.parametrize("barrier", [False, True])
def test_tessellated_grid_round_trips_cells_and_connections(barrier):
    tessellator = _tessellated(barrier)
    grid = tessellator.grid
    disv = tessellator.to_disv()

    assert disv["ncpl"] == len(grid)
    np.testing.assert_array_equal(disv["node_id"], grid["node_id"].to_numpy())
    for ring, cell in zip(_cell_rings(disv), grid.geometry):
        assert _signed_area(ring) < 0
        assert Polygon(ring).symmetric_difference(cell).area < 1e-9
    np.testing.assert_allclose([c[1] for c in disv["cell2d"]], grid["x"])

    # Every face build_connectivity finds is a vertex pair both cells list.
    shared = {}
    for cell in disv["cell2d"]:
        ids = cell[4:4 + cell[3]]
        for e in zip(ids, ids[1:] + ids[:1]):
            shared.setdefault(frozenset(e), set()).add(cell[0])
    disv_pairs = {frozenset(cells) for cells in shared.values() if len(cells) == 2}
    connectivity = build_connectivity(grid)
    expected = {frozenset(p) for p in zip(connectivity["cell_id_1"], connectivity["cell_id_2"])}
    assert expected <= disv_pairs


def test_to_disv_before_generate_raises():
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 1, 1), zone_id=1)
    clean_polys, _, _ = cm.generate()
    with pytest.raises(RuntimeError, match="generate"):
        VoronoiTessellator(FakeMeshGenerator(clean_polys), cm).to_disv()


def test_to_modflowgwfdisv_builds_flopy_package(tmp_path):
    flopy = pytest.importorskip("flopy")
    tessellator = _tessellated(barrier=True)
    sim = flopy.mf6.MFSimulation(sim_ws=str(tmp_path))
    gwf = flopy.mf6.ModflowGwf(sim, modelname="vorflow")

    disv = tessellator.to_modflowgwfdisv(gwf, top=10.0, botm=[0.0, -10.0], nlay=2, length_units="meters")

    assert disv.ncpl.get_data() == len(tessellator.grid)
    assert disv.nlay.get_data() == 2
    assert gwf.modelgrid.ncpl == len(tessellator.grid)


def test_to_modflowgwfdisv_without_flopy_raises(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "flopy", None)
    with pytest.raises(ImportError, match="to_disv"):
        _tessellated().to_modflowgwfdisv(gwf=None, top=1.0, botm=0.0)
