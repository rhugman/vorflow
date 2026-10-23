"""Independent bend-crossing acceptance and adversarial tracking cases."""

import geopandas as gpd
import pytest
from shapely.affinity import scale, translate
from shapely.geometry import LineString, Polygon, box

from vorflow.utils import grid_barrier_report


def two_cells(*, horizontal=False):
    if horizontal:
        geometry = [box(-2, -2, 2, 0), box(-2, 0, 2, 2)]
        x, y = [0.0, 0.0], [-1.0, 1.0]
    else:
        geometry = [box(-2, -2, 0, 2), box(0, -2, 2, 2)]
        x, y = [-1.0, 1.0], [0.0, 0.0]
    return gpd.GeoDataFrame({"node_id": [17, 29], "x": x, "y": y}, geometry=geometry)


def track(cells, coordinates):
    lines = gpd.GeoDataFrame(
        {"line_id": ["bend"], "is_barrier": [True]},
        geometry=[LineString(coordinates)],
    )
    return grid_barrier_report(cells, lines, approximate=True)


@pytest.mark.parametrize(
    "factor,offset", [(1e-3, (700000.0, 6000000.0)), (1e6, (0.0, 0.0))]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_bend_labels_are_invariant_to_extreme_scale(factor, offset, reverse):
    cells = two_cells()
    target = LineString([(-0.2, -2), (0, 0), (-0.2, 2)])
    cells.geometry = cells.geometry.map(
        lambda geom: translate(scale(geom, factor, factor, origin=(0, 0)), *offset)
    )
    centres = cells[["x", "y"]].to_numpy() * factor + offset
    cells[["x", "y"]] = centres
    target = translate(scale(target, factor, factor, origin=(0, 0)), *offset)
    coordinates = list(target.coords)[::-1] if reverse else list(target.coords)
    before = cells.copy(deep=True)
    result = track(cells, coordinates)
    face = result["approximate_faces"].iloc[0]
    expected = (29, 17) if reverse else (17, 29)
    assert (face.left_node_id, face.right_node_id) == expected
    assert face.usable
    assert result["approximate_summary"].iloc[0].ambiguous_connectors == 0
    shared = cells.geometry.iloc[0].boundary.intersection(
        cells.geometry.iloc[1].boundary
    )
    assert face.geometry.hausdorff_distance(shared) <= 2 * result["tolerance"]
    assert cells.equals(before)


def test_retraced_u_turn_does_not_create_a_barrier_connection():
    result = track(two_cells(), [(0, -1), (0, 0), (0, -1)])
    assert result["approximate_faces"].empty


def test_numerical_centre_on_the_target_is_still_excluded():
    result = track(two_cells(), [(-1, -2), (-1, 0), (0, 1)])
    assert result["approximate_faces"].empty
    assert result["approximate_summary"].iloc[0].boundary_centre_ties == 1


def test_self_intersection_is_not_misclassified_as_one_simple_bend():
    # Nonadjacent branches cross at the SAME point, so geometric intersection
    # is Point rather than MultiPoint. Endpoint signs are consistent against
    # both branches, but this still does not define one simple directed bend.
    target = [(-2, -2), (2, 2), (-2, 2), (2, -2)]
    result = track(two_cells(horizontal=True), target)
    assert result["approximate_faces"].empty


def test_real_near_straight_quad_cell_195_is_tracked_across_the_bend():
    # Retained original Gmsh/Voronoi cells 195/214 from the measured near-straight
    # case. Their connector hits the line vertex (100, 45) exactly.
    cells = gpd.GeoDataFrame(
        {
            "node_id": [8, 27],
            "x": [100.0, 100.0],
            "y": [49.00780488547034, 40.99219511452966],
        },
        geometry=[
            Polygon(
                [
                    (103.52532427914723, 45.0),
                    (96.47467572085277, 45.0),
                    (96.07082891915103, 51.46154882722788),
                    (99.61046488174195, 53.69977733959533),
                    (103.92867842004759, 51.45366625440568),
                ]
            ),
            Polygon(
                [
                    (99.81265219769524, 36.94364405875445),
                    (96.59150895482013, 39.12286337105187),
                    (96.238879791326, 44.76492998695787),
                    (96.47467572085277, 45.0),
                    (103.52532427914723, 45.0),
                    (103.761120208674, 44.764929986957874),
                    (103.3330147016072, 37.91524187388919),
                ]
            ),
        ],
        index=[195, 214],
    )
    target = [(20.0, 40.0), (100.0, 45.0), (180.0, 40.0)]
    before = cells.copy(deep=True)
    result = track(cells, target)
    assert len(result["approximate_faces"]) == 1
    face = result["approximate_faces"].iloc[0]
    assert (face.left_node_id, face.right_node_id) == (8, 27)
    assert face.usable
    assert result["faces"].empty  # Tracking is explicitly an approximation.
    assert cells.equals(before)
    reversed_face = track(cells, target[::-1])["approximate_faces"].iloc[0]
    assert (reversed_face.left_node_id, reversed_face.right_node_id) == (27, 8)
