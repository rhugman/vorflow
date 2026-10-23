import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from vorflow.blueprint import ConceptualMesh
import vorflow.tessellator as tessellator_module
from vorflow.tessellator import VoronoiTessellator
from vorflow.utils import boundary_connectivity_report, build_connectivity


class DummyMeshGenerator:
    def __init__(self, nodes, tags, zones_gdf):
        self.nodes = nodes
        self.node_tags = tags
        self.zones_gdf = zones_gdf


def _build_conceptual_mesh():
    cm = ConceptualMesh()
    box = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    cm.add_polygon(box, zone_id=42)
    return cm


def _plain_barrier_tessellator():
    cm = _build_conceptual_mesh()
    cm.add_line(
        LineString([(0.75, 0), (0.75, 1)]),
        line_id="barrier",
        resolution=1.0,
        is_barrier=True,
    )
    cm.generate()
    mesh_gen = DummyMeshGenerator(nodes=np.empty((0, 2)), tags=[], zones_gdf=cm.clean_polygons)
    return VoronoiTessellator(mesh_gen, cm), cm.clean_lines.iloc[0].geometry


def test_enforce_barriers_preserves_uint64_ids_and_split_fragments():
    """Catches coercion of newly assigned barrier-fragment IDs to float."""
    tessellator, barrier = _plain_barrier_tessellator()
    original_id = 2**64 - 2
    grid = gpd.GeoDataFrame(
        {"node_id": np.array([original_id], dtype=np.uint64), "x": [0.5], "y": [0.5]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
    )

    result = tessellator._enforce_barriers(grid)

    assert result["node_id"].dtype == np.dtype("uint64")
    assert result["node_id"].is_unique
    assert set(result["node_id"]) == {original_id, original_id + 1}
    assert result.loc[result["node_id"] == original_id, "geometry"].iloc[0].area == pytest.approx(0.75)
    assert result.loc[result["node_id"] == original_id + 1, "geometry"].iloc[0].area == pytest.approx(0.25)
    assert result.geometry.area.sum() == pytest.approx(1.0)
    assert not result.geometry.crosses(barrier).any()


def _barrier_tessellator(line, **line_kwargs):
    """Return a tessellator whose conceptual mesh holds one barrier line in a 2 x 1 box."""
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]), zone_id=1)
    cm.add_line(line, line_id="barrier", resolution=1.0, is_barrier=True, **line_kwargs)
    cm.generate()
    mesh_gen = DummyMeshGenerator(nodes=np.empty((0, 2)), tags=[], zones_gdf=cm.clean_polygons)
    return VoronoiTessellator(mesh_gen, cm)


def _two_cell_grid():
    """Return two unit cells sharing the face x=1."""
    return gpd.GeoDataFrame(
        {"node_id": np.array([1, 2], dtype=np.int64), "x": [0.5, 1.5], "y": [0.5, 0.5]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        ],
    )


def test_raw_voronoi_keeps_bounded_cells_with_their_tags():
    tessellator, _ = _plain_barrier_tessellator()
    inner = np.array([[0.4, 0.5], [0.6, 0.5], [0.5, 0.4], [0.5, 0.6]])
    ghosts = np.array([[-5.0, -5.0], [6.0, -5.0], [6.0, 6.0], [-5.0, 6.0], [0.5, 0.5]])
    tags = np.array([11, 12, 13, 14], dtype=np.uint64)

    raw = tessellator._build_raw_voronoi(np.vstack([inner, ghosts]), tags)

    # Only the inner four cells and the untagged centre node are bounded.
    assert raw["node_id"].tolist() == [11, 12, 13, 14, -1]
    assert raw[["x", "y"]].to_numpy().tolist() == np.vstack([inner, ghosts[-1:]]).tolist()
    assert raw.geometry.is_valid.all()
    assert shapely.contains_xy(raw.geometry.to_numpy(), raw["x"].to_numpy(), raw["y"].to_numpy()).all()


def test_enforce_barriers_leaves_cells_whose_face_lies_on_the_barrier():
    """Guards against splitting cells whose face merely lies on the barrier."""
    tessellator = _barrier_tessellator(LineString([(1, 0), (1, 1)]), densify=False)

    result = tessellator._enforce_barriers(_two_cell_grid())

    assert sorted(result["node_id"]) == [1, 2]
    assert result.geometry.area.tolist() == pytest.approx([1.0, 1.0])


def test_enforce_barriers_cuts_straddle_width_barrier_crossing_a_cell():
    """Catches straddle-width barriers being skipped where a cell still straddles them."""
    line = LineString([(0.25, 0), (0.25, 1)])
    tessellator = _barrier_tessellator(line, straddle_width=0.1, densify=False)

    result = tessellator._enforce_barriers(_two_cell_grid())

    assert sorted(result["node_id"]) == [1, 2, 3]
    kept = result[result["node_id"] == 1].iloc[0]
    fragment = result[result["node_id"] == 3].iloc[0]
    assert kept.geometry.area == pytest.approx(0.75)
    assert (kept["x"], kept["y"]) == (0.5, 0.5)
    assert fragment.geometry.area == pytest.approx(0.25)
    assert fragment["x"] == pytest.approx(0.125)
    assert not result.geometry.crosses(line).any()


def test_enforce_barriers_keeps_id_on_piece_holding_the_generator():
    """Catches the node_id moving to a larger piece that does not hold the generator."""
    line = LineString([(0.25, 0), (0.25, 1)])
    tessellator = _barrier_tessellator(line, densify=False)
    grid = _two_cell_grid()
    grid.loc[0, "x"] = 0.1

    result = tessellator._enforce_barriers(grid)

    kept = result[result["node_id"] == 1].iloc[0]
    assert kept.geometry.area == pytest.approx(0.25)
    assert kept["x"] == 0.1
    assert result[result["node_id"] == 3].iloc[0].geometry.area == pytest.approx(0.75)


def test_enforce_barriers_legacy_merge_retains_historical_small_fragment_rule():
    """Legacy mode reproduces the old union, even with a bent shared face."""
    line = LineString([(1, 0), (0.9, 0.5), (1, 1)])
    tessellator = _barrier_tessellator(line, densify=False, barrier_merge='legacy')

    result = tessellator._enforce_barriers(_two_cell_grid())

    assert sorted(result["node_id"]) == [1, 2]
    areas = result.set_index("node_id").geometry.area
    assert areas[1] == pytest.approx(0.95)
    assert areas[2] == pytest.approx(1.05)
    assert result.geometry.is_valid.all()
    assert not any(tessellator_module._straddled_pieces(cell, line) for cell in result.geometry)


def test_split_pieces_keep_area_where_barrier_runs_along_a_face():
    """Catches split() dropping a piece where the barrier follows a cell face to roundoff."""
    # A straddle pair's face on a curved barrier: the barrier bends at the
    # cell vertex and runs along the faces either side of it.
    cell = Polygon([
        (57.96922267793483, 11.86440677966101), (57.48847447458607, 9.36193501881629),
        (55.011979711113504, 10.297986205530755), (55.174633511821476, 13.730206996037758),
        (57.92501721293334, 14.299737684624162),
    ])
    line = LineString([
        (57.16349866562289, 8.950243290450704), (57.643613623698016, 10.169491525423728),
        (57.96922267793483, 11.864406779661017), (57.938457036682884, 13.559322033898304),
        (57.66593350278956, 14.756695707921759),
    ])

    pieces = tessellator_module._straddled_pieces(cell, line)

    assert sum(piece.area for piece in pieces) == pytest.approx(cell.area, rel=1e-12)
    assert len(pieces) == 3


def test_straddled_pieces_merge_cut_point_into_nearby_cell_vertex():
    """Catches split() leaving a zero-length edge where a barrier passes through a cell vertex."""
    vertex = (1.0, 1.0)
    cell = Polygon([(0, 0), (2, 0), (2, 1.5), vertex, (0, 1.2)])
    direction = np.array([1.0, 1.7]) / np.hypot(1.0, 1.7)
    # The barrier misses the vertex by roundoff, as on a Voronoi vertex of a straddle pair.
    start = np.array(vertex) - 3 * direction + [3e-14, 0.0]
    line = LineString([start, start + 6 * direction])

    pieces = tessellator_module._straddled_pieces(cell, line)

    assert len(pieces) == 2
    assert sum(piece.area for piece in pieces) == pytest.approx(cell.area)
    for piece in pieces:
        coords = np.asarray(piece.exterior.coords)
        assert np.hypot(*np.diff(coords, axis=0).T).min() > 1e-9
        assert tuple(vertex) in {tuple(c) for c in coords}


def test_merge_close_vertices_keeps_cell_whose_hole_collapses(caplog):
    """Catches GEOS raising for the whole grid when merging collapses one cell's hole to a point."""
    # Coordinate scale 20, so vertices within 2e-11 merge.
    holed = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(5, 5), (5 + 1e-12, 5), (5, 5 + 1e-12)]])
    # A near-duplicate vertex that should still be merged away.
    plain = Polygon([(10, 0), (20, 0), (20, 5), (20, 5 + 1e-12), (20, 10), (10, 10)])
    grid = gpd.GeoDataFrame({"node_id": [0, 1]}, geometry=[holed, plain])

    vorflow_logger = logging.getLogger("vorflow")
    old_propagate = vorflow_logger.propagate
    vorflow_logger.propagate = True
    try:
        with caplog.at_level("WARNING", logger="vorflow.tessellator"):
            result = tessellator_module._merge_close_vertices(grid)
    finally:
        vorflow_logger.propagate = old_propagate

    assert result.geometry.iloc[0].equals_exact(holed, 0)
    assert len(result.geometry.iloc[1].exterior.coords) == len(plain.exterior.coords) - 1
    assert "  -> Kept 1 cells unmerged: merging close vertices made them invalid" in caplog.messages


@pytest.mark.parametrize('piece', [
    # A sliver at a cell vertex, narrower than the snap tolerance (4e-8 here).
    Polygon([(10, 10), (10 - 1e-8, 10), (10, 10 - 1e-8)]),
    # A piece with a hole narrower than the snap tolerance.
    Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(5, 5), (5 + 1e-8, 5), (5, 5 + 1e-8)]]),
])
def test_snap_to_cell_vertices_keeps_piece_whose_ring_collapses(piece):
    """Catches GEOS raising when snapping collapses a ring of a split piece to two points."""
    cell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    # GEOS 3.13 raises on the collapsed ring; GEOS 3.11 leaves the ring as is.
    assert tessellator_module._snap_to_cell_vertices(piece, cell).equals(piece)


def test_enforce_barriers_retains_cell_and_logs_warning_when_split_fails(monkeypatch, caplog):
    """Catches removing a cell when Shapely raises while splitting it."""
    tessellator, _ = _plain_barrier_tessellator()
    grid = gpd.GeoDataFrame(
        {"node_id": np.array([7], dtype=np.uint64), "x": [0.5], "y": [0.5]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
    )

    def raise_split_error(*args, **kwargs):
        raise RuntimeError("split failed")

    monkeypatch.setattr(tessellator_module, "split", raise_split_error)
    vorflow_logger = logging.getLogger("vorflow")
    old_propagate = vorflow_logger.propagate
    vorflow_logger.propagate = True
    try:
        with caplog.at_level("WARNING", logger="vorflow.tessellator"):
            result = tessellator._enforce_barriers(grid)
    finally:
        vorflow_logger.propagate = old_propagate

    assert result.equals(grid)
    assert "Warning: Failed to split cell 7: split failed" in caplog.messages


def test_voronoi_clips_to_domain_and_assigns_zones():
    cm = _build_conceptual_mesh()
    clean_polys, _, _ = cm.generate()

    nodes = np.array(
        [
            [0.2, 0.2],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.8, 0.8],
        ]
    )
    tags = [1, 2, 3, 4]

    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=clean_polys)

    tessellator = VoronoiTessellator(mesh_gen, cm, clip_to_boundary=True)
    grid = tessellator.generate()

    assert not grid.empty
    assert grid["zone_id"].nunique() == 1
    assert grid["zone_id"].iloc[0] == 42

    domain = clean_polys.iloc[0].geometry
    assert grid.geometry.apply(lambda cell: cell.intersects(domain)).all()

    assert set(grid["node_id"]) == set(tags)
    assert "centroid_x" in grid.columns
    assert "centroid_y" in grid.columns


def test_voronoi_assigns_equal_priority_shared_border_nodes_by_zone_index(monkeypatch):
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), zone_id="left", z_order=0)
    cm.add_polygon(Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]), zone_id="right", z_order=0)
    clean_polys, _, _ = cm.generate()

    nodes = np.array([[0.5, 0.5], [1.0, 0.5], [1.5, 0.5], [1.0, 0.25]])
    tags = np.arange(1, len(nodes) + 1)
    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=clean_polys)

    def reversed_shared_border_matches(points, zones, how, predicate):
        records = []
        for _, point in points.iterrows():
            records.extend(
                [
                    {
                        "node_id": point["node_id"],
                        "geometry": point.geometry,
                        "index_right": 1,
                        "zone_id": "right",
                        "z_order": 0,
                    },
                    {
                        "node_id": point["node_id"],
                        "geometry": point.geometry,
                        "index_right": 0,
                        "zone_id": "left",
                        "z_order": 0,
                    },
                ]
            )
        return gpd.GeoDataFrame(records, geometry="geometry", crs=points.crs)

    monkeypatch.setattr(gpd, "sjoin", reversed_shared_border_matches)
    grid = VoronoiTessellator(mesh_gen, cm, clip_to_boundary=True).generate()

    shared_border_zones = grid.loc[grid["node_id"].isin([2, 4]), "zone_id"]
    assert shared_border_zones.tolist() == ["left", "left"]


def test_boundary_centering_default_matches_clip_mode():
    cm = _build_conceptual_mesh()
    clean_polys, _, _ = cm.generate()

    nodes = np.array(
        [
            [0.0, 0.25],
            [0.0, 0.75],
            [0.5, 0.25],
            [0.5, 0.75],
            [1.0, 0.25],
            [1.0, 0.75],
        ]
    )
    tags = np.arange(1, len(nodes) + 1)
    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=clean_polys)

    default_grid = VoronoiTessellator(mesh_gen, cm, clip_to_boundary=True).generate()
    clip_grid = VoronoiTessellator(
        mesh_gen,
        cm,
        clip_to_boundary=True,
        boundary_centering="clip",
    ).generate()

    ordered_default = default_grid.sort_values("node_id").reset_index(drop=True)
    ordered_clip = clip_grid.sort_values("node_id").reset_index(drop=True)

    assert ordered_default["node_id"].tolist() == ordered_clip["node_id"].tolist()
    assert np.allclose(ordered_default["x"], ordered_clip["x"])
    assert np.allclose(ordered_default["y"], ordered_clip["y"])
    assert all(
        geom_a.equals_exact(geom_b, tolerance=1e-12)
        for geom_a, geom_b in zip(ordered_default.geometry, ordered_clip.geometry)
    )
    assert "source_x" not in default_grid.columns
    assert "boundary_centered" not in default_grid.columns


def test_boundary_inset_mirror_shifts_non_corner_boundary_centers_inward():
    cm = _build_conceptual_mesh()
    clean_polys, _, _ = cm.generate()

    nodes = np.array(
        [
            [0.0, 0.25],
            [0.0, 0.75],
            [0.5, 0.25],
            [0.5, 0.75],
            [1.0, 0.25],
            [1.0, 0.75],
        ]
    )
    tags = np.arange(1, len(nodes) + 1)
    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=clean_polys)

    grid = VoronoiTessellator(
        mesh_gen,
        cm,
        clip_to_boundary=True,
        boundary_centering="inset_mirror",
    ).generate()

    assert not grid.empty
    assert {"source_x", "source_y", "boundary_centering", "boundary_inset", "boundary_centered"}.issubset(
        grid.columns
    )

    shifted_left = grid[np.isclose(grid["source_x"], 0.0)]
    shifted_right = grid[np.isclose(grid["source_x"], 1.0)]
    assert shifted_left["boundary_centered"].all()
    assert shifted_right["boundary_centered"].all()
    assert (shifted_left["x"] > shifted_left["source_x"]).all()
    assert (shifted_right["x"] < shifted_right["source_x"]).all()
    assert (grid["boundary_inset"] >= 0.0).all()


def test_boundary_inset_mirror_offsets_scale_with_local_spacing():
    cm = _build_conceptual_mesh()
    clean_polys, _, _ = cm.generate()

    nodes = np.array(
        [
            [0.0, 0.25],
            [0.0, 0.75],
            [1.0, 0.20],
            [1.0, 0.40],
            [0.5, 0.5],
        ]
    )
    tags = np.arange(1, len(nodes) + 1)
    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=clean_polys)
    tessellator = VoronoiTessellator(mesh_gen, cm, boundary_centering="inset_mirror")

    prepared, _, ghosts, metadata = tessellator._prepare_boundary_centered_nodes(nodes, tags)

    coarse_inset = metadata.loc[metadata["source_x"] == 0.0, "boundary_inset"].iloc[0]
    fine_inset = metadata.loc[metadata["source_x"] == 1.0, "boundary_inset"].iloc[0]

    assert coarse_inset > fine_inset
    # Default fraction 0.25 of the nearest boundary-node spacing (0.5 and 0.2).
    assert np.isclose(coarse_inset, 0.125)
    assert np.isclose(fine_inset, 0.05)
    assert len(ghosts) == int(metadata["boundary_centered"].sum())
    assert not np.allclose(prepared, nodes)


def test_boundary_inset_mirror_skips_sharp_corners():
    cm = _build_conceptual_mesh()
    clean_polys, _, _ = cm.generate()

    nodes = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
        ]
    )
    tags = np.arange(1, len(nodes) + 1)
    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=clean_polys)
    tessellator = VoronoiTessellator(mesh_gen, cm, boundary_centering="inset_mirror")

    prepared, _, _, metadata = tessellator._prepare_boundary_centered_nodes(nodes, tags)

    corner_row = metadata[metadata["node_id"] == 1].iloc[0]
    assert not bool(corner_row["boundary_centered"])
    assert corner_row["boundary_inset"] == 0.0
    assert np.allclose(prepared[0], nodes[0])


def test_boundary_inset_mirror_keeps_node_when_ghost_lands_inside_domain():
    """Catches mirror ghosts placed back inside the domain across a narrow hole."""
    slit = [(1.0, 1.95), (3.0, 1.95), (3.0, 2.05), (1.0, 2.05)]
    domain = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], [slit])
    cm = ConceptualMesh()
    cm.add_polygon(domain, zone_id=1, densify=False)
    clean_polys, _, _ = cm.generate()

    # Two nodes on the slit's lower face are 1.0 apart, so their inset (0.25)
    # and mirror ghost reach across the 0.1-wide slit into the domain.
    slit_nodes = [[1.5, 1.95], [2.5, 1.95]]
    outer_nodes = [[2.0, 0.0], [0.0, 2.0], [4.0, 2.0], [2.0, 4.0]]
    interior_nodes = [[1.0, 1.0], [3.0, 1.0], [1.0, 3.0], [3.0, 3.0]]
    nodes = np.array(slit_nodes + outer_nodes + interior_nodes, dtype=float)
    tags = np.arange(1, len(nodes) + 1)
    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=clean_polys)
    tessellator = VoronoiTessellator(mesh_gen, cm, boundary_centering="inset_mirror")

    prepared, _, ghosts, metadata = tessellator._prepare_boundary_centered_nodes(nodes, tags)

    assert not metadata.loc[:1, "boundary_centered"].any()
    assert (metadata.loc[:1, "boundary_inset"] == 0.0).all()
    assert np.array_equal(prepared[:2], nodes[:2])
    assert metadata.loc[2:5, "boundary_centered"].all()
    assert len(ghosts) == int(metadata["boundary_centered"].sum())
    assert not shapely.contains_xy(domain, ghosts[:, 0], ghosts[:, 1]).any()

    grid = tessellator.generate()
    assert grid["node_id"].is_unique
    assert grid.geometry.area.sum() == pytest.approx(domain.area)


def _staggered_lattice_nodes():
    """Unit-box lattice with boundary nodes on the edges and staggered interior rows."""
    nodes = []
    for x in (0.25, 0.5, 0.75):
        nodes.append([x, 0.0])
        nodes.append([x, 1.0])
    for y in (0.25, 0.5, 0.75):
        nodes.append([0.0, y])
        nodes.append([1.0, y])
    for x in (0.375, 0.625):
        nodes.append([x, 0.25])
        nodes.append([x, 0.75])
    for x in (0.25, 0.5, 0.75):
        nodes.append([x, 0.5])
    return np.array(nodes, dtype=float)


def test_boundary_connectivity_report_restricts_to_boundary_cells():
    cm = _build_conceptual_mesh()
    clean_polys, _, _ = cm.generate()
    domain_geom = unary_union(clean_polys.geometry)

    coords = [0.125, 0.375, 0.625, 0.875]
    nodes = np.array([[x, y] for x in coords for y in coords])
    tags = np.arange(1, len(nodes) + 1)
    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=clean_polys)

    grid = VoronoiTessellator(mesh_gen, cm, clip_to_boundary=True).generate()
    full = build_connectivity(grid, center="centroid")
    report = boundary_connectivity_report(grid, domain_geom, center="centroid")

    assert not report.empty
    assert len(report) < len(full)

    boundary = domain_geom.boundary
    cell_geoms = grid.geometry.reset_index(drop=True)
    boundary_cells = {
        i for i, geom in enumerate(cell_geoms) if geom.distance(boundary) <= 1e-8
    }
    assert (
        report["cell_id_1"].isin(boundary_cells) | report["cell_id_2"].isin(boundary_cells)
    ).all()

    interior_pairs = full[
        ~full["cell_id_1"].isin(boundary_cells) & ~full["cell_id_2"].isin(boundary_cells)
    ]
    assert not interior_pairs.empty
    merged = report.merge(
        interior_pairs[["cell_id_1", "cell_id_2"]],
        on=["cell_id_1", "cell_id_2"],
        how="inner",
    )
    assert merged.empty


def test_inset_mirror_improves_boundary_cell_centroid_orthogonality():
    cm = _build_conceptual_mesh()
    clean_polys, _, _ = cm.generate()
    domain_geom = unary_union(clean_polys.geometry)

    nodes = _staggered_lattice_nodes()
    tags = np.arange(1, len(nodes) + 1)
    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=clean_polys)

    clip_grid = VoronoiTessellator(
        mesh_gen, cm, clip_to_boundary=True, boundary_centering="clip"
    ).generate()
    mirror_grid = VoronoiTessellator(
        mesh_gen, cm, clip_to_boundary=True, boundary_centering="inset_mirror"
    ).generate()

    clip_report = boundary_connectivity_report(clip_grid, domain_geom, center="centroid")
    mirror_report = boundary_connectivity_report(mirror_grid, domain_geom, center="centroid")

    assert not clip_report.empty
    assert not mirror_report.empty

    clip_error = clip_report["ortho_error"].mean()
    mirror_error = mirror_report["ortho_error"].mean()
    assert clip_error > 1.0  # the staggered clip-mode grid is measurably non-orthogonal
    assert mirror_error < clip_error


def test_boundary_centering_rejects_invalid_mode():
    cm = _build_conceptual_mesh()
    clean_polys, _, _ = cm.generate()
    mesh_gen = DummyMeshGenerator(
        nodes=np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]]),
        tags=np.array([1, 2, 3]),
        zones_gdf=clean_polys,
    )

    with pytest.raises(ValueError, match="boundary_centering"):
        VoronoiTessellator(mesh_gen, cm, boundary_centering="mirror")


class TestExportToShapefile:
    def _tessellator_with_grid(self):
        cm = _build_conceptual_mesh()
        clean_polys, _, _ = cm.generate()
        nodes = np.array([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]])
        mesh_gen = DummyMeshGenerator(nodes=nodes, tags=[1, 2, 3, 4], zones_gdf=clean_polys)
        return VoronoiTessellator(mesh_gen, cm, clip_to_boundary=True)

    def test_writes_readable_shapefile(self, tmp_path):
        tess = self._tessellator_with_grid()
        grid = tess.generate()
        path = tmp_path / "grid.shp"
        with pytest.warns(DeprecationWarning, match="to_file"):
            tess.export_to_shapefile(str(path))
        assert path.exists()
        back = gpd.read_file(path)
        assert len(back) == len(grid)
        assert back.geometry.is_valid.all()
        assert back.geometry.area.sum() == pytest.approx(grid.geometry.area.sum())

    def test_no_grid_writes_nothing(self, tmp_path):
        tess = self._tessellator_with_grid()  # generate() never called
        path = tmp_path / "grid.shp"
        with pytest.warns(DeprecationWarning):
            tess.export_to_shapefile(str(path))
        assert not path.exists()

    def test_to_file_writes_geopackage_layer(self, tmp_path):
        tess = self._tessellator_with_grid()
        grid = tess.generate()
        path = tmp_path / "grid.gpkg"
        tess.to_file(path, layer="cells")
        back = gpd.read_file(path, layer="cells")
        assert len(back) == len(grid)

    def test_to_file_before_generate_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="generate"):
            self._tessellator_with_grid().to_file(tmp_path / "grid.gpkg")


def _lattice_nodes(minx, miny, maxx, maxy, n):
    """Return an n x n lattice of generator nodes over a bounding box."""
    xs = np.linspace(minx, maxx, n)
    ys = np.linspace(miny, maxy, n)
    return np.array([[x, y] for x in xs for y in ys])


def test_field_only_polygon_does_not_assign_zones():
    """Catches a sizing-only (embed=False) polygon stamping its zone_id onto cells."""
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), zone_id=1)
    cm.add_polygon(
        Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
        zone_id=99,
        z_order=5,
        resolution=0.5,
        embed=False,
    )
    clean_polys, _, _ = cm.generate()
    assert 99 in set(clean_polys["zone_id"])

    nodes = _lattice_nodes(0.0, 0.0, 4.0, 4.0, 9)
    tags = np.arange(1, len(nodes) + 1)
    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=clean_polys)
    grid = VoronoiTessellator(mesh_gen, cm, clip_to_boundary=True).generate()

    assert set(grid["zone_id"]) == {1}
    assert grid.geometry.area.sum() == pytest.approx(16.0)


def test_field_only_polygon_does_not_extend_clip_domain():
    """Catches a field-only polygon outside the embedded domain enlarging the clip."""
    cm = _build_conceptual_mesh()
    cm.generate()
    field_only = gpd.GeoDataFrame(
        {"zone_id": [99], "z_order": [5], "embed": [False]},
        geometry=[Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])],
    )
    cm.clean_polygons = gpd.GeoDataFrame(
        pd.concat([cm.clean_polygons, field_only], ignore_index=True)
    )

    nodes = _lattice_nodes(0.0, 0.0, 2.0, 1.0, 5)
    tags = np.arange(1, len(nodes) + 1)
    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=cm.clean_polygons)
    grid = VoronoiTessellator(mesh_gen, cm, clip_to_boundary=True).generate()

    assert grid.geometry.area.sum() == pytest.approx(1.0)
    assert set(grid["zone_id"]) == {42}


def test_missing_embed_values_are_treated_as_embedded():
    """Catches NaN embed flags dropping real zones from the zone join."""
    cm = _build_conceptual_mesh()
    cm.generate()
    cm.clean_polygons["embed"] = None

    nodes = _lattice_nodes(0.1, 0.1, 0.9, 0.9, 3)
    tags = np.arange(1, len(nodes) + 1)
    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=cm.clean_polygons)
    grid = VoronoiTessellator(mesh_gen, cm, clip_to_boundary=True).generate()

    assert set(grid["zone_id"]) == {42}


def test_multipart_clip_keeps_node_ids_unique():
    """Catches clip-then-explode duplicating node_id when a cell reaches across a notch."""
    cm = ConceptualMesh()
    u_shape = Polygon([(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)])
    cm.add_polygon(u_shape, zone_id=7)
    clean_polys, _, _ = cm.generate()
    # Node 3 sits in the left prong but its raw cell also covers most of the
    # empty right prong, so clipping yields a larger detached part.
    nodes = np.array([[1.2, 0.6], [0.3, 1.7], [0.9, 2.0], [0.6, 2.8], [1.1, 0.3]])
    tags = np.arange(1, len(nodes) + 1)
    mesh_gen = DummyMeshGenerator(nodes=nodes, tags=tags, zones_gdf=clean_polys)

    grid = VoronoiTessellator(mesh_gen, cm, clip_to_boundary=True).generate()

    assert grid["node_id"].is_unique
    assert (grid.geom_type == "Polygon").all()
    assert grid.geometry.area.sum() == pytest.approx(u_shape.area)
    fresh_ids = sorted(set(grid["node_id"]) - set(tags))
    assert len(fresh_ids) >= 1
    assert fresh_ids == list(range(6, 6 + len(fresh_ids)))
    generator_cell = grid[grid["node_id"] == 3].iloc[0]
    assert generator_cell.geometry.covers(Point(0.9, 2.0))
    assert (generator_cell["x"], generator_cell["y"]) == (0.9, 2.0)
    detached = grid[grid.geometry.covers(Point(2.5, 2.3))].iloc[0]
    assert detached["node_id"] in fresh_ids
    assert detached.geometry.area > generator_cell.geometry.area
    assert detached["x"] == pytest.approx(detached.geometry.centroid.x)
    assert detached["y"] == pytest.approx(detached.geometry.centroid.y)
    assert detached["zone_id"] == 7


def test_generators_just_outside_a_slanted_edge_still_get_a_zone():
    # Mesh nodes on a slanted domain edge can sit a floating-point hair
    # outside the zone polygon, so an intersects join alone misses them.
    domain = Polygon([(0, 0), (1, 0), (1.2, 1), (0, 1)])
    cm = ConceptualMesh()
    cm.add_polygon(domain, zone_id=7)
    cm.generate()

    edge = LineString([(1, 0), (1.2, 1)])
    outward = np.array([1.0, -0.2]) / np.hypot(1.0, 0.2)
    on_edge = [np.array(edge.interpolate(f, normalized=True).coords[0]) + 1e-12 * outward
               for f in (0.25, 0.5, 0.75)]
    interior = [(0.2, 0.2), (0.5, 0.5), (0.2, 0.8), (0.6, 0.2), (0.6, 0.8)]
    nodes = np.vstack([np.array(interior), np.array(on_edge)])
    assert not any(domain.intersects(Point(p)) for p in on_edge)

    mesher = DummyMeshGenerator(nodes, np.arange(1, len(nodes) + 1), cm.clean_polygons)
    grid = VoronoiTessellator(mesher, cm, clip_to_boundary=True).generate()

    assert grid["zone_id"].notna().all()
    assert set(grid["zone_id"]) == {7}


@pytest.mark.slow
def test_real_mesh_on_slanted_domain_assigns_every_cell_a_zone():
    from vorflow.engine import MeshGenerator

    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (120, 0), (130, 60), (60, 90), (0, 70)]), zone_id="dom", resolution=6)
    clean = cm.generate()
    mesher = MeshGenerator(background_lc=6, verbosity=0)
    assert mesher.generate(*clean)
    grid = VoronoiTessellator(mesher, cm).generate()
    assert grid["zone_id"].eq("dom").all()
