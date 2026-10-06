from __future__ import annotations

import logging
import warnings
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union, split
from shapely.validation import make_valid

from . import _lloyd
from ._features import polygon_parts

logger = logging.getLogger(__name__)

# Split pieces smaller than this fraction of the cell area are treated as
# floating-point slivers from a barrier that runs along a cell face.
BARRIER_SLIVER_FRACTION = 1e-6

# Split pieces without the generator smaller than this fraction of the cell
# area join the neighbouring cell on their side of the barrier instead of
# becoming cells of their own. A curved barrier leaves such pieces all along
# it: the straddle pairs' Voronoi faces are chords of the curve, which bulges
# across them by the sagitta (pieces of 0.03-13% of the cell on curves with
# radius 1-4 lc).
BARRIER_FRAGMENT_MERGE_FRACTION = 0.2

# Rounds of barrier mirroring before leftover crossings go to the post-hoc split.
BARRIER_MIRROR_PASSES = 3

# A mirror generator is dropped when an existing node lies closer to it than
# this fraction of its distance to the original node: that node already
# acts as the partner, and a mirror beside it would squeeze both cells. On a
# curved barrier a straddle point's partner reflected across the nearest
# segment lands 4-19% of the reflection distance from it (lc 2, radius 1.7-8).
BARRIER_MIRROR_MERGE_FRACTION = 0.1

# Cell vertices closer than this fraction of the grid's coordinate scale are
# one vertex. Clipping puts a cut point within roundoff of a domain vertex
# wherever a Voronoi face meets the boundary there, which leaves a
# zero-length edge; MODFLOW 6 can crash on those.
VERTEX_MERGE_FRACTION = 1e-12


def _merged_vertex_index(vertices: np.ndarray, tolerance: float) -> np.ndarray:
    """Map each vertex index to the lowest index in its cluster of vertices chained within tolerance."""
    pairs = cKDTree(vertices).query_pairs(tolerance, output_type='ndarray')
    n = len(vertices)
    if len(pairs) == 0:
        return np.arange(n)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    representative = np.full(labels.max() + 1, n)
    np.minimum.at(representative, labels, np.arange(n))
    return representative[labels]


def _merge_close_vertices(grid_gdf: gpd.GeoDataFrame, rel_tol: float = VERTEX_MERGE_FRACTION) -> gpd.GeoDataFrame:
    """
    Merge cell vertices within rel_tol x coordinate scale of each other across the whole grid.

    Every vertex moves to its cluster representative, so neighbouring cells
    keep identical shared vertices; consecutive repeats are then dropped.
    The scale includes the coordinate magnitude because roundoff grows with
    it (e.g. UTM northings). A cell the merge would make invalid keeps its
    geometry, including one with a ring (e.g. a tiny hole) that collapses to
    fewer than three points, which GEOS rejects.
    """
    if grid_gdf.empty:
        return grid_gdf
    geoms = grid_gdf.geometry.to_numpy()
    coords, owner = shapely.get_coordinates(geoms, return_index=True)
    scale = max(float(np.ptp(coords, axis=0).max()), float(np.abs(coords).max()), 1.0)
    index = _merged_vertex_index(coords, rel_tol * scale)
    # Vertices shared exactly between neighbours map to one index but do not move.
    moved = (coords[index] != coords).any(axis=1)
    if not moved.any():
        return grid_gdf

    merged = shapely.set_coordinates(geoms.copy(), coords[index])
    changed = np.unique(owner[moved])
    collapsed = np.zeros(len(changed), dtype=bool)
    try:
        merged[changed] = shapely.remove_repeated_points(merged[changed])
    except shapely.errors.GEOSException:
        # One collapsed ring aborts the vectorised call; retry cell by cell.
        for k, i in enumerate(changed):
            try:
                merged[i] = shapely.remove_repeated_points(merged[i])
            except shapely.errors.GEOSException:
                collapsed[k] = True
    invalid = changed[collapsed | ~shapely.is_valid(merged[changed])]
    merged[invalid] = geoms[invalid]
    if len(invalid):
        logger.warning(f"  -> Kept {len(invalid)} cells unmerged: merging close vertices made them invalid")
    logger.info(f"  -> Merged {int(moved.sum())} cell vertices within roundoff of another")

    result = grid_gdf.copy()
    result[grid_gdf.geometry.name] = gpd.GeoSeries(merged, index=grid_gdf.index, crs=grid_gdf.crs)
    return result


def _boundary_node_spacing(nodes, boundary_indices):
    """Return each boundary node's distance to its nearest distinct boundary node (NaN elsewhere)."""
    spacing = np.full(len(nodes), np.nan)
    if len(boundary_indices) < 2:
        return spacing
    boundary_xy = nodes[boundary_indices]
    # Querying k=2 against de-duplicated coordinates returns the node's own
    # coordinate (distance 0) and the nearest *different* coordinate.
    unique_xy = np.unique(boundary_xy, axis=0)
    if len(unique_xy) > 1:
        dists, _ = cKDTree(unique_xy).query(boundary_xy, k=2)
        spacing[boundary_indices] = dists[:, 1]
    return spacing


def _ring_vertex_angles(domain_geom):
    """Return every open ring vertex of the domain and the angle between its two edges in degrees (NaN if degenerate)."""
    if isinstance(domain_geom, Polygon):
        polygons = [domain_geom]
    elif isinstance(domain_geom, MultiPolygon):
        polygons = list(domain_geom.geoms)
    else:
        polygons = []

    vertices, angles = [], []
    for poly in polygons:
        for ring in (poly.exterior, *poly.interiors):
            coords = np.asarray(ring.coords, dtype=float)[:, :2]
            if len(coords) < 4:
                continue
            open_coords = coords[:-1]
            v1 = np.roll(open_coords, 1, axis=0) - open_coords
            v2 = np.roll(open_coords, -1, axis=0) - open_coords
            magnitude = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                cos_theta = np.einsum('ij,ij->i', v1, v2) / magnitude
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            angle[magnitude == 0] = np.nan
            vertices.append(open_coords)
            angles.append(angle)
    if not vertices:
        return np.empty((0, 2)), np.empty(0)
    return np.concatenate(vertices), np.concatenate(angles)


def _ring_angles_at_points(points_xy, vertices, angles, tolerance):
    """Return the ring angle at each point from the first ring vertex within tolerance (NaN if none)."""
    result = np.full(len(points_xy), np.nan)
    if len(points_xy) == 0 or len(vertices) == 0:
        return result
    matches = cKDTree(vertices).query_ball_point(points_xy, r=tolerance)
    for i, vertex_indices in enumerate(matches):
        if vertex_indices:
            result[i] = angles[min(vertex_indices)]
    return result


def _boundary_tangents(boundary, points, spacing):
    """Return unit boundary tangents at points from samples +/- spacing/4 along the boundary (NaN if degenerate)."""
    length = boundary.length
    distance = shapely.line_locate_point(boundary, points)
    eps = np.maximum(np.maximum(spacing * 0.25, length * 1e-9), 1e-9)
    before = np.maximum(0.0, distance - eps)
    after = np.minimum(length, distance + eps)
    same = before == after
    before[same] = np.maximum(0.0, distance[same] - 1e-9)
    after[same] = np.minimum(length, distance[same] + 1e-9)
    p1 = shapely.get_coordinates(shapely.line_interpolate_point(boundary, before))
    p2 = shapely.get_coordinates(shapely.line_interpolate_point(boundary, after))
    tangents = p2 - p1
    norm = np.linalg.norm(tangents, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        tangents = tangents / norm[:, None]
    tangents[norm == 0] = np.nan
    return tangents


def _inward_normals(domain_geom, points_xy, tangents, inset):
    """Return the unit normal pointing into the domain at each point (NaN if neither side probes inside)."""
    left = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    right = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    probe = np.maximum(inset * 0.5, 1e-9)[:, None]
    left_inside = _probe_inside(domain_geom, points_xy + left * probe)
    right_inside = _probe_inside(domain_geom, points_xy + right * probe)
    # The left normal takes precedence when both probes land inside.
    normals = np.where(right_inside[:, None], right, np.nan)
    return np.where(left_inside[:, None], left, normals)


def _probe_inside(domain_geom, probe_xy):
    """Return whether each probe point is covered by the domain (False for NaN probes)."""
    inside = np.zeros(len(probe_xy), dtype=bool)
    finite = np.isfinite(probe_xy).all(axis=1)
    # A point is covered by a polygon exactly when it intersects it.
    inside[finite] = shapely.intersects_xy(domain_geom, probe_xy[finite, 0], probe_xy[finite, 1])
    return inside


def _strictly_inside(domain_geom, points_xy):
    """Return whether each point lies in the domain interior (False for NaN points)."""
    inside = np.zeros(len(points_xy), dtype=bool)
    finite = np.isfinite(points_xy).all(axis=1)
    inside[finite] = shapely.contains_xy(domain_geom, points_xy[finite, 0], points_xy[finite, 1])
    return inside


def _split_polygon_pieces(cell_poly, line):
    """Return the polygon pieces of split(cell_poly, line), covering the whole cell.

    Where the line runs along a cell face to within roundoff (a straddle
    pair's face lies on the barrier segment through the pair), split() can
    drop a piece. Snapping the cell to the line's vertices first makes the
    overlap exact, so split is retried that way when area goes missing.
    """
    tolerance = 1e-9 * cell_poly.area
    pieces = [piece for piece in split(cell_poly, line).geoms if isinstance(piece, (Polygon, MultiPolygon))]
    if abs(cell_poly.area - sum(piece.area for piece in pieces)) <= tolerance:
        return pieces
    snapped = shapely.snap(cell_poly, line, 1e-9 * cell_poly.length)
    retried = [piece for piece in split(snapped, line).geoms if isinstance(piece, (Polygon, MultiPolygon))]
    if abs(cell_poly.area - sum(piece.area for piece in retried)) <= tolerance:
        return retried
    return pieces


def _straddled_pieces(cell_poly, line, sliver_fraction=BARRIER_SLIVER_FRACTION):
    """Return the polygon pieces of a cell split by a line, or [] when the line does not straddle it."""
    pieces = _split_polygon_pieces(cell_poly, line)
    min_area = sliver_fraction * cell_poly.area
    significant = [piece for piece in pieces if piece.area > min_area]
    if len(significant) < 2:
        return []
    return [_snap_to_cell_vertices(piece, cell_poly) for piece in pieces]


def _polygonal_part(geom):
    """Return the polygons of a geometry (dropping line/point slivers from an intersection)."""
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    polygons = [part for part in shapely.get_parts(geom) if isinstance(part, Polygon)]
    return MultiPolygon(polygons) if polygons else Polygon()


def _snap_to_cell_vertices(piece, cell_poly):
    """Return a split piece with cut points that land on a cell vertex merged into that vertex.

    A barrier through (or within roundoff of) a cell vertex makes split()
    insert its own copy of the vertex next to the original, i.e. a
    zero-length edge the neighbouring cell does not share. A piece (or hole)
    smaller than the tolerance would collapse to a ring of fewer than three
    points, which GEOS rejects, so it keeps its unsnapped geometry.
    """
    tolerance = 1e-9 * cell_poly.length
    try:
        snapped = shapely.remove_repeated_points(shapely.snap(piece, cell_poly, tolerance), tolerance)
    except shapely.errors.GEOSException:
        return piece
    if not snapped.is_valid or abs(snapped.area - piece.area) > 1e-9 * cell_poly.area:
        return piece
    return snapped


def _line_segments(line):
    """Return the (start, end) coordinates of every segment of a LineString or MultiLineString."""
    parts = getattr(line, 'geoms', [line])
    starts, ends = [], []
    for part in parts:
        coords = np.asarray(part.coords, dtype=float)[:, :2]
        starts.append(coords[:-1])
        ends.append(coords[1:])
    return np.concatenate(starts), np.concatenate(ends)


def _reflect_across_nearest_segment(point_xy, line, tolerance):
    """Return the mirror of a point across the line through its nearest segment, or None if it lies on the line."""
    starts, ends = _line_segments(line)
    direction = ends - starts
    length_sq = np.einsum('ij,ij->i', direction, direction)
    valid = length_sq > 0
    if not valid.any():
        return None
    starts, direction, length_sq = starts[valid], direction[valid], length_sq[valid]
    t = np.clip(np.einsum('ij,ij->i', point_xy - starts, direction) / length_sq, 0.0, 1.0)
    nearest = int(np.argmin(np.hypot(*(starts + t[:, None] * direction - point_xy).T)))
    # Reflect across the segment's infinite line, not its closest point, so a
    # node beyond a segment end still mirrors onto the other side of the line.
    start, d = starts[nearest], direction[nearest]
    foot = start + np.dot(point_xy - start, d) / length_sq[nearest] * d
    if np.hypot(*(point_xy - foot)) <= tolerance:
        return None
    return 2.0 * foot - point_xy


def _primary_piece_position(pieces, generator):
    """Return the position of the piece that keeps the node_id: the largest covering the generator, else the largest."""
    covering = [i for i, piece in enumerate(pieces) if piece.covers(generator)]
    candidates = covering if covering else range(len(pieces))
    return max(candidates, key=lambda i: pieces[i].area)


def _fragment_merge(piece, geometries, tree, barriers):
    """Return (position, merged geometry) of the cell ``piece`` joins, or None if no cell qualifies.

    The cell must share a boundary with the piece, the union must be one
    valid polygon without holes, and no barrier may straddle the union; so
    the piece never joins a cell across a barrier (e.g. the cell it was cut
    from, unless it is a roundoff crumb). Of the cells left, the one sharing
    the longest boundary wins.

    The piece's cut points can sit a roundoff distance off the cell's face
    (a barrier vertex on a straddle pair's face), which would leave the two
    touching at a point only; snapping the cell to the piece's vertices
    first inserts them into its faces.
    """
    best = None
    best_shared = 1e-9 * piece.length
    for position in tree.query(piece, predicate='intersects'):
        cell = shapely.snap(geometries[position], piece, 1e-9 * piece.length)
        merged = cell.union(piece)
        if merged.geom_type != 'Polygon' or merged.interiors or not merged.is_valid:
            continue
        if abs(merged.area - cell.area - piece.area) > 1e-9 * merged.area:
            continue
        # Union drops the shared boundary from both perimeters.
        shared = 0.5 * (cell.length + piece.length - merged.length)
        if shared <= best_shared:
            continue
        if any(_straddled_pieces(merged, line) for line in barriers if merged.intersects(line)):
            continue
        best, best_shared = (position, merged), shared
    return best


def _clip_to_domain(gdf, domain_geom):
    """Return the rows of ``gdf`` that intersect ``domain_geom``, clipped to it, in their original order.

    Replaces ``gpd.clip``, which with pandas 1.5 / GeoPandas 0.13 writes the
    clipped geometries back onto the wrong rows (its spatial-index query
    reorders them), so cells lose their generator's x/y and node_id.
    """
    hits = np.sort(gdf.sindex.query(domain_geom, predicate='intersects'))
    clipped = gdf.iloc[hits].copy()
    clipped[clipped.geometry.name] = clipped.geometry.intersection(domain_geom)
    return clipped


def _merge_barrier_fragments(geometries, fragments, barriers):
    """Merge small barrier-split fragments into neighbouring cells; return the fragments left over.

    ``geometries`` (the grid's cell geometries, an object array) is updated
    in place. ``fragments`` are (piece, cell area, cell row) tuples. A
    fragment joins a neighbour (see _fragment_merge) when its area is below
    BARRIER_FRAGMENT_MERGE_FRACTION of the cell it was cut from. Fragments
    that only touch other fragments are retried once those have merged.
    """
    pending = [f for f in fragments if f[0].area < BARRIER_FRAGMENT_MERGE_FRACTION * f[1]]
    leftover = [f for f in fragments if f[0].area >= BARRIER_FRAGMENT_MERGE_FRACTION * f[1]]
    while pending:
        # Shapely 2.0's STRtree makes the array it is given read-only.
        tree = shapely.STRtree(geometries.copy())
        unmerged = []
        for fragment in pending:
            merge = _fragment_merge(fragment[0], geometries, tree, barriers)
            if merge is None:
                unmerged.append(fragment)
            else:
                geometries[merge[0]] = merge[1]
        if len(unmerged) == len(pending):
            break
        pending = unmerged
    kept = {id(f) for f in leftover + pending}
    return [f for f in fragments if id(f) in kept]


def _join_unmatched_to_nearest_zone(joined, pts_gdf, zones):
    """Give generators that matched no zone the nearest zone instead.

    Mesh nodes on a slanted domain edge can sit a floating-point hair outside
    every zone polygon, so the ``intersects`` join leaves them without a zone
    even though their cell lies inside the domain.
    """
    matched = joined['index_right'].notna()
    missing = pts_gdf[~pts_gdf['node_id'].isin(joined.loc[matched, 'node_id'])]
    if missing.empty or zones.empty:
        return joined
    logger.debug(f"  -> {len(missing)} generator(s) outside every zone; using the nearest zone.")
    nearest = gpd.sjoin_nearest(missing, zones, how='left')
    return pd.concat([joined[matched], nearest])


def _mirror_metadata(tags, mirrors):
    """Return node metadata rows for barrier mirror generators (never boundary-centred)."""
    return pd.DataFrame(
        {
            "node_id": tags,
            "source_x": mirrors[:, 0],
            "source_y": mirrors[:, 1],
            "boundary_centering": "clip",
            "boundary_inset": 0.0,
            "boundary_centered": False,
        }
    )


def _explode_with_unique_ids(grid_gdf):
    """Explode multipart cells into polygons, giving every non-primary part a fresh node_id."""
    exploded = grid_gdf.explode(index_parts=False)
    # GeometryCollections from clipping can carry line/point slivers; drop them.
    exploded = exploded[exploded.geom_type == 'Polygon'].reset_index(drop=True)
    duplicated = exploded['node_id'].duplicated(keep=False).to_numpy()
    if not duplicated.any():
        return exploded

    secondary = []
    for _, group in exploded[duplicated].groupby('node_id', sort=False):
        pieces = list(group.geometry)
        generator = Point(float(group['x'].iloc[0]), float(group['y'].iloc[0]))
        primary = _primary_piece_position(pieces, generator)
        secondary.extend(label for i, label in enumerate(group.index) if i != primary)

    # Python ints avoid NumPy scalar overflow for large (e.g. uint64) node IDs.
    node_ids = exploded['node_id'].to_numpy(copy=True)
    max_id = int(node_ids.max())
    new_ids = [max_id + 1 + i for i in range(len(secondary))]
    node_ids[secondary] = np.array(new_ids, dtype=node_ids.dtype)
    exploded['node_id'] = node_ids
    # Detached parts have no generator of their own, so x/y fall back to the centroid.
    centroids = exploded.geometry.iloc[secondary].centroid
    exploded.loc[secondary, 'x'] = centroids.x.to_numpy()
    exploded.loc[secondary, 'y'] = centroids.y.to_numpy()
    logger.info(f"  -> Re-numbered {len(secondary)} detached cell parts")
    return exploded


def _zone_pieces(polygons, footprints):
    """Polygon parts of ``polygons`` (zones) with the quad-buffer ``footprints`` (or None) cut out."""
    pieces = []
    for geom in polygons:
        if geom is None or geom.is_empty:
            continue
        if footprints is not None and geom.intersects(footprints):
            geom = make_valid(geom.difference(footprints))
        pieces.extend(polygon_parts(geom))
    return [piece for piece in pieces if piece.area > 0]


def _owner_polygons(nodes, free, pieces):
    """Owner polygon of each free node (the zone piece containing it) and the free mask of nodes that have one.

    Returns (owners (len(nodes),) object array, None for fixed nodes; free).
    A free node in no piece (or on a piece boundary) is made fixed.
    """
    owners = np.full(len(nodes), None, dtype=object)
    free = np.asarray(free, dtype=bool).copy()
    free_idx = np.flatnonzero(free)
    if len(free_idx) == 0 or not pieces:
        free[:] = False
        return owners, free
    piece_array = np.empty(len(pieces), dtype=object)
    piece_array[:] = pieces
    point_pos, piece_pos = shapely.STRtree(piece_array).query(
        shapely.points(nodes[free_idx]), predicate='within'
    )
    # Clean polygons do not overlap; keep the first piece should two touch.
    point_pos, first = np.unique(point_pos, return_index=True)
    owners[free_idx[point_pos]] = piece_array[piece_pos[first]]
    free[free_idx] = False
    free[free_idx[point_pos]] = True
    return owners, free


class VoronoiTessellator:
    def __init__(
        self,
        mesh_generator,
        conceptual_mesh,
        clip_to_boundary=True,
        boundary_centering="clip",
        boundary_inset_fraction=0.25,
        boundary_corner_angle=135.0,
        boundary_tolerance=None,
        lloyd_iterations=0,
        lloyd_damping=1.0,
        lloyd_tolerance=1e-3,
        lloyd_size_smoothing=5,
    ):
        """
        Initializes the Voronoi tessellator.

        This class takes a triangular mesh (typically from `MeshGenerator`) and
        computes its dual: a Voronoi diagram. The resulting grid of polygonal
        cells is suitable for use in cell-centered finite volume models.

        Args:
            mesh_generator (MeshGenerator): An instance of the mesh generator that
                contains the generated triangular mesh nodes.
            conceptual_mesh (ConceptualMesh): The conceptual model, used for CRS,
                domain boundaries, and feature information.
            clip_to_boundary (bool): If True, the final Voronoi grid will be
                clipped to the domain boundary defined in the conceptual model.
            boundary_centering (str): ``"clip"`` keeps the historical behavior.
                ``"inset_mirror"`` shifts boundary generators inward and adds
                mirrored outside ghosts so boundary-cell centers move off the
                clipped face. On Gmsh meshes this lowers the median
                centroid-to-centroid orthogonality error of boundary
                connections from about 6 to under 2 degrees. Nodes at sharp
                corners stay in place, so the worst connections next to
                corners do not improve.
            boundary_inset_fraction (float): Fraction of local boundary-node
                spacing used for the inward shift in ``"inset_mirror"`` mode.
                A boundary cell reaches about halfway to the first interior
                row, so its generator sits near its centroid when the inset is
                about a third of that row's depth. On Gmsh meshes the row lies
                about 0.87 x spacing inside, so 0.2-0.3 works best. At 0.5
                boundary nodes get closer to the interior row than to each
                other and orthogonality ends up worse than ``"clip"``.
            boundary_corner_angle (float): Boundary vertices with a local angle
                below this value are treated as sharp corners and left unchanged.
            boundary_tolerance (float, optional): Distance tolerance used to
                classify generator nodes as boundary nodes.
            lloyd_iterations (int): Number of Lloyd relaxation passes over the
                generators before the Voronoi grid is built (0, the default,
                turns it off). Each pass moves every free generator towards
                the centroid of its Voronoi cell, so generators end up near
                their cell centroids (lower ``drift_ratio`` in
                ``vorflow.utils.calculate_mesh_quality``). The centroids are
                weighted by the density h**-4, where h is the local mesh size
                interpolated from ``MeshGenerator.node_sizes`` after
                ``lloyd_size_smoothing``; this keeps the mesh grading (the
                cell sizes around refined features) instead of evening out
                the cell sizes. Only free nodes move: mesh
                nodes inside the embedded polygon surfaces. Nodes on the domain
                and zone boundaries, embedded points (and their hex-ring
                seeds), embedded and barrier lines (including straddle pairs)
                and the quad-buffer strips and bands stay fixed. A move is
                rejected if it would leave the node's zone or cross an embedded
                line. With Lloyd on, the cells are no longer the exact dual of
                the triangle mesh (``MeshGenerator.get_element_grid()``) and
                the grid gets a ``lloyd_shift`` column; ``lloyd_report`` holds
                the run summary: passes run (``iterations``), the last pass's
                largest relative residual (``max_rel_shift``, as for
                ``lloyd_tolerance``), moves rejected (``rejected``) and the
                number of free generators (``n_free``).

                Lloyd is opt-in and mainly for models that use cell centroids
                as cell centres, or for visually more regular cells. Head
                accuracy is set mainly by ``growth_factor`` (how fine the
                cells are relative to the curvature of the head), not by cell
                shape. In a steady radial-flow (Thiem) test with generator
                centres (rhugman/vorflow#31), 100 passes lowered the head
                RMSE by at most about 5%. Each pass costs about one Voronoi
                diagram of all nodes plus the weighted centroids of the free
                cells; 100 passes took 5-10x the time of meshing and
                tessellating once. Use 100 or more passes, together with
                ``hex_ring=True`` on refined points. Avoid about 20 passes: a
                partly relaxed grid passes through near-cocircular generators,
                and the number of very short cell faces peaks there.
                Requires a mesh generator that ran ``MeshGenerator.generate()``
                before this tessellator was constructed.
            lloyd_damping (float): Fraction of the way each free generator
                moves towards its centroid per pass, in (0, 1]. Default 1.0.
            lloyd_tolerance (float): Stop early after a pass in which the
                largest distance from an accepted free generator to its
                weighted centroid, relative to the local mesh size, is below
                this (independent of ``lloyd_damping``). Default 1e-3.
                Graded meshes usually run all ``lloyd_iterations`` passes.
            lloyd_size_smoothing (int): Passes of smoothing applied to the
                per-node sizes behind the Lloyd density (only used with
                ``lloyd_iterations > 0``). Gmsh's node sizes jitter by about
                10% between neighbouring nodes, i.e. +-30-45% in h**-4, and
                Lloyd would otherwise converge to that noise. Each pass moves
                the log size of every free node halfway to the mean over its
                mesh-edge neighbours (``MeshGenerator.node_edges``); fixed
                nodes keep their size and anchor the grading. Default 5;
                from 10-20 passes on it starts to flatten steep grading. 0
                uses ``node_sizes`` unchanged (the 0.2.0 behaviour).
        """
        if boundary_centering not in {"clip", "inset_mirror"}:
            raise ValueError("boundary_centering must be either 'clip' or 'inset_mirror'.")
        if boundary_inset_fraction <= 0:
            raise ValueError("boundary_inset_fraction must be positive.")
        if boundary_corner_angle <= 0 or boundary_corner_angle >= 180:
            raise ValueError("boundary_corner_angle must be between 0 and 180 degrees.")
        if boundary_tolerance is not None and boundary_tolerance < 0:
            raise ValueError("boundary_tolerance must be non-negative when provided.")
        _lloyd.validate_settings(lloyd_iterations, lloyd_damping, lloyd_tolerance, prefix="lloyd_",
                                 size_smoothing=lloyd_size_smoothing)

        self.mg = mesh_generator
        self.cm = conceptual_mesh
        self.voronoi_gdf = None
        self.final_grid = None
        self.n_barrier_mirrors = 0
        self.nodes = mesh_generator.nodes
        self.node_tags = mesh_generator.node_tags
        # Lloyd inputs, captured with the nodes so they stay paired with them;
        # None on mesh generators that do not provide them.
        self.node_is_free = getattr(mesh_generator, 'node_is_free', None)
        self.node_sizes = getattr(mesh_generator, 'node_sizes', None)
        self.node_edges = getattr(mesh_generator, 'node_edges', None)
        self.buffer_footprints = getattr(mesh_generator, 'buffer_footprints', None)
        self.zones_gdf = mesh_generator.zones_gdf
        self.clip_to_boundary = clip_to_boundary
        self.boundary_centering = boundary_centering
        self.boundary_inset_fraction = float(boundary_inset_fraction)
        self.boundary_corner_angle = float(boundary_corner_angle)
        self.boundary_tolerance = boundary_tolerance
        self.lloyd_iterations = int(lloyd_iterations)
        self.lloyd_damping = float(lloyd_damping)
        self.lloyd_tolerance = float(lloyd_tolerance)
        self.lloyd_size_smoothing = int(lloyd_size_smoothing)
        # {"iterations", "max_rel_shift", "rejected", "n_free"} after a Lloyd
        # run; max_rel_shift is the largest residual |centroid - node| / local
        # size over the accepted moves of the last pass (see _lloyd.relax).
        self.lloyd_report = None

    def _lloyd_mesh_inputs(self, n_nodes):
        """(node_is_free, node_sizes, buffer_footprints) captured from the mesh generator, checked against ``n_nodes``."""
        node_is_free, node_sizes = self.node_is_free, self.node_sizes
        if node_is_free is None or node_sizes is None:
            raise ValueError(
                "lloyd_iterations > 0 needs MeshGenerator.node_is_free and node_sizes; "
                "run MeshGenerator.generate() before tessellating."
            )
        node_is_free = np.asarray(node_is_free, dtype=bool)
        node_sizes = np.asarray(node_sizes, dtype=float)
        if node_is_free.shape != (n_nodes,) or node_sizes.shape != (n_nodes,):
            raise ValueError(
                "MeshGenerator.node_is_free and node_sizes must be aligned with its nodes "
                f"({n_nodes}); got {node_is_free.shape} and {node_sizes.shape}. "
                "Re-run MeshGenerator.generate() before tessellating."
            )
        return node_is_free, node_sizes, self.buffer_footprints

    def _lloyd_node_edges(self, n_nodes):
        """Mesh edges (m, 2) captured from the mesh generator, checked against ``n_nodes``."""
        if self.node_edges is None:
            raise ValueError(
                "lloyd_size_smoothing > 0 needs MeshGenerator.node_edges; run MeshGenerator.generate() "
                "before tessellating, or set lloyd_size_smoothing=0."
            )
        edges = np.asarray(self.node_edges)
        if edges.size and (edges.ndim != 2 or edges.shape[1] != 2 or edges.min() < 0 or edges.max() >= n_nodes):
            raise ValueError(
                f"MeshGenerator.node_edges must be (m, 2) positions into its nodes ({n_nodes}). "
                "Re-run MeshGenerator.generate() before tessellating."
            )
        return edges.reshape(-1, 2)

    def _lloyd_sizes(self, node_sizes, free, n_nodes):
        """Per-node sizes for the Lloyd density: ``node_sizes`` after ``lloyd_size_smoothing`` passes over the free nodes."""
        if self.lloyd_size_smoothing == 0:
            return node_sizes
        logger.info(f"  -> Lloyd density from node sizes smoothed with {self.lloyd_size_smoothing} passes")
        return _lloyd.smooth_log_sizes(node_sizes, free, self._lloyd_node_edges(n_nodes), self.lloyd_size_smoothing)

    def _lloyd_constraint_lines(self):
        """Union of the embedded clean lines (barriers and straddle lines included), or None."""
        lines = self.cm.clean_lines
        if lines is None or lines.empty:
            return None
        if 'embed' in lines.columns:
            # Missing embed values default to embedded, matching MeshGenerator.
            embedded = lines['embed'].map(lambda value: True if pd.isna(value) else bool(value))
            lines = lines[embedded.astype(bool)]
        geoms = [geom for geom in lines.geometry if geom is not None and not geom.is_empty]
        return unary_union(geoms) if geoms else None

    def _relax_generators(self, nodes):
        """Density-weighted Lloyd relaxation of the free mesh nodes (see ``lloyd_iterations``); returns the moved nodes."""
        node_is_free, node_sizes, footprints = self._lloyd_mesh_inputs(len(nodes))
        pieces = _zone_pieces(self._embedded_polygons().geometry, footprints)
        owners, free = _owner_polygons(nodes, node_is_free, pieces)
        report = {"iterations": 0, "max_rel_shift": 0.0, "rejected": 0}
        if free.any():
            logger.info(f"Lloyd relaxation of {int(free.sum())} free generators...")
            sizes = self._lloyd_sizes(node_sizes, free, len(nodes))
            nodes, report = _lloyd.relax(
                nodes,
                free,
                _lloyd.size_interpolator(nodes, sizes),
                self._domain_geometry(),
                owners,
                self._lloyd_constraint_lines(),
                self.lloyd_iterations,
                damping=self.lloyd_damping,
                tolerance=self.lloyd_tolerance,
            )
            logger.info(
                f"  -> {report['iterations']} passes, last max residual |centroid - node| "
                f"{report['max_rel_shift']:.2e} x local size, {report['rejected']} moves rejected"
            )
        self.lloyd_report = {**report, "n_free": int(free.sum())}
        return nodes

    def _embedded_polygons(self):
        """Return the clean polygons that define the domain and zones (embed=True)."""
        polygons = self.cm.clean_polygons
        if 'embed' not in polygons.columns:
            return polygons
        # Missing embed values default to embedded, matching MeshGenerator.
        embedded = polygons['embed'].map(lambda value: True if pd.isna(value) else bool(value))
        return polygons[embedded.astype(bool)]

    def _domain_geometry(self):
        """Return the current meshing domain geometry."""
        polygons = self._embedded_polygons()
        if not polygons.empty:
            domain_geom = unary_union(polygons.geometry)
            if not domain_geom.is_valid:
                domain_geom = make_valid(domain_geom)
            return domain_geom
        if hasattr(self.cm, 'domain_boundary') and self.cm.domain_boundary:
            domain_geom = self.cm.domain_boundary
            if not domain_geom.is_valid:
                domain_geom = make_valid(domain_geom)
            return domain_geom
        return None

    def _boundary_tolerance(self, nodes, domain_geom):
        if self.boundary_tolerance is not None:
            return float(self.boundary_tolerance)
        minx, miny, maxx, maxy = domain_geom.bounds
        domain_scale = max(maxx - minx, maxy - miny, 1.0)
        node_scale = 1.0
        if len(nodes) > 0:
            node_scale = max(np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]), 1.0)
        return max(domain_scale, node_scale) * 1e-8

    def _prepare_boundary_centered_nodes(self, nodes, node_tags):
        """
        Shift non-corner boundary nodes inward and add mirrored outside ghosts.

        A node whose mirror ghost would land inside the domain (e.g. across
        a narrow hole) keeps its original position and gets no ghost.

        Returns prepared nodes, prepared tags, and a metadata frame keyed by
        node_id. Tags only cover real nodes; appended ghosts receive -1 in
        _build_raw_voronoi.
        """
        if self.boundary_centering != "inset_mirror":
            metadata = pd.DataFrame(
                {
                    "node_id": node_tags,
                    "source_x": nodes[:, 0],
                    "source_y": nodes[:, 1],
                    "boundary_centering": "clip",
                    "boundary_inset": 0.0,
                    "boundary_centered": False,
                }
            )
            return nodes, node_tags, np.empty((0, 2)), metadata

        domain_geom = self._domain_geometry()
        if domain_geom is None or domain_geom.is_empty:
            raise RuntimeError(
                "boundary_centering='inset_mirror' requires a generated domain geometry."
            )

        boundary = domain_geom.boundary
        tolerance = self._boundary_tolerance(nodes, domain_geom)
        shapely.prepare(boundary)
        shapely.prepare(domain_geom)
        points = shapely.points(nodes)

        boundary_indices = np.flatnonzero(shapely.dwithin(boundary, points, tolerance))
        spacing = _boundary_node_spacing(nodes, boundary_indices)
        candidates = np.flatnonzero(spacing > 0)

        vertices, vertex_angles = _ring_vertex_angles(domain_geom)
        corner_angles = _ring_angles_at_points(nodes[candidates], vertices, vertex_angles, tolerance)
        # NaN (no matching ring vertex) compares False, so the node is not a corner.
        candidates = candidates[~(corner_angles < self.boundary_corner_angle)]

        inset = spacing[candidates] * self.boundary_inset_fraction
        tangents = _boundary_tangents(boundary, points[candidates], spacing[candidates])
        normals = _inward_normals(domain_geom, nodes[candidates], tangents, inset)
        ghosts = nodes[candidates] - normals * inset[:, None]
        has_normal = ~np.isnan(normals).any(axis=1)
        # In narrow holes or concave spots the mirror ghost can land back
        # inside the domain; keep those nodes at their original position.
        ghost_inside = _strictly_inside(domain_geom, ghosts)
        keep = has_normal & ~ghost_inside
        if ghost_inside.any():
            logger.info(f"  -> Left {int(ghost_inside.sum())} boundary nodes in place: mirror ghost inside domain")

        centered_indices = candidates[keep]
        prepared = nodes.astype(float, copy=True)
        prepared[centered_indices] = nodes[centered_indices] + normals[keep] * inset[keep, None]

        centered = np.zeros(len(nodes), dtype=bool)
        centered[centered_indices] = True
        node_inset = np.zeros(len(nodes), dtype=float)
        node_inset[centered_indices] = inset[keep]
        metadata = pd.DataFrame(
            {
                "node_id": node_tags,
                "source_x": nodes[:, 0].astype(float),
                "source_y": nodes[:, 1].astype(float),
                "boundary_centering": np.where(centered, "inset_mirror", "clip").astype(object),
                "boundary_inset": node_inset,
                "boundary_centered": centered,
            }
        )
        ghosts = ghosts[keep] if keep.any() else np.empty((0, 2))
        return prepared, node_tags, ghosts, metadata

    def _build_raw_voronoi(self, nodes, node_tags):
        """
        Computes the mathematical Voronoi diagram from a set of generator points.

        This method uses `scipy.spatial.Voronoi` to calculate the unbounded
        Voronoi diagram. It filters out invalid or infinite regions and returns
        the finite polygons as a GeoDataFrame.

        Args:
            nodes (np.ndarray): An array of (x, y) coordinates for the generator points.
            node_tags (np.ndarray): An array of IDs corresponding to each node.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the Voronoi polygons, with
                columns for the generator's node_id, x, and y coordinates.
        """
        if len(nodes) < 3:
            warnings.warn(
                "Not enough generator nodes (<3) to build a Voronoi diagram; "
                "returning an empty grid. Check that meshing succeeded and "
                "the domain is not degenerate.",
                stacklevel=2,
            )
            return gpd.GeoDataFrame()

        # Qhull loses precision on coordinates with a large offset relative
        # to their extent (e.g. UTM northings around a small domain) and
        # returns overlapping cells, so build the diagram about the bbox
        # centre and shift the vertices back afterwards.
        nodes = np.asarray(nodes, dtype=float)
        origin = (nodes.min(axis=0) + nodes.max(axis=0)) / 2.0
        vor = Voronoi(nodes - origin)
        vertices = vor.vertices + origin
        polygons = []
        ids = []
        gen_x = []
        gen_y = []
        
        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            # Skip infinite regions (those containing -1).
            if not region or -1 in region:
                continue
            
            verts = vertices[region]
            poly = Polygon(verts)
            
            if poly.is_valid:
                polygons.append(poly)
                # Store the coordinates of the generator point for this cell.
                gen_x.append(nodes[i][0])
                gen_y.append(nodes[i][1])
                
                # Assign the node tag (ID) to the cell. Ghost nodes (used to
                # bound the diagram) will not have a tag.
                if i < len(node_tags):
                    ids.append(node_tags[i])
                else:
                    ids.append(-1) 
        
        gdf = gpd.GeoDataFrame(
            {'node_id': ids, 'x': gen_x, 'y': gen_y}, 
            geometry=polygons, 
            crs=self.cm.crs
        )
        return gdf

    def _barrier_lines(self):
        """Return the geometries of the clean lines marked is_barrier=True."""
        lines = self.cm.clean_lines
        if lines is None or lines.empty or 'is_barrier' not in lines.columns:
            return []
        mask = lines['is_barrier'].fillna(False).astype(bool)
        return [geom for geom in lines.loc[mask].geometry if geom is not None and not geom.is_empty]

    def _barrier_mirror_points(self, raw_gdf, generators, domain_geom):
        """
        Return mirror generators for nodes whose Voronoi cell a barrier crosses.

        The straddle mechanism in MeshGenerator puts node pairs on either
        side of a barrier so that their shared Voronoi face lies on it. Other
        nodes near the line (mesh nodes between the pairs, nodes of a line
        crossing the barrier) have no partner, and their cells straddle it.
        Reflecting such a node across the barrier gives it that partner: the
        bisector of the two is the barrier line, so after re-tessellation
        both cells stop at the line and every face stays a Voronoi bisector.

        A node on the line and a mirror within BARRIER_MIRROR_MERGE_FRACTION
        of an existing generator are skipped; the post-hoc split in
        _enforce_barriers handles what is left. A mirror outside the domain
        (a boundary node near a barrier end) is kept when the piece beyond
        the line is large enough to stay a cell of its own after that split
        (see BARRIER_FRAGMENT_MERGE_FRACTION): every point of the piece is
        nearer to the mirror than to any other node, so the mirror's clipped
        cell covers it. That cell's x/y then lies outside the cell, as with
        the engine's straddle pair at a barrier end. A smaller piece would
        leave the mirror a corner cell of a few thousandths of lc^2; the
        split merges it into a neighbour instead.
        """
        barriers = self._barrier_lines()
        cells = raw_gdf[raw_gdf['node_id'] != -1]
        if not barriers or cells.empty:
            return np.empty((0, 2))

        scale = max(float(np.ptp(generators, axis=0).max()), 1.0)
        tolerance = scale * 1e-8
        tree = cKDTree(generators)
        mirrors = []
        for line in barriers:
            for position in cells.sindex.query(line, predicate='intersects'):
                cell = cells.iloc[position]
                # A barrier ending on the domain boundary ends inside the raw
                # cell of a boundary node, so test the cell as it will be clipped.
                cell_geom = cell.geometry
                if domain_geom is not None:
                    cell_geom = _polygonal_part(cell_geom.intersection(domain_geom))
                if cell_geom.is_empty:
                    continue
                pieces = _straddled_pieces(cell_geom, line)
                if not pieces:
                    continue
                node_xy = np.array([cell['x'], cell['y']], dtype=float)
                mirror = _reflect_across_nearest_segment(node_xy, line, tolerance)
                if mirror is None:
                    continue
                if domain_geom is not None and not domain_geom.covers(Point(mirror)):
                    primary = _primary_piece_position(pieces, Point(node_xy))
                    beyond = cell_geom.area - pieces[primary].area
                    if beyond < BARRIER_FRAGMENT_MERGE_FRACTION * cell_geom.area:
                        continue
                merge_distance = BARRIER_MIRROR_MERGE_FRACTION * np.hypot(*(mirror - node_xy))
                if tree.query(mirror)[0] < merge_distance:
                    continue
                if any(np.hypot(*(mirror - other)) < merge_distance for other in mirrors):
                    continue
                mirrors.append(mirror)
        return np.array(mirrors) if mirrors else np.empty((0, 2))

    def _build_barrier_conforming_voronoi(self, nodes, tags, boundary_ghosts, far_ghosts, node_metadata):
        """
        Build the raw Voronoi diagram, adding barrier mirror generators until no cell straddles a barrier.

        Mirror generators are real nodes: they get fresh node_ids above the
        current maximum, x/y at the mirror point, and a row in node_metadata.
        Their count is stored in ``self.n_barrier_mirrors``.

        Returns:
            tuple: (raw Voronoi GeoDataFrame, node metadata including mirrors).
        """
        domain_geom = self._domain_geometry() if self.clip_to_boundary else None
        raw_gdf = self._build_raw_voronoi(np.vstack([nodes, boundary_ghosts, far_ghosts]), tags)
        self.n_barrier_mirrors = 0
        if raw_gdf.empty:
            return raw_gdf, node_metadata

        for _ in range(BARRIER_MIRROR_PASSES):
            generators = np.vstack([nodes, boundary_ghosts])
            mirrors = self._barrier_mirror_points(raw_gdf, generators, domain_geom)
            if len(mirrors) == 0:
                break
            # Python ints avoid NumPy scalar overflow for large (e.g. uint64) node IDs.
            first_id = int(np.max(tags)) + 1 if len(tags) else 0
            new_tags = np.array([first_id + i for i in range(len(mirrors))], dtype=tags.dtype)
            nodes = np.vstack([nodes, mirrors])
            tags = np.concatenate([tags, new_tags])
            node_metadata = pd.concat(
                [node_metadata, _mirror_metadata(new_tags, mirrors)], ignore_index=True
            )
            self.n_barrier_mirrors += len(mirrors)
            raw_gdf = self._build_raw_voronoi(np.vstack([nodes, boundary_ghosts, far_ghosts]), tags)

        if self.n_barrier_mirrors:
            logger.info(f"  -> Added {self.n_barrier_mirrors} barrier mirror generators")
        return raw_gdf, node_metadata

    def _enforce_barriers(self, grid_gdf):
        """
        Splits Voronoi cells that are still straddled by barrier lines.

        Every line marked `is_barrier=True` is checked, whatever its meshing
        method. Barrier mirror generators (_barrier_mirror_points) already
        stop most cells at the line; what is left here is mainly nodes that
        lie on the line itself (quad_buffer_thickness=2 puts a node row on
        it), cells the mirror passes did not finish, and the bulges of a
        curved barrier across its straddle pairs' faces (chords of the
        curve). A cell is split only when the line leaves at least two
        pieces larger than BARRIER_SLIVER_FRACTION of its area; cells the
        line merely runs along are left untouched. Cut points within
        roundoff of a cell vertex are merged into it, so the split adds no
        zero-length edges.

        The piece covering the generator point (else the largest piece)
        keeps the original node_id and x/y. A piece smaller than
        BARRIER_FRAGMENT_MERGE_FRACTION of the cell joins the neighbouring
        cell on its side of the barrier (_merge_barrier_fragments), so the
        barrier face follows the line instead of leaving a sliver cell. The
        other pieces get new, unique IDs above the current maximum. These
        fragments have no generator of their own, so their x/y are set to
        the fragment centroid and do not denote a mesh node; their
        connections are not orthogonal.

        Args:
            grid_gdf (gpd.GeoDataFrame): The current Voronoi grid.

        Returns:
            gpd.GeoDataFrame: An updated grid with cells split along barrier lines.
        """
        if self.cm.clean_lines.empty:
            return grid_gdf

        # fillna keeps the old `== True` semantics: rows with a missing
        # is_barrier value are treated as non-barriers, not as errors.
        mask_barrier = self.cm.clean_lines['is_barrier'].fillna(False).astype(bool)
        barriers_to_cut = self.cm.clean_lines[mask_barrier]

        if barriers_to_cut.empty:
            return grid_gdf

        logger.info(f"Enforcing Barrier Cuts on {len(barriers_to_cut)} lines...")

        current_grid = grid_gdf
        barrier_lines = list(barriers_to_cut.geometry)

        # Keep the caller's integer dtype while assigning fragment IDs with
        # Python integers, avoiding NumPy scalar arithmetic during increments.
        node_id_dtype = grid_gdf['node_id'].dtype
        max_id = int(grid_gdf['node_id'].max())
        n_merged = 0

        for idx, row in barriers_to_cut.iterrows():
            line = row.geometry

            # Use a spatial index to quickly find cells that might intersect the line.
            possible_matches_index = list(current_grid.sindex.query(line, predicate='intersects'))
            candidate_cells = current_grid.iloc[possible_matches_index]

            primary_rows = []
            fragments = []
            cells_to_remove_indices = []

            for cell_idx, cell_row in candidate_cells.iterrows():
                cell_poly = cell_row.geometry

                if not cell_poly.intersects(line):
                    continue

                try:
                    pieces = _straddled_pieces(cell_poly, line)
                    if not pieces:
                        continue

                    generator = Point(float(cell_row['x']), float(cell_row['y']))
                    primary = _primary_piece_position(pieces, generator)
                    # Fresh IDs follow descending area, as before.
                    others = sorted(
                        (piece for i, piece in enumerate(pieces) if i != primary),
                        key=lambda p: p.area,
                        reverse=True,
                    )

                    primary_row = cell_row.copy()
                    primary_row.geometry = pieces[primary]
                    primary_rows.append(primary_row)
                    fragments.extend((piece, cell_poly.area, cell_row) for piece in others)
                    cells_to_remove_indices.append(cell_idx)

                except Exception as e:
                    logger.warning(f"Warning: Failed to split cell {cell_row['node_id']}: {e}")

            if not cells_to_remove_indices:
                continue

            # Rebuild the grid with the primary pieces, then let small
            # fragments join a neighbour on their side of the barrier.
            current_grid = pd.concat(
                [current_grid.drop(cells_to_remove_indices), gpd.GeoDataFrame(primary_rows, crs=current_grid.crs)],
                ignore_index=True,
            )
            geometries = current_grid.geometry.to_numpy().copy()
            leftover = _merge_barrier_fragments(geometries, fragments, barrier_lines)
            n_merged += len(fragments) - len(leftover)
            current_grid[current_grid.geometry.name] = gpd.GeoSeries(
                geometries, index=current_grid.index, crs=current_grid.crs
            )

            new_rows = []
            for piece, _, cell_row in leftover:
                max_id += 1
                new_row = cell_row.copy()
                new_row.geometry = piece
                new_row['node_id'] = max_id
                new_row['x'] = piece.centroid.x
                new_row['y'] = piece.centroid.y
                new_rows.append(new_row)
            if new_rows:
                current_grid = pd.concat(
                    [current_grid, gpd.GeoDataFrame(new_rows, crs=current_grid.crs)], ignore_index=True
                )

        if n_merged:
            logger.info(f"  -> Merged {n_merged} barrier fragments into the neighbouring cell on their side")
        current_grid['node_id'] = current_grid['node_id'].astype(node_id_dtype)
        return current_grid

    def generate(self):
        """
        Executes the full Voronoi tessellation workflow.

        This method orchestrates the process of:
        1. With ``lloyd_iterations > 0``, moving the free generators towards
           their density-weighted cell centroids (Lloyd relaxation).
        2. With ``boundary_centering="inset_mirror"``, shifting boundary
           generators inward and adding their mirrored outside ghosts.
        3. Adding "ghost" nodes to create a bounded Voronoi diagram.
        4. Computing the raw Voronoi polygons, adding a mirror generator
           across each barrier for every node whose cell straddles it.
        5. Clipping the grid to the model domain.
        6. Assigning zone IDs to cells based on their generator point location.
        7. Enforcing barrier lines by splitting cells that still straddle them.
        8. Merging cell vertices a roundoff distance apart, so no cell has
           a zero-length edge.
        9. Calculating final cell properties. After Lloyd relaxation this
           includes ``lloyd_shift``: how far each mesh node moved (0 for fixed
           nodes; NaN for cells without a mesh node of their own: barrier
           mirrors, barrier fragments and detached cell parts).

        Returns:
            gpd.GeoDataFrame: The final, clean Voronoi grid.
        """
        if self.nodes is None or len(self.nodes) == 0:
            warnings.warn(
                "No nodes found in MeshGenerator; returning an empty grid. "
                "Did MeshGenerator.generate() run successfully?",
                stacklevel=2,
            )
            return gpd.GeoDataFrame()
        
        logger.info(f"Extracting {len(self.nodes)} Nodes from Gmsh...")
        nodes, tags = self.nodes, self.node_tags
        nodes = np.asarray(nodes, dtype=float)
        tags = np.asarray(tags)
        if len(tags) != len(nodes):
            raise ValueError("MeshGenerator nodes and node_tags must have the same length.")

        self.lloyd_report = None
        lloyd_shift = None
        if self.lloyd_iterations > 0:
            relaxed = self._relax_generators(nodes)
            lloyd_shift = pd.Series(np.hypot(*(relaxed - nodes).T), index=tags)
            nodes = relaxed

        nodes, tags, boundary_ghost_nodes, node_metadata = self._prepare_boundary_centered_nodes(nodes, tags)
        
        # To create a bounded Voronoi diagram from a finite set of points, a common
        # technique is to add "ghost" nodes far outside the area of interest. The
        # large, unwanted cells generated by these ghosts can then be clipped away.
        minx, miny = np.min(nodes, axis=0)
        maxx, maxy = np.max(nodes, axis=0)
        w, h = maxx - minx, maxy - miny
        buffer = max(w, h) * 10
        
        ghost_nodes = np.array([
            [minx - buffer, miny - buffer],
            [maxx + buffer, miny - buffer],
            [maxx + buffer, maxy + buffer],
            [minx - buffer, maxy + buffer]
        ])
        
        logger.info("Computing Mathematical Voronoi...")
        raw_gdf, node_metadata = self._build_barrier_conforming_voronoi(
            nodes, tags, boundary_ghost_nodes, ghost_nodes, node_metadata
        )
        logger.info(f"  -> Raw Polygons: {len(raw_gdf)}")
        
        # Remove the cells generated by the ghost nodes.
        raw_gdf = raw_gdf[raw_gdf['node_id'] != -1]
        logger.info(f"  -> After Ghost Filter: {len(raw_gdf)}")
        
        if raw_gdf.crs is None and self.cm.crs:
            raw_gdf.set_crs(self.cm.crs, inplace=True)

        
        if self.clip_to_boundary:
            logger.info("Clipping to Domain Boundary...")
            domain_geom = self._domain_geometry()
            if domain_geom is None:
                warnings.warn(
                    "No domain geometry found (no polygons); returning an "
                    "empty grid. Add at least one embedded polygon to the "
                    "ConceptualMesh, or use clip_to_boundary=False.",
                    stacklevel=2,
                )
                return gpd.GeoDataFrame()

            bounded_voronoi = _clip_to_domain(raw_gdf, domain_geom)
            logger.info(f"  -> After Domain Clip: {len(bounded_voronoi)}")
        
            if len(bounded_voronoi) == 0:
                logger.warning("Warning: Clipping resulted in 0 cells. Check CRS or Domain Box.")
                return bounded_voronoi
        else:
            bounded_voronoi = raw_gdf

        logger.info("Enforcing Hydrogeological Zones (Optimization: Point Sampling)...")
        # Field-only (embed=False) polygons only size the mesh; they must not
        # stamp their zone_id onto cells.
        zones = self._embedded_polygons()[['geometry', 'zone_id', 'z_order']]

        # To assign a zone ID to each Voronoi cell, we perform a spatial join
        # between the cell's generator point and the zone polygons. This is much
        # faster than doing a polygon-on-polygon overlay.
        
        # 1. Create a temporary GeoDataFrame of the generator points.
        pts_gdf = gpd.GeoDataFrame(
            {'node_id': bounded_voronoi['node_id']},
            geometry=gpd.points_from_xy(bounded_voronoi.x, bounded_voronoi.y),
            crs=bounded_voronoi.crs
        )
        
        # 2. Spatially join the points to the zones.
        joined = gpd.sjoin(pts_gdf, zones, how='left', predicate='intersects')
        joined = _join_unmatched_to_nearest_zone(joined, pts_gdf, zones)

        # 3. If a point falls on a boundary between zones, it may have multiple
        # matches. We use the `z_order` from the conceptual model to pick the
        # highest-priority zone.
        sort_columns, ascending = ['node_id'], [True]
        if 'z_order' in joined.columns:
            sort_columns += ['z_order', 'index_right']
            ascending += [False, True]
        else:
            sort_columns += ['index_right']
            ascending += [True]
        joined = joined.sort_values(sort_columns, ascending=ascending, kind='mergesort')
        
        joined = joined.drop_duplicates(subset='node_id')
        
        # 4. Merge the zone information back into the main grid.
        zoned_grid = bounded_voronoi.merge(
            joined[['node_id', 'zone_id', 'z_order']],
            on='node_id',
            how='left'
        )
        if self.boundary_centering == "inset_mirror":
            zoned_grid = zoned_grid.merge(
                node_metadata,
                on="node_id",
                how="left",
            )
        
        logger.info(f"  -> Zones Assigned: {len(zoned_grid)}")
        
        # Clipping can create MultiPolygons; explode them into single parts
        # while keeping node_id unique (it is the merge key for zones).
        zoned_grid = _explode_with_unique_ids(zoned_grid)

        # Enforce barriers by splitting cells.
        self.final_grid = self._enforce_barriers(zoned_grid)
        logger.info(f"  -> After Barrier Cuts: {len(self.final_grid)}")

        # Final cleanup after potential splits.
        self.final_grid = _explode_with_unique_ids(self.final_grid)
        self.final_grid = _merge_close_vertices(self.final_grid)
        if lloyd_shift is not None:
            # Cells with a node_id that is no mesh node get NaN.
            self.final_grid['lloyd_shift'] = self.final_grid['node_id'].map(lloyd_shift)

        # The 'x' and 'y' columns should always refer to the generator point
        # coordinates, which are essential for quality analysis. We add separate
        # columns for the geometric centroid of the final cell.
        self.final_grid['centroid_x'] = self.final_grid.geometry.centroid.x
        self.final_grid['centroid_y'] = self.final_grid.geometry.centroid.y
        
        logger.info(f"Final Voronoi Grid Generated: {len(self.final_grid)} cells.")
        return self.final_grid

    def export_to_shapefile(self, filepath):
        if self.final_grid is not None and not self.final_grid.empty:
            self.final_grid.to_file(filepath)
            logger.info(f"Saved to {filepath}")
        else:
            logger.info("No grid to export.")
