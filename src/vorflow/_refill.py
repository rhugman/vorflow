"""Re-mesh the inside of a barrier outline so the new cells share the outline's vertices.

Pure geometry for :func:`vorflow.disu.refill`. The outline nodes are the
vertices that the base grid has on the outline. Each outline segment gets a
pair of generators mirrored across it, on circles centred at its two end
nodes (the empty-circle rule of the exact barriers, Berge et al. 2019): with
no other generator inside a circle, the segment is exactly the Voronoi face
between the pair. The inside generator of each pair is kept as a cell, the
outside one only shapes that face, and new generators fill the rest of the
polygon outside the circles. Where a long segment's circle would hold a
neighbouring segment's generator, the long segment is halved.
"""

from __future__ import annotations

import numpy as np
import shapely
from scipy.spatial import Voronoi, cKDTree

RADIUS_FACTOR = 0.6     # circle radius / longer adjacent segment; must exceed 0.5
CLEARANCE = 1.05        # fill nodes must lie this much outside every circle
INSIDE_EPS = 1e-9


def outline_nodes(geoms, ring, tolerance):
    """Vertices of ``geoms`` on ``ring``, ordered along it (the ring's own vertices included)."""
    coords = shapely.get_coordinates(shapely.boundary(geoms))
    on_ring = shapely.distance(shapely.points(coords), ring) <= tolerance
    points = np.vstack([coords[on_ring], np.asarray(ring.coords)[:-1]])
    position = shapely.line_locate_point(ring, shapely.points(points))
    order = np.argsort(position)
    position, points = position[order], points[order]
    keep = np.r_[True, np.diff(position) > tolerance]
    keep[position > ring.length - tolerance] = False    # The start point, reached again at the end.
    return points[keep]


def subdivide_outline(nodes, divisions):
    """Split every outline segment into ``divisions`` equal parts (one number, or one per segment)."""
    divisions = np.broadcast_to(divisions, len(nodes))
    if (divisions == 1).all():
        return nodes
    nxt = np.roll(nodes, -1, axis=0)
    seg = np.repeat(np.arange(len(nodes)), divisions)
    steps = (np.arange(len(seg)) - np.repeat(np.cumsum(divisions) - divisions, divisions)) / divisions[seg]
    return nodes[seg] + steps[:, None] * (nxt - nodes)[seg]


def segment_circles(n):
    """The two circles (end nodes) of each of the ``n`` outline segments."""
    return [(i, (i + 1) % n) for i in range(n)]


def even_outline(nodes, polygon):
    """Halve the longest face around any breached circle until no circle holds another segment's site.

    A long face next to a short one gives a circle large enough to hold a
    neighbouring segment's mirrored site. Halving faces keeps every node; the
    new ones are hanging vertices on the outside cells' faces, as with
    ``outline_split``. Faces no longer than the shortest one at the start are
    not halved, so a breach that remains (at a corner sharper than about 68
    degrees) is left to :func:`check_circles`. An outline without a breach is
    returned unchanged.
    """
    shortest = np.linalg.norm(np.roll(nodes, -1, axis=0) - nodes, axis=1).min()
    while True:
        n = len(nodes)
        inside, outside, centres, radii = mirrored_sites(nodes, polygon)
        own = segment_circles(n) * 2
        bad = breaches(np.vstack([inside, outside]), centres, radii, own)
        if not bad:
            return nodes
        length = np.linalg.norm(np.roll(nodes, -1, axis=0) - nodes, axis=1)
        faces = np.unique([f % n for k, h in bad for f in (k - 1, k, own[h][0])])
        longest = (1 - 1e-9) * length[faces].max()     # ties to roundoff are halved together
        if longest <= shortest:
            return nodes
        halve = np.zeros(n, dtype=bool)
        halve[faces[length[faces] >= longest]] = True
        nodes = subdivide_outline(nodes, np.where(halve, 2, 1))


def mirrored_sites(nodes, polygon):
    """One generator pair per outline segment, mirrored across it.

    Returns ``(inside, outside, centres, radii)``: the inside generators (one
    per segment), their mirrors outside the polygon, and the circle at each
    node.
    """
    nxt = np.roll(nodes, -1, axis=0)
    length = np.linalg.norm(nxt - nodes, axis=1)
    radii = RADIUS_FACTOR * np.maximum(length, np.roll(length, 1))
    r_a, r_b = radii, np.roll(radii, -1)
    # Intersections of the circles at both ends of each segment.
    along = (length ** 2 + r_a ** 2 - r_b ** 2) / (2 * length)
    offset = np.sqrt(np.maximum(r_a ** 2 - along ** 2, 0.0))
    unit = (nxt - nodes) / length[:, None]
    normal = np.column_stack([-unit[:, 1], unit[:, 0]])
    foot = nodes + along[:, None] * unit
    left, right = foot + offset[:, None] * normal, foot - offset[:, None] * normal
    left_inside = shapely.contains_xy(polygon, left[:, 0], left[:, 1])
    inside = np.where(left_inside[:, None], left, right)
    outside = np.where(left_inside[:, None], right, left)
    return inside, outside, nodes, radii


def fill_nodes(candidates, centres, radii, sites, spacing):
    """Candidate generators outside every circle and not crowding a site."""
    candidates = np.asarray(candidates, dtype=float).reshape(-1, 2)
    if not len(candidates):
        return candidates
    clear = np.ones(len(candidates), dtype=bool)
    for hits in cKDTree(candidates).query_ball_point(centres, CLEARANCE * radii):
        clear[hits] = False
    gap, _ = cKDTree(sites).query(candidates)
    return candidates[clear & (gap > 0.5 * spacing)]


def breaches(generators, centres, radii, own):
    """(circle, generator) pairs with the generator strictly inside a circle other than the two it sits on.

    ``own`` gives, per generator, the indices of the two circles it sits on
    (-1 for fill nodes).
    """
    hits = cKDTree(generators).query_ball_point(centres, radii * (1 - INSIDE_EPS))
    return [(k, h) for k, inside in enumerate(hits) for h in inside if k not in own[h]]


def check_circles(generators, centres, radii, own):
    """Raise if a generator lies strictly inside a circle other than the two of its own segment.

    Fill nodes are kept out of every circle, so the offender is a mirrored
    outline site that :func:`even_outline` could not clear: the outline faces
    around that circle meet too sharply (or are too uneven).
    """
    bad = breaches(generators, centres, radii, own)
    if bad:
        k, h = bad[0]
        length = np.linalg.norm(np.roll(centres, -1, axis=0) - centres, axis=1)
        near = length[[k - 1, k, own[h][0]]]
        x, y = centres[k]
        raise ValueError(
            f"refill: the outline faces near ({x:.2f}, {y:.2f}), from {near.min():.3g} to {near.max():.3g} long, "
            "meet too sharply (or are too uneven) for every outline face to stay exact; faces of similar length "
            "need a corner of at least about 68 degrees. Blunt that corner of the outline, or give the base grid "
            "more even faces there."
        )


def voronoi_cells(generators, polygon, keep, nodes, tolerance):
    """Voronoi cells of ``generators`` clipped to ``polygon``, for the generators in ``keep``.

    Built with Qhull as in vorflow's tessellator (about the bounding-box
    centre, a roundoff-crossed ring replaced by its convex hull): the
    mirrored pairs put four generators on each outline circle, a degenerate
    input on which GEOS can return a wrong cell. Cell vertices within
    ``tolerance`` of an outline node are set to it exactly, so they coincide
    with the base grid's vertices instead of leaving roundoff-length faces.
    """
    generators = np.asarray(generators, dtype=float)
    origin = (generators.min(axis=0) + generators.max(axis=0)) / 2.0
    vor = Voronoi(generators - origin)
    regions = [vor.regions[r] for r in vor.point_region[keep]]
    if any(not region or -1 in region for region in regions):
        raise ValueError("refill: a cell inside the outline is unbounded; please report this.")
    ring_of = np.repeat(np.arange(len(regions)), [len(region) for region in regions])
    cells = shapely.polygons(shapely.linearrings(vor.vertices[np.concatenate(regions)] + origin, indices=ring_of))
    crossed = ~shapely.is_valid(cells)
    cells[crossed] = shapely.convex_hull(cells[crossed])
    cells = shapely.intersection(cells, polygon)
    coords = shapely.get_coordinates(cells)
    distance, nearest = cKDTree(nodes).query(coords)
    close = distance <= tolerance
    coords[close] = nodes[nearest[close]]
    return shapely.normalize(shapely.make_valid(shapely.set_coordinates(cells, coords)))
