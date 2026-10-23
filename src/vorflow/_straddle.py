"""Pure-geometry placement of barrier straddle pairs and of the lines that cross them.

Shapely only, no Gmsh calls; ``MeshGenerator`` turns the result into OCC
points and curves (see ``engine.py``).

Methodology
-----------
*Straddle pairs.* A barrier or straddle line (no quad buffer) is not meshed
as a curve. Pairs of points at +/-eps along its normal (eps =
``straddle_width / 2``, else ``0.2 * lc``) are placed about ``lc`` apart, so
the Voronoi edge between the two points of a pair, their perpendicular
bisector, lies on the line. End pairs on an oblique domain boundary slide
inward along the line until both points are inside (``_end_pair_slide``).

*Kinks.* A pair placed across a vertex where the line turns cannot be
symmetric about both legs, so the cell next to the corner reaches across the
line and has to be cut after tessellation. Instead each kink gets one pair
on each leg, both at the same distance ``h`` from the vertex
(``straddle_kinks``). The vertex is then equidistant from those four points
and becomes a Voronoi vertex: the barrier faces of the two legs meet there.
No pair is placed between the two, and pairs elsewhere on each leg are
spaced evenly as before. On the outer side of a sharp kink the two pairs
leave a gap that the mesher would fill with a node of its own, nearer the
vertex than the pairs; points on the same circle about the vertex fill it
instead (``kink_fill_points``). On the inner side of a sharp kink the two
legs' inner points would come closer to each other than to their own
partners, so the two kink pairs use a smaller offset, ``h * tan(theta / 2)
/ KINK_INNER_MARGIN`` (``theta`` the interior angle), where that is below
``eps`` (``kink_offset``). Moving the pairs away from the vertex instead
would leave an empty wedge that the mesher fills with nodes of its own;
reducing the offset of further pairs too lets mesh nodes reach the thinner
pairs. Crossing anchors and mirror pairs keep
clear of kinks.

*Crossings.* An embedded standard line that crosses an embedded straddle line
would otherwise be trimmed back by the barrier corridor (1.2 * eps), leaving
its end node at an arbitrary offset from the nearest pair. Instead the
crossing becomes an anchor: a pair is placed exactly at it, the other pairs
are spaced evenly between consecutive anchors and barrier ends, and the line
ends on the pair point on each side (``_align_line_to_crossings``).

*Oblique crossings.* The pair stays perpendicular to the barrier, because a
pair laid along the crossing line would put its bisector (the Voronoi edge)
off the barrier. The line bends instead: its last segment runs from its
first node beyond the crossing to the pair point. On the acute side of an
oblique crossing the line's next nodes lie close to the barrier, where their
cells would reach across it (or squeeze the pairs there). Those nodes are
placed at a known spacing and mirrored across the barrier; each node and its
mirror act as one more pair (a wider one) and as a fixed point for the even
spacing of the regular pairs. A generator set that is mirror-symmetric about
the barrier has no Voronoi edge crossing it, so the barrier cells stay
closed and symmetric.

A crossing that cannot be anchored (too close to a barrier end or another
anchor, pair outside the domain, collinear overlap, or no room to bend) keeps
the old behaviour: the line is trimmed by the barrier zone.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import substring

from ._features import feature_lc, is_embedded, positive_number, row_bool

# Anchors closer than this many barrier cell sizes to a barrier end pair or
# to another anchor are rejected (that line is trimmed as before), so no two
# pairs get squeezed together.
CROSSING_ANCHOR_MIN_GAP = 0.5
# Nodes of a crossing line nearer the barrier than this many barrier cell
# sizes are mirrored across it. Nearer than about half a cell a node's cell
# reaches across the barrier; mirroring nodes further out squeezes the
# regular pairs between them. Results were flat for 0.55-0.7 over crossings
# of 20-90 degrees.
CROSSING_MIRROR_DISTANCE = 0.6
# A mirrored node closer than this many barrier cell sizes (along the
# barrier) to another fixed pair is left unmirrored.
CROSSING_MIRROR_MIN_GAP = 0.25
# A vertex where a straddle line turns by more than this many degrees is a
# kink and gets its own pairs. Below it the line counts as straight; the
# vertices that densification adds turn by round-off only.
KINK_MIN_TURN_DEGREES = 1.0
# Both legs at a kink must be at least this many cell sizes long. Shorter
# legs (a finely sampled curve) keep the evenly spaced pairs, and the
# tessellator cuts what still reaches across the line.
KINK_MIN_LEG = 0.5
# The two pairs of a sharp kink use the offset h * tan(theta / 2) /
# KINK_INNER_MARGIN (theta the interior angle) when that is below eps,
# keeping each leg's inner points nearer their own partners than the other
# leg's.
KINK_INNER_MARGIN = 1.1
# Below this interior angle the pairs cannot keep every cell on its side of
# the line at the line's resolution (in tests, spikes of 25 degrees or less
# still needed a cut), so MeshGenerator warns.
KINK_WARN_DEGREES = 30.0


def _unit_tangent(line, d, probe):
    """Unit tangent of ``line`` at distance ``d`` along it.

    The direction is estimated from a short chord of length ``probe``. The
    caller chooses ``probe`` proportional to the line length so the estimate
    is CRS-unit independent (a fixed absolute step would span whole features
    on short lines and blunt corners on curved ones).
    """
    length = line.length
    if d >= length - probe:
        p1 = line.interpolate(max(d - probe, 0.0))
        p2 = line.interpolate(d)
    else:
        p1 = line.interpolate(d)
        p2 = line.interpolate(d + probe)
    dx, dy = p2.x - p1.x, p2.y - p1.y
    mag = math.hypot(dx, dy)
    if mag == 0:
        # Degenerate (zero-length) input: any unit vector keeps the straddle
        # pair perpendicular and non-coincident.
        return 1.0, 0.0
    return dx / mag, dy / mag


def _straddle_pair(line, d, epsilon, probe):
    """The two straddle points at distance ``d`` along ``line``, +/-``epsilon`` along its normal."""
    p = line.interpolate(d)
    dx, dy = _unit_tangent(line, d, probe)
    nx, ny = -dy, dx
    return [(p.x + sign * nx * epsilon, p.y + sign * ny * epsilon) for sign in (1.0, -1.0)]


def _end_pair_slide(line, at_end, epsilon, probe, domain, limit, tol):
    """Distance to move an end pair inward along the line so both points lie in ``domain``.

    Each point of the pair moves parallel to the line, so the pair stays
    mirror-symmetric about it. The slide stops where the last point to enter
    reaches the domain boundary; 0 if both already lie inside, None if one
    does not enter within ``limit``.
    """
    d = line.length if at_end else 0.0
    dx, dy = _unit_tangent(line, d, probe)
    if at_end:
        dx, dy = -dx, -dy
    slide = 0.0
    for x, y in _straddle_pair(line, d, epsilon, probe):
        if shapely.dwithin(domain, Point(x, y), tol):
            continue
        path = shapely.LineString([(x, y), (x + dx * limit, y + dy * limit)])
        entered = path.intersection(domain)
        if entered.is_empty:
            return None
        entry = min(path.project(Point(c)) for c in shapely.get_coordinates(entered))
        slide = max(slide, float(entry))
    return slide


def straddle_kinks(line, lc, epsilon):
    """(distance along ``line``, half-spacing ``h``, offset slope) of each kink that gets its own pairs.

    A kink is a vertex where the line turns by more than
    KINK_MIN_TURN_DEGREES, with both legs (to the neighbouring kinks or the
    line ends) at least KINK_MIN_LEG * ``lc`` long. ``h`` is ``lc / 2``,
    capped at a quarter of each adjacent leg so the pairs of two
    neighbouring kinks never cross. The slope is ``tan(theta / 2) /
    KINK_INNER_MARGIN``; see kink_offset.
    """
    xy = np.asarray(line.coords, dtype=float)[:, :2]
    steps = np.hypot(*np.diff(xy, axis=0).T)
    xy = np.vstack([xy[:1], xy[1:][steps > 0]])
    if len(xy) < 3:
        return []
    seg = np.diff(xy, axis=0)
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    turn = np.abs((np.diff(heading) + np.pi) % (2 * np.pi) - np.pi)
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*seg.T))])
    is_kink = turn > math.radians(KINK_MIN_TURN_DEGREES)
    kinks = list(along[1:-1][is_kink])
    half_angle = dict(zip(kinks, (np.pi - turn[is_kink]) / 2.0))
    # Dropping a kink lengthens its neighbours' legs, so repeat until stable.
    while True:
        legs = np.diff([0.0, *kinks, line.length])
        short = {k for k in range(len(kinks)) if min(legs[k], legs[k + 1]) < KINK_MIN_LEG * lc}
        if not short:
            break
        kinks = [d for k, d in enumerate(kinks) if k not in short]
    legs = np.diff([0.0, *kinks, line.length])
    out = []
    for k, d in enumerate(kinks):
        h = min(0.5 * lc, legs[k] / 4.0, legs[k + 1] / 4.0)
        out.append((float(d), h, math.tan(half_angle[d]) / KINK_INNER_MARGIN))
    return out


def kink_interior_degrees(slope):
    """Interior angle of a kink from its offset slope (see straddle_kinks)."""
    return math.degrees(2.0 * math.atan(slope * KINK_INNER_MARGIN))


def kink_offset(d, kinks, epsilon):
    """Offset of the pair at distance ``d``: ``epsilon``, or less for the two pairs of a sharp kink.

    A kink pair, ``h`` from the vertex, is offset by at most ``h * slope``
    (see straddle_kinks). The bisector of a pair stays on the leg whatever
    the offset.
    """
    for k, h, slope in kinks:
        if d in (k - h, k + h):
            return min(epsilon, h * slope)
    return epsilon


def kink_fill_points(line, kinks, epsilon):
    """Points on the outer side of each kink, on the circle through its four pair points.

    The pairs at ``d -/+ h`` sit on a circle of radius ``sqrt(h**2 +
    eps_k**2)`` about the vertex, ``eps_k`` their offset (kink_offset). On a straight line the outer points of two
    neighbouring pairs span an arc of ``pi - 2 * alpha`` on that circle
    (``alpha = atan(eps_k / h)``); at a kink the outer arc is wider. It is
    split into pieces no wider than that, and a point is placed at each
    division, so the mesher has no room to put a node nearer the vertex.
    """
    points = []
    for d, h, _ in kinks:
        eps_k = kink_offset(d - h, kinks, epsilon)
        vertex = line.interpolate(d)
        before = np.array(line.interpolate(d - h).coords[0])
        after = np.array(line.interpolate(d + h).coords[0])
        centre = np.array(vertex.coords[0])
        a1 = math.atan2(*(before - centre)[::-1])
        a2 = math.atan2(*(after - centre)[::-1])
        alpha = math.atan2(eps_k, h)
        radius = math.hypot(h, eps_k)
        ccw = (a2 - a1) % (2 * math.pi)
        # The interior wedge is the smaller angle between the legs.
        if ccw < math.pi:
            start, span = a2 + alpha, 2 * math.pi - ccw - 2 * alpha
        else:
            start, span = a1 + alpha, ccw - 2 * alpha
        pieces = math.ceil(span / (math.pi - 2 * alpha) - 1e-9)
        for j in range(1, pieces):
            angle = start + span * j / pieces
            points.append((centre[0] + radius * math.cos(angle), centre[1] + radius * math.sin(angle)))
    return points


def _near_kink(barrier, d, gap):
    """True if distance ``d`` lies within ``gap`` of the pairs of one of ``barrier``'s kinks."""
    return any(abs(d - k) < h + gap for k, h, _ in barrier.kinks)


def _anchored_grid(length, lc, anchors, kinks=()):
    """Distances 0..length through every anchor, each gap split evenly into ~``lc`` steps.

    Each kink ``(d, h, eps_k)`` adds breaks at ``d - h`` and ``d + h`` and leaves
    the gap between them empty. Returns (distances, first gap spacing, last
    gap spacing). Without anchors or kinks this is
    ``np.linspace(0, length, ceil(length / lc) + 1)``.
    """
    windows = [(d - h, d + h) for d, h, _ in kinks]
    breaks = sorted([0.0, *anchors, length, *(edge for window in windows for edge in window)])
    pieces, spacings = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        if (a, b) in windows:
            pieces.append(np.array([a]))
            spacings.append(b - a)
            continue
        num_segments = int(max(1, np.ceil((b - a) / lc)))
        pieces.append(np.linspace(a, b, num_segments + 1)[:-1])
        spacings.append((b - a) / num_segments)
    distances = np.concatenate(pieces + [np.array([length])])
    return distances, spacings[0], spacings[-1]


def _straddle_distances(line, lc, epsilon, probe, domain, tol, anchors=(), kinks=()):
    """Distances along ``line`` of its straddle pairs, with end pairs slid inside ``domain``.

    Pairs are spaced about ``lc`` apart and include both endpoints, every
    distance in ``anchors`` and the pairs ``d -/+ h`` of every kink in
    ``kinks`` (see straddle_kinks); each gap between consecutive breaks is
    split evenly, and no pair is placed inside a kink. Where a line meets the domain boundary obliquely, a pair at the
    endpoint has one point outside; that pair moves inward along the line
    (see _end_pair_slide) and interior pairs it comes within half a spacing
    of are dropped. Anchors are always kept (plan_barrier_crossings only
    accepts anchors inside the slid ends). Without a domain the distances
    are returned unchanged.
    """
    length = line.length
    anchors = sorted(anchors)
    distances, first_spacing, last_spacing = _anchored_grid(length, lc, anchors, kinks)
    if domain is None:
        return distances
    start = _end_pair_slide(line, False, epsilon, probe, domain, length / 2.0, tol)
    end = _end_pair_slide(line, True, epsilon, probe, domain, length / 2.0, tol)
    lo = 0.0 if start is None else start + first_spacing / 2.0
    hi = length if end is None else length - end - last_spacing / 2.0
    anchor_set = set(anchors) | {edge for d, h, _ in kinks for edge in (d - h, d + h)}
    positions = [] if start is None else [start]
    positions += [d for d in distances[1:-1] if lo < d < hi or d in anchor_set]
    # On a short line the two slid end pairs can meet; keep only the first.
    if end is not None and (not positions or length - end - positions[-1] >= last_spacing / 2.0):
        positions.append(length - end)
    return np.array(positions)


def straddle_epsilon(lc, straddle):
    """Half-width of a straddle pair: ``straddle_width / 2``, else ``0.2 * lc``."""
    return straddle / 2.0 if straddle else lc * 0.20


def straddle_probe(line):
    """Tangent probe for ``line``: proportional to its length, so CRS-unit independent."""
    return max(line.length * 1e-4, 1e-12)


def is_straddle_line(row):
    """True for a line meshed as straddle pairs: a barrier or straddle line without a quad buffer."""
    if row_bool(row, 'quad_buffer', False):
        return False
    return row_bool(row, 'is_barrier', False) or positive_number(row.get('straddle_width')) is not None


@dataclasses.dataclass
class _Barrier:
    """An embedded straddle line and its pair geometry."""

    idx: int
    line: LineString
    lc: float
    epsilon: float
    probe: float
    tol: float
    lo: float  # first and last pair positions (slid ends)
    hi: float
    kinks: list = dataclasses.field(default_factory=list)  # see straddle_kinks

    @property
    def corridor(self):
        """Half-width of the barrier's protection corridor (see buffer.corridor_geometry)."""
        return 1.20 * self.epsilon

    def normal(self, d):
        tx, ty = _unit_tangent(self.line, d, self.probe)
        return -ty, tx

    def mirror(self, xy):
        """(distance along the barrier, reflection of ``xy`` across its tangent there)."""
        d = self.line.project(Point(xy))
        foot = self.line.interpolate(d)
        nx, ny = self.normal(d)
        off = (xy[0] - foot.x) * nx + (xy[1] - foot.y) * ny
        return d, (xy[0] - 2.0 * off * nx, xy[1] - 2.0 * off * ny)


@dataclasses.dataclass(frozen=True)
class _LineCrossing:
    """Where an embedded standard line crosses (or ends on) a straddle line."""

    line_distance: float  # along the standard line
    barrier: _Barrier
    point: tuple  # crossing point on the barrier
    normal: tuple  # unit normal of the barrier there
    pair: tuple  # (C + eps * normal, C - eps * normal), exactly as the barrier places them
    spacing: float  # node spacing of the line near the crossing


@dataclasses.dataclass
class StraddlePlan:
    """Fixed pairs per straddle line and replacement geometry per crossing line."""

    # barrier id -> {distance along it: (point, point)} pairs placed exactly.
    fixed: dict = dataclasses.field(default_factory=dict)
    # line id -> [LineString] to add instead of the zone-trimmed line.
    line_parts: dict = dataclasses.field(default_factory=dict)


def _barrier_info(idx, row, background_lc, domain):
    """_Barrier for a straddle-line row, with its slid end positions."""
    line = row.geometry
    lc = feature_lc(row, background_lc)
    epsilon = straddle_epsilon(lc, positive_number(row.get('straddle_width')))
    probe = straddle_probe(line)
    tol = epsilon * 1e-6
    length = line.length
    lo, hi = 0.0, length
    if domain is not None:
        start = _end_pair_slide(line, False, epsilon, probe, domain, length / 2.0, tol)
        end = _end_pair_slide(line, True, epsilon, probe, domain, length / 2.0, tol)
        lo = 0.0 if start is None else start
        hi = length if end is None else length - end
    return _Barrier(int(idx), line, lc, epsilon, probe, tol, lo, hi,
                    straddle_kinks(line, lc, epsilon))


def _crossing_candidates(line, barrier, corridor):
    """(kind, point on the barrier) where ``line`` crosses or ends on ``barrier``.

    kind 0 is a crossing or touch point; kind 1 an endpoint of ``line`` that
    stops inside the barrier corridor without touching it (a T-junction),
    projected onto the barrier. Collinear overlaps are ignored.
    """
    inter = line.intersection(barrier)
    hits = [(0, g) for g in getattr(inter, 'geoms', [inter]) if g.geom_type == 'Point']
    for xy in (line.coords[0], line.coords[-1]):
        end = Point(xy)
        if barrier.distance(end) < corridor and all(end.distance(g) > corridor for _, g in hits):
            hits.append((1, barrier.interpolate(barrier.project(end))))
    return hits


def _anchor_crossings(barrier, standard, background_lc, domain):
    """Accepted anchor distances on ``barrier`` and the _LineCrossing records per line id."""
    line = barrier.line
    min_gap = CROSSING_ANCHOR_MIN_GAP * barrier.lc
    candidates = []
    for l_idx, l_row in standard:
        for kind, point in _crossing_candidates(l_row.geometry, line, barrier.corridor):
            candidates.append((kind, line.project(point), l_idx, l_row))
    # Crossings before T-junction ends, then along the barrier.
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    accepted, crossings = [], {}
    for _kind, d, l_idx, l_row in candidates:
        if not (barrier.lo + min_gap <= d <= barrier.hi - min_gap):
            continue
        if _near_kink(barrier, d, min_gap):
            continue
        shared = next((a for a in accepted if abs(a - d) <= barrier.tol), None)
        if shared is None and any(abs(a - d) < min_gap for a in accepted):
            continue
        d = d if shared is None else shared
        pair = tuple(_straddle_pair(line, d, barrier.epsilon, barrier.probe))
        if domain is not None and not all(domain.covers(Point(xy)) for xy in pair):
            continue
        if shared is None:
            accepted.append(d)
        c = line.interpolate(d)
        crossings.setdefault(l_idx, []).append(_LineCrossing(
            line_distance=float(l_row.geometry.project(c)),
            barrier=barrier,
            point=(c.x, c.y),
            normal=barrier.normal(d),
            pair=pair,
            spacing=min(feature_lc(l_row, background_lc), barrier.lc),
        ))
    return accepted, crossings


def _circle_exit(piece, center, radius):
    """Distance along ``piece`` where it first gets ``radius`` away from ``center``, or None."""
    coords = list(piece.coords)
    walked = 0.0
    cx, cy = center
    for (ax, ay), (bx, by) in zip(coords[:-1], coords[1:]):
        seg = math.hypot(bx - ax, by - ay)
        if math.hypot(bx - cx, by - cy) >= radius and seg > 0:
            # |A + t (B - A) - C| = radius, larger root (A lies inside).
            dx, dy = (bx - ax) / seg, (by - ay) / seg
            fx, fy = ax - cx, ay - cy
            b = fx * dx + fy * dy
            disc = b * b - (fx * fx + fy * fy - radius * radius)
            return walked + (-b + math.sqrt(max(disc, 0.0)))
        walked += seg
    return None


def _crossing_head(piece, crossing, barrier_zone, other_zone):
    """Nodes a line piece starts with at ``crossing``, or None to keep it trimmed as before.

    ``piece`` runs from the crossing outward. Returns (nodes, resume, mirrors):
    ``nodes`` starts at the pair point on the piece's side, then nodes on the
    piece at ``crossing.spacing`` apart (the first that far from the pair
    point) until one is at least CROSSING_MIRROR_DISTANCE barrier cells from
    the barrier; ``resume`` is the distance along ``piece`` from which its
    own vertices are kept; ``mirrors`` are the nodes to mirror.
    """
    barrier = crossing.barrier
    spacing = crossing.spacing
    cx, cy = crossing.point
    nx, ny = crossing.normal
    probe = piece.interpolate(min(0.5 * spacing, 0.5 * piece.length))
    side = (probe.x - cx) * nx + (probe.y - cy) * ny
    if side == 0:
        return None
    pair_point = crossing.pair[0] if side > 0 else crossing.pair[1]
    s = _circle_exit(piece, pair_point, spacing)
    if s is None:
        return None
    nodes, mirrors = [pair_point], []
    mirror_distance = CROSSING_MIRROR_DISTANCE * barrier.lc
    last = s
    while s <= piece.length:
        node = piece.interpolate(s)
        offset = barrier.line.distance(node)
        if offset <= barrier.corridor or (barrier_zone is not None and barrier_zone.intersects(node)):
            return None
        nodes.append((node.x, node.y))
        last = s
        if offset >= mirror_distance:
            break
        mirrors.append((node.x, node.y))
        s += spacing
    head = LineString(nodes)
    if head.intersects(barrier.line) or (other_zone is not None and head.intersects(other_zone)):
        return None
    return nodes, last + 0.5 * spacing, mirrors


def _vertex_distances(coords):
    """Cumulative distance along a coordinate list at each vertex."""
    xy = np.asarray(coords, dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])


def _align_piece(piece, start, end, barrier_zone, other_zones):
    """Parts of one piece between crossings; returns (LineStrings, [(barrier, node) to mirror])."""
    length = piece.length
    reverse = LineString(list(piece.coords)[::-1])
    head = tail = None
    if start is not None:
        head = _crossing_head(piece, start, barrier_zone, other_zones.get(start.barrier.idx))
    if end is not None:
        tail = _crossing_head(reverse, end, barrier_zone, other_zones.get(end.barrier.idx))
    if head is not None and tail is not None and head[1] + tail[1] > length:
        tail = None
    lo = head[1] if head is not None else 0.0
    hi = length - tail[1] if tail is not None else length
    coords = [tuple(xy[:2]) for xy in piece.coords]
    along = _vertex_distances(coords)
    middle = [xy for xy, d in zip(coords, along) if lo < d < hi]
    if head is None:
        middle = [coords[0]] + middle
    if tail is None:
        middle = middle + [coords[-1]]
    core = ([head[0][-1]] if head else []) + middle + ([tail[0][-1]] if tail else [])
    if len(core) == 1:
        trimmed = [core] if barrier_zone is None or not barrier_zone.intersects(Point(core[0])) else []
    elif len(core) >= 2:
        geom = LineString(core)
        if barrier_zone is not None and geom.intersects(barrier_zone):
            geom = geom.difference(barrier_zone)
        # Trimming keeps the orientation and the vertices it does not cut.
        trimmed = [[tuple(xy[:2]) for xy in g.coords] for g in getattr(geom, 'geoms', [geom])
                   if g.geom_type == 'LineString' and not g.is_empty]
    else:
        trimmed = []
    parts, mirrors = [], []
    for part in trimmed:
        if head is not None and part[0] == core[0]:
            part = head[0][:-1] + part
            mirrors.extend((start.barrier, xy) for xy in head[2])
        if tail is not None and part[-1] == core[-1]:
            part = part + tail[0][:-1][::-1]
            mirrors.extend((end.barrier, xy) for xy in tail[2])
        if len(part) >= 2:
            parts.append(LineString(part))
    return parts, mirrors


def _align_line_to_crossings(line, crossings, barrier_zone, other_zones):
    """Parts of ``line`` that end on the straddle pairs of its crossings, plus the nodes to mirror.

    The line is split at its crossings. From each crossing end the piece
    starts at the pair point on its side, runs to the piece at one node
    spacing from it and continues along the piece (_crossing_head); the rest
    is trimmed by ``barrier_zone`` as for any line. A crossing end falls back
    to plain trimming if its nodes would enter a protected corridor
    (``other_zones[barrier id]`` is every corridor but the crossed one's).
    """
    length = line.length
    crossings = sorted(crossings, key=lambda c: c.line_distance)
    tol = 1e-9 * max(length, 1.0)
    breaks = [(0.0, None)] + [(min(max(c.line_distance, 0.0), length), c) for c in crossings]
    breaks.append((length, None))
    parts, mirrors = [], []
    for (a, start), (b, end) in zip(breaks[:-1], breaks[1:]):
        if b - a <= tol:
            continue
        piece_parts, piece_mirrors = _align_piece(
            substring(line, a, b), start, end, barrier_zone, other_zones
        )
        parts.extend(piece_parts)
        mirrors.extend(piece_mirrors)
    return parts, mirrors


def _add_mirrors(plan, mirrors, domain):
    """Record each (barrier, node) as a fixed pair of the node and its mirror, where there is room."""
    for barrier, xy in mirrors:
        d, image = barrier.mirror(xy)
        fixed = plan.fixed.setdefault(barrier.idx, {})
        min_gap = CROSSING_MIRROR_MIN_GAP * barrier.lc
        if not (barrier.lo + min_gap <= d <= barrier.hi - min_gap):
            continue
        if _near_kink(barrier, d, min_gap):
            continue
        if any(abs(d - other) < min_gap for other in fixed):
            continue
        if domain is not None and not domain.covers(Point(image)):
            continue
        fixed[d] = (xy, image)


def plan_barrier_crossings(lines_gdf, background_lc, domain, barrier_zone, corridor_zones):
    """Plan the anchor pairs and bent line ends where standard lines cross straddle lines.

    Only embedded LineString features take part. ``corridor_zones`` maps a
    barrier id to the union of every other protected corridor (or None).
    Lines and barriers without an accepted crossing are left out of the
    plan, so they are meshed exactly as before.
    """
    plan = StraddlePlan()
    if lines_gdf is None or lines_gdf.empty:
        return plan
    barriers, standard = [], []
    for idx, row in lines_gdf.iterrows():
        geom = row.geometry
        if not is_embedded(row) or geom is None or geom.geom_type != 'LineString' or geom.length <= 0:
            continue
        if is_straddle_line(row):
            barriers.append(_barrier_info(idx, row, background_lc, domain))
        elif not row_bool(row, 'quad_buffer', False):
            standard.append((int(idx), row))
    if not barriers or not standard:
        return plan

    crossings = {}
    for barrier in barriers:
        accepted, by_line = _anchor_crossings(barrier, standard, background_lc, domain)
        if not accepted:
            continue
        plan.fixed[barrier.idx] = {
            d: tuple(_straddle_pair(barrier.line, d, barrier.epsilon, barrier.probe))
            for d in accepted
        }
        for l_idx, records in by_line.items():
            crossings.setdefault(l_idx, []).extend(records)

    other_zones = {b: corridor_zones(b) for b in plan.fixed}
    rows = dict(standard)
    for l_idx, records in crossings.items():
        parts, mirrors = _align_line_to_crossings(
            rows[l_idx].geometry, records, barrier_zone, other_zones
        )
        plan.line_parts[l_idx] = parts
        _add_mirrors(plan, mirrors, domain)
    return plan
