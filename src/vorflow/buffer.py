"""Pure-geometry planning for structured quad buffers.

Everything here is shapely-only: no Gmsh calls. ``MeshGenerator`` turns the
plans into OCC surfaces and applies the transfinite/recombine constraints
after fragmentation (see ``engine.py``).

Methodology
-----------
*Footprints.* A quad-buffered line becomes one strip polygon per line part:
the line is simplified (legacy default ``1.5 * lc``, or the feature's explicit
``simplify_tolerance``; zero disables it) and segmentized (``lc``) so its two
mitred offsets at ``+/- thickness * lc / 2`` are symmetric and split into
~lc-long segments, keeping the transfinite divisions equal on opposite sides.
The offset lines (not the strip) are clipped to the domain so their endpoints
stay the true strip corners. Closed lines and quad-buffered polygons become
annular bands between the +/- offsets of their simplified outlines; bands
have no end caps or 4-corner structure, so they are meshed recombine-only.

*Crossing priority.* Every footprint is planned before any is trimmed. When
two cross, the one with the lower priority key stays continuous and only the
other is trimmed -- against the winner's footprint plus a gap of
``QUAD_BUFFER_CROSSING_GAP`` local cell widths, so the loser's quads butt up
against the winner's structured row. Priority key (lower wins): user
``z_order`` (higher first), then finer ``lc``, then wider strip, then line
over polygon, then insertion order.

*Protection corridors.* Barrier, straddle and quad-buffer features each get a
corridor (the feature buffered by ``1.2 * eps``). Their union, the barrier
zone, trims standard lines away from these sensitive regions; corridors of
non-quad features also act as obstacles for quad buffers.

*Slivers.* One-sided trimming can leave thin wedges at shallow-angle overlaps.
Pieces are morphologically opened and dropped when thinner than ~0.8 cells;
the gap is filled with unstructured elements.

*Crossing refinement.* From the loser's side, each crossing region is recorded
as a refinement disk (``Crossing``) so ``_setup_fields`` can pin the gap fill
to ``min(lc)`` rather than letting it jump to the background size.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import pandas as pd
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import linemerge, unary_union
from shapely.validation import make_valid

from ._features import (
    feature_lc,
    is_embedded,
    line_parts,
    polygon_parts,
    positive_number,
    row_bool,
    sanitize_coords,
)

# Half-cell gap left between a trimmed (lower-priority) quad buffer and the
# continuous (higher-priority) one it crosses. The loser is trimmed to the
# winner's footprint plus this many local cell widths, so its quads butt up
# against the winner's structured node row with one clean unstructured row in
# between. Tunable; larger values widen the gap if slivers appear.
QUAD_BUFFER_CROSSING_GAP = 0.5

# Rank used as the fourth priority-key element: lines win ties over polygons.
_KIND_RANK = {'line': 0, 'poly': 1}


@dataclass
class StripPart:
    """Untrimmed strip for one part of a quad-buffered line."""

    strip: Polygon
    corners: list  # Four ordered corners for an open strip; [] for an annulus.
    side_lines: tuple  # Open-strip offsets; () for an annulus with no end caps.


@dataclass
class BufferPlan:
    """Planned footprint and priority of one quad-buffered feature."""

    kind: str  # 'line' or 'poly'
    footprint: object
    lc: float
    thickness: int
    priority_key: tuple
    parts: list = field(default_factory=list)  # StripPart list (lines)
    band: object = None  # annular band geometry (polygons)

    @property
    def width(self) -> float:
        """Full strip width, ``thickness * lc``."""
        return self.thickness * self.lc


@dataclass(frozen=True)
class Crossing:
    """Refinement disk over a region where two quad buffers cross."""

    x: float
    y: float
    size: float
    radius: float


def quad_buffer_thickness(row) -> int:
    """The row's ``quad_buffer_thickness`` (default 1); only 1 or 2 are allowed."""
    value = row.get('quad_buffer_thickness', 1)
    if value is None or pd.isna(value):
        return 1
    value = int(value)
    if value not in (1, 2):
        raise ValueError("quad_buffer_thickness must be either 1 or 2.")
    return value


def feature_z_order(row) -> float:
    """The row's ``z_order`` as a float; missing or non-numeric gives 0.0."""
    val = row.get('z_order', 0)
    if val is None or pd.isna(val):
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        # Non-numeric z_order is treated as the default layer.
        return 0.0


def coerce_offset_line(geom):
    """Reduce an offset result to a single LineString (longest merged part), or None."""
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged
        lines = [part for part in merged.geoms if part.length > 0] if hasattr(merged, "geoms") else []
        return max(lines, key=lambda line: line.length) if lines else None
    return None


def domain_union_geometry(polygons_gdf):
    """Valid union of the embedded polygons (the meshed domain), or None."""
    if polygons_gdf is None or polygons_gdf.empty:
        return None
    embedded = []
    for _, poly_row in polygons_gdf.iterrows():
        if is_embedded(poly_row):
            embedded.append(poly_row.geometry)
    if not embedded:
        return None
    return make_valid(unary_union(embedded))


def _buffer_simplification(lc, tolerance):
    """Resolve the legacy buffer default while honoring explicit zero."""
    if tolerance is None or pd.isna(tolerance):
        return 1.5 * lc
    if isinstance(tolerance, bool):
        raise ValueError("simplify_tolerance must be a finite non-negative distance or None.")
    try:
        value = float(tolerance)
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value) or value < 0:
        raise ValueError("simplify_tolerance must be a finite non-negative distance or None.")
    return value


def plan_line_strip(line, lc, thickness, domain, feature_name,
                    simplify_tolerance=None) -> list[StripPart]:
    """Untrimmed strips for open lines and annular bands for closed lines.

    Closed offsets cannot be joined like an open strip: GEOS can return an
    open exterior offset at the ring's starting vertex. Build the band from
    polygon buffers instead, independent of ring winding and starting vertex.
    Its interior remains part of the original domain, not a material zone.
    Explicit ``simplify_tolerance=0`` preserves all source turns; ``None``
    retains the legacy buffer simplification of ``1.5 * lc``.
    """
    offset = thickness * lc / 2.0
    simplify = _buffer_simplification(lc, simplify_tolerance)
    plans = []
    for part in line_parts(line):
        if part.length <= 0:
            continue
        if part.is_closed:
            polygon = Polygon(part.coords)
            if not polygon.is_valid or polygon.area <= 0:
                warnings.warn(
                    f"Skipping structured buffer for closed line feature {feature_name}: "
                    "the line must form a simple polygon ring."
                )
                continue
            band = plan_polygon_band(
                polygon, lc, thickness, feature_name, simplify_tolerance=simplify
            )
            if band is not None and not band.is_empty:
                # An annulus has two closed boundary loops, not two offset
                # sides joined by caps. Do not invent four corners for the
                # engine's open-strip transfinite interpolation.
                plans.append(StripPart(strip=band, corners=[], side_lines=()))
            continue
        work = part.simplify(simplify) if simplify else part
        work = work.segmentize(lc)
        pos = coerce_offset_line(
            work.offset_curve(offset, quad_segs=1, join_style=2, mitre_limit=5.0)
        )
        neg = coerce_offset_line(
            work.offset_curve(-offset, quad_segs=1, join_style=2, mitre_limit=5.0)
        )
        if pos is None or neg is None:
            warnings.warn(
                f"Skipping structured buffer for line feature {feature_name} after offset split."
            )
            continue
        if domain is not None and not domain.is_empty:
            pos = coerce_offset_line(pos.intersection(domain))
            neg = coerce_offset_line(neg.intersection(domain))
            if pos is None or neg is None:
                warnings.warn(
                    f"Skipping structured buffer for line feature {feature_name} after domain clipping."
                )
                continue

        pos_coords = sanitize_coords(list(pos.coords), min_points=2)
        neg_coords = sanitize_coords(list(neg.coords), min_points=2)
        if len(pos_coords) < 2 or len(neg_coords) < 2:
            continue

        strip = Polygon(pos_coords + list(reversed(neg_coords)))
        if not strip.is_valid:
            strip = make_valid(strip)
        # Corner order matches gmshflow's setTransfiniteSurface(..., "Left", ...).
        corners = [
            tuple(neg_coords[0]),
            tuple(neg_coords[-1]),
            tuple(pos_coords[-1]),
            tuple(pos_coords[0]),
        ]
        plans.append(StripPart(strip=strip, corners=corners, side_lines=(pos, neg)))
    return plans


def plan_polygon_band(geom, lc, thickness, feature_name, simplify_tolerance=None):
    """Untrimmed annular band around a quad-buffered polygon's outline, or None.

    gmshflow parity (create_surfacegrid_from_buffer_poly): the zone interior
    is meshed from the inner offset, so the original outline never becomes
    mesh edges. thickness=1 leaves no nodes on the outline (the Voronoi faces
    trace the shape); thickness=2 puts a node row on it.
    """
    offset = thickness * lc / 2.0
    # Retain the legacy simplification unless explicitly disabled, including
    # for closed lines routed through this shared annular planner.
    simplify = _buffer_simplification(lc, simplify_tolerance)
    work = make_valid(geom.simplify(simplify) if simplify else geom)
    inner = make_valid(work.buffer(-offset, quad_segs=1, join_style=2, mitre_limit=5.0))
    outer = make_valid(work.buffer(offset, quad_segs=1, join_style=2, mitre_limit=5.0))
    inner_parts = [
        p for p in polygon_parts(inner) if not p.is_empty and p.area > 0
    ]
    if not inner_parts or outer.is_empty:
        warnings.warn(
            f"Polygon feature {feature_name} is too narrow for a quad_buffer band of "
            f"width {2.0 * offset:g}; meshing it without the structured buffer."
        )
        return None
    inner = inner_parts[0] if len(inner_parts) == 1 else MultiPolygon(inner_parts)
    # Difference (rather than boundary.buffer) so the band's inner ring and the
    # interior surface share exact coordinates and OCC merges them.
    return make_valid(outer.difference(inner))


def priority_key(z_order, lc, thickness, kind, order) -> tuple:
    """Crossing priority (lower wins): z_order desc, lc asc, width desc, line first, order."""
    return (-z_order, lc, -(thickness * lc), _KIND_RANK[kind], order)


def plan_quad_buffers(polygons_gdf, lines_gdf, background_lc, domain) -> dict:
    """Plan every quad-buffer footprint, keyed ('line'|'poly', index): lines then polygons, in input order."""
    plans = {}
    order = 0
    for idx, row in lines_gdf.iterrows():
        if not row_bool(row, 'quad_buffer', False):
            continue
        lc = feature_lc(row, background_lc)
        thickness = quad_buffer_thickness(row)
        parts = plan_line_strip(
            row.geometry, lc, thickness, domain, row.name,
            simplify_tolerance=row.get('simplify_tolerance'),
        )
        if not parts:
            continue
        plans[('line', int(idx))] = BufferPlan(
            kind='line',
            footprint=make_valid(unary_union([p.strip for p in parts])),
            lc=lc,
            thickness=thickness,
            priority_key=priority_key(feature_z_order(row), lc, thickness, 'line', order),
            parts=parts,
        )
        order += 1
    if not polygons_gdf.empty:
        for idx, row in polygons_gdf.iterrows():
            if not (is_embedded(row) and row_bool(row, 'quad_buffer', False)):
                continue
            lc = feature_lc(row, background_lc)
            thickness = quad_buffer_thickness(row)
            band = plan_polygon_band(
                row.geometry, lc, thickness, row.name,
                simplify_tolerance=row.get('simplify_tolerance'),
            )
            if band is None or band.is_empty:
                continue
            plans[('poly', int(idx))] = BufferPlan(
                kind='poly',
                footprint=make_valid(band),
                lc=lc,
                thickness=thickness,
                priority_key=priority_key(feature_z_order(row), lc, thickness, 'poly', order),
                band=band,
            )
            order += 1
    return plans


def protection_epsilon(row, background_lc) -> float:
    """Half-width protected around a feature: buffer half-width, straddle/2, or 0.2*lc."""
    lc = feature_lc(row, background_lc)
    if row_bool(row, 'quad_buffer', False):
        return quad_buffer_thickness(row) * lc / 2.0
    straddle = positive_number(row.get('straddle_width'))
    if straddle:
        return straddle / 2.0
    return lc * 0.20


def corridor_geometry(basis, eps, min_half_width=0.0):
    """Flat-capped corridor slightly wider (1.2x) than the feature's half-width."""
    return basis.buffer(max(eps * 1.20, min_half_width), cap_style=2)


def protected_corridors(polygons_gdf, lines_gdf, background_lc) -> dict:
    """(basis, eps) per barrier/straddle/quad-buffer line and embedded quad-buffer polygon."""
    corridors = {}
    for idx, row in lines_gdf.iterrows():
        if (
            row_bool(row, 'is_barrier', False)
            or row_bool(row, 'quad_buffer', False)
            or positive_number(row.get('straddle_width'))
        ):
            corridors[('line', int(idx))] = (
                row.geometry, protection_epsilon(row, background_lc)
            )
    if not polygons_gdf.empty:
        for idx, row in polygons_gdf.iterrows():
            if row_bool(row, 'quad_buffer', False) and is_embedded(row):
                corridors[('poly', int(idx))] = (
                    row.geometry.boundary, protection_epsilon(row, background_lc)
                )
    return corridors


def barrier_zone(corridors):
    """Union of all protection corridors (trims standard lines), or None."""
    if not corridors:
        return None
    return make_valid(unary_union([
        corridor_geometry(basis, eps)
        for basis, eps in corridors.values()
    ]))


def higher_priority_obstacles(self_key, plans, corridors):
    """Geometry a quad buffer must keep clear of, or None.

    The footprints of strictly higher-priority quad buffers (grown by the
    crossing gap) plus the corridors of non-quad protected features
    (barrier/straddle lines, which still trim mutually).
    """
    self_plan = plans.get(self_key)
    if self_plan is None:
        return None
    self_pkey = self_plan.priority_key
    lc_self = self_plan.lc
    geoms = []
    for key, plan in plans.items():
        if key == self_key:
            continue
        if plan.priority_key < self_pkey:
            geoms.append(
                plan.footprint.buffer(
                    QUAD_BUFFER_CROSSING_GAP * lc_self, cap_style=2
                )
            )
    for key, (basis, eps) in corridors.items():
        if key in plans or key == self_key:
            continue
        geoms.append(corridor_geometry(basis, eps, min_half_width=0.6 * lc_self))
    if not geoms:
        return None
    return make_valid(unary_union(geoms))


def find_crossings(self_key, plans) -> list[Crossing]:
    """Refinement disks where ``self_key``'s footprint overlaps higher-priority ones."""
    self_plan = plans.get(self_key)
    if self_plan is None:
        return []
    self_fp = self_plan.footprint
    self_pkey = self_plan.priority_key
    lc_self = self_plan.lc
    w_self = self_plan.width
    crossings = []
    for key, plan in plans.items():
        if key == self_key or not (plan.priority_key < self_pkey):
            continue
        inter = make_valid(self_fp.intersection(plan.footprint))
        for part in polygon_parts(inter):
            if part.is_empty or part.area <= 0:
                continue
            minx, miny, maxx, maxy = part.bounds
            part_radius = 0.5 * math.hypot(maxx - minx, maxy - miny)
            size = min(lc_self, plan.lc)
            radius = part_radius + 0.5 * (w_self + plan.width) + size
            crossings.append(Crossing(
                x=part.centroid.x,
                y=part.centroid.y,
                size=size,
                radius=radius,
            ))
    return crossings


def find_all_crossings(plans) -> list[Crossing]:
    """Crossing disks for every planned buffer, in plan order."""
    crossings = []
    for key in plans:
        crossings.extend(find_crossings(key, plans))
    return crossings


def clean_trimmed_pieces(geom, lc, feature_label) -> list:
    """Drop sliver pieces (thinner than ~0.8 cells) left by one-sided trimming.

    Morphological opening (mitre joins keep rectangles square) removes
    whiskers; the area and erosion tests drop the rest.
    """
    kept = []
    dropped = 0
    for part in polygon_parts(make_valid(geom)):
        if part.is_empty or part.area <= 0:
            continue
        opened = make_valid(
            part.buffer(-0.25 * lc, join_style=2).buffer(0.25 * lc, join_style=2)
        )
        candidates = polygon_parts(opened) if not opened.is_empty else []
        if not candidates:
            dropped += 1
            continue
        for sub in candidates:
            sub = make_valid(sub.simplify(0.1 * lc))
            if sub.is_empty or sub.area < 0.5 * lc * lc:
                dropped += 1
                continue
            eroded = sub.buffer(-0.4 * lc)
            if eroded.is_empty or getattr(eroded, 'area', 0.0) <= 0:
                dropped += 1
                continue
            kept.append(sub)
    if dropped:
        warnings.warn(
            f"Structured buffer for {feature_label} dropped {dropped} sliver "
            "piece(s) at a crossing (too thin to mesh); that gap is filled with "
            "unstructured elements. Flip z_order or simplify the geometry to avoid it."
        )
    return kept


def trim_against_obstacles(geom, obstacles, lc, feature_label):
    """Trim a footprint off ``obstacles``; returns (geometry or None if nothing survives, was_trimmed)."""
    if obstacles is None or not geom.intersects(obstacles):
        return geom, False
    warnings.warn(
        f"Structured buffer for {feature_label} crosses a higher-priority "
        "protected feature; it is trimmed at the crossing (set z_order to "
        "choose which feature stays continuous)."
    )
    pieces = clean_trimmed_pieces(make_valid(geom.difference(obstacles)), lc, feature_label)
    if not pieces:
        return None, True
    return make_valid(unary_union(pieces) if len(pieces) > 1 else pieces[0]), True
