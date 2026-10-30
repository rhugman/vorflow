"""Feature-row accessors and small geometry helpers.

Shared by the Gmsh engine (``engine.py``) and the pure-geometry quad-buffer
planning (``buffer.py``). Rows are clean ``ConceptualMesh`` GeoDataFrame rows,
so optional columns may be missing or NaN; every accessor resolves those to the
same defaults the engine has always used.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.validation import make_valid

DEFAULT_CONNECTIVITY_TOLERANCE = 1e-3
# A hex_ring point's ring seeds sit at 1 x lc; every other node-bearing feature
# (barrier sites included) must stay at least HEX_RING_CLEARANCE x lc away, so
# at least one cell width separates the ring from foreign nodes (closer ones
# would cut the hexagon).
HEX_RING_CLEARANCE = 2.0


def is_embedded(row) -> bool:
    """True unless the row sets ``embed`` to a falsy, non-NaN value."""
    val = row.get('embed', True)
    if pd.isna(val):
        return True
    return bool(val)


def row_bool(row, column, default=False) -> bool:
    """Read a boolean-ish column (True, 'true', '1', 'yes'); NaN gives ``default``."""
    val = row.get(column, default)
    if pd.isna(val):
        return bool(default)
    return (val is True) or (str(val).lower() in ['true', '1', 'yes'])


def positive_number(value) -> float | None:
    """``value`` as a float if it is a positive number, else None."""
    if value is None or pd.isna(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        # Non-numeric column content means "not set".
        return None
    return value if value > 0 else None


def is_face_line(row) -> bool:
    """True for a line whose cell faces follow it exactly: an embedded barrier (or a line with the
    deprecated ``straddle_width``) without a quad buffer. These go through ``vorflow._barrier``."""
    if row_bool(row, 'quad_buffer', False) or not is_embedded(row):
        return False
    return row_bool(row, 'is_barrier', False) or positive_number(row.get('straddle_width')) is not None


def ring_seed_coords(row) -> list:
    """The row's hex-ring seed ``(x, y)`` list; empty when the column is missing, None or NaN."""
    seeds = row.get('ring_seeds')
    if isinstance(seeds, (list, tuple)):
        return list(seeds)
    return []


def feature_lc(row, background_lc) -> float:
    """The row's ``lc``, falling back to ``background_lc`` then 10.0 (floored at 0.001)."""
    lc = positive_number(row.get('lc'))
    if lc is None:
        lc = positive_number(background_lc)
    return max(lc if lc is not None else 10.0, 0.001)


def polygon_parts(geom) -> list:
    """Flatten a (Multi)Polygon or collection into its Polygon parts."""
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        parts = []
        for part in geom.geoms:
            parts.extend(polygon_parts(part))
        return parts
    return []


def _merge_close_vertices(ring, tolerance):
    """A ring's coordinates with vertex pairs closer than ``tolerance`` made exact; None if none are non-adjacent."""
    coords = np.asarray(ring.coords, dtype=float)[:-1].copy()
    n = len(coords)
    pairs = sorted(cKDTree(coords).query_pairs(tolerance))
    if not any(1 < j - i < n - 1 for i, j in pairs):
        return None
    # Sorted pairs visit (k, i) before (i, j), so c[i] is already final.
    for i, j in pairs:
        coords[j] = coords[i]
    return np.vstack([coords, coords[:1]])


def unpinch_polygons(geom, tolerance):
    """Rebuild polygons whose rings pass within ``tolerance`` of themselves as valid polygons.

    Overlay output can visit a touch point twice with the two copies a few ulp
    apart -- e.g. a zone minus a buffer footprint whose corner lies on the
    zone outline. GEOS treats that ring as simple, but OCC merges the copies
    within its tolerance and cannot close the wire, so the surface is lost.
    Making the copies exact turns the ring into a plain self-touch, which
    ``make_valid`` resolves into a shell with a touching hole or into
    separate polygons. Other geometries are returned unchanged.
    """
    parts = polygon_parts(geom)
    rebuilt = []
    changed = False
    for poly in parts:
        rings = [poly.exterior, *poly.interiors]
        merged = [_merge_close_vertices(ring, tolerance) for ring in rings]
        if all(coords is None for coords in merged):
            rebuilt.append(poly)
            continue
        changed = True
        coords = [
            np.asarray(ring.coords) if m is None else m
            for ring, m in zip(rings, merged)
        ]
        rebuilt.extend(polygon_parts(make_valid(Polygon(coords[0], coords[1:]))))
    if not changed:
        return geom
    return MultiPolygon(rebuilt)


def line_parts(geom) -> list:
    """Flatten a (Multi)LineString or collection into its non-empty LineString parts."""
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [part for part in geom.geoms if part.length > 0]
    if hasattr(geom, "geoms"):
        parts = []
        for part in geom.geoms:
            parts.extend(line_parts(part))
        return parts
    return []


def sanitize_coords(coords, *, min_spacing=1e-5, require_closed=False, min_points=2) -> list:
    """Drop non-finite and near-duplicate coordinates; [] if fewer than ``min_points`` remain."""
    clean_coords = []
    for pt in coords:
        if len(pt) < 2:
            continue
        x = float(pt[0])
        y = float(pt[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if clean_coords:
            dist = math.sqrt((x - clean_coords[-1][0])**2 + (y - clean_coords[-1][1])**2)
            if dist <= min_spacing:
                continue
        clean_coords.append((x, y))

    if require_closed and len(clean_coords) > 1:
        dist = math.sqrt(
            (clean_coords[0][0] - clean_coords[-1][0])**2 +
            (clean_coords[0][1] - clean_coords[-1][1])**2
        )
        if dist <= min_spacing:
            clean_coords.pop()

    if len(clean_coords) < min_points:
        return []
    return clean_coords
