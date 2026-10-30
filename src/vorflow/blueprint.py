from __future__ import annotations

import logging
import math
import warnings
from contextlib import contextmanager

import geopandas as gpd
import shapely
import shapely.affinity
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union, snap, polygonize
from shapely.prepared import prep
from shapely.validation import make_valid

from . import _checks, _junctions
from ._log import verbosity_scope, warn_user
from ._features import line_parts as _line_parts
from ._features import DEFAULT_CONNECTIVITY_TOLERANCE, HEX_RING_CLEARANCE, is_face_line, positive_number, row_bool
from ._plot import features_frame, get_ax, plot_features
from ._barrier import lines as _barrier_lines
from ._barrier import pipeline as _barrier_pipeline
from .buffer import strip_cells_across
from .fields import DEFAULT_GROWTH_FACTOR, GapField, GeometricGrowthField, MeshField, ThresholdField
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)

# Constants for geometry simplification and reporting
SIGNIFICANT_REDUCTION_PCT = 1.0
# Where embedded lines meet, ends that miss another line by less than this
# fraction of their vertex spacing are snapped onto it, shorter overshoots are
# trimmed, and line vertices closer than it to the junction are dropped (#31).
# 0.5 keeps the nearest line node at least half a cell from the junction node
# while moving any line end by at most half a cell.
DEFAULT_LINE_JUNCTION_FRACTION = 0.5
HEX_RING_ANGLES_DEG = tuple(30.0 + 60.0 * k for k in range(6))
# Gmsh keeps the ring's radius-r edges intact only while the target size inside
# the ring stays above ~0.85 r (measured with a uniform size: 0.85 r kept the
# hexagon, 0.80 r split it). 0.9 leaves a margin for the linear size estimate.
HEX_RING_MIN_SIZE_RATIO = 0.9


def _validate_growth_factor(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"growth_factor must be a number greater than 1.0 (or None). Got {value!r}."
        )
    if not math.isfinite(value):
        raise ValueError(f"growth_factor must be finite. Got {value}.")
    if value <= 1.0:
        raise ValueError(f"growth_factor must be greater than 1.0. Got {value}.")
    return float(value)


def _feature_options(fields, gap_wall):
    """Validated ``fields`` (None becomes []) and ``gap_wall`` for an add_* call."""
    if not isinstance(gap_wall, (bool, np.bool_)):
        raise ValueError(f"gap_wall must be True or False. Got {gap_wall!r}.")
    if fields is None:
        fields = []
    items = fields if isinstance(fields, (list, tuple, set)) else [fields]
    if any(isinstance(f, GapField) for f in items):
        raise ValueError(
            "GapField is model-wide: pass it as MeshGenerator(gap_field=GapField(...)), "
            "not in a feature's fields."
        )
    return fields, bool(gap_wall)


def _coerce_line_junction_fraction(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"line_junction_fraction must be a non-negative number. Got {value!r}.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"line_junction_fraction must be finite and non-negative. Got {value}.")
    return float(value)


def _line_densify_spacing(feature) -> float | None:
    """Densification spacing of a line: a numeric ``densify``, else its ``lc``; None when ``densify=False``."""
    d = feature.get('densify')
    if d is False:
        return None
    if isinstance(d, (int, float)) and not isinstance(d, bool) and d > 0:
        return d
    return feature.get('lc')


def _line_vertex_spacing(feature) -> float | None:
    """Spacing of a line's nodes: its densification spacing, else (``densify=False``) its ``lc``."""
    spacing = positive_number(_line_densify_spacing(feature))
    return spacing if spacing is not None else positive_number(feature.get('lc'))


def _empty_junction_diagnostics() -> dict:
    """The ``junction_diagnostics`` of a run that joined no lines."""
    return {'junctions': [], 'snapped_ends': [], 'trimmed_ends': []}


def _junction_diagnostics(line_ids, junctions, moves, trims) -> dict:
    """Readable record of the junction step, keyed by ``line_id``."""
    end_name = {0: 'start', -1: 'end'}
    lines_at = {}
    for k, points in enumerate(junctions):
        for xy in points:
            lines_at.setdefault(xy, []).append(line_ids[k])
    return {
        'junctions': [{'point': xy, 'line_ids': ids} for xy, ids in lines_at.items()],
        'snapped_ends': [
            {'line_id': line_ids[k], 'end': end_name[end], 'from': old, 'to': new}
            for k, end, old, new in moves
        ],
        'trimmed_ends': [
            {'line_id': line_ids[k], 'end': end_name[end], 'length': cut}
            for k, end, cut in trims
        ],
    }


def _takes_part_in_junctions(feature) -> bool:
    """True for embedded plain lines: not field-only, straddle/barrier or quad-buffer lines."""
    geom = feature.get('geometry')
    return (
        isinstance(geom, LineString)
        and geom.length > 0
        and row_bool(feature, 'embed', True)
        and not row_bool(feature, 'quad_buffer', False)
        and not is_face_line(feature)
        and _line_vertex_spacing(feature) is not None
    )


def _coerce_connectivity_tolerance(value, parameter_name="connectivity_tolerance"):
    if isinstance(value, bool):
        raise ValueError(
            f"{parameter_name} must be a non-negative number. Boolean values are not supported."
        )
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{parameter_name} must be a non-negative number. Got {type(value).__name__}."
        )
    if value < 0:
        raise ValueError(f"{parameter_name} must be non-negative. Got {value}.")
    return float(value)


def _simplify_keeping_shared_edges(geom, tol, neighbours):
    """Simplify a polygon without moving boundary it shares with ``neighbours``.

    Simplifying adjacent polygons independently pulls their common edge apart
    and opens gaps in the domain. Boundary that coincides with a neighbour's
    boundary is kept verbatim; only the free parts are simplified, and the
    polygon is rebuilt from the resulting linework. Falls back to the
    unsimplified geometry (with a warning) if the rebuild is not a valid
    areal geometry.
    """
    others = [n.boundary for n in neighbours if n.intersects(geom)]
    shared = unary_union(others).intersection(geom.boundary) if others else None
    shared_lines = _line_parts(shared) if shared is not None else []
    if not shared_lines:
        return geom.simplify(tol, preserve_topology=True)

    shared_union = unary_union(shared_lines)
    free = geom.boundary.difference(shared_union)
    free_lines = [part.simplify(tol, preserve_topology=True) for part in _line_parts(free)]
    faces = list(polygonize(unary_union(shared_lines + free_lines)))
    kept = [face for face in faces if face.intersection(geom).area > 0.5 * face.area]
    rebuilt = unary_union(kept) if kept else None
    if rebuilt is None or rebuilt.is_empty or not rebuilt.is_valid or rebuilt.area <= 0:
        warnings.warn(
            "Could not simplify a polygon while preserving its shared edges; "
            "keeping it unsimplified.",
            stacklevel=3,
        )
        return geom
    return rebuilt


def _deduplicate_points(points):
    """Drop points within tolerance of a finer point; keep insertion order.

    Points are visited finest-first (by ``lc``). A point is dropped when it
    lies closer than ``max(its tolerance, the kept point's tolerance)`` to an
    already kept point, so the result does not depend on which of a close
    pair carries the tolerance. Returns the kept points in their original
    order.
    """
    def tolerance(point):
        tol = point.get("simplify_tolerance")
        return float(tol) if tol is not None and tol > 0 else 0.0

    order = sorted(
        range(len(points)),
        key=lambda i: points[i]["lc"] if points[i]["lc"] is not None else float("inf"),
    )
    max_tol = max(tolerance(p) for p in points)
    if max_tol == 0.0:
        return list(points)

    geoms = [p["geometry"] for p in points]
    tree = STRtree(geoms)
    kept = set()
    for i in order:
        own_tol = tolerance(points[i])
        candidates = tree.query(geoms[i].buffer(max_tol))
        merged_into = next(
            (
                j for j in candidates
                if j in kept and geoms[j].distance(geoms[i]) < max(own_tol, tolerance(points[j]))
            ),
            None,
        )
        if merged_into is None:
            kept.add(i)
        elif points[i].get("hex_ring", False):
            warnings.warn(
                f"Point {points[i]['point_id']!r} was merged into point "
                f"{points[merged_into]['point_id']!r} by simplify_tolerance; "
                "its hex_ring is dropped.",
                UserWarning,
                stacklevel=5,
            )
    return [points[i] for i in sorted(kept)]


def _hex_ring_seeds(point, lc: float) -> list[tuple[float, float]]:
    """The six hex-ring seeds at radius ``lc`` around ``point``, at 30 + k * 60 degrees."""
    return [
        (point.x + lc * math.cos(math.radians(angle)), point.y + lc * math.sin(math.radians(angle)))
        for angle in HEX_RING_ANGLES_DEG
    ]


def _meshed_half_width(feature) -> float:
    """Half-width of the node band a line or polygon feature meshes about its geometry.

    Quad buffers straddle the geometry by ``cells_across * lc / 2``; plain
    features put their nodes on the geometry itself, and barrier sites stay
    within half a face length of the line, which the barrier sizing keeps
    clear of every feature node (zero width here). An unset ``lc``
    (background resolution, unknown here) counts as zero width.
    """
    lc = positive_number(feature.get('lc'))
    if row_bool(feature, 'quad_buffer', False):
        return strip_cells_across(feature) * lc / 2.0 if lc is not None else 0.0
    return 0.0


def _hex_ring_conflict(point, lc: float, obstacles) -> str | None:
    """Why a hex ring around ``point`` would collide with an obstacle, or None if it fits.

    ``obstacles`` holds ``(label, geometry, half_width)``; each must keep
    ``HEX_RING_CLEARANCE * lc`` between the point and its meshed band
    (``half_width`` either side of the geometry).
    """
    clearance = HEX_RING_CLEARANCE * lc
    for label, geometry, half_width in obstacles:
        gap = geometry.distance(point) - half_width
        if gap < clearance:
            return f"{label} is {max(gap, 0.0):.6g} away (needs >= {clearance:.6g})"
    return None


def _optional_number(feature, key):
    """``feature[key]`` as a float, or None when it is missing or NaN."""
    value = feature.get(key)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _size_ramps(feature) -> list[tuple[float, float, float | None]]:
    """Linear models ``(size0, plateau, gradient)`` of the size fields a feature sets.

    The size is ``size0`` up to ``plateau`` from the feature and grows at
    ``gradient`` beyond it (None: unknown, as the growth needs
    ``background_lc``). Mirrors the engine's ``_feature_fields``: the default
    GeometricGrowthField, the legacy dist_min/dist_max ThresholdField, and
    explicit GeometricGrowthField/ThresholdField entries. Other explicit fields
    and the deprecated border grading are not modelled; the engine's
    post-mesh ring check covers them.
    """
    lc = positive_number(feature.get('lc'))
    fields = feature.get('fields')
    explicit = [f for f in fields if isinstance(f, MeshField)] if isinstance(fields, (list, tuple)) else []
    ramps = []
    dist_min = _optional_number(feature, 'dist_min')
    if lc is not None and (dist_min is not None or _optional_number(feature, 'dist_max') is not None):
        plateau = max(dist_min if dist_min is not None else lc, 0.5 * lc)
        ramps.append((lc, plateau, None))
    elif lc is not None and not explicit:
        growth_factor = _optional_number(feature, 'growth_factor') or DEFAULT_GROWTH_FACTOR
        ramps.append((lc, 0.0, growth_factor - 1.0))
    for field in explicit:
        if isinstance(field, GeometricGrowthField) and lc is not None:
            ramps.append((lc, 0.0, field.gradient))
        elif isinstance(field, ThresholdField):
            span = field.dist_max - field.dist_min
            gradient = None
            if field.size_max is not None and span > 0:
                gradient = (field.size_max - field.size_min) / span
            ramps.append((field.size_min, field.dist_min, gradient))
    return ramps


def _ramp_size(ramp, distance: float) -> float:
    """Target size a ``(size0, plateau, gradient)`` ramp sets at ``distance``; inf if unknown."""
    size0, plateau, gradient = ramp
    if distance <= plateau:
        return size0
    if gradient is None:
        return math.inf
    return size0 + gradient * (distance - plateau)


def _hex_ring_size_conflict(ring, lc: float, sources) -> str | None:
    """Why the target mesh size inside a hex ring would be too fine to keep it, or None.

    ``ring`` is the hexagon through the ring seeds; ``sources`` holds
    ``(label, geometry, half_width, ramps)``. Each source's size is
    estimated at its nearest approach to the hexagon and must stay at or
    above ``HEX_RING_MIN_SIZE_RATIO * lc``, or Gmsh splits the ring edges.
    """
    min_size = HEX_RING_MIN_SIZE_RATIO * lc
    for label, geometry, half_width, ramps in sources:
        if not ramps:
            continue
        distance = max(geometry.distance(ring) - half_width, 0.0)
        size = min(_ramp_size(ramp, distance) for ramp in ramps)
        if size < min_size:
            return (
                f"{label} sets a mesh size of ~{size:.6g} at the ring "
                f"(needs >= {min_size:.6g})"
            )
    return None


def _check_background_lc(background_lc, clean_frames) -> float:
    """``background_lc`` for check(): the given value, else the coarsest feature resolution."""
    if background_lc is None:
        resolutions = [
            lc for frame in clean_frames if 'lc' in frame.columns
            for lc in (positive_number(value) for value in frame['lc']) if lc is not None
        ]
        if not resolutions:
            raise ValueError(
                "check() needs background_lc when no feature has a resolution."
            )
        return max(resolutions)
    value = None if isinstance(background_lc, bool) else positive_number(background_lc)
    if value is None or not math.isfinite(value):
        raise ValueError(
            f"background_lc must be a positive finite number. Got {background_lc!r}."
        )
    return value


def _warn_deprecated(message):
    """Emit a DeprecationWarning pointing at the caller of add_polygon()."""
    warnings.warn(message, DeprecationWarning, stacklevel=4)


def _resolve_legacy_polygon_args(densify, dist_max, mesh_refinement, border_density,
                                 dist_max_in, dist_max_out):
    """Map pre-0.1 add_polygon keywords onto the current API.

    Returns ``(densify, dist_max, border_lc)``.
    """
    if mesh_refinement is not None:
        _warn_deprecated("add_polygon(mesh_refinement=...) is deprecated and has no effect; omit it.")
    if dist_max_out is not None:
        _warn_deprecated("add_polygon(dist_max_out=...) is deprecated; use dist_max instead.")
        assert dist_max is None or dist_max == dist_max_out, (
            "dist_max and dist_max_out were both given with different values."
        )
        dist_max = dist_max_out
    border_lc = None
    if border_density is not None:
        _warn_deprecated(
            "add_polygon(border_density=...) is deprecated; use densify=<spacing> for "
            "boundary densification."
        )
        assert border_density > 0, f"border_density must be positive. Got {border_density}."
        border_lc = float(border_density)
        if densify is None:
            densify = border_lc
    if dist_max_in is not None:
        _warn_deprecated("add_polygon(dist_max_in=...) is deprecated.")
        if border_density is None:
            warnings.warn(
                "dist_max_in only applies together with border_density; it is ignored.",
                UserWarning,
                stacklevel=4,
            )
    return densify, dist_max, border_lc


def _warn_if_geographic_crs(crs):
    """Warn when the project CRS is geographic (lat/lon degrees).

    Mesh sizes (resolution, background_lc, buffer thicknesses) are expressed
    in CRS units, and MODFLOW needs real length units for cell areas and
    conductances -- meshing in degrees is almost certainly a mistake.
    """
    if crs is None:
        return
    try:
        from pyproj import CRS
        is_geographic = CRS.from_user_input(crs).is_geographic
    except Exception:
        # Unparseable CRS spec: GeoDataFrame construction will surface it.
        return
    if is_geographic:
        warnings.warn(
            f"The project CRS ({crs}) is geographic (latitude/longitude "
            "degrees). Mesh sizes will be in degrees and the resulting "
            "MODFLOW grid will be physically meaningless (1 deg of longitude "
            "and latitude differ in length, and cell areas/conductances need "
            "length units). Reproject your data to a projected CRS (e.g. a "
            "UTM zone) with GeoDataFrame.to_crs() before building the mesh.",
            stacklevel=3,
        )


class ConceptualMesh:
    """
    Conceptual model: the raw geometric inputs of a vorflow mesh.

    This class acts as a staging area for geometric features (polygons, lines,
    points) before they are processed into a clean, non-overlapping set of
    inputs for the mesh generator.

    Args:
        crs: The coordinate reference system for the project. Use a
            *projected* CRS (e.g. a UTM zone like "EPSG:32618") so mesh
            sizes are in real length units. Geographic CRSs (lat/lon
            degrees, e.g. "EPSG:4326") produce physically meaningless
            grids and trigger a warning -- reproject your data first with
            GeoDataFrame.to_crs(). None (default) means local/unspecified
            coordinates; output GeoDataFrames then carry no CRS.
        connectivity_tolerance (float, optional): Default snapping tolerance used
            during topology cleanup in generate(). Larger values make lines and
            points connect more aggressively to nearby geometry.
        line_junction_fraction (float, optional): Where embedded lines meet
            or nearly meet, a line end that misses another line by less than
            this fraction of its vertex spacing (``densify`` if numeric,
            else ``resolution``) is snapped onto it, an overshoot shorter
            than that is trimmed, each junction becomes a vertex of every
            line through it, and line vertices closer than that to a
            junction are dropped, so one node sits at the junction instead
            of two a few units apart. Barrier/straddle, quad-buffer and
            field-only lines are left out; line ends on polygon boundaries
            never move. Default 0.5; 0 turns the step off. Each
            ``generate()`` records what it did in ``junction_diagnostics``:
            ``junctions`` (``point`` and the ``line_ids`` through it),
            ``snapped_ends`` (``line_id``, ``end``, ``from``, ``to``) and
            ``trimmed_ends`` (``line_id``, ``end``, cut ``length``).
        keep_close_rings (bool): Keep a point's ``hex_ring`` when a barrier
            passes within ``HEX_RING_CLEARANCE * resolution`` of the point.
            By default such a ring is dropped with a warning, as next to any
            embedded line; with True the barrier faces are made shorter
            there so the ring seeds stay clear of the barrier
            construction. It is a ``ConceptualMesh`` option because the
            rings are decided here, in ``generate()``. Default False.
    """

    def __init__(self, crs=None, connectivity_tolerance=DEFAULT_CONNECTIVITY_TOLERANCE,
                 line_junction_fraction=DEFAULT_LINE_JUNCTION_FRACTION, keep_close_rings=False):
        self.crs = crs
        _warn_if_geographic_crs(crs)
        self.connectivity_tolerance = _coerce_connectivity_tolerance(connectivity_tolerance)
        self.line_junction_fraction = _coerce_line_junction_fraction(line_junction_fraction)
        if not isinstance(keep_close_rings, (bool, np.bool_)):
            raise ValueError(f"keep_close_rings must be True or False. Got {keep_close_rings!r}.")
        self.keep_close_rings = bool(keep_close_rings)
        self.junction_diagnostics = _empty_junction_diagnostics()
        # Store raw geometric inputs before processing.
        self.raw_polygons = [] 
        self.raw_lines = []
        self.raw_points = []
        
        # Geometries after cleaning, snapping, and processing.
        self.domain_boundary = None
        self.clean_polygons = gpd.GeoDataFrame()
        self.clean_lines = gpd.GeoDataFrame()
        self.clean_points = gpd.GeoDataFrame()

    def add_polygon(
        self,
        geometry,
        zone_id,
        resolution=None,
        z_order=0,
        dist_min=None,
        dist_max=None,
        densify=None,
        fields=None,
        embed=True,
        simplify_tolerance=None,
        quad_buffer=False,
        quad_buffer_thickness=1,
        growth_factor=None,
        mesh_refinement=None,
        border_density=None,
        dist_max_in=None,
        dist_max_out=None,
        label=None,
        gap_wall=True,
    ):
        """
        Adds a polygon feature, such as a model boundary or a refinement zone.

        Args:
            geometry (shapely.Polygon): The polygon geometry.
            zone_id (int or str): A unique identifier for the zone.
            resolution (float, optional): Target mesh size within this polygon. If None,
                the background mesh size will be used.
            z_order (int): Stacking order for resolving overlaps. Higher values are
                processed first and will "cut" into lower-order polygons. Also acts
                as the crossing priority when this polygon's quad buffer crosses
                another quad-buffered feature (higher stays continuous). To control
                the two independently, add the zone without ``quad_buffer`` and add
                its boundary as a separate quad-buffered line with its own z_order.
            growth_factor (float, optional): Cell-to-cell growth ratio (>1.0) for the default
                GeometricGrowthField size transition away from the feature. Defaults to 1.2.
                Ignored when an explicit ``fields`` list or the legacy dist_min/dist_max is given.
            densify (float|bool|None, optional): Controls polygon boundary densification:
                - If False, disables densification.
                - If True, densifies using `resolution` (lc). Requires `resolution` to be set.
                - If a float, densifies so no boundary segment is longer than this value.
                Raises ValueError if non-positive when specified as a float.
            simplify_tolerance (float|int|None, optional): If a number > 0, applies Douglas-Peucker
                simplification with this tolerance. If None or 0, no simplification is applied.
                Boundary shared exactly with another polygon is left unsimplified so
                neighbouring zones never open gaps. Raises ValueError if negative.
                Boolean values are not supported.
            fields (list, optional): List of MeshField objects.
            embed (bool): If True, the polygon is embedded in the mesh. If False, it is used only for
                fields: it refines the mesh to ``resolution`` throughout its interior (holes excluded)
                and grows from its boundary like an embedded polygon, but adds no mesh edges, does not
                cut the domain and never becomes a zone.
            quad_buffer (bool): If True, replaces the meshed polygon outline
                with a quad band straddling it (the annulus between the
                ``+/- thickness * resolution / 2`` offsets, meshed as
                recombined quads; annuli cannot be 4-corner transfinite). The
                outline itself never becomes mesh edges: with
                ``quad_buffer_thickness=1`` no nodes fall on it, so the Voronoi
                cell faces trace the shapefile shape; with ``2`` a node row
                lands on it, giving a row of ~square cells centered on the
                shape. (For the triangular element-grid deliverable the reading
                is opposite: ``2`` puts element edges along the outline.)
                When the band crosses another quad buffer, the higher-priority
                feature (``z_order``, then finer resolution, wider strip, lines
                over polygons) stays continuous and only the other is trimmed
                with a warning; polygons narrower than the band fall back to a
                plain outline with a warning.
            quad_buffer_thickness (int): Band width in local cell widths
                (multiples of ``resolution``, like gmshflow's ``cs_thick``).
                Supported values are 1 and 2.
            label (str, optional): Legend label for ``plot()``. Features sharing a
                label share one legend entry; unlabelled features get none.
            gap_wall (bool): If True (default), the polygon is a wall that
                ``GapField`` measures gaps against. Set False for features whose
                narrow gaps should not be refined, e.g. a sliver polygon. Its whole
                boundary, including edges shared with other polygons, is then
                exempt.

        Deprecated parameters:
            dist_min (float, optional): Distance from the polygon boundary where
                the mesh size is held constant at the boundary's resolution. Supplying
                dist_min/dist_max selects the legacy linear ThresholdField transition and
                emits a DeprecationWarning; omit them to use the default GeometricGrowthField.
            dist_max (float, optional): Distance from the polygon boundary over
                which the mesh transitions to the background resolution. See dist_min.
            mesh_refinement (bool, optional): Has no effect (it was
                never read by the mesh generator).
            border_density (float, optional): Densifies the boundary
                to this spacing (use ``densify``) and grades the interior from this
                finer border size up to ``resolution`` over ``dist_min`` to
                ``dist_max_in``.
            dist_max_in (float, optional): Interior grading distance
                for ``border_density``.
            dist_max_out (float, optional): Alias for ``dist_max``.
        """
        if not geometry.is_valid:
            geometry = make_valid(geometry)

        densify, dist_max, border_lc = _resolve_legacy_polygon_args(
            densify, dist_max, mesh_refinement, border_density, dist_max_in, dist_max_out
        )

        if isinstance(simplify_tolerance, bool):
            raise ValueError(
                "simplify_tolerance must be a non-negative number (or None/0 to disable). "
                "Boolean values are not supported."
            )
        if isinstance(simplify_tolerance, (int, float)) and simplify_tolerance < 0:
            raise ValueError(f"simplify_tolerance must be non-negative. Got {simplify_tolerance}.")

        if isinstance(densify, (int, float)) and not isinstance(densify, bool) and densify <= 0:
            raise ValueError(f"densify must be positive when specified as a float. Got {densify}.")
        if densify is True and (resolution is None or resolution <= 0):
            raise ValueError("densify=True for polygons requires a positive `resolution` (lc).")

        if resolution is not None and resolution <= 0:
            raise ValueError(f"resolution must be positive (or None). Got {resolution}.")

        if dist_max is not None and dist_max < 0:
            raise ValueError(f"dist_max must be non-negative (or None). Got {dist_max}.")
    
        if dist_min is not None and dist_min < 0:
            raise ValueError(f"dist_min must be non-negative (or None). Got {dist_min}.")

        if quad_buffer_thickness not in (1, 2):
            raise ValueError("quad_buffer_thickness must be either 1 or 2.")
        if quad_buffer and not embed:
            raise ValueError("quad_buffer=True requires embed=True: a quad buffer is meshed geometry.")

        growth_factor = _validate_growth_factor(growth_factor)

        fields, gap_wall = _feature_options(fields, gap_wall)

        self.raw_polygons.append(
            {
                "geometry": geometry,
                "zone_id": zone_id,
                "lc": resolution,
                "z_order": z_order,
                "dist_min": dist_min,
                "dist_max": dist_max,
                "densify": densify,
                "simplify_tolerance": simplify_tolerance,
                'fields': fields,
                'embed': embed,
                'quad_buffer': bool(quad_buffer),
                'quad_buffer_thickness': int(quad_buffer_thickness),
                'growth_factor': growth_factor,
                'border_lc': border_lc,
                'dist_max_in': dist_max_in if border_lc is not None else None,
                'label': label,
                'gap_wall': gap_wall,
            }
        )

    def add_line(self, geometry, line_id, resolution, snap_to_polygons=True, is_barrier=False,
                  dist_min=None, dist_max=None, straddle_width=None, fields=None, embed=True, densify=True,
                  simplify_tolerance=None, quad_buffer=False, quad_buffer_thickness=1, z_order=0,
                  growth_factor=None, label=None, color=None, gap_wall=True):
        """
        Adds a line feature, such as a river, fault, or other linear boundary.

        Args:
            geometry (shapely.LineString): The line geometry.
            line_id (str): A unique identifier for the line.
            resolution (float): Target mesh size along the line.
            snap_to_polygons (bool): If True, the line's endpoints will be snapped to
                nearby polygon boundaries to ensure connectivity.
            is_barrier (bool): If True, the line is treated as a flow barrier: the
                cells are built so that the line is made of whole cell faces, each
                orthogonal to the line joining its two generators (see the barrier
                notes below).
            fields (list, optional): List of MeshField objects.
            embed (bool): If True, the line is embedded in the mesh. If False, it is used only for fields.
            densify (float or bool, optional): Controls line densification:
                - If False, disables densification.
                - If True, densifies the line using the `resolution` value.
                - If a float, densifies the line so that no segment is longer than this value.
                Raises ValueError if negative or zero.
            simplify_tolerance (float|int|None, optional): If a number > 0, simplifies the line with
                this tolerance using Douglas-Peucker algorithm. If None or 0, no simplification is applied.
                Raises ValueError if negative. Boolean values are not supported. For a barrier it is
                also the snap radius for near-miss ends, and the simplification keeps junctions
                and crossings (see the barrier notes below).
            quad_buffer (bool): If True, creates an opt-in structured quad buffer
                strip around the line instead of the exact barrier construction.
                The strip is meshed as a 4-corner transfinite surface (structured
                quad rows): with ``quad_buffer_thickness=1`` no nodes fall on
                the line, so the Voronoi faces align with it (sharp barrier);
                with ``2`` a node row lands on it, giving a row of ~square
                Voronoi cells centered on the line. (For the triangular
                element-grid deliverable the reading is opposite: ``2`` puts
                element edges along the line.) On a barrier
                (``is_barrier=True``) no node row ever lands on the line:
                ``quad_buffer_thickness`` is the number of cell rows on each
                side of it, so ``2`` gives a strip three cells wide whose
                middle faces follow the line. Where the line leaves the
                domain, the strip runs to the boundary on both sides. When two quad buffers cross,
                the higher-priority one stays continuous and only the other is
                trimmed at the crossing with a warning (priority: ``z_order``,
                then finer resolution, wider strip, lines over polygons,
                insertion order). Strips crossing a barrier's corridor
                are still trimmed there. Independent of ``is_barrier``.
            quad_buffer_thickness (int): Strip width in local cell widths
                (multiples of ``resolution``, like gmshflow's ``cs_thick``);
                on a barrier, cell rows on each side of the line (a strip
                ``2 * thickness - 1`` cells wide). Supported values are 1 and 2.
            z_order (int): Crossing priority for quad buffers. When this
                feature's quad buffer crosses another one, the feature with the
                higher ``z_order`` keeps its strip continuous and the other is
                trimmed. Lines do not participate in polygon overlap stacking.
            growth_factor (float, optional): Cell-to-cell growth ratio (>1.0) for the default
                GeometricGrowthField size transition away from the line. Defaults to 1.2.
                Ignored when an explicit ``fields`` list or the legacy dist_min/dist_max is given.
            label (str, optional): Legend label for ``plot()``. Features sharing a
                label share one legend entry; unlabelled features get none.
            color (optional): Matplotlib colour for the line in ``plot()``. None
                (default) draws barriers red and other lines blue.
            gap_wall (bool): If True (default), the line is a wall that
                ``GapField`` measures gaps against. Set False for features whose
                narrow gaps should not be refined, e.g. a boundary-condition line
                digitised just inside the domain edge.

        Barriers:
            ``generate()`` cleans every barrier (embedded, no quad buffer) before
            meshing: an end within ``max(connectivity_tolerance, simplify_tolerance)``
            of another barrier, of its vertex or of the domain boundary is snapped
            onto it (a short overshoot is trimmed, and a branch that collapses onto
            the barrier it starts on is dropped); parts lying on the domain boundary
            are dropped with a warning; junctions and crossings become
            shared vertices; a ``simplify_tolerance`` simplifies by the fewest chords
            while keeping them; and vertices closer than 0.1 x resolution to a
            neighbour (digitising noise) are always dropped, with a warning giving
            the largest move. Barriers are not densified. ``MeshGenerator`` then
            places the barrier's cell generators in closed form so each barrier
            segment is exactly one cell face (``MeshGenerator.barrier_report`` lists
            what it had to give up, such as the tip of a very sharp spike), and
            raises ``vorflow.BarrierMeshError`` when it cannot, or when a point
            feature lies on a barrier.

        Deprecated parameters:
            dist_min (float, optional): Distance from the line where the mesh size
                is held constant at the line's resolution. Supplying dist_min/dist_max selects
                the legacy linear ThresholdField transition and emits a DeprecationWarning;
                omit them to use the default GeometricGrowthField.
            dist_max (float, optional): Distance from the line over which the mesh
                transitions to the background resolution. See dist_min.
            straddle_width (float, optional): Deprecated, with a FutureWarning; its value is
                ignored (it no longer sets any offset or trimming width). It set the width of the
                node pairs that made cell faces follow the line; the faces are now placed exactly
                and automatically along every barrier. On a line that is not a barrier, any
                positive value still makes the cell faces follow the line exactly: that is the
                only switch for exact faces without ``is_barrier=True`` (which gives the same
                faces and lists the line with ``is_barrier`` True in
                ``MeshGenerator.barrier_lines``).
        """
        if straddle_width is not None:
            warn_user(
                "straddle_width is deprecated and its value is ignored. Along a barrier (is_barrier=True) the "
                "cell faces follow the line exactly and automatically, so remove it there. On a line that is not "
                "a barrier, any positive straddle_width still makes the faces follow the line exactly; it is the "
                "only way to get that without is_barrier=True, which gives the same faces and marks the line "
                "as a barrier in MeshGenerator.barrier_lines.",
                FutureWarning,
            )
        if not geometry.is_valid:
            geometry = make_valid(geometry)

        if isinstance(simplify_tolerance, bool):
            raise ValueError(
                "simplify_tolerance must be a non-negative number (or None/0 to disable). "
                "Boolean values are not supported."
            )
        if isinstance(simplify_tolerance, (int, float)) and simplify_tolerance < 0:
            raise ValueError(f"simplify_tolerance must be non-negative. Got {simplify_tolerance}.")
        
        if isinstance(densify, (int, float)) and not isinstance(densify, bool) and densify <= 0:
            raise ValueError(f"densify must be positive when specified as a float. Got {densify}.")

        if quad_buffer_thickness not in (1, 2):
            raise ValueError("quad_buffer_thickness must be either 1 or 2.")
        if quad_buffer and not embed:
            raise ValueError("quad_buffer=True requires embed=True: a quad buffer is meshed geometry.")

        growth_factor = _validate_growth_factor(growth_factor)

        fields, gap_wall = _feature_options(fields, gap_wall)

        self.raw_lines.append({
            'geometry': geometry,
            'line_id': line_id,
            'lc': resolution,
            'snap_to_polygons': snap_to_polygons,
            'is_barrier': is_barrier,
            'dist_min': dist_min,
            'dist_max': dist_max,
            'straddle_width': straddle_width,
            'fields': fields,
            'embed': embed,
            'densify': densify,
            'simplify_tolerance': simplify_tolerance,
            'quad_buffer': bool(quad_buffer),
            'quad_buffer_thickness': int(quad_buffer_thickness),
            'z_order': z_order,
            'growth_factor': growth_factor,
            'label': label,
            'color': color,
            'gap_wall': gap_wall,
        })

    def add_point(self, geometry, point_id, resolution, dist_min=None, dist_max=None, fields=None,
                  embed=True, simplify_tolerance=None, growth_factor=None, hex_ring=False,
                  label=None, gap_wall=True):
        """
        Adds a point feature, such as a well or an observation point.

        Args:
            geometry (shapely.Point): The point geometry.
            point_id (str): A unique identifier for the point.
            resolution (float): Target mesh size at the point.
            fields (list, optional): List of MeshField objects. When given, they replace
                the default size transition around the point.
            embed (bool): If True (default), the point becomes a mesh node (and so a
                Voronoi generator). If False, it is used only for size fields.
            simplify_tolerance (float|int|None, optional): If a number > 0, merges points that are closer
                than this tolerance, keeping the one with the finest resolution. Two points merge
                when either one's tolerance covers their separation. If None or 0, this point does
                not trigger merging. Raises ValueError if negative. Boolean values are not supported.
            growth_factor (float, optional): Cell-to-cell growth ratio (>1.0) for the default
                GeometricGrowthField size transition away from the point. Defaults to 1.2.
                Ignored when an explicit ``fields`` list or the legacy dist_min/dist_max is given.
            hex_ring (bool): If True, six fixed seed nodes are added around the point
                at radius ``resolution`` (angles 30 + k * 60 degrees), so the point's
                Voronoi cell is a regular hexagon centred on it with apothem
                ``resolution / 2``. Requires ``embed=True`` and a positive
                ``resolution``. The ring adds cells around the point (about 40 with
                the default growth_factor). It is built about the point's final position
                after snapping and clipping in ``generate()``, and dropped with a
                warning when another embedded feature (polygon boundary, line or
                its buffer band, other point) lies closer than
                ``HEX_RING_CLEARANCE * resolution`` to the point, when a size field
                is estimated finer than ``HEX_RING_MIN_SIZE_RATIO * resolution``
                inside the ring (e.g. an enclosing zone with a finer resolution, or
                a nearby fine line), or when ``simplify_tolerance`` merges the point
                into another. Size fields the estimate cannot model (explicit fields
                other than GeometricGrowthField/ThresholdField, or a background_lc
                below the ring radius) are caught after meshing: ``MeshGenerator``
                warns when a ring was split and records the outcome per point in
                ``diagnostics['hex_rings']``.
            label (str, optional): Legend label for ``plot()``. Features sharing a
                label share one legend entry; unlabelled features get none.
            gap_wall (bool): If True (default), the point is a wall that
                ``GapField`` measures gaps against. Set False for features whose
                narrow gaps should not be refined, e.g. an observation point close to
                a line.

        Deprecated parameters:
            dist_min (float, optional): Distance from the point where the mesh size
                is held constant at the point's resolution. Supplying dist_min/dist_max selects
                the legacy linear ThresholdField transition and emits a DeprecationWarning;
                omit them to use the default GeometricGrowthField.
            dist_max (float, optional): Distance from the point over which the mesh
                transitions to the background resolution. See dist_min.
        """
        if not isinstance(hex_ring, (bool, np.bool_)):
            raise ValueError(f"hex_ring must be True or False. Got {hex_ring!r}.")
        if hex_ring and not embed:
            raise ValueError("hex_ring=True requires embed=True: the ring seeds are mesh nodes.")
        if hex_ring and positive_number(resolution) is None:
            raise ValueError(
                f"hex_ring=True requires a positive resolution (the ring radius). Got {resolution!r}."
            )
        if isinstance(simplify_tolerance, bool):
            raise ValueError(
                "simplify_tolerance must be a non-negative number (or None/0 to disable). "
                "Boolean values are not supported."
            )
        if isinstance(simplify_tolerance, (int, float)) and simplify_tolerance < 0:
            raise ValueError(f"simplify_tolerance must be non-negative. Got {simplify_tolerance}.")

        growth_factor = _validate_growth_factor(growth_factor)

        fields, gap_wall = _feature_options(fields, gap_wall)
        self.raw_points.append({
            'geometry': geometry,
            'point_id': point_id,
            'lc': resolution,
            'dist_min': dist_min,
            'dist_max': dist_max,
            'fields': fields,
            'embed': embed,
            'simplify_tolerance': simplify_tolerance,
            'growth_factor': growth_factor,
            'hex_ring': bool(hex_ring),
            'label': label,
            'gap_wall': gap_wall,
        })

    def _apply_simplification(self):
        """
        Applies geometry simplification to raw polygons, lines, and points
        based on their specified tolerances to reduce geometric complexity.

        For polygons and lines the Douglas-Peucker algorithm is used.
        For points, this method performs deduplication: points that are within
        a specified tolerance of each other are merged, and only the point with the
        finest (smallest) resolution is kept.
        """
        # Simplify Polygons. Neighbours are compared in their original form so
        # both sides of a shared edge keep it intact.
        original_polygons = [poly_data['geometry'] for poly_data in self.raw_polygons]
        for i, poly_data in enumerate(self.raw_polygons):
            tol = poly_data.get('simplify_tolerance')
            if isinstance(tol, bool):
                raise ValueError(
                    "simplify_tolerance must be a non-negative number (or None/0 to disable). "
                    "Boolean values are not supported."
                )

            if tol is not None and tol > 0:
                org_area = poly_data['geometry'].area
                neighbours = original_polygons[:i] + original_polygons[i + 1:]
                simplified_geom = _simplify_keeping_shared_edges(poly_data['geometry'], tol, neighbours)
                self.raw_polygons[i]['geometry'] = simplified_geom
                new_area = simplified_geom.area
                if new_area < org_area and org_area > 0:
                    reduction_pct = 100 * (org_area - new_area) / org_area
                    if reduction_pct > SIGNIFICANT_REDUCTION_PCT:
                        logger.info(
                                  f"Simplified polygon (zone_id={poly_data['zone_id']}) "
                                  f"reduced area by {reduction_pct:.2f}% using tolerance {tol}."
                        )

        # Simplify Lines
        for i, line_data in enumerate(self.raw_lines):
            tol = line_data.get('simplify_tolerance')
            if isinstance(tol, bool):
                raise ValueError(
                    "simplify_tolerance must be a non-negative number (or None/0 to disable). "
                    "Boolean values are not supported."
                )

            # Barriers are simplified after their junctions are joined (_clean_barrier_lines).
            if tol is not None and tol > 0 and not is_face_line(line_data):
                org_length = line_data['geometry'].length
                simplified_geom = line_data['geometry'].simplify(tol, preserve_topology=True)
                self.raw_lines[i]['geometry'] = simplified_geom
                new_length = simplified_geom.length
                if new_length < org_length and org_length > 0:
                    reduction_pct = 100 * (org_length - new_length) / org_length
                    if reduction_pct > SIGNIFICANT_REDUCTION_PCT:
                        logger.info(
                                  f"Simplified line (line_id={line_data['line_id']}) "
                                  f"reduced length by {reduction_pct:.2f}% using tolerance {tol}."
                        )

        # Merge points that are very close to each other (deduplication)
        if self.raw_points:
            for point_data in self.raw_points:
                if isinstance(point_data.get('simplify_tolerance'), bool):
                    raise ValueError(
                        "simplify_tolerance must be a non-negative number (or None/0 to disable). "
                        "Boolean values are not supported."
                    )
            final_points = _deduplicate_points(self.raw_points)
            if len(self.raw_points) != len(final_points):
                logger.info(
                          f"Simplification merged {len(self.raw_points) - len(final_points)} "
                          f"points out of {len(self.raw_points)}"
                )
            self.raw_points = final_points

    def _resolve_overlaps(self):
        """
        Processes polygons based on their `z_order` to create a flat,
        non-overlapping planar partition. Higher `z_order` polygons "cookie-cut"
        lower ones.
        """
        # Sort polygons by priority, with the highest z_order processed first.
        if not self.raw_polygons:
            # If this is empty, just create an empty GeoDataFrame.
            self.clean_polygons = gpd.GeoDataFrame(
                columns=[
                    "geometry",
                    "zone_id",
                    "lc",
                    "z_order",
                    "dist_min",
                    "dist_max",
                    "densify",
                    "simplify_tolerance",
                    "fields",
                    "embed",
                    "quad_buffer",
                    "quad_buffer_thickness",
                    "growth_factor",
                    "border_lc",
                    "dist_max_in",
                    "label",
                    "gap_wall",
                ],
                crs=self.crs,
            )
            return
        df = pd.DataFrame(self.raw_polygons)
        df = df.sort_values(by='z_order', ascending=False, kind='mergesort')
        
        occupied_space = None # Tracks the union of all higher-priority polygons.
        
        final_features = []

        for idx, row in df.iterrows():
            current_geo = row['geometry']
            
            if occupied_space is None:
                # The first (highest priority) polygon is added unmodified.
                final_geo = current_geo
                occupied_space = current_geo
            else:
                # Subtract the already-occupied space from the current polygon.
                try:
                    final_geo = current_geo.difference(occupied_space)
                except Exception:
                    # If the standard difference fails, try again with valid geometries.
                    current_geo = make_valid(current_geo)
                    occupied_space = make_valid(occupied_space)
                    final_geo = current_geo.difference(occupied_space)

                # Add the current polygon's footprint to the occupied space.
                occupied_space = unary_union([occupied_space, current_geo])
            
            # Skip if the polygon was completely covered by higher-priority ones.
            if final_geo.is_empty:
                continue
                
            # Ensure the resulting geometry is valid before exploding.
            final_geo = make_valid(final_geo)

            # If the difference operation resulted in a MultiPolygon, explode it into
            # individual Polygons, each inheriting the parent's attributes.
            if final_geo.geom_type == 'MultiPolygon':
                for part in final_geo.geoms:
                    feat = row.copy()
                    feat['geometry'] = make_valid(part) # Ensure each part is valid
                    final_features.append(feat)
            else:
                feat = row.copy()
                feat['geometry'] = final_geo
                final_features.append(feat)

        self.clean_polygons = gpd.GeoDataFrame(final_features, crs=self.crs)

    def _resolve_connectivity_tolerance(self, connectivity_tolerance=None):
        """The per-run override if given (validated), else the instance default."""
        if connectivity_tolerance is None:
            return self.connectivity_tolerance
        return _coerce_connectivity_tolerance(connectivity_tolerance)

    def _enforce_connectivity(self, connectivity_tolerance=None):
        """
        Snaps features together to ensure they are topologically connected before
        being passed to the mesher. This is crucial for Gmsh to correctly

        interpret shared boundaries.
        """
        tolerance = self._resolve_connectivity_tolerance(connectivity_tolerance)

        # 1. Collect all polygon boundaries into a single geometry.
        # We snap to the linear boundaries, not the polygon areas.
        if not self.clean_polygons.empty:
            poly_boundaries = unary_union(self.clean_polygons.geometry.boundary)
        else:
            poly_boundaries = None

        # 2. Snap lines to polygon boundaries.
        # This ensures that features like rivers connect precisely to zone edges.
        if self.raw_lines and poly_boundaries is not None and not poly_boundaries.is_empty:
            logger.info(f"Snapping {len(self.raw_lines)} lines to polygon boundaries (tol={tolerance})...")
            for i, line_data in enumerate(self.raw_lines):
                if not line_data.get('snap_to_polygons', True):
                    continue
                original_line = line_data['geometry']
                snapped_line = snap(original_line, poly_boundaries, tolerance)
                self.raw_lines[i]['geometry'] = snapped_line

        # 3. Snap points to all other geometries (lines and polygon boundaries).
        # This ensures points like wells are located exactly on a feature.
        if self.raw_points:
            geoms_to_snap_to = []
            if poly_boundaries is not None and not poly_boundaries.is_empty:
                geoms_to_snap_to.append(poly_boundaries)
            
            if self.raw_lines:
                # Use the (potentially modified) snapped lines for snapping points.
                lines_union = unary_union([d['geometry'] for d in self.raw_lines])
                geoms_to_snap_to.append(lines_union)
            
            if geoms_to_snap_to:
                reference_geom = unary_union(geoms_to_snap_to)
                
                logger.info(f"Snapping {len(self.raw_points)} points to geometry (tol={tolerance})...")
                for i, point_data in enumerate(self.raw_points):
                    original_point = point_data['geometry']
                    snapped_point = snap(original_point, reference_geom, tolerance)
                    self.raw_points[i]['geometry'] = snapped_point


    def _clip_features_to_domain(self):
        """Remove or trim line/point features that remain outside the meshing domain."""
        if self.clean_polygons.empty:
            return

        domain_union = unary_union(self.clean_polygons.geometry)
        if domain_union.is_empty:
            return
        domain_union = make_valid(domain_union)
        # Prepared-geometry fast paths: most features are entirely inside
        # (or outside) the domain, where a cheap predicate avoids the full
        # boolean intersection.
        domain_prep = prep(domain_union)

        clipped_lines = []
        for line_data in self.raw_lines:
            geom = line_data.get("geometry")
            if geom is None or geom.is_empty:
                continue
            if not domain_prep.intersects(geom):
                continue
            if domain_prep.covers(geom):
                clipped = geom
            else:
                try:
                    clipped = geom.intersection(domain_union)
                except Exception:
                    clipped = make_valid(geom).intersection(domain_union)

            if clipped.is_empty:
                continue

            line_parts = []
            if clipped.geom_type in ("LineString", "MultiLineString"):
                line_parts = [clipped] if clipped.geom_type == "LineString" else list(clipped.geoms)
            elif clipped.geom_type == "GeometryCollection":
                line_parts = [
                    part for part in clipped.geoms
                    if part.geom_type in ("LineString", "MultiLineString") and not part.is_empty
                ]

            for part in line_parts:
                if part.geom_type == "MultiLineString":
                    for subpart in part.geoms:
                        if subpart.length > 0:
                            feat = line_data.copy()
                            feat["geometry"] = subpart
                            clipped_lines.append(feat)
                elif part.length > 0:
                    feat = line_data.copy()
                    feat["geometry"] = part
                    clipped_lines.append(feat)

        removed_lines = len(self.raw_lines) - len(clipped_lines)
        if removed_lines > 0:
            logger.info(f"Clipped/removed {removed_lines} line feature(s) outside the domain.")
        self.raw_lines = clipped_lines

        kept_points = []
        for point_data in self.raw_points:
            geom = point_data.get("geometry")
            if geom is None or geom.is_empty:
                continue
            if domain_union.covers(geom):
                kept_points.append(point_data)

        removed_points = len(self.raw_points) - len(kept_points)
        if removed_points > 0:
            logger.info(f"Removed {removed_points} point feature(s) outside the domain.")
        self.raw_points = kept_points


    def _join_line_junctions(self, connectivity_tolerance=None):
        """Snap, trim, pin and thin embedded lines where they meet (see ``_junctions``).

        Only line geometries change; the list keeps its order and length, so
        the clean-line index (the feature ID) is unchanged.
        """
        self.junction_diagnostics = _empty_junction_diagnostics()
        if self.line_junction_fraction <= 0:
            return
        members = [i for i, line in enumerate(self.raw_lines) if _takes_part_in_junctions(line)]
        if len(members) < 2:
            return
        eps = max(self._resolve_connectivity_tolerance(connectivity_tolerance), _junctions.TOUCH_FLOOR)
        tolerances = [
            self.line_junction_fraction * _line_vertex_spacing(self.raw_lines[i]) for i in members
        ]
        anchors = None
        if not self.clean_polygons.empty:
            anchors = unary_union(self.clean_polygons.geometry.boundary)

        lines = [self.raw_lines[i]['geometry'] for i in members]
        lines, moves = _junctions.snap_line_ends(lines, tolerances, eps, anchors)
        junctions = _junctions.find_junctions(lines, eps)
        lines, trims = _junctions.trim_short_stubs(lines, junctions, tolerances, eps, anchors)
        if trims:
            junctions = _junctions.find_junctions(lines, eps)
        for k, i in enumerate(members):
            self.raw_lines[i]['geometry'] = _junctions.pin_junctions(
                lines[k], junctions[k], tolerances[k], eps, anchors
            )

        line_ids = [self.raw_lines[i].get('line_id') for i in members]
        self.junction_diagnostics = _junction_diagnostics(line_ids, junctions, moves, trims)
        n_junctions = len(self.junction_diagnostics['junctions'])
        if moves or trims or n_junctions:
            logger.info(
                f"Line junctions: {n_junctions} pinned, {len(moves)} line end(s) snapped, "
                f"{len(trims)} overshoot(s) trimmed."
            )

    def _clean_barrier_lines(self, connectivity_tolerance=None):
        """Snap, node, simplify and de-noise the barrier lines (see ``vorflow._barrier.lines``).

        Works in a local frame (origin at the lower-left corner of the
        domain). Each barrier row gets a ``barrier_events`` list recording
        what was done to it (global coordinates); ``MeshGenerator`` passes
        them on in ``barrier_report``. A row split where it runs along the
        domain boundary becomes several rows; a row dropped whole (on the
        boundary, or collapsed to a point) leaves the line and its events
        are returned. Noise cleaning and dropped boundary parts warn.
        """
        members = [i for i, line in enumerate(self.raw_lines) if is_face_line(line)]
        for line in self.raw_lines:
            line['barrier_events'] = [] if is_face_line(line) else None
        if not members or self.clean_polygons.empty:
            return []
        domain_global = make_valid(unary_union(self.clean_polygons.geometry))
        origin = np.asarray(domain_global.bounds[:2], dtype=float)
        domain = shapely.affinity.translate(domain_global, -origin[0], -origin[1])
        standard = [
            shapely.affinity.translate(part, -origin[0], -origin[1])
            for line in self.raw_lines
            if row_bool(line, 'embed', True) and not row_bool(line, 'quad_buffer', False)
            and not is_face_line(line)
            for part in _line_parts(line['geometry'])
        ]
        barriers = [
            _barrier_lines.BarrierLine(
                self.raw_lines[i].get('line_id'),
                np.asarray(self.raw_lines[i]['geometry'].coords, dtype=float)[:, :2] - origin,
                float(positive_number(self.raw_lines[i].get('lc'))
                      or self._zone_resolution(self.raw_lines[i]['geometry'])),
                float(positive_number(self.raw_lines[i].get('simplify_tolerance')) or 0.0),
                source=i,
            )
            for i in members
        ]
        tolerance = self._resolve_connectivity_tolerance(connectivity_tolerance)
        try:
            kept, dropped = _barrier_lines.clean_barrier_lines(barriers, standard, domain, tolerance)
        except _barrier_pipeline.GEOMETRY_ERRORS as err:
            raise _barrier_pipeline.internal_failure("line cleaning", err) from err
        rows = []
        for i, line in enumerate(self.raw_lines):
            if i not in members:
                rows.append(line)
                continue
            for barrier in (b for b in kept if b.source == i):
                rows.append(dict(line, geometry=LineString(barrier.xy + origin),
                                 barrier_events=self._barrier_events(barrier, origin)))
        self.raw_lines = rows
        return [event for barrier in dropped for event in self._barrier_events(barrier, origin)]

    def _zone_resolution(self, geometry) -> float:
        """Finest resolution of the clean polygons ``geometry`` meets (inf if none has one).

        Stands in for the resolution of a barrier added without one, so its
        noise cleaning and simplification still have a length scale.
        """
        found = [positive_number(lc) for lc, poly in zip(self.clean_polygons.get('lc', []),
                                                        self.clean_polygons.geometry)
                 if poly.intersects(geometry)]
        found = [lc for lc in found if lc is not None]
        return float(min(found)) if found else math.inf

    @staticmethod
    def _barrier_events(barrier, origin) -> list:
        """The cleaning events of ``barrier`` in global coordinates, with its line_id; warns or logs each."""
        events = [dict(e, line_id=barrier.line_id,
                       location=(float(e['location'][0] + origin[0]), float(e['location'][1] + origin[1])))
                  for e in barrier.events]
        for e in events:
            moved = f"; moved {e['distance']:.3g}" if e['distance'] > 0 and e['kind'] in ('snap', 'trim', 'noise') else ""
            message = (f"Barrier {barrier.line_id!r}: {e['note']}{moved} "
                       f"near ({e['location'][0]:.2f}, {e['location'][1]:.2f}).")
            if e['kind'] in ('noise', 'boundary'):
                warn_user(message)
            else:
                logger.info(message)
        return events

    def _hex_ring_obstacles(self):
        """Embedded polygon boundaries and lines as ``(label, geometry, half_width)`` hex-ring obstacles.

        Run after overlap resolution, so ``clean_polygons`` holds only the
        embedded (domain) polygons. Field-only lines create no nodes and are
        skipped.
        """
        obstacles = []
        for _, poly in self.clean_polygons.iterrows():
            obstacles.append((
                f"polygon {poly['zone_id']!r} boundary",
                poly.geometry.boundary,
                _meshed_half_width(poly),
            ))
        for line_data in self.raw_lines:
            if not bool(line_data.get('embed', True)):
                continue
            if self.keep_close_rings and is_face_line(line_data):
                continue
            obstacles.append((
                f"line {line_data['line_id']!r}",
                line_data['geometry'],
                _meshed_half_width(line_data),
            ))
        return obstacles

    def _hex_ring_size_sources(self, field_only_polys):
        """Polygons and lines as ``(label, geometry, half_width, ramps)`` size sources.

        Unlike the clearance obstacles, field-only features count: they
        create no nodes but their size fields still reach the ring. The
        engine holds every polygon, embedded or field-only, at its
        resolution throughout its interior, so polygons are measured as
        areas.
        """
        sources = [
            (f"polygon {poly['zone_id']!r}", poly['geometry'], _meshed_half_width(poly), _size_ramps(poly))
            for _, poly in self.clean_polygons.iterrows()
        ]
        sources.extend(
            (f"polygon {poly['zone_id']!r}", poly['geometry'], 0.0, _size_ramps(poly))
            for poly in field_only_polys
        )
        sources.extend(
            (f"line {line['line_id']!r}", line['geometry'], _meshed_half_width(line), _size_ramps(line))
            for line in self.raw_lines
        )
        return sources

    def _assign_hex_ring_seeds(self, field_only_polys=()):
        """Set each working point's ``ring_seeds``: its six ring seeds, or None when off or conflicting.

        ``ring_near_barrier`` is True for a dropped ring that a barrier passes too close to.

        A ring is dropped when another embedded feature is too close (see
        ``_hex_ring_conflict``) or when any feature's size field is too fine
        inside the ring (see ``_hex_ring_size_conflict``). Other embedded
        points are obstacles too; one with its own hex ring also claims the
        ring radius around it.
        """
        requested = [i for i, p in enumerate(self.raw_points) if p.get('hex_ring', False)]
        for point_data in self.raw_points:
            point_data['ring_seeds'] = None
            point_data['ring_near_barrier'] = False
        if not requested:
            return

        feature_obstacles = self._hex_ring_obstacles()
        barrier_obstacles = [
            (None, line['geometry'], 0.0) for line in self.raw_lines
            if bool(line.get('embed', True)) and is_face_line(line) and not self.keep_close_rings
        ]
        feature_sources = self._hex_ring_size_sources(field_only_polys)
        point_obstacles = [
            (
                f"point {p['point_id']!r}",
                p['geometry'],
                float(p['lc']) if p.get('hex_ring', False) else 0.0,
            )
            if bool(p.get('embed', True)) else None
            for p in self.raw_points
        ]
        point_sources = [
            (f"point {p['point_id']!r}", p['geometry'], 0.0, _size_ramps(p)) for p in self.raw_points
        ]
        for i in requested:
            point_data = self.raw_points[i]
            lc = float(point_data['lc'])
            seeds = _hex_ring_seeds(point_data['geometry'], lc)
            others = [ob for j, ob in enumerate(point_obstacles) if j != i and ob is not None]
            other_sources = [src for j, src in enumerate(point_sources) if j != i]
            reason = _hex_ring_conflict(point_data['geometry'], lc, feature_obstacles + others)
            if reason is None:
                reason = _hex_ring_size_conflict(
                    Polygon(seeds), lc, feature_sources + other_sources
                )
            if reason is not None:
                # barrier_report lists the rings a barrier alone would have dropped.
                point_data['ring_near_barrier'] = _hex_ring_conflict(
                    point_data['geometry'], lc, barrier_obstacles) is not None
                warnings.warn(
                    f"Point {point_data['point_id']!r}: {reason}; hex_ring ignored.",
                    UserWarning,
                    stacklevel=4,
                )
                continue
            point_data['ring_seeds'] = seeds

    def plot(self, ax=None, clean=False, column="zone_id", **kwargs):
        """
        Plots the conceptual model: polygons coloured by ``column``, with lines
        and points on top. Lines take the ``color`` given to ``add_line``
        (default: barriers red, others blue), and features with a ``label``
        get one legend entry per label for ``ax.legend()``.

        Field-only (``embed=False``) features are drawn as dashed outlines,
        dashed lines and hollow points. Needs Matplotlib
        (``pip install vorflow[plot]``).

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to draw on; a new figure
                is created when None.
            clean (bool): Plot the features from the last ``generate()``
                instead of the registered raw features.
            column (str, optional): Polygon column to colour by. Default
                ``"zone_id"``; None draws them in one colour.
            **kwargs: Passed to ``GeoDataFrame.plot`` for the polygons.

        Returns:
            matplotlib.axes.Axes: The axes drawn on.

        Raises:
            RuntimeError: If ``clean=True`` before ``generate()`` has run.
        """
        ax = get_ax(ax)
        if clean:
            if self.clean_polygons.empty:
                raise RuntimeError(
                    "Clean features are not available. Call ConceptualMesh.generate() first."
                )
            frames = (self.clean_polygons, self.clean_lines, self.clean_points)
        else:
            frames = tuple(
                features_frame(features, self.crs)
                for features in (self.raw_polygons, self.raw_lines, self.raw_points)
            )
        return plot_features(ax, *frames, column=column, **kwargs)

    def generate(self, connectivity_tolerance=None):
        """
        Runs the full preprocessing workflow: resolves polygon overlaps,
        ensures topological connectivity, and prepares clean GeoDataFrames
        for the mesher. Registered raw features remain unchanged so repeated
        calls with the same inputs produce the same clean features.

        Args:
            connectivity_tolerance (float, optional): Override for the instance's
                default topology snapping tolerance during this preprocessing run.
        """
        with self._working_copies():
            return self._generate_working_features(connectivity_tolerance)

    def check(self, background_lc=None):
        """
        Reports input defects that tend to produce bad Voronoi cells, before meshing.

        Each defect is measured against the local mesh size ``h`` at its
        location: the size field the mesher will aim for there, i.e.
        ``background_lc`` capped by every feature's size transition (by
        default ``resolution + (growth_factor - 1) * distance``, zero distance
        inside a polygon). ``ratio = distance / h`` sets the severity:

        ====================  ================  ===================
        kind                  error if ratio <  warning if ratio <
        ====================  ================  ===================
        short_segment         0.1               0.25
        close_vertices        0.1               0.5
        line_end_near_miss    0.5               1.0 (inclusive)
        point_near_miss       0.5               1.0 (inclusive)
        sliver_polygon        0.5               1.0
        narrow_gap            never             1.0 (inclusive)
        ====================  ================  ===================

        - ``short_segment``: two consecutive vertices of a polygon ring or line.
        - ``close_vertices``: two distinct mesh points (vertices, points, or
          where a line crosses another line or a polygon edge), e.g. a line
          crossing a polygon edge just beside one of its vertices.
        - ``line_end_near_miss`` / ``point_near_miss``: a line end or point
          that touches no other feature (it is further than
          ``connectivity_tolerance`` from all of them) but lies close to one.
        - ``sliver_polygon``: a polygon part, or a strip of one, narrower than
          the local size, often left over by overlap resolution.
        - ``narrow_gap``: a stretch of line running close to another feature
          (or to itself), away from where they meet. The grid there has only
          one or two cells across, which may be acceptable.

        The checks run on the features as ``generate()`` cleans them
        (simplification, overlap resolution, snapping, clipping and line
        junctions), but before densification: the vertices densification adds
        are vorflow's own and are not reported. Only embedded features are
        checked; field-only (``embed=False``) features only set ``h``.
        Barrier, straddle and quad-buffer features are checked as the plain
        lines and polygons they are drawn from, except that a barrier end
        running alongside another barrier or the domain outline is not a
        near miss: the barrier construction gives that gap one row of cells,
        and it is reported as a ``narrow_gap``.

        One defect often trips several checks. Findings whose geometries lie
        within ``2 * h`` of each other are treated as one site, and only the most
        severe (then the lowest ratio) is returned; its ``site_kinds`` lists
        every kind found there, so a site may need more than one fix.

        The registered features and the results of an earlier ``generate()``
        are left unchanged.

        Args:
            background_lc (float, optional): The background mesh size you
                will pass to ``MeshGenerator``. If None, the coarsest feature
                ``resolution`` is used, which caps ``h`` lower than a coarser
                background would: defects in coarse areas far from any
                feature may then go unreported.

        Returns:
            geopandas.GeoDataFrame: One row per defect site, errors first and
            then by ratio, with columns ``feature_type`` (``"polygon"``,
            ``"line"`` or ``"point"``), ``feature_id`` (the ``zone_id``,
            ``line_id`` or ``point_id``), ``other_type`` and ``other_id``
            (the other feature involved, or None), ``kind``,
            ``site_kinds`` (comma-separated kinds found at the site),
            ``severity``
            (``"error"`` or ``"warning"``), ``distance``, ``h``, ``ratio``,
            ``x`` and ``y`` (where the defect is narrowest) and ``geometry``
            (the segment, gap or polygon part that shows it). Empty when no
            defect is found.

        Raises:
            ValueError: If ``background_lc`` is not a positive number, or is
                None and no feature has a resolution.
        """
        saved_state = (self.clean_polygons, self.clean_lines, self.clean_points,
                       self.junction_diagnostics)
        try:
            with verbosity_scope(0), self._working_copies():
                clean = self._generate_working_features(None, densify=False)
        finally:
            (self.clean_polygons, self.clean_lines, self.clean_points,
             self.junction_diagnostics) = saved_state
        background_lc = _check_background_lc(background_lc, clean)
        sources = [
            (row.geometry, ramp)
            for frame in clean
            for _, row in frame.iterrows()
            for ramp in _size_ramps(row)
        ]
        size = _checks.size_field(sources, background_lc, DEFAULT_GROWTH_FACTOR - 1.0)
        findings = _checks.find_defects(*clean, size, self.connectivity_tolerance, crs=self.crs)
        n_errors = int((findings["severity"] == "error").sum())
        logger.info(
            "Input check: %d error(s), %d warning(s).", n_errors, len(findings) - n_errors
        )
        return findings

    @contextmanager
    def _working_copies(self):
        """Run preprocessing on per-run copies of the raw feature records, then restore them."""
        raw_inputs = self.raw_polygons, self.raw_lines, self.raw_points
        # Preprocessing replaces geometries and filters feature records, but does
        # not modify their nested field objects. Copy each record for this run.
        self.raw_polygons, self.raw_lines, self.raw_points = (
            [feature.copy() for feature in raw_features]
            for raw_features in raw_inputs
        )
        try:
            yield
        finally:
            self.raw_polygons, self.raw_lines, self.raw_points = raw_inputs

    def _generate_working_features(self, connectivity_tolerance, densify=True):
        logger.info("Applying optional geometry simplification...")
        self._apply_simplification()

        # --- Embed semantics for polygons ---
        # Polygons with embed=True define the actual meshing domain and therefore
        # participate in the cookie-cutter (overlap resolution) process.
        # Polygons with embed=False are refinement-only (field-only) regions and
        # must NOT affect domain topology.
        embedded_polys = [p for p in self.raw_polygons if bool(p.get("embed", True))]
        field_only_polys = [p for p in self.raw_polygons if not bool(p.get("embed", True))]

        # Only embedded polygons are used to build the domain partition.
        self.raw_polygons = embedded_polys

        logger.info("Resolving polygon overlaps...")
        self._resolve_overlaps()
        
        logger.info("Enforcing strict topology...")
        self._enforce_connectivity(connectivity_tolerance=connectivity_tolerance)

        logger.info("Clipping features to domain...")
        self._clip_features_to_domain()

        logger.info("Joining line junctions...")
        self._join_line_junctions(connectivity_tolerance=connectivity_tolerance)

        dropped_barrier_events = self._clean_barrier_lines(connectivity_tolerance=connectivity_tolerance)

        # After snapping and clipping, so each ring is built about the final point.
        self._assign_hex_ring_seeds(field_only_polys)
        
        # Promote the processed raw geometries to final "clean" GeoDataFrames.
        if self.raw_lines:
            self.clean_lines = gpd.GeoDataFrame(self.raw_lines, crs=self.crs)
        else:
            self.clean_lines = gpd.GeoDataFrame(
                columns=[
                    'geometry',
                    'line_id',
                    'lc',
                    'snap_to_polygons',
                    'is_barrier',
                    'dist_min',
                    'dist_max',
                    'straddle_width',
                    'fields',
                    'embed',
                    'densify',
                    'simplify_tolerance',
                    'quad_buffer',
                    'quad_buffer_thickness',
                    'z_order',
                    'growth_factor',
                    'label',
                    'color',
                    'barrier_events',
                    'gap_wall',
                ],
                crs=self.crs,
            )
        # Barrier rows dropped whole by the cleaning have no row to carry their events.
        self.clean_lines.attrs['dropped_barrier_events'] = dropped_barrier_events

        # Clean Points
        if self.raw_points:
            self.clean_points = gpd.GeoDataFrame(self.raw_points, crs=self.crs)
        else:
            self.clean_points = gpd.GeoDataFrame(
                columns=[
                    'geometry',
                    'point_id',
                    'lc',
                    'dist_min',
                    'dist_max',
                    'fields',
                    'embed',
                    'simplify_tolerance',
                    'growth_factor',
                    'hex_ring',
                    'label',
                    'ring_seeds',
                    'ring_near_barrier',
                    'gap_wall',
                ],
                crs=self.crs,
            )

        # Clip field-only polygons to the final embedded domain.
        # Field-only polygons should not extend outside the domain, but they also
        # should not cut/modify domain topology.
        if field_only_polys and not self.clean_polygons.empty:
            domain_union = unary_union(self.clean_polygons.geometry)
            domain_union = make_valid(domain_union)
            domain_prep = prep(domain_union)

            clipped_features = []
            for poly_data in field_only_polys:
                geom = poly_data.get("geometry")
                if geom is None or geom.is_empty:
                    continue
                geom = make_valid(geom)
                if not domain_prep.intersects(geom):
                    continue
                if domain_prep.covers(geom):
                    clipped = geom
                else:
                    try:
                        clipped = geom.intersection(domain_union)
                    except Exception:
                        clipped = make_valid(geom).intersection(make_valid(domain_union))

                if clipped.is_empty:
                    continue

                feat = poly_data.copy()
                feat["geometry"] = make_valid(clipped)
                clipped_features.append(feat)

            if clipped_features:
                field_only_gdf = gpd.GeoDataFrame(clipped_features, crs=self.crs)
                # Align schemas before concat to avoid pandas dtype inference warnings
                # while preserving the canonical polygon columns.
                polygon_columns = [
                    'geometry',
                    'zone_id',
                    'lc',
                    'z_order',
                    'dist_min',
                    'dist_max',
                    'fields',
                    'embed',
                    'densify',
                    'simplify_tolerance',
                    'quad_buffer',
                    'quad_buffer_thickness',
                    'growth_factor',
                    'label',
                    'gap_wall',
                ]
                self.clean_polygons = self.clean_polygons.reindex(columns=polygon_columns)
                field_only_gdf = field_only_gdf.reindex(columns=polygon_columns)
                # Drop all-null non-geometry columns only during concat; they are
                # restored immediately after so the public GeoDataFrame shape is unchanged.
                concat_frames = [
                    frame.dropna(axis=1, how='all')
                    for frame in (self.clean_polygons, field_only_gdf)
                ]
                self.clean_polygons = gpd.GeoDataFrame(
                    pd.concat(concat_frames, ignore_index=True).reindex(columns=polygon_columns),
                    crs=self.crs,
                )

        if densify:
            logger.info("Densifying geometry...")
            self._apply_densification()
        
        return self.clean_polygons, self.clean_lines, self.clean_points

    
    def _densify_geometry(self, geometry, resolution):
        """
        Recursively adds vertices to LineStrings and Polygon boundaries to ensure
        that no segment is longer than the specified resolution. This is critical
        for forcing the mesh to respect a desired element size along a feature.
        """
        def densify_line(line, max_segment_length):
            if not isinstance(line, LineString):
                return line
            
            coords = list(line.coords)
            new_coords = [coords[0]]
            
            for i in range(len(coords) - 1):
                p1 = np.array(coords[i])
                p2 = np.array(coords[i+1])
                segment_length = np.linalg.norm(p2 - p1)
                
                if segment_length > max_segment_length:
                    num_segments = int(np.ceil(segment_length / max_segment_length))
                    # Add intermediate points along the segment.
                    for j in range(1, num_segments):
                        t = j / num_segments
                        p_new = p1 + t * (p2 - p1)
                        new_coords.append(tuple(p_new))
                
                # Always add the original endpoint of the segment.
                new_coords.append(coords[i+1])
            
            return LineString(new_coords)

        if geometry.geom_type == 'LineString':
            return densify_line(geometry, resolution)
            
            
        elif geometry.geom_type == 'Polygon':
            # Densify the exterior ring.
            new_exterior = densify_line(geometry.exterior, resolution)
            
            # Densify all interior rings (holes).
            new_interiors = []
            for interior in geometry.interiors:
                new_interiors.append(densify_line(interior, resolution))
                
            return Polygon(new_exterior, new_interiors)
            
        elif geometry.geom_type == 'MultiPolygon':
            parts = [self._densify_geometry(p, resolution) for p in geometry.geoms]
            return MultiPolygon(parts)
            
        return geometry


    def _apply_densification(self):
        """Applies densification to the clean polygon and line features."""
        # Densify polygon boundaries based on `densify`.
        if not self.clean_polygons.empty:

            def get_poly_resolution(row):
                d = row.get("densify")
                if d is False or pd.isna(d):
                    return None
                if d is True:
                    return row.get("lc")
                if isinstance(d, (int, float)) and not isinstance(d, bool) and d > 0:
                    return d
                return None

            def _poly_densify(row):
                res = get_poly_resolution(row)
                return self._densify_geometry(row["geometry"], res) if res is not None else row["geometry"]

            self.clean_polygons["geometry"] = self.clean_polygons.apply(_poly_densify, axis=1)

        # Densify lines based on their target resolution ('lc').
        if not self.clean_lines.empty:
            def _line_densify(row):
                # Barriers place their own nodes (vorflow._barrier).
                res = None if is_face_line(row) else _line_densify_spacing(row)
                return self._densify_geometry(row['geometry'], res) if res is not None else row['geometry']

            self.clean_lines['geometry'] = self.clean_lines.apply(_line_densify, axis=1)

    def features(self):
        """
        Return all cleaned lines, points and polygons as one GeoDataFrame.

        Returns:
            features (GeoDataFrame): Returns a GeoDataFrame of all cleaned features of the ConceptualModel with 
            the columns ``'geometry'``, ``'embed'``, ``'feature_type'``, and ``'feature_id'``.

        Raises:
            ValueError: If there are no cleaned features.
        """
        lines = gpd.GeoDataFrame()
        if not self.clean_lines.empty:
            lines = self.clean_lines[["geometry", "line_id", "embed"]].copy()
            lines = lines.rename(columns={"line_id": "feature_id"})
            lines["feature_type"] = "line"

        points = gpd.GeoDataFrame()
        if not self.clean_points.empty:
            points = self.clean_points[["geometry", "point_id", "embed"]].copy()
            points = points.rename(columns={"point_id": "feature_id"})
            points["feature_type"] = "point"

        polygons = gpd.GeoDataFrame()
        if not self.clean_polygons.empty:
            polygons = self.clean_polygons[["geometry", "zone_id", "embed"]].copy()
            polygons = polygons.rename(columns={"zone_id": "feature_id"})
            polygons["feature_type"] = "polygon"

        features = gpd.GeoDataFrame(pd.concat([
            lines,
            points,
            polygons
        ], ignore_index=True))

        if features.empty:
            raise ValueError("Run ``.generate()`` before exporting features.")

        return features
