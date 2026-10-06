"""Gmsh meshing engine: clean ConceptualMesh features -> 2D triangular/quad mesh.

Geometry transfer (``MeshGenerator._build_occ_model``) runs in a fixed order,
because the order in which OCC entities are created determines their tags:

1. point features;
2. protection corridors, the barrier zone, quad-buffer plans and the
   straddle pairs fixed where standard lines cross barriers (pure geometry,
   see ``vorflow.buffer`` and ``vorflow._straddle``);
3. lines: quad-buffer strip surfaces, straddle point pairs for barrier and
   straddle lines, plain curves trimmed off the barrier zone, or, for a line
   crossing a barrier, curves that end on the barrier's straddle pairs;
4. polygons: quad-buffer bands first, then every embedded polygon minus the
   buffer footprints; field-only (embed=False) polygons are deferred;
5. ``fragment`` + ``removeAllDuplicates`` (+ optional ``healShapes``), with
   the fragment map remapped by coordinates wherever OCC renumbers entities;
6. deferred field-only polygons, the rebuilt feature map, orphan-surface
   recovery and the transfinite/recombine buffer constraints.

The resulting ``gmsh_map`` (feature id -> dimtags, per kind) drives
``_embed_features`` and ``_setup_fields``; the quad-buffer crossings are
passed on to ``_setup_fields`` as Ball refinement fields.
"""
from __future__ import annotations

import logging
import dataclasses

import gmsh
import math
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points, unary_union
from shapely.validation import make_valid
from scipy.spatial import cKDTree
from .fields import (
    DEFAULT_GROWTH_FACTOR,
    ConstantField,
    GeometricGrowthField,
    MeshField,
    ThresholdField,
    _BorderGradingField,
)
from ._log import current_verbosity, verbosity_scope
from . import buffer
from ._straddle import (
    _straddle_distances,
    _straddle_pair,
    plan_barrier_crossings,
    straddle_epsilon,
    straddle_probe,
)
from ._features import (
    feature_lc,
    is_embedded,
    polygon_parts,
    positive_number,
    ring_seed_coords,
    row_bool,
    sanitize_coords,
    unpinch_polygons,
)


logger = logging.getLogger(__name__)



def _to_key(dim, tag):
    """Normalize a gmsh (dim, tag) pair into a hashable dict key."""
    return (int(dim), int(tag))


# Tolerances for matching OCC entities across removeAllDuplicates (~um) and
# healShapes (0.1 mm, absorbing its ~1e-6 drift). After healing, a curve or
# surface matches only within _HEAL_MATCH_REL_TOL of its own extent, so a piece
# shorter than _HEAL_MATCH_TOL cannot match its neighbour.
_DEDUP_MATCH_TOL = 1e-6
_HEAL_MATCH_TOL = 1e-4
_HEAL_MATCH_REL_TOL = 0.5
# Vertices closer than this are one point when building OCC curve loops.
_VERTEX_MERGE_TOL = 1e-5


def _entity_signature(dim, tag):
    """Location signature of an OCC entity: point coordinates, or bbox plus centre of mass.

    A bounding box alone is ambiguous for curves and surfaces (two triangles
    tiling a square share one), so they add their centre of mass.
    """
    bb = gmsh.model.occ.getBoundingBox(dim, tag)
    if dim == 0:
        return tuple(bb[:3])
    return tuple(bb[:6]) + tuple(gmsh.model.occ.getCenterOfMass(dim, tag))


def _snapshot_signatures(dimtags, stage):
    """Signatures of the given (dim, tag) entities, keyed by (dim, tag); unmeasurable ones are skipped."""
    signatures = {}
    for d, t in dimtags:
        if (d, t) in signatures:
            continue
        try:
            signatures[(d, t)] = _entity_signature(d, t)
        except Exception:
            # gmsh raises plain Exception for an entity it cannot measure;
            # without a signature it cannot be matched across the rebuild.
            logger.debug("%s: no signature for entity (dim %d, tag %d); it "
                         "cannot be remapped.", stage, d, t)
    return signatures


class _SurvivorIndex:
    """Entities surviving an OCC rebuild, searchable by signature within a tolerance."""

    def __init__(self, signatures, tolerance):
        self.tolerance = tolerance
        by_dim = {}
        for (d, t), signature in signatures.items():
            by_dim.setdefault(d, []).append((t, signature))
        self._trees = {
            d: (np.array([t for t, _ in items]), cKDTree(np.array([sig for _, sig in items])))
            for d, items in by_dim.items()
        }

    def candidates(self, dim, tag, signature, tolerance=None):
        """(offset, surviving tag) within the tolerance of this signature, nearest first.

        The same tag only breaks ties: healShapes reuses tag numbers for
        different entities, so a surviving tag is not evidence of identity.
        """
        if dim not in self._trees:
            return []
        tags, tree = self._trees[dim]
        radius = self.tolerance if tolerance is None else tolerance
        hits = tree.query_ball_point(signature, r=radius, p=np.inf)
        if not hits:
            return []
        offsets = np.abs(tree.data[hits] - np.asarray(signature)).max(axis=1)
        return sorted(((float(offset), int(t)) for offset, t in zip(offsets, tags[hits])),
                      key=lambda pair: (pair[0], pair[1] != tag, pair[1]))

    def match(self, dim, tag, signature, tolerance=None):
        """Nearest surviving tag within the tolerance (the same tag on a tie), or None."""
        found = self.candidates(dim, tag, signature, tolerance)
        return found[0][1] if found else None


def _heal_match_tolerance(dim, signature):
    """_HEAL_MATCH_TOL, capped for a curve or surface at _HEAL_MATCH_REL_TOL of its extent."""
    if dim == 0:
        return _HEAL_MATCH_TOL
    extent = max(signature[3] - signature[0], signature[4] - signature[1], signature[5] - signature[2])
    return min(_HEAL_MATCH_TOL, _HEAL_MATCH_REL_TOL * extent)


def _match_heal_survivors(pre_heal, survivors):
    """Surviving tag (or None) for each pre-heal entity, keyed by (dim, tag).

    Points take the nearest survivor: healing merges close points, and the
    merged point still marks each of them. Curves and surfaces are matched
    one-to-one, nearest pair first, within _heal_match_tolerance. Two
    collinear neighbours differ by at least the longer one's length, so a
    piece that healing deleted or absorbed into its neighbour is pruned
    rather than aliased onto that neighbour (or onto another feature's piece).
    """
    matches = {}
    pairs = []
    for (d, t), signature in pre_heal.items():
        if d == 0:
            matches[(d, t)] = survivors.match(d, t, signature)
            continue
        matches[(d, t)] = None
        pairs.extend(
            (offset, new_tag != t, d, t, new_tag)
            for offset, new_tag in survivors.candidates(d, t, signature, _heal_match_tolerance(d, signature))
        )
    claimed = set()
    for _offset, _retagged, d, t, new_tag in sorted(pairs):
        if matches[(d, t)] is None and (d, new_tag) not in claimed:
            matches[(d, t)] = new_tag
            claimed.add((d, new_tag))
    return matches


def _embedded_zones(zones_gdf):
    """Zones that belong to the meshed domain (drops field-only polygons)."""
    if zones_gdf is None or zones_gdf.empty or "embed" not in zones_gdf.columns:
        return zones_gdf
    embed = zones_gdf["embed"].fillna(True).astype(bool)
    return zones_gdf[embed]


def _assign_zones_to_elements(grid, zones_gdf):
    """Assign a zone to each element by spatially joining element centroids.

    When a centroid intersects several zones (overlaps or shared borders) the
    tie is broken deterministically: highest ``z_order`` wins, then the zone
    that appears earliest in ``zones_gdf``.
    """
    if zones_gdf is None or zones_gdf.empty or "zone_id" not in zones_gdf.columns:
        grid["zone_id"] = pd.NA
        grid["z_order"] = pd.NA
        return grid

    zone_cols = ["geometry", "zone_id"]
    if "z_order" in zones_gdf.columns:
        zone_cols.append("z_order")
    zones = zones_gdf[zone_cols].reset_index(drop=True)
    centroids = gpd.GeoDataFrame(
        {"element_tag": grid["element_tag"]},
        geometry=gpd.points_from_xy(grid["centroid_x"], grid["centroid_y"]),
        crs=grid.crs,
    )
    joined = gpd.sjoin(centroids, zones, how="left", predicate="intersects")
    sort_cols, ascending = ["element_tag"], [True]
    if "z_order" in joined.columns:
        sort_cols += ["z_order", "index_right"]
        ascending += [False, True]
    else:
        sort_cols += ["index_right"]
        ascending += [True]
    joined = joined.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    joined = joined.drop_duplicates(subset="element_tag")
    merge_cols = ["element_tag", "zone_id"]
    if "z_order" in joined.columns:
        merge_cols.append("z_order")
    return grid.merge(joined[merge_cols], on="element_tag", how="left")


def _accumulate_nodes(dim, ent_tag, tag_to_xy):
    """Add the (x, y) of an entity's mesh nodes, boundary included, to ``tag_to_xy`` (first seen wins)."""
    nt, nc, _ = gmsh.model.mesh.getNodes(dim, int(ent_tag), includeBoundary=True)
    if len(nt) == 0:
        return
    pts = np.array(nc, dtype=float).reshape(-1, 3)
    for t, p in zip(nt, pts):
        tt = int(t)
        if tt not in tag_to_xy:
            tag_to_xy[tt] = (float(p[0]), float(p[1]))


def _embedded_constraint_entities(gmsh_map, clean_points, clean_lines):
    """Yield (dim, tag) of embedded point features, line curves and straddle points."""
    if clean_points is not None and not clean_points.empty and 'embed' in clean_points.columns:
        for fid, row in clean_points.iterrows():
            if not is_embedded(row):
                continue
            for dimtag in gmsh_map.get('points', {}).get(int(fid), []):
                if isinstance(dimtag, (tuple, list)) and len(dimtag) >= 2 and int(dimtag[0]) == 0:
                    yield 0, int(dimtag[1])

    if clean_lines is not None and not clean_lines.empty and 'embed' in clean_lines.columns:
        for fid, row in clean_lines.iterrows():
            if not is_embedded(row):
                continue
            if int(fid) in gmsh_map.get('lines', {}):
                for dimtag in gmsh_map['lines'][int(fid)]:
                    if isinstance(dimtag, (tuple, list)) and len(dimtag) >= 2 and int(dimtag[0]) == 1:
                        yield 1, int(dimtag[1])
            # Straddle/barrier lines are represented by point pairs.
            elif int(fid) in gmsh_map.get('straddle_points', {}):
                for dimtag in gmsh_map['straddle_points'][int(fid)]:
                    if isinstance(dimtag, (tuple, list)) and len(dimtag) >= 2 and int(dimtag[0]) == 0:
                        yield 0, int(dimtag[1])


def _free_node_tags(gmsh_map, polygons_gdf):
    """Tags of the mesh nodes inside embedded polygon surfaces, off every curve and point.

    gmsh classifies a node on the lowest-dimension entity it lies on, so
    getNodes(2, s, includeBoundary=False) leaves out the nodes of the
    surface's boundary and of the points and curves embedded in it (hex-ring
    seeds, line nodes, straddle pairs). Quad-buffer surfaces (strips and
    bands) are left out entirely.
    """
    if polygons_gdf is None or polygons_gdf.empty:
        return set()
    structured = {
        int(tag)
        for dimtags in gmsh_map.get('structured_buffer_surfs', {}).values()
        for tag in _dimtag_tags(dimtags)
    }
    free = set()
    for surf_tag in sorted(_domain_surface_tags(gmsh_map, polygons_gdf) - structured):
        node_tags, _, _ = gmsh.model.mesh.getNodes(2, int(surf_tag), includeBoundary=False)
        free.update(int(t) for t in node_tags)
    return free


def _element_edges(element_data):
    """Unique element edges as sorted (m, 2) node-tag pairs; (0, 2) without elements.

    ``element_data`` is the dict from ``MeshGenerator._capture_element_data``
    (primary-node connectivity per element block, all node tags and xy).
    """
    blocks = [np.asarray(block["connectivity"], dtype=np.int64) for block in element_data["blocks"]]
    if not blocks:
        return np.empty((0, 2), dtype=np.int64)
    # Each element's edges join consecutive corners (closing back to the first).
    edges = np.concatenate([
        np.stack([conn, np.roll(conn, -1, axis=1)], axis=-1).reshape(-1, 2) for conn in blocks
    ])
    return np.unique(np.sort(edges, axis=1), axis=0)


def _node_edges_from_elements(element_data, node_tags):
    """Unique element edges between two of ``node_tags``, as (m, 2) positions into ``node_tags``.

    Edges with an end outside ``node_tags`` are dropped. ``element_data`` is
    as for ``_element_edges``.
    """
    node_tags = np.asarray(node_tags, dtype=np.int64)
    edges = _element_edges(element_data)
    if len(edges) == 0 or len(node_tags) == 0:
        return np.empty((0, 2), dtype=np.int64)
    order = np.argsort(node_tags, kind="stable")
    sorted_tags = node_tags[order]
    at = np.minimum(np.searchsorted(sorted_tags, edges), len(sorted_tags) - 1)
    known = (sorted_tags[at] == edges).all(axis=1)
    return order[at[known]].astype(np.int64)


def _node_sizes_from_elements(element_data, node_tags, fallback):
    """Mean length of the unique element edges at each of ``node_tags``; ``fallback`` where a node has none.

    ``element_data`` is as for ``_element_edges``.
    """
    node_tags = np.asarray(node_tags, dtype=np.int64)
    sizes = np.full(len(node_tags), float(fallback))
    edges = _element_edges(element_data)
    if len(edges) == 0 or len(node_tags) == 0:
        return sizes

    known_tags, first = np.unique(np.asarray(element_data["node_tags"], dtype=np.int64), return_index=True)
    known_xy = np.asarray(element_data["node_xy"], dtype=float)[first]
    position = np.searchsorted(known_tags, edges)
    position = np.minimum(position, len(known_tags) - 1)
    resolved = (known_tags[position] == edges).all(axis=1)
    edges, position = edges[resolved], position[resolved]
    if len(edges) == 0:
        return sizes
    length = np.hypot(*(known_xy[position[:, 0]] - known_xy[position[:, 1]]).T)

    end_tags, inverse = np.unique(edges.ravel(), return_inverse=True)
    total = np.bincount(inverse, weights=np.repeat(length, 2), minlength=len(end_tags))
    count = np.bincount(inverse, minlength=len(end_tags))
    mean = total / np.maximum(count, 1)

    at = np.minimum(np.searchsorted(end_tags, node_tags), len(end_tags) - 1)
    has_edges = (end_tags[at] == node_tags) & (mean[at] > 0)
    sizes[has_edges] = mean[at[has_edges]]
    return sizes


# gmsh_map key for each fragment-input kind recorded in _GeometryInventory.
_MAP_KEY_BY_KIND = {
    'point': 'points',
    'straddle_point': 'straddle_points',
    'line': 'lines',
    'surface': 'surfaces',
    'structured_buffer_surf': 'structured_buffer_surfs',
}


@dataclasses.dataclass
class _GeometryInventory:
    """Per-call bookkeeping of the OCC entities created by ``_add_geometry``."""

    # Embedded entities take part in fragmentation; input_tag_info maps each
    # pre-fragment (dim, tag) to {'type', 'id'} so the fragment map can be
    # traced back to features.
    input_tag_info: dict = dataclasses.field(default_factory=dict)
    embedded_point_tags: list = dataclasses.field(default_factory=list)
    embedded_line_tags: list = dataclasses.field(default_factory=list)
    embedded_surface_tags: list = dataclasses.field(default_factory=list)
    # Non-embedded (field-only) entities per feature id; they do not fragment
    # but still receive size fields. Straddle pairs are keyed by their *line*
    # id, apart from point features (whose ids share the same 0..n range).
    # Field-only polygons also keep their boundary curves (poly_curves).
    nonembedded_point_tags: dict = dataclasses.field(default_factory=dict)
    nonembedded_straddle_tags: dict = dataclasses.field(default_factory=dict)
    nonembedded_line_tags: dict = dataclasses.field(default_factory=dict)
    nonembedded_surface_tags: dict = dataclasses.field(default_factory=dict)
    nonembedded_poly_curve_tags: dict = dataclasses.field(default_factory=dict)
    # (feature id, polygon) created only after fragmentation.
    pending_nonembedded_polys: list = dataclasses.field(default_factory=list)
    # Per quad-buffered feature: lc, thickness, strip corners/side lines.
    structured_buffer_specs: dict = dataclasses.field(default_factory=dict)

    def record_embedded(self, key, kind, feature_id):
        """Register an entity that takes part in fragmentation."""
        self.input_tag_info[key] = {'type': kind, 'id': feature_id}
        by_dim = (self.embedded_point_tags, self.embedded_line_tags, self.embedded_surface_tags)
        by_dim[key[0]].append(key)

    def object_tags(self):
        """Fragment inputs: surfaces, then lines, then points, each in creation order."""
        return self.embedded_surface_tags + self.embedded_line_tags + self.embedded_point_tags

    def feature_map(self):
        """A gmsh_map seeded with the non-embedded tags (shallow copies)."""
        return {
            'points': dict(self.nonembedded_point_tags),
            'straddle_points': dict(self.nonembedded_straddle_tags),
            'lines': dict(self.nonembedded_line_tags),
            'surfaces': dict(self.nonembedded_surface_tags),
            'structured_buffer_surfs': {},
            'poly_curves': dict(self.nonembedded_poly_curve_tags),
        }


# --- Mesh-size field helpers (used by MeshGenerator._setup_fields) ---

_FIELD_GEOM_TYPES = ('points', 'lines', 'surfaces')


def _optional_float(row, key, default=None):
    """``row[key]`` as a float, or ``default`` when it is missing or NaN."""
    value = row.get(key, None)
    if value is None or pd.isna(value):
        return default
    return float(value)


def _dimtag_tags(entries):
    """Integer tags from a list of (dim, tag) pairs or raw tags."""
    return [item[1] if isinstance(item, (tuple, list)) and len(item) >= 2 else item
            for item in entries]


def _explicit_fields(value):
    """A feature's ``fields`` value as a list of MeshField (None/NaN and non-fields dropped)."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, MeshField):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if isinstance(v, MeshField)]
    return []


def _implicit_size_field(row, background_lc, has_explicit_fields):
    """The size field backing a feature's resolution, or None.

    Default: a GeometricGrowthField that grows the mesh from the feature size
    up to the background size at the feature's growth_factor
    (DEFAULT_GROWTH_FACTOR when unset). Created only when the feature is finer
    than the background and has no explicit ``fields``.

    Legacy (deprecated): if dist_min/dist_max are supplied, honor them as the
    old linear ThresholdField and emit a DeprecationWarning. This path is kept
    (even alongside explicit fields) so existing models still mesh.
    """
    lc = _optional_float(row, 'lc')
    if lc is None:
        return None
    dist_min = _optional_float(row, 'dist_min')
    dist_max = _optional_float(row, 'dist_max')

    if dist_min is not None or dist_max is not None:
        warnings.warn(
            "dist_min/dist_max are deprecated for feature size transitions; they "
            "select the legacy linear ThresholdField. Omit them to use the default "
            "GeometricGrowthField (tune it with growth_factor), or pass an explicit "
            "ThresholdField in `fields` to keep a linear ramp.",
            DeprecationWarning,
            stacklevel=4,
        )
        # DistMin: at least one local element size; DistMax: broad scale.
        if dist_min is None:
            dist_min = lc
        if dist_max is None:
            dist_max = background_lc * 5.0
        dist_min = max(dist_min, lc * 0.5)
        # Enforce a gentle gradient relative to SizeMax.
        min_span = 3.0 * background_lc
        if (dist_max - dist_min) < min_span:
            dist_max = dist_min + min_span
        if dist_max <= dist_min:
            dist_max = dist_min + max(background_lc, lc, 1e-3)
        return ThresholdField(size_min=lc, dist_min=dist_min, dist_max=dist_max, size_max=background_lc)

    if has_explicit_fields or lc >= background_lc:
        return None
    return GeometricGrowthField(growth_factor=_optional_float(row, 'growth_factor', DEFAULT_GROWTH_FACTOR))


def _border_grading_field(row):
    """Border grading backing the deprecated add_polygon(border_density=...), or None."""
    border_lc = _optional_float(row, 'border_lc')
    if border_lc is None:
        return None
    return _BorderGradingField(
        border_size=border_lc,
        dist_min=_optional_float(row, 'dist_min', 0.0),
        dist_max=_optional_float(row, 'dist_max_in'),
    )


def _feature_fields(row, geom_type, background_lc):
    """Every MeshField sizing one feature: explicit, implicit, then border grading."""
    explicit = _explicit_fields(row.get('fields', None))
    fields = list(explicit)
    implicit = _implicit_size_field(row, background_lc, has_explicit_fields=bool(explicit))
    if implicit is not None:
        fields.append(implicit)
    if geom_type == 'surfaces':
        border = _border_grading_field(row)
        if border is not None:
            fields.append(border)
    return fields


@dataclasses.dataclass
class _FieldGroup:
    """One MeshField (at one feature lc) and the feature ids it sizes, per geometry type."""
    field: MeshField
    feature_lc: float | None
    feature_ids: dict = dataclasses.field(
        default_factory=lambda: {geom_type: [] for geom_type in _FIELD_GEOM_TYPES}
    )


def _group_feature_fields(points_gdf, lines_gdf, polygons_gdf, background_lc):
    """Group features by (field, feature lc) so each group becomes one Gmsh field.

    Groups are not split by geometry type: a single Distance/Threshold field
    can target points, curves and surfaces at once. ``feature_lc`` is part of
    the key because growth fields compute their transition from it.
    """
    groups = {}
    for gdf, geom_type in zip((points_gdf, lines_gdf, polygons_gdf), _FIELD_GEOM_TYPES):
        for idx, row in gdf.iterrows():
            fields = _feature_fields(row, geom_type, background_lc)
            if not fields:
                continue
            lc = _optional_float(row, 'lc')
            for field in fields:
                group = groups.setdefault((hash(field), lc), _FieldGroup(field, lc))
                group.feature_ids[geom_type].append(int(idx))
    return list(groups.values())


def _field_target_tags(gmsh_map, feature_ids, field_only_polygons):
    """Gmsh tags a field group targets, as the tags_dict MeshField.create expects.

    ``field_only_polygons`` maps field-only polygon ids to their geometry.
    Their surfaces are not part of the domain mesh, so fields locate their
    interiors from these geometries ('field_only_polygons') instead.
    """
    tags = {'points': [], 'lines': [], 'surfaces': [],
            'embedded_surfaces': [], 'field_only_surfaces': [],
            'field_only_polygons': []}
    buffer_surfs = gmsh_map.get('structured_buffer_surfs', {})

    for fid in feature_ids['points']:
        if fid in gmsh_map.get('points', {}):
            tags['points'].extend(_dimtag_tags(gmsh_map['points'][fid]))

    for fid in feature_ids['lines']:
        if fid in gmsh_map.get('lines', {}):
            tags['lines'].extend(_dimtag_tags(gmsh_map['lines'][fid]))
        elif fid in gmsh_map.get('straddle_points', {}):
            # Straddle/barrier lines are represented by point pairs.
            tags['points'].extend(_dimtag_tags(gmsh_map['straddle_points'][fid]))
        elif ('line', fid) in buffer_surfs:
            # Buffer strips are embedded surfaces; list them as such so
            # distance-growth fields target their boundary curves (an empty
            # 'embedded_surfaces' would disable the field).
            surface_tags = _dimtag_tags(buffer_surfs[('line', fid)])
            tags['surfaces'].extend(surface_tags)
            tags['embedded_surfaces'].extend(surface_tags)

    for fid in feature_ids['surfaces']:
        if fid in gmsh_map.get('surfaces', {}):
            surface_tags = _dimtag_tags(gmsh_map['surfaces'][fid])
            # A buffered polygon's outline lives in its band surfaces (the
            # interior is inset), so include them for field targeting too.
            if ('poly', fid) in buffer_surfs:
                surface_tags = surface_tags + _dimtag_tags(buffer_surfs[('poly', fid)])
            tags['surfaces'].extend(surface_tags)
            if fid in field_only_polygons:
                tags['field_only_surfaces'].extend(surface_tags)
                tags['field_only_polygons'].append(field_only_polygons[fid])
            else:
                tags['embedded_surfaces'].extend(surface_tags)
        elif fid in gmsh_map.get('poly_curves', {}):
            # Field-only polygons without a surface: size from their boundary curves.
            tags['lines'].extend(_dimtag_tags(gmsh_map['poly_curves'][fid]))
    return tags


def _activate_size_fields(field_ids):
    """Set Min(field_ids) as the background mesh and turn off Gmsh's own sizing."""
    if field_ids:
        # At any (x, y), Gmsh takes the smallest requested element size.
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [float(f) for f in field_ids])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
    # Otherwise sizing from points/curvature/boundary competes with the fields.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


# --- Explicit embedding helpers (used by MeshGenerator._embed_features) ---

@dataclasses.dataclass
class _EmbedStats:
    """Outcome counts of explicit embedding, reported in diagnostics['embedding']."""
    ok: int = 0
    failed: int = 0
    skip_bbox: int = 0
    skip_no_cand: int = 0
    skip_no_match: int = 0
    boundary_skip: int = 0
    nonconforming_skip: int = 0
    multi_match: int = 0
    inside_failed: int = 0
    boundary_tags: list = dataclasses.field(default_factory=list)
    nonconforming_tags: list = dataclasses.field(default_factory=list)
    multi_tags: list = dataclasses.field(default_factory=list)
    unmatched_tags: list = dataclasses.field(default_factory=list)
    fail_tags: list = dataclasses.field(default_factory=list)
    inside_fail_tags: list = dataclasses.field(default_factory=list)
    records: list = dataclasses.field(default_factory=list)

    def as_diagnostics(self, include_records):
        """The diagnostics['embedding'] dict."""
        report = {name: getattr(self, name) for name in (
            'ok', 'failed', 'skip_bbox', 'skip_no_cand', 'skip_no_match',
            'boundary_skip', 'nonconforming_skip', 'multi_match', 'inside_failed')}
        report.update(boundary_tags=list(self.boundary_tags),
                      nonconforming_tags=list(self.nonconforming_tags),
                      multi_tags=list(self.multi_tags),
                      unmatched_tags=list(self.unmatched_tags),
                      fail_tags=list(self.fail_tags))
        if include_records:
            report['records'] = list(self.records)
        return report


def _domain_surface_tags(gmsh_map, polygons_gdf):
    """Surface tags of the embedded polygons: the pool features are embedded into."""
    tags = set()
    for idx, row in polygons_gdf.iterrows():
        if is_embedded(row) and idx in gmsh_map.get('surfaces', {}):
            for dt in gmsh_map['surfaces'][idx]:
                if isinstance(dt, (tuple, list)) and len(dt) >= 2 and dt[0] == 2:
                    tags.add(dt[1])
    return tags


def _surface_bboxes(surface_tags):
    """Bounding box (xmin, ymin, zmin, xmax, ymax, zmax) per surface; unmeasurable ones are left out."""
    bboxes = {}
    for tag in surface_tags:
        try:
            bboxes[tag] = gmsh.model.getBoundingBox(2, tag)
        except Exception:
            logger.debug("No bounding box for domain surface %d; it is "
                         "excluded from the embedding candidate pool.", tag)
    return bboxes


def _bbox_contains_point(bbox, pt, eps=1e-4):
    """True if ``pt`` lies inside ``bbox`` grown by ``eps``."""
    return (bbox[0] - eps <= pt[0] <= bbox[3] + eps and
            bbox[1] - eps <= pt[1] <= bbox[4] + eps and
            bbox[2] - eps <= pt[2] <= bbox[5] + eps)


def _entity_sample_points(dim, tag, bbox):
    """Points identifying the surface that owns an entity: a point's location, or three interior samples of a curve."""
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    if dim == 0:
        return [((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)]
    if dim != 1:
        return []

    pmin, pmax = gmsh.model.getParametrizationBounds(1, tag)
    lo = float(pmin[0])
    hi = float(pmax[0])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return []
    if hi < lo:
        lo, hi = hi, lo

    # Avoid exact endpoints: line ends commonly lie on partition boundaries
    # and are ambiguous. Interior samples identify the trimmed surface that
    # actually owns the line fragment.
    points = []
    seen = set()
    for f in (0.25, 0.5, 0.75):
        val = gmsh.model.getValue(1, tag, [lo + (hi - lo) * f])
        pt = (float(val[0]), float(val[1]), float(val[2]))
        key = (round(pt[0], 8), round(pt[1], 8), round(pt[2], 8))
        if key not in seen:
            seen.add(key)
            points.append(pt)
    return points


def _unmatched_record(dim, tag, bbox):
    """(dim, tag, bbox centre) of an entity no domain surface was found for."""
    return (int(dim), int(tag),
            (round((bbox[0] + bbox[3]) / 2.0, 6), round((bbox[1] + bbox[4]) / 2.0, 6)))


def _surface_area(surf_tag):
    """Area of a surface (inf if it cannot be measured)."""
    try:
        return float(gmsh.model.occ.getMass(2, int(surf_tag)))
    except Exception:
        try:
            return float(gmsh.model.getMass(2, int(surf_tag)))
        except Exception:
            return float("inf")


def _curve_adjacent_surfaces(curve_tag):
    """Surfaces a curve bounds (none if the adjacency lookup fails)."""
    try:
        up, _down = gmsh.model.getAdjacencies(1, curve_tag)
    except Exception:
        logger.debug("getAdjacencies failed for curve %d; treating it as "
                     "interior for embedding.", curve_tag)
        return []
    return sorted({int(v) for v in up})


def _endpoint_surfaces(curve_tag):
    """Surfaces bounded by a curve at each of its endpoints (empty sets if the lookup fails)."""
    try:
        _up, endpoints = gmsh.model.getAdjacencies(1, curve_tag)
        surfaces = []
        for point_tag in endpoints:
            curves, _down = gmsh.model.getAdjacencies(0, int(point_tag))
            surfaces.append({int(s) for c in curves for s in gmsh.model.getAdjacencies(1, int(c))[0]})
    except Exception:
        logger.debug("getAdjacencies failed for the endpoints of curve %d; "
                     "treating them as interior for embedding.", curve_tag)
        return []
    return surfaces


def _point_entity_nodes(dimtags):
    """Mesh node tag -> (x, y) of the dim-0 entities in a point feature's map entry."""
    nodes = {}
    for dimtag in dimtags:
        if int(dimtag[0]) != 0:
            continue
        node_tags, coords, _ = gmsh.model.mesh.getNodes(0, int(dimtag[1]))
        for tag, xy in zip(node_tags, np.asarray(coords, dtype=float).reshape(-1, 3)):
            nodes[int(tag)] = (float(xy[0]), float(xy[1]))
    return nodes


def _hex_ring_intact(dimtags, centre, element_nodes):
    """True when the meshed ring around ``centre`` is the fan of 6 triangles onto its 6 seeds.

    ``dimtags`` is the point feature's gmsh_map entry (centre plus seeds);
    ``element_nodes`` holds one (n_elements, n_nodes) tag array per 2D element type.
    """
    nodes = _point_entity_nodes(dimtags)
    if len(nodes) != 7:
        return False
    centre_tag = min(nodes, key=lambda t: math.dist(nodes[t], (centre.x, centre.y)))
    incident = [block[(block == centre_tag).any(axis=1)] for block in element_nodes]
    incident = [block for block in incident if len(block)]
    if len(incident) != 1 or incident[0].shape != (6, 3):
        return False
    neighbours = {int(t) for t in incident[0].ravel()} - {centre_tag}
    return neighbours == set(nodes) - {centre_tag}


def _entities_to_embed(gmsh_map, points_gdf, lines_gdf):
    """(dim, tag) of every embedded point, line and barrier straddle point, in embedding order."""
    entities = []
    if points_gdf is not None:
        for idx, row in points_gdf.iterrows():
            if is_embedded(row):
                entities.extend((0, dt[1]) for dt in gmsh_map.get('points', {}).get(idx, [])
                                if dt[0] == 0)
    if lines_gdf is not None:
        for idx, row in lines_gdf.iterrows():
            if is_embedded(row):
                entities.extend((1, dt[1]) for dt in gmsh_map.get('lines', {}).get(idx, [])
                                if dt[0] == 1)
                entities.extend((0, dt[1]) for dt in gmsh_map.get('straddle_points', {}).get(idx, [])
                                if dt[0] == 0)
    return entities


class MeshGenerator:
    def __init__(self, background_lc=None, verbosity=None, mesh_algorithm=6,
                 smoothing_steps=10, optimization_cycles=2,
                 tolerance_initial_delaunay=1e-8,
                 heal_shapes=False, heal_tolerance=1e-8,
                 heal_fix_degenerated=True, heal_fix_small_edges=True,
                 heal_fix_small_faces=True, diagnose=False):
        """
        Initializes the Gmsh-based mesh generator.

        This class is responsible for taking clean geometric inputs and using Gmsh
        to produce a high-quality triangular mesh.

        Args:
            background_lc (float, optional): The default target mesh size for areas
                not controlled by a specific refinement field.
            verbosity (int, optional): Output level while ``generate()`` runs
                (0=warnings only, 1=progress, 2=debug diagnostics). It applies to
                both vorflow's logger and Gmsh, and only for the duration of
                ``generate()``. None (default) follows the package-wide level set
                with ``vorflow.set_verbosity()``.
            mesh_algorithm (int): The 2D mesh algorithm to use. Common choices are
                5 (Delaunay) for speed or 6 (Frontal-Delaunay) for quality.
            smoothing_steps (int): Gmsh ``Mesh.Smoothing``: the number of Laplacian
                smoothing passes over the triangle-mesh nodes during mesh generation.
            optimization_cycles (int): Number of extra gmsh ``Relocate2D`` +
                ``Laplace2D`` passes over the triangles after the initial mesh is
                generated. Both options improve triangle shape; neither centres the
                Voronoi generators within their cells. For Lloyd (centroidal
                Voronoi) relaxation of the generators, use
                ``VoronoiTessellator(lloyd_iterations=...)``.
            tolerance_initial_delaunay (float): Tolerance for the initial Delaunay
                point insertion. Increase this (e.g. 1e-4, 1e-2) to handle
                "Could not insert point" errors caused by near-degenerate geometry
                after fragmentation. This is a meshing-phase tolerance — it does NOT
                alter the CAD topology, so no surfaces or lines are lost.
                Default is 1e-8 (Gmsh default).
            heal_shapes (bool): If True, run OCC topology healing after
                fragmentation. This can fix degenerate geometry that causes
                meshing failures, but may also merge or delete small entities.
                Use with caution on complex models — keep heal_tolerance small.
                Default is False.
            heal_tolerance (float): Size threshold for healShapes. Entities
                smaller than this may be removed or merged. Default 1e-8. only works if heal_shapes=True.
            heal_fix_degenerated (bool): Fix degenerated edges/faces. Default True. Only works if heal_shapes=True.
            heal_fix_small_edges (bool): Remove edges smaller than tolerance. Default True. Only works if heal_shapes=True.
            heal_fix_small_faces (bool): Remove faces smaller than tolerance. Default True. Only works if heal_shapes=True.
            diagnose (bool): If True, retain structured diagnostic details from
                geometry transfer, embedding, and meshing steps.
        """
        self.background_lc = background_lc
        self.verbosity = verbosity
        # Resolved level used by internal gates; refreshed at each generate().
        self._verbosity = self._resolve_verbosity()
        self.mesh_algorithm = mesh_algorithm
        self.smoothing_steps = smoothing_steps
        self.optimization_cycles = optimization_cycles
        self.tolerance_initial_delaunay = tolerance_initial_delaunay
        self.heal_shapes = heal_shapes
        self.heal_tolerance = heal_tolerance
        self.heal_fix_degenerated = heal_fix_degenerated
        self.heal_fix_small_edges = heal_fix_small_edges
        self.heal_fix_small_faces = heal_fix_small_faces
        self.diagnose = bool(diagnose)

        self.initialized = False
        self.nodes = None
        self.node_tags = None
        self.zones_gdf = None
        self.triangular_quality = None
        self.element_grid = None
        self._element_data = None
        # Inputs to VoronoiTessellator(lloyd_iterations=...), set by generate():
        # node_is_free / node_sizes are aligned with self.nodes (True for
        # nodes inside embedded polygon surfaces, off every curve and point;
        # mean incident mesh-edge length), node_edges holds the unique mesh
        # edges between two of self.nodes as (m, 2) positions into it (the
        # graph the Lloyd density is smoothed over), buffer_footprints is the
        # union of the quad-buffer strip and band footprints (None without any).
        self.node_is_free = None
        self.node_sizes = None
        self.node_edges = None
        self.buffer_footprints = None
        self.diagnostics = {}

    def _validate_background_lc(self):
        """Fail before any Gmsh work if background_lc is missing or not positive."""
        if self.background_lc is None:
            raise ValueError(
                "MeshGenerator.background_lc must be provided. "
                "If you don't want to constrain the mesh, pass a very large value."
            )
        value = float(self.background_lc)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "MeshGenerator.background_lc must be a positive finite number. "
                f"Got {self.background_lc!r}."
            )

    def _resolve_verbosity(self) -> int:
        """Verbosity in effect: the explicit setting, else the package level."""
        if self.verbosity is None:
            return current_verbosity()
        return int(self.verbosity)

    def _force_close_polygon(self, poly):
        """Ensure a polygon's exterior and interior rings are closed."""
        if not isinstance(poly, Polygon):
            return poly

        # Close exterior ring
        if poly.exterior.coords[0] != poly.exterior.coords[-1]:
            exterior_coords = list(poly.exterior.coords)
            exterior_coords.append(exterior_coords[0])
            poly = Polygon(exterior_coords, [list(i.coords) for i in poly.interiors])

        # Close interior rings
        new_interiors = []
        for interior in poly.interiors:
            if interior.coords[0] != interior.coords[-1]:
                interior_coords = list(interior.coords)
                interior_coords.append(interior_coords[0])
                new_interiors.append(interior_coords)
            else:
                new_interiors.append(list(interior.coords))
        
        return Polygon(poly.exterior, new_interiors)

    def _initialize_gmsh(self):
        # If Gmsh is already initialized (e.g. leftover from a previous failed
        # run in the same Jupyter kernel), tear it down first so we start clean.
        if gmsh.is_initialized():
            gmsh.finalize()
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", self._verbosity)
        gmsh.option.setNumber("Geometry.Tolerance", 1e-6)
        gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)
        gmsh.model.add("mesh_model")
        self.initialized = True

    def _finalize_gmsh(self):
        if gmsh.is_initialized():
            gmsh.finalize()
            self.initialized = False

    @staticmethod
    def _meshed_surface_tags(gmsh_map, clean_polys):
        """Surface tags composing the meshed domain.

        Embedded polygon surfaces plus structured-buffer strips.
        Field-only (embed=False) surfaces are excluded: gmsh meshes them as
        standalone entities, but they are not part of the deliverable mesh and
        must not pollute element/quality/node collection.
        """
        if clean_polys is not None and not clean_polys.empty:
            if 'embed' in clean_polys.columns:
                poly_ids = [int(i) for i, r in clean_polys.iterrows() if is_embedded(r)]
            else:
                poly_ids = [int(i) for i in clean_polys.index]
        else:
            poly_ids = []

        tags, seen = [], set()

        def add_dimtags(dimtags):
            for dimtag in dimtags:
                if isinstance(dimtag, (tuple, list)) and len(dimtag) >= 2 and int(dimtag[0]) == 2:
                    tag = int(dimtag[1])
                    if tag not in seen:
                        seen.add(tag)
                        tags.append(tag)

        for fid in poly_ids:
            add_dimtags(gmsh_map.get('surfaces', {}).get(fid, []))
        for dimtags in gmsh_map.get('structured_buffer_surfs', {}).values():
            add_dimtags(dimtags)
        return tags

    @staticmethod
    def _get_2d_elements(surface_tags=None):
        """getElements(dim=2), optionally restricted to specific surfaces."""
        if not surface_tags:
            return gmsh.model.mesh.getElements(dim=2)
        by_type = {}
        for tag in surface_tags:
            try:
                element_types, element_tags, element_nodes = gmsh.model.mesh.getElements(2, int(tag))
            except Exception:
                continue
            for etype, etags, enodes in zip(element_types, element_tags, element_nodes):
                bucket = by_type.setdefault(int(etype), ([], []))
                bucket[0].append(np.asarray(etags, dtype=np.int64))
                bucket[1].append(np.asarray(enodes, dtype=np.int64))
        types = list(by_type.keys())
        tags = [np.concatenate(by_type[t][0]) for t in types]
        nodes = [np.concatenate(by_type[t][1]) for t in types]
        return types, tags, nodes

    def _collect_triangular_quality(self, surface_tags=None):
        """Collect gmsh 2D element quality metrics while the model is live."""
        quality_columns = [
            "minSICN",
            "minDetJac",
            "maxDetJac",
            "minSJ",
            "minSIGE",
            "gamma",
            "innerRadius",
            "outerRadius",
            "minIsotropy",
            "angleShape",
            "minEdge",
            "maxEdge",
        ]
        metadata_columns = ["element_tag", "element_type", "element_name", "is_triangle"]
        element_types, element_tags, _ = self._get_2d_elements(surface_tags)
        if len(element_tags) == 0:
            return pd.DataFrame(columns=metadata_columns + quality_columns)

        frames = []
        unavailable_quality_measures = {}
        for element_type, tags_for_type in zip(element_types, element_tags):
            tags = np.asarray(tags_for_type, dtype=np.int64)
            if len(tags) == 0:
                continue

            element_name, _, _, _, _, _ = gmsh.model.mesh.getElementProperties(int(element_type))
            qualities = {
                "element_tag": tags,
                "element_type": int(element_type),
                "element_name": element_name,
                "is_triangle": "triangle" in element_name.lower(),
            }
            for measure in quality_columns:
                if measure in unavailable_quality_measures:
                    qualities[measure] = np.full(len(tags), np.nan)
                    continue
                try:
                    qualities[measure] = gmsh.model.mesh.getElementQualities(tags, measure)
                except Exception as exc:
                    if "Unknown quality name" not in str(exc):
                        raise
                    unavailable_quality_measures[measure] = str(exc)
                    qualities[measure] = np.full(len(tags), np.nan)

            frames.append(pd.DataFrame(qualities))

        if not frames:
            return pd.DataFrame(columns=metadata_columns + quality_columns)

        if unavailable_quality_measures:
            logger.warning(
                "Gmsh %s does not provide quality measures %s; "
                "their report columns contain NaN.",
                getattr(gmsh, "__version__", "unknown"),
                ", ".join(sorted(unavailable_quality_measures)),
            )

        return pd.concat(frames, ignore_index=True)[metadata_columns + quality_columns]

    def get_triangular_quality(self):
        """
        Return cached gmsh 2D element quality metrics for the generated mesh.

        The metrics are collected during ``generate()`` before gmsh is finalized,
        so this method can be called after the normal mesh-generation lifecycle.
        The report includes all 2D element types and marks triangle elements in
        ``is_triangle`` so mixed tri/quad meshes are explicit.
        """
        if self.triangular_quality is None:
            raise RuntimeError(
                "Triangular quality is not available. Call MeshGenerator.generate() first."
            )
        return self.triangular_quality.copy()

    def _empty_element_grid(self, crs=None):
        return gpd.GeoDataFrame(
            columns=[
                "element_tag",
                "element_type",
                "element_name",
                "is_triangle",
                "is_quad",
                "node_tags",
                "centroid_x",
                "centroid_y",
                "zone_id",
                "z_order",
                "geometry",
            ],
            geometry="geometry",
            crs=crs,
        )

    def _capture_element_data(self, surface_tags=None):
        """Copy raw 2D element connectivity and node coordinates while gmsh is live."""
        element_types, element_tags, element_node_tags = self._get_2d_elements(surface_tags)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        blocks = []
        for element_type, tags_for_type, nodes_for_type in zip(
            element_types, element_tags, element_node_tags
        ):
            element_name, _, _, num_nodes, _, num_primary_nodes = gmsh.model.mesh.getElementProperties(
                int(element_type)
            )
            num_nodes = int(num_nodes)
            num_primary_nodes = int(num_primary_nodes) if int(num_primary_nodes) > 0 else num_nodes
            tags = np.asarray(tags_for_type, dtype=np.int64)
            flat_nodes = np.asarray(nodes_for_type, dtype=np.int64)
            if num_nodes <= 0 or num_primary_nodes < 3 or len(tags) == 0 or len(flat_nodes) == 0:
                continue
            connectivity = flat_nodes.reshape((len(tags), num_nodes))[:, :num_primary_nodes]
            blocks.append({
                "element_type": int(element_type),
                "element_name": element_name,
                "tags": tags,
                "connectivity": connectivity.copy(),
            })
        return {
            "blocks": blocks,
            "node_tags": np.asarray(node_tags, dtype=np.int64),
            "node_xy": np.asarray(node_coords, dtype=float).reshape(-1, 3)[:, :2].copy(),
        }

    @staticmethod
    def _element_block_frame(block, node_index, node_xy, crs):
        """Build the element polygons of one gmsh element type."""
        connectivity = block["connectivity"]
        known = np.isin(connectivity, node_index.index.to_numpy()).all(axis=1)
        tags = block["tags"][known]
        connectivity = connectivity[known]
        if len(tags) == 0:
            return None
        rows = node_index.loc[connectivity.ravel()].to_numpy().reshape(connectivity.shape)
        polygons = shapely.polygons(node_xy[rows])
        invalid = ~shapely.is_valid(polygons)
        if invalid.any():
            polygons[invalid] = shapely.make_valid(polygons[invalid])
        usable = (
            (shapely.get_type_id(polygons) == shapely.GeometryType.POLYGON)
            & ~shapely.is_empty(polygons)
            & (shapely.area(polygons) > 0)
        )
        polygons, tags, connectivity = polygons[usable], tags[usable], connectivity[usable]
        if len(tags) == 0:
            return None
        name = block["element_name"]
        name_lower = name.lower()
        centroids = shapely.centroid(polygons)
        return gpd.GeoDataFrame(
            {
                "element_tag": tags,
                "element_type": block["element_type"],
                "element_name": name,
                "is_triangle": "triangle" in name_lower,
                "is_quad": "quadrangle" in name_lower or "quadrilateral" in name_lower,
                "node_tags": [tuple(int(t) for t in row) for row in connectivity],
                "centroid_x": shapely.get_x(centroids),
                "centroid_y": shapely.get_y(centroids),
            },
            geometry=polygons,
            crs=crs,
        )

    def _build_element_grid(self, element_data, zones_gdf=None):
        """Turn captured element data into a zoned element-polygon GeoDataFrame."""
        crs = getattr(zones_gdf, "crs", None)
        if not element_data["blocks"]:
            warnings.warn("gmsh returned no 2D elements; element grid is empty.")
            return self._empty_element_grid(crs)

        node_index = pd.Series(
            np.arange(len(element_data["node_tags"])), index=element_data["node_tags"]
        )
        node_index = node_index[~node_index.index.duplicated()]
        frames = [
            self._element_block_frame(block, node_index, element_data["node_xy"], crs)
            for block in element_data["blocks"]
        ]
        frames = [frame for frame in frames if frame is not None]
        if not frames:
            warnings.warn("gmsh returned no usable 2D elements; element grid is empty.")
            return self._empty_element_grid(crs)

        grid = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs)
        grid = grid.sort_values("element_tag").reset_index(drop=True)
        return _assign_zones_to_elements(grid, _embedded_zones(zones_gdf))

    def get_element_grid(self, element_filter="all"):
        """
        Return gmsh 2D element polygons for the generated mesh.

        The raw element data is captured during ``generate()``; the polygons
        are built on the first call and cached, so users who only need the
        Voronoi grid do not pay for them.

        ``element_filter`` may be ``"all"``, ``"triangles"``, or ``"quads"``.
        The exporter is independent of the Voronoi tessellator and can represent
        mixed tri/quad meshes produced by future structured-buffer workflows.

        Each element is assigned the zone whose polygon intersects the element
        centroid; field-only (``embed=False``) polygons never assign zones. Ties (overlapping zones or centroids on shared borders) are
        broken deterministically: highest ``z_order`` wins, then the zone that
        appears earliest in the conceptual-mesh polygon table.
        """
        if self.element_grid is None and self._element_data is None:
            raise RuntimeError(
                "Element grid is not available. Call MeshGenerator.generate() first."
            )
        if element_filter not in {"all", "triangles", "quads"}:
            raise ValueError("element_filter must be one of 'all', 'triangles', or 'quads'.")
        if self.element_grid is None:
            self.element_grid = self._build_element_grid(self._element_data, self.zones_gdf)
            self._element_data = None

        grid = self.element_grid
        if element_filter == "triangles":
            grid = grid[grid["is_triangle"]]
        elif element_filter == "quads":
            grid = grid[grid["is_quad"]]
        return grid.copy()

    def _dedup_and_remap_fragment_map(self, out_map, object_tags, input_tag_info):
        """Run removeAllDuplicates and remap the fragment map by entity location, in place.

        Unlike healShapes this does not delete or merge entities by a size
        tolerance, so it cannot destroy surfaces or turn interior lines into
        boundaries. It does merge coincident entities and renumber others
        (points, curves and surfaces alike), even reusing a freed tag number
        for a different entity, without returning a mapping. Entity
        signatures are therefore snapshotted beforehand and every out_map
        entry is matched to the surviving entity of the same dimension within
        _DEDUP_MATCH_TOL (the same tag on a tie). Dedup moves nothing, so a
        merged duplicate maps onto the entity it was merged into.
        """
        pre_dedup = _snapshot_signatures(
            {(int(d), int(t)) for entries in out_map for d, t in entries if int(d) in (0, 1, 2)},
            "Pre-dedup snapshot",
        )
        gmsh.model.occ.removeAllDuplicates()
        if len(out_map) == 0:
            return

        occ_alive = {(int(d), int(t)) for dim in range(3) for d, t in gmsh.model.occ.getEntities(dim)}
        survivors = _SurvivorIndex(
            _snapshot_signatures(sorted(occ_alive), "Post-dedup survey"),
            _DEDUP_MATCH_TOL,
        )

        remapped = 0
        pruned = 0
        for i in range(len(out_map)):
            new_entries = []
            for dt in out_map[i]:
                d, t = int(dt[0]), int(dt[1])
                if (d, t) not in pre_dedup:
                    # Unmeasurable before dedup: keep it only if still alive.
                    if (d, t) in occ_alive:
                        new_entries.append(dt)
                    else:
                        pruned += 1
                    continue
                new_tag = survivors.match(d, t, pre_dedup[(d, t)])
                if new_tag is None:
                    pruned += 1
                elif new_tag == t:
                    new_entries.append(dt)
                else:
                    new_entries.append((d, new_tag))
                    remapped += 1
            out_map[i] = new_entries
        if pruned > 0 or remapped > 0:
            logger.info(f"removeAllDuplicates: remapped {remapped}, pruned {pruned} tag(s) from fragment map.")

        self._log_point_survival_after_dedup(object_tags, out_map, input_tag_info, occ_alive)

    def _heal_and_remap_fragment_map(self, out_map, object_tags, input_tag_info):
        """Optionally heal OCC shapes, synchronize, and remap out_map by coordinates, in place.

        healShapes rebuilds OCC topology - renumbering entities and even
        reusing a tag number for a DIFFERENT entity - so remapping matches
        entity signatures (location, not tag identity). Always
        synchronizes the OCC model, even when heal_shapes is off.
        """
        if not self.heal_shapes:
            gmsh.model.occ.synchronize()
            return

        pre_heal = _snapshot_signatures(
            {(int(d), int(t)) for entries in out_map for d, t in entries if int(d) in (0, 1, 2)},
            "Pre-heal snapshot",
        )
        pre_heal_entities = {(int(d), int(t)) for dim in range(3) for d, t in gmsh.model.occ.getEntities(dim)}

        self._heal_occ_shapes()
        gmsh.model.occ.synchronize()
        if len(out_map) == 0:
            return

        surviving, remapped, pruned = self._remap_after_heal(out_map, pre_heal)
        if remapped > 0 or pruned > 0:
            logger.info(f"Heal post-processing: remapped {remapped}, pruned {pruned} tag(s) from fragment map.")

        self._log_point_survival_after_heal(
            object_tags, out_map, input_tag_info, surviving, remapped, pruned
        )
        self._log_heal_dim0_changes(pre_heal_entities, surviving)

    def _heal_occ_shapes(self):
        """Run OCC healShapes with the configured tolerance and fix flags."""
        if self.heal_tolerance > 1e-2:
            logger.info(f"WARNING: heal_tolerance={self.heal_tolerance} is large. "
                        f"This may destroy fragment boundaries and lose surfaces/lines. "
                        f"Consider values <= 1e-3.")
        logger.info(f"Healing OCC shapes (tolerance={self.heal_tolerance}, "
                    f"degenerated={self.heal_fix_degenerated}, "
                    f"small_edges={self.heal_fix_small_edges}, "
                    f"small_faces={self.heal_fix_small_faces})...")
        gmsh.model.occ.healShapes(
            [], tolerance=self.heal_tolerance,
            fixDegenerated=self.heal_fix_degenerated,
            fixSmallEdges=self.heal_fix_small_edges,
            fixSmallFaces=self.heal_fix_small_faces,
            sewFaces=False,
            makeSolids=False,
        )

    @staticmethod
    def _remap_after_heal(out_map, pre_heal):
        """Remap out_map entries to the healed entity with the same signature; returns (surviving, remapped, pruned).

        healShapes introduces ~1e-6 coordinate drift, so signatures match
        within a tolerance (see _match_heal_survivors) that absorbs the drift
        but stays below the size of the entity being matched.
        """
        surviving = {(int(d), int(t)) for dim in range(3) for d, t in gmsh.model.getEntities(dim)}
        survivors = _SurvivorIndex(_snapshot_signatures(sorted(surviving), "Post-heal survey"),
                                   _HEAL_MATCH_TOL)
        matches = _match_heal_survivors(pre_heal, survivors)

        remapped = 0
        pruned = 0
        for i in range(len(out_map)):
            new_entries = []
            for dt in out_map[i]:
                d, t = int(dt[0]), int(dt[1])
                if (d, t) not in pre_heal:
                    # Entity wasn't snapshotted (shouldn't happen); keep if alive.
                    if (d, t) in surviving:
                        new_entries.append(dt)
                    else:
                        pruned += 1
                    continue
                new_tag = matches[(d, t)]
                if new_tag is None:
                    pruned += 1
                elif new_tag == t:
                    new_entries.append(dt)
                else:
                    new_entries.append((d, new_tag))
                    remapped += 1
            out_map[i] = new_entries
        return surviving, remapped, pruned

    @staticmethod
    def _point_feature_survival(object_tags, out_map, input_tag_info, alive):
        """(feature id, n dim-0 map entries, n alive) for each point feature in the fragment map."""
        status = []
        for i, input_dimtag in enumerate(object_tags):
            info = input_tag_info.get(_to_key(input_dimtag[0], input_dimtag[1]), {})
            if info.get('type') != 'point':
                continue
            dim0 = [dt for dt in (out_map[i] if i < len(out_map) else [])
                    if int(dt[0]) == 0]
            n_alive = len([dt for dt in dim0 if (int(dt[0]), int(dt[1])) in alive])
            status.append((info['id'], len(dim0), n_alive))
        return status

    def _log_point_survival_after_dedup(self, object_tags, out_map, input_tag_info, occ_alive):
        """[DIAG] Point features left with no alive tag after removeAllDuplicates."""
        if self._verbosity < 2:
            return
        status = self._point_feature_survival(object_tags, out_map, input_tag_info, occ_alive)
        n_empty = sum(1 for _, _, n_alive in status if n_alive == 0)
        logger.debug(f"[DIAG] Post-dedup point features: {len(status)} total, "
                     f"{n_empty} with 0 alive tags")
        if n_empty > 0:
            for fid, n_entries, n_alive in status:
                if n_alive == 0:
                    logger.debug(f"  [DIAG] Point feat_id={fid}: {n_entries} map entries, 0 alive")

    def _log_point_survival_after_heal(self, object_tags, out_map, input_tag_info,
                                       surviving, remapped, pruned):
        """[DIAG] Point features left with no alive tag after healShapes."""
        if self._verbosity < 2:
            return
        status = self._point_feature_survival(object_tags, out_map, input_tag_info, surviving)
        n_empty = sum(1 for _, _, n_alive in status if n_alive == 0)
        logger.debug(f"[DIAG] Post-heal point features: {len(status)} total, "
                     f"{n_empty} with 0 alive tags (remapped {remapped}, pruned {pruned})")
        if n_empty > 0:
            for fid, n_entries, n_alive in status:
                if n_alive == 0:
                    logger.debug(f"  [DIAG] Point feat_id={fid}: {n_entries} map entries, 0 alive after heal")

    def _log_heal_dim0_changes(self, pre_heal, surviving):
        """[DIAG] Point tags healShapes removed or added."""
        if self._verbosity < 2:
            return
        dim0_removed = [(d, t) for d, t in pre_heal - surviving if d == 0]
        dim0_added = [(d, t) for d, t in surviving - pre_heal if d == 0]
        if dim0_removed or dim0_added:
            logger.debug(f"[DIAG] Heal dim-0 changes: removed {len(dim0_removed)}, added {len(dim0_added)}")
            if dim0_removed:
                logger.debug(f"  [DIAG] Removed point tags: {sorted(t for _, t in dim0_removed)}")
            if dim0_added:
                logger.debug(f"  [DIAG] Added point tags: {sorted(t for _, t in dim0_added)}")

    def _add_geometry(self, polygons_gdf, lines_gdf, points_gdf, launch_gmsh_gui=False):
        """Transfer the clean features into the OCC model and fragment them; returns gmsh_map."""
        gmsh_map, _, _ = self._build_occ_model(
            polygons_gdf, lines_gdf, points_gdf, launch_gmsh_gui=launch_gmsh_gui
        )
        return gmsh_map

    def _build_occ_model(self, polygons_gdf, lines_gdf, points_gdf, launch_gmsh_gui=False):
        """Build and fragment the OCC model.

        Returns (gmsh_map, quad-buffer crossings for _setup_fields, union of
        the quad-buffer strip and band footprints or None).
        """
        inventory = _GeometryInventory()
        domain = buffer.domain_union_geometry(polygons_gdf)

        self._add_point_features(points_gdf, inventory)

        corridors = buffer.protected_corridors(polygons_gdf, lines_gdf, self.background_lc)
        barrier_zone = self._build_barrier_zone(corridors)
        plans = buffer.plan_quad_buffers(polygons_gdf, lines_gdf, self.background_lc, domain)
        crossings = buffer.find_all_crossings(plans)
        straddle_plan = plan_barrier_crossings(
            lines_gdf, self.background_lc, domain, barrier_zone,
            lambda b_idx: buffer.barrier_zone(
                {k: v for k, v in corridors.items() if k != ('line', b_idx)}
            ),
        )

        line_strips = self._add_line_features(
            lines_gdf, inventory, plans, corridors, barrier_zone, domain, straddle_plan
        )
        footprints = self._add_polygon_features(
            polygons_gdf, inventory, plans, corridors, domain, line_strips
        )

        # Call the GUI before fragmentation for debugging.
        if self._verbosity > 1 and launch_gmsh_gui:
            gmsh.model.occ.synchronize()
            gmsh.fltk.run()

        self._log_pre_fragment_diagnostics(inventory)

        object_tags = inventory.object_tags()
        if not object_tags:
            logger.warning("Warning: No geometry to mesh.")
            return inventory.feature_map(), crossings, footprints

        # "Fragment" combines all the individual geometries into a single,
        # topologically consistent model. Only embedded geometry takes part.
        logger.info(f"Fragmenting {len(object_tags)} objects...")
        out_dt, out_map = gmsh.model.occ.fragment(object_tags, [])
        self._dedup_and_remap_fragment_map(out_map, object_tags, inventory.input_tag_info)
        self._heal_and_remap_fragment_map(out_map, object_tags, inventory.input_tag_info)

        self._log_post_fragment_diagnostics(object_tags, out_map, inventory.input_tag_info)

        self._add_field_only_polygons(inventory)
        final_map = self._rebuild_feature_map(object_tags, out_map, inventory)

        self._log_final_map_diagnostics(final_map)
        self._recover_orphan_surfaces(final_map, polygons_gdf)
        self._apply_structured_buffer_meshing(final_map, inventory.structured_buffer_specs)
        return final_map, crossings, footprints

    def _add_point_features(self, points_gdf, inventory):
        """Add one OCC point per point feature, plus its hex-ring seeds when it has them.

        Ring seeds are recorded under the point's own feature id, so
        embedding, node collection and size fields treat them like the point.
        """
        for idx, row in points_gdf.iterrows():
            tag = gmsh.model.occ.addPoint(row.geometry.x, row.geometry.y, 0)
            key = _to_key(0, tag)
            if is_embedded(row):
                inventory.record_embedded(key, 'point', idx)
                for x, y in ring_seed_coords(row):
                    seed_key = _to_key(0, gmsh.model.occ.addPoint(x, y, 0))
                    inventory.record_embedded(seed_key, 'point', idx)
            else:
                inventory.nonembedded_point_tags.setdefault(int(idx), []).append(key)

    def _build_barrier_zone(self, corridors):
        """Union of the protection corridors (None if there are none), logged at verbosity 1."""
        zone = buffer.barrier_zone(corridors)
        if zone is not None and self._verbosity > 0:
            logger.info(f"Constructed Barrier Zone from {len(corridors)} protected features.")
        return zone

    def _add_line_features(self, lines_gdf, inventory, plans, corridors, barrier_zone, domain,
                           straddle_plan=None):
        """Add each line as a quad buffer, straddle point pairs or plain curves; returns the strip footprints.

        ``straddle_plan`` (see _straddle.plan_barrier_crossings) fixes the
        straddle pairs at crossings of standard lines and gives those lines
        the geometry that ends on them.
        """
        fixed = straddle_plan.fixed if straddle_plan is not None else {}
        line_parts = straddle_plan.line_parts if straddle_plan is not None else {}
        line_strips = []
        for idx, row in lines_gdf.iterrows():
            is_barrier = row_bool(row, 'is_barrier', False)
            quad_buffer = row_bool(row, 'quad_buffer', False)
            straddle = positive_number(row.get('straddle_width'))
            lc = feature_lc(row, self.background_lc)
            embedded = is_embedded(row)
            if quad_buffer:
                line_strips.extend(self._add_line_quad_buffer(
                    idx, row, lc, inventory, plans, corridors, domain
                ))
            elif is_barrier or straddle is not None:
                self._add_straddle_points(
                    idx, row.geometry, lc, straddle, embedded, inventory, domain,
                    fixed=fixed.get(int(idx)),
                )
            elif int(idx) in line_parts:
                if self._verbosity > 1:
                    new_len = sum(part.length for part in line_parts[int(idx)])
                    logger.info(f"  Line {idx} ends on barrier straddle pairs "
                                f"(Len: {row.geometry.length:.2f} -> {new_len:.2f})")
                for part in line_parts[int(idx)]:
                    if part.length >= 1e-6:
                        self._add_polyline(idx, part, embedded, inventory)
            else:
                self._add_standard_line(idx, row.geometry, embedded, barrier_zone, inventory)
        return line_strips

    def _add_line_quad_buffer(self, idx, row, lc, inventory, plans, corridors, domain):
        """Create a quad-buffered line's strip surfaces, trimmed by priority; returns the strip footprints."""
        key = ('line', int(idx))
        plan = plans.get(key)
        strips, created = [], []
        if plan is not None:
            obstacles = buffer.higher_priority_obstacles(key, plans, corridors)
            feature_label = f"line feature {row.name}"
            for part in plan.parts:
                strip, trimmed = buffer.trim_against_obstacles(
                    part.strip, obstacles, plan.lc, feature_label
                )
                if strip is None:
                    continue
                strips.append(strip)
                created.extend(self._add_buffer_surfaces(
                    strip, key, domain, inventory,
                    corners=None if trimmed else part.corners,
                    side_lines=part.side_lines,
                ))
        if created:
            inventory.structured_buffer_specs[key] = {
                'lc': lc,
                'thickness': buffer.quad_buffer_thickness(row),
                'kind': 'line',
                'strips': [info for _, info in created],
                'n_surfaces_created': len(created),
            }
        elif self._verbosity > 0:
            logger.warning(f"Warning: Structured buffer requested for line {idx}, but no buffer surface was created.")
        return strips

    def _add_straddle_points(self, idx, line, lc, straddle, embedded, inventory, domain=None,
                             fixed=None):
        """Place point pairs at +/-eps along a barrier/straddle line so Voronoi edges follow it.

        The line itself is not added; the pairs become mesh nodes whose
        Voronoi edges trace the original line. For an embedded line, every
        point is kept inside ``domain``: end pairs on an oblique boundary
        slide inward (see _straddle_distances) and any other point outside
        the domain is dropped, since no surface could embed it. ``fixed``
        maps distances along the line to pairs placed exactly there (at
        crossings of standard lines, whose nodes they share; OCC dedup
        merges the coincident points).
        """
        fixed = fixed or {}
        epsilon = straddle_epsilon(lc, straddle)
        # Tangent probe proportional to line length so the offsets work for
        # any CRS units and for lines shorter than a fixed step.
        probe = straddle_probe(line)
        clip = domain if embedded else None
        tol = epsilon * 1e-6
        distances = _straddle_distances(line, lc, epsilon, probe, clip, tol, anchors=list(fixed))
        coords = [xy for d in distances
                  for xy in (fixed[d] if d in fixed else _straddle_pair(line, d, epsilon, probe))]
        if clip is not None and coords:
            coords = self._clip_straddle_points(idx, coords, clip, tol)
        for x, y in coords:
            key = _to_key(0, gmsh.model.occ.addPoint(x, y, 0))
            if embedded:
                inventory.record_embedded(key, 'straddle_point', idx)
            else:
                inventory.nonembedded_straddle_tags.setdefault(int(idx), []).append(key)

    def _clip_straddle_points(self, idx, coords, domain, tol):
        """Drop straddle points outside ``domain``; snap those within ``tol`` of it onto its boundary.

        A slid end pair lands one point on the boundary only up to round-off;
        snapping puts it exactly there, as for a pair on a perpendicular end.
        """
        points = shapely.points(coords)
        near = shapely.dwithin(domain, points, tol)
        if self._verbosity > 1 and not near.all():
            logger.debug(f"[DIAG] Line {idx}: dropped {int((~near).sum())} straddle "
                         f"point(s) outside the domain")
        kept = []
        for xy, point, keep in zip(coords, points, near):
            if not keep:
                continue
            if not domain.covers(point):
                snapped = nearest_points(domain.boundary, point)[0]
                xy = (snapped.x, snapped.y)
            kept.append(xy)
        return kept

    def _add_standard_line(self, idx, geom, embedded, barrier_zone, inventory):
        """Add a plain constraint line, trimmed off the barrier zone, as OCC segments."""
        if barrier_zone and geom.intersects(barrier_zone):
            try:
                original_len = geom.length
                geom = geom.difference(barrier_zone)
                if self._verbosity > 1:
                    logger.info(f"  Line {idx} trimmed by barrier (Len: {original_len:.2f} -> {geom.length:.2f})")
            except Exception as e:
                # Broad on purpose: a failed trim keeps the untrimmed line
                # rather than aborting the whole mesh.
                logger.warning(f"Warning: Failed to trim line {idx}: {e}")
        if geom.is_empty:
            return
        # A line might be split into multiple parts after being trimmed.
        if geom.geom_type == 'LineString':
            parts = [geom]
        elif geom.geom_type == 'MultiLineString':
            parts = geom.geoms
        else:
            parts = []
        for part in parts:
            # Filter out tiny fragments that might remain after trimming.
            if part.length < 1e-6:
                continue
            self._add_polyline(idx, part, embedded, inventory)

    def _add_polyline(self, idx, part, embedded, inventory):
        """Add one LineString as a chain of OCC segments."""
        coords = sanitize_coords(list(part.coords), min_points=2)
        if len(coords) < 2:
            if self._verbosity > 0:
                logger.warning(f"Warning: Skipping degenerate line part for feature {idx} after coordinate cleanup.")
            return
        pt_tags = [gmsh.model.occ.addPoint(x, y, 0) for x, y in coords]
        created_segments = 0
        for i in range(len(pt_tags) - 1):
            try:
                line_tag = gmsh.model.occ.addLine(pt_tags[i], pt_tags[i+1])
            except Exception as e:
                # gmsh's Python API raises plain Exception on OCC errors.
                logger.warning(
                    f"Warning: Skipping invalid line segment {i} for feature {idx} "
                    f"between {coords[i]} and {coords[i+1]}: {e}"
                )
                continue
            key = _to_key(1, line_tag)
            created_segments += 1
            if embedded:
                inventory.record_embedded(key, 'line', idx)
            else:
                inventory.nonembedded_line_tags.setdefault(int(idx), []).append(key)
        if created_segments == 0 and self._verbosity > 0:
            logger.warning(f"Warning: No valid line segments were created for feature {idx}.")

    def _add_polygon_features(self, polygons_gdf, inventory, plans, corridors, domain, line_strips):
        """Add quad-buffer bands, then every polygon minus the buffer footprints; returns their union or None."""
        if polygons_gdf.empty:
            return None
        logger.info(f"Adding {len(polygons_gdf)} polygons to Gmsh...")
        band_geoms = self._add_polygon_quad_buffers(polygons_gdf, inventory, plans, corridors, domain)
        footprints = band_geoms + line_strips
        footprints_union = make_valid(unary_union(footprints)) if footprints else None
        for idx, row in polygons_gdf.iterrows():
            self._add_polygon_feature(idx, row, footprints_union, inventory)
        return footprints_union

    def _add_polygon_quad_buffers(self, polygons_gdf, inventory, plans, corridors, domain):
        """Create the band surfaces of embedded quad-buffered polygons; returns the band footprints.

        A band hugs the full feature outline, so it is created once per
        feature rather than once per MultiPolygon part.
        """
        band_geoms = []
        for idx, row in polygons_gdf.iterrows():
            if not (is_embedded(row) and row_bool(row, 'quad_buffer', False)):
                continue
            key = ('poly', int(idx))
            created, band = self._add_polygon_band(key, inventory, plans, corridors, domain)
            if created:
                inventory.structured_buffer_specs[key] = {
                    'lc': feature_lc(row, self.background_lc),
                    'thickness': buffer.quad_buffer_thickness(row),
                    'kind': 'polygon',
                    'strips': [],
                    'n_surfaces_created': len(created),
                }
                band_geoms.append(band)
            elif self._verbosity > 0:
                logger.warning(f"Warning: Structured buffer requested for polygon {idx}, but no buffer surface was created.")
        return band_geoms

    def _add_polygon_band(self, key, inventory, plans, corridors, domain):
        """Create one polygon's band surfaces, trimmed by priority; returns (created, band) or ([], None)."""
        plan = plans.get(key)
        if plan is None:
            return [], None
        obstacles = buffer.higher_priority_obstacles(key, plans, corridors)
        band, _ = buffer.trim_against_obstacles(
            plan.band, obstacles, plan.lc, f"polygon feature {key[1]}"
        )
        if band is None:
            return [], None
        created = self._add_buffer_surfaces(band, key, domain, inventory)
        if not created:
            return [], None
        return created, band

    def _add_polygon_feature(self, idx, row, footprints_union, inventory):
        """Add an embedded polygon (minus buffer footprints) as surfaces, or defer a field-only one."""
        embedded = is_embedded(row)
        geom = row['geometry']
        if geom.geom_type not in ('Polygon', 'MultiPolygon'):
            return
        # Mesh every embedded polygon minus the band/strip footprints, so the
        # buffer surfaces tile the plane with their neighbours exactly (shared
        # curves merged by removeAllDuplicates) instead of relying on OCC
        # fragment to cut overlapping faces -- which silently refuses in some
        # trimmed-crossing configurations and leaves double-meshed regions. It
        # also keeps a buffered zone's outline out of the mesh entirely:
        # overlap resolution makes neighbours share that outline, so
        # subtracting only from the buffered zone itself would still pin mesh
        # nodes onto it. Zone assignment uses the original polygons, so zone
        # extents are unchanged.
        if (
            embedded
            and footprints_union is not None
            and geom.intersects(footprints_union)
        ):
            geom = make_valid(geom.difference(footprints_union))
        # A footprint touching the outline pinches the ring (see unpinch_polygons).
        geom = unpinch_polygons(geom, _VERTEX_MERGE_TOL)
        for poly in polygon_parts(geom):
            if poly.is_empty:
                continue
            if not embedded:
                # Defer field-only polygon creation until after
                # fragmentation/dedup/healing: overlapping surfaces present
                # during global OCC cleanup can cut or renumber embedded
                # domain surfaces, which violates embed=False semantics.
                inventory.pending_nonembedded_polys.append((int(idx), poly))
                continue
            s_tag = self._create_required_surface(poly, f"polygon {idx}")
            if s_tag is None:
                logger.warning(f"Warning: Skipping degenerate polygon {idx}")
                continue
            inventory.record_embedded(_to_key(2, s_tag), 'surface', idx)

    def _create_required_surface(self, poly, label):
        """Create a surface for an embedded part; None for a sliver OCC rejects, raises for anything wider.

        The part's footprint is already cut out of its neighbours, so losing
        it leaves a hole in the domain: the points and lines inside are then
        not embedded and the Voronoi grid is silently wrong.
        """
        s_tag, _ = self._create_polygon_surface(poly)
        if s_tag is None and poly.area > _VERTEX_MERGE_TOL * poly.length:
            raise RuntimeError(
                f"Could not create an OCC surface for {label} (area {poly.area:.6g}); "
                f"meshing without it would leave a hole in the domain.")
        return s_tag

    def _add_buffer_surfaces(self, buffer_geom, feature_key, domain, inventory,
                             corners=None, side_lines=None):
        """Create OCC surfaces for a buffer footprint clipped to the domain; returns [(key, strip_info)]."""
        if domain is not None and not domain.is_empty:
            buffer_geom = buffer_geom.intersection(domain)
        buffer_geom = unpinch_polygons(make_valid(buffer_geom), _VERTEX_MERGE_TOL)
        parts = [
            poly for poly in polygon_parts(buffer_geom)
            if not poly.is_empty and poly.area > 0
        ]
        if corners is not None and len(parts) != 1:
            # The recorded whole-strip corners no longer apply; pieces are
            # re-cornered individually from the side lines post-fragment.
            corners = None
        created = []
        for poly in parts:
            s_tag = self._create_required_surface(poly, f"the quad buffer of {feature_key[0]} {feature_key[1]}")
            if s_tag is None:
                continue
            key = _to_key(2, s_tag)
            inventory.record_embedded(key, 'structured_buffer_surf', feature_key)
            created.append((key, {'corners': corners, 'side_lines': side_lines}))
        return created

    def _add_field_only_polygons(self, inventory):
        """Create the deferred field-only (embed=False) polygon surfaces after fragmentation."""
        pending = inventory.pending_nonembedded_polys
        if not pending:
            return
        if self._verbosity > 0:
            logger.info(f"Adding {len(pending)} field-only polygon surface(s)...")
        for idx, poly in pending:
            s_tag, boundary_curve_tags = self._create_polygon_surface(poly)
            if s_tag is None:
                if self._verbosity > 0:
                    logger.warning(f"Warning: Skipping degenerate field-only polygon {idx}")
                continue
            inventory.nonembedded_surface_tags.setdefault(int(idx), []).append(_to_key(2, s_tag))
            inventory.nonembedded_poly_curve_tags.setdefault(int(idx), []).extend(
                [(1, int(t)) for t in boundary_curve_tags]
            )
        gmsh.model.occ.synchronize()

    def _create_polygon_surface(self, poly):
        """Create an OCC plane surface; returns (surface tag, boundary curve tags) or (None, [])."""
        if poly.is_empty:
            return None, []
        poly = self._force_close_polygon(poly)
        exterior_loop_tag, exterior_lines = self._create_curve_loop(list(poly.exterior.coords))
        if exterior_loop_tag is None:
            return None, []
        loops = [exterior_loop_tag]
        boundary_curve_tags = list(exterior_lines)
        holes = []
        for interior in poly.interiors:
            # OCC addPlaneSurface reverses the hole loops itself, so it needs
            # them wound like the exterior. Shapely overlay output winds holes
            # opposite to the shell; passed as is, they become an inverted
            # face (area = shell + holes) on which isInside() is wrong, and
            # features inside it are never embedded.
            coords = list(interior.coords)
            if interior.is_ccw != poly.exterior.is_ccw:
                coords.reverse()
            interior_loop_tag, interior_lines = self._create_curve_loop(coords)
            if interior_loop_tag is not None:
                loops.append(interior_loop_tag)
                boundary_curve_tags.extend(interior_lines)
                holes.append(interior)
        try:
            s_tag = gmsh.model.occ.addPlaneSurface(loops)
        except Exception as e:
            # gmsh's Python API raises plain Exception on OCC errors.
            logger.error(f"Error creating surface: {e}")
            return None, []
        if holes:
            self._warn_if_holes_inverted(s_tag, poly.exterior, holes)
        return s_tag, boundary_curve_tags

    @staticmethod
    def _warn_if_holes_inverted(s_tag, exterior, holes):
        """Warn if OCC measures a holed surface with any hole added instead of removed."""
        hole_areas = [Polygon(hole).area for hole in holes]
        expected = Polygon(exterior).area - sum(hole_areas)
        try:
            occ_area = float(gmsh.model.occ.getMass(2, s_tag))
        except Exception:
            # gmsh raises plain Exception for entities without mass properties.
            return
        # An inverted hole adds twice its area.
        if abs(occ_area - expected) > 0.5 * min(hole_areas):
            logger.warning(
                f"Surface {s_tag}: OCC area {occ_area:.6g} differs from the polygon "
                f"area {expected:.6g}, so its holes are probably inverted. Points and "
                f"lines inside it may not be embedded.")

    @staticmethod
    def _create_curve_loop(coords):
        """Create an OCC curve loop through a ring's coordinates; returns (loop tag, curve tags) or (None, [])."""
        clean_coords = sanitize_coords(
            coords, min_spacing=_VERTEX_MERGE_TOL, require_closed=True, min_points=3)
        if len(clean_coords) < 3:
            return None, []
        p_tags = [gmsh.model.occ.addPoint(x, y, 0) for x, y in clean_coords]
        l_tags = []
        for i in range(len(p_tags)):
            p1 = p_tags[i]
            p2 = p_tags[(i + 1) % len(p_tags)]
            try:
                l_tags.append(gmsh.model.occ.addLine(p1, p2))
            except Exception as e:
                # gmsh's Python API raises plain Exception on OCC errors.
                logger.error(f"Error adding line {p1}-{p2}: {e}")
                return None, []
        try:
            return gmsh.model.occ.addCurveLoop(l_tags), l_tags
        except Exception as e:
            # gmsh's Python API raises plain Exception on OCC errors.
            logger.error(f"Error adding curve loop: {e}")
            return None, []

    def _rebuild_feature_map(self, object_tags, out_map, inventory):
        """Map each feature id to its post-fragment dimtags, starting from the non-embedded tags."""
        final_map = inventory.feature_map()
        logger.info(f"Reconstructing Map (Input Tags: {len(object_tags)}, Out Map Len: {len(out_map)})...")
        for i, input_dimtag in enumerate(object_tags):
            res_tags = out_map[i] if i < len(out_map) else [input_dimtag]
            info = inventory.input_tag_info.get(_to_key(input_dimtag[0], input_dimtag[1]))
            assert info is not None, f"fragment input {input_dimtag} was never recorded"
            # Structured-buffer ids are ('line'|'poly', idx) tuples so line
            # and polygon features with the same index cannot collide.
            feat_id = info['id'] if isinstance(info['id'], tuple) else int(info['id'])
            final_map[_MAP_KEY_BY_KIND[info['type']]].setdefault(feat_id, []).extend(res_tags)
        return final_map

    def _recover_orphan_surfaces(self, final_map, polygons_gdf):
        """Attach surfaces the fragment map dropped to the embedded polygon covering (or nearest) them.

        OCC's fragment map can omit pieces of an input surface (observed when
        a buffer strip with boundaries coincident to the densified domain edge
        splits the domain). An unclaimed surface would silently lose its mesh
        nodes and field sizing downstream.
        """
        claimed_surfaces = set()
        for map_key in ('surfaces', 'structured_buffer_surfs'):
            for dimtags in final_map.get(map_key, {}).values():
                for dt in dimtags:
                    if isinstance(dt, (tuple, list)) and len(dt) >= 2 and int(dt[0]) == 2:
                        claimed_surfaces.add(int(dt[1]))
        orphan_surfaces = [
            int(tag) for dim, tag in gmsh.model.getEntities(2)
            if int(tag) not in claimed_surfaces
        ]
        if not orphan_surfaces or polygons_gdf is None or polygons_gdf.empty:
            return
        embedded_polys = [
            (int(idx), row.geometry)
            for idx, row in polygons_gdf.iterrows()
            if is_embedded(row)
        ]
        recovered = 0
        for surf_tag in orphan_surfaces:
            try:
                cx, cy, _ = gmsh.model.occ.getCenterOfMass(2, surf_tag)
            except Exception:
                # gmsh raises plain Exception for entities without mass
                # properties; such a surface cannot be located, so skip it.
                continue
            center = Point(cx, cy)
            owner = None
            for fid, geom in embedded_polys:
                if geom.covers(center):
                    owner = fid
                    break
            if owner is None and embedded_polys:
                owner = min(embedded_polys, key=lambda item: item[1].distance(center))[0]
            if owner is not None:
                final_map['surfaces'].setdefault(owner, []).append((2, surf_tag))
                recovered += 1
        if recovered:
            logger.info(
                f"Recovered {recovered} orphan surface(s) the fragment map had "
                "dropped; re-attached to their containing polygon features."
            )

    def _log_pre_fragment_diagnostics(self, inventory):
        """[DIAG] Counts of the entities about to be fragmented."""
        if self._verbosity < 2:
            return
        line_feats = sorted(set(
            inventory.input_tag_info.get(_to_key(dt[0], dt[1]), {}).get('id', '?')
            for dt in inventory.embedded_line_tags
        )) if inventory.embedded_line_tags else []
        logger.debug(f"\n[DIAG] Pre-fragment: {len(inventory.embedded_surface_tags)} surfs, "
                     f"{len(inventory.embedded_line_tags)} lines, {len(inventory.embedded_point_tags)} pts "
                     f"| line features: {line_feats}")

    def _log_post_fragment_diagnostics(self, object_tags, out_map, input_tag_info):
        """[DIAG] Model size after fragmentation and how line fragments ended up."""
        if self._verbosity < 2:
            return
        all_surfs = gmsh.model.getEntities(2)
        all_lines = gmsh.model.getEntities(1)
        all_pts = gmsh.model.getEntities(0)

        # Classify line fragments: boundary vs interior vs orphan.
        n_boundary, n_interior, n_orphan, n_dim0 = 0, 0, 0, 0
        boundary_feats = set()  # feature ids whose lines became boundaries
        for i, input_dimtag in enumerate(object_tags):
            info = input_tag_info.get(_to_key(input_dimtag[0], input_dimtag[1]), {})
            if info.get('type') != 'line':
                continue
            res = out_map[i] if i < len(out_map) else [input_dimtag]
            for dt in res:
                dim_r, tag_r = int(dt[0]), int(dt[1])
                if dim_r == 0:
                    n_dim0 += 1
                    continue
                try:
                    gmsh.model.getBoundingBox(dim_r, tag_r)
                    up, _ = gmsh.model.getAdjacencies(1, tag_r)
                    if len(up) > 0:
                        n_boundary += 1
                        boundary_feats.add(info.get('id', '?'))
                    else:
                        n_interior += 1
                except Exception:
                    # gmsh raises plain Exception for an entity no longer in
                    # the model; count it as an orphan.
                    n_orphan += 1

        n_auto = 0
        for s in all_surfs:
            try:
                if gmsh.model.mesh.getEmbedded(2, s[1]):
                    n_auto += 1
            except Exception:
                # gmsh raises plain Exception; diagnostics only, so log and go on.
                logger.debug("getEmbedded failed for surface %d during "
                             "post-fragment diagnostics.", s[1])

        logger.debug(f"[DIAG] Post-fragment: {len(all_surfs)} surfs, "
                     f"{len(all_lines)} lines, {len(all_pts)} pts")
        logger.debug(f"[DIAG] Line fragments: {n_interior} interior, "
                     f"{n_boundary} BOUNDARY, {n_orphan} orphan, "
                     f"{n_dim0} became-points | auto-embed surfs: {n_auto}")
        if boundary_feats:
            logger.debug(f"[DIAG] *** Lines from these features became BOUNDARIES: "
                         f"{sorted(boundary_feats)} ***")

    def _log_final_map_diagnostics(self, final_map):
        """[DIAG] Point features with no, or stale, tags in the rebuilt map."""
        if self._verbosity < 2:
            return
        model_ents = set()
        for dim in range(3):
            for dt in gmsh.model.getEntities(dim):
                model_ents.add((int(dt[0]), int(dt[1])))
        empty_feats = []
        stale_feats = []
        for fid, dimtags in final_map.get('points', {}).items():
            dim0 = [dt for dt in dimtags if isinstance(dt, (tuple, list)) and int(dt[0]) == 0]
            if not dim0:
                empty_feats.append(fid)
            else:
                for dt in dim0:
                    if (int(dt[0]), int(dt[1])) not in model_ents:
                        stale_feats.append((fid, int(dt[1])))
        logger.debug(f"[DIAG] Final map: {len(final_map.get('points', {}))} point features, "
                     f"{len(empty_feats)} empty, {len(stale_feats)} with stale tags")
        if empty_feats:
            logger.debug(f"  [DIAG] Empty point feat_ids: {sorted(empty_feats)}")
        if stale_feats:
            logger.debug(f"  [DIAG] Stale point (feat_id, tag): {stale_feats}")

    def _log_post_embed_diagnostics(self, gmsh_map):
        """[DIAG] Surfaces carrying embedded entities and how mapped lines sit in the model."""
        if self._verbosity < 2:
            return
        all_surfs = gmsh.model.getEntities(2)
        with_emb, without_emb, total_emb = 0, 0, 0
        for surf_dt in all_surfs:
            try:
                emb = gmsh.model.mesh.getEmbedded(2, surf_dt[1])
                if emb:
                    with_emb += 1
                    total_emb += len(emb)
                else:
                    without_emb += 1
            except Exception:
                # gmsh raises plain Exception; count the surface as empty.
                without_emb += 1
        # Count line boundary vs interior in gmsh_map.
        map_bnd, map_int, map_miss = 0, 0, 0
        for dimtags in gmsh_map.get('lines', {}).values():
            for dt in dimtags:
                if not (isinstance(dt, (tuple, list)) and len(dt) >= 2):
                    continue
                if int(dt[0]) != 1:
                    continue
                try:
                    up, _ = gmsh.model.getAdjacencies(1, int(dt[1]))
                    if len(up) > 0:
                        map_bnd += 1
                    else:
                        map_int += 1
                except Exception:
                    # gmsh raises plain Exception for a curve not in the model.
                    map_miss += 1
        logger.debug(f"[DIAG] Post-embed: {with_emb}/{len(all_surfs)} surfaces have embeddings "
                     f"({total_emb} total entities) | "
                     f"{without_emb} surfaces empty")
        logger.debug(f"[DIAG] Line map: {map_int} interior, {map_bnd} boundary, {map_miss} missing")

    @staticmethod
    def _entity_length(dim, tag):
        try:
            return float(gmsh.model.occ.getMass(int(dim), int(tag)))
        except Exception:
            try:
                return float(gmsh.model.getMass(int(dim), int(tag)))
            except Exception:
                return None

    @staticmethod
    def _surface_boundary_point_coords(surf_tag):
        """Map of point tag -> (x, y) for a surface's boundary points."""
        try:
            boundary_points = gmsh.model.getBoundary(
                [(2, int(surf_tag))], oriented=False, recursive=True
            )
        except Exception:
            return {}
        candidates = {}
        for dim, tag in boundary_points:
            if int(dim) != 0:
                continue
            try:
                xyz = gmsh.model.getValue(0, int(tag), [])
            except Exception:
                continue
            candidates[int(tag)] = (float(xyz[0]), float(xyz[1]))
        return candidates

    def _derive_strip_corners_on_surface(self, surf_tag, side_lines, tol):
        """Derive the 4 corner point tags of a strip *piece* from its side lines.

        When fragmentation splits a strip (e.g. an embedded zone boundary
        crosses it), each piece is still a 4-sided strip whose corners are the
        extreme boundary points lying on the original positive/negative offset
        curves. Returns corner tags in "Left" order, or None.
        """
        if not side_lines:
            return None
        pos, neg = side_lines
        if pos is None or neg is None:
            return None
        candidates = self._surface_boundary_point_coords(surf_tag)
        if len(candidates) < 4:
            return None

        def extremes_on(line):
            hits = []
            for tag, (x, y) in candidates.items():
                point = Point(x, y)
                if line.distance(point) <= tol:
                    hits.append((float(line.project(point)), tag))
            if len(hits) < 2:
                return None
            hits.sort()
            return hits[0][1], hits[-1][1]

        neg_ends = extremes_on(neg)
        pos_ends = extremes_on(pos)
        if neg_ends is None or pos_ends is None:
            return None
        corner_tags = [neg_ends[0], neg_ends[1], pos_ends[1], pos_ends[0]]
        if len(set(corner_tags)) != 4:
            return None
        return corner_tags

    def _locate_corner_tags_on_surface(self, surf_tag, corner_coords, tol):
        """Match recorded strip corner coordinates to point tags on a surface.

        Only the surface's own boundary points are considered, so coordinate
        collisions with the rest of the model are impossible. Returns the four
        point tags in corner order, or None if any corner has no boundary
        point within ``tol``.
        """
        candidates = self._surface_boundary_point_coords(surf_tag)
        if len(candidates) < 4:
            return None

        corner_tags = []
        for cx, cy in corner_coords:
            best_tag, best_dist = None, None
            for tag, (px, py) in candidates.items():
                dist = math.hypot(px - cx, py - cy)
                if best_dist is None or dist < best_dist:
                    best_tag, best_dist = tag, dist
            if best_dist is None or best_dist > tol:
                return None
            corner_tags.append(best_tag)
        if len(set(corner_tags)) != 4:
            return None
        return corner_tags

    def _partition_boundary_chains(self, surf_tag, corner_tags):
        """Order a surface's boundary curves into 4 chains cut at the corners.

        Returns a list of (start_corner, end_corner, [curve_tags]) tuples, or
        None when the boundary is not a single closed loop through all four
        corner points (e.g. the strip was split by fragmentation).
        """
        try:
            boundary = gmsh.model.getBoundary(
                [(2, int(surf_tag))], oriented=False, recursive=False
            )
        except Exception:
            return None
        curve_tags = [int(tag) for dim, tag in boundary if int(dim) == 1]
        if len(curve_tags) < 4:
            return None

        endpoints = {}
        point_curves = {}
        for curve in curve_tags:
            try:
                pts = gmsh.model.getBoundary([(1, curve)], oriented=False, recursive=False)
            except Exception:
                return None
            point_pair = [int(tag) for dim, tag in pts if int(dim) == 0]
            if len(point_pair) != 2 or point_pair[0] == point_pair[1]:
                return None
            endpoints[curve] = point_pair
            for point in point_pair:
                point_curves.setdefault(point, []).append(curve)
        if any(len(curves) != 2 for curves in point_curves.values()):
            return None

        corner_set = {int(tag) for tag in corner_tags}
        if len(corner_set) != 4 or not corner_set.issubset(point_curves.keys()):
            return None

        start = int(corner_tags[0])
        point = start
        curve = point_curves[start][0]
        chains = []
        chain_start = start
        current = []
        visited = set()
        for _ in range(len(curve_tags)):
            if curve in visited:
                return None
            visited.add(curve)
            current.append(curve)
            a, b = endpoints[curve]
            point = b if point == a else a
            if point in corner_set:
                chains.append((chain_start, point, current))
                chain_start = point
                current = []
            next_curves = [c for c in point_curves[point] if c != curve]
            if len(next_curves) != 1:
                return None
            curve = next_curves[0]
        if current or len(chains) != 4 or chains[-1][1] != start:
            return None
        return chains

    @staticmethod
    def _distribute_chain_points(lengths, total_points):
        """Split a chain's transfinite point budget across its curves.

        Returns per-curve point counts whose segment total matches
        ``total_points - 1`` exactly, or None if the chain has more curves
        than segments.
        """
        total_segments = total_points - 1
        n = len(lengths)
        if total_segments < n:
            return None
        total_length = sum(lengths)
        segments = [
            max(1, int(round(total_segments * length / total_length)))
            for length in lengths
        ]
        drift = total_segments - sum(segments)
        order = sorted(range(n), key=lambda i: -lengths[i])
        attempts = 0
        while drift != 0 and attempts < 10 * n:
            i = order[attempts % n]
            step = 1 if drift > 0 else -1
            if segments[i] + step >= 1:
                segments[i] += step
                drift -= step
            attempts += 1
        if drift != 0:
            return None
        return [s + 1 for s in segments]

    def _apply_transfinite_strip(self, surf_tag, corner_tags, lc, thickness):
        """Apply a 4-corner transfinite structure to a relocated buffer strip.

        Opposite sides of a transfinite surface must carry equal point counts,
        so the along-feature target is computed once from the longer side and
        distributed across each side's curves. End caps get ``thickness + 1``
        points, matching gmshflow. Returns True on success.
        """
        chains = self._partition_boundary_chains(surf_tag, corner_tags)
        if chains is None:
            return False

        ct = [int(tag) for tag in corner_tags]
        roles = {
            frozenset((ct[0], ct[1])): 'side',
            frozenset((ct[2], ct[3])): 'side',
            frozenset((ct[1], ct[2])): 'cap',
            frozenset((ct[3], ct[0])): 'cap',
        }
        sides, caps = [], []
        for start_corner, end_corner, curves in chains:
            role = roles.get(frozenset((start_corner, end_corner)))
            if role == 'side':
                sides.append(curves)
            elif role == 'cap':
                caps.append(curves)
            else:
                return False
        if len(sides) != 2 or len(caps) != 2:
            return False

        def chain_lengths(chains_group):
            result = []
            for curves in chains_group:
                lengths = [self._entity_length(1, curve) for curve in curves]
                if any(v is None or not math.isfinite(v) or v <= 0 for v in lengths):
                    return None
                result.append(lengths)
            return result

        side_lengths = chain_lengths(sides)
        cap_lengths = chain_lengths(caps)
        if side_lengths is None or cap_lengths is None:
            return False

        total_points = max(
            2,
            int(round(max(sum(lengths) for lengths in side_lengths) / max(lc, 1e-12))) + 1,
        )
        # A cap subdivided into more curves than the strip has cell rows (e.g.
        # by a densified domain-boundary vertex) cannot carry thickness+1
        # points; forcing more would interpolate a node row onto the feature
        # line, so the caller falls back to recombine-only instead.
        cap_divisions = [
            self._distribute_chain_points(lengths, int(thickness) + 1)
            for lengths in cap_lengths
        ]
        if any(divisions is None for divisions in cap_divisions):
            return False

        try:
            for curves, lengths in zip(sides, side_lengths):
                divisions = self._distribute_chain_points(lengths, total_points)
                if divisions is None:
                    return False
                for curve, points in zip(curves, divisions):
                    gmsh.model.mesh.setTransfiniteCurve(int(curve), int(points))
            for curves, divisions in zip(caps, cap_divisions):
                for curve, points in zip(curves, divisions):
                    gmsh.model.mesh.setTransfiniteCurve(int(curve), int(points))
            gmsh.model.mesh.setTransfiniteSurface(int(surf_tag), "Left", ct)
        except Exception as e:
            warnings.warn(
                f"Could not apply transfinite structure to buffer surface {surf_tag}: {e}"
            )
            return False
        return True

    def _set_default_buffer_curve_divisions(self, surf_tag, lc):
        """Recombine-only fallback: seed each boundary curve at ~lc spacing."""
        try:
            boundary = gmsh.model.getBoundary(
                [(2, int(surf_tag))], oriented=False, recursive=False
            )
        except Exception:
            boundary = []
        for dim, tag in boundary:
            if int(dim) != 1:
                continue
            length = self._entity_length(1, tag)
            if length is None or not math.isfinite(length) or length <= 0:
                continue
            divisions = max(2, int(round(length / lc)) + 1)
            try:
                gmsh.model.mesh.setTransfiniteCurve(int(tag), divisions)
            except Exception as e:
                warnings.warn(f"Could not set transfinite divisions on curve {tag}: {e}")

    def _apply_structured_buffer_meshing(self, final_map, structured_buffer_specs):
        """Apply transfinite/recombine constraints to relocated buffer surfaces.

        Runs after fragmentation/dedup so OCC re-tagging cannot break the
        structured constraints. Line strips get a true 4-corner transfinite
        structure located by their recorded corner coordinates; polygon bands
        (annuli) and any strip that was trimmed or split are meshed
        recombine-only.
        """
        structured_surfaces = final_map.get('structured_buffer_surfs', {})
        if not structured_surfaces:
            return

        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
        transfinite_count = 0
        recombine_only_count = 0
        for feat_id, dimtags in structured_surfaces.items():
            spec = structured_buffer_specs.get(feat_id, {})
            lc = max(float(spec.get('lc', self.background_lc or 1.0)), 0.001)
            thickness = int(spec.get('thickness', 1))
            strips = [info for info in spec.get('strips', []) if isinstance(info, dict)]
            corner_sets = [info['corners'] for info in strips if info.get('corners')]
            side_line_sets = [info['side_lines'] for info in strips if info.get('side_lines')]
            surf_tags = [
                int(dt[1])
                for dt in dimtags
                if isinstance(dt, (tuple, list)) and len(dt) >= 2 and int(dt[0]) == 2
            ]

            n_created = int(spec.get('n_surfaces_created', 0) or 0)
            if n_created and len(surf_tags) > n_created and self._verbosity > 0:
                logger.info(
                          f"Structured buffer for feature {feat_id} was split by fragmentation "
                          f"({n_created} surface(s) became {len(surf_tags)}); applying the "
                          "transfinite structure per piece."
                )

            tol = max(1e-4, lc * 1e-3)
            for surf_tag in surf_tags:
                structured = False
                # Fast path: the recorded whole-strip corners survived intact.
                for corners in corner_sets:
                    corner_tags = self._locate_corner_tags_on_surface(surf_tag, corners, tol)
                    if corner_tags is not None:
                        structured = self._apply_transfinite_strip(
                            surf_tag, corner_tags, lc, thickness
                        )
                        if structured:
                            break
                # Split/trimmed pieces: re-derive each piece's corners from the
                # extreme boundary points on the original offset side lines.
                if not structured:
                    for side_lines in side_line_sets:
                        corner_tags = self._derive_strip_corners_on_surface(
                            surf_tag, side_lines, tol
                        )
                        if corner_tags is not None:
                            structured = self._apply_transfinite_strip(
                                surf_tag, corner_tags, lc, thickness
                            )
                            if structured:
                                break
                if not structured and (corner_sets or side_line_sets):
                    warnings.warn(
                        f"Could not apply transfinite structure to buffer surface "
                        f"{surf_tag} of feature {feat_id} (the strip was altered by "
                        "fragmentation, e.g. end caps subdivided where the strip meets "
                        "the domain boundary); meshing it recombine-only."
                    )
                if not structured:
                    self._set_default_buffer_curve_divisions(surf_tag, lc)
                try:
                    gmsh.model.mesh.setRecombine(2, int(surf_tag))
                    gmsh.model.mesh.setAlgorithm(2, int(surf_tag), 8)
                except Exception as e:
                    warnings.warn(
                        f"Could not apply recombination to buffer surface {surf_tag}: {e}"
                    )
                if structured:
                    transfinite_count += 1
                else:
                    recombine_only_count += 1

        if self._verbosity > 0:
            logger.info(
                      f"Applied structured quad-buffer meshing to "
                      f"{transfinite_count + recombine_only_count} surface(s) "
                      f"({transfinite_count} transfinite, {recombine_only_count} recombine-only)."
            )
    
    def _setup_fields(self, gmsh_map, polygons_gdf, lines_gdf, points_gdf, crossings=()):
        """Build the mesh-size fields and set their Min as the background mesh.

        Each feature contributes its explicit ``fields`` plus the implicit
        field backing its resolution (see ``_implicit_size_field``); features
        sharing a field and lc become one Gmsh field targeting all of their
        tags. ``crossings`` are the quad-buffer crossing disks from
        ``_build_occ_model``; each becomes a Ball refinement field. A Constant
        field at ``background_lc`` bounds everything from above.
        """
        self._log_field_setup_diagnostics(gmsh_map, polygons_gdf)
        self._validate_background_lc()
        background_lc = float(self.background_lc)
        field_only_polygons = {
            int(idx): row.geometry for idx, row in polygons_gdf.iterrows() if not is_embedded(row)
        }

        field_ids = []
        for group in _group_feature_fields(points_gdf, lines_gdf, polygons_gdf, background_lc):
            tags_dict = _field_target_tags(gmsh_map, group.feature_ids, field_only_polygons)
            if not any(tags_dict.values()):
                continue
            # Private metadata for built-in field helpers; custom MeshField
            # implementations can ignore it because tag lists remain unchanged.
            tags_dict['_verbosity'] = self._verbosity
            f_id = group.field.create(
                gmsh_api=gmsh, tags_dict=tags_dict,
                background_lc=background_lc, feature_lc=group.feature_lc,
            )
            if f_id is not None:
                field_ids.append(f_id)

        field_ids.extend(self._add_crossing_fields(crossings, background_lc))
        field_ids.append(ConstantField(size=self.background_lc).create(
            gmsh_api=gmsh, tags_dict={}, background_lc=self.background_lc, feature_lc=None,
        ))
        _activate_size_fields(field_ids)

    def _log_field_setup_diagnostics(self, gmsh_map, polygons_gdf):
        """[DIAG] Polygon rows versus surface-map keys (a mismatch detaches polygon fields)."""
        if self._verbosity < 2:
            return
        surfaces = gmsh_map.get('surfaces', {})
        logger.debug(f"[DIAG] Setup fields: {len(polygons_gdf)} polygon rows, "
                     f"{len(surfaces)} surface map entries")
        if not polygons_gdf.empty and surfaces:
            first_idx = polygons_gdf.index[0]
            first_key = next(iter(surfaces))
            logger.debug(f"[DIAG]   first polygon index {first_idx!r} ({type(first_idx).__name__}), "
                         f"first map key {first_key!r} ({type(first_key).__name__}), "
                         f"match={first_idx in surfaces}")

    @staticmethod
    def _add_crossing_fields(crossings, background_lc):
        """Add one Ball field per quad-buffer crossing; returns the field ids.

        Where two quad buffers cross, the lower-priority one is trimmed away,
        leaving a small gap the unstructured mesher would otherwise fill at the
        background size right next to the dense strip rows. Each Ball pins the
        crossing region to min(lc) of the two features (rotation-agnostic, no
        OCC geometry added). The Min field takes the smallest requested size,
        and transfinite strips ignore size fields, so the continuous winner is
        unaffected.
        """
        field_ids = []
        for crossing in crossings:
            ball = gmsh.model.mesh.field.add("Ball")
            gmsh.model.mesh.field.setNumber(ball, "Radius", float(crossing.radius))
            gmsh.model.mesh.field.setNumber(ball, "XCenter", float(crossing.x))
            gmsh.model.mesh.field.setNumber(ball, "YCenter", float(crossing.y))
            gmsh.model.mesh.field.setNumber(ball, "ZCenter", 0.0)
            gmsh.model.mesh.field.setNumber(ball, "VIn", float(crossing.size))
            gmsh.model.mesh.field.setNumber(ball, "VOut", background_lc)
            gmsh.model.mesh.field.setNumber(ball, "Thickness", 3.0 * float(crossing.size))
            field_ids.append(ball)
        return field_ids

    def _embed_features(self, gmsh_map, polygons_gdf, lines_gdf, points_gdf):
        """Embed points, lines and straddle points into the domain surfaces that contain them.

        Fragmentation splits surfaces, so each entity's owner is found
        geometrically. Candidates are pre-filtered by bounding box, then
        confirmed with gmsh.model.isInside() on the trimmed surfaces. Do not
        use getClosestPoint() here: for coplanar OCC surfaces it can project
        onto the support plane outside the trimmed face, causing false
        multi-surface embeds and over-constraining Gmsh. isInside() is only
        right on faces whose holes are wound correctly (see
        _create_polygon_surface); entities it places in no surface are
        skipped with a warning.
        """
        if self._verbosity > 0:
            logger.info("Explicitly embedding features into domain surfaces...")
        domain_surface_tags = _domain_surface_tags(gmsh_map, polygons_gdf)
        self._log_embed_pool_diagnostics(gmsh_map, polygons_gdf, domain_surface_tags)
        if not domain_surface_tags:
            return

        surface_bboxes = _surface_bboxes(domain_surface_tags)
        stats = _EmbedStats()
        for dim, tag in _entities_to_embed(gmsh_map, points_gdf, lines_gdf):
            self._embed_entity(dim, tag, surface_bboxes, stats)

        if stats.unmatched_tags:
            logger.warning(
                f"Skipped embedding {len(stats.unmatched_tags)} point(s)/line fragment(s) "
                f"that no domain surface contains (dim, tag, bbox centre): "
                f"{stats.unmatched_tags[:5]}. Gmsh still meshes them, but not as part "
                f"of the triangulation, so their nodes can create sliver Voronoi cells. "
                f"This usually follows heal_shapes=True or geometry outside the domain.")
        if stats.nonconforming_skip:
            logger.warning(
                f"Skipped embedding {stats.nonconforming_skip} line fragment(s) whose "
                f"endpoints lie on another surface's boundary but not in the target "
                f"surface's topology (curve, target surface, endpoint surfaces): "
                f"{stats.nonconforming_tags[:5]}. This usually follows heal_shapes=True; "
                f"a smaller heal_tolerance may keep them.")
        self._log_embed_summary_diagnostics(stats)
        self.diagnostics['embedding'] = stats.as_diagnostics(include_records=self.diagnose)

    def _embed_entity(self, dim, tag, surface_bboxes, stats):
        """Embed one point or curve into the smallest domain surface containing it."""
        try:
            bbox = gmsh.model.getBoundingBox(dim, tag)
        except Exception:
            stats.skip_bbox += 1
            return

        # Boundary curves already constrain their adjacent surfaces; embedding
        # them elsewhere duplicates constraints and can make Gmsh
        # non-terminating on dense partitioned geometries.
        if dim == 1:
            adjacent = _curve_adjacent_surfaces(tag)
            if adjacent:
                stats.boundary_skip += 1
                stats.boundary_tags.append((int(tag), adjacent))
                return

        try:
            sample_points = _entity_sample_points(dim, tag, bbox)
        except Exception:
            sample_points = []
        if not sample_points:
            stats.skip_no_match += 1
            stats.unmatched_tags.append(_unmatched_record(dim, tag, bbox))
            return

        candidates = sorted({
            int(surf_tag) for surf_tag, sbb in surface_bboxes.items()
            if any(_bbox_contains_point(sbb, pt) for pt in sample_points)
        })
        if not candidates:
            stats.skip_no_cand += 1
            stats.unmatched_tags.append(_unmatched_record(dim, tag, bbox))
            return

        matches = self._surfaces_containing(tag, sample_points, candidates, stats)
        if not matches:
            stats.skip_no_match += 1
            stats.unmatched_tags.append(_unmatched_record(dim, tag, bbox))
            return
        if len(matches) > 1:
            stats.multi_match += 1
            stats.multi_tags.append((int(tag), list(matches)))
            # Nested or overlapping source polygons can still produce several
            # containing faces. Choose the smallest trimmed surface as the most
            # local owner instead of embedding into every containing face.
            matches = [min(matches, key=_surface_area)]

        # In a conforming model a curve ending on a surface boundary ends at a
        # vertex of that surface. healShapes can break this (a hole edge
        # duplicated rather than shared), leaving the endpoint on the target's
        # boundary but not in its topology; Gmsh then never terminates while
        # recovering the edge.
        if dim == 1:
            foreign = [sorted(owners) for owners in _endpoint_surfaces(tag)
                       if owners and not owners & set(matches)]
            if foreign:
                stats.nonconforming_skip += 1
                stats.nonconforming_tags.append((int(tag), int(matches[0]), foreign))
                return

        for surf_tag in matches:
            try:
                gmsh.model.mesh.embed(dim, [tag], 2, surf_tag)
            except Exception as e:
                stats.failed += 1
                stats.fail_tags.append((tag, str(e)[:60]))
                continue
            stats.ok += 1
            if self.diagnose:
                stats.records.append({
                    'dim': int(dim),
                    'tag': int(tag),
                    'surface': int(surf_tag),
                    'bbox': tuple(float(v) for v in bbox),
                    'sample_points': sample_points,
                })

    @staticmethod
    def _surfaces_containing(tag, sample_points, candidates, stats):
        """Candidate surfaces that contain every sample point (per gmsh isInside)."""
        flat_points = [coord for pt in sample_points for coord in pt]
        matches = []
        for surf_tag in candidates:
            try:
                inside_count = int(gmsh.model.isInside(2, int(surf_tag), flat_points))
            except Exception:
                stats.inside_failed += 1
                stats.inside_fail_tags.append((int(tag), int(surf_tag)))
                continue
            if inside_count == len(sample_points):
                matches.append(surf_tag)
        return matches

    def _log_embed_pool_diagnostics(self, gmsh_map, polygons_gdf, domain_surface_tags):
        """[DIAG] Which surfaces form the embedding pool and which feature owns each."""
        if self._verbosity < 2:
            return
        surfaces = gmsh_map.get('surfaces', {})
        gdf_idxs = list(polygons_gdf.index)
        matching = [i for i in gdf_idxs if i in surfaces]
        logger.debug(f"[DIAG] Embed pool: GDF indices={gdf_idxs}, map keys={list(surfaces)}, "
                     f"matched={len(matching)}, domain_surface_tags={sorted(domain_surface_tags)}")
        # Bounding box of every surface shows which surfaces cover which area.
        for _dim, surf_tag in gmsh.model.getEntities(2):
            in_pool = "POOL" if surf_tag in domain_surface_tags else "----"
            try:
                sbb = gmsh.model.getBoundingBox(2, surf_tag)
                logger.debug(f"[DIAG]   surf {surf_tag:3d} [{in_pool}] "
                             f"x=[{sbb[0]:7.1f},{sbb[3]:7.1f}] "
                             f"y=[{sbb[1]:7.1f},{sbb[4]:7.1f}]")
            except Exception:
                logger.debug(f"[DIAG]   surf {surf_tag:3d} [{in_pool}] bbox FAILED")
        for feat_id, dimtags in surfaces.items():
            surf_tags = [int(dt[1]) for dt in dimtags if isinstance(dt, (tuple, list)) and dt[0] == 2]
            logger.debug(f"[DIAG]   map[surfaces][{feat_id}] -> tags {surf_tags}")

    def _log_embed_summary_diagnostics(self, stats):
        """[DIAG] Embedding outcome counts and the first few problem entities."""
        if self._verbosity < 2:
            return
        logger.debug(f"[DIAG] Embed results: {stats.ok} OK, "
                     f"{stats.failed} failed, "
                     f"{stats.skip_bbox} no-bbox, "
                     f"{stats.skip_no_cand} empty-bbox, "
                     f"{stats.skip_no_match} no-match, "
                     f"{stats.boundary_skip} boundary-skip, "
                     f"{stats.nonconforming_skip} nonconforming-skip, "
                     f"{stats.multi_match} multi-match, "
                     f"{stats.inside_failed} inside-failed")
        if stats.fail_tags:
            logger.debug(f"[DIAG] *** Failed embeds: {stats.fail_tags[:5]} ***")
        if stats.boundary_tags:
            logger.debug(f"[DIAG] Boundary line fragments skipped (first 10): {stats.boundary_tags[:10]}")
        if stats.multi_tags:
            logger.debug(f"[DIAG] Multi-surface embed candidates collapsed "
                         f"(first 10): {stats.multi_tags[:10]}")


    def generate(self, clean_polys, clean_lines, clean_points, output_file=None, launch_gmsh_gui=False):
        """
        Executes the full mesh generation workflow.

        This method orchestrates the entire process:
        1. Initializes Gmsh.
        2. Transfers geometries into the Gmsh model.
        3. Sets up mesh size fields.
        4. Generates the 2D triangular mesh.
        5. Performs optional post-generation optimization.
        6. Extracts the resulting nodes and their tags, plus the per-node
           ``node_is_free`` and ``node_sizes``, the mesh ``node_edges`` and
           the ``buffer_footprints`` that ``VoronoiTessellator(lloyd_iterations=...)`` relies on.

        Args:
            clean_polys (GeoDataFrame): Non-overlapping polygons.
            clean_lines (GeoDataFrame): Snapped and cleaned lines.
            clean_points (GeoDataFrame): Snapped and cleaned points.
            output_file (str, optional): If provided, saves the mesh to this path.
            launch_gmsh_gui (Boolean, optional): This allow to see triangular mesh results
                using the GMSH GUI, and allow to review visually the fields and the triangular
                mesh quality

        Returns:
            bool: True if generation was successful.
        
        Raises:
            Exception: If any step in the Gmsh process fails.
        """
        self._validate_background_lc()
        with verbosity_scope(self.verbosity):
            self._verbosity = self._resolve_verbosity()
            return self._generate(clean_polys, clean_lines, clean_points,
                                  output_file=output_file, launch_gmsh_gui=launch_gmsh_gui)

    def _generate(self, clean_polys, clean_lines, clean_points, output_file=None, launch_gmsh_gui=False):
        """Run the Gmsh workflow behind ``generate()``."""
        self.triangular_quality = None
        self.element_grid = None
        self._element_data = None
        self.node_is_free = None
        self.node_sizes = None
        self.node_edges = None
        self.buffer_footprints = None
        self._initialize_gmsh()
        try:
            logger.info("Transferring Geometry to Gmsh...")
            gmsh_map, crossings, self.buffer_footprints = self._build_occ_model(
                clean_polys, clean_lines, clean_points, launch_gmsh_gui=launch_gmsh_gui
            )

            # Ensure features are correctly embedded in surfaces before meshing
            self._embed_features(gmsh_map, clean_polys, clean_lines, clean_points)
            self._log_post_embed_diagnostics(gmsh_map)

            logger.info("Setting up Resolution Fields...")
            self._setup_fields(gmsh_map, clean_polys, clean_lines, clean_points, crossings=crossings)

            self._mesh_2d()
            self._check_hex_rings(gmsh_map, clean_points)

            meshed_surface_tags = self._meshed_surface_tags(gmsh_map, clean_polys)
            self.triangular_quality = self._collect_triangular_quality(meshed_surface_tags)
            self._element_data = self._capture_element_data(meshed_surface_tags)

            if output_file:
                gmsh.write(output_file)

            self.nodes, self.node_tags = self._collect_domain_nodes(
                gmsh_map, meshed_surface_tags, clean_points, clean_lines
            )
            free_tags = _free_node_tags(gmsh_map, clean_polys)
            self.node_is_free = np.isin(np.asarray(self.node_tags, dtype=np.int64),
                                        np.fromiter(free_tags, dtype=np.int64, count=len(free_tags)))
            self.node_sizes = _node_sizes_from_elements(
                self._element_data, self.node_tags, self.background_lc
            )
            self.node_edges = _node_edges_from_elements(self._element_data, self.node_tags)
            self.zones_gdf = clean_polys
            if launch_gmsh_gui:
                gmsh.fltk.run()
            self._finalize_gmsh()
            return True

        except Exception as e:
            # Broad on purpose: gmsh is process-global state and must be
            # finalized whatever failed, before the error is re-raised.
            logger.error(f"Mesh Generation Failed: {e}")
            self._finalize_gmsh()
            raise

    def _check_hex_rings(self, gmsh_map, points_gdf):
        """Warn for each hex_ring point whose ring the mesher split; record the outcome in diagnostics.

        The blueprint drops rings it can see a conflict for, but explicit
        size fields it cannot model (or a background_lc below the ring
        radius) can still make Gmsh insert nodes inside a ring.
        """
        rings = [(idx, row) for idx, row in points_gdf.iterrows()
                 if is_embedded(row) and ring_seed_coords(row)]
        self.diagnostics['hex_rings'] = {}
        if not rings:
            return
        element_types, _, element_node_tags = gmsh.model.mesh.getElements(2)
        element_nodes = [
            np.asarray(tags, dtype=np.int64).reshape(
                -1, gmsh.model.mesh.getElementProperties(etype)[3]
            )
            for etype, tags in zip(element_types, element_node_tags)
        ]
        for idx, row in rings:
            point_id = row.get('point_id', idx)
            intact = _hex_ring_intact(
                gmsh_map.get('points', {}).get(int(idx), []), row.geometry, element_nodes
            )
            self.diagnostics['hex_rings'][point_id] = intact
            if not intact:
                warnings.warn(
                    f"hex_ring of point {point_id!r} was broken by a size field finer than "
                    f"its ring radius ({row.get('lc')}); its cell will not be a regular "
                    "hexagon.",
                    UserWarning,
                    stacklevel=4,
                )

    def _mesh_2d(self):
        """Set the meshing options, generate the 2D mesh and run the optimization cycles."""
        gmsh.option.setNumber("Mesh.Algorithm", self.mesh_algorithm)
        # Laplacian smoothing passes over the triangle-mesh nodes.
        gmsh.option.setNumber("Mesh.Smoothing", self.smoothing_steps)
        # Tolerance for the initial Delaunay insertion - helps with
        # "Could not insert point" from near-degenerate geometry.
        gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", self.tolerance_initial_delaunay)

        logger.info("Generating Triangular Mesh...")
        gmsh.model.mesh.generate(2)

        if self.optimization_cycles > 0:
            if self._verbosity > 0:
                logger.info(f"Running {self.optimization_cycles} Optimization Cycles (Relocate2D & Laplace2D)...")
            for i in range(self.optimization_cycles):
                if self._verbosity > 1:
                    logger.info(f"  -> Cycle {i+1}/{self.optimization_cycles}")
                # Moves nodes to improve element shape (compactness).
                gmsh.model.mesh.optimize("Relocate2D", niter=1)
                # Laplacian smoothing: moves each free node towards the mean of its neighbours.
                gmsh.model.mesh.optimize("Laplace2D", niter=1)

    @staticmethod
    def _collect_domain_nodes(gmsh_map, domain_surface_tags, clean_points, clean_lines):
        """Unique mesh nodes of the domain surfaces and embedded constraints; returns (nodes_xy, node_tags).

        gmsh.model.mesh.getNodes() without arguments is avoided on purpose: it
        also returns nodes of standalone 1D meshes on curves (e.g. field-only
        rivers) and of non-fragmented 2D surfaces, which would constrain the
        Voronoi tessellation. Without known domain surfaces, all 2D nodes are
        used (still avoiding 1D-only nodes).
        """
        if not domain_surface_tags:
            node_tags, coords, _ = gmsh.model.mesh.getNodes(2, -1, includeBoundary=True)
            nodes_3d = np.array(coords, dtype=float).reshape(-1, 3)
            return nodes_3d[:, :2], node_tags

        tag_to_xy = {}
        for surf_tag in domain_surface_tags:
            _accumulate_nodes(2, surf_tag, tag_to_xy)
        for dim, tag in _embedded_constraint_entities(gmsh_map, clean_points, clean_lines):
            _accumulate_nodes(dim, tag, tag_to_xy)
        node_tags = np.array(list(tag_to_xy.keys()), dtype=np.uint64)
        nodes_xy = np.array([tag_to_xy[int(t)] for t in node_tags], dtype=float)
        return nodes_xy, node_tags
