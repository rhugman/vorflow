"""Barrier policies and optional geometry-to-cell tracking, without solver imports."""

from __future__ import annotations

import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import LineString, Point

from ._features import row_bool
from ._geometry import straight_face


POLICIES = {'auto', 'preserve', 'strict', 'cut'}
MERGE_POLICIES = {'none', 'checked', 'restore', 'legacy'}


def barrier_policy(row):
    """Resolve old/missing feature columns as well as newly authored features."""
    policy = row.get('barrier_policy', 'auto')
    if policy is None or (pd.api.types.is_scalar(policy) and pd.isna(policy)):
        policy = 'auto'
    if not isinstance(policy, str) or policy not in POLICIES:
        raise ValueError("barrier_policy must be 'auto', 'preserve', 'strict' or 'cut'.")
    if policy == 'auto':
        return 'cut'
    return policy


def barrier_merge(row):
    mode = row.get('barrier_merge', 'legacy')
    if mode is None or (pd.api.types.is_scalar(mode) and pd.isna(mode)):
        mode = 'legacy'
    if not isinstance(mode, str) or mode not in MERGE_POLICIES:
        raise ValueError("barrier_merge must be 'none', 'checked', 'restore' or 'legacy'.")
    return mode


def barrier_mirrors(row):
    """Resolve an explicit mirror switch, retaining the historical cut default."""
    value = row.get('barrier_mirrors')
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return barrier_policy(row) == 'cut'
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('barrier_mirrors must be a boolean or None.')
    return bool(value)


def barrier_rows(lines):
    if lines is None or lines.empty or 'is_barrier' not in lines:
        return []
    return [(position, row) for position, (_, row) in enumerate(lines.iterrows())
            if row_bool(row, 'is_barrier')]


def line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'LineString':
        return [geometry]
    return [p for g in getattr(geometry, 'geoms', []) for p in line_parts(g)]


def tracking_tolerance(grid, lines, tolerance):
    if tolerance is not None:
        if isinstance(tolerance, (bool, np.bool_)) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('tolerance must be a finite non-negative distance.')
        return float(tolerance)
    arrays = [shapely.get_coordinates(frame.geometry) for frame in (grid, lines)
              if frame is not None and not frame.empty]
    if not arrays:
        return 0.0
    xy = np.vstack(arrays)
    scale = max(1.0, float(np.abs(xy).max()))
    span = max(1.0, float(np.ptp(xy, axis=0).max()))
    return max(32 * np.finfo(float).eps * scale, 1e-9 * span)


def crossing_records(grid, lines, tolerance):
    """Positive-length barrier sections in cell interiors, including internal ends."""
    records = []
    if grid.empty:
        return records
    for index, row in barrier_rows(lines):
        line = row.geometry
        for position in grid.sindex.query(line, predicate='intersects'):
            polygon = grid.geometry.iloc[position]
            boundary = polygon.boundary
            band = boundary.buffer(tolerance) if tolerance else boundary
            interior = line.intersection(polygon).difference(band)
            parts = line_parts(interior)
            if not parts or sum(p.length for p in parts) <= tolerance:
                continue
            records.append(dict(barrier_index=index, line_id=row.get('line_id', index),
                                cell_id=int(position), orig_index=grid.index[position],
                                node_id=grid['node_id'].iloc[position] if 'node_id' in grid else None,
                                length=sum(p.length for p in parts),
                                geometry=shapely.union_all(parts)))
    return records


def _frame(records, columns, crs):
    return gpd.GeoDataFrame(records, columns=[*columns, 'geometry'], geometry='geometry', crs=crs)


def _intervals(polygon, start, direction, normal, length, tolerance):
    """Intervals where a polygon boundary lies along a directed barrier segment."""
    intervals = []
    for ring in (polygon.exterior, *polygon.interiors):
        xy = np.asarray(ring.coords)[:, :2] - start
        distance = xy @ normal
        projection = xy @ direction
        for i in range(len(xy) - 1):
            if max(abs(distance[i]), abs(distance[i + 1])) > tolerance:
                continue
            a, b = sorted(projection[i:i + 2])
            a, b = max(0.0, float(a)), min(length, float(b))
            if b - a > tolerance:
                intervals.append((a, b))
    return intervals


def _local_side(polygon, midpoint, normal, tolerance, length):
    # Use polygon interiors rather than generator signs: a cut can leave the
    # original generator on the new face, and a concave cell's centroid can
    # misrepresent the side of an individual segment.
    epsilon = max(tolerance * 4, length * 1e-7)
    for factor in (1.0, .1, 10.0):
        left = polygon.covers(Point(midpoint + normal * epsilon * factor))
        right = polygon.covers(Point(midpoint - normal * epsilon * factor))
        if left != right:
            return 1 if left else -1
    return 0


def grid_barrier_report(grid, lines, tolerance=None, approximate=False, closed_ties='exclude'):
    """Prepare segment-wise two-sided pairs plus explicit unmatched/crossing data."""
    if not isinstance(lines, gpd.GeoDataFrame):
        raise ValueError('barriers must be a GeoDataFrame with line geometry and is_barrier flags.')
    if not isinstance(closed_ties, str) or closed_ties not in {'exclude', 'majority'}:
        raise ValueError("closed_ties must be 'exclude' or 'majority'.")
    required = {'x', 'y'}
    if not required <= set(grid.columns):
        raise ValueError("Barrier tracking requires finite generator columns 'x' and 'y'.")
    if not np.isfinite(grid[['x', 'y']].to_numpy(dtype=float)).all():
        raise ValueError("Barrier tracking requires finite generator columns 'x' and 'y'.")
    if any(g is None or g.is_empty or g.geom_type != 'Polygon' or not g.is_valid
           for g in grid.geometry):
        raise ValueError('Barrier tracking requires valid, non-empty Polygon cells.')
    if grid.crs is not None and lines.crs is not None and grid.crs != lines.crs:
        raise ValueError('Grid and barrier CRS must match; reproject the barriers first.')
    rows = barrier_rows(lines)
    if any(row.geometry is None or row.geometry.is_empty or not row.geometry.is_valid
           or row.geometry.geom_type not in {'LineString', 'MultiLineString'} for _, row in rows):
        raise ValueError('Barriers must have valid, non-empty line geometry.')
    tolerance = tracking_tolerance(grid, lines, tolerance)
    crossings = crossing_records(grid, lines, tolerance)
    faces, unmatched, summary = [], [], []
    tree = shapely.STRtree(grid.geometry.to_numpy())
    centres = grid[['x', 'y']].to_numpy(dtype=float)
    for index, row in rows:
        line_id = row.get('line_id', index)
        covered, missed = [], []
        for part_id, part in enumerate(line_parts(row.geometry)):
            xy = np.asarray(part.coords)[:, :2]
            for segment_id, (start, end) in enumerate(zip(xy[:-1], xy[1:])):
                length = float(np.linalg.norm(end - start))
                if length <= tolerance:
                    continue
                direction = (end - start) / length
                normal = np.array([-direction[1], direction[0]])
                segment = LineString([start, end])
                positions = tree.query(segment, predicate='dwithin', distance=tolerance)
                intervals = [(int(p), a, b) for p in sorted(positions)
                             for a, b in _intervals(grid.geometry.iloc[p], start, direction,
                                                   normal, length, tolerance)]
                stops = sorted({0.0, length, *(a for _, a, _ in intervals),
                                *(b for _, _, b in intervals)})
                for a, b in zip(stops[:-1], stops[1:]):
                    if b - a <= tolerance:
                        continue
                    midpoint = start + ((a + b) / 2) * direction
                    candidates = {p for p, lo, hi in intervals if lo <= (a + b) / 2 <= hi}
                    sides = {p: _local_side(grid.geometry.iloc[p], midpoint, normal,
                                           tolerance, b - a) for p in candidates}
                    left = [p for p, side in sides.items() if side == 1]
                    right = [p for p, side in sides.items() if side == -1]
                    geometry = LineString([start + a * direction, start + b * direction])
                    common = dict(barrier_index=index, line_id=line_id, part_id=part_id,
                                  segment_id=segment_id, length=b - a, geometry=geometry)
                    if len(left) != 1 or len(right) != 1 or any(side == 0 for side in sides.values()):
                        unmatched.append(dict(**common, reason='no_unique_two_sided_face'))
                        missed.append(geometry)
                        continue
                    left_cell, right_cell = left[0], right[0]
                    signed = (centres[[left_cell, right_cell]] - midpoint) @ normal
                    faces.append(dict(**common, left_cell=left_cell, right_cell=right_cell,
                                      left_index=grid.index[left_cell], right_index=grid.index[right_cell],
                                      left_node_id=grid['node_id'].iloc[left_cell] if 'node_id' in grid else None,
                                      right_node_id=grid['node_id'].iloc[right_cell] if 'node_id' in grid else None,
                                      centres_opposite=bool(signed[0] > tolerance and signed[1] < -tolerance)))
                    covered.append(geometry)
        policy = barrier_policy(row)
        summary.append(dict(barrier_index=index, line_id=line_id, policy=policy,
                            merge_mode=barrier_merge(row) if policy == 'cut' else 'not_used',
                            length=row.geometry.length,
                            matched_length=shapely.union_all(covered).length if covered else 0.0,
                            unmatched_length=shapely.union_all(missed).length if missed else 0.0,
                            crossing_cells=sum(r['barrier_index'] == index for r in crossings),
                            geometry=row.geometry))
    common = ['barrier_index', 'line_id', 'part_id', 'segment_id', 'length']
    result = dict(
        faces=_frame(faces, common + ['left_cell', 'right_cell', 'left_index', 'right_index',
                                    'left_node_id', 'right_node_id', 'centres_opposite'], grid.crs),
        crossings=_frame(crossings, ['barrier_index', 'line_id', 'cell_id', 'orig_index',
                                     'node_id', 'length'], grid.crs),
        unmatched=_frame(unmatched, common + ['reason'], grid.crs),
        summary=_frame(summary, ['barrier_index', 'line_id', 'policy', 'merge_mode', 'length',
                                 'matched_length', 'unmatched_length', 'crossing_cells'], grid.crs),
        tolerance=tolerance,
    )
    if approximate:
        result.update(_adjacent_barrier_faces(grid, lines, tolerance, closed_ties))
    return result


def _route_topology(records, tolerance):
    """Describe a route without mistaking an invalid connection for closure."""
    from collections import defaultdict

    edges = defaultdict(set)
    quantum = max(tolerance, np.finfo(float).eps)
    for record in records:
        for part in line_parts(record['geometry']):
            xy = np.asarray(part.coords)
            for s, e in zip(xy[:-1], xy[1:]):
                p, q = tuple(np.rint(s / quantum)), tuple(np.rint(e / quantum))
                if p != q:
                    edges[p].add(q)
                    edges[q].add(p)
    unseen, components = set(edges), 0
    while unseen:
        components += 1
        pending = [unseen.pop()]
        while pending:
            for other in edges[pending.pop()] & unseen:
                unseen.remove(other)
                pending.append(other)
    ends = sum(len(n) == 1 for n in edges.values())
    branches = sum(len(n) > 2 for n in edges.values())
    return dict(components=components,
                single_cycle=bool(edges) and components == 1 and
                all(len(n) == 2 for n in edges.values()),
                single_path=components == 1 and ends == 2 and not branches,
                end_vertices=ends, branch_vertices=branches)


def _adjacent_barrier_faces(grid, lines, tolerance, closed_ties='exclude'):
    """Offer real adjacent faces crossed by centre connections, never fake pairs.

    Closed rings use inside/outside centre labels, giving a cell-partition
    approximation. Open lines use single proper connector crossings, including
    bends when both incident segments agree on the sides. Tangential touches,
    multiple crossings and centre ties are excluded; inspect routes before HFB use.
    """
    from shapely.geometry import Polygon

    geoms = grid.geometry.to_numpy()
    centres = grid[['x', 'y']].to_numpy(float)
    tree = shapely.STRtree(geoms)
    a, b = tree.query(geoms, predicate='dwithin', distance=tolerance)
    adjacent = []
    for i, j in zip(a[a < b], b[a < b]):
        shared = shapely.intersection(geoms[i].boundary, geoms[j].boundary, grid_size=tolerance)
        if shared.length > tolerance:
            adjacent.append((int(i), int(j), shared))
    records, summaries, tie_records, excluded = [], [], [], []
    for index, row in barrier_rows(lines):
        for part_id, target in enumerate(line_parts(row.geometry)):
            closed = target.is_ring
            region = Polygon(target) if closed else None
            xy = np.asarray(target.coords)[:, :2]
            segments = [LineString([s, e]) for s, e in zip(xy[:-1], xy[1:])]
            tangents = np.diff(xy, axis=0)
            lengths = np.linalg.norm(tangents, axis=1)
            ties = shapely.distance(shapely.points(centres), target) <= tolerance
            labels = shapely.contains_xy(region, centres[:, 0], centres[:, 1]) if closed else None
            resolve = closed and closed_ties == 'majority'
            balanced = 0
            for cell in np.flatnonzero(ties):
                fraction, inside = np.nan, None
                reason = 'centre_on_target'
                if closed:
                    overlap = geoms[cell].intersection(region).area
                    fraction = overlap / geoms[cell].area
                    # Boundary roundoff can perturb a perfectly balanced cell.
                    # Use a boundary-band area bound and a declared outside
                    # convention for these ties, independent of row ordering.
                    area_band = tolerance * geoms[cell].length
                    is_balanced = abs(overlap - geoms[cell].area / 2) <= area_band
                    inside = bool(overlap > geoms[cell].area / 2 + area_band)
                    if resolve:
                        labels[cell] = inside
                        balanced += int(is_balanced)
                        reason = 'balanced_outside' if is_balanced else 'majority_area'
                tie_records.append(dict(barrier_index=index, line_id=row.get('line_id', index),
                                        part_id=part_id, cell_id=int(cell),
                                        node_id=grid.node_id.iloc[cell] if 'node_id' in grid else None,
                                        inside_fraction=fraction,
                                        assigned_inside=inside if resolve else None,
                                        resolved=resolve, reason=reason, geometry=geoms[cell]))
            selected, ambiguous = [], 0
            for i, j, shared in adjacent:
                if not resolve and (ties[i] or ties[j]):
                    continue
                if closed:
                    if labels[i] == labels[j]:
                        continue
                    left_inside = region.exterior.is_ccw
                    left, right = (i, j) if labels[i] == left_inside else (j, i)
                else:
                    connector = LineString(centres[[i, j]])
                    if not connector.crosses(target):
                        continue
                    hit = connector.intersection(target)
                    if hit.geom_type != 'Point':
                        ambiguous += 1
                        excluded.append(dict(barrier_index=index, line_id=row.get('line_id', index),
                                             part_id=part_id, cell_a=i, cell_b=j,
                                             reason='multiple_or_overlapping_intersections',
                                             geometry=connector))
                        continue
                    near = [k for k, segment in enumerate(segments)
                            if lengths[k] > tolerance and segment.distance(hit) <= tolerance]
                    hit_xy = np.asarray(hit.coords[0])
                    # Several nearby segments count as a bend only when they
                    # meet at this vertex in source order. A self-intersection
                    # can also produce one Point, but has no unique local side.
                    bend = (len(near) == 2 and
                            (lengths[near[0]+1:near[1]] <= tolerance).all() and
                            np.linalg.norm(xy[near[0]+1] - hit_xy) <= tolerance and
                            np.linalg.norm(xy[near[1]] - hit_xy) <= tolerance)
                    if not near or (len(near) > 1 and not bend):
                        ambiguous += 1
                        excluded.append(dict(barrier_index=index, line_id=row.get('line_id', index),
                                             part_id=part_id, cell_a=i, cell_b=j,
                                             reason='nonlocal_or_ambiguous_vertex', geometry=connector))
                        continue
                    unit = tangents[near] / lengths[near, None]
                    normals = np.column_stack((-unit[:, 1], unit[:, 0]))
                    signed = (centres[[i, j]] - hit_xy) @ normals.T
                    # At a proper bend crossing, each endpoint is on the same
                    # directed side of both incident segments. A tangent touch
                    # instead gives conflicting signs, even when GEOS calls
                    # the line/line point intersection a crossing.
                    left_sides = (signed > tolerance).all(axis=1)
                    right_sides = (signed < -tolerance).all(axis=1)
                    if left_sides[0] and right_sides[1]:
                        left, right = i, j
                    elif left_sides[1] and right_sides[0]:
                        left, right = j, i
                    else:
                        ambiguous += 1
                        excluded.append(dict(barrier_index=index, line_id=row.get('line_id', index),
                                             part_id=part_id, cell_a=i, cell_b=j,
                                             reason='tangent_touch_or_uncertain_side', geometry=connector))
                        continue
                reason = None
                try:
                    face = straight_face(shared, centres[[left, right]], (left, right), tolerance)
                    if face is None or min(face['distance_1'], face['distance_2']) <= tolerance:
                        reason = 'nonpositive_distance'
                except ValueError as error:
                    reason = str(error)
                # Vertex-only distances miss a face spanning a concave target.
                # Distance to a set is 1-Lipschitz: dense samples plus half the
                # sample spacing provide a conservative directed offset bound.
                step = shared.length / 128
                sampled = shapely.segmentize(shared, step)
                offset = float(shapely.distance(shapely.points(shapely.get_coordinates(sampled)), target).max())
                record = dict(barrier_index=index, line_id=row.get('line_id', index), part_id=part_id,
                              left_cell=left, right_cell=right,
                              left_node_id=grid.node_id.iloc[left] if 'node_id' in grid else None,
                              right_node_id=grid.node_id.iloc[right] if 'node_id' in grid else None,
                              length=shared.length, max_offset=offset,
                              offset_upper_bound=offset + step/2, usable=reason is None,
                              reason=reason, geometry=shared)
                records.append(record)
                selected.append(record)
            topology = _route_topology(selected, tolerance)
            usable_topology = _route_topology([r for r in selected if r['usable']], tolerance)
            target_offset, target_bound = np.nan, np.nan
            if selected:
                route = shapely.union_all([r['geometry'] for r in selected])
                spacing = target.length / 512
                samples = shapely.segmentize(target, spacing)
                target_offset = float(shapely.distance(
                    shapely.points(shapely.get_coordinates(samples)), route).max())
                target_bound = target_offset + spacing / 2
            summaries.append(dict(barrier_index=index, line_id=row.get('line_id', index), part_id=part_id,
                                  closed=closed, pairs=len(selected), **topology,
                                  **{'usable_' + k: v for k, v in usable_topology.items()},
                                  boundary_centre_ties=int(ties.sum()), ambiguous_connectors=ambiguous,
                                  resolved_centre_ties=int(ties.sum()) if resolve else 0,
                                  unresolved_centre_ties=0 if resolve else int(ties.sum()),
                                  balanced_centre_ties=balanced, closed_ties=closed_ties,
                                  unusable_pairs=sum(not r['usable'] for r in selected),
                                  max_offset=max((r['max_offset'] for r in selected), default=np.nan),
                                  offset_upper_bound=max((r['offset_upper_bound'] for r in selected), default=np.nan),
                                  target_max_offset=target_offset, target_offset_upper_bound=target_bound))
    return dict(
        approximate_faces=_frame(records, ['barrier_index', 'line_id', 'part_id', 'left_cell',
                                            'right_cell', 'left_node_id', 'right_node_id', 'length',
                                            'max_offset', 'offset_upper_bound', 'usable', 'reason'], grid.crs),
        approximate_summary=pd.DataFrame(summaries),
        approximate_ties=_frame(tie_records, ['barrier_index', 'line_id', 'part_id', 'cell_id',
                                             'node_id', 'inside_fraction', 'assigned_inside',
                                             'resolved', 'reason'], grid.crs),
        approximate_excluded=_frame(excluded, ['barrier_index', 'line_id', 'part_id',
                                               'cell_a', 'cell_b', 'reason'], grid.crs),
    )
