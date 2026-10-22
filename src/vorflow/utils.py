from __future__ import annotations

import logging
import warnings

import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, Point
from .blueprint import DEFAULT_CONNECTIVITY_TOLERANCE


logger = logging.getLogger(__name__)

# Default tolerance of grid_to_disv, as a fraction of the coordinate scale,
# for a vertex of one cell to count as lying on an edge of another.
DISV_VERTEX_FRACTION = 1e-9


CONNECTIVITY_COLUMNS = [
    "cell_id_1",
    "cell_id_2",
    "orig_index_1",
    "orig_index_2",
    "node_id_1",
    "node_id_2",
    "center_mode",
    "center_1",
    "center_2",
    "angle",
    "ortho_error",
    "skewness",
    "connector",
    "shared_edge",
]


def _empty_connectivity(crs=None):
    return gpd.GeoDataFrame(
        columns=CONNECTIVITY_COLUMNS,
        geometry="shared_edge",
        crs=crs,
    )


def _longest_line(geom):
    """Return the longest line component from a shared-boundary geometry."""
    if geom.is_empty:
        return None
    if isinstance(geom, LineString):
        return geom if geom.length > 0 else None
    if isinstance(geom, MultiLineString):
        lines = [line for line in geom.geoms if line.length > 0]
        return max(lines, key=lambda line: line.length) if lines else None
    if hasattr(geom, "geoms"):
        lines = []
        for part in geom.geoms:
            line = _longest_line(part)
            if line is not None:
                lines.append(line)
        return max(lines, key=lambda line: line.length) if lines else None
    return None


def _cross_2d(a, b):
    """Return the z-component of the cross product of (n, 2) vector arrays."""
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _face_skewness(center_1, center_2, face_start, face_end, parallel_tol=1e-12):
    """Return per-face CVFD skewness |X - M| / |face|, NaN where connector and face are parallel."""
    center_1, center_2, face_start, face_end = (
        np.asarray(a, dtype=float).reshape(-1, 2)
        for a in (center_1, center_2, face_start, face_end)
    )
    assert center_1.shape == center_2.shape == face_start.shape == face_end.shape
    connector = center_2 - center_1
    face = face_end - face_start
    midpoint = 0.5 * (face_start + face_end)
    # X = M + u * face lies on the connector line, so |X - M| / |face| = |u|.
    denom = _cross_2d(face, connector)
    numer = _cross_2d(center_1 - midpoint, connector)
    scale = np.hypot(*connector.T) * np.hypot(*face.T)
    valid = np.abs(denom) > parallel_tol * scale
    skewness = np.full(len(face), np.nan)
    skewness[valid] = np.abs(numer[valid] / denom[valid])
    return skewness


def _neighbour_pairs(geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row positions (i, j), i < j, of every pair of geometries that touch."""
    left, right = shapely.STRtree(geoms).query(geoms, predicate="touches")
    keep = left < right
    return left[keep], right[keep]


def _shared_edges(geoms: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Longest shared line of each (left, right) pair; None where they share no line."""
    edges = shapely.intersection(geoms[left], geoms[right])
    is_line = shapely.get_type_id(edges) == shapely.GeometryType.LINESTRING
    edges[~is_line] = [_longest_line(geom) for geom in edges[~is_line]]
    edges[is_line & (shapely.length(edges) <= 0)] = None
    return edges


def _cell_centers(gdf: gpd.GeoDataFrame, center: str) -> np.ndarray:
    """One center point per row: the generator (``x``, ``y``) or the polygon centroid."""
    if center == "generator":
        return shapely.points(gdf["x"].to_numpy(dtype=float), gdf["y"].to_numpy(dtype=float))
    if center == "centroid":
        return shapely.centroid(gdf.geometry.to_numpy())
    raise ValueError("center must be either 'generator' or 'centroid'.")


def _end_points(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, 2) first and last coordinates of each line; NaN rows where the line is None."""
    present = ~shapely.is_missing(lines)
    start = np.full((len(lines), 2), np.nan)
    end = np.full((len(lines), 2), np.nan)
    start[present] = shapely.get_coordinates(shapely.get_point(lines[present], 0))
    end[present] = shapely.get_coordinates(shapely.get_point(lines[present], -1))
    return start, end


def build_connectivity(gdf: gpd.GeoDataFrame, center: str = "generator") -> gpd.GeoDataFrame:
    """
    Build a per-face connectivity report for a polygonal grid.

    ``center="generator"`` uses the ``x`` and ``y`` columns and reports the
    mathematical Voronoi-dual connectivity. ``center="centroid"`` uses polygon
    centroids and reports MODFLOW-facing cell-center connectivity for exported
    cells. ``angle`` is the connector-vs-shared-face angle in degrees, with an
    ideal value of 90. ``ortho_error`` is the corresponding orthogonality error,
    with an ideal value of 0. ``skewness`` is the CVFD face skewness: the
    distance between the face midpoint and the point where the (infinite)
    connector line meets the (infinite) line through the shared face's end
    points, divided by the face length. Its ideal value is 0; values above 0.5
    mean the connector misses the face; it is NaN when connector and face are
    parallel. ``cell_id_1`` and ``cell_id_2`` are zero-based row positions in
    ``gdf`` and are therefore unique even when the GeoDataFrame index is not. ``orig_index_1``
    and ``orig_index_2`` preserve the input index for traceability. The active
    geometry is ``shared_edge``; ``connector`` stores the center-to-center line.
    """
    if center not in {"generator", "centroid"}:
        raise ValueError("center must be either 'generator' or 'centroid'.")

    required = {"geometry"}
    if center == "generator":
        required.update({"x", "y"})
    missing = required.difference(gdf.columns)
    if missing:
        raise ValueError(
            f"build_connectivity missing required columns: {sorted(missing)}; "
            f"required columns for center='{center}': {sorted(required)}"
        )

    if gdf.empty:
        return _empty_connectivity(gdf.crs)

    geoms = gdf.geometry.to_numpy()
    left, right = _neighbour_pairs(geoms)
    edges = _shared_edges(geoms, left, right)
    start, end = _end_points(edges)

    centers = _cell_centers(gdf, center)
    center_ok = shapely.is_geometry(centers) & ~shapely.is_empty(centers)
    center_xy = np.full((len(centers), 2), np.nan)
    center_xy[center_ok] = shapely.get_coordinates(centers[center_ok])
    xy1, xy2 = center_xy[left], center_xy[right]

    gx, gy = (xy2 - xy1).T
    ex, ey = (end - start).T
    gmag = np.sqrt(gx * gx + gy * gy)
    emag = np.sqrt(ex * ex + ey * ey)
    valid = (
        ~shapely.is_missing(edges)
        & center_ok[left]
        & center_ok[right]
        & (gmag != 0)
        & (emag != 0)
    )
    if not valid.any():
        return _empty_connectivity(gdf.crs)

    left, right, edges = left[valid], right[valid], edges[valid]
    start, end, xy1, xy2 = start[valid], end[valid], xy1[valid], xy2[valid]
    gx, gy, ex, ey = gx[valid], gy[valid], ex[valid], ey[valid]
    cos_theta = np.abs(gx * ex + gy * ey) / (gmag[valid] * emag[valid])
    angle = np.degrees(np.arccos(np.clip(cos_theta, 0.0, 1.0)))

    # infer_objects matches the per-row values the old records path produced,
    # e.g. uint64 tags held in an object column come back as uint64.
    orig_index = gdf.index.to_series().infer_objects().to_numpy()
    columns = {
        "cell_id_1": left.astype(np.int64),
        "cell_id_2": right.astype(np.int64),
        "orig_index_1": orig_index[left],
        "orig_index_2": orig_index[right],
        "center_mode": center,
        "center_1": centers[left],
        "center_2": centers[right],
        "angle": angle,
        "ortho_error": np.abs(90.0 - angle),
        "skewness": _face_skewness(xy1, xy2, start, end),
        "connector": shapely.linestrings(np.stack([xy1, xy2], axis=1)),
        "shared_edge": edges,
    }
    if "node_id" in gdf.columns:
        node_id = gdf["node_id"].infer_objects().to_numpy()
        columns["node_id_1"] = node_id[left]
        columns["node_id_2"] = node_id[right]
    return gpd.GeoDataFrame(columns, geometry="shared_edge", crs=gdf.crs)


def boundary_connectivity_report(
    grid_gdf: gpd.GeoDataFrame,
    domain_geom,
    *,
    center: str = "centroid",
    tolerance: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Connectivity report restricted to pairs touching the domain boundary.

    Runs :func:`build_connectivity` on ``grid_gdf`` and keeps only the rows
    where at least one of the two cells touches the boundary of
    ``domain_geom``. Useful to compare the boundary-cell angle/orthogonality
    distribution between ``boundary_centering="clip"`` and ``"inset_mirror"``
    runs of the tessellator. ``center="centroid"`` is the meaningful mode for
    that comparison: with ``center="generator"`` Voronoi faces are exact
    perpendicular bisectors and always report 90 degrees.

    ``tolerance`` defaults to a domain-bbox-scaled value matching the
    tessellator's boundary-node classification.
    """
    if domain_geom is None or domain_geom.is_empty:
        raise ValueError("boundary_connectivity_report requires a non-empty domain geometry.")
    if tolerance is None:
        minx, miny, maxx, maxy = domain_geom.bounds
        tolerance = max(maxx - minx, maxy - miny, 1.0) * 1e-8
    elif tolerance < 0:
        raise ValueError("tolerance must be non-negative when provided.")

    connectivity = build_connectivity(grid_gdf, center=center)
    if connectivity.empty:
        return connectivity

    boundary = domain_geom.boundary
    distances = grid_gdf.geometry.reset_index(drop=True).distance(boundary)
    boundary_cells = set(np.flatnonzero(distances.to_numpy() <= tolerance))

    mask = connectivity["cell_id_1"].isin(boundary_cells) | connectivity["cell_id_2"].isin(
        boundary_cells
    )
    return connectivity[mask].reset_index(drop=True)


def _validate_connectivity_report(
    connectivity: gpd.GeoDataFrame,
    n_cells: int,
    *,
    require_ortho: bool = False,
    require_skewness: bool = False,
) -> None:
    required = {"cell_id_1", "cell_id_2"}
    if require_ortho:
        required.add("ortho_error")
    if require_skewness:
        required.add("skewness")

    missing = required.difference(connectivity.columns)
    if missing:
        raise ValueError(
            f"connectivity missing required columns: {sorted(missing)}; "
            f"required columns for requested metrics: {sorted(required)}"
        )

    if connectivity.empty:
        return

    cell_ids = pd.concat(
        [connectivity["cell_id_1"], connectivity["cell_id_2"]],
        ignore_index=True,
    )
    numeric_ids = pd.to_numeric(cell_ids, errors="coerce")
    if numeric_ids.isna().any():
        raise ValueError("connectivity cell_id_1/cell_id_2 must be integer row positions.")

    int_ids = numeric_ids.astype(np.int64)
    if not np.array_equal(numeric_ids.to_numpy(), int_ids.to_numpy()):
        raise ValueError("connectivity cell_id_1/cell_id_2 must be integer row positions.")

    if (int_ids < 0).any() or (int_ids >= n_cells).any():
        raise ValueError(
            "connectivity cell_id_1/cell_id_2 values must align to zero-based "
            "row positions in the input GeoDataFrame."
        )


def _max_pair_metric_by_cell(connectivity: gpd.GeoDataFrame, metric: pd.Series, n_cells: int) -> np.ndarray:
    values = np.zeros(n_cells, dtype=float)
    if connectivity.empty:
        return values

    s1 = metric.groupby(connectivity["cell_id_1"]).max()
    s2 = metric.groupby(connectivity["cell_id_2"]).max()
    combined = pd.concat([s1, s2], axis=1).max(axis=1)
    for cell_id, value in combined.items():
        values[int(cell_id)] = float(value)
    return values

def calculate_orthogonality(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Calculates the maximum orthogonality error for each cell in a Voronoi grid.

    Orthogonality is a critical measure of grid quality for finite volume methods.
    It is the angle between two vectors:
    1. The vector connecting the generator points of two adjacent cells (G).
    2. The normal vector of their shared edge (N).

    In a perfect Delaunay-Voronoi dual grid, this angle is 0 degrees. This
    function calculates the deviation from this ideal for every internal edge.

    Args:
        gdf (gpd.GeoDataFrame): The Voronoi grid, which must contain 'x' and 'y'
            columns corresponding to the generator point coordinates.

    Returns:
        pd.Series: A series containing the maximum orthogonality error (in degrees)
            for each cell. Cells with no valid neighbors will have an error of 0.
    """
    if 'x' not in gdf.columns or 'y' not in gdf.columns:
        return pd.Series(np.nan, index=gdf.index)

    # Ensure the index is unique for reliable mapping.
    df = gdf.copy()
    if not df.index.is_unique:
        df = df.reset_index(drop=True)
    
    # 1. Identify all neighboring cell pairs using a spatial join.
    # The 'touches' predicate finds all polygons that share a boundary.
    neighbors = gpd.sjoin(df, df, how='inner', predicate='touches')
    
    # Filter out self-matches and duplicates to process each pair only once.
    pairs = neighbors[neighbors.index < neighbors.index_right].copy()
    
    if pairs.empty:
        return pd.Series(0.0, index=gdf.index)

    # 2. Set up vectorized calculations for the generator-to-generator vectors.
    g1_x = df.loc[pairs.index, 'x'].values
    g1_y = df.loc[pairs.index, 'y'].values
    g2_x = df.loc[pairs.index_right, 'x'].values
    g2_y = df.loc[pairs.index_right, 'y'].values
    
    # Vector G (from generator 1 to generator 2).
    Gx = g2_x - g1_x
    Gy = g2_y - g1_y
    
    # 3. Iterate through pairs to compute the normal of the shared edge.
    # This part is not easily vectorized in GeoPandas.
    errors = []
    
    geoms1 = df.loc[pairs.index, 'geometry'].values
    geoms2 = df.loc[pairs.index_right, 'geometry'].values
    
    for i in range(len(pairs)):
        poly1 = geoms1[i]
        poly2 = geoms2[i]
        
        # The intersection of two adjacent polygons is their shared edge.
        inter = poly1.intersection(poly2)
        
        # Skip if the intersection is not a line (e.g., a single point).
        if inter.is_empty or inter.geom_type not in ['LineString', 'MultiLineString']:
            errors.append(np.nan)
            continue
            
        if inter.geom_type == 'MultiLineString':
            # If they touch at multiple places, use the longest shared segment.
            if not inter.geoms:
                errors.append(np.nan)
                continue
            edge = max(inter.geoms, key=lambda x: x.length)
        else:
            edge = inter
            
        # Get the vector for the edge itself.
        coords = list(edge.coords)
        if len(coords) < 2:
            errors.append(np.nan)
            continue
            
        Ex = coords[-1][0] - coords[0][0]
        Ey = coords[-1][1] - coords[0][1]
        
        # The 2D normal vector is (-Ey, Ex).
        Nx, Ny = -Ey, Ex
        
        # Calculate the cosine of the angle between the generator vector (G)
        # and the edge normal vector (N) using the dot product.
        dot = Gx[i]*Nx + Gy[i]*Ny
        mag_g = np.sqrt(Gx[i]**2 + Gy[i]**2)
        mag_n = np.sqrt(Nx**2 + Ny**2)
        
        if mag_g == 0 or mag_n == 0:
            errors.append(np.nan)
            continue
            
        cos_theta = abs(dot) / (mag_g * mag_n)
        
        # Clamp to handle potential floating point inaccuracies.
        cos_theta = min(1.0, max(0.0, cos_theta))
        
        # The orthogonality error is the angle whose cosine we just found.
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        
        errors.append(angle_deg)

    pairs['ortho_error'] = errors
    
    # 4. Aggregate the errors. Each cell's error is the maximum error from
    # all of its edges.
    s1 = pairs['ortho_error'].groupby(pairs.index).max()
    s2 = pairs['ortho_error'].groupby(pairs.index_right).max()
    
    # Combine the errors (since each edge belongs to two cells) and reindex
    # to match the original GeoDataFrame.
    combined = pd.concat([s1, s2], axis=1).max(axis=1)
    final_series = combined.reindex(gdf.index).fillna(0.0)
    
    return final_series

def calculate_mesh_quality(
    gdf: gpd.GeoDataFrame,
    calc_ortho: bool = False,
    calc_skewness: bool = False,
    connectivity: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """
    Calculates a suite of geometric quality metrics for a Voronoi grid.

    Args:
        gdf (gpd.GeoDataFrame): A GeoDataFrame containing the Voronoi cells.
            It is expected to have 'x' and 'y' columns for the generator points.
        calc_ortho (bool): If True, the orthogonality error will be calculated.
            This is a more expensive calculation and is disabled by default.
        calc_skewness (bool): If True, the per-cell skewness (the maximum
            CVFD face skewness over the cell's faces, see
            :func:`build_connectivity`; ideal 0) will be calculated. Faces
            with undefined (NaN) skewness are ignored; a cell with no defined
            face skewness gets NaN. Disabled by default.
        connectivity (GeoDataFrame, optional): A precomputed report from
            ``build_connectivity(gdf, center=...)``. It must include
            ``cell_id_1`` and ``cell_id_2`` as zero-based row positions in
            ``gdf`` plus ``ortho_error`` and/or ``skewness`` for the requested
            metrics. When provided, it is reused for per-cell aggregation.

    Returns:
        gpd.GeoDataFrame: The input GeoDataFrame with added columns for each
            quality metric (e.g., 'area', 'compactness', 'drift_ratio').
    """
    df = gdf.copy()
    
    # 1. Basic geometric properties.
    df['area'] = df.geometry.area
    df['perimeter'] = df.geometry.length
    
    # 2. Compactness (Isoperimetric Quotient): A measure of how "circular" a
    # polygon is. A perfect circle has a compactness of 1.0.
    df['compactness'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
    
    # 3. Convexity (Solidity): The ratio of the cell's area to the area of its
    # convex hull. A perfectly convex polygon has a convexity of 1.0.
    df['convexity'] = df['area'] / df.geometry.convex_hull.area
    
    connectivity_report = None
    if connectivity is not None:
        connectivity_report = connectivity
        _validate_connectivity_report(
            connectivity_report,
            len(df),
            require_ortho=calc_ortho,
            require_skewness=calc_skewness,
        )
    elif calc_skewness and 'x' in df.columns and 'y' in df.columns:
        connectivity_report = build_connectivity(df, center="generator")

    # 4. Generator-based drift metrics (require generator point coordinates).
    if 'x' in df.columns and 'y' in df.columns:
        centroids = df.geometry.centroid
        dx = df['x'] - centroids.x
        dy = df['y'] - centroids.y
        # Distance between the cell's generator and its geometric centroid.
        df['centroid_dist'] = np.sqrt(dx*dx + dy*dy)
        # A non-dimensional measure of the generator/centroid drift.
        df['drift_ratio'] = df['centroid_dist'] / np.sqrt(df['area'])

    if calc_ortho:
        if connectivity_report is not None and 'ortho_error' in connectivity_report.columns:
            df['ortho_error'] = _max_pair_metric_by_cell(
                connectivity_report,
                connectivity_report['ortho_error'],
                len(df),
            )
        elif 'x' in df.columns and 'y' in df.columns:
            df['ortho_error'] = calculate_orthogonality(df)

    if calc_skewness:
        if connectivity_report is None or connectivity_report.empty:
            df['skewness'] = 0.0
        else:
            df['skewness'] = _max_pair_metric_by_cell(
                connectivity_report,
                connectivity_report['skewness'].astype(float),
                len(df),
            )
        
    return df

def summarize_quality(gdf: gpd.GeoDataFrame):
    """
    Prints a summary report of mesh quality metrics, separating statistics
    for internal cells versus boundary cells.
    """
    # Ensure the required quality metrics have been calculated.
    if 'compactness' not in gdf.columns:
        gdf = calculate_mesh_quality(gdf, calc_ortho=False)
        
    logger.info("\n--- Mesh Quality Report ---")
    logger.info(f"Total Cells: {len(gdf)}")
    
    # 1. Distinguish between internal and boundary cells.
    # A simple heuristic is that for a boundary cell, its generator point
    # will lie on or very close to the cell's own geometric boundary.
    is_boundary = np.zeros(len(gdf), dtype=bool)
    if 'x' in gdf.columns and 'y' in gdf.columns:
        gens = gpd.GeoSeries(gpd.points_from_xy(gdf.x, gdf.y), index=gdf.index,crs=gdf.crs)
        dists = gdf.geometry.boundary.distance(gens)
        
        # Use a small tolerance relative to the cell size.
        tols = np.sqrt(gdf['area']) * 0.01
        is_boundary = dists < tols

    internal_df = gdf[~is_boundary]
    boundary_df = gdf[is_boundary]
    
    logger.info(f"  - Internal Cells: {len(internal_df)}")
    logger.info(f"  - Boundary Cells: {len(boundary_df)}")

    metrics = ['area', 'compactness', 'convexity']
    if 'drift_ratio' in gdf.columns:
        metrics.append('drift_ratio')
    if 'ortho_error' in gdf.columns:
        metrics.append('ortho_error')
    if 'skewness' in gdf.columns:
        metrics.append('skewness')
        
    logger.info("\n-- Internal Cells Statistics --")
    if not internal_df.empty:
        stats_in = internal_df[metrics].describe(percentiles=[0.05, 0.5, 0.95])
        logger.info(stats_in.T[['min', '5%', '50%', '95%', 'max']].to_string())
    else:
        logger.info("No internal cells.")

    logger.info("\n-- Boundary Cells Statistics --")
    if not boundary_df.empty:
        stats_bnd = boundary_df[metrics].describe(percentiles=[0.05, 0.5, 0.95])
        logger.info(stats_bnd.T[['min', '5%', '50%', '95%', 'max']].to_string())
    else:
        logger.info("No boundary cells.")
    
    # Provide some diagnostic warnings based on common quality thresholds.
    logger.info("\n--- Diagnostics ---")
    
    # Internal cells should be high quality.
    if not internal_df.empty:
        slivers = len(internal_df[internal_df['compactness'] < 0.6])
        if slivers > 0:
            logger.info(f"[WARNING] {slivers} INTERNAL cells have low compactness (< 0.6).")
            
        if 'drift_ratio' in internal_df.columns:
            high_drift = len(internal_df[internal_df['drift_ratio'] > 0.25])
            if high_drift > 0:
                logger.info(f"[WARNING] {high_drift} INTERNAL cells have high drift (> 0.25).")
            else:
                logger.info("[OK] Internal drift is excellent.")

    # Boundary cells have different geometric norms.
    if not boundary_df.empty:
        # Boundary cells are naturally less compact.
        bad_bnd = len(boundary_df[boundary_df['compactness'] < 0.4])
        if bad_bnd > 0:
            logger.info(f"[WARNING] {bad_bnd} BOUNDARY cells are potential slivers (< 0.4).")
            
        # Boundary cells also have a naturally higher drift.
        if 'drift_ratio' in boundary_df.columns:
            high_drift_bnd = len(boundary_df[boundary_df['drift_ratio'] > 0.45])
            if high_drift_bnd > 0:
                logger.info(f"[WARNING] {high_drift_bnd} BOUNDARY cells have excessive drift (> 0.45).")
            else:
                logger.info("[OK] Boundary drift is within geometric norms (~0.34).")


def check_geometry_resolution(gdf):
    """
    Analyzes the vertex spacing (resolution) of geometries in a GeoDataFrame.
    Returns a summary of min, max, and mean segment lengths.
    """
    all_lengths = []

    for geom in gdf.geometry:
        # Handle different geometry types
        if geom.geom_type == 'Polygon':
            geoms = [geom]
        elif geom.geom_type == 'MultiPolygon':
            geoms = geom.geoms
        elif geom.geom_type == 'LineString':
            geoms = [geom]
        else:
            continue

        for g in geoms:
            # For polygons, check exterior and interiors
            if g.geom_type == 'Polygon':
                coords = list(g.exterior.coords)
                for interior in g.interiors:
                    coords.extend(list(interior.coords))
            else:
                coords = list(g.coords)

            # Calculate distances between consecutive points
            if len(coords) > 1:
                points = np.array(coords)
                # Vectorized distance calculation
                diffs = np.diff(points, axis=0)
                dists = np.sqrt((diffs**2).sum(axis=1))
                all_lengths.extend(dists)

    if not all_lengths:
        warnings.warn(
            "check_geometry_resolution: no valid segments found in the "
            "GeoDataFrame; returning empty statistics.",
            stacklevel=2,
        )
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "count": 0,
        }

    all_lengths = np.array(all_lengths)
    return {
        "min": all_lengths.min(),
        "max": all_lengths.max(),
        "mean": all_lengths.mean(),
        "median": np.median(all_lengths),
        "count": len(all_lengths)
    }


def resample_geometry(geom, target_spacing):
    """
    Resamples a geometry so that vertices are evenly spaced at `target_spacing`.
    This is useful for creating uniform boundaries for meshing.
    """
    if geom.is_empty:
        return geom

    if geom.geom_type == 'LineString':
        length = geom.length
        # Calculate number of segments needed
        num_segments = max(int(np.ceil(length / target_spacing)), 1)
        # Generate distances along the line
        distances = np.linspace(0, length, num_segments + 1)
        # Interpolate points at these distances
        points = [geom.interpolate(d).coords[0] for d in distances]
        return LineString(points)
    
    elif geom.geom_type == 'Polygon':
        # Resample exterior ring
        ext_len = geom.exterior.length
        num_ext = max(int(np.ceil(ext_len / target_spacing)), 3) # Min 3 pts for polygon
        ext_dists = np.linspace(0, ext_len, num_ext + 1)
        # Note: interpolate(0) and interpolate(length) are the same for rings
        ext_points = [geom.exterior.interpolate(d).coords[0] for d in ext_dists]
        
        # Resample interior rings (holes)
        interiors = []
        for interior in geom.interiors:
            int_len = interior.length
            num_int = max(int(np.ceil(int_len / target_spacing)), 3)
            int_dists = np.linspace(0, int_len, num_int + 1)
            int_points = [interior.interpolate(d).coords[0] for d in int_dists]
            interiors.append(int_points)
            
        return Polygon(ext_points, interiors)
        
    elif geom.geom_type == 'MultiPolygon':
        parts = [resample_geometry(p, target_spacing) for p in geom.geoms]
        return MultiPolygon(parts)
        
    return geom


def grid_to_disv(gdf: gpd.GeoDataFrame, tolerance: float | None = None) -> dict:
    """
    Build MODFLOW 6 DISV grid arrays from a polygonal grid.

    Returns a dict with ``ncpl``, ``nvert``, ``vertices`` (``[iv, x, y]``)
    and ``cell2d`` (``[icell2d, xc, yc, ncvert, iv1, iv2, ...]``), in the
    zero-based form flopy's ``ModflowGwfdisv`` takes, plus ``node_id``: the
    grid's ``node_id`` per ``icell2d`` (row order), to map results back.

    Cell centres ``xc``/``yc`` are the ``x``/``y`` columns, i.e. the
    generator points on a vorflow grid, which the CVFD connection geometry
    assumes. Each cell lists its vertices clockwise without repeating the
    first. Neighbouring cells share vertex indices; a vertex of one cell
    that lies on another cell's edge (a hanging node, e.g. where a barrier
    split a cell) is inserted into that edge too, because MODFLOW 6 finds
    connections from shared vertex pairs.

    Args:
        gdf (GeoDataFrame): Grid of single-part polygons without holes, with
            ``x`` and ``y`` columns (e.g. ``VoronoiTessellator.grid``).
        tolerance (float, optional): Distance within which a vertex counts as
            lying on an edge. Defaults to ``DISV_VERTEX_FRACTION`` times the
            coordinate scale.

    Returns:
        dict: ``ncpl``, ``nvert``, ``vertices``, ``cell2d`` and ``node_id``.

    Raises:
        ValueError: If columns are missing, the grid is empty, or a cell is
            not a single polygon without holes.
    """
    missing = {"geometry", "x", "y"}.difference(gdf.columns)
    if missing:
        raise ValueError(f"grid_to_disv missing required columns: {sorted(missing)}")
    if gdf.empty:
        raise ValueError("grid_to_disv needs a non-empty grid.")
    geoms = gdf.geometry.to_numpy()
    bad = np.flatnonzero(
        (shapely.get_type_id(geoms) != shapely.GeometryType.POLYGON)
        | (shapely.get_num_interior_rings(geoms) > 0)
    )
    if len(bad):
        raise ValueError(
            f"DISV cells must be single polygons without holes; {len(bad)} are not "
            f"(first at row {bad[0]})."
        )

    exteriors = shapely.get_exterior_ring(geoms)
    vertex_xy, cell_vertices = _ring_vertex_ids(exteriors)

    if tolerance is None:
        scale = max(float(np.ptp(vertex_xy, axis=0).max()), float(np.abs(vertex_xy).max()), 1.0)
        tolerance = DISV_VERTEX_FRACTION * scale
    cell_vertices = _insert_hanging_vertices(cell_vertices, vertex_xy, tolerance)
    # Hanging vertices are collinear with their edge, so each ring keeps its
    # orientation; MODFLOW 6 expects clockwise (negative shoelace area).
    counter_clockwise = shapely.is_ccw(exteriors)
    cell_vertices = [ids[::-1] if ccw else ids for ids, ccw in zip(cell_vertices, counter_clockwise)]

    # Drop vertices no cell uses (none today; kept for robustness) and renumber.
    used = np.unique(np.concatenate([np.asarray(ids) for ids in cell_vertices]))
    renumber = np.full(len(vertex_xy), -1, dtype=np.int64)
    renumber[used] = np.arange(len(used))
    vertices = [[i, x, y] for i, (x, y) in enumerate(vertex_xy[used].tolist())]

    xc = gdf["x"].to_numpy(dtype=float).tolist()
    yc = gdf["y"].to_numpy(dtype=float).tolist()
    cell2d = [
        [icell, xc[icell], yc[icell], len(ids), *renumber[ids].tolist()]
        for icell, ids in enumerate(cell_vertices)
    ]
    node_id = gdf["node_id"].to_numpy() if "node_id" in gdf.columns else np.arange(len(gdf))
    return {
        "ncpl": len(cell2d),
        "nvert": len(vertices),
        "vertices": vertices,
        "cell2d": cell2d,
        "node_id": node_id,
    }


def _ring_vertex_ids(rings: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Unique (m, 2) vertex coordinates and, per ring, its vertex ids without the closing repeat."""
    coords, ring_of = shapely.get_coordinates(rings, return_index=True)
    sizes = np.bincount(ring_of, minlength=len(rings))
    is_open = np.ones(len(coords), dtype=bool)
    is_open[np.cumsum(sizes) - 1] = False
    vertex_xy, vertex_of = np.unique(coords[is_open], axis=0, return_inverse=True)
    splits = np.cumsum(sizes - 1)[:-1]
    return vertex_xy, [ids.tolist() for ids in np.split(vertex_of.ravel(), splits)]


def _insert_hanging_vertices(cell_vertices, vertex_xy, tolerance):
    """Cell rings with every grid vertex that lies inside one of their edges inserted, in order along the edge."""
    edge_cell = np.repeat(np.arange(len(cell_vertices)), [len(ids) for ids in cell_vertices])
    edge_pos = np.concatenate([np.arange(len(ids)) for ids in cell_vertices])
    start_ids = np.concatenate([np.asarray(ids) for ids in cell_vertices])
    end_ids = np.concatenate([np.roll(np.asarray(ids), -1) for ids in cell_vertices])
    starts, ends = vertex_xy[start_ids], vertex_xy[end_ids]

    tree = shapely.STRtree(shapely.points(vertex_xy))
    edge_idx, vertex_idx = tree.query(
        shapely.linestrings(np.stack([starts, ends], axis=1)), predicate="dwithin", distance=tolerance
    )
    keep = (vertex_idx != start_ids[edge_idx]) & (vertex_idx != end_ids[edge_idx])
    edge_idx, vertex_idx = edge_idx[keep], vertex_idx[keep]
    if edge_idx.size == 0:
        return cell_vertices

    edge = ends[edge_idx] - starts[edge_idx]
    rel = vertex_xy[vertex_idx] - starts[edge_idx]
    t = np.einsum("ij,ij->i", rel, edge) / np.einsum("ij,ij->i", edge, edge)
    inside = (t > 0.0) & (t < 1.0)
    hanging = {}
    for e, v, ti in zip(edge_idx[inside], vertex_idx[inside], t[inside]):
        hanging.setdefault(int(e), []).append((ti, int(v)))

    out = [list(ids) for ids in cell_vertices]
    # Insert from the last edge of each ring backwards so earlier positions stay valid.
    for e in sorted(hanging, reverse=True):
        inserted = [v for _, v in sorted(hanging[e])]
        ring = out[edge_cell[e]]
        pos = edge_pos[e] + 1
        ring[pos:pos] = inserted
    return out


def cells_on(gdf, geometry):
    """Return cells intersecting a point or line with area and line-length metrics.

    ``cell_area`` is each full cell's area. For line inputs, ``length`` is the
    portion of the input line inside each cell; for point inputs it is zero.
    """
    if not isinstance(geometry, (Point, LineString, MultiLineString)):
        raise TypeError("geometry must be a Point, LineString, or MultiLineString.")

    cells = gdf.loc[gdf.geometry.intersects(geometry)].copy()
    cells["cell_area"] = cells.geometry.area
    if isinstance(geometry, (LineString, MultiLineString)):
        cells["length"] = cells.geometry.intersection(geometry).length
    else:
        cells["length"] = 0.0
    cells["cell_id"] = cells.index
    return cells


def faces_on(gdf, geometry, tolerance=None):
    """Return rows from a connectivity GeoDataFrame whose faces are near a line.

    Pass the result of :func:`build_connectivity` as ``gdf``. ``tolerance``
    buffers the line for selecting candidate shared edges; ``face_length`` is
    the length of each full shared edge.
    """
    if tolerance is None:
        tolerance = DEFAULT_CONNECTIVITY_TOLERANCE
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    if not isinstance(geometry, (LineString, MultiLineString)):
        raise TypeError("geometry must be a LineString or MultiLineString.")

    required = {"cell_id_1", "cell_id_2", "shared_edge"}
    missing = required.difference(gdf.columns)
    if missing:
        raise ValueError(
            f"faces_on requires a connectivity GeoDataFrame with columns: {sorted(required)}; "
            f"missing {sorted(missing)}."
        )

    faces = gdf.loc[gdf.geometry.within(geometry.buffer(tolerance))].copy()
    faces["face_length"] = faces.geometry.length
    return faces