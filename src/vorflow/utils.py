import numpy as np
import geopandas as gpd
import pandas as pd

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

def calculate_mesh_quality(gdf: gpd.GeoDataFrame, calc_ortho: bool = False) -> gpd.GeoDataFrame:
    """
    Calculates a suite of geometric quality metrics for a Voronoi grid.

    Args:
        gdf (gpd.GeoDataFrame): A GeoDataFrame containing the Voronoi cells.
            It is expected to have 'x' and 'y' columns for the generator points.
        calc_ortho (bool): If True, the orthogonality error will be calculated.
            This is a more expensive calculation and is disabled by default.

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
    
    # 4. Generator-based metrics (require generator point coordinates).
    if 'x' in df.columns and 'y' in df.columns:
        centroids = df.geometry.centroid
        dx = df['x'] - centroids.x
        dy = df['y'] - centroids.y
        # Distance between the cell's generator and its geometric centroid.
        df['centroid_dist'] = np.sqrt(dx*dx + dy*dy)
        # A non-dimensional measure of the generator/centroid drift.
        df['drift_ratio'] = df['centroid_dist'] / np.sqrt(df['area'])
        
        if calc_ortho:
            df['ortho_error'] = calculate_orthogonality(df)
        
    return df

def summarize_quality(gdf: gpd.GeoDataFrame):
    """
    Prints a summary report of mesh quality metrics, separating statistics
    for internal cells versus boundary cells.
    """
    # Ensure the required quality metrics have been calculated.
    if 'compactness' not in gdf.columns:
        gdf = calculate_mesh_quality(gdf, calc_ortho=False)
        
    print("\n--- Mesh Quality Report ---")
    print(f"Total Cells: {len(gdf)}")
    
    # 1. Distinguish between internal and boundary cells.
    # A simple heuristic is that for a boundary cell, its generator point
    # will lie on or very close to the cell's own geometric boundary.
    is_boundary = np.zeros(len(gdf), dtype=bool)
    if 'x' in gdf.columns and 'y' in gdf.columns:
        gens = gpd.GeoSeries(gpd.points_from_xy(gdf.x, gdf.y), index=gdf.index)
        dists = gdf.geometry.boundary.distance(gens)
        
        # Use a small tolerance relative to the cell size.
        tols = np.sqrt(gdf['area']) * 0.01
        is_boundary = dists < tols

    internal_df = gdf[~is_boundary]
    boundary_df = gdf[is_boundary]
    
    print(f"  - Internal Cells: {len(internal_df)}")
    print(f"  - Boundary Cells: {len(boundary_df)}")

    metrics = ['area', 'compactness', 'convexity']
    if 'drift_ratio' in gdf.columns:
        metrics.append('drift_ratio')
    if 'ortho_error' in gdf.columns:
        metrics.append('ortho_error')
        
    print("\n-- Internal Cells Statistics --")
    if not internal_df.empty:
        stats_in = internal_df[metrics].describe(percentiles=[0.05, 0.5, 0.95])
        print(stats_in.T[['min', '5%', '50%', '95%', 'max']].to_string())
    else:
        print("No internal cells.")

    print("\n-- Boundary Cells Statistics --")
    if not boundary_df.empty:
        stats_bnd = boundary_df[metrics].describe(percentiles=[0.05, 0.5, 0.95])
        print(stats_bnd.T[['min', '5%', '50%', '95%', 'max']].to_string())
    else:
        print("No boundary cells.")
    
    # Provide some diagnostic warnings based on common quality thresholds.
    print("\n--- Diagnostics ---")
    
    # Internal cells should be high quality.
    if not internal_df.empty:
        slivers = len(internal_df[internal_df['compactness'] < 0.6])
        if slivers > 0:
            print(f"[WARNING] {slivers} INTERNAL cells have low compactness (< 0.6).")
            
        if 'drift_ratio' in internal_df.columns:
            high_drift = len(internal_df[internal_df['drift_ratio'] > 0.25])
            if high_drift > 0:
                print(f"[WARNING] {high_drift} INTERNAL cells have high drift (> 0.25).")
            else:
                print("[OK] Internal drift is excellent.")

    # Boundary cells have different geometric norms.
    if not boundary_df.empty:
        # Boundary cells are naturally less compact.
        bad_bnd = len(boundary_df[boundary_df['compactness'] < 0.4])
        if bad_bnd > 0:
            print(f"[WARNING] {bad_bnd} BOUNDARY cells are potential slivers (< 0.4).")
            
        # Boundary cells also have a naturally higher drift.
        if 'drift_ratio' in boundary_df.columns:
            high_drift_bnd = len(boundary_df[boundary_df['drift_ratio'] > 0.45])
            if high_drift_bnd > 0:
                print(f"[WARNING] {high_drift_bnd} BOUNDARY cells have excessive drift (> 0.45).")
            else:
                print("[OK] Boundary drift is within geometric norms (~0.34).")