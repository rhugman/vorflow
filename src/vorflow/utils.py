import numpy as np
import geopandas as gpd
import pandas as pd

def calculate_orthogonality(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Calculates the maximum orthogonality error (in degrees) for each cell's internal faces.
    
    Orthogonality is defined as the angle between the line connecting two generators
    and the normal of their shared face. In a perfect Voronoi grid, this is 0.
    
    Returns:
        pd.Series: Max error in degrees for each cell. 
                   Returns 0.0 for cells with no neighbors or perfect orthogonality.
    """
    if 'x' not in gdf.columns or 'y' not in gdf.columns:
        return pd.Series(np.nan, index=gdf.index)

    # Ensure unique index for mapping
    df = gdf.copy()
    if not df.index.is_unique:
        df = df.reset_index(drop=True)
    
    # 1. Identify Neighbors
    # predicate='touches' finds all adjacent polygons
    # We use inner join to get pairs
    neighbors = gpd.sjoin(df, df, how='inner', predicate='touches')
    
    # Filter out self-loops and ensure we only process each pair once (i < j)
    pairs = neighbors[neighbors.index < neighbors.index_right].copy()
    
    if pairs.empty:
        return pd.Series(0.0, index=gdf.index)

    # 2. Vectorized Calculation setup
    # Get Generators
    g1_x = df.loc[pairs.index, 'x'].values
    g1_y = df.loc[pairs.index, 'y'].values
    g2_x = df.loc[pairs.index_right, 'x'].values
    g2_y = df.loc[pairs.index_right, 'y'].values
    
    # Vector G (Generator to Generator)
    Gx = g2_x - g1_x
    Gy = g2_y - g1_y
    
    # 3. Compute Shared Edges
    # We iterate because vectorized intersection is not standard/easy in geopandas
    errors = []
    
    # Pre-fetch geometries to avoid repeated indexing overhead
    geoms1 = df.loc[pairs.index, 'geometry'].values
    geoms2 = df.loc[pairs.index_right, 'geometry'].values
    
    for i in range(len(pairs)):
        poly1 = geoms1[i]
        poly2 = geoms2[i]
        
        # Intersection should be the shared edge
        inter = poly1.intersection(poly2)
        
        # Handle cases where they touch at a point or multipoint (not an edge)
        if inter.is_empty or inter.geom_type not in ['LineString', 'MultiLineString']:
            errors.append(np.nan)
            continue
            
        if inter.geom_type == 'MultiLineString':
            # Take the longest segment if multiple touches
            if not inter.geoms:
                errors.append(np.nan)
                continue
            edge = max(inter.geoms, key=lambda x: x.length)
        else:
            edge = inter
            
        # Vector E (Edge)
        coords = list(edge.coords)
        if len(coords) < 2:
            errors.append(np.nan)
            continue
            
        # Vector from start to end of edge
        Ex = coords[-1][0] - coords[0][0]
        Ey = coords[-1][1] - coords[0][1]
        
        # Normal N (-Ey, Ex)
        Nx, Ny = -Ey, Ex
        
        # Cosine of angle between G and N
        # dot = Gx*Nx + Gy*Ny
        dot = Gx[i]*Nx + Gy[i]*Ny
        mag_g = np.sqrt(Gx[i]**2 + Gy[i]**2)
        mag_n = np.sqrt(Nx**2 + Ny**2)
        
        if mag_g == 0 or mag_n == 0:
            errors.append(np.nan)
            continue
            
        cos_theta = abs(dot) / (mag_g * mag_n)
        
        # Clamp for float errors
        cos_theta = min(1.0, max(0.0, cos_theta))
        
        # Angle in degrees (0 is perfect)
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        
        errors.append(angle_deg)

    pairs['ortho_error'] = errors
    
    # 4. Aggregate back to cells (Max error per cell)
    # We need to assign this error to BOTH neighbors involved in the edge
    s1 = pairs['ortho_error'].groupby(pairs.index).max()
    s2 = pairs['ortho_error'].groupby(pairs.index_right).max()
    
    # Combine and fill missing (cells with no neighbors or only point contacts get 0.0)
    combined = pd.concat([s1, s2], axis=1).max(axis=1)
    final_series = combined.reindex(gdf.index).fillna(0.0)
    
    return final_series

def calculate_mesh_quality(gdf: gpd.GeoDataFrame, calc_ortho: bool = False) -> gpd.GeoDataFrame:
    """
    Calculates geometric quality metrics for a Voronoi grid.
    
    Args:
        gdf: GeoDataFrame containing Voronoi cells. 
        calc_ortho: If True, calculates orthogonality error (slower).
             
    Returns:
        GeoDataFrame with added quality columns.
    """
    df = gdf.copy()
    
    # 1. Basic Geometry
    df['area'] = df.geometry.area
    df['perimeter'] = df.geometry.length
    
    # 2. Compactness (Isoperimetric Quotient)
    df['compactness'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
    
    # 3. Convexity (Solidity)
    df['convexity'] = df['area'] / df.geometry.convex_hull.area
    
    # 4. Generator Metrics (if available)
    if 'x' in df.columns and 'y' in df.columns:
        centroids = df.geometry.centroid
        dx = df['x'] - centroids.x
        dy = df['y'] - centroids.y
        df['centroid_dist'] = np.sqrt(dx*dx + dy*dy)
        df['drift_ratio'] = df['centroid_dist'] / np.sqrt(df['area'])
        
        if calc_ortho:
            df['ortho_error'] = calculate_orthogonality(df)
        
    return df

def summarize_quality(gdf: gpd.GeoDataFrame):
    """
    Prints a summary report of mesh quality metrics, separating Internal vs Boundary cells.
    """
    # Ensure metrics exist
    if 'compactness' not in gdf.columns:
        gdf = calculate_mesh_quality(gdf, calc_ortho=False)
        
    print("\n--- Mesh Quality Report ---")
    print(f"Total Cells: {len(gdf)}")
    
    # 1. Detect Boundary Cells
    # Heuristic: If generator is on the polygon boundary, it's a boundary cell.
    is_boundary = np.zeros(len(gdf), dtype=bool)
    if 'x' in gdf.columns and 'y' in gdf.columns:
        # Create points from generators
        gens = gpd.GeoSeries(gpd.points_from_xy(gdf.x, gdf.y), index=gdf.index)
        # Calculate distance to the boundary of the cell itself
        # For boundary cells, the generator lies ON the cell boundary (dist ~ 0)
        dists = gdf.geometry.boundary.distance(gens)
        
        # Tolerance: 1% of cell characteristic length
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
    
    # Diagnostic Warnings
    print("\n--- Diagnostics ---")
    
    # Internal Checks (Strict)
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

    # Boundary Checks (Geometric Norms)
    if not boundary_df.empty:
        # Boundary cells are naturally less compact (~0.5-0.7)
        bad_bnd = len(boundary_df[boundary_df['compactness'] < 0.4])
        if bad_bnd > 0:
            print(f"[WARNING] {bad_bnd} BOUNDARY cells are potential slivers (< 0.4).")
            
        # Boundary cells naturally have drift ~0.34. We warn if > 0.45
        if 'drift_ratio' in boundary_df.columns:
            high_drift_bnd = len(boundary_df[boundary_df['drift_ratio'] > 0.45])
            if high_drift_bnd > 0:
                print(f"[WARNING] {high_drift_bnd} BOUNDARY cells have excessive drift (> 0.45).")
            else:
                print("[OK] Boundary drift is within geometric norms (~0.34).")