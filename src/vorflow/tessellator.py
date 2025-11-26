import numpy as np
import geopandas as gpd
import pandas as pd
import gmsh
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union, split
from shapely.validation import make_valid

class VoronoiTessellator:
    def __init__(self, mesh_generator, conceptual_mesh,clip_to_boundary=True):
        """
        Converts the Triangular Mesh into a Polgonal Voronoi Grid.
        """
        self.mg = mesh_generator
        self.cm = conceptual_mesh
        self.voronoi_gdf = None
        self.final_grid = None
        self.nodes = mesh_generator.nodes
        self.node_tags = mesh_generator.node_tags
        self.zones_gdf = mesh_generator.zones_gdf
        self.clip_to_boundary = clip_to_boundary

    def _build_raw_voronoi(self, nodes, node_tags):
        """
        Uses Scipy to compute the mathematical Voronoi diagram.
        """
        if len(nodes) < 3:
            print("Error: Not enough nodes to generate Voronoi.")
            return gpd.GeoDataFrame()

        vor = Voronoi(nodes)
        polygons = []
        ids = []
        gen_x = []
        gen_y = []
        
        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if not region or -1 in region:
                continue
            
            verts = vor.vertices[region]
            poly = Polygon(verts)
            
            if poly.is_valid:
                polygons.append(poly)
                # Capture generator coordinates from the input nodes
                gen_x.append(nodes[i][0])
                gen_y.append(nodes[i][1])
                
                # Handle ghost nodes (which have no tag)
                if i < len(node_tags):
                    ids.append(node_tags[i])
                else:
                    ids.append(-1) 
        
        gdf = gpd.GeoDataFrame(
            {'node_id': ids, 'x': gen_x, 'y': gen_y}, 
            geometry=polygons, 
            crs=self.cm.crs
        )
        return gdf

    def _enforce_barriers(self, grid_gdf):
        """
        Splits Voronoi cells along 'Barrier' lines.
        SKIPs lines that have 'straddle_width' defined, as they are already aligned.
        """
        if self.cm.clean_lines.empty:
            return grid_gdf
            
        # Filter: Only barriers that do NOT have straddle_width
        # If straddle_width is NaN or 0, we cut. If it has a value, we skip.
        mask_barrier = self.cm.clean_lines['is_barrier'] == True
        
        # Handle case where straddle_width column might not exist yet
        if 'straddle_width' in self.cm.clean_lines.columns:
            mask_no_straddle = (self.cm.clean_lines['straddle_width'].isna()) | (self.cm.clean_lines['straddle_width'] <= 0)
        else:
            mask_no_straddle = True
            
        barriers_to_cut = self.cm.clean_lines[mask_barrier & mask_no_straddle]
        
        if barriers_to_cut.empty:
            return grid_gdf
            
        print(f"Enforcing Barrier Cuts on {len(barriers_to_cut)} lines (Straddle lines skipped)...")
        
        current_grid = grid_gdf

        # Track the maximum node ID to assign new IDs to split pieces
        max_id = grid_gdf['node_id'].max()

        for idx, row in barriers_to_cut.iterrows():
            # ...existing cut logic...
            line = row.geometry
            
            # Find candidate cells (spatial index query)
            possible_matches_index = list(current_grid.sindex.query(line, predicate='intersects'))
            candidate_cells = current_grid.iloc[possible_matches_index]
            
            cells_to_keep = []
            cells_to_remove_indices = []
            
            for cell_idx, cell_row in candidate_cells.iterrows():
                cell_poly = cell_row.geometry
                
                # Strict intersection check
                if not cell_poly.intersects(line):
                    continue
                    
                try:
                    split_result = split(cell_poly, line)
                    
                    # Only proceed if we actually split the polygon into multiple pieces
                    if len(split_result.geoms) > 1:
                        valid_pieces = []
                        
                        # Sort pieces by area, largest keeps the original ID
                        sorted_pieces = sorted(list(split_result.geoms), key=lambda p: p.area, reverse=True)
                        
                        for i, piece in enumerate(sorted_pieces):
                            if isinstance(piece, (Polygon, MultiPolygon)):
                                new_row = cell_row.copy()
                                new_row.geometry = piece
                                
                                if i == 0:
                                    # Largest piece keeps original ID
                                    pass 
                                else:
                                    # Smaller pieces get new unique IDs
                                    max_id += 1
                                    new_row['node_id'] = max_id
                                    # Note: The x/y generator is now technically "wrong" for this new cell
                                    # Ideally, calculate a new centroid for it
                                    new_row['x'] = piece.centroid.x
                                    new_row['y'] = piece.centroid.y
                                    
                                valid_pieces.append(new_row)
                        
                        # SAFETY CHECK: Only remove original if we have valid pieces to replace it
                        if valid_pieces:
                            cells_to_remove_indices.append(cell_idx)
                            cells_to_keep.extend(valid_pieces)
                            
                except Exception as e:
                    print(f"Warning: Failed to split cell {cell_row['node_id']}: {e}")
            
            if cells_to_remove_indices:
                current_grid = current_grid.drop(cells_to_remove_indices)
                new_df = gpd.GeoDataFrame(cells_to_keep, crs=current_grid.crs)
                current_grid = pd.concat([current_grid, new_df], ignore_index=True)
        
        return current_grid

# ...existing code...
    def generate(self):
        """
        Main execution workflow with debug logging.
        """
        if self.nodes is None or len(self.nodes) == 0:
            print("Error: No nodes found in MeshGenerator.")
            return gpd.GeoDataFrame()
        
        print(f"Extracting {len(self.nodes)} Nodes from Gmsh...")
        nodes, tags = self.nodes, self.node_tags
        
        # Ghost Node Trick
        minx, miny = np.min(nodes, axis=0)
        maxx, maxy = np.max(nodes, axis=0)
        w, h = maxx - minx, maxy - miny
        buffer = max(w, h) * 10
        
        ghost_nodes = np.array([
            [minx - buffer, miny - buffer],
            [maxx + buffer, miny - buffer],
            [maxx + buffer, maxy + buffer],
            [minx - buffer, maxy + buffer]
        ])
        
        combined_nodes = np.vstack([nodes, ghost_nodes])
        
        print("Computing Mathematical Voronoi...")
        raw_gdf = self._build_raw_voronoi(combined_nodes, tags)
        print(f"  -> Raw Polygons: {len(raw_gdf)}")
        
        # Filter ghost cells
        raw_gdf = raw_gdf[raw_gdf['node_id'] != -1]
        print(f"  -> After Ghost Filter: {len(raw_gdf)}")
        
        # Ensure CRS matches
        if raw_gdf.crs is None and self.cm.crs:
            raw_gdf.set_crs(self.cm.crs, inplace=True)

        
        if self.clip_to_boundary:
            print("Clipping to Domain Boundary...")
            if not self.cm.clean_polygons.empty:
                domain_geom = unary_union(self.cm.clean_polygons.geometry)
                if not domain_geom.is_valid:
                    domain_geom = make_valid(domain_geom)
            elif hasattr(self.cm, 'domain_boundary') and self.cm.domain_boundary:
                domain_geom = self.cm.domain_boundary
            else:
                print("Error: No domain geometry found (no polygons).")
                return gpd.GeoDataFrame()

            domain_gdf = gpd.GeoDataFrame(
                geometry=[domain_geom], 
                crs=self.cm.crs
            )
            
            bounded_voronoi = gpd.clip(raw_gdf, domain_gdf)
            print(f"  -> After Domain Clip: {len(bounded_voronoi)}")
        
            if len(bounded_voronoi) == 0:
                print("Warning: Clipping resulted in 0 cells. Check CRS or Domain Box.")
                return bounded_voronoi
        else:
            bounded_voronoi = raw_gdf

        print("Enforcing Hydrogeological Zones (Optimization: Point Sampling)...")
        zones = self.cm.clean_polygons[['geometry', 'zone_id', 'z_order']]
        
        # Optimization: Instead of expensive polygon overlay/intersection,
        # we sample the zone at the generator node location.
        # This avoids creating sliver polygons and is much faster.
        
        # 1. Create temporary points for spatial join
        pts_gdf = gpd.GeoDataFrame(
            {'node_id': bounded_voronoi['node_id']},
            geometry=gpd.points_from_xy(bounded_voronoi.x, bounded_voronoi.y),
            crs=bounded_voronoi.crs
        )
        
        # 2. Spatial Join to find zone for each node
        # Use 'intersects' to catch nodes on boundaries
        joined = gpd.sjoin(pts_gdf, zones, how='left', predicate='intersects')
        
        # 3. Handle nodes on boundaries (multiple matches)
        # Sort by z_order to prioritize overlaying zones
        if 'z_order' in joined.columns:
            joined = joined.sort_values('z_order', ascending=False)
        
        # Keep one entry per node
        joined = joined.drop_duplicates(subset='node_id')
        
        # 4. Map attributes back to the Voronoi polygons
        zoned_grid = bounded_voronoi.merge(
            joined[['node_id', 'zone_id', 'z_order']],
            on='node_id',
            how='left'
        )
        
        print(f"  -> Zones Assigned: {len(zoned_grid)}")
        
        # Explode MultiPolygons (safety for clip artifacts)
        zoned_grid = zoned_grid.explode(index_parts=True).reset_index(drop=True)
        
        # Enforce Barriers
        #self.final_grid = self._enforce_barriers(zoned_grid)
        self.final_grid = zoned_grid
        print(f"  -> After Barrier Cuts: {len(self.final_grid)}")
        
        # Final Cleanup
        self.final_grid = self.final_grid.explode(index_parts=True).reset_index(drop=True)

        # FIX: Do NOT overwrite 'x' and 'y' with centroids.
        # We keep 'x' and 'y' as the generator coordinates for quality analysis.
        # We add explicit centroid columns for reference.
        self.final_grid['centroid_x'] = self.final_grid.geometry.centroid.x
        self.final_grid['centroid_y'] = self.final_grid.geometry.centroid.y
        
        print(f"Final Voronoi Grid Generated: {len(self.final_grid)} cells.")
        return self.final_grid
# ...existing code...

    def export_to_shapefile(self, filepath):
        if self.final_grid is not None and not self.final_grid.empty:
            self.final_grid.to_file(filepath)
            print(f"Saved to {filepath}")
        else:
            print("No grid to export.")