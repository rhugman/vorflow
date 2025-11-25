import numpy as np
import geopandas as gpd
import pandas as pd
import gmsh
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union, split
from shapely.validation import make_valid

class VoronoiTessellator:
    def __init__(self, mesh_generator, conceptual_mesh):
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
        
        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if not region or -1 in region:
                continue
            
            verts = vor.vertices[region]
            poly = Polygon(verts)
            
            if poly.is_valid:
                polygons.append(poly)
                # Handle ghost nodes (which have no tag)
                if i < len(node_tags):
                    ids.append(node_tags[i])
                else:
                    ids.append(-1) 
        
        gdf = gpd.GeoDataFrame(
            {'node_id': ids}, 
            geometry=polygons, 
            crs=self.cm.crs
        )
        return gdf

    def _enforce_barriers(self, grid_gdf):
        """
        Splits Voronoi cells along 'Barrier' lines.
        """
        if self.cm.clean_lines.empty:
            return grid_gdf
            
        barriers = self.cm.clean_lines[self.cm.clean_lines['is_barrier'] == True]
        if barriers.empty:
            return grid_gdf
            
        print(f"Enforcing Barrier Cuts on {len(barriers)} barrier lines...")
        
        current_grid = grid_gdf
        
        for idx, row in barriers.iterrows():
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
                        for piece in split_result.geoms:
                            if isinstance(piece, (Polygon, MultiPolygon)):
                                new_row = cell_row.copy()
                                new_row.geometry = piece
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

        print("Clipping to Domain Boundary...")
        
        # FIX: Robust Domain Definition
        # Instead of relying on self.cm.domain_boundary (which might be unset),
        # we calculate the union of the actual meshed polygons.
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
            print(f"  -> Raw Bounds: {raw_gdf.total_bounds}")
            print(f"  -> Domain Bounds: {domain_gdf.total_bounds}")
            return bounded_voronoi

        print("Enforcing Hydrogeological Zones...")
        zones = self.cm.clean_polygons[['geometry', 'zone_id', 'z_order']]
        
        # Intersection with Zones
        zoned_grid = gpd.overlay(
            bounded_voronoi, 
            zones, 
            how='intersection', 
            keep_geom_type=True
        )
        print(f"  -> After Zone Overlay: {len(zoned_grid)}")
        
        # Explode MultiPolygons
        zoned_grid = zoned_grid.explode(index_parts=True).reset_index(drop=True)
        
        # Enforce Barriers
        self.final_grid = self._enforce_barriers(zoned_grid)
        print(f"  -> After Barrier Cuts: {len(self.final_grid)}")
        
        # Final Cleanup
        self.final_grid = self.final_grid.explode(index_parts=True).reset_index(drop=True)
        
        # Calculate centroids
        self.final_grid['x'] = self.final_grid.geometry.centroid.x
        self.final_grid['y'] = self.final_grid.geometry.centroid.y
        
        print(f"Final Voronoi Grid Generated: {len(self.final_grid)} cells.")
        return self.final_grid
# ...existing code...

    def export_to_shapefile(self, filepath):
        if self.final_grid is not None and not self.final_grid.empty:
            self.final_grid.to_file(filepath)
            print(f"Saved to {filepath}")
        else:
            print("No grid to export.")