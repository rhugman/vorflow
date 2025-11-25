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
        Converts the Triangular Mesh into a Polgonal Voronoi Grid,
        strictly respecting the boundaries defined in ConceptualMesh.
        """
        self.mg = mesh_generator
        self.cm = conceptual_mesh
        self.voronoi_gdf = None
        self.final_grid = None
        self.nodes = mesh_generator.nodes
        self.node_tags = mesh_generator.node_tags
        self.zones_gdf = mesh_generator.zones_gdf

    def _extract_nodes(self):
        """
        Pulls node coordinates from the active Gmsh model.
        """
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes_3d = np.array(coords).reshape(-1, 3)
        nodes_2d = nodes_3d[:, :2]
        return nodes_2d, node_tags

    def _build_raw_voronoi(self, nodes, node_tags):
        """
        Uses Scipy to compute the mathematical Voronoi diagram.
        """
        vor = Voronoi(nodes)
        polygons = []
        ids = []
        
        # Map point_region to input index
        # vor.point_region is an array where index i corresponds to input point i
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
                    ids.append(-1) # Ghost node ID
        
        gdf = gpd.GeoDataFrame(
            {'node_id': ids}, 
            geometry=polygons, 
            crs=self.cm.crs
        )
        return gdf

    def _enforce_barriers(self, grid_gdf):
        """
        Splits Voronoi cells along 'Barrier' lines.
        Since nodes are on the line, the cells straddle the line.
        We cut them to create a face along the fault.
        """
        # 1. Identify Barrier Lines
        if self.cm.clean_lines.empty:
            return grid_gdf
            
        barriers = self.cm.clean_lines[self.cm.clean_lines['is_barrier'] == True]
        if barriers.empty:
            return grid_gdf
            
        print("Enforcing Barrier Cuts...")
        
        # We process the grid list-wise to handle splits
        # Converting to list of dicts is often faster for manipulation than pure GDF
        new_cells = []
        
        # Spatial Index for speed
        sindex = grid_gdf.sindex
        
        # Track which cells have been processed (split)
        processed_indices = set()
        
        # Iterate over barriers
        # Note: If barriers intersect each other, this simple loop might need recursion,
        # but for now we assume barriers are handled sequentially.
        
        # To do this robustly using GeoPandas:
        # We can use the 'split' operation on the whole geometry column? 
        # No, shapely.ops.split works on single geometries.
        
        # Strategy:
        # 1. Find all cells intersecting ANY barrier.
        # 2. For each such cell, split it by the barrier(s).
        # 3. Keep the pieces.
        
        # Let's do it per barrier to ensure we handle the specific line geometry
        current_grid = grid_gdf
        
        for idx, row in barriers.iterrows():
            line = row.geometry
            
            # Find candidate cells
            possible_matches_index = list(current_grid.sindex.query(line, predicate='intersects'))
            candidate_cells = current_grid.iloc[possible_matches_index]
            
            cells_to_keep = []
            cells_to_remove_indices = []
            
            for cell_idx, cell_row in candidate_cells.iterrows():
                cell_poly = cell_row.geometry
                
                # Check actual intersection (sindex is bounding box)
                if not cell_poly.intersects(line):
                    continue
                    
                # Perform Split
                # shapely.split(polygon, line) -> GeometryCollection
                try:
                    split_result = split(cell_poly, line)
                    
                    if len(split_result.geoms) > 1:
                        # Successful split!
                        cells_to_remove_indices.append(cell_idx)
                        
                        # Create new entries for the pieces
                        for piece in split_result.geoms:
                            if isinstance(piece, (Polygon, MultiPolygon)):
                                new_row = cell_row.copy()
                                new_row.geometry = piece
                                # We might want to update node_id or add a flag, 
                                # but for now we keep the parent ID (duplicate IDs allowed in DISV)
                                cells_to_keep.append(new_row)
                except Exception as e:
                    print(f"Warning: Failed to split cell {cell_row['node_id']} along barrier: {e}")
            
            if cells_to_remove_indices:
                # Remove old cells
                current_grid = current_grid.drop(cells_to_remove_indices)
                # Add new pieces
                new_df = gpd.GeoDataFrame(cells_to_keep, crs=current_grid.crs)
                current_grid = pd.concat([current_grid, new_df], ignore_index=True)
        
        return current_grid

    def generate(self):
        """
        Main execution workflow.
        """
        if self.nodes is None:
            raise RuntimeError("MeshGenerator data (nodes) is missing. Run MeshGenerator.generate() first.")
        
        print("Extracting Nodes from Gmsh...")
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
        
        # Filter ghost cells
        raw_gdf = raw_gdf[raw_gdf['node_id'] != -1]
        
        print("Clipping to Domain Boundary...")
        domain_gdf = gpd.GeoDataFrame(
            geometry=[self.cm.domain_boundary], 
            crs=self.cm.crs
        )
        bounded_voronoi = gpd.clip(raw_gdf, domain_gdf)
        
        print("Enforcing Hydrogeological Zones...")
        zones = self.cm.clean_polygons[['geometry', 'zone_id', 'z_order']]
        
        # Intersection with Zones
        zoned_grid = gpd.overlay(
            bounded_voronoi, 
            zones, 
            how='intersection', 
            keep_geom_type=True
        )
        
        # Explode MultiPolygons
        zoned_grid = zoned_grid.explode(index_parts=True).reset_index(drop=True)
        
        # --- NEW: Enforce Barriers ---
        self.final_grid = self._enforce_barriers(zoned_grid)
        
        # Final Cleanup
        self.final_grid = self.final_grid.explode(index_parts=True).reset_index(drop=True)
        
        # Calculate centroids
        self.final_grid['x'] = self.final_grid.geometry.centroid.x
        self.final_grid['y'] = self.final_grid.geometry.centroid.y
        
        print(f"Final Voronoi Grid Generated: {len(self.final_grid)} cells.")
        return self.final_grid

    def export_to_shapefile(self, filepath):
        if self.final_grid is not None:
            self.final_grid.to_file(filepath)
            print(f"Saved to {filepath}")