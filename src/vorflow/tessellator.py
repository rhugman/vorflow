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
        Initializes the Voronoi tessellator.

        This class takes a triangular mesh (typically from `MeshGenerator`) and
        computes its dual: a Voronoi diagram. The resulting grid of polygonal
        cells is suitable for use in cell-centered finite volume models.

        Args:
            mesh_generator (MeshGenerator): An instance of the mesh generator that
                contains the generated triangular mesh nodes.
            conceptual_mesh (ConceptualMesh): The conceptual model, used for CRS,
                domain boundaries, and feature information.
            clip_to_boundary (bool): If True, the final Voronoi grid will be
                clipped to the domain boundary defined in the conceptual model.
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
        Computes the mathematical Voronoi diagram from a set of generator points.

        This method uses `scipy.spatial.Voronoi` to calculate the unbounded
        Voronoi diagram. It filters out invalid or infinite regions and returns
        the finite polygons as a GeoDataFrame.

        Args:
            nodes (np.ndarray): An array of (x, y) coordinates for the generator points.
            node_tags (np.ndarray): An array of IDs corresponding to each node.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the Voronoi polygons, with
                columns for the generator's node_id, x, and y coordinates.
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
            # Skip infinite regions (those containing -1).
            if not region or -1 in region:
                continue
            
            verts = vor.vertices[region]
            poly = Polygon(verts)
            
            if poly.is_valid:
                polygons.append(poly)
                # Store the coordinates of the generator point for this cell.
                gen_x.append(nodes[i][0])
                gen_y.append(nodes[i][1])
                
                # Assign the node tag (ID) to the cell. Ghost nodes (used to
                # bound the diagram) will not have a tag.
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
        Splits Voronoi cells that are crossed by barrier lines.

        This method iterates through all line features marked as `is_barrier=True`
        (and that do not use the `straddle_width` method) and cuts any Voronoi
        cell they intersect. The largest resulting piece of a split cell retains
        the original cell's ID, while smaller pieces are assigned new, unique IDs.

        Args:
            grid_gdf (gpd.GeoDataFrame): The current Voronoi grid.

        Returns:
            gpd.GeoDataFrame: An updated grid with cells split along barrier lines.
        """
        if self.cm.clean_lines.empty:
            return grid_gdf
            
        # We only need to cut barriers that were NOT handled by the "straddle"
        # method in the mesh generator. Straddled barriers are already aligned.
        mask_barrier = self.cm.clean_lines['is_barrier'] == True
        
        if 'straddle_width' in self.cm.clean_lines.columns:
            mask_no_straddle = (self.cm.clean_lines['straddle_width'].isna()) | (self.cm.clean_lines['straddle_width'] <= 0)
        else:
            mask_no_straddle = True
            
        barriers_to_cut = self.cm.clean_lines[mask_barrier & mask_no_straddle]
        
        if barriers_to_cut.empty:
            return grid_gdf
            
        print(f"Enforcing Barrier Cuts on {len(barriers_to_cut)} lines (Straddle lines skipped)...")
        
        current_grid = grid_gdf

        # Keep track of the highest node ID to assign to new cell fragments.
        max_id = grid_gdf['node_id'].max()

        for idx, row in barriers_to_cut.iterrows():
            line = row.geometry
            
            # Use a spatial index to quickly find cells that might intersect the line.
            possible_matches_index = list(current_grid.sindex.query(line, predicate='intersects'))
            candidate_cells = current_grid.iloc[possible_matches_index]
            
            cells_to_keep = []
            cells_to_remove_indices = []
            
            for cell_idx, cell_row in candidate_cells.iterrows():
                cell_poly = cell_row.geometry
                
                if not cell_poly.intersects(line):
                    continue
                    
                try:
                    # Split the cell polygon by the barrier line.
                    split_result = split(cell_poly, line)
                    
                    if len(split_result.geoms) > 1:
                        valid_pieces = []
                        
                        # Sort the pieces by area. The largest piece will keep the original ID.
                        sorted_pieces = sorted(list(split_result.geoms), key=lambda p: p.area, reverse=True)
                        
                        for i, piece in enumerate(sorted_pieces):
                            if isinstance(piece, (Polygon, MultiPolygon)):
                                new_row = cell_row.copy()
                                new_row.geometry = piece
                                
                                if i == 0:
                                    # The largest piece keeps the original ID and attributes.
                                    pass 
                                else:
                                    # Smaller pieces get a new ID and their own centroid.
                                    max_id += 1
                                    new_row['node_id'] = max_id
                                    new_row['x'] = piece.centroid.x
                                    new_row['y'] = piece.centroid.y
                                    
                                valid_pieces.append(new_row)
                        
                        # If the split was successful, mark the original cell for removal.
                        if valid_pieces:
                            cells_to_remove_indices.append(cell_idx)
                            cells_to_keep.extend(valid_pieces)
                            
                except Exception as e:
                    print(f"Warning: Failed to split cell {cell_row['node_id']}: {e}")
            
            # Rebuild the grid with the split cells.
            if cells_to_remove_indices:
                current_grid = current_grid.drop(cells_to_remove_indices)
                new_df = gpd.GeoDataFrame(cells_to_keep, crs=current_grid.crs)
                current_grid = pd.concat([current_grid, new_df], ignore_index=True)
        
        return current_grid

    def generate(self):
        """
        Executes the full Voronoi tessellation workflow.

        This method orchestrates the process of:
        1. Adding "ghost" nodes to create a bounded Voronoi diagram.
        2. Computing the raw Voronoi polygons.
        3. Clipping the grid to the model domain.
        4. Assigning zone IDs to cells based on their generator point location.
        5. Enforcing barrier lines by splitting cells.
        6. Calculating final cell properties.

        Returns:
            gpd.GeoDataFrame: The final, clean Voronoi grid.
        """
        if self.nodes is None or len(self.nodes) == 0:
            print("Error: No nodes found in MeshGenerator.")
            return gpd.GeoDataFrame()
        
        print(f"Extracting {len(self.nodes)} Nodes from Gmsh...")
        nodes, tags = self.nodes, self.node_tags
        
        # To create a bounded Voronoi diagram from a finite set of points, a common
        # technique is to add "ghost" nodes far outside the area of interest. The
        # large, unwanted cells generated by these ghosts can then be clipped away.
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
        
        # Remove the cells generated by the ghost nodes.
        raw_gdf = raw_gdf[raw_gdf['node_id'] != -1]
        print(f"  -> After Ghost Filter: {len(raw_gdf)}")
        
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
        
        # To assign a zone ID to each Voronoi cell, we perform a spatial join
        # between the cell's generator point and the zone polygons. This is much
        # faster than doing a polygon-on-polygon overlay.
        
        # 1. Create a temporary GeoDataFrame of the generator points.
        pts_gdf = gpd.GeoDataFrame(
            {'node_id': bounded_voronoi['node_id']},
            geometry=gpd.points_from_xy(bounded_voronoi.x, bounded_voronoi.y),
            crs=bounded_voronoi.crs
        )
        
        # 2. Spatially join the points to the zones.
        joined = gpd.sjoin(pts_gdf, zones, how='left', predicate='intersects')
        
        # 3. If a point falls on a boundary between zones, it may have multiple
        # matches. We use the `z_order` from the conceptual model to pick the
        # highest-priority zone.
        if 'z_order' in joined.columns:
            joined = joined.sort_values('z_order', ascending=False)
        
        joined = joined.drop_duplicates(subset='node_id')
        
        # 4. Merge the zone information back into the main grid.
        zoned_grid = bounded_voronoi.merge(
            joined[['node_id', 'zone_id', 'z_order']],
            on='node_id',
            how='left'
        )
        
        print(f"  -> Zones Assigned: {len(zoned_grid)}")
        
        # Clipping can sometimes create MultiPolygons; explode them into single parts.
        zoned_grid = zoned_grid.explode(index_parts=True).reset_index(drop=True)
        
        # Enforce barriers by splitting cells.
        self.final_grid = self._enforce_barriers(zoned_grid)
        print(f"  -> After Barrier Cuts: {len(self.final_grid)}")
        
        # Final cleanup after potential splits.
        self.final_grid = self.final_grid.explode(index_parts=True).reset_index(drop=True)

        # The 'x' and 'y' columns should always refer to the generator point
        # coordinates, which are essential for quality analysis. We add separate
        # columns for the geometric centroid of the final cell.
        self.final_grid['centroid_x'] = self.final_grid.geometry.centroid.x
        self.final_grid['centroid_y'] = self.final_grid.geometry.centroid.y
        
        print(f"Final Voronoi Grid Generated: {len(self.final_grid)} cells.")
        return self.final_grid

    def export_to_shapefile(self, filepath):
        if self.final_grid is not None and not self.final_grid.empty:
            self.final_grid.to_file(filepath)
            print(f"Saved to {filepath}")
        else:
            print("No grid to export.")