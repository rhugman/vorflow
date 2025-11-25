import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, LineString, Point, box, MultiPolygon
from shapely.ops import unary_union, snap, linemerge
from shapely.validation import make_valid

class ConceptualMesh:
    def __init__(self, crs="EPSG:4326"):
        """
        Orchestrates the preprocessing of geometry before meshing.
        
        Args:
            crs: Coordinate reference system for the project.
        """
        self.crs = crs
        # Storing raw inputs with their metadata (resolution, z_order, etc.)
        self.raw_polygons = [] 
        self.raw_lines = []
        self.raw_points = []
        
        # The calculated "clean" domain
        self.domain_boundary = None
        self.clean_polygons = gpd.GeoDataFrame()
        self.clean_lines = gpd.GeoDataFrame()
        self.clean_points = gpd.GeoDataFrame()

    def add_polygon(self, geometry, zone_id, resolution, z_order=0, mesh_refinement=True):
        """
        Ingests polygon zones (aquifers, lakes, exclusion zones).
        
        Args:
            geometry (shapely.Polygon): The geometry object.
            zone_id (str/int): Identifier for the material property/zone.
            resolution (float): Target mesh edge length in this zone.
            z_order (int): Priority. Higher z_order cuts through lower z_order.
            mesh_refinement (bool): If True, adds interior points for generation.
        """
        # Ensure validity immediately upon entry
        if not geometry.is_valid:
            geometry = make_valid(geometry)
            
        self.raw_polygons.append({
            'geometry': geometry,
            'zone_id': zone_id,
            'lc': resolution, # 'lc' is gmsh shorthand for characteristic length
            'z_order': z_order,
            'refine': mesh_refinement
        })


    def add_line(self, geometry, line_id, resolution, snap_to_polygons=True, is_barrier=False):
        """Ingests features like rivers, faults, or boundary conditions."""
        if not geometry.is_valid:
            geometry = make_valid(geometry)
            
        self.raw_lines.append({
            'geometry': geometry,
            'line_id': line_id,
            'lc': resolution,
            'is_barrier': is_barrier 
        })

    def add_point(self, geometry, point_id, resolution):
        """Ingests wells or observation points."""
        self.raw_points.append({
            'geometry': geometry,
            'point_id': point_id,
            'lc': resolution
        })

    def _resolve_overlaps(self):
        """
        The 'Cookie Cutter' Algorithm.
        Processes polygons by z_order to create a flat, non-overlapping planar graph.
        """
        # 1. Sort by Priority (Highest Z-order first)
        df = pd.DataFrame(self.raw_polygons)
        df = df.sort_values(by='z_order', ascending=False)
        
        processed_geoms = []
        occupied_space = None # Shapely geometry representing filled area
        
        final_features = []

        for idx, row in df.iterrows():
            current_geo = row['geometry']
            
            if occupied_space is None:
                # First (highest priority) polygon sits as is
                final_geo = current_geo
                occupied_space = current_geo
            else:
                # Subtract the already occupied space from the current polygon
                try:
                    final_geo = current_geo.difference(occupied_space)
                except Exception as e:
                    # Fallback for complex topology errors
                    current_geo = make_valid(current_geo)
                    occupied_space = make_valid(occupied_space)
                    final_geo = current_geo.difference(occupied_space)

                # Update occupied space
                occupied_space = unary_union([occupied_space, current_geo])
            
            # If the difference resulted in empty geometry (fully covered), skip
            if final_geo.is_empty:
                continue
                
            # Handle MultiPolygons (explode them)
            if final_geo.geom_type == 'MultiPolygon':
                for part in final_geo.geoms:
                    feat = row.copy()
                    feat['geometry'] = part
                    final_features.append(feat)
            else:
                feat = row.copy()
                feat['geometry'] = final_geo
                final_features.append(feat)

        self.clean_polygons = gpd.GeoDataFrame(final_features, crs=self.crs)

    def _enforce_connectivity(self, tolerance=1e-3):
        """
        Ensures lines snap to polygon boundaries and points snap to lines/polygons
        to ensure Gmsh treats them as connected.
        """
        # Implementation strategy:
        # 1. Snap Points to Lines
        # 2. Snap Lines to Polygon Boundaries
        # 3. Ensure Polygon shared edges are identical (no slivers)
        
        # (Placeholder for complex snapping logic using shapely.ops.snap)
        pass

    def generate(self):
        """
        Main execution method.
        1. Resolves overlaps.
        2. Cleans topology.
        3. Applies densification to polygons and lines.
        4. Prepares data for the MeshGenerator.
        """
        print("Resolving polygon overlaps...")
        self._resolve_overlaps()
        
        print("Enforcing strict topology...")
        self._enforce_connectivity()
        
        # Clean Lines
        if self.raw_lines:
            # Construct GeoDataFrame directly from the raw_lines list
            self.clean_lines = gpd.GeoDataFrame(self.raw_lines, crs=self.crs)
        else:
            self.clean_lines = gpd.GeoDataFrame(columns=['geometry', 'line_id', 'lc', 'is_barrier'], crs=self.crs)

        # Clean Points
        if self.raw_points:
            self.clean_points = gpd.GeoDataFrame(self.raw_points, crs=self.crs)
        else:
            self.clean_points = gpd.GeoDataFrame(columns=['geometry', 'point_id', 'lc'], crs=self.crs)


        print("Densifying geometry...")
        self._apply_densification()

        # Set the global domain boundary based on the union of all polygons
        self.domain_boundary = unary_union(self.clean_polygons.geometry)
        
        return self.clean_polygons, self.clean_lines, self.clean_points
    
    
# ...existing code...
    def _densify_geometry(self, geometry, resolution):
        """
        Adds vertices to LineStrings and Polygon boundaries to ensure segments
        are no longer than the specified resolution.
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
                    # Add intermediate points
                    for j in range(1, num_segments):
                        t = j / num_segments
                        p_new = p1 + t * (p2 - p1)
                        new_coords.append(tuple(p_new))
                
                # Always add the end point of the segment to preserve original vertices
                new_coords.append(coords[i+1])
            
            return LineString(new_coords)

        if geometry.geom_type == 'LineString':
            return densify_line(geometry, resolution)
            
        elif geometry.geom_type == 'Polygon':
            # Densify exterior
            new_exterior = densify_line(geometry.exterior, resolution)
            
            # Densify interiors (holes)
            new_interiors = []
            for interior in geometry.interiors:
                new_interiors.append(densify_line(interior, resolution))
                
            return Polygon(new_exterior, new_interiors)
            
        elif geometry.geom_type == 'MultiPolygon':
            parts = [self._densify_geometry(p, resolution) for p in geometry.geoms]
            return MultiPolygon(parts)
            
        return geometry


    def _apply_densification(self):
        """Applies densification to clean polygons and lines."""
        # Densify Polygons
        if not self.clean_polygons.empty:
            self.clean_polygons['geometry'] = self.clean_polygons.apply(
                lambda row: self._densify_geometry(row['geometry'], row['lc']), axis=1
            )

        # Densify Lines
        if not self.clean_lines.empty:
            self.clean_lines['geometry'] = self.clean_lines.apply(
                lambda row: self._densify_geometry(row['geometry'], row['lc']), axis=1
            )