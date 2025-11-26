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

    def add_polygon(self, geometry, zone_id, resolution=None, z_order=0, mesh_refinement=True, dist_min=None, dist_max=None, dist_max_in=None, dist_max_out=None, border_density=None):
        """
        Registers a polygon zone.
        
        Args:
            geometry (shapely.Polygon): The geometry.
            zone_id (int/str): Identifier.
            resolution (float, optional): Mesh size inside the zone. Defaults to background LC if None.
            z_order (int): Stacking order.
            mesh_refinement (bool): If False, just for tagging.
            dist_min (float): Distance from boundary where size remains constant (Wall width).
            dist_max (float): Legacy shorthand for dist_max_out.
            dist_max_in (float): Distance inside where size reaches internal resolution.
            dist_max_out (float): Distance outside where size reaches background.
            border_density (float, optional): Max segment length for polygon boundary densification.
        """
        if not geometry.is_valid:
            geometry = make_valid(geometry)
            
        # Handle legacy dist_max
        if dist_max is not None and dist_max_out is None:
            dist_max_out = dist_max

        self.raw_polygons.append({
            'geometry': geometry,
            'zone_id': zone_id,
            'lc': resolution,
            'z_order': z_order,
            'refine': mesh_refinement,
            'dist_min': dist_min,
            'dist_max_in': dist_max_in,
            'dist_max_out': dist_max_out,
            'border_density': border_density
        })

    def add_line(self, geometry, line_id, resolution, snap_to_polygons=True, is_barrier=False, dist_min=None, dist_max=None, straddle_width=None):
        """
        Ingests features like rivers, faults, or boundary conditions.
        
        Args:
            geometry (shapely.LineString): The geometry.
            line_id (str): Identifier.
            resolution (float): Target mesh size on the line.
            snap_to_polygons (bool): Whether to snap to zone boundaries.
            is_barrier (bool): If True, acts as a flow barrier.
            dist_min (float, optional): Distance from line where mesh size remains constant.
            dist_max (float, optional): Distance from line where mesh size reaches background size.
            straddle_width (float, optional): If set, replaces the line with two parallel lines 
                                              separated by this width. Forces Voronoi face alignment.
        """
        if not geometry.is_valid:
            geometry = make_valid(geometry)
            
        self.raw_lines.append({
            'geometry': geometry,
            'line_id': line_id,
            'lc': resolution,
            'is_barrier': is_barrier,
            'dist_min': dist_min,
            'dist_max': dist_max,
            'straddle_width': straddle_width # New Parameter
        })

    def add_point(self, geometry, point_id, resolution, dist_min=None, dist_max=None):
        """
        Ingests wells or observation points.
        
        Args:
            geometry (shapely.Point): The geometry.
            point_id (str): Identifier.
            resolution (float): Target mesh size at the point.
            dist_min (float, optional): Distance from point where mesh size remains constant.
            dist_max (float, optional): Distance from point where mesh size reaches background size.
        """
        self.raw_points.append({
            'geometry': geometry,
            'point_id': point_id,
            'lc': resolution,
            'dist_min': dist_min,
            'dist_max': dist_max
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
        # 1. Collect Polygon Boundaries
        # We snap to boundaries (LineStrings), not the filled Polygons
        if not self.clean_polygons.empty:
            poly_boundaries = unary_union(self.clean_polygons.geometry.boundary)
        else:
            poly_boundaries = None

        # 2. Snap Lines to Polygon Boundaries
        # This ensures that if a river ends near a zone boundary, it connects exactly.
        if self.raw_lines and poly_boundaries is not None and not poly_boundaries.is_empty:
            print(f"Snapping {len(self.raw_lines)} lines to polygon boundaries (tol={tolerance})...")
            for i, line_data in enumerate(self.raw_lines):
                original_line = line_data['geometry']
                # snap(input, reference, tolerance)
                snapped_line = snap(original_line, poly_boundaries, tolerance)
                self.raw_lines[i]['geometry'] = snapped_line

        # 3. Snap Points to Everything (Lines + Polygons)
        # This ensures wells sit exactly on faults or boundaries if they are close.
        if self.raw_points:
            geoms_to_snap_to = []
            if poly_boundaries is not None and not poly_boundaries.is_empty:
                geoms_to_snap_to.append(poly_boundaries)
            
            if self.raw_lines:
                # Use the potentially updated lines
                lines_union = unary_union([d['geometry'] for d in self.raw_lines])
                geoms_to_snap_to.append(lines_union)
            
            if geoms_to_snap_to:
                reference_geom = unary_union(geoms_to_snap_to)
                
                print(f"Snapping {len(self.raw_points)} points to geometry (tol={tolerance})...")
                for i, point_data in enumerate(self.raw_points):
                    original_point = point_data['geometry']
                    snapped_point = snap(original_point, reference_geom, tolerance)
                    self.raw_points[i]['geometry'] = snapped_point


    def generate(self):
        """
        Main execution method.
        """
        print("Resolving polygon overlaps...")
        self._resolve_overlaps()
        
        print("Enforcing strict topology...")
        self._enforce_connectivity()
        
        # Clean Lines
        if self.raw_lines:
            self.clean_lines = gpd.GeoDataFrame(self.raw_lines, crs=self.crs)
        else:
            self.clean_lines = gpd.GeoDataFrame(columns=['geometry', 'line_id', 'lc', 'is_barrier', 'dist_min', 'dist_max', 'straddle_width'], crs=self.crs)
   
        # Clean Points
        if self.raw_points:
            self.clean_points = gpd.GeoDataFrame(self.raw_points, crs=self.crs)
        else:
            self.clean_points = gpd.GeoDataFrame(columns=['geometry', 'point_id', 'lc', 'dist_min', 'dist_max'], crs=self.crs)

        print("Densifying geometry...")
        self._apply_densification()
        
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
            # Use border_density if available. If not, do NOT densify (return original geometry).
            self.clean_polygons['geometry'] = self.clean_polygons.apply(
                lambda row: self._densify_geometry(row['geometry'], row['border_density']) 
                if pd.notna(row.get('border_density')) else row['geometry'], axis=1
            )

        # Densify Lines
        if not self.clean_lines.empty:
            self.clean_lines['geometry'] = self.clean_lines.apply(
                lambda row: self._densify_geometry(row['geometry'], row['lc']), axis=1
            )