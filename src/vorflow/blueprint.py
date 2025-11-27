import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, LineString, Point, box, MultiPolygon
from shapely.ops import unary_union, snap, linemerge
from shapely.validation import make_valid

class ConceptualMesh:
    def __init__(self, crs="EPSG:4326"):
        """
        Initializes the conceptual model, which holds raw geometric inputs.

        This class acts as a staging area for geometric features (polygons, lines,
        points) before they are processed into a clean, non-overlapping set of
        inputs for the mesh generator.

        Args:
            crs: The coordinate reference system for the project (e.g., "EPSG:4326").
        """
        self.crs = crs
        # Store raw geometric inputs before processing.
        self.raw_polygons = [] 
        self.raw_lines = []
        self.raw_points = []
        
        # Geometries after cleaning, snapping, and processing.
        self.domain_boundary = None
        self.clean_polygons = gpd.GeoDataFrame()
        self.clean_lines = gpd.GeoDataFrame()
        self.clean_points = gpd.GeoDataFrame()

    def add_polygon(self, geometry, zone_id, resolution=None, z_order=0, mesh_refinement=True, dist_min=None, dist_max=None, dist_max_in=None, dist_max_out=None, border_density=None):
        """
        Adds a polygon feature, such as a model boundary or a refinement zone.

        Args:
            geometry (shapely.Polygon): The polygon geometry.
            zone_id (int or str): A unique identifier for the zone.
            resolution (float, optional): Target mesh size within this polygon. If None,
                the background mesh size will be used.
            z_order (int): Stacking order for resolving overlaps. Higher values are
                processed first and will "cut" into lower-order polygons.
            mesh_refinement (bool): If True, this polygon will be used to control mesh
                refinement. If False, it is used only for tagging the final cells.
            dist_min (float, optional): Distance from the polygon boundary where the mesh
                size is held constant at the boundary's resolution.
            dist_max (float, optional): Legacy alias for `dist_max_out`.
            dist_max_in (float, optional): Distance inside the polygon over which the mesh
                transitions from the boundary resolution to the internal resolution.
            dist_max_out (float, optional): Distance outside the polygon over which the
                mesh transitions to the background resolution.
            border_density (float, optional): If set, densifies the polygon's boundary
                by adding vertices, ensuring no segment is longer than this value.
        """
        if not geometry.is_valid:
            geometry = make_valid(geometry)
            
        # For backward compatibility, allow 'dist_max' to function as 'dist_max_out'.
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
        Adds a line feature, such as a river, fault, or other linear boundary.

        Args:
            geometry (shapely.LineString): The line geometry.
            line_id (str): A unique identifier for the line.
            resolution (float): Target mesh size along the line.
            snap_to_polygons (bool): If True, the line's endpoints will be snapped to
                nearby polygon boundaries to ensure connectivity.
            is_barrier (bool): If True, the line is treated as a flow barrier. The mesh
                will be constructed to prevent cell faces from crossing it.
            dist_min (float, optional): Distance from the line where the mesh size is
                held constant at the line's resolution.
            dist_max (float, optional): Distance from the line over which the mesh
                transitions to the background resolution.
            straddle_width (float, optional): If set, forces Voronoi cell edges to align
                perfectly with the line by creating a "virtual straddle" of mesh nodes.
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
            'straddle_width': straddle_width
        })

    def add_point(self, geometry, point_id, resolution, dist_min=None, dist_max=None):
        """
        Adds a point feature, such as a well or an observation point.

        Args:
            geometry (shapely.Point): The point geometry.
            point_id (str): A unique identifier for the point.
            resolution (float): Target mesh size at the point.
            dist_min (float, optional): Distance from the point where the mesh size is
                held constant at the point's resolution.
            dist_max (float, optional): Distance from the point over which the mesh
                transitions to the background resolution.
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
        Processes polygons based on their `z_order` to create a flat,
        non-overlapping planar partition. Higher `z_order` polygons "cookie-cut"
        lower ones.
        """
        # Sort polygons by priority, with the highest z_order processed first.
        df = pd.DataFrame(self.raw_polygons)
        df = df.sort_values(by='z_order', ascending=False)
        
        processed_geoms = []
        occupied_space = None # Tracks the union of all higher-priority polygons.
        
        final_features = []

        for idx, row in df.iterrows():
            current_geo = row['geometry']
            
            if occupied_space is None:
                # The first (highest priority) polygon is added unmodified.
                final_geo = current_geo
                occupied_space = current_geo
            else:
                # Subtract the already-occupied space from the current polygon.
                try:
                    final_geo = current_geo.difference(occupied_space)
                except Exception as e:
                    # If the standard difference fails, try again with valid geometries.
                    current_geo = make_valid(current_geo)
                    occupied_space = make_valid(occupied_space)
                    final_geo = current_geo.difference(occupied_space)

                # Add the current polygon's footprint to the occupied space.
                occupied_space = unary_union([occupied_space, current_geo])
            
            # Skip if the polygon was completely covered by higher-priority ones.
            if final_geo.is_empty:
                continue
                
            # If the difference operation resulted in a MultiPolygon, explode it into
            # individual Polygons, each inheriting the parent's attributes.
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
        Snaps features together to ensure they are topologically connected before
        being passed to the mesher. This is crucial for Gmsh to correctly

        interpret shared boundaries.
        """
        # 1. Collect all polygon boundaries into a single geometry.
        # We snap to the linear boundaries, not the polygon areas.
        if not self.clean_polygons.empty:
            poly_boundaries = unary_union(self.clean_polygons.geometry.boundary)
        else:
            poly_boundaries = None

        # 2. Snap lines to polygon boundaries.
        # This ensures that features like rivers connect precisely to zone edges.
        if self.raw_lines and poly_boundaries is not None and not poly_boundaries.is_empty:
            print(f"Snapping {len(self.raw_lines)} lines to polygon boundaries (tol={tolerance})...")
            for i, line_data in enumerate(self.raw_lines):
                original_line = line_data['geometry']
                snapped_line = snap(original_line, poly_boundaries, tolerance)
                self.raw_lines[i]['geometry'] = snapped_line

        # 3. Snap points to all other geometries (lines and polygon boundaries).
        # This ensures points like wells are located exactly on a feature.
        if self.raw_points:
            geoms_to_snap_to = []
            if poly_boundaries is not None and not poly_boundaries.is_empty:
                geoms_to_snap_to.append(poly_boundaries)
            
            if self.raw_lines:
                # Use the (potentially modified) snapped lines for snapping points.
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
        Runs the full preprocessing workflow: resolves polygon overlaps,
        ensures topological connectivity, and prepares clean GeoDataFrames
        for the mesher.
        """
        print("Resolving polygon overlaps...")
        self._resolve_overlaps()
        
        print("Enforcing strict topology...")
        self._enforce_connectivity()
        
        # Promote the processed raw geometries to final "clean" GeoDataFrames.
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

    
    def _densify_geometry(self, geometry, resolution):
        """
        Recursively adds vertices to LineStrings and Polygon boundaries to ensure
        that no segment is longer than the specified resolution. This is critical
        for forcing the mesh to respect a desired element size along a feature.
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
                    # Add intermediate points along the segment.
                    for j in range(1, num_segments):
                        t = j / num_segments
                        p_new = p1 + t * (p2 - p1)
                        new_coords.append(tuple(p_new))
                
                # Always add the original endpoint of the segment.
                new_coords.append(coords[i+1])
            
            return LineString(new_coords)

        if geometry.geom_type == 'LineString':
            return densify_line(geometry, resolution)
            
            
        elif geometry.geom_type == 'Polygon':
            # Densify the exterior ring.
            new_exterior = densify_line(geometry.exterior, resolution)
            
            # Densify all interior rings (holes).
            new_interiors = []
            for interior in geometry.interiors:
                new_interiors.append(densify_line(interior, resolution))
                
            return Polygon(new_exterior, new_interiors)
            
        elif geometry.geom_type == 'MultiPolygon':
            parts = [self._densify_geometry(p, resolution) for p in geometry.geoms]
            return MultiPolygon(parts)
            
        return geometry


    def _apply_densification(self):
        """Applies densification to the clean polygon and line features."""
        # Densify polygon boundaries where a 'border_density' is specified.
        if not self.clean_polygons.empty:
            self.clean_polygons['geometry'] = self.clean_polygons.apply(
                lambda row: self._densify_geometry(row['geometry'], row['border_density']) 
                if pd.notna(row.get('border_density')) else row['geometry'], axis=1
            )

        # Densify lines based on their target resolution ('lc').
        if not self.clean_lines.empty:
            self.clean_lines['geometry'] = self.clean_lines.apply(
                lambda row: self._densify_geometry(row['geometry'], row['lc']), axis=1
            )