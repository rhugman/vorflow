import gmsh
import sys
import math
import numpy as np

class MeshGenerator:
    def __init__(self, verbosity=2):
        """
        Wraps the GMSH API.
        verbosity: 0 (silent) to 9 (debug).
        """
        self.initialized = False
        self.verbosity = verbosity
        self.nodes = None         # Node coordinates (for Voronoi generators)
        self.node_tags = None     # Node IDs (for linking results)
        self.zones_gdf = None     # Reference to the cleaned ConceptualMesh polygons
    
    def _initialize_gmsh(self):
        if not gmsh.is_initialized():
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", self.verbosity)
            # Relax tolerances slightly to handle floating point noise in inputs
            gmsh.option.setNumber("Geometry.Tolerance", 1e-6)
            gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)
            gmsh.model.add("mesh_model")
            self.initialized = True

    def _finalize_gmsh(self):
        if gmsh.is_initialized():
            gmsh.finalize()
            self.initialized = False

    def _add_geometry(self, polygons_gdf, lines_gdf, points_gdf):
        """
        Transfers Shapely geometry to Gmsh OCC kernel.
        Strategy: Fragment EVERYTHING (Surfaces, Lines, AND Points).
        
        This resolves all topological relationships:
        - Overlapping polygons are stitched.
        - Lines cut surfaces.
        - Points are inserted as vertices into surfaces/lines.
        """
        gmsh_map = {'points': {}, 'lines': {}, 'surfaces': {}}
        
        # 1. Add Points (Wells)
        all_point_tags = []
        for idx, row in points_gdf.iterrows():
            x, y = row.geometry.x, row.geometry.y
            tag = gmsh.model.occ.addPoint(x, y, 0)
            gmsh_map['points'][idx] = tag
            all_point_tags.append(tag)
            
        # 2. Add Lines (Rivers, Faults)
        all_line_tags = []
        for idx, row in lines_gdf.iterrows():
            coords = list(row.geometry.coords)
            pt_tags = []
            for x, y in coords:
                pt_tags.append(gmsh.model.occ.addPoint(x, y, 0))
            
            line_segs = []
            for i in range(len(pt_tags) - 1):
                l_tag = gmsh.model.occ.addLine(pt_tags[i], pt_tags[i+1])
                line_segs.append(l_tag)
            
            gmsh_map['lines'][idx] = line_segs
            all_line_tags.extend(line_segs)

        # 3. Add Polygons (Zones)
        initial_surface_tags = []
        for idx, row in polygons_gdf.iterrows():
            ext_coords = list(row.geometry.exterior.coords)
            if ext_coords[0] == ext_coords[-1]:
                ext_coords = ext_coords[:-1]
                
            pt_tags = [gmsh.model.occ.addPoint(x, y, 0) for x, y in ext_coords]
            
            l_tags = []
            for i in range(len(pt_tags)):
                p1, p2 = pt_tags[i], pt_tags[(i+1)%len(pt_tags)]
                l_tags.append(gmsh.model.occ.addLine(p1, p2))
            
            curve_loop = gmsh.model.occ.addCurveLoop(l_tags)
            surf = gmsh.model.occ.addPlaneSurface([curve_loop])
            
            gmsh_map['surfaces'][idx] = surf
            initial_surface_tags.append((2, surf))

        # 4. Fragment Everything
        # We include POINTS in the tool tags. 
        # This ensures points become hard vertices in the mesh topology.
        object_tags = initial_surface_tags
        
        # Tools = Lines + Points
        tool_tags = [(1, t) for t in all_line_tags]
        tool_tags.extend([(0, t) for t in all_point_tags])
        
        # Fragment
        out_dt, out_map = gmsh.model.occ.fragment(object_tags, tool_tags)
        
        # CRITICAL: Remove Duplicates
        # This merges the overlapping surfaces (Background + Refined) and 
        # snaps points to lines/surfaces if they are within tolerance.
        gmsh.model.occ.removeAllDuplicates()
        
        # Synchronize to finalize the B-Rep
        gmsh.model.occ.synchronize()
        
        # Retrieve valid 2D surfaces
        final_surface_tags = [tag for dim, tag in gmsh.model.getEntities(2)]

        # Note: We do NOT need to manually embed points anymore.
        # The fragment operation has already inserted them into the topology.

        # 5. Physical Groups
        gmsh.model.addPhysicalGroup(2, final_surface_tags, tag=2000)
        gmsh.model.setPhysicalName(2, 2000, "Meshed Domain")
        
        return gmsh_map, final_surface_tags
    
    def _setup_fields(self, gmsh_map, polygons_gdf, lines_gdf, points_gdf):
        """
        Sets up the resolution (Size) fields.
        """
        field_list = []
        
        # 0. Determine Global Background Size
        # We need this to cap the Threshold fields correctly.
        # If SizeMax is too huge, the gradation is too steep.
        global_max_lc = 100.0
        if 'lc' in polygons_gdf.columns and not polygons_gdf.empty:
            global_max_lc = float(polygons_gdf['lc'].max())
        if global_max_lc <= 0: global_max_lc = 100.0

        # Helper to find point tags robustly (since fragment might change them)
        def get_point_tag_at(x, y):
            eps = 1e-4
            # Search for 0D entities (points) near coordinates
            ents = gmsh.model.getEntitiesInBoundingBox(x-eps, y-eps, -eps, x+eps, y+eps, eps, dim=0)
            if ents:
                return ents[0][1] # Return tag
            return None

        def add_refinement(entity_dim, entity_tags, size_target, dist_min, dist_max):
            if not isinstance(entity_tags, list):
                entity_tags = [entity_tags]
            
            valid_tags = [float(t) for t in entity_tags]
            if not valid_tags: return None

            f_dist = gmsh.model.mesh.field.add("Distance")
            if entity_dim == 0: 
                gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", valid_tags)
            elif entity_dim == 1: 
                gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", valid_tags)
            
            f_thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", float(size_target))
            
            # FIX: Set SizeMax to the global background size.
            # This ensures the linear interpolation 1.0 -> 20.0 happens over DistMax.
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", float(global_max_lc))
            
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", float(dist_min))
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", float(dist_max))
            return f_thresh

        # 1. Point Resolutions
        for idx, row in points_gdf.iterrows():
            # Robust lookup: Find the tag currently at this location
            tag = get_point_tag_at(row.geometry.x, row.geometry.y)
            
            if tag:
                lc = max(row.get('lc', 5.0), 0.001)
                # Refine to 'lc' within 2m, fade to background over 150m
                fid = add_refinement(0, [tag], lc, 2.0, 150.0)
                if fid: field_list.append(fid)

        # 2. Line Resolutions
        for idx, row in lines_gdf.iterrows():
            if idx in gmsh_map['lines']:
                tags = gmsh_map['lines'][idx]
                lc = max(row.get('lc', 10.0), 0.001)
                # Refine to 'lc' within 5m, fade to background over 200m
                fid = add_refinement(1, tags, lc, 5.0, 200.0)
                if fid: field_list.append(fid)

        # 3. Global Background Field
        f_bg = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(f_bg, "F", str(global_max_lc))
        field_list.append(f_bg)

        # 4. Combine
        if field_list:
            min_field = gmsh.model.mesh.field.add("Min")
            field_list = [float(f) for f in field_list]
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_list)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        
        gmsh.option.setNumber("Mesh.Algorithm", 5) 

        
    def generate(self, clean_polys, clean_lines, clean_points, output_file=None):
        self._initialize_gmsh()
        
        try:
            print("Transferring Geometry to Gmsh...")
            gmsh_map, final_surfaces = self._add_geometry(clean_polys, clean_lines, clean_points)
            
            print("Setting up Resolution Fields...")
            self._setup_fields(gmsh_map, clean_polys, clean_lines, clean_points)
            
            print("Generating Triangular Mesh...")
            gmsh.model.mesh.generate(2)
            
            print("Optimizing Mesh (Lloyd)...")
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.model.mesh.optimize( niter=5)
            
            if output_file:
                gmsh.write(output_file)
                
            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            nodes_3d = np.array(coords).reshape(-1, 3)
            self.nodes = nodes_3d[:, :2]
            self.node_tags = node_tags
            self.zones_gdf = clean_polys
            
            self._finalize_gmsh()
            return True

        except Exception as e:
            print(f"Mesh Generation Failed: {e}")
            self._finalize_gmsh()
            raise e