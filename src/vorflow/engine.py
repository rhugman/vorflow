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
            gmsh.model.add("mesh_model")
            self.initialized = True

    def _finalize_gmsh(self):
        if gmsh.is_initialized():
            gmsh.finalize()
            self.initialized = False

    def _add_geometry(self, polygons_gdf, lines_gdf, points_gdf):
        """
        Transfers Shapely geometry to Gmsh OCC kernel.
        Strategy: Fragment EVERYTHING.
        This resolves all intersections between barriers, rivers, and zone boundaries.
        """
        gmsh_map = {'points': {}, 'lines': {}, 'surfaces': {}}
        
        # 1. Add Points (Wells)
        for idx, row in points_gdf.iterrows():
            x, y = row.geometry.x, row.geometry.y
            tag = gmsh.model.occ.addPoint(x, y, 0)
            gmsh_map['points'][idx] = tag
            
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
            # Exterior
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
        # We cut the surfaces with ALL lines (Barriers AND Conductive).
        # This ensures that if a river crosses a fault, the geometry is correctly split.
        # This places nodes on ALL lines, which is required for both cases.
        
        object_tags = initial_surface_tags
        tool_tags = [(1, t) for t in all_line_tags]
        
        # Remove duplicates before fragmenting to clean up dirty input
        gmsh.model.occ.removeAllDuplicates()
        
        out_dt, out_map = gmsh.model.occ.fragment(object_tags, tool_tags)
        
        # The valid surfaces are now the result of the fragment operation
        final_surface_tags = [tag for dim, tag in out_dt if dim == 2]

        # 5. Synchronize
        gmsh.model.occ.synchronize()

        # 6. Embed Points (Wells)
        # Points must still be embedded into the new surfaces
        point_tags = [tag for tag in gmsh_map['points'].values()]
        if point_tags:
            # Embed in all surfaces (robust)
            for s_tag in final_surface_tags:
                try:
                    gmsh.model.mesh.embed(0, point_tags, 2, s_tag)
                except:
                    pass

        # 7. Physical Groups
        gmsh.model.addPhysicalGroup(2, final_surface_tags, tag=2000)
        gmsh.model.setPhysicalName(2, 2000, "Meshed Domain")
        
        return gmsh_map, final_surface_tags
    
    
    def _setup_fields(self, gmsh_map, polygons_gdf, lines_gdf, points_gdf):
        """
        Sets up the resolution (Size) fields.
        Uses a 'Min' field strategy to handle overlapping resolution requests.
        """
        field_list = []
        
        # Function to add a threshold field
        def add_refinement(entity_dim, entity_tags, size_target, dist_min, dist_max):
            # Safety: Ensure entity_tags is a list of standard python floats
            if not isinstance(entity_tags, list):
                entity_tags = [entity_tags]
            # Explicit cast to float/int for Gmsh API
            entity_tags = [t for t in entity_tags]
            
            # 1. Distance Field
            f_dist = gmsh.model.mesh.field.add("Distance")
            if entity_dim == 0: 
                gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", entity_tags)
            elif entity_dim == 1: 
                gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", entity_tags)
            
            # 2. Threshold Field
            f_thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", float(size_target))
            # SizeMax is set very high so this field doesn't artificially constrain 
            # the mesh far away. The global background field will cap it.
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", 1e22)
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", float(dist_min))
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", float(dist_max))
            
            return f_thresh

        # 1. Point Resolutions (Wells)
        for idx, row in points_gdf.iterrows():
            if idx in gmsh_map['points']:
                tag = gmsh_map['points'][idx]
                lc = max(row.get('lc', 5.0), 0.001) # Clamp to avoid 0
                fid = add_refinement(0, [tag], lc, 1.0, 500.0)
                field_list.append(fid)

        # 2. Line Resolutions
        for idx, row in lines_gdf.iterrows():
            if idx in gmsh_map['lines']:
                tags = gmsh_map['lines'][idx]
                lc = max(row.get('lc', 10.0), 0.001)
                fid = add_refinement(1, tags, lc, 5.0, 200.0)
                field_list.append(fid)

        # 3. Global Background Field
        f_bg = gmsh.model.mesh.field.add("MathEval")
        bg_size = 100.0
        if 'lc' in polygons_gdf.columns and not polygons_gdf.empty:
            bg_size = float(polygons_gdf['lc'].max())
            
        gmsh.model.mesh.field.setString(f_bg, "F", str(bg_size))
        field_list.append(f_bg)

        # 4. Combine with Min Field
        if field_list:
            min_field = gmsh.model.mesh.field.add("Min")
            field_list = [float(f) for f in field_list]
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_list)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        
        # Mesh Options - Algo 5 (Delaunay) is often more robust for complex fields than 6
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
            
            print("Optimizing Mesh...")
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.model.mesh.optimize( niter=5)
            
            if output_file:
                gmsh.write(output_file)
                
            # Extract nodes
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