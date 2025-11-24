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
        self.field_ids = [] # Track field IDs for the final Min calculation
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
        Returns a mapping of shapely_index -> gmsh_tag for fields setup.
        """
        gmsh_map = {'points': {}, 'lines': {}, 'surfaces': {}}
        
        # 1. Add Points (Wells)
        for idx, row in points_gdf.iterrows():
            x, y = row.geometry.x, row.geometry.y
            # Add point to model
            tag = gmsh.model.occ.addPoint(x, y, 0)
            gmsh_map['points'][idx] = tag
            
        # 2. Add Lines (Rivers, Faults)
        for idx, row in lines_gdf.iterrows():
            # Extract coords
            coords = list(row.geometry.coords)
            pt_tags = []
            for x, y in coords:
                # Check if point exists (optimization needed here for shared nodes)
                pt_tags.append(gmsh.model.occ.addPoint(x, y, 0))
            
            # Create splines/lines between points
            line_segs = []
            for i in range(len(pt_tags) - 1):
                l_tag = gmsh.model.occ.addLine(pt_tags[i], pt_tags[i+1])
                line_segs.append(l_tag)
            
            # If multiple segments, wire them; otherwise just the line
            if len(line_segs) > 1:
                wire = gmsh.model.occ.addWire(line_segs)
                gmsh_map['lines'][idx] = wire # Store wire tag
            else:
                gmsh_map['lines'][idx] = line_segs[0]

        # 3. Add Polygons (Zones)
        # Note: We assume ConceptualMesh has already removed overlaps.
        # We just need to stitch them.
        all_surfaces = []
        for idx, row in polygons_gdf.iterrows():
            # Simplified: assumes exterior ring only for brevity. 
            # Real imp needs to handle holes (interiors).
            ext_coords = list(row.geometry.exterior.coords)
            pt_tags = [gmsh.model.occ.addPoint(x, y, 0) for x, y in ext_coords[:-1]]
            print(f"DEBUG: pt_tags value (scalar integers expected): {pt_tags}")
            l_tags = []
            for i in range(len(pt_tags)):
                p1, p2 = pt_tags[i], pt_tags[(i+1)%len(pt_tags)]
                l_tags.append(gmsh.model.occ.addLine(p1, p2))
            
            #wire = gmsh.model.occ.addWire(l_tags)
            #print(f"DEBUG: Wire tag type: {type(wire)}, value: {wire}")
            #surf = gmsh.model.occ.addPlaneSurface([wire])

            # 1. Create the Curve Loop (The boundary)
            # This function expects a list of curve tags, which l_tags is.
            curve_loop_tag = gmsh.model.occ.addCurveLoop(l_tags) 
            
            # 2. Create the Surface
            # The argument here is a list of curve loop tags.
            # CRITICAL: We pass the scalar tag inside a list.
            surf = gmsh.model.occ.addPlaneSurface([curve_loop_tag])

            gmsh_map['surfaces'][idx] = surf
            all_surfaces.append((2, surf)) # (dim, tag)

        # 4. Sync OCC with standard model
        gmsh.model.occ.synchronize()
        
        # 5. HANDLING LINE INTERPRETATIONS (Fragment vs Embed)
        # This is where we handle "Barrier" vs "Conductive"
        
        # Filter lines based on behavior
        barrier_lines = lines_gdf[lines_gdf['is_barrier'] == True]
        conductive_lines = lines_gdf[lines_gdf['is_barrier'] == False]
        # Get the list of scalar tags for all non-barrier lines
        conductive_line_tags = [gmsh_map['lines'][i] for i in conductive_lines.index]

        if conductive_line_tags:
            # Use a high number for the physical group tag to avoid conflicts
            # Group the dimension 1 entities (lines)
            gmsh.model.addPhysicalGroup(1, conductive_line_tags, tag=1000)
            gmsh.model.setPhysicalName(1, 1000, "Conductive Lines")

        # Add Physical Groups for surfaces as well (required for mesh output)
        # You should also group all final surfaces (dim 2) here.
        all_surface_tags = [tag for dim, tag in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, all_surface_tags, tag=2000)
        gmsh.model.setPhysicalName(2, 2000, "Meshed Domain")
        return gmsh_map

    def _setup_fields(self, gmsh_map, polygons_gdf, lines_gdf, points_gdf):
        """
        Sets up the resolution (Size) fields.
        Uses a 'Min' field strategy to handle overlapping resolution requests.
        """
        field_list = []
        
        # Function to add a threshold field
        def add_threshold(entity_dim, entity_tag, size_in, size_out, dist_min, dist_max):
            # 1. Distance Field
            fid_dist = gmsh.model.mesh.field.add("Distance")
            if entity_dim == 0: gmsh.model.mesh.field.setNumbers(fid_dist, "PointsList", [entity_tag])
            elif entity_dim == 1: gmsh.model.mesh.field.setNumbers(fid_dist, "CurvesList", [entity_tag])
            elif entity_dim == 2: gmsh.model.mesh.field.setNumbers(fid_dist, "SurfacesList", [entity_tag]) # Requires MathEval usually
            
            # 2. Threshold Field
            fid_thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(fid_thresh, "InField", fid_dist)
            gmsh.model.mesh.field.setNumber(fid_thresh, "SizeMin", size_in)
            gmsh.model.mesh.field.setNumber(fid_thresh, "SizeMax", size_out)
            gmsh.model.mesh.field.setNumber(fid_thresh, "DistMin", dist_min)
            gmsh.model.mesh.field.setNumber(fid_thresh, "DistMax", dist_max)
            
            return fid_thresh

        # 1. Point Resolutions (Wells)
        for idx, row in points_gdf.iterrows():
            tag = gmsh_map['points'][idx]
            # High res at well, fading out over 500m
            fid = add_threshold(0, tag, row['lc'], row['lc']*10, 1.0, 500.0)
            field_list.append(fid)

        # 2. Line Resolutions
        for idx, row in lines_gdf.iterrows():
            tag = gmsh_map['lines'][idx]
            # High res at river/fault, fading out
            fid = add_threshold(1, tag, row['lc'], row['lc']*5, 10.0, 200.0)
            field_list.append(fid)

        # 3. Polygon Resolutions (Constant inside zone)
        # Using "MathEval" for polygons is often easier than Distance
        for idx, row in polygons_gdf.iterrows():
            tag = gmsh_map['surfaces'][idx]
            fid = gmsh.model.mesh.field.add("MathEval")
            # Create a string function that restricts this field to the surface
            # (This is simplified; typically we set mesh size on points/corners for zones)
            # A robust way is "Restrict" field in Gmsh
            
            # Alternative: Just set point mesh sizes if field logic is too heavy
            # But for field consistency:
            gmsh.model.mesh.field.setString(fid, "F", f"{row['lc']}")
            
            # Restrict application of this field to the specific surface
            fid_restrict = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(fid_restrict, "InField", fid)
            gmsh.model.mesh.field.setNumbers(fid_restrict, "SurfacesList", [tag])
            field_list.append(fid_restrict)

        # 4. Combine with Min Field
        if field_list:
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_list)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        
        # Important: Mesh Options
        gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal-Delaunay for 2D

    def generate(self, clean_polys, clean_lines, clean_points, output_file=None):
        self._initialize_gmsh()
        
        try:
            print("Transferring Geometry to Gmsh...")
            mapping = self._add_geometry(clean_polys, clean_lines, clean_points)
            
            print("Setting up Resolution Fields...")
            self._setup_fields(mapping, clean_polys, clean_lines, clean_points)
            
            print("Generating Triangular Mesh...")
            gmsh.model.mesh.generate(2)
            
            print("Optimizing Mesh (Lloyd)...")
            gmsh.option.setNumber("Mesh.Optimize", 1)
            #gmsh.option.setNumber("Mesh.Lloyd", 10) # 10 iterations of smoothing
            gmsh.model.mesh.optimize("")
            
            if output_file:
                gmsh.write(output_file)
                
            # Extract the nodes (generators)
            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            nodes_3d = np.array(coords).reshape(-1, 3)
            self.nodes = nodes_3d[:, :2]
            self.node_tags = node_tags
            
            # Store the zone constraints for the tesselator
            self.zones_gdf = clean_polys[['geometry', 'zone_id', 'z_order']].copy()
            
            # Finalize Gmsh Kernel Safely
            gmsh.finalize() 
            self.initialized = False # Update internal state            
            return True

        except Exception as e:
            print(f"Mesh Generation Failed: {e}")
            self._finalize_gmsh()
            return False