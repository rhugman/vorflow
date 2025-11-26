import gmsh
import sys
import math
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon

class MeshGenerator:
    def __init__(self, background_lc, verbosity=2):
        self.initialized = False
        self.verbosity = verbosity
        self.background_lc = float(background_lc)
        self.nodes = None
        self.node_tags = None
        self.zones_gdf = None
    
    
    def _initialize_gmsh(self):
        if not gmsh.is_initialized():
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", self.verbosity)
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
        Uses Explicit Ladder Construction for Straddle Lines.
        """
        input_tag_info = {} 
        
        # Helper to ensure tags are standard python ints for dict keys
        def to_key(dim, tag):
            return (int(dim), int(tag))
        
        # 1. Add Points
        all_point_tags = []
        for idx, row in points_gdf.iterrows():
            tag = gmsh.model.occ.addPoint(row.geometry.x, row.geometry.y, 0)
            key = to_key(0, tag)
            input_tag_info[key] = {'type': 'point', 'id': idx}
            all_point_tags.append(key)
            
        # 2. Add Lines
        all_line_tags = []
        all_surface_tags = [] 
        
        for idx, row in lines_gdf.iterrows():
            # Trigger Virtual Straddle if 'straddle_width' is set OR 'is_barrier' is True
            straddle = row.get('straddle_width')
            is_barrier = row.get('is_barrier', False)
            lc = max(row.get('lc', 10.0), 0.001)
            
            use_virtual_straddle = is_barrier or (straddle is not None and straddle > 0)
            
            if use_virtual_straddle:
                if self.verbosity > 1:
                    print(f"Line {idx}: Virtual Straddle Active (Barrier={is_barrier}, Width={straddle})")

                # --- VIRTUAL STRADDLE (Point Pairs) ---
                line = row.geometry
                length = line.length
                num_segments = int(max(1, np.ceil(length / lc)))
                distances = np.linspace(0, length, num_segments + 1)
                
                # Determine offset distance (epsilon)
                if straddle and straddle > 0:
                    # Physical width specified
                    epsilon = straddle / 2.0
                else:
                    # Virtual Barrier: Use a stable fraction of LC
                    # 0.01 was too small (optimizer collapsed it).
                    # 0.20 (20%) creates a 1:2.5 aspect ratio, which survives optimization.
                    epsilon = lc * 0.20
                
                for d in distances:
                    p = line.interpolate(d)
                    
                    # Calculate Normal
                    t_val = d
                    p_near = line.interpolate(min(t_val + 0.01, length))
                    if t_val >= length - 0.001:
                         p_near = line.interpolate(max(t_val - 0.01, 0))
                         dx, dy = p.x - p_near.x, p.y - p_near.y
                    else:
                         dx, dy = p_near.x - p.x, p_near.y - p.y
                    
                    mag = np.sqrt(dx*dx + dy*dy)
                    if mag == 0: mag = 1
                    dx, dy = dx/mag, dy/mag
                    nx, ny = -dy, dx
                    
                    # Add Pair of Hard Points
                    # Left Node
                    lx, ly = p.x + nx*epsilon, p.y + ny*epsilon
                    lt = gmsh.model.occ.addPoint(lx, ly, 0)
                    k_l = to_key(0, lt)
                    input_tag_info[k_l] = {'type': 'point', 'id': idx}
                    all_point_tags.append(k_l)
                    
                    # Right Node
                    rx, ry = p.x - nx*epsilon, p.y - ny*epsilon
                    rt = gmsh.model.occ.addPoint(rx, ry, 0)
                    k_r = to_key(0, rt)
                    input_tag_info[k_r] = {'type': 'point', 'id': idx}
                    all_point_tags.append(k_r)
                
                # Note: We do NOT add the line curve itself to Gmsh.

            else:
                # --- STANDARD CONSTRAINT --
                # Nodes are placed ON the line.
                coords = list(row.geometry.coords)
                pt_tags = [gmsh.model.occ.addPoint(x, y, 0) for x, y in coords]
                for i in range(len(pt_tags) - 1):
                    l = gmsh.model.occ.addLine(pt_tags[i], pt_tags[i+1])
                    
                    key = to_key(1, l)
                    all_line_tags.append(key)
                    input_tag_info[key] = {'type': 'line', 'id': idx}

        # 3. Add Polygons
        if not polygons_gdf.empty:
            print(f"Adding {len(polygons_gdf)} polygons to Gmsh...")
            for idx, row in polygons_gdf.iterrows():
                geom = row['geometry']
                if geom.geom_type == 'Polygon':
                    polys = [geom]
                elif geom.geom_type == 'MultiPolygon':
                    polys = geom.geoms
                else:
                    continue
                
                for poly in polys:
                    # Exterior
                    ext_coords = list(poly.exterior.coords)
                    # Create Loop
                    p_tags = []
                    for x, y in ext_coords[:-1]: # Skip duplicate end
                        p_tags.append(gmsh.model.occ.addPoint(x, y, 0))
                    
                    l_tags = []
                    for i in range(len(p_tags)):
                        p1 = p_tags[i]
                        p2 = p_tags[(i + 1) % len(p_tags)]
                        l_tags.append(gmsh.model.occ.addLine(p1, p2))
                    
                    cl_tag = gmsh.model.occ.addCurveLoop(l_tags)
                    s_tag = gmsh.model.occ.addPlaneSurface([cl_tag])
                    
                    # FIX: Use to_key for Polygons (Already present, but kept for consistency)
                    key = to_key(2, s_tag)
                    input_tag_info[key] = {'type': 'surface', 'id': idx}
                    all_surface_tags.append(key)

        # 4. Fragment
        object_tags = all_surface_tags + all_line_tags + all_point_tags 
        
        if not object_tags:
            print("Warning: No geometry to mesh.")
            return {'points': {}, 'lines': {}, 'surfaces': {}, 'straddle_surfs': {}}

        print(f"Fragmenting {len(object_tags)} objects...")
        out_dt, out_map = gmsh.model.occ.fragment(object_tags, [])
        gmsh.model.occ.synchronize()
        
        # 5. Reconstruct Map
        final_map = {'points': {}, 'lines': {}, 'surfaces': {}, 'straddle_surfs': {}}
        
        print(f"Reconstructing Map (Input Tags: {len(object_tags)}, Out Map Len: {len(out_map)})...")
        
        for i, input_dimtag in enumerate(object_tags):
            if i < len(out_map):
                res_tags = out_map[i]
            else:
                res_tags = [input_dimtag]

            # Lookup using to_key
            key = to_key(input_dimtag[0], input_dimtag[1])
            
            if key in input_tag_info:
                info = input_tag_info[key]
                kind = info['type']
                feat_id = info['id']
                
                if kind == 'point':
                    # FIX: Use extend to avoid overwriting if multiple features share an ID
                    # (e.g. a Line generating points and a Point having the same index)
                    if feat_id not in final_map['points']:
                        final_map['points'][feat_id] = []
                    final_map['points'][feat_id].extend(res_tags)
                    
                elif kind == 'line':
                    final_map['lines'][feat_id] = res_tags
                elif kind == 'surface':
                    final_map['surfaces'][feat_id] = res_tags
                elif kind == 'straddle_surf':
                    if feat_id not in final_map['straddle_surfs']:
                        final_map['straddle_surfs'][feat_id] = []
                    final_map['straddle_surfs'][feat_id].extend(res_tags)
            else:
                print(f"Warning: Tag {key} lost during fragmentation mapping.")

        return final_map
    
# ...existing code...
    def _setup_fields(self, gmsh_map, polygons_gdf, lines_gdf, points_gdf):
        """
        Sets up fields using the robust tag map.
        """
        # --- DIAGNOSTIC START ---
        if self.verbosity > 0:
            print(f"--- Setup Fields Debug ---")
            print(f"Polygons GDF: {len(polygons_gdf)} rows")
            print(f"Gmsh Surface Map: {len(gmsh_map.get('surfaces', {}))} entries")
            if not polygons_gdf.empty:
                first_idx = polygons_gdf.index[0]
                print(f"First Poly Index: {first_idx} (Type: {type(first_idx)})")
                if gmsh_map['surfaces']:
                    first_key = list(gmsh_map['surfaces'].keys())[0]
                    print(f"First Map Key: {first_key} (Type: {type(first_key)})")
                    print(f"Match? {first_idx in gmsh_map['surfaces']}")
                else:
                    print("Gmsh Surface Map is EMPTY.")
        # --- DIAGNOSTIC END ---

        field_list = []
        
        # Determine global background size from instance variable
        global_max_lc = self.background_lc

        def extract_tags(entry_list):
            """Helper to handle both [(dim, tag)] and [tag] formats."""
            clean_tags = []
            for item in entry_list:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    clean_tags.append(item[1])
                else:
                    clean_tags.append(item)
            return clean_tags

        def get_row_param(row, key, default):
            if key in row and not pd.isna(row[key]): return float(row[key])
            return float(default)

        def add_refinement(entity_dim, entity_tags, size_target, dist_min, dist_max, size_max_limit=None):
            if not entity_tags: return None
            valid_tags = [float(t) for t in entity_tags]
            
            # Default upper limit is global background
            if size_max_limit is None:
                size_max_limit = global_max_lc

            # 1. GRADIENT LIMITER (Crucial for Convergence)
            # The mesher struggles if size grows faster than ~1.5x per element.
            # Min Distance ~= (SizeMax - SizeMin) / (GrowthRate * SizeMin)
            # We enforce a conservative max slope to prevent stalling.
            
            size_diff = size_max_limit - size_target
            if size_diff > 0:
                # Max allowed slope: grow by 50% of current size per unit distance
                # This is a heuristic to keep the mesh quality high and generation fast.
                min_span_required = size_diff / (0.5 * size_target)
                
                # Ensure dist_max provides enough room
                current_span = dist_max - dist_min
                if current_span < min_span_required:
                    # Extend dist_max to satisfy gradient limit
                    dist_max = dist_min + min_span_required

            # Sanity check
            if dist_max <= dist_min:
                dist_max = dist_min + max(size_target, 1e-3)
            
            # 2. Create Distance Field
            f_dist = gmsh.model.mesh.field.add("Distance")
            if entity_dim == 0: gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", valid_tags)
            elif entity_dim == 1: gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", valid_tags)
            
            # 3. Use Native Threshold (Linear) for Efficiency
            # MathEval (Exponential) is too slow for large meshes. 
            # The Gradient Limiter above ensures this Linear field is steep enough to be 
            # efficient but smooth enough to converge quickly.
            
            f_thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", float(size_target))
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", float(size_max_limit))
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", float(dist_min))
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", float(dist_max))
            
            return f_thresh

       # 1. Points
        for idx, row in points_gdf.iterrows():
            if idx in gmsh_map['points']:
                tags = extract_tags(gmsh_map['points'][idx])
                lc = max(get_row_param(row, 'lc', 5.0), 0.001)
                d_min = get_row_param(row, 'dist_min', lc * 2.0)
                d_max = get_row_param(row, 'dist_max', global_max_lc * 1.5)
                
                fid = add_refinement(0, tags, lc, d_min, d_max)
                if fid: field_list.append(fid)

        # 2. Lines (and Straddle Barriers)
        for idx, row in lines_gdf.iterrows():
            # Case A: Standard Lines (Curves exist in Gmsh)
            if idx in gmsh_map['lines']:
                tags = extract_tags(gmsh_map['lines'][idx])
                lc = max(get_row_param(row, 'lc', 10.0), 0.001)
                d_min = get_row_param(row, 'dist_min', lc * 1.0)
                d_max = get_row_param(row, 'dist_max', global_max_lc * 1.5)
                
                fid = add_refinement(1, tags, lc, d_min, d_max)
                if fid: field_list.append(fid)
            
            # Case B: Virtual Barriers (Only Points exist in Gmsh)
            # We must refine around these points to resolve the straddle gap
            elif idx in gmsh_map['points']:
                # Verify it is actually a barrier/straddle line
                is_barrier = row.get('is_barrier', False)
                straddle = row.get('straddle_width', 0)
                if is_barrier or (straddle and straddle > 0):
                    tags = extract_tags(gmsh_map['points'][idx])
                    lc = max(get_row_param(row, 'lc', 10.0), 0.001)
                    
                    # Refine around the straddle points
                    d_min = get_row_param(row, 'dist_min', lc * 2.0)
                    d_max = get_row_param(row, 'dist_max', global_max_lc * 1.5)
                    
                    # Apply Point Refinement (dim 0)
                    fid = add_refinement(0, tags, lc, d_min, d_max)
                    if fid: field_list.append(fid)

        # 3. Polygons (Border Density -> Interior Gradation)
        for idx, row in polygons_gdf.iterrows():
            if idx in gmsh_map['surfaces']:
                tags = extract_tags(gmsh_map['surfaces'][idx])
                
                # 1. Determine Targets
                target_lc = get_row_param(row, 'lc', global_max_lc)
                border_dens = get_row_param(row, 'border_density', target_lc)
                boundary_lc = min(target_lc, border_dens)

                if self.verbosity > 1:
                    print(f"Poly {idx}: Target={target_lc}, Border={boundary_lc}, Global={global_max_lc}")

                # 2. Setup Interior Field (Restricted to Surface)
                if boundary_lc < global_max_lc or target_lc < global_max_lc:
                    
                    dim_tags = [(2, int(t)) for t in tags]
                    boundaries = gmsh.model.getBoundary(dim_tags, combined=True, oriented=False, recursive=False)
                    curve_tags = [b[1] for b in boundaries if b[0] == 1]
                    
                    f_inner = None
                    
                    if curve_tags and boundary_lc < target_lc:
                        # GRADATION: Border is finer than Interior.
                        d_min = get_row_param(row, 'dist_min', 0.0)
                        
                        # Use specific internal max distance
                        d_max_in = get_row_param(row, 'dist_max_in', -1.0)
                        
                        if d_max_in > d_min:
                            d_max_inner = d_max_in
                        else:
                            # HEURISTIC: 5x Boundary Elements
                            d_max_inner = d_min + (boundary_lc * 5.0)
                            d_max_inner = max(d_max_inner, d_min + (target_lc - boundary_lc) * 0.2)
                        
                        if self.verbosity > 1:
                            print(f"  -> Grading Interior: DistMin={d_min}, DistMax={d_max_inner}, SizeMax={target_lc}")

                        f_inner = add_refinement(1, curve_tags, boundary_lc, d_min, d_max_inner, size_max_limit=target_lc)
                        
                    else:
                        # CONSTANT: Uniform interior
                        if self.verbosity > 1:
                            print(f"  -> Constant Interior: Size={target_lc}")
                        
                        # WORKAROUND: Use a Threshold field that is "always close"
                        # Sometimes Constant fields behave oddly with Min in older Gmsh versions
                        # We create a Distance field to the boundary curves
                        f_dist_inner = gmsh.model.mesh.field.add("Distance")
                        gmsh.model.mesh.field.setNumbers(f_dist_inner, "CurvesList", curve_tags)
                        
                        f_const = gmsh.model.mesh.field.add("Threshold")
                        gmsh.model.mesh.field.setNumber(f_const, "InField", f_dist_inner)
                        gmsh.model.mesh.field.setNumber(f_const, "SizeMin", target_lc)
                        gmsh.model.mesh.field.setNumber(f_const, "SizeMax", target_lc) # Force constant
                        gmsh.model.mesh.field.setNumber(f_const, "DistMin", 1e22)      # Always "close"
                        gmsh.model.mesh.field.setNumber(f_const, "DistMax", 1e22)
                        
                        f_inner = f_const

                    # Restrict the chosen field to the Polygon Surface
                    if f_inner:
                        f_rest = gmsh.model.mesh.field.add("Restrict")
                        gmsh.model.mesh.field.setNumber(f_rest, "IField", f_inner)
                        gmsh.model.mesh.field.setNumbers(f_rest, "SurfacesList", [float(t) for t in tags])
                        field_list.append(f_rest)

                # 3. Setup Exterior Field (Gradation to Global)
                # Use specific external max distance
                d_max_out = get_row_param(row, 'dist_max_out', 0.0)
                
                if d_max_out > 0:
                    dim_tags = [(2, int(t)) for t in tags]
                    boundaries = gmsh.model.getBoundary(dim_tags, combined=True, oriented=False, recursive=False)
                    curve_tags = [b[1] for b in boundaries if b[0] == 1]
                    
                    if curve_tags:
                        d_min = get_row_param(row, 'dist_min', 0.0)
                        if self.verbosity > 1:
                            print(f"  -> Grading Exterior: DistMax={d_max_out}")
                        
                        # Grades from boundary_lc -> Global Background
                        fid_grad = add_refinement(1, curve_tags, boundary_lc, d_min, d_max_out, size_max_limit=global_max_lc)
                        if fid_grad: field_list.append(fid_grad)

        # 4. Straddle Surfaces (Transfinite Enforcement)
        for idx, tags in gmsh_map.get('straddle_surfs', {}).items():
            clean_tags = extract_tags(tags)
            for s_tag in clean_tags:
                # ...existing code...
                pass

        # 5. Global Background
        f_bg = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(f_bg, "F", str(global_max_lc))
        field_list.append(f_bg)

        # 6. Combine all fields using Min
        if field_list:
            min_field = gmsh.model.mesh.field.add("Min")
            field_list = [float(f) for f in field_list]
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_list)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        
        # KEY FIX: Disable default boundary extension so our fields have full control
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        gmsh.option.setNumber("Mesh.Algorithm", 5) 


    def generate(self, clean_polys, clean_lines, clean_points, output_file=None):
        self._initialize_gmsh()
        try:
            print("Transferring Geometry to Gmsh...")
            gmsh_map = self._add_geometry(clean_polys, clean_lines, clean_points)
            
            print("Setting up Resolution Fields...")
            self._setup_fields(gmsh_map, clean_polys, clean_lines, clean_points)
            
            
            # 2. Smoothing: Applies Laplacian smoothing to internal nodes.
            # This relaxes the mesh, making triangles more equilateral.
            # Equilateral triangles -> Compact, Hexagonal Voronoi cells -> Lower Drift.
            gmsh.option.setNumber("Mesh.Smoothing", 10) 

            print("Generating Triangular Mesh...")
            gmsh.model.mesh.generate(2)
            # NEW: Explicit Optimization Passes
            if self.verbosity > 0:
                print("Optimizing Mesh (Relocate2D & Laplace2D)...")
            
            # Relocate2D: Moves nodes to improve element quality (Compactness)
            gmsh.model.mesh.optimize("Relocate2D",niter=100)
            
            # Laplace2D: Smooths the mesh to relax gradients (Drift reduction)
            gmsh.model.mesh.optimize("Laplace2D",niter=100)

            
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