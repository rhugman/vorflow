import gmsh
import sys
import math
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon

class MeshGenerator:
    def __init__(self, verbosity=2):
        self.initialized = False
        self.verbosity = verbosity
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

        # 1. Add Points
        all_point_tags = []
        for idx, row in points_gdf.iterrows():
            tag = gmsh.model.occ.addPoint(row.geometry.x, row.geometry.y, 0)
            input_tag_info[(0, tag)] = {'type': 'point', 'id': idx}
            all_point_tags.append((0, tag))
            
        # 2. Add Lines (and Straddle Ladders)
        all_line_tags = []
        all_surface_tags = [] 
        
        for idx, row in lines_gdf.iterrows():
            straddle = row.get('straddle_width')
            lc = max(row.get('lc', 10.0), 0.001)
            
            if straddle and straddle > 0:
                # --- LADDER CONSTRUCTION ---
                # 1. Resample the line at resolution 'lc'
                line = row.geometry
                length = line.length
                num_segments = int(max(1, np.ceil(length / lc)))
                
                # Generate points along the line
                distances = np.linspace(0, length, num_segments + 1)
                points = [line.interpolate(d) for d in distances]
                
                left_tags = []
                right_tags = []
                
                # 2. Create Left/Right points
                for i, p in enumerate(points):
                    # Calculate tangent/normal
                    # Use a small delta for tangent approximation
                    t_val = distances[i]
                    p_near = line.interpolate(min(t_val + 0.01, length))
                    if t_val >= length - 0.001: # End of line
                         p_near = line.interpolate(max(t_val - 0.01, 0))
                         dx, dy = p.x - p_near.x, p.y - p_near.y
                    else:
                         dx, dy = p_near.x - p.x, p_near.y - p.y
                    
                    # Normalize
                    mag = np.sqrt(dx*dx + dy*dy)
                    if mag == 0: mag = 1
                    dx, dy = dx/mag, dy/mag
                    
                    # Normal vector (-dy, dx)
                    nx, ny = -dy, dx
                    
                    # Offset points
                    w = straddle / 2.0
                    lx, ly = p.x + nx*w, p.y + ny*w
                    rx, ry = p.x - nx*w, p.y - ny*w
                    
                    lt = gmsh.model.occ.addPoint(lx, ly, 0)
                    rt = gmsh.model.occ.addPoint(rx, ry, 0)
                    left_tags.append(lt)
                    right_tags.append(rt)
                
                # 3. Build Quad Patches
                # We build a series of 4-sided surfaces connecting the points
                for i in range(len(left_tags) - 1):
                    p1, p2 = left_tags[i], left_tags[i+1]
                    p3, p4 = right_tags[i+1], right_tags[i]
                    
                    l1 = gmsh.model.occ.addLine(p1, p2) # Left edge
                    l2 = gmsh.model.occ.addLine(p2, p3) # Cap/Rung
                    l3 = gmsh.model.occ.addLine(p3, p4) # Right edge
                    l4 = gmsh.model.occ.addLine(p4, p1) # Cap/Rung
                    
                    cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
                    s = gmsh.model.occ.addPlaneSurface([cl])
                    
                    all_surface_tags.append((2, s))
                    input_tag_info[(2, s)] = {'type': 'straddle_surf', 'id': idx}
                    
                    # Track the outer edges (l1, l3) as lines for resolution control
                    input_tag_info[(1, l1)] = {'type': 'line', 'id': idx}
                    input_tag_info[(1, l3)] = {'type': 'line', 'id': idx}
                    all_line_tags.extend([(1, l1), (1, l3), (1, l2), (1, l4)])

            else:
                # Standard Line
                coords = list(row.geometry.coords)
                pt_tags = [gmsh.model.occ.addPoint(x, y, 0) for x, y in coords]
                for i in range(len(pt_tags) - 1):
                    l = gmsh.model.occ.addLine(pt_tags[i], pt_tags[i+1])
                    all_line_tags.append((1, l))
                    input_tag_info[(1, l)] = {'type': 'line', 'id': idx}

        # 3. Add Polygons
        for idx, row in polygons_gdf.iterrows():
            ext = list(row.geometry.exterior.coords)[:-1]
            pt_tags = [gmsh.model.occ.addPoint(x, y, 0) for x, y in ext]
            l_tags = []
            for i in range(len(pt_tags)):
                l = gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i+1)%len(pt_tags)])
                l_tags.append(l)
            
            cl = gmsh.model.occ.addCurveLoop(l_tags)
            s = gmsh.model.occ.addPlaneSurface([cl])
            
            all_surface_tags.append((2, s))
            input_tag_info[(2, s)] = {'type': 'polygon', 'id': idx}

        # 4. Fragment
        object_tags = all_surface_tags + all_line_tags + all_point_tags
        out_dt, out_map = gmsh.model.occ.fragment(object_tags, [])
        gmsh.model.occ.synchronize()
        
        # 5. Reconstruct Map
        final_map = {'points': {}, 'lines': {}, 'surfaces': {}, 'straddle_surfs': {}}
        for i, input_dimtag in enumerate(object_tags):
            if input_dimtag in input_tag_info:
                info = input_tag_info[input_dimtag]
                res_tags = out_map[i]
                kind = info['type']
                feat_id = info['id']
                
                target_dict = None
                if kind == 'point': target_dict = final_map['points']
                elif kind == 'line': target_dict = final_map['lines']
                elif kind == 'polygon': target_dict = final_map['surfaces']
                elif kind == 'straddle_surf': target_dict = final_map['straddle_surfs']
                
                if target_dict is not None:
                    if feat_id not in target_dict: target_dict[feat_id] = []
                    tags_only = [t for d, t in res_tags if d == input_dimtag[0]]
                    target_dict[feat_id].extend(tags_only)

        return final_map
    
# ...existing code...
    def _setup_fields(self, gmsh_map, polygons_gdf, lines_gdf, points_gdf):
        """
        Sets up fields using the robust tag map.
        """
        field_list = []
        
        # Determine global background size
        global_max_lc = 100.0
        if 'lc' in polygons_gdf.columns and not polygons_gdf.empty:
            global_max_lc = float(polygons_gdf['lc'].max())
        if global_max_lc <= 0: global_max_lc = 100.0

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

        def add_refinement(entity_dim, entity_tags, size_target, dist_min, dist_max):
            if not entity_tags: return None
            valid_tags = [float(t) for t in entity_tags]
            
            # SANITIZATION: Ensure dist_max is logically valid
            # 1. dist_max must be > dist_min
            if dist_max <= dist_min:
                # If invalid, push dist_max out to allow at least some gradation
                # or assume the user meant "width" and add it to min
                dist_max = dist_min + max(size_target, 1e-3)
            
            f_dist = gmsh.model.mesh.field.add("Distance")
            if entity_dim == 0: gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", valid_tags)
            elif entity_dim == 1: gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", valid_tags)
            
            f_thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", float(size_target))
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", float(global_max_lc))
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

        # 2. Lines (and Straddle Edges)
        for idx, row in lines_gdf.iterrows():
            if idx in gmsh_map['lines']:
                tags = extract_tags(gmsh_map['lines'][idx])
                lc = max(get_row_param(row, 'lc', 10.0), 0.001)
                d_min = get_row_param(row, 'dist_min', lc * 1.0)
                # Use provided dist_max or a reasonable default
                d_max = get_row_param(row, 'dist_max', global_max_lc * 1.5)
                
                fid = add_refinement(1, tags, lc, d_min, d_max)
                if fid: field_list.append(fid)

        # 3. Polygons (Zone-specific resolution AND Gradation)
        for idx, row in polygons_gdf.iterrows():
            if idx in gmsh_map['surfaces']:
                tags = extract_tags(gmsh_map['surfaces'][idx])
                lc = get_row_param(row, 'lc', global_max_lc)
                
                # Only apply fields if this zone is finer than global
                if lc < global_max_lc:
                    # A. Internal Resolution: Constant Field Restricted to Surface
                    f_const = gmsh.model.mesh.field.add("Constant")
                    gmsh.model.mesh.field.setNumber(f_const, "VIn", lc)
                    gmsh.model.mesh.field.setNumber(f_const, "VOut", global_max_lc * 10) # Ignore outside
                    
                    f_rest = gmsh.model.mesh.field.add("Restrict")
                    gmsh.model.mesh.field.setNumber(f_rest, "IField", f_const)
                    gmsh.model.mesh.field.setNumbers(f_rest, "SurfacesList", [float(t) for t in tags])
                    field_list.append(f_rest)

                    # B. Exterior Gradation: Threshold Field on Boundary Curves
                    # This respects dist_max for polygons
                    d_max = get_row_param(row, 'dist_max', 0.0)
                    if d_max > 0:
                        # Get boundary curves for these surfaces
                        dim_tags = [(2, int(t)) for t in tags]
                        boundaries = gmsh.model.getBoundary(dim_tags, combined=True, oriented=False, recursive=False)
                        curve_tags = [b[1] for b in boundaries if b[0] == 1]
                        
                        if curve_tags:
                            d_min = get_row_param(row, 'dist_min', 0.0)
                            # Add threshold field based on these curves
                            fid_grad = add_refinement(1, curve_tags, lc, d_min, d_max)
                            if fid_grad: field_list.append(fid_grad)

        # 4. Straddle Surfaces (Transfinite Enforcement)
        for idx, tags in gmsh_map.get('straddle_surfs', {}).items():
            clean_tags = extract_tags(tags)
            for s_tag in clean_tags:
                # Check topology: Get all boundary points
                bnd_pts = gmsh.model.getBoundary([(2, s_tag)], combined=False, oriented=False, recursive=True)
                unique_corners = set(t for d, t in bnd_pts if d == 0)
                
                if len(unique_corners) == 4:
                    try:
                        gmsh.model.mesh.setTransfiniteSurface(s_tag)
                        gmsh.model.mesh.setRecombine(2, s_tag)
                    except Exception:
                        gmsh.model.mesh.setRecombine(2, s_tag)
                else:
                    gmsh.model.mesh.setRecombine(2, s_tag)

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
        
        gmsh.option.setNumber("Mesh.Algorithm", 5) 
# ...existing code...

    def generate(self, clean_polys, clean_lines, clean_points, output_file=None):
        self._initialize_gmsh()
        try:
            print("Transferring Geometry to Gmsh...")
            gmsh_map = self._add_geometry(clean_polys, clean_lines, clean_points)
            
            print("Setting up Resolution Fields...")
            self._setup_fields(gmsh_map, clean_polys, clean_lines, clean_points)
            
            print("Generating Triangular Mesh...")
            gmsh.model.mesh.generate(2)
            
            print("Optimizing Mesh Quality...")
            gmsh.model.mesh.optimize("Netgen")
            
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