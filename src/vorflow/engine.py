import gmsh
import sys
import math
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

class MeshGenerator:
    def __init__(self, background_lc=None,verbosity=0, mesh_algorithm=6, smoothing_steps=10, optimization_cycles=2):
        """
        Initializes the Gmsh-based mesh generator.

        This class is responsible for taking clean geometric inputs and using Gmsh
        to produce a high-quality triangular mesh.

        Args:
            background_lc (float, optional): The default target mesh size for areas
                not controlled by a specific refinement field.
            verbosity (int): Gmsh verbosity level (0=silent, 1=basic, 2=debug).
            mesh_algorithm (int): The 2D mesh algorithm to use. Common choices are
                5 (Delaunay) for speed or 6 (Frontal-Delaunay) for quality.
            smoothing_steps (int): Number of internal Lloyd smoothing iterations
                performed by Gmsh during mesh generation.
            optimization_cycles (int): Number of explicit optimization passes
                (e.g., Relocate2D, Laplace2D) to run after the initial mesh is generated.
        """
        self.background_lc = background_lc
        self.verbosity = verbosity
        self.mesh_algorithm = mesh_algorithm
        self.smoothing_steps = smoothing_steps
        self.optimization_cycles = optimization_cycles

        self.initialized = False
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
        Transfers Shapely geometries from GeoDataFrames into the Gmsh model.

        This method adds points, lines, and polygons to Gmsh's internal CAD
        kernel (OCC). It also handles special cases like "straddle" lines and
        pre-processes barrier features before fragmenting all geometries to
        create a consistent topological model.
        """
        input_tag_info = {} 
        
        def to_key(dim, tag):
            return (int(dim), int(tag))
        
        # Add all point features to the Gmsh model first.
        all_point_tags = []
        for idx, row in points_gdf.iterrows():
            tag = gmsh.model.occ.addPoint(row.geometry.x, row.geometry.y, 0)
            key = to_key(0, tag)
            input_tag_info[key] = {'type': 'point', 'id': idx}
            all_point_tags.append(key)
            
        # Create a buffer zone around barrier lines. This is used to trim back
        # other lines, preventing their endpoints from interfering with the
        # meshing of the barrier features.
        barrier_buffers = []
        for idx, row in lines_gdf.iterrows():
            val = row.get('is_barrier', False)
            is_barrier = (val is True) or (str(val).lower() in ['true', '1', 'yes'])
            straddle = row.get('straddle_width')
            
            if is_barrier: 
                lc = max(row.get('lc', 10.0), 0.001)
                if straddle and straddle > 0:
                    eps = straddle / 2.0
                else:
                    eps = lc * 0.20
                
                # The trim buffer is made slightly larger than the feature's half-width
                # to ensure a clean separation between standard lines and the
                # sensitive node pairs used for straddle barriers.
                trim_eps = eps * 1.20
                
                buf = row.geometry.buffer(trim_eps, cap_style=2)
                barrier_buffers.append(buf)
        
        barrier_zone = None
        if barrier_buffers:
            barrier_zone = unary_union(barrier_buffers)
            barrier_zone = make_valid(barrier_zone)
            if self.verbosity > 0:
                print(f"Constructed Barrier Zone from {len(barrier_buffers)} barriers.")

        # Add line features to the model, handling barriers and standard lines differently.
        all_line_tags = []
        all_surface_tags = [] 
        
        for idx, row in lines_gdf.iterrows():
            val = row.get('is_barrier', False)
            is_barrier = (val is True) or (str(val).lower() in ['true', '1', 'yes'])
            straddle = row.get('straddle_width')
            lc = max(row.get('lc', 10.0), 0.001)
            
            use_virtual_straddle = is_barrier or (straddle is not None and straddle > 0)
            
            if use_virtual_straddle:
                # For barriers or "straddle" lines, we don't add the line itself.
                # Instead, we place pairs of points along the line's path. These
                # points will become nodes in the triangular mesh, forcing the
                # subsequent Voronoi cell edges to align with the original line.
                line = row.geometry
                length = line.length
                num_segments = int(max(1, np.ceil(length / lc)))
                distances = np.linspace(0, length, num_segments + 1)
                
                if straddle and straddle > 0:
                    epsilon = straddle / 2.0
                else:
                    epsilon = lc * 0.20
                
                for d in distances:
                    p = line.interpolate(d)
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
                    
                    # Create two points, offset from the original line by the normal.
                    lx, ly = p.x + nx*epsilon, p.y + ny*epsilon
                    lt = gmsh.model.occ.addPoint(lx, ly, 0)
                    k_l = to_key(0, lt)
                    input_tag_info[k_l] = {'type': 'point', 'id': idx}
                    all_point_tags.append(k_l)
                    
                    rx, ry = p.x - nx*epsilon, p.y - ny*epsilon
                    rt = gmsh.model.occ.addPoint(rx, ry, 0)
                    k_r = to_key(0, rt)
                    input_tag_info[k_r] = {'type': 'point', 'id': idx}
                    all_point_tags.append(k_r)

            else:
                # This is a standard line feature that will act as a constraint
                # in the mesh, but not a hard barrier.
                geom = row.geometry
                
                # Trim the line against the barrier zone to avoid intersections.
                if barrier_zone:
                    if geom.intersects(barrier_zone):
                        try:
                            original_len = geom.length
                            geom = geom.difference(barrier_zone)
                            
                            if self.verbosity > 1:
                                print(f"  Line {idx} trimmed by barrier (Len: {original_len:.2f} -> {geom.length:.2f})")
                                
                        except Exception as e:
                            print(f"Warning: Failed to trim line {idx}: {e}")
                
                if geom.is_empty:
                    continue
                
                # A line might be split into multiple parts after being trimmed.
                if geom.geom_type == 'LineString':
                    parts = [geom]
                elif geom.geom_type == 'MultiLineString':
                    parts = geom.geoms
                else:
                    parts = []
                
                for part in parts:
                    # Filter out tiny fragments that might remain after trimming.
                    if part.length < 1e-6: continue
                    
                    coords = list(part.coords)
                    if len(coords) < 2: continue
                    
                    # Add each segment of the line to Gmsh.
                    pt_tags = [gmsh.model.occ.addPoint(x, y, 0) for x, y in coords]
                    for i in range(len(pt_tags) - 1):
                        l = gmsh.model.occ.addLine(pt_tags[i], pt_tags[i+1])
                        
                        key = to_key(1, l)
                        all_line_tags.append(key)
                        input_tag_info[key] = {'type': 'line', 'id': idx}

        # Add polygon features to the model.
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
                    # Define the exterior boundary of the polygon.
                    ext_coords = list(poly.exterior.coords)
                    p_tags = []
                    for x, y in ext_coords[:-1]: # Skip duplicate end point.
                        p_tags.append(gmsh.model.occ.addPoint(x, y, 0))
                    
                    # Create the line segments forming the boundary.
                    l_tags = []
                    for i in range(len(p_tags)):
                        p1 = p_tags[i]
                        p2 = p_tags[(i + 1) % len(p_tags)]
                        l_tags.append(gmsh.model.occ.addLine(p1, p2))
                    
                    # Create a curve loop and a plane surface from the boundary.
                    cl_tag = gmsh.model.occ.addCurveLoop(l_tags)
                    s_tag = gmsh.model.occ.addPlaneSurface([cl_tag])
                    
                    key = to_key(2, s_tag)
                    input_tag_info[key] = {'type': 'surface', 'id': idx}
                    all_surface_tags.append(key)

        # "Fragment" combines all the individual geometries into a single,
        # topologically consistent model. This is where intersections are
        # calculated and new, smaller entities are created at overlaps.
        object_tags = all_surface_tags + all_line_tags + all_point_tags 
        
        if not object_tags:
            print("Warning: No geometry to mesh.")
            return {'points': {}, 'lines': {}, 'surfaces': {}, 'straddle_surfs': {}}

        print(f"Fragmenting {len(object_tags)} objects...")
        out_dt, out_map = gmsh.model.occ.fragment(object_tags, [])
        gmsh.model.occ.synchronize()
        
        # After fragmentation, we need to rebuild our map of which original
        # feature corresponds to which new Gmsh tags.
        final_map = {'points': {}, 'lines': {}, 'surfaces': {}, 'straddle_surfs': {}}
        
        print(f"Reconstructing Map (Input Tags: {len(object_tags)}, Out Map Len: {len(out_map)})...")
        
        for i, input_dimtag in enumerate(object_tags):
            if i < len(out_map):
                res_tags = out_map[i]
            else:
                res_tags = [input_dimtag]

            # Look up the original feature ID using the pre-fragmentation tag.
            key = to_key(input_dimtag[0], input_dimtag[1])
            
            if key in input_tag_info:
                info = input_tag_info[key]
                kind = info['type']
                feat_id = info['id']
                
                if kind == 'point':
                    if feat_id not in final_map['points']:
                        final_map['points'][feat_id] = []
                    final_map['points'][feat_id].extend(res_tags)
                    
                elif kind == 'line':
                    if feat_id not in final_map['lines']:
                        final_map['lines'][feat_id] = []
                    final_map['lines'][feat_id].extend(res_tags)
                    
                elif kind == 'surface':
                    if feat_id not in final_map['surfaces']:
                        final_map['surfaces'][feat_id] = []
                    final_map['surfaces'][feat_id].extend(res_tags)
                    
                elif kind == 'straddle_surf':
                    if feat_id not in final_map['straddle_surfs']:
                        final_map['straddle_surfs'][feat_id] = []
                    final_map['straddle_surfs'][feat_id].extend(res_tags)
            else:
                print(f"Warning: Tag {key} lost during fragmentation mapping.")

        return final_map
    
    def _setup_fields(self, gmsh_map, polygons_gdf, lines_gdf, points_gdf):
        """
        Configures Gmsh mesh size fields based on the input features.

        This method creates and combines various fields (`Distance`, `Threshold`,
        `MathEval`) to control the mesh element size across the domain. It uses
        the parameters (e.g., `lc`, `dist_min`, `dist_max`) from the
        original conceptual model features to define how the mesh should be
        refined near points, along lines, and within polygons.
        """
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

        field_list = []
        
        # The global background mesh size is the fallback resolution.
        global_max_lc = self.background_lc

        def extract_tags(entry_list):
            """Helper to get a clean list of integer tags from Gmsh's output."""
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
            
            if size_max_limit is None:
                size_max_limit = global_max_lc

            # To ensure the meshing algorithm converges efficiently, the rate of
            # change in element size must be controlled. This heuristic enforces
            # a minimum transition distance to prevent the mesh size gradient
            # from becoming too steep, which can stall the mesher.
            size_diff = size_max_limit - size_target
            if size_diff > 0:
                min_span_required = size_diff / (0.5 * size_target)
                
                current_span = dist_max - dist_min
                if current_span < min_span_required:
                    dist_max = dist_min + min_span_required

            if dist_max <= dist_min:
                dist_max = dist_min + max(size_target, 1e-3)
            
            # Create a `Distance` field, which calculates the distance from the specified entities.
            f_dist = gmsh.model.mesh.field.add("Distance")
            if entity_dim == 0: gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", valid_tags)
            elif entity_dim == 1: gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", valid_tags)
            
            # Create a `Threshold` field, which uses the `Distance` field to
            # define a mesh size that varies linearly from `SizeMin` to `SizeMax`
            # over the range `DistMin` to `DistMax`. This is more efficient than `MathEval`.
            f_thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", float(size_target))
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", float(size_max_limit))
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", float(dist_min))
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", float(dist_max))
            
            return f_thresh

       # 1. Point-based refinement fields.
        for idx, row in points_gdf.iterrows():
            if idx in gmsh_map['points']:
                tags = extract_tags(gmsh_map['points'][idx])
                lc = max(get_row_param(row, 'lc', 5.0), 0.001)
                d_min = get_row_param(row, 'dist_min', lc * 2.0)
                d_max = get_row_param(row, 'dist_max', global_max_lc * 1.5)
                
                fid = add_refinement(0, tags, lc, d_min, d_max)
                if fid: field_list.append(fid)

        # 2. Line-based refinement fields (including straddle barriers).
        for idx, row in lines_gdf.iterrows():
            # Standard lines that exist as curves in Gmsh.
            if idx in gmsh_map['lines']:
                tags = extract_tags(gmsh_map['lines'][idx])
                lc = max(get_row_param(row, 'lc', 10.0), 0.001)
                d_min = get_row_param(row, 'dist_min', lc * 1.0)
                d_max = get_row_param(row, 'dist_max', global_max_lc * 1.5)
                
                fid = add_refinement(1, tags, lc, d_min, d_max)
                if fid: field_list.append(fid)
            
            # "Straddle" lines, which were converted into pairs of points.
            # Refinement must be applied to these points to resolve the gap.
            elif idx in gmsh_map['points']:
                is_barrier = row.get('is_barrier', False)
                straddle = row.get('straddle_width', 0)
                if is_barrier or (straddle and straddle > 0):
                    tags = extract_tags(gmsh_map['points'][idx])
                    lc = max(get_row_param(row, 'lc', 10.0), 0.001)
                    
                    d_min = get_row_param(row, 'dist_min', lc * 2.0)
                    d_max = get_row_param(row, 'dist_max', global_max_lc * 1.5)
                    
                    fid = add_refinement(0, tags, lc, d_min, d_max)
                    if fid: field_list.append(fid)

        # 3. Polygon-based refinement fields.
        for idx, row in polygons_gdf.iterrows():
            if idx in gmsh_map['surfaces']:
                tags = extract_tags(gmsh_map['surfaces'][idx])
                
                target_lc = get_row_param(row, 'lc', global_max_lc)
                border_dens = get_row_param(row, 'border_density', target_lc)
                boundary_lc = min(target_lc, border_dens)

                if self.verbosity > 1:
                    print(f"Poly {idx}: Target={target_lc}, Border={boundary_lc}, Global={global_max_lc}")

                # Set up a field for the polygon's interior.
                if boundary_lc < global_max_lc or target_lc < global_max_lc:
                    
                    dim_tags = [(2, int(t)) for t in tags]
                    boundaries = gmsh.model.getBoundary(dim_tags, combined=True, oriented=False, recursive=False)
                    curve_tags = [b[1] for b in boundaries if b[0] == 1]
                    
                    f_inner = None
                    
                    if curve_tags and boundary_lc < target_lc:
                        # Create a gradient from the finer boundary to the coarser interior.
                        d_min = get_row_param(row, 'dist_min', 0.0)
                        d_max_in = get_row_param(row, 'dist_max_in', -1.0)
                        
                        if d_max_in > d_min:
                            d_max_inner = d_max_in
                        else:
                            d_max_inner = d_min + (boundary_lc * 5.0)
                            d_max_inner = max(d_max_inner, d_min + (target_lc - boundary_lc) * 0.2)
                        
                        if self.verbosity > 1:
                            print(f"  -> Grading Interior: DistMin={d_min}, DistMax={d_max_inner}, SizeMax={target_lc}")

                        f_inner = add_refinement(1, curve_tags, boundary_lc, d_min, d_max_inner, size_max_limit=target_lc)
                        
                    else:
                        # Apply a constant mesh size throughout the interior.
                        if self.verbosity > 1:
                            print(f"  -> Constant Interior: Size={target_lc}")
                        
                        f_dist_inner = gmsh.model.mesh.field.add("Distance")
                        gmsh.model.mesh.field.setNumbers(f_dist_inner, "CurvesList", curve_tags)
                        
                        f_const = gmsh.model.mesh.field.add("Threshold")
                        gmsh.model.mesh.field.setNumber(f_const, "InField", f_dist_inner)
                        gmsh.model.mesh.field.setNumber(f_const, "SizeMin", target_lc)
                        gmsh.model.mesh.field.setNumber(f_const, "SizeMax", target_lc)
                        gmsh.model.mesh.field.setNumber(f_const, "DistMin", 1e22)
                        gmsh.model.mesh.field.setNumber(f_const, "DistMax", 1e22)
                        
                        f_inner = f_const

                    # Restrict this field to apply only inside the polygon surface.
                    if f_inner:
                        f_rest = gmsh.model.mesh.field.add("Restrict")
                        gmsh.model.mesh.field.setNumber(f_rest, "IField", f_inner)
                        gmsh.model.mesh.field.setNumbers(f_rest, "SurfacesList", [float(t) for t in tags])
                        field_list.append(f_rest)

                # Set up a field for the polygon's exterior, grading to the global size.
                d_max_out = get_row_param(row, 'dist_max_out', 0.0)
                
                if d_max_out > 0:
                    dim_tags = [(2, int(t)) for t in tags]
                    boundaries = gmsh.model.getBoundary(dim_tags, combined=True, oriented=False, recursive=False)
                    curve_tags = [b[1] for b in boundaries if b[0] == 1]
                    
                    if curve_tags:
                        d_min = get_row_param(row, 'dist_min', 0.0)
                        if self.verbosity > 1:
                            print(f"  -> Grading Exterior: DistMax={d_max_out}")
                        
                        fid_grad = add_refinement(1, curve_tags, boundary_lc, d_min, d_max_out, size_max_limit=global_max_lc)
                        if fid_grad: field_list.append(fid_grad)

        # 4. Straddle surfaces (placeholder for future transfinite enforcement).
        for idx, tags in gmsh_map.get('straddle_surfs', {}).items():
            clean_tags = extract_tags(tags)
            for s_tag in clean_tags:
                pass

        # 5. Set the final background field. This is a constant field that
        # provides the mesh size for any area not covered by other fields.
        f_bg = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(f_bg, "F", str(global_max_lc))
        field_list.append(f_bg)

        # 6. Combine all fields using a `Min` field. At any point in the
        # domain, the mesh size will be the minimum of all active fields.
        if field_list:
            min_field = gmsh.model.mesh.field.add("Min")
            field_list = [float(f) for f in field_list]
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_list)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        
        # Disable Gmsh's default size-setting mechanisms. We want our fields
        # to have complete control over the mesh size.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    def generate(self, clean_polys, clean_lines, clean_points, output_file=None):
        """
        Executes the full mesh generation workflow.

        This method orchestrates the entire process:
        1. Initializes Gmsh.
        2. Transfers geometries into the Gmsh model.
        3. Sets up mesh size fields.
        4. Generates the 2D triangular mesh.
        5. Performs optional post-generation optimization.
        6. Extracts the resulting nodes and their tags.

        Args:
            clean_polys (GeoDataFrame): Non-overlapping polygons.
            clean_lines (GeoDataFrame): Snapped and cleaned lines.
            clean_points (GeoDataFrame): Snapped and cleaned points.
            output_file (str, optional): If provided, saves the mesh to this path.

        Returns:
            bool: True if generation was successful.
        
        Raises:
            Exception: If any step in the Gmsh process fails.
        """
        self._initialize_gmsh()
        try:
            print("Transferring Geometry to Gmsh...")
            gmsh_map = self._add_geometry(clean_polys, clean_lines, clean_points)
            
            print("Setting up Resolution Fields...")
            self._setup_fields(gmsh_map, clean_polys, clean_lines, clean_points)
            
            # Set the core meshing algorithm.
            gmsh.option.setNumber("Mesh.Algorithm", self.mesh_algorithm) 
            
            # Set the number of internal smoothing steps.
            gmsh.option.setNumber("Mesh.Smoothing", self.smoothing_steps)

            print("Generating Triangular Mesh...")
            gmsh.model.mesh.generate(2)
            
            # Run explicit optimization passes after generation for higher quality.
            if self.optimization_cycles > 0:
                if self.verbosity > 0:
                    print(f"Running {self.optimization_cycles} Optimization Cycles (Relocate2D & Laplace2D)...")
                
                for i in range(self.optimization_cycles):
                    if self.verbosity > 1:
                        print(f"  -> Cycle {i+1}/{self.optimization_cycles}")
                    # Moves nodes to improve element shape (compactness).
                    gmsh.model.mesh.optimize("Relocate2D",niter=1)
                    # Smooths the mesh to relax gradients (reduces drift).
                    gmsh.model.mesh.optimize("Laplace2D",niter=1)

            
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