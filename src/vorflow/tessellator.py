import numpy as np
import geopandas as gpd
import pandas as pd
import gmsh
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

class VoronoiTessellator:
    def __init__(self, mesh_generator, conceptual_mesh):
        """
        Converts the Triangular Mesh into a Polgonal Voronoi Grid,
        strictly respecting the boundaries defined in ConceptualMesh.
        
        Args:
            mesh_generator (MeshGenerator): The instance holding the active Gmsh model.
            conceptual_mesh (ConceptualMesh): The instance holding the clean zones.
        """
        self.mg = mesh_generator
        self.cm = conceptual_mesh
        self.voronoi_gdf = None
        self.final_grid = None
        self.nodes = mesh_generator.nodes
        self.node_tags = mesh_generator.node_tags
        self.zones_gdf = mesh_generator.zones_gdf

    def _extract_nodes(self):
        """
        Pulls node coordinates from the active Gmsh model.
        Returns:
            nodes (np.array): (N, 2) array of XY coordinates.
            node_tags (np.array): Array of Gmsh node IDs.
        """
        # Get all nodes from the model (dim=-1 means all dimensions)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        
        # coords comes as a flat 1D array [x1, y1, z1, x2, y2, z2...]
        # Reshape to (N, 3) and slice to (N, 2) for 2D operations
        nodes_3d = np.array(coords).reshape(-1, 3)
        nodes_2d = nodes_3d[:, :2]
        
        return nodes_2d, node_tags

    def _build_raw_voronoi(self, nodes, node_tags):
        """
        Uses Scipy to compute the mathematical Voronoi diagram of the nodes.
        Then converts infinite regions into finite Shapely polygons.
        """
        vor = Voronoi(nodes)
        
        polygons = []
        ids = []
        
        # Loop through point_region map to ensure we align with node_tags
        # vor.point_region gives the index of the region for each input point
        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            
            # Skip empty regions or regions with -1 (indicates infinity/outer boundary)
            if not region or -1 in region:
                continue
            
            # Get vertices for this region
            verts = vor.vertices[region]
            
            # Create Polygon
            poly = Polygon(verts)
            
            if poly.is_valid:
                polygons.append(poly)
                ids.append(node_tags[i]) # Map back to Gmsh Node ID
        
        # Create a preliminary GDF
        # Note: This set excludes the "Infinite" boundary cells. 
        # We will handle the boundary by constructing a bounding box clip.
        gdf = gpd.GeoDataFrame(
            {'node_id': ids}, 
            geometry=polygons, 
            crs=self.cm.crs
        )
        return gdf

    def _repair_boundary_cells(self, raw_gdf, nodes):
        """
        The Scipy Voronoi alg leaves 'infinite' cells at the boundary.
        We fix this by creating a synthetic bounding box larger than the domain,
        generating Voronoi for that, and clipping to the Domain Boundary.
        """
        # 1. Get the Domain Boundary from ConceptualMesh
        domain_shape = self.cm.domain_boundary
        
        # 2. Filter raw_gdf to only those strictly inside or intersecting
        # (This removes garbage infinite approximations if any slipped through)
        # Actually, for Option A, we want to maximize coverage.
        
        # STRATEGY UPDATE for Robustness:
        # Scipy's finite regions are often insufficient for the hull.
        # A simple engineering fix: 
        # Add "Ghost Nodes" far outside the domain before Voronoi generation
        # to force the domain hull nodes to have finite cells. 
        # (Implemented in generate() wrapper for cleanliness).
        pass 

    def generate(self):
        """
        Main execution workflow.
        """
        if self.nodes is None:
            raise RuntimeError("MeshGenerator data (nodes) is missing. Run MeshGenerator.generate() first.")
        
        print("Extracting Nodes from Gmsh...")
        nodes, tags = self.nodes, self.node_tags
        
        # --- GHOST NODE TRICK ---
        # Add 4 ghost nodes very far away to force closure of the convex hull
        # This ensures 'scipy.spatial.Voronoi' returns finite polygons for our domain.
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
        # Note: tags for ghost nodes don't matter, we will filter them out later
        
        print("Computing Mathematical Voronoi...")
        raw_gdf = self._build_raw_voronoi(combined_nodes, tags)
        
        # Filter out the ghost cells (they will be massive)
        # Simple check: Keep cells that intersect the domain
        print("Clipping to Domain Boundary...")
        domain_gdf = gpd.GeoDataFrame(
            geometry=[self.cm.domain_boundary], 
            crs=self.cm.crs
        )
        
        # Clip 1: Global Domain Cut
        # This ensures the outer boundary is exactly the user input boundary
        bounded_voronoi = gpd.clip(raw_gdf, domain_gdf)
        
        print("Enforcing Hydrogeological Zones (The Cookie Cutter)...")
        # Clip 2: Internal Zone Enforcement
        # We overlay the bounded voronoi with the clean polygons from ConceptualMesh.
        # This splits any cell crossing a line.
        
        # Ensure the clean_polygons have the metadata we want to keep (Zone ID, etc)
        zones = self.cm.clean_polygons[['geometry', 'zone_id', 'z_order']]
        
        # The Intersection logic
        # keep_geom_type=True ensures we don't get LineStrings from touching edges
        self.final_grid = gpd.overlay(
            bounded_voronoi, 
            zones, 
            how='intersection', 
            keep_geom_type=True
        )
        
        # Post-Processing: Cleanup
        # The overlay might produce MultiPolygons or tiny slivers.
        self.final_grid = self.final_grid.explode(index_parts=True).reset_index(drop=True)
        
        # Calculate cell centers (for MF6 DISV)
        self.final_grid['x'] = self.final_grid.geometry.centroid.x
        self.final_grid['y'] = self.final_grid.geometry.centroid.y
        
        print(f"Final Voronoi Grid Generated: {len(self.final_grid)} cells.")
        return self.final_grid

    def export_to_shapefile(self, filepath):
        if self.final_grid is not None:
            self.final_grid.to_file(filepath)
            print(f"Saved to {filepath}")