#%%
"""Field capabilities example (pseudo-notebook).

Run this file in VS Code with the Python extension. The `#%%` markers create
cell-like execution, similar to a notebook.

This example duplicates polygon and line geometries to exercise multiple mesh
field types:
- Implicit threshold via `dist_min`/`dist_max`
- Explicit `ThresholdField`
- `GeometricGrowthField` with edge-ratio and continuous-metric conventions
- Field-only polygon via `embed=False`
- Barrier/straddle line to validate point-pair representation
- `hex_ring=True` point, whose cell is a regular hexagon centred on it
- Lloyd relaxation (`lloyd_iterations`) compared with the default grid
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from shapely.affinity import translate
from shapely.geometry import LineString, Point, box

from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator
from vorflow.fields import (
    GeometricGrowthField,
    ThresholdField,
)
from vorflow.utils import build_connectivity, calculate_mesh_quality, summarize_quality

#%%
# Geometry
domain = box(0, 0, 400, 200)
# Base refinement polygon (simplified set)
poly_base = box(40, 40, 70, 70)

# Create a 2x3 layout (upper/lower × left/center/right)
ul = translate(poly_base, xoff=0, yoff=80)
uc = translate(poly_base, xoff=120, yoff=80)
ur = translate(poly_base, xoff=240, yoff=80)
ll = translate(poly_base, xoff=0, yoff=0)
lc = translate(poly_base, xoff=120, yoff=0)
lr = translate(poly_base, xoff=240, yoff=0)

polys = {
    "upper-left": ul,
    "upper-center": uc,
    "upper-right": ur,
    "lower-left": ll,
    "lower-center": lc,
    "lower-right": lr,
}

# A gentle sine-wave river
xs = np.linspace(-10, 430, 45)
river_y = 140 + 18 * np.sin(0.06 * xs)
river_base = LineString(list(zip(xs, river_y)))
rivers = {
    "river-center-up": translate(river_base, xoff=0, yoff=0),
    "river-center-down": translate(river_base, xoff=0, yoff=-60),
}

# Points (simplified)
points = {
    "pt-lower-left": Point(25, 25),
    "pt-lower-center": Point(185, 25),
    "pt-lower-right": Point(325, 25),
    # At least 2 x resolution from every other embedded feature, so the ring is kept.
    "pt-hex-ring": Point(110, 30),
}
#%%
# Fields used below

edge_growth = GeometricGrowthField(growth_factor=1.1)
metric_growth = GeometricGrowthField(
    growth_factor=1.1, growth_model="continuous_metric"
)
threshold = ThresholdField(size_min=5.0, dist_min=5.0, dist_max=50.0, size_max=20.0)





#%%
# 1) Setup Blueprint + 2) Mesh Generation + 3) Voronoi Conversion

background_lc = 100
feature_lc = 10  # representative base feature length for features

blueprint = ConceptualMesh(crs="EPSG:3857")
blueprint.add_polygon(domain, zone_id="domain")  # color: black (domain boundary)

# Polygons (IDs are location-based)
blueprint.add_polygon(
    polys["upper-left"],
    zone_id="upper-left",
    resolution=feature_lc/5,
    z_order=10,
    dist_min=feature_lc/2,#using the implicit threshold approach here (instead of an explicit ThresholdField) to validate both code paths
    dist_max=background_lc * 5.0,
)

blueprint.add_polygon(
    polys["upper-center"],
    zone_id="upper-center",
    resolution=feature_lc/5,
    z_order=10,
    fields=[edge_growth],
    embed=False,  # field-only polygon 
)

blueprint.add_polygon(
    polys["upper-right"],
    zone_id="upper-right",
    resolution=feature_lc/5,
    z_order=10,
    fields=[metric_growth],
)


blueprint.add_line(
    polys["lower-left"].boundary,
    line_id='lower-left',# zone_id="lower-left",
    resolution=feature_lc/5,
    # z_order=5,
    dist_min=feature_lc/2,
    dist_max=background_lc * 1.5,
    fields=[metric_growth],
    embed=False,  # field-only boundary line
)

blueprint.add_polygon(
    polys["lower-center"],
    zone_id="lower-center",
    resolution=feature_lc/5,
    z_order=5,
    fields=[edge_growth],
)

blueprint.add_polygon(
    polys["lower-right"],
    zone_id="lower-right",
    resolution=feature_lc/5,
    z_order=5,
    fields=[threshold],
    embed=False,  # field-only polygon
)

blueprint.add_line(
    rivers["river-center-up"],
    line_id="river-center-up",
    resolution=feature_lc/2,
    is_barrier=False,
    fields=[threshold],
)

blueprint.add_line(
    rivers["river-center-down"],
    line_id="river-center-down",
    resolution=feature_lc/4,
    is_barrier=False,
    fields=[edge_growth],
    embed=False,  # field-only line
)

# Points (IDs are location-based)
# Colors: pt-lower-left -> 'tab:red', pt-lower-center -> 'tab:purple'
blueprint.add_point(
    points["pt-lower-left"],
    point_id="pt-lower-left",
    resolution=feature_lc/5,
    dist_min=feature_lc/5,
    dist_max=background_lc * 1.5,
)
blueprint.add_point(
    points["pt-lower-center"],
    point_id="pt-lower-center",
    resolution=feature_lc/5,
    fields=[edge_growth],
    embed=False,  # field-only point
)

blueprint.add_point(
    points["pt-lower-right"],
    point_id="pt-lower-right",
    resolution=feature_lc/5,
    fields=[metric_growth],
    embed=False,  # field-only point
)

blueprint.add_point(
    points["pt-hex-ring"],
    point_id="pt-hex-ring",
    resolution=feature_lc/5,
    growth_factor=1.2,
    hex_ring=True,  # six fixed seeds at radius `resolution` -> regular hexagonal cell
)

clean_polys, clean_lines, clean_pts = blueprint.generate()

mesher = MeshGenerator(background_lc=background_lc, verbosity=0)
mesher.generate(clean_polys, clean_lines, clean_pts, launch_gmsh_gui=False)

#%% 3) Triangular quality + Voronoi conversion + quality reports

tri_quality = mesher.get_triangular_quality()
print("\nTriangular element quality summary:")
print(tri_quality["element_name"].value_counts().to_string())

tri_quality_metrics = [
    "minSICN",
    "minDetJac",
    "maxDetJac",
    "minSJ",
    "minSIGE",
    "gamma",
    "innerRadius",
    "outerRadius",
    "minIsotropy",
    "angleShape",
    "minEdge",
    "maxEdge",
]
print(
    tri_quality[tri_quality_metrics]
    .describe(percentiles=[0.05, 0.5, 0.95])
    .T[["min", "5%", "50%", "95%", "max"]]
)

tessellator = VoronoiTessellator(mesher, blueprint, clip_to_boundary=True)
grid_gdf = tessellator.generate()

modflow_connectivity_report = build_connectivity(grid_gdf, center="centroid")
voronoi_dual_connectivity_report = build_connectivity(grid_gdf, center="generator")

print("\nMODFLOW-facing centroid connectivity summary:")
print(
    modflow_connectivity_report[["angle", "ortho_error", "skewness"]]
    .describe(percentiles=[0.05, 0.5, 0.95])
    .T[["min", "5%", "50%", "95%", "max"]]
)

print("\nVoronoi-dual generator connectivity summary:")
print(
    voronoi_dual_connectivity_report[["angle", "ortho_error", "skewness"]]
    .describe(percentiles=[0.05, 0.5, 0.95])
    .T[["min", "5%", "50%", "95%", "max"]]
)

worst_columns = [
    col
    for col in ["node_id_1", "node_id_2", "angle", "ortho_error", "skewness"]
    if col in modflow_connectivity_report.columns
]
print("\nWorst MODFLOW-facing centroid connectivity pairs by orthogonality error:")
print(
    modflow_connectivity_report
    .sort_values("ortho_error", ascending=False)
    .head(10)[worst_columns]
    .to_string(index=False)
)

quality_gdf = calculate_mesh_quality(
    grid_gdf,
    calc_ortho=True,
    calc_skewness=True,
    connectivity=modflow_connectivity_report,
)
summarize_quality(quality_gdf)


#%%
# 4) Visual sanity-check

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.set_aspect("equal")

# Domain
ax.plot(*domain.exterior.xy, color="black", lw=1)

# Polygons
plot_polys = polys
poly_colors = {
    "upper-left": "tab:orange",
    "upper-center": "tab:green",
    "upper-right": "tab:blue",
    "lower-left": "tab:purple",
    "lower-center": "tab:brown",
    "lower-right": "tab:pink",
}
field_only = {
    "upper-center": "polygon",
    "lower-left": "boundary line",
    "lower-right": "polygon",
}
for name, poly in plot_polys.items():
    field_only_kind = field_only.get(name)
    label = f"{name} (field-only {field_only_kind})" if field_only_kind else name
    ax.plot(
        *poly.exterior.xy,
        lw=1,
        ls=":" if field_only_kind else "--",
        color=poly_colors.get(name),
        label=label,
    )

line_colors = {
    "river-center-up": "tab:cyan",
    "river-center-down": "tab:cyan",
}

ax.plot(
    *rivers["river-center-up"].xy,
    lw=1,
    color=line_colors["river-center-up"],
    label="river-center-up",
)
ax.plot(
    *rivers["river-center-down"].xy,
    lw=1,
    color=line_colors["river-center-down"],
    label="river-center-down (field-only)",
)

# Points
point_colors = {
    "pt-lower-left": "tab:red",
    "pt-lower-center": "tab:purple",
    "pt-lower-right": "tab:blue",
    "pt-hex-ring": "tab:olive",
}
for name, pt in points.items():
    ax.scatter(pt.x, pt.y, s=25, marker="x", color=point_colors.get(name), label=name)

grid_gdf.plot(ax=ax, alpha=0.35, edgecolor="k", linewidth=0.15)
ax.legend(loc="upper right", fontsize=7, ncol=2)
fig.tight_layout()
plt.show()


#%%
# Quick check: smaller cells => smaller polygon areas (proxy for refinement)

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_aspect("equal")
ax.plot(*domain.exterior.xy, color="black", lw=1)
quality_gdf.plot(ax=ax, column="area", cmap="viridis", legend=True, linewidth=0.0)
ax.set_title("Voronoi cell area (proxy for refinement)")
fig.tight_layout()
plt.show()

#%% 5) Quality diagnostics plots

fig, ax = plt.subplots(figsize=(10, 5))
for element_name, group in tri_quality.groupby("element_name"):
    group["gamma"].plot.hist(
        ax=ax,
        bins=40,
        alpha=0.55,
        label=element_name,
    )
ax.set_title("Gmsh 2D element quality: gamma")
ax.set_xlabel("gamma (higher is better)")
ax.legend()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect("equal")
grid_gdf.plot(ax=ax, color="white", edgecolor="0.85", linewidth=0.2)
modflow_connectivity_report.set_geometry("connector").plot(
    ax=ax,
    color="0.35",
    linewidth=0.35,
    alpha=0.35,
)
# ortho_error is in degrees (0 is ideal); let the color scale auto-range
# instead of saturating at 1 degree.
modflow_connectivity_report.plot(
    column="ortho_error",
    ax=ax,
    legend=True,
    cmap="Reds",
    linewidth=1.2,
    vmin=0,
)
ax.plot(*domain.exterior.xy, color="black", lw=1)
ax.set_title("MODFLOW-facing centroid connectivity: shared-face orthogonality error")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect("equal")
quality_gdf.plot(column="ortho_error", ax=ax, legend=True, cmap="Reds", vmin=0)
ax.plot(*domain.exterior.xy, color="black", lw=1)
ax.set_title("Per-cell orthogonality error (degrees, worst face per cell)")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect("equal")
quality_gdf.plot(column="skewness", ax=ax, legend=True, cmap="Purples", vmin=0, vmax=0.5)
ax.plot(*domain.exterior.xy, color="black", lw=1)
ax.set_title("Per-cell face skewness (0 = connector through face midpoint)")
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
dashboard_metrics = [
    ("area", "Cell area", "viridis", None, None),
    ("drift_ratio", "Generator drift ratio", "magma", None, None),
    ("ortho_error", "Orthogonality error (degrees)", "Reds", 0, None),
    ("skewness", "Face skewness", "Purples", 0, 0.5),
]
for ax, (column, title, cmap, vmin, vmax) in zip(axes.ravel(), dashboard_metrics):
    ax.set_aspect("equal")
    quality_gdf.plot(
        column=column,
        ax=ax,
        legend=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0.0,
    )
    ax.plot(*domain.exterior.xy, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_axis_off()
fig.tight_layout()
plt.show()

#%% 6) hex_ring cell and Lloyd relaxation

hex_cell = grid_gdf[grid_gdf.contains(points["pt-hex-ring"])].iloc[0]
hex_coords = np.array(hex_cell.geometry.exterior.coords)[:-1]
hex_radii = np.hypot(*(hex_coords - [points["pt-hex-ring"].x, points["pt-hex-ring"].y]).T)
print(
    f"\npt-hex-ring cell: {len(hex_coords)} vertices at {hex_radii.min():.4f}-{hex_radii.max():.4f} m "
    f"(resolution / sqrt(3) = {feature_lc / 5 / np.sqrt(3):.4f} m); "
    f"rings intact after meshing: {mesher.diagnostics['hex_rings']}"
)

# Lloyd is opt-in; use 100 or more passes (about 20 leaves the most very short faces).
lloyd_tessellator = VoronoiTessellator(mesher, blueprint, clip_to_boundary=True, lloyd_iterations=100)
lloyd_grid = lloyd_tessellator.generate()
print(f"Lloyd report: {lloyd_tessellator.lloyd_report}")
lloyd_connectivity = build_connectivity(lloyd_grid, center="centroid")
print(
    "p95 centroid ortho_error (degrees): "
    f"default {modflow_connectivity_report['ortho_error'].quantile(0.95):.2f}, "
    f"lloyd_iterations=100 {lloyd_connectivity['ortho_error'].quantile(0.95):.2f}"
)

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_aspect("equal")
lloyd_grid.plot(
    ax=ax,
    column=lloyd_grid["lloyd_shift"] / np.sqrt(lloyd_grid.geometry.area),
    cmap="viridis",
    legend=True,
    vmin=0,
    linewidth=0.0,
)
ax.plot(*domain.exterior.xy, color="black", lw=1)
ax.set_title("Lloyd relaxation: generator shift / sqrt(cell area) (0 for fixed nodes)")
fig.tight_layout()
plt.show()
# %%
