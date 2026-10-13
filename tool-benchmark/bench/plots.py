"""Figures: meshes, cell size vs distance, convergence, and the Tier-0 verification summary.

Each tool keeps one colour and marker in every figure (``TOOL_STYLE``): the
first four categorical slots of the dataviz reference palette, in fixed order,
with a distinct marker as secondary encoding. Slot 4 (yellow) is below 3:1
contrast on white, so every figure carries a legend.

The module leaves the Matplotlib backend alone, so notebooks can import
``TOOL_STYLE``; ``workflow.make_figures`` selects Agg before importing it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.collections import PolyCollection

from .case import Case
from .grid import open_rings
from .metrics import face_metrics, face_table

TOOL_STYLE = {
    "vorflow": {"color": "#2a78d6", "marker": "o"},
    "vorflow_lloyd": {"color": "#8a4fd6", "marker": "o"},
    "mf6voronoi": {"color": "#eb6834", "marker": "s"},
    "flopy": {"color": "#1baf7a", "marker": "^"},
    "vorogridgen": {"color": "#eda100", "marker": "D"},
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style_axes(ax) -> None:
    """Recessive grid and spines, text in ink colours."""
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _save(fig, path: Path) -> None:
    """Write a figure and close it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_meshes(case: Case, grids: dict, path: Path) -> None:
    """One panel per tool: cells coloured by MF6-seen orthogonality error, features on top."""
    fig, axes = plt.subplots(1, len(grids), figsize=(4.5 * len(grids), 4.8), squeeze=False)
    for ax, (tool, grid) in zip(axes[0], grids.items()):
        polys = [grid.vertices[r] for r in open_rings(grid.iverts)]
        pc = PolyCollection(polys, array=_cell_max_ortho(grid), cmap="viridis", clim=(0, 10),
                            edgecolors="k", linewidths=0.1)
        ax.add_collection(pc)
        _plot_features(ax, case)
        ax.set_title(f"{tool}: {grid.ncpl} cells", color=INK)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(pc, ax=axes[0].tolist(), shrink=0.8, label="max orthogonality error per cell, deg (as MF6 reads it)")
    fig.suptitle(case.id, color=INK)
    _save(fig, path)


def _cell_max_ortho(grid) -> np.ndarray:
    """Largest orthogonality error over each cell's faces, using the written centres."""
    faces = face_table(grid)
    ortho = face_metrics(grid, faces, "written")["ortho_deg"]
    per_cell = np.zeros(grid.ncpl)
    np.maximum.at(per_cell, faces["ci"], ortho)
    np.maximum.at(per_cell, faces["cj"], ortho)
    return per_cell


def _plot_features(ax, case: Case) -> None:
    """Draw case lines, polygon outlines, points and the domain outline."""
    ax.plot(*case.domain.exterior.xy, color="k", lw=1)
    for ring in case.domain.interiors:
        ax.plot(*ring.xy, color="k", lw=1)
    for f in case.features:
        colour = "#e34948" if f.barrier else "#ffffff"
        if f.kind == "point":
            ax.plot(*f.geometry.xy, "o", color="#e34948", ms=4, mec="white", mew=0.8)
        elif f.kind == "polygon":
            ax.plot(*f.geometry.exterior.xy, color=colour, lw=1)
        else:
            for part in getattr(f.geometry, "geoms", [f.geometry]):
                ax.plot(*part.xy, color=colour, lw=1.2)


def plot_size_vs_distance(case: Case, grids: dict, path: Path) -> None:
    """Cell size against distance from the features, both divided by each grid's spec scale.

    The spec is linear in scale, h(d; s) = s * h(d / s; 1), so normalised points
    from every tool fall on the unscaled spec curve of the finest feature.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    features = shapely.union_all([f.geometry for f in case.features])
    for tool, grid in grids.items():
        scale = grid.info.get("scale", 1.0)
        area = shapely.area([shapely.Polygon(grid.vertices[r]) for r in open_rings(grid.iverts)])
        d = shapely.distance(features, shapely.points(grid.xc))
        ax.plot(d / scale, np.sqrt(2 * area / np.sqrt(3)) / scale, TOOL_STYLE[tool]["marker"],
                color=TOOL_STYLE[tool]["color"], ms=2, alpha=0.5, ls="none",
                label=f"{tool} ({grid.ncpl} cells, scale {scale:.2f})")
    h_f = min(f.h for f in case.features)
    d_line = np.linspace(0, 1.3 * case.h_max / (case.growth - 1), 200)
    ax.plot(d_line, np.minimum(case.h_max, h_f + (case.growth - 1) * d_line), color=INK, lw=2, label="spec")
    ax.set_xlabel("distance from feature / scale")
    ax.set_ylabel("cell size sqrt(2A/sqrt(3)) / scale")
    ax.set_title(case.id, color=INK)
    ax.legend(markerscale=4, fontsize=8, frameon=False)
    _style_axes(ax)
    _save(fig, path)


def plot_convergence(mf6: pd.DataFrame, case_id: str, path: Path) -> None:
    """L2 head error against cell count, one line per tool and centre convention, XT3D off/on."""
    data = mf6[(mf6["case"] == case_id) & mf6["mf6_ok"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, xt3d in zip(axes, (False, True)):
        sub = data[data["xt3d"] == xt3d]
        for (tool, centres), line in sub.groupby(["tool", "centres"], sort=False):
            line = line.sort_values("ncpl")
            style = TOOL_STYLE[tool]
            written = centres == "written"
            ax.plot(line["ncpl"], line["l2"], color=style["color"], marker=style["marker"], ms=6, lw=2,
                    ls="-" if written else "--", mfc=style["color"] if written else "white",
                    label=f"{tool}, {centres} centres")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cells (ncpl)")
        ax.set_title(f"XT3D {'on' if xt3d else 'off'}", color=INK)
        _style_axes(ax)
    axes[0].set_ylabel("L2 head error")
    axes[1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(f"{case_id}: solid = centres as the tool writes them, dashed = the other convention",
                 color=INK, fontsize=10)
    _save(fig, path)


def plot_verification(mf6: pd.DataFrame, case_ids: list, path: Path) -> None:
    """Dot plot of L2 error per case/problem and tool, at each case's first target.

    Filled marker: centres as the tool writes them. Open marker joined by a thin
    line: the same grid with the other centre convention (generator for tools
    that write centroids, centroid for tools that write generators). Grids
    that missed the matched cell count are left out.
    """
    first = mf6.groupby("case")["target"].transform("first")
    data = mf6[(mf6["target"] == first) & mf6["case"].isin(case_ids) & mf6["mf6_ok"] & mf6["matched"]]
    rows = list(dict.fromkeys(zip(data["case"], data["problem"])))
    tools = [t for t in TOOL_STYLE if t in set(data["tool"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(rows) + 1.5), sharey=True)
    for ax, xt3d in zip(axes, (False, True)):
        sub = data[data["xt3d"] == xt3d]
        labelled = set()
        for k, tool in enumerate(tools):
            style = TOOL_STYLE[tool]
            offset = (k - (len(tools) - 1) / 2) * 0.18
            for i, (case_id, problem) in enumerate(rows):
                cell = sub[(sub["case"] == case_id) & (sub["problem"] == problem) & (sub["tool"] == tool)]
                written = cell[cell["centres"] == "written"]["l2"]
                other = cell[cell["centres"] != "written"]["l2"]
                y = i + offset
                if len(written) and len(other):
                    ax.plot([written.iloc[0], other.iloc[0]], [y, y], color=style["color"], lw=0.8, alpha=0.6)
                if len(other):
                    ax.plot(other.iloc[0], y, style["marker"], ms=7, mfc="white", mec=style["color"], mew=1.5)
                if len(written):
                    ax.plot(written.iloc[0], y, style["marker"], ms=7, color=style["color"],
                            label=None if tool in labelled else tool)
                    labelled.add(tool)
        ax.set_xscale("log")
        ax.set_xlabel("L2 head error")
        ax.set_title(f"XT3D {'on' if xt3d else 'off'}", color=INK)
        _style_axes(ax)
    axes[0].set_yticks(range(len(rows)), [f"{c}\n{p}" for c, p in rows], fontsize=8)
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Tier 0 head error at matched cell count. Filled: centres as written; "
                 "open: same grid, other centre convention",
                 color=INK, fontsize=10)
    _save(fig, path)
