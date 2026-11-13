"""Optional Matplotlib plotting behind the ``plot()`` methods.

Matplotlib is not a dependency of vorflow: it is imported only when a
``plot()`` method is called, and installed with the ``plot`` extra.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from ._features import is_embedded, row_bool

# Columns the raw feature dicts are reduced to before plotting; the rest
# (fields, refinement settings) cannot go into a GeoDataFrame cleanly.
RAW_PLOT_COLUMNS = ("geometry", "zone_id", "embed", "is_barrier", "label", "color")


def require_matplotlib():
    """Import and return ``matplotlib.pyplot``, with an install hint if it is missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "Plotting needs Matplotlib; install it with `pip install vorflow[plot]`."
        ) from err
    return plt


def get_ax(ax=None):
    """``ax``, or the axes of a new figure when None, set to an equal aspect ratio."""
    plt = require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    ax.set_aspect("equal")
    return ax


def features_frame(features, crs):
    """GeoDataFrame of raw feature dicts (as stored by ``ConceptualMesh.add_*``), reduced to ``RAW_PLOT_COLUMNS``."""
    if not features:
        return gpd.GeoDataFrame()
    rows = [{key: feature[key] for key in RAW_PLOT_COLUMNS if key in feature} for feature in features]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def split_embedded(gdf):
    """(embedded, field_only) rows of ``gdf``."""
    if gdf.empty:
        return gdf, gdf
    mask = gdf.apply(is_embedded, axis=1).astype(bool)
    return gdf[mask], gdf[~mask]


def label_groups(gdf):
    """``(label, positions)`` of ``gdf`` rows grouped by their ``label`` column.

    Unlabelled rows come first with label None; labelled groups follow in
    order of first appearance, so legend entries follow insertion order.
    """
    labels = gdf["label"] if "label" in gdf.columns else pd.Series(None, index=gdf.index)
    groups = {}
    for position, label in enumerate(labels):
        groups.setdefault(None if pd.isna(label) else str(label), []).append(position)
    unlabelled = groups.pop(None, [])
    return ([(None, unlabelled)] if unlabelled else []) + list(groups.items())


def plot_by_label(gdf, ax, colors=None, **kwargs):
    """Plot ``gdf`` with one artist per label group, so each label gets a legend entry.

    ``colors`` (optional) holds one colour per row of ``gdf``.
    """
    for label, positions in label_groups(gdf):
        group_kwargs = dict(kwargs)
        if colors is not None:
            group_kwargs["color"] = [colors[i] for i in positions]
        if label is not None:
            group_kwargs["label"] = label
        gdf.iloc[positions].plot(ax=ax, **group_kwargs)


def line_colors(gdf, barrier_color="tab:red"):
    """Colour of each line row: its ``color``, else ``barrier_color`` for barriers and blue otherwise."""
    colors = []
    for _, row in gdf.iterrows():
        color = row.get("color")
        if color is None or (isinstance(color, float) and pd.isna(color)):
            color = barrier_color if row_bool(row, "is_barrier") else "tab:blue"
        colors.append(color)
    return colors


def category_colors(values, cmap):
    """Colour of each value as GeoPandas draws a categorical column with ``cmap``.

    Categories are the sorted unique values, spread evenly over the colormap;
    missing values are light grey.
    """
    from matplotlib import colormaps

    if isinstance(cmap, str):
        cmap = colormaps[cmap]
    categorical = pd.Categorical(values)
    top_code = max(len(categorical.categories) - 1, 1)
    return [cmap(code / top_code) if code >= 0 else "lightgrey" for code in categorical.codes]


def add_polygon_legend_entries(ax, gdf, colors, polygon_kwargs):
    """Add a legend entry per label of the polygons ``gdf`` drawn with ``polygon_kwargs``.

    Polygon collections make poor legend handles (GeoPandas >= 1.2 also
    labels them by category), so each label gets an empty patch styled like
    its first polygon. Without ``colors`` (one per row) the patch is hollow.
    """
    for label, positions in label_groups(gdf):
        if label is None:
            continue
        ax.fill(
            [], [],
            facecolor=colors[positions[0]] if colors is not None else "none",
            edgecolor=polygon_kwargs.get("edgecolor"),
            linewidth=polygon_kwargs.get("linewidth"),
            alpha=polygon_kwargs.get("alpha"),
            label=label,
        )


# Polygon keywords that take over GeoPandas' colouring, so the categorical
# colours are left to GeoPandas and are not known for the legend.
GEOPANDAS_COLOUR_KEYWORDS = {"color", "facecolor", "norm", "vmin", "vmax", "categories", "legend"}


def plot_features(ax, polygons, lines, points, column="zone_id", **kwargs):
    """Draw conceptual-model features on ``ax``; returns ``ax``.

    Embedded polygons are filled by ``column`` (``kwargs`` go to their
    ``GeoDataFrame.plot``); lines and points are drawn on top, each line in
    its ``color`` (default: barriers red, others blue). Field-only (``embed=False``) features are dashed outlines, dashed
    lines and hollow points, since they only size the mesh. Features with a
    ``label`` get one legend entry per label, shown by ``ax.legend()``.
    """
    from matplotlib.colors import is_color_like

    embedded, field_only = split_embedded(polygons)
    if not embedded.empty:
        polygon_kwargs = {"edgecolor": "black", "linewidth": 0.8, "alpha": 0.6, **kwargs}
        colors = None
        if column is not None and column in embedded.columns:
            categorical = polygon_kwargs.pop("categorical", True)
            cmap = polygon_kwargs.pop("cmap", "Pastel1")
            if categorical and not GEOPANDAS_COLOUR_KEYWORDS & set(kwargs):
                colors = category_colors(embedded[column], cmap)
                polygon_kwargs["color"] = colors
            else:
                polygon_kwargs.update(column=column, categorical=categorical, cmap=cmap)
        elif "facecolor" not in polygon_kwargs:
            # One colour for all polygons: GeoPandas' default unless given.
            polygon_kwargs.setdefault("color", "C0")
        single_color = polygon_kwargs.get("facecolor", polygon_kwargs.get("color"))
        if colors is None and is_color_like(single_color):
            colors = [single_color] * len(embedded)
        n_collections = len(ax.collections)
        embedded.plot(ax=ax, **polygon_kwargs)
        for collection in ax.collections[n_collections:]:
            collection.set_label("_nolegend_")
        add_polygon_legend_entries(ax, embedded, colors, polygon_kwargs)
    if not field_only.empty:
        outlines = field_only.assign(geometry=field_only.boundary)
        plot_by_label(outlines, ax, color="grey", linestyle="--", linewidth=0.8)

    embedded, field_only = split_embedded(lines)
    if not embedded.empty:
        plot_by_label(embedded, ax, colors=line_colors(embedded), linewidth=1.5)
    if not field_only.empty:
        plot_by_label(field_only, ax, colors=line_colors(field_only, barrier_color="tab:blue"), linestyle="--", linewidth=1.0)

    embedded, field_only = split_embedded(points)
    if not embedded.empty:
        plot_by_label(embedded, ax, color="black", markersize=12)
    if not field_only.empty:
        plot_by_label(field_only, ax, facecolor="none", edgecolor="black", markersize=12)
    return ax


def draw_connections(ax, start, end, values, cmap=None, norm=None, linewidth=1.2, colorbar=True, label=None):
    """Connections from ``start`` to ``end`` (``(n, 2)`` points) coloured by ``values``; returns the LineCollection.

    Drawn in order of value, so the largest end on top. Connections of zero
    length (a vertical connection within one cell column) are dots.
    """
    plt = require_matplotlib()
    from matplotlib.collections import LineCollection

    order = np.argsort(values, kind="stable")
    start, end, values = start[order], end[order], values[order]
    if norm is None:
        norm = plt.Normalize(*(np.nanmin(values), np.nanmax(values)) if len(values) else (0, 1))
    dot = (start == end).all(axis=1)
    lines = LineCollection(np.stack([start[~dot], end[~dot]], axis=1), cmap=cmap, norm=norm, linewidths=linewidth)
    lines.set_array(values[~dot])
    ax.add_collection(lines)
    if dot.any():
        ax.scatter(*start[dot].T, c=values[dot], cmap=lines.cmap, norm=norm, s=4 * linewidth, zorder=3)
    ax.autoscale_view()
    if colorbar:
        ax.figure.colorbar(lines, ax=ax, label=label)
    return lines
