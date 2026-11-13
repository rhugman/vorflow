import sys

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from vorflow.blueprint import ConceptualMesh
from vorflow.tessellator import VoronoiTessellator

from test_pipeline import FakeMeshGenerator

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _model():
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id=1, resolution=2)
    cm.add_polygon(Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]), zone_id=2, resolution=1, z_order=1)
    cm.add_polygon(Polygon([(6, 6), (8, 6), (8, 8), (6, 8)]), zone_id=3, resolution=1, embed=False)
    cm.add_line(LineString([(1, 8), (9, 8)]), line_id="river", resolution=1)
    cm.add_line(LineString([(5, 1), (5, 9)]), line_id="fault", resolution=1, embed=False)
    cm.add_point(Point(7, 3), point_id="well", resolution=0.5)
    return cm


@pytest.mark.parametrize("clean", [False, True])
def test_conceptual_mesh_plot_draws_all_feature_types(clean):
    cm = _model()
    if clean:
        cm.generate()

    ax = cm.plot(clean=clean)

    assert ax.collections  # polygons, lines and points are all collections
    assert ax.get_aspect() == 1.0


def test_conceptual_mesh_plot_clean_before_generate_raises():
    with pytest.raises(RuntimeError, match="generate"):
        _model().plot(clean=True)


def test_conceptual_mesh_plot_uses_given_axes():
    _, ax = plt.subplots()
    assert _model().plot(ax=ax, column=None) is ax


def _labelled_model():
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id=1, label="Region")
    cm.add_polygon(Polygon([(6, 6), (8, 6), (8, 8), (6, 8)]), zone_id=2, embed=False, label="Refine")
    cm.add_line(LineString([(5, 0), (5, 8)]), line_id="fault", resolution=1, is_barrier=True,
                label="Fault")
    cm.add_line(LineString([(0, 2), (10, 2)]), line_id="left", resolution=1, label="Boundary")
    cm.add_line(LineString([(0, 4), (10, 4)]), line_id="right", resolution=1, label="Boundary")
    cm.add_line(LineString([(0, 9), (10, 9)]), line_id="unlabelled", resolution=1)
    cm.add_point(Point(2, 6), point_id="well", resolution=0.5, label="Well")
    return cm


@pytest.mark.parametrize("clean", [False, True])
def test_conceptual_mesh_plot_gives_one_legend_entry_per_label(clean):
    cm = _labelled_model()
    if clean:
        cm.generate()

    legend = cm.plot(clean=clean).legend()

    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["Region", "Refine", "Fault", "Boundary", "Well"]


def test_conceptual_mesh_plot_polygon_legend_entry_matches_fill():
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id=1, label="A")
    cm.add_polygon(Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]), zone_id=2, z_order=1, label="B")

    ax = cm.plot()

    fills = {tuple(fill) for collection in ax.collections for fill in collection.get_facecolor()}
    handles, labels = ax.get_legend_handles_labels()
    assert labels == ["A", "B"]
    handle_fills = {tuple(handle.get_facecolor()) for handle in handles}
    assert len(handle_fills) == 2
    assert handle_fills == fills


@pytest.mark.parametrize("plot_kwargs, expected", [
    ({"column": None}, "C0"),
    ({"column": None, "color": "wheat"}, "wheat"),
    ({"facecolor": "none"}, "none"),
])
def test_conceptual_mesh_plot_single_colour_polygon_legend_entry(plot_kwargs, expected):
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id=1, label="A")

    handles, _ = cm.plot(alpha=1, **plot_kwargs).get_legend_handles_labels()

    assert tuple(handles[0].get_facecolor()) == pytest.approx(matplotlib.colors.to_rgba(expected))


@pytest.mark.parametrize("clean", [False, True])
def test_conceptual_mesh_plot_uses_line_colour(clean):
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), zone_id=1)
    cm.add_line(LineString([(0, 2), (10, 2)]), line_id="river", resolution=1, label="River",
                color="navy")
    cm.add_line(LineString([(0, 6), (10, 6)]), line_id="drain", resolution=1, label="Drain",
                color=(0.0, 0.5, 0.0))
    cm.add_line(LineString([(5, 0), (5, 9)]), line_id="fault", resolution=1, is_barrier=True,
                label="Fault")
    if clean:
        cm.generate()

    handles, labels = cm.plot(clean=clean).get_legend_handles_labels()

    colors = {label: tuple(handle.get_color()[0]) for handle, label in zip(handles, labels)}
    to_rgba = matplotlib.colors.to_rgba
    assert colors == {
        "River": pytest.approx(to_rgba("navy")),
        "Drain": pytest.approx(to_rgba((0.0, 0.5, 0.0))),
        "Fault": pytest.approx(to_rgba("tab:red")),
    }


def test_conceptual_mesh_plot_without_labels_has_no_legend_entries():
    _, labels = _model().plot().get_legend_handles_labels()
    assert labels == []


def test_tessellator_plot_draws_cells_and_generators():
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), zone_id=99, densify=0.5)
    clean_polys, _, _ = cm.generate()
    tessellator = VoronoiTessellator(FakeMeshGenerator(clean_polys), cm)
    tessellator.generate()

    ax = tessellator.plot(show_generators=True)

    assert len(ax.collections) == 2
    assert len(ax.collections[1].get_offsets()) == len(tessellator.grid)


def test_tessellator_plot_before_generate_raises():
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), zone_id=99)
    clean_polys, _, _ = cm.generate()
    with pytest.raises(RuntimeError, match="generate"):
        VoronoiTessellator(FakeMeshGenerator(clean_polys), cm).plot()


def test_plot_without_matplotlib_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"vorflow\[plot\]"):
        _model().plot()


@pytest.mark.slow
def test_mesh_generator_plot_draws_elements():
    from vorflow.engine import MeshGenerator

    mesher = MeshGenerator(background_lc=2, verbosity=0, conceptual_mesh=_model())
    mesher.generate()

    ax = mesher.plot(column="zone_id")

    assert ax.collections


def _offset_disu():
    """Two columns of offset layers (see test_disu_full): A 20 | 12 | 4 and B 20 | 16 | 0."""
    from test_disu_full import _two_columns

    return _two_columns(np.array([20.0, 20.0]), [12, 4], [16, 0])


def test_disu_plot_quality_draws_the_connections_of_a_layer():
    from matplotlib.collections import LineCollection, PathCollection

    d = _offset_disu()
    q = d.quality()
    ax = d.plot_quality(0, "ortho_error_3d", quality=q)
    lines = [c for c in ax.collections if isinstance(c, LineCollection) and c.get_array() is not None]
    dots = [c for c in ax.collections if isinstance(c, PathCollection)]
    layer = d.cellmap[d.cellmap.node >= 0].sort_values("node").layer.to_numpy()
    touching = q[(layer[q.n] == 0) | (layer[q.m] == 0)]
    assert len(lines[0].get_segments()) == (touching.ihc != 0).sum()
    assert len(dots[0].get_offsets()) == (touching.ihc == 0).sum()
    assert sorted(lines[0].get_array()) == sorted(touching[touching.ihc != 0].ortho_error_3d)
    assert len(ax.figure.axes) == 2                                      # the colour bar
    ax = d.plot_quality(0, quality=q[q.ihc == 0], colorbar=False)
    assert not [c for c in ax.collections if isinstance(c, LineCollection) and len(c.get_segments())
                and c.get_array() is not None]
    with pytest.raises(ValueError, match="layer"):
        d.plot_quality(5)


def test_disu_plot_quality_section_draws_cells_and_connections():
    from matplotlib.collections import LineCollection, PatchCollection

    d = _offset_disu()
    ax = d.plot_quality_section([(0.0, 0.5), (2.0, 0.5)], "ortho_error_3d")
    boxes = [c for c in ax.collections if isinstance(c, PatchCollection)][0]
    lines = [c for c in ax.collections if isinstance(c, LineCollection)][0]
    assert len(boxes.get_paths()) == d.nodes                            # the line crosses every cell
    assert len(lines.get_segments()) == (d.quality().ihc != 0).sum() + (d.quality().ihc == 0).sum()
    assert max(lines.get_array()) == pytest.approx(np.degrees(np.arctan(8.0)))
