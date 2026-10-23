"""USG-specific export geometry, serialization and optional FloPy integration."""

import os
import shutil
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from vorflow import ConceptualMesh, MeshGenerator, ThresholdField, VoronoiTessellator
from vorflow.utils import grid_to_gsf, grid_to_mfusg_disu

from test_pipeline import FakeMeshGenerator


def _grid(polygons=None, centres=None):
    polygons = polygons if polygons is not None else [box(0, 0, 1, 1), box(1, 0, 2, 1)]
    centres = centres if centres is not None else [p.centroid.coords[0] for p in polygons]
    return gpd.GeoDataFrame({
        "node_id": np.arange(10, 10 + len(polygons)),
        "zone_id": ["aquifer"] * len(polygons),
        "x": [p[0] for p in centres], "y": [p[1] for p in centres],
    }, geometry=polygons, index=["original"] * len(polygons))


def _rows(disu, key):
    return np.split(disu[key], np.cumsum(disu["iac"])[:-1])


def test_usg_arrays_for_two_cells_with_asymmetric_distances():
    grid = _grid(centres=[(0.25, 0.5), (1.5, 0.5)])
    disu = grid_to_mfusg_disu(grid, top=10, botm=0)
    assert disu["nodes"] == 2 and disu["njag"] == 4
    assert disu["ivsd"] == -1 and disu["idsymrd"] == 0
    assert {"ihc", "hwva", "nja", "botm"}.isdisjoint(disu)
    np.testing.assert_array_equal(disu["iac"], [2, 2])
    np.testing.assert_array_equal(disu["ja"], [0, 1, 1, 0])
    np.testing.assert_allclose(disu["cl12"], [0, 0.75, 0, 0.5])
    np.testing.assert_allclose(disu["fahl"], [0, 10, 0, 10])
    np.testing.assert_allclose(disu["area"], [1, 1])
    np.testing.assert_array_equal(disu["node_id"], [10, 11])
    assert disu["cell_map"]["orig_index"].tolist() == ["original", "original"]
    assert disu["cell_map"]["cell_id"].tolist() == [0, 1]
    assert disu["connectivity"]["ortho_error"].max() == 0


def test_multilayer_connections_use_mean_thickness_and_own_half_lengths():
    disu = grid_to_mfusg_disu(_grid(), top=[10, 12], botm=[[4, 5], [-1, -4]], nlay=2)
    np.testing.assert_array_equal(disu["nodelay"], [2, 2])
    np.testing.assert_array_equal(disu["iac"], [3, 3, 3, 3])
    np.testing.assert_array_equal(disu["ja"], [0, 1, 2, 1, 0, 3, 2, 0, 3, 3, 1, 2])
    np.testing.assert_allclose(disu["top"], [10, 12, 4, 5])
    np.testing.assert_allclose(disu["bot"], [4, 5, -1, -4])
    np.testing.assert_allclose(disu["cl12"], [0, .5, 3, 0, .5, 3.5, 0, 2.5, .5, 0, 4.5, .5])
    np.testing.assert_allclose(disu["fahl"], [0, 6.5, 1, 0, 6.5, 1, 0, 1, 7, 0, 1, 7])
    assert disu["cell_map"]["model_node"].tolist() == [0, 1, 2, 3]
    assert disu["cell_map"]["layer"].tolist() == [0, 0, 1, 1]
    assert disu["node_id"].tolist() == [10, 11, 10, 11]
    assert disu["area"].shape == (2,)  # IVSD=-1 reads one layer of areas.
    neighbours = _rows(disu, "ja")
    areas = _rows(disu, "fahl")
    for n, row in enumerate(neighbours):
        assert row[0] == n
        assert list(row[1:]) == sorted(row[1:])
        for pos, m in enumerate(row[1:], 1):
            reverse = np.flatnonzero(neighbours[m] == n)[0]
            assert areas[n][pos] == areas[m][reverse]
    assert disu["iac"].sum() == disu["njag"] == len(disu["ja"])


def test_no_corner_connection_and_single_cell_grid():
    disu = grid_to_mfusg_disu(_grid([box(0, 0, 1, 1), box(1, 1, 2, 2)]), 10, 0)
    np.testing.assert_array_equal(disu["iac"], [1, 1])
    assert disu["connectivity"].empty
    single = grid_to_mfusg_disu(_grid([box(0, 0, 1, 1)]), 10, 0)
    np.testing.assert_array_equal(single["ja"], [0])


def test_hanging_vertex_and_contiguous_collinear_face_segments(tmp_path):
    tall = Polygon([(0, 0), (1, 0), (1, .5), (1, 2), (0, 2)])
    grid = _grid([tall, box(1, 0, 2, 1), box(1, 1, 2, 2)])
    disu = grid_to_mfusg_disu(grid, 10, 0)
    np.testing.assert_array_equal(disu["iac"], [3, 3, 3])
    assert sorted(disu["connectivity"]["shared_length"]) == [1, 1, 1]
    path = grid_to_gsf(grid, tmp_path / "hanging.gsf", 10, 0)
    vertices, cells = _read_gsf(path)
    tall_xy = vertices[np.asarray(cells[0][6:], dtype=int) - 1, :2]
    assert np.any(np.all(tall_xy == (1, 1), axis=1))


def test_nonorthogonality_is_diagnosed_without_replacing_centres(tmp_path):
    grid = _grid(centres=[(.5, .2), (1.5, .8)])
    disu = grid_to_mfusg_disu(grid, 10, 0)
    assert disu["connectivity"]["ortho_error"].iloc[0] == pytest.approx(np.degrees(np.arctan(.6)))
    _, cells = _read_gsf(grid_to_gsf(grid, tmp_path / "centres.gsf", 10, 0))
    np.testing.assert_allclose(np.array(cells, dtype=object)[:, 1:3].astype(float), grid[["x", "y"]])


@pytest.mark.parametrize("offset", [0, 5_000_000])
def test_boundary_roundoff_preserves_centres_but_rejects_real_errors(tmp_path, offset):
    eps = np.spacing(float(offset + 1))
    polygon = box(offset, offset, offset + 1, offset + 1)
    centre = (offset - 2 * eps, offset + .5)
    grid = _grid([polygon], [centre])
    assert not polygon.covers(Point(centre))
    disu = grid_to_mfusg_disu(grid, 10, 0)
    assert disu["nodes"] == 1
    _, cells = _read_gsf(grid_to_gsf(grid, tmp_path / "roundoff.gsf", 10, 0))
    np.testing.assert_array_equal(np.array(cells, dtype=object)[:, 1:3].astype(float), [centre])
    bad = _grid([polygon], [(offset - 1e-4, offset + .5)])
    with pytest.raises(ValueError, match="outside"):
        grid_to_mfusg_disu(bad, 10, 0)


@pytest.mark.parametrize("offset", [0, 5_000_000])
def test_roundoff_overlap_keeps_reciprocal_face_and_rejects_real_overlap(offset):
    eps = np.spacing(float(offset + 1))
    grid = _grid([
        box(offset, offset, offset + 1, offset + 1),
        box(offset + 1 - 2 * eps, offset, offset + 2, offset + 1),
    ])
    assert grid.geometry.iloc[0].intersection(grid.geometry.iloc[1]).area > 0
    original = grid.copy(deep=True)
    disu = grid_to_mfusg_disu(grid, 10, 0)
    np.testing.assert_array_equal(disu["ja"], [0, 1, 1, 0])
    np.testing.assert_allclose(disu["fahl"], [0, 10, 0, 10], rtol=1e-6)
    np.testing.assert_allclose(disu["cl12"], [0, .5, 0, .5], rtol=1e-6)
    assert grid.equals(original)
    grid.iloc[1, grid.columns.get_loc("geometry")] = box(
        offset + 1 - 1e-4, offset, offset + 2, offset + 1,
    )
    with pytest.raises(ValueError, match="Overlapping"):
        grid_to_mfusg_disu(grid, 10, 0)


def test_overlap_tolerance_does_not_connect_corners_or_hide_small_deep_overlaps():
    eps = np.spacing(1.0)
    corner = _grid([box(0, 0, 1, 1), box(1 - eps, 1 - eps, 2, 2)])
    disu = grid_to_mfusg_disu(corner, 10, 0)
    np.testing.assert_array_equal(disu["iac"], [1, 1])
    # Small area alone does not imply coordinate roundoff.
    deep = _grid([box(0, 0, 1, 1), box(1 - 1e-7, 1 - 1e-7, 2, 2)])
    with pytest.raises(ValueError, match="Overlapping"):
        grid_to_mfusg_disu(deep, 10, 0)


@pytest.mark.slow
def test_slanted_concave_domain_exports_with_default_boundary_centres(tmp_path):
    cm = ConceptualMesh()
    cm.add_polygon(Polygon([(0, 0), (300, 0), (300, 100), (120, 130),
                            (300, 300), (0, 300)]), zone_id=1, resolution=20)
    mg = MeshGenerator(background_lc=20, conceptual_mesh=cm)
    mg.generate()
    vt = VoronoiTessellator(mg, cm)
    grid = vt.generate()
    original = grid.copy(deep=True)
    disu = vt.to_mfusg_disu(10, 0)
    assert disu["nodes"] == len(grid)
    _, cells = _read_gsf(vt.to_gsf(tmp_path / "concave.gsf", 10, 0))
    np.testing.assert_array_equal(np.array([cell[1:3] for cell in cells], dtype=float), grid[["x", "y"]])
    assert grid.equals(original)


@pytest.mark.parametrize("nlay", [0, -1, True, 1.5])
def test_bad_layer_count(nlay):
    with pytest.raises(ValueError, match="positive integer"):
        grid_to_mfusg_disu(_grid(), 10, 0, nlay=nlay)


@pytest.mark.parametrize("top, botm, nlay, match", [
    ([10], 0, 1, "top.*shape"),
    (10, [0, -5, -10], 2, "botm.*shape"),
    (np.nan, 0, 1, "finite"),
    (10, np.inf, 1, "finite"),
    (1e308, -1e308, 1, "finite"),
    (10, 10, 1, "thickness"),
    (10, 11, 1, "thickness"),
    (10, [0, 0], 2, "pinch-outs"),
    (10, 0, 2, "thickness"),
])
def test_bad_surfaces(top, botm, nlay, match):
    with pytest.raises(ValueError, match=match):
        grid_to_mfusg_disu(_grid(), top, botm, nlay)


@pytest.mark.parametrize("polygons, centres, match", [
    ([box(0, 0, 2, 2), box(1, 0, 3, 2)], [(1, 1), (2, 1)], "Overlapping"),
    ([Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])], [(.5, .5)], "valid"),
    ([Polygon()], [(0, 0)], "non-empty"),
    ([box(0, 0, 1, 1)], [(2, 2)], "outside"),
    ([box(0, 0, 1, 1)], [(np.nan, .5)], "finite"),
    ([box(0, 0, 3, 3).difference(box(1, 1, 2, 2))], [(.5, .5)], "holes"),
    ([box(0, 0, 1, 1).union(box(2, 0, 3, 1))], [(.5, .5)], "single polygon"),
    ([box(0, 0, 1, 1), box(1, 0, 2, 1)], [(1, .5), (1.5, .5)], "opposite sides"),
])
def test_bad_geometry(polygons, centres, match):
    with pytest.raises(ValueError, match=match):
        grid_to_mfusg_disu(_grid(polygons, centres), 10, 0)


def test_missing_columns_empty_and_no_node_id():
    grid = _grid()
    with pytest.raises(ValueError, match="missing required"):
        grid_to_mfusg_disu(grid.drop(columns="x"), 10, 0)
    with pytest.raises(ValueError, match="non-empty"):
        grid_to_mfusg_disu(grid.iloc[:0], 10, 0)
    disu = grid_to_mfusg_disu(grid.drop(columns="node_id"), 10, 0)
    assert disu["node_id"].tolist() == [0, 1]


def test_bent_and_disconnected_shared_faces_are_rejected():
    lower_left = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    upper_right = box(0, 0, 2, 2).difference(lower_left)
    with pytest.raises(ValueError, match="Noncollinear"):
        grid_to_mfusg_disu(_grid([lower_left, upper_right], [(.5, .5), (1.5, 1.5)]), 10, 0)
    notched = Polygon([(0, 0), (1, 0), (1, 1), (.5, 1), (.5, 2), (1, 2), (1, 3), (0, 3)])
    with pytest.raises(ValueError, match="Disconnected"):
        grid_to_mfusg_disu(_grid([notched, box(1, 0, 2, 3)], [(.25, 1.5), (1.5, 1.5)]), 10, 0)


def _read_gsf(path):
    lines = path.read_text().splitlines()
    assert lines[0] == "UNSTRUCTURED"
    nodes, nlay, iz, ic = map(int, lines[1].split())
    assert nlay > 0 and iz == ic == 1
    nvert = int(lines[2])
    vertices = np.array([list(map(float, line.split())) for line in lines[3:3 + nvert]])
    cells = [line.split() for line in lines[3 + nvert:]]
    assert len(cells) == nodes
    return vertices, cells


@pytest.mark.parametrize("width", [.01, .0001])
def test_original_3d_gsf_round_trip_and_large_coordinates(tmp_path, width):
    flopy = pytest.importorskip("flopy", minversion="3.10")
    x0, y0 = 500000., 5000000.
    grid = _grid([box(x0, y0, x0 + width, y0 + width),
                  box(x0 + width, y0, x0 + 2 * width, y0 + width)])
    before = grid.copy(deep=True)
    top, botm = [10, 11], [[4, 5], [-2, -1]]
    path = grid_to_gsf(grid, tmp_path / "model.gsf", top, botm, nlay=2)
    vertices, cells = _read_gsf(path)
    assert len(set(map(tuple, vertices))) == len(vertices)
    imported = flopy.discretization.UnstructuredGrid.from_gridspec(path)
    np.testing.assert_array_equal(imported.ncpl, [2, 2])
    np.testing.assert_allclose(imported.xcellcenters, np.tile(grid.x, 2), rtol=0, atol=1e-10)
    disu = grid_to_mfusg_disu(grid, top, botm, nlay=2)
    np.testing.assert_allclose(imported.top, disu["top"])
    np.testing.assert_allclose(imported.botm, disu["bot"])
    for n, cell in enumerate(cells):
        assert int(cell[0]) == n + 1 and int(cell[4]) == n // 2 + 1
        ids = np.asarray(cell[6:], dtype=int) - 1
        assert int(cell[5]) == len(ids)
        half = len(ids) // 2
        for ring_ids in (ids[:half], ids[half:]):
            xy = vertices[ring_ids, :2]
            local = xy - xy[0]
            assert np.sum(local[:, 0] * np.roll(local[:, 1], -1)
                          - np.roll(local[:, 0], -1) * local[:, 1]) < 0
            assert Polygon(xy).equals(grid.geometry.iloc[n % 2])
        assert float(cell[3]) == pytest.approx((disu["top"][n] + disu["bot"][n]) / 2)
    assert grid.equals(before)


def _tessellator(generate=True):
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 4, 3), zone_id=1, densify=.5)
    clean_polys, _, _ = cm.generate()
    vt = VoronoiTessellator(FakeMeshGenerator(clean_polys, spacing=.5), cm)
    if generate:
        vt.generate()
    return vt


@pytest.mark.parametrize("method, args", [
    ("to_mfusg_disu", (10, 0)), ("to_gsf", ("unused.gsf", 10, 0)),
])
def test_exports_before_generate_raise(method, args):
    with pytest.raises(RuntimeError, match="generate"):
        getattr(_tessellator(False), method)(*args)


def test_plain_exports_work_without_flopy(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "flopy", None)
    vt = _tessellator()
    disu = vt.to_mfusg_disu(10, 0)
    assert disu["nodes"] == len(vt.grid)
    assert vt.to_gsf(tmp_path / "model.gsf", 10, 0).is_file()
    with pytest.raises(ImportError, match="to_mfusg_disu"):
        vt.to_mfusgdisu(None, 10, 0)


def test_flopy_disu_round_trip_and_independent_results(tmp_path):
    flopy = pytest.importorskip("flopy", minversion="3.10")
    vt = _tessellator()
    model = flopy.mfusg.MfUsg(modelname="vorflow", structured=False, model_ws=tmp_path)
    original = vt.to_mfusg_disu(10, [0, -10], nlay=2)
    disu = vt.to_mfusgdisu(model, 10, [0, -10], nlay=2, nper=2, perlen=[1, 2], steady=[True, False])
    disu.write_file()
    loaded_model = flopy.mfusg.MfUsg(structured=False, model_ws=tmp_path)
    loaded = flopy.mfusg.MfUsgDisU.load(disu.fn_path, loaded_model)
    np.testing.assert_array_equal(loaded.ja.array, original["ja"] + 1)
    np.testing.assert_array_equal(loaded.iac.array, original["iac"])
    np.testing.assert_allclose(loaded.cl12.array, original["cl12"], rtol=2e-6)
    np.testing.assert_allclose(loaded.fahl.array, original["fahl"], rtol=2e-6)
    np.testing.assert_allclose(loaded.top.array.ravel(), original["top"])
    assert loaded.nper == 2
    original["ja"][:] = -9
    original["cell_map"].loc[:, "node_id"] = -9
    assert vt.to_mfusg_disu(10, [0, -10], 2)["ja"][0] == 0
    assert vt.to_mfusg_disu(10, [0, -10], 2)["cell_map"]["node_id"].min() >= 0


def test_wrapper_rejects_mf6_and_grid_overrides(tmp_path):
    flopy = pytest.importorskip("flopy", minversion="3.10")
    vt = _tessellator()
    mf6 = flopy.mf6.ModflowGwf(flopy.mf6.MFSimulation(sim_ws=tmp_path))
    with pytest.raises(ValueError, match="not an MF6"):
        vt.to_mfusgdisu(mf6, 10, 0)
    model = flopy.mfusg.MfUsg(structured=False, model_ws=tmp_path)
    with pytest.raises(ValueError, match="override"):
        vt.to_mfusgdisu(model, 10, 0, ivsd=1)
    with pytest.raises(ValueError, match="Confining beds"):
        vt.to_mfusgdisu(model, 10, 0, laycbd=1)


def _run_linear_model(tmp_path, props, ibound, exact):
    flopy = pytest.importorskip("flopy", minversion="3.10")
    executable = os.environ.get("MFUSG_EXE") or shutil.which("mfusg")
    if not executable:
        pytest.skip("MFUSG executable not installed (set MFUSG_EXE or PATH).")
    model = flopy.mfusg.MfUsg(
        modelname="linear", structured=False, model_ws=tmp_path, exe_name=executable,
    )
    metadata = {"node_id", "cell_map", "connectivity"}
    flopy.mfusg.MfUsgDisU(model, **{k: v for k, v in props.items() if k not in metadata})
    flopy.mfusg.MfUsgBas(model, ibound=ibound, strt=exact)
    flopy.mfusg.MfUsgLpf(model, laytyp=0, hk=10, vka=10, ipakcb=53)
    # The confined isotropic system is symmetric. CG also handles the
    # single-unknown vertical fixture, where USG 1.5's BCGS oscillates.
    flopy.mfusg.MfUsgSms(model, clin="cg", hclose=1e-8, hiclose=1e-8,
                       rclosepcgu=1e-8, mxiter=100, iter1=500)
    flopy.modflow.ModflowOc(model, stress_period_data={(0, 0): ["save head", "print budget"]})
    # Perturb interior starting heads so the solver must recover the solution.
    strt = exact.copy()
    strt[ibound == 1] = 12.0
    model.bas6.strt = strt
    model.write_input()
    success, output = model.run_model(silent=True, report=True)
    assert success, "\n".join(output)
    heads = np.concatenate(flopy.utils.HeadUFile(tmp_path / "linear.hds").get_data())
    np.testing.assert_allclose(heads, exact, rtol=0, atol=1e-5)
    incremental, _ = flopy.utils.MfusgListBudget(tmp_path / "linear.list").get_budget()
    assert np.isfinite(incremental["PERCENT_DISCREPANCY"]).all()
    assert np.abs(incremental["PERCENT_DISCREPANCY"]).max() < .1
    return heads, incremental


@pytest.mark.parametrize("direction", ["horizontal", "vertical"])
def test_layered_linear_flow_with_mfusg_executable(tmp_path, direction):
    """An independent analytic check of horizontal and vertical conductances."""
    polygons = ([box(i, j, i + 1, j + 1) for j in range(3) for i in range(3)]
                if direction == "horizontal" else [box(0, 0, 1, 1)])
    grid = _grid(polygons)
    bottoms = [0, -5] if direction == "horizontal" else [0, -5, -25]
    props = grid_to_mfusg_disu(grid, 10, bottoms, nlay=len(bottoms))
    if direction == "horizontal":
        x = np.tile(grid.x.to_numpy(), 2)
        exact = 20.0 - 5.0 * (x - .5)
        ibound = np.where((x == .5) | (x == 2.5), -1, 1)
    else:
        # Centre elevations 5, -2.5, -15; distances 7.5 and 12.5 m.
        # Series resistance places the interior head at 16.25, not 15 m.
        exact = np.array([20.0, 16.25, 10.0])
        ibound = np.array([-1, 1, -1])
    _, incremental = _run_linear_model(tmp_path, props, ibound, exact)
    # Heads and budget balance alone cannot detect uniform conductance errors.
    # Q = K * A * delta_h / distance_between_fixed_head_centres.
    expected_flow = 10 * (3 * 15) * 10 / 2 if direction == "horizontal" else 10 * 1 * 10 / 20
    np.testing.assert_allclose(incremental["CONSTANT_HEAD_IN"], expected_flow, rtol=1e-5)
    np.testing.assert_allclose(incremental["CONSTANT_HEAD_OUT"], expected_flow, rtol=1e-5)


@pytest.mark.slow
def test_refined_multilayer_voronoi_flow_with_mfusg_executable(tmp_path):
    """Exercise Gmsh, Voronoi geometry, DISU and USG on one refined model."""
    pytest.importorskip("flopy", minversion="3.10")
    if not (os.environ.get("MFUSG_EXE") or shutil.which("mfusg")):
        pytest.skip("MFUSG executable not installed (set MFUSG_EXE or PATH).")
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 200, 100), zone_id=1, resolution=20)
    cm.add_point(Point(100, 50), "refinement", resolution=5,
                 fields=[ThresholdField(size_min=5, dist_min=10, dist_max=60)])
    mg = MeshGenerator(background_lc=20, conceptual_mesh=cm)
    mg.generate()
    vt = VoronoiTessellator(mg, cm)
    grid = vt.generate()
    assert grid.geometry.area.max() / grid.geometry.area.min() > 4
    props = vt.to_mfusg_disu(10, [0, -5, -25], nlay=3)
    x = np.tile(grid.x.to_numpy(), 3)
    # Boundary generators are at x=0/200, so the imposed gradient spans 200 m.
    left, right = np.isclose(x, 0), np.isclose(x, 200)
    assert left.any() and right.any()
    ibound = np.where(left | right, -1, 1)
    exact = 20 - x / 20
    _, incremental = _run_linear_model(tmp_path, props, ibound, exact)
    # Three layers, total thickness 35 m, width 100 m, delta_h=10 m.
    expected_flow = 10 * (100 * 35) * 10 / 200
    np.testing.assert_allclose(incremental["CONSTANT_HEAD_IN"], expected_flow, rtol=1e-5)
    np.testing.assert_allclose(incremental["CONSTANT_HEAD_OUT"], expected_flow, rtol=1e-5)
