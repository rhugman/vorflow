"""Check benchmark safeguards separately from production meshing behavior."""

import importlib.util
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from vorflow.utils import grid_barrier_report


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_barrier_flow.py"
SPEC = importlib.util.spec_from_file_location("barrier_flow_validation", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_linear_reference_checks_physical_flow_magnitude():
    heads, flow = benchmark.reference_solution([0, 50, 100, 150, 200])
    np.testing.assert_allclose(heads, [20, 17.5, 15, 12.5, 10])
    assert flow == pytest.approx(500)
    # Correct heads and perfect mass balance cannot excuse a conductance error.
    result = benchmark.assess_accuracy(heads, heads, 800, 800, flow, 0)
    assert result["checks"]["head_accuracy"]
    assert result["checks"]["water_budget"]
    assert not result["checks"]["flow_accuracy"]
    assert not result["accepted"]


def test_hfb_reference_has_expected_resistance_and_head_jump():
    heads, flow = benchmark.reference_solution([0, 96, 104, 200], 0.01)
    assert flow == pytest.approx(1000 * 10 / (200 / 10 + 1 / 0.01))
    np.testing.assert_allclose(heads, [20, 19.2, 10.8, 10])
    assert heads[1] - heads[2] > 8


@pytest.mark.parametrize("hydchr", [0, -0.1, np.nan, np.inf])
def test_hfb_reference_rejects_bad_resistance(hydchr):
    with pytest.raises(ValueError, match="Hydraulic characteristic"):
        benchmark.reference_solution([0, 200], hydchr)


def test_strict_baseline_and_practical_repair_acceptance_are_distinct():
    expected = np.asarray([20.0, 15.0, 10.0])
    heads = expected + [0, 1e-4, 0]
    practical = benchmark.assess_accuracy(heads, expected, 500, 500, 500, 0)
    strict = benchmark.assess_accuracy(
        heads, expected, 500, 500, 500, 0, strict_baseline=True
    )
    assert practical["accepted"]
    assert not practical["strict_baseline_accuracy"]
    assert not strict["accepted"]
    assert strict["tolerance_class"] == "strict_baseline"


@pytest.mark.parametrize("failure", ["normal_termination", "expected_end_time"])
def test_fresh_run_health_is_required_even_for_correct_results(failure):
    result = benchmark.assess_accuracy(
        [20, 15, 10], [20, 15, 10], 500, 500, 500, 0, **{failure: False}
    )
    assert not result["accepted"]


def test_nonfinite_and_wrong_shape_heads_do_not_pass():
    for heads in ([20, np.nan, 10], [20, 10]):
        result = benchmark.assess_accuracy(heads, [20, 15, 10], 500, 500, 500, 0)
        assert not result["accepted"]
        assert not result["checks"]["finite_heads"]


def _tracking_fixture():
    polygons = [box(0, 0, 100, 100), box(100, 0, 200, 100)]
    grid = gpd.GeoDataFrame(
        dict(x=[50, 150], y=[50, 50], node_id=[720, 91]),
        geometry=polygons,
        index=["west_source", "east_source"],
    )
    # Segmenting the input creates two interval rows but only one HFB pair.
    lines = gpd.GeoDataFrame(
        dict(line_id=["wall"], is_barrier=[True]),
        geometry=[LineString([(100, 0), (100, 40), (100, 100)])],
    )
    return grid, grid_barrier_report(grid, lines)


def test_hfb_mapping_deduplicates_intervals_and_retains_source_traceability():
    grid, report = _tracking_fixture()
    assert len(report["faces"]) == 2
    pairs = benchmark.exact_hfb_pairs(grid, report)
    assert pairs == [
        dict(cell_a=0, cell_b=1, node_id_a=720, node_id_b=91, length_m=100)
    ]


def test_hfb_mapping_refuses_unmatched_sections_and_wrong_side_centres():
    grid, report = _tracking_fixture()
    bad = {
        key: value.copy() if hasattr(value, "copy") else value
        for key, value in report.items()
    }
    bad["summary"].loc[:, "unmatched_length"] = 1.0
    with pytest.raises(ValueError, match="complete two-sided"):
        benchmark.exact_hfb_pairs(grid, bad)
    report["faces"].loc[:, "centres_opposite"] = False
    with pytest.raises(ValueError, match="opposite sides"):
        benchmark.exact_hfb_pairs(grid, report)


def test_closed_cases_are_exactly_closed_and_shapes_fit_model_domain():
    domain = box(0, 0, benchmark.LENGTH, benchmark.WIDTH)
    for name, case in benchmark.benchmark_cases().items():
        line = LineString(case["coords"])
        assert domain.covers(line), name
        assert line.is_closed == (
            name
            in {"square", "square_small", "square_large", "round", "hexagon", "concave"}
        ), name
        if name not in {"round"}:
            lengths = np.linalg.norm(np.diff(case["coords"], axis=0), axis=1)
            assert lengths.min() >= 4 * case["h"] - 1e-10, name


def test_bent_shared_face_is_rejected_before_starting_mfusg(tmp_path, monkeypatch):
    from vorflow import utils
    if not hasattr(utils, "grid_to_mfusg_disu"):
        pytest.skip("Requires optional MFUSG support from PR #58.")
    pytest.importorskip("flopy")
    polygons = [
        Polygon([(0, 0), (100, 0), (110, 50), (100, 100), (0, 100)]),
        Polygon([(100, 0), (200, 0), (200, 100), (100, 100), (110, 50)]),
    ]
    grid = gpd.GeoDataFrame(
        dict(node_id=[9, 42], x=[0, 200], y=[50, 50]), geometry=polygons
    )
    executable = tmp_path / "unused-executable"
    executable.write_bytes(b"This file must never be executed.")

    def unexpected_solver(*args, **kwargs):
        pytest.fail("Unsupported geometry reached the solver process.")

    monkeypatch.setattr(benchmark, "_run_process", unexpected_solver)
    result = benchmark.run_grid(grid, {}, "mfusg", tmp_path, executable=executable)
    assert not result["accepted"]
    assert "Noncollinear shared face" in result["error"]


def test_unavailable_mfusg_support_is_reported_without_starting_solver(tmp_path, monkeypatch):
    from vorflow import utils

    monkeypatch.delattr(utils, "grid_to_mfusg_disu", raising=False)

    def unexpected_executable(*args, **kwargs):
        pytest.fail("Unavailable MFUSG support reached executable discovery.")

    monkeypatch.setattr(benchmark, "executable_for", unexpected_executable)
    grid = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)])
    result = benchmark.run_grid(grid, {}, "mfusg", tmp_path)
    assert not result["accepted"]
    assert "MFUSG grid export is unavailable" in result["error"]
    assert "--engines mf6" in result["error"]


@pytest.mark.parametrize("engine", ["mfusg", "mf6"])
def test_generated_straight_grid_with_executable_when_explicitly_enabled(
    tmp_path, engine
):
    """Opt in with BARRIER_FLOW_INTEGRATION=1; no downloads or tracked inputs."""
    if os.environ.get("BARRIER_FLOW_INTEGRATION") != "1":
        pytest.skip("Set BARRIER_FLOW_INTEGRATION=1 for actual solver integration.")
    from vorflow import utils
    if engine == "mfusg" and not hasattr(utils, "grid_to_mfusg_disu"):
        pytest.skip("Requires optional MFUSG support from PR #58.")
    pytest.importorskip("flopy")
    executable = benchmark.executable_for(engine)
    grid, report, _, _ = benchmark.generate_grid(
        benchmark.benchmark_cases()["straight"], "straddle", "none", "restore"
    )
    result = benchmark.run_grid(
        grid, report, engine, tmp_path, executable=executable, strict_baseline=True
    )
    assert result["accepted"], result
    if engine == "mf6":
        hfb = benchmark.run_grid(
            grid,
            report,
            engine,
            tmp_path,
            barrier=True,
            executable=executable,
            strict_baseline=True,
        )
        assert hfb["accepted"], hfb
