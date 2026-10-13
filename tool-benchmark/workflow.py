"""Benchmark driver: vorflow vs mf6Voronoi, FloPy (Triangle + VoronoiGrid), VOROGRIDGEN.

Plan: docs/benchmark-plan.md. Configure by editing the flags in ``__main__``
and rerunning (CI calls ``main`` through ``ci.py``). Steps:

1. calibrate -- fit mf6Voronoi's ring multiplier to the spec growth (c0 case).
2. build     -- per case, tool and target cell count: tune the spec scale until
                NCPL is within 5 % (matched count), record metrics, and (with
                run_models) solve the case's MF6 verification problems for every
                centre convention, XT3D off and on.
3. report    -- compile all result rows into CSV tables and draw the figures.

Each (case, tool, target) writes its own JSON rows under ``results/rows`` and
its grid to ``work/<case>/<tool>/n<target>/grid.pkl``, so runs on different
machines (VOROGRIDGEN only runs on Windows) merge by copying files together.
"""

import json
import pickle
import platform
from pathlib import Path

import numpy as np
import pandas as pd

import fetch
from bench.adapters import TOOLS
from bench.case import load_case, size_field
from bench.isolate import run_isolated
from bench.matching import match_count
from bench.metrics import cell_metrics, summarize
from bench.models import run_problem

HERE = Path(__file__).resolve().parent
CASES_D = HERE / "cases"
WORK_D = HERE / "work"
RESULTS_D = HERE / "results"
BIN_D = HERE / ".bin"

CASES = ("c0_point_grading", "v1_linear", "v2_mms_uniform", "v2_mms_graded",
         "v3_thiem", "v4_barrier", "f4_vorflow_demo")
# The other tools' own problems: FloPy autotest grids, VOROGRIDGEN's example and
# mf6Voronoi test cases (the last two fetched into data/ by fetch.py).
OTHER_TOOL_CASES = ("f1_flopy_circle", "f1_flopy_nested_circles", "f2_flopy_polygons",
                    "f2_flopy_many_polygons", "f3_flopy_polygon", "f6_vorogridgen_example",
                    "r1_mf6voronoi_river_aquifer", "r3_mf6voronoi_trench_excavation")
CALIBRATION_CASE = "c0_point_grading"
MULTIPLIERS = (1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0)


def exe_path(bin_d: Path, name: str) -> Path:
    """Executable in bin_d, with the .exe suffix on Windows."""
    return bin_d / (f"{name}.exe" if platform.system() == "Windows" else name)


def tool_kwargs(bin_d: Path, mf6voronoi_multiplier: float | None) -> dict:
    """Per-tool keyword arguments for the adapters' build()."""
    vorogridgen_exe = bin_d / "vorogridgen.exe"
    kwargs = {
        "vorflow": {},
        "vorflow_lloyd": {},
        "mf6voronoi": {},
        "flopy": {"triangle_exe": str(exe_path(bin_d, "triangle"))},
        "vorogridgen": {"exe": str(vorogridgen_exe) if vorogridgen_exe.exists() else None},
    }
    if mf6voronoi_multiplier is not None:
        kwargs["mf6voronoi"]["multiplier"] = mf6voronoi_multiplier
    return kwargs


def calibrate_mf6voronoi(case, multipliers, work_d: Path, results_d: Path) -> float:
    """Pick the ring multiplier whose cell sizes best follow the spec in the growth zone."""
    rows = []
    for m in multipliers:
        result = run_isolated("mf6voronoi", case, 1.0, work_d / f"m{m:g}", {"multiplier": m})
        assert result["status"] == "ok", f"mf6Voronoi calibration failed at multiplier {m}:\n{result['error']}"
        grid = result["grid"]
        growing = size_field(case, grid.xc) < case.h_max
        misfit = float(np.median(np.abs(np.log(cell_metrics(grid, case, 1.0)["h_ratio"][growing]))))
        rows.append({"multiplier": m, "ncpl": grid.ncpl, "median_abs_log_h_ratio": misfit})
    table = pd.DataFrame(rows)
    table.to_csv(results_d / "mf6voronoi_multiplier_calibration.csv", index=False)
    best = float(table.loc[table["median_abs_log_h_ratio"].idxmin(), "multiplier"])
    print(table.to_string(index=False), f"\n-> multiplier {best:g}")
    return best


def build_one(case, tool: str, target: int, kwargs: dict, run_models: bool,
              work_d: Path, rows_d: Path, mf6_exe: str) -> None:
    """Matched-count build of one case/tool/target, its metrics and MF6 problems."""
    ws = work_d / case.id / tool / f"n{target}"
    result = match_count(tool, case, target, ws, kwargs)
    grid = result["grid"]
    row = {"case": case.id, "tool": tool, "target": target, "status": result["status"],
           "matched": result["matched"], "scale": result["scale"], "n_builds": len(result["history"]),
           "n_failed_builds": len(result["failures"]),
           "wall_s": result.get("wall_s"), "peak_rss_mb": result.get("peak_rss_mb"),
           "build_rss_mb": result.get("build_rss_mb"),
           "error": result["error"].strip().splitlines()[-1] if result["error"] else ""}
    mf6_rows = []
    if grid is not None:
        grid.info["scale"] = result["scale"]
        row.update(grid.timings)
        row.update(summarize(grid, case, result["scale"]))
        row.update({f"info_{k}": v for k, v in grid.info.items()})
        ws.mkdir(parents=True, exist_ok=True)
        with open(ws / "grid.pkl", "wb") as f:
            pickle.dump({"grid": grid, "scale": result["scale"], "history": result["history"]}, f)
        if run_models:
            mf6_rows = run_models_on(case, grid, target, ws, mf6_exe)
    key = f"{case.id}__{tool}__n{target}"
    _write_rows(rows_d / "metrics" / f"{key}.json", [row])
    _write_rows(rows_d / "mf6" / f"{key}.json", mf6_rows)
    print(f"{case.id:18s} {tool:12s} n={target:<6d} {result['status']:11s} "
          f"ncpl={getattr(grid, 'ncpl', '-')} scale={result['scale']:.3f} builds={len(result['history'])}")


def run_saved_models(case, target: int, work_d: Path, rows_d: Path, mf6_exe: str) -> None:
    """MF6 problems on the grids saved by an earlier build, replacing their rows."""
    for tool, grid in load_grids(case, work_d, target).items():
        ws = work_d / case.id / tool / f"n{target}"
        _write_rows(rows_d / "mf6" / f"{case.id}__{tool}__n{target}.json",
                    run_models_on(case, grid, target, ws, mf6_exe))
        print(f"{case.id:18s} {tool:12s} n={target:<6d} MF6 rerun on the saved grid")


def run_models_on(case, grid, target: int, ws: Path, mf6_exe: str) -> list:
    """The case's MF6 problems on one grid, for every distinct centre convention."""
    rows = []
    for centres in _distinct_centres(grid):
        for problem in case.mf6_problems:
            for xt3d in (False, True):
                model_ws = ws / f"mf6_{problem['name']}_{centres}_{'xt3d' if xt3d else 'std'}"
                out = run_problem(grid, problem, case, centres, xt3d, model_ws, mf6_exe)
                rows.append({"case": case.id, "tool": grid.tool, "target": target, "ncpl": grid.ncpl,
                             "problem": problem["name"], "centres": centres, "xt3d": xt3d, **out})
    return rows


def _distinct_centres(grid) -> list:
    """Centre conventions that differ from the written one (written always first)."""
    conventions = ["written"]
    for name in ("generator", "centroid"):
        if name == "generator" and grid.generators is None:
            continue
        if not np.allclose(grid.centres(name), grid.xc, rtol=0.0, atol=1e-9):
            conventions.append(name)
    return conventions


def _write_rows(path: Path, rows: list) -> None:
    """Rows as JSON, with numpy scalars converted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1, default=_jsonable))


def _jsonable(value):
    """JSON encoder fallback for numpy scalars and arrays."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def compile_tables(rows_d: Path, results_d: Path) -> tuple:
    """Concatenate all JSON rows into metrics.csv and mf6_verification.csv (with each grid's matched flag)."""
    tables = []
    for name in ("metrics", "mf6"):
        rows = [r for f in sorted((rows_d / name).glob("*.json")) for r in json.loads(f.read_text())]
        assert rows, f"no {name} rows in {rows_d / name}; did the build jobs run?"
        tables.append(pd.DataFrame(rows))
    metrics, mf6 = tables
    key = ["case", "tool", "target"]
    mf6 = mf6.merge(metrics[key + ["matched"]], on=key, how="left")
    metrics.to_csv(results_d / "metrics.csv", index=False)
    mf6.to_csv(results_d / "mf6_verification.csv", index=False)
    return metrics, mf6


def load_grids(case, work_d: Path, target: int) -> dict:
    """Saved grids of one case at one target, keyed by tool, in TOOLS order."""
    grids = {}
    for tool in TOOLS:
        path = work_d / case.id / tool / f"n{target}" / "grid.pkl"
        if path.exists():
            with open(path, "rb") as f:
                grids[tool] = pickle.load(f)["grid"]
    return grids


def make_figures(cases, mf6_table: pd.DataFrame, work_d: Path, figures_d: Path) -> None:
    """Mesh and size figures per case, convergence per sweep case, and the verification summary."""
    import matplotlib

    matplotlib.use("Agg")       # files only; CI runners have no display
    from bench import plots

    for case in cases:
        grids = load_grids(case, work_d, case.targets[0])
        if grids:
            plots.plot_meshes(case, grids, figures_d / f"{case.id}_meshes.png")
            if case.features:
                plots.plot_size_vs_distance(case, grids, figures_d / f"{case.id}_size.png")
        if len(case.targets) > 1 and not mf6_table.empty:
            plots.plot_convergence(mf6_table, case.id, figures_d / f"{case.id}_convergence.png")
    if not mf6_table.empty:
        plots.plot_verification(mf6_table, [c.id for c in cases if c.tier == 0],
                                figures_d / "verification_summary.png")


def main(cases=CASES + OTHER_TOOL_CASES, tools=TOOLS, calibrate=False, build=True, run_models=True, report=True,
         rerun_models=False):
    """Run the benchmark steps selected by the flags.

    ``rerun_models`` reruns the MF6 problems on the grids saved in work/ (with
    ``build=False``), e.g. after a change to bench/models.py.
    """
    RESULTS_D.mkdir(parents=True, exist_ok=True)
    fetch.ensure()
    loaded = [load_case(CASES_D / f"{case_id}.yml") for case_id in cases]
    multiplier = None
    if calibrate:
        cal_case = load_case(CASES_D / f"{CALIBRATION_CASE}.yml")
        multiplier = calibrate_mf6voronoi(cal_case, MULTIPLIERS, WORK_D / "calibration", RESULTS_D)
    if build:
        kwargs = tool_kwargs(BIN_D, multiplier)
        for case in loaded:
            for tool in tools:
                for target in case.targets:
                    build_one(case, tool, target, kwargs[tool], run_models,
                              WORK_D, RESULTS_D / "rows", str(exe_path(BIN_D, "mf6")))
    elif rerun_models:
        for case in loaded:
            for target in case.targets:
                run_saved_models(case, target, WORK_D, RESULTS_D / "rows", str(exe_path(BIN_D, "mf6")))
    if report:
        _, mf6_table = compile_tables(RESULTS_D / "rows", RESULTS_D)
        make_figures(loaded, mf6_table, WORK_D, RESULTS_D / "figures")


if __name__ == "__main__":
    main()
