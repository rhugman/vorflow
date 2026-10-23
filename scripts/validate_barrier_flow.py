"""Run reproducible head *and* flow checks on genuinely generated barrier grids.

The unmodified confined problem has an exact linear solution. A second problem
puts a zero-thickness, low-conductance HFB across the full model width; its exact
solution includes the head jump and barrier resistance. The latter is deliberately
restricted to a straight, completely matched barrier: assigning only its matched
pieces would silently leave hydraulic gaps.

Example (downloads stay outside this script)::

    python scripts/validate_barrier_flow.py --engines mf6

Set MFUSG_EXE/MF6_EXE, put executables on PATH, or place them in .bin/. Results,
model inputs, list files, heads and budgets are retained in a fresh directory under
.bin/barrier-system-flow/. No cached model outputs are reused. FloPy is optional
until a simulation is requested. The script pins imports to this checkout because
an editable environment may point at a different worktree.

This is a small benchmark, not certification for arbitrary geometries, anisotropy,
unconfined flow or different layer meshes. MFUSG DISU and MF6 DISV remain separate
export paths. HFB is tested with MF6 only; no MFUSG HFB wrapper is assumed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import warnings

import numpy as np
import shapely
from shapely.geometry import LineString, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator, set_verbosity  # noqa: E402
from vorflow.utils import grid_to_disv  # noqa: E402


LENGTH = 200.0
WIDTH = 100.0
THICKNESS = 10.0
CONDUCTIVITY = 10.0
LEFT_HEAD = 20.0
RIGHT_HEAD = 10.0
HYDRAULIC_CHARACTERISTIC = 0.01  # K_barrier / thickness_barrier, day^-1
HEAD_ATOL = 1e-3  # m; 0.01% of the imposed head difference
FLOW_RTOL = 5e-3  # 0.5%; checks physical magnitude, independently of balance
STRICT_HEAD_ATOL = 1e-5
STRICT_FLOW_RTOL = 1e-5
BUDGET_ATOL = 0.1  # percent, rather than fraction
CLOSED_PUMPING = 50.0  # m3/day, uniformly distributed inside the 64 m square
CLOSED_DRAWDOWN_RTOL = 0.02
CLOSED_REFINEMENT_RTOL = 0.005
CLOSED_SAMPLE_HEAD_ATOL = 0.05  # m, including nearest refined-centre sampling


def source_hashes():
    """Identify the exact dirty-worktree Python sources behind this evidence."""
    files = [Path(__file__), *(ROOT / "src" / "vorflow").glob("*.py")]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }


LOADED_SOURCE_HASHES = source_hashes()


def _flopy():
    try:
        import flopy
    except ImportError as error:
        raise ImportError(
            "Flow benchmarks need FloPy; install the examples extra with "
            "`python -m pip install -e '.[examples]'`."
        ) from error
    return flopy


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def reference_solution(x, hydraulic_characteristic=None):
    """Return heads and total Q for confined uniform flow, optionally with HFB.

    Parameters are in metres and days. The HFB lies at x=LENGTH/2, spans WIDTH,
    and has hydraulic characteristic K_b/b. There is no recharge or pumping;
    the other two boundaries are no-flow. Head at the ideal discontinuity itself
    is undefined, so callers must not place an HFB cell centre there.
    """
    x = np.asarray(x, dtype=float)
    if hydraulic_characteristic is not None and (
        not np.isfinite(hydraulic_characteristic) or hydraulic_characteristic <= 0
    ):
        raise ValueError("Hydraulic characteristic must be finite and positive.")
    barrier_resistance = (
        0.0 if hydraulic_characteristic is None else 1.0 / hydraulic_characteristic
    )
    specific_discharge = (LEFT_HEAD - RIGHT_HEAD) / (
        LENGTH / CONDUCTIVITY + barrier_resistance
    )
    head = LEFT_HEAD - specific_discharge * (
        x / CONDUCTIVITY + barrier_resistance * (x > LENGTH / 2)
    )
    return head, specific_discharge * WIDTH * THICKNESS


def assess_accuracy(
    heads,
    expected_heads,
    inflow,
    outflow,
    expected_flow,
    budget_percent,
    *,
    normal_termination=True,
    expected_end_time=True,
    strict_baseline=False,
):
    """Apply independent numerical, head-error and total-flow acceptance checks."""
    heads = np.asarray(heads, float)
    expected_heads = np.asarray(expected_heads, float)
    same_shape = heads.shape == expected_heads.shape and heads.size > 0
    finite = same_shape and bool(np.isfinite(heads).all())
    error = heads - expected_heads if finite else np.asarray([np.inf])
    max_error = float(np.abs(error).max())
    flow_values = np.asarray([inflow, outflow, expected_flow], float)
    flow_finite = bool(np.isfinite(flow_values).all()) and expected_flow > 0
    relative_in = (
        float(abs(inflow - expected_flow) / expected_flow) if flow_finite else None
    )
    relative_out = (
        float(abs(outflow - expected_flow) / expected_flow) if flow_finite else None
    )
    budget_ok = bool(np.isfinite(budget_percent) and abs(budget_percent) < BUDGET_ATOL)
    head_tolerance = STRICT_HEAD_ATOL if strict_baseline else HEAD_ATOL
    flow_tolerance = STRICT_FLOW_RTOL if strict_baseline else FLOW_RTOL
    checks = dict(
        normal_termination=bool(normal_termination),
        expected_end_time=bool(expected_end_time),
        finite_heads=finite,
        boundary_head_range=bool(
            finite
            and heads.min() >= RIGHT_HEAD - HEAD_ATOL
            and heads.max() <= LEFT_HEAD + HEAD_ATOL
        ),
        head_accuracy=max_error <= head_tolerance,
        flow_accuracy=bool(
            flow_finite
            and relative_in <= flow_tolerance
            and relative_out <= flow_tolerance
        ),
        water_budget=budget_ok,
    )
    return dict(
        accepted=all(checks.values()),
        tolerance_class="strict_baseline" if strict_baseline else "practical_repaired",
        head_tolerance_m=head_tolerance,
        flow_tolerance_relative=flow_tolerance,
        strict_baseline_accuracy=bool(
            max_error <= STRICT_HEAD_ATOL
            and flow_finite
            and relative_in <= STRICT_FLOW_RTOL
            and relative_out <= STRICT_FLOW_RTOL
        ),
        checks=checks,
        max_head_error_m=max_error if np.isfinite(max_error) else None,
        rms_head_error_m=float(np.sqrt(np.mean(error**2))) if finite else None,
        inflow_m3_day=float(inflow) if np.isfinite(inflow) else None,
        outflow_m3_day=float(outflow) if np.isfinite(outflow) else None,
        expected_flow_m3_day=float(expected_flow),
        inflow_relative_error=relative_in,
        outflow_relative_error=relative_out,
        budget_discrepancy_percent=(
            float(budget_percent) if np.isfinite(budget_percent) else None
        ),
    )


def exact_hfb_pairs(grid, report, *, expected_length=WIDTH):
    """Deduplicate real, complete shared faces and preserve source IDs in mapping.

    Report intervals may split one shared face at multiple source vertices. A
    package entry is needed once per cell pair, not once per interval. Reject
    incomplete coverage, crossing cells, centres on the barrier, and a pair whose
    actual shared face extends beyond the tracked feature.
    """
    summary = report["summary"]
    if len(summary) != 1:
        raise ValueError("This analytical HFB benchmark needs one barrier.")
    tolerance = max(float(report["tolerance"]), 1e-7)
    if (
        len(report["crossings"])
        or abs(float(summary.iloc[0].matched_length) - expected_length) > tolerance
        or float(summary.iloc[0].unmatched_length) > tolerance
    ):
        raise ValueError("HFB requires complete two-sided coverage without crossings.")
    faces = report["faces"]
    if faces.empty or not bool(faces.centres_opposite.all()):
        raise ValueError("HFB requires numerical centres strictly on opposite sides.")
    mapped = {}
    for _, face in faces.iterrows():
        pair = tuple(sorted((int(face.left_cell), int(face.right_cell))))
        if pair[0] == pair[1] or pair[0] < 0 or pair[1] >= len(grid):
            raise ValueError("Invalid HFB model-row pair.")
        mapped.setdefault(pair, []).append(face.geometry)
    result = []
    for (a, b), intervals in sorted(mapped.items()):
        shared = grid.geometry.iloc[a].boundary.intersection(
            grid.geometry.iloc[b].boundary
        )
        tracked = shapely.union_all(intervals)
        if (
            shared.length <= tolerance
            or shared.difference(tracked.buffer(tolerance)).length > tolerance
            or tracked.difference(shared.buffer(tolerance)).length > tolerance
        ):
            raise ValueError(
                "HFB pair does not represent exactly one tracked shared face."
            )
        result.append(
            dict(
                cell_a=a,
                cell_b=b,
                node_id_a=int(grid.iloc[a].node_id),
                node_id_b=int(grid.iloc[b].node_id),
                length_m=float(shared.length),
            )
        )
    if abs(sum(row["length_m"] for row in result) - expected_length) > 2 * tolerance:
        raise ValueError("HFB pairs overlap, omit, or duplicate feature length.")
    return result


def benchmark_cases():
    """Common logical shapes; physical feature sizes are at least four local h."""
    rounded = [
        (100 + 32 * np.cos(a), 50 + 32 * np.sin(a))
        for a in np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ]
    rounded.append(rounded[0])  # exact closure, not sin(2*pi) roundoff
    hexagon = [
        (100 + 32 * np.cos(a), 50 + 32 * np.sin(a))
        for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ]
    hexagon.append(hexagon[0])
    return {
        "straight": dict(h=8.0, coords=[(100, 0), (100, 100)]),
        "near_straight": dict(h=8.0, coords=[(20, 40), (100, 45), (180, 40)]),
        "gentle": dict(h=8.0, coords=[(20, 30), (100, 50), (180, 30)]),
        "right_angle": dict(h=8.0, coords=[(40, 20), (100, 20), (100, 80)]),
        # Rui's original zigzag, uniformly scaled by 1/3 and translated in x.
        # Its shortest leg is 9.43 m, so h=2 m respects the four-cell-size rule.
        "zigzag": dict(
            h=2.0,
            coords=[
                (50 + x / 3, y / 3)
                for x, y in [(30, 40), (140, 170), (160, 150), (270, 260)]
            ],
        ),
        "square": dict(
            h=8.0,
            coords=[(68, 18), (132, 18), (132, 82), (68, 82), (68, 18)],
        ),
        "square_small": dict(
            h=8.0,
            coords=[(84, 34), (116, 34), (116, 66), (84, 66), (84, 34)],
        ),
        "square_large": dict(
            h=8.0,
            coords=[(60, 10), (140, 10), (140, 90), (60, 90), (60, 10)],
        ),
        "hexagon": dict(h=8.0, coords=hexagon),
        "concave": dict(
            h=8.0,
            coords=[
                (60, 10),
                (140, 10),
                (140, 42),
                (92, 42),
                (92, 90),
                (60, 90),
                (60, 10),
            ],
        ),
        "round": dict(
            h=8.0,
            coords=rounded,
        ),
    }


def generate_grid(case, layout, repair, merge):
    """Exercise only public source/repair APIs; never patch tessellation methods."""
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, LENGTH, WIDTH), zone_id=1, resolution=16.0)
    line = LineString(case["coords"])
    # The report compares physical feature shapes. Never simplify them silently.
    cm.add_line(
        line,
        line_id="test_barrier",
        resolution=case["h"],
        is_barrier=True,
        quad_buffer=layout == 'quad',
        quad_buffer_thickness=1,
        barrier_policy='cut' if 'clip' in repair else 'preserve',
        barrier_mirrors=repair in {'mirror', 'mirror_clip'},
        barrier_merge=merge,
        simplify_tolerance=0,
    )
    mg = MeshGenerator(background_lc=16.0, conceptual_mesh=cm)
    mg.generate()
    vt = VoronoiTessellator(mg, cm)
    grid = vt.generate()
    return grid, vt.barrier_report(), mg, vt


def executable_for(engine):
    """Find an executable without downloading or mutating an environment."""
    name = "mfusg" if engine == "mfusg" else "mf6"
    explicit = os.environ.get(f"{name.upper()}_EXE")
    candidates = [explicit, shutil.which(name), ROOT / ".bin" / f"{name}.exe"]
    for value in candidates:
        if value and Path(value).is_file():
            return Path(value).resolve()
    raise FileNotFoundError(
        f"Install {name}, set {name.upper()}_EXE, or add it to PATH."
    )


def _boundary_arrays(grid, barrier):
    x = grid.x.to_numpy(float)
    left = np.isclose(x, 0.0, rtol=0, atol=1e-7)
    right = np.isclose(x, LENGTH, rtol=0, atol=1e-7)
    if not left.any() or not right.any():
        raise ValueError("Benchmark requires boundary generators at x=0 and x=L.")
    if barrier and np.any(np.isclose(x, LENGTH / 2, rtol=0, atol=1e-7)):
        raise ValueError(
            "An HFB centre is on the discontinuity; reference is undefined."
        )
    exact, total_q = reference_solution(
        x, HYDRAULIC_CHARACTERISTIC if barrier else None
    )
    initial = exact.copy()
    initial[~(left | right)] = 12.345  # require actual solver recovery
    return left, right, initial, exact, total_q


def _run_process(command, workspace, timeout=60):
    completed = subprocess.run(
        [str(value) for value in command],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
    )
    transcript = completed.stdout + completed.stderr
    (workspace / "solver-output.txt").write_text(transcript, encoding="utf-8")
    normal = completed.returncode == 0 and "normal termination" in transcript.lower()
    if not normal:
        raise RuntimeError(f"Solver did not terminate normally: {transcript[-3000:]}")
    return transcript


def _mfusg_export():
    """Resolve optional USG support without making the MF6 benchmark depend on it."""
    try:
        from vorflow.utils import grid_to_mfusg_disu
    except ImportError as error:
        raise ImportError(
            "MFUSG grid export is unavailable in this checkout. Use a checkout "
            "containing the MFUSG support from PR #58, or select --engines mf6."
        ) from error
    return grid_to_mfusg_disu


def _run_mfusg(grid, workspace, executable, initial, left, right):
    flopy = _flopy()

    props = _mfusg_export()(grid, top=THICKNESS, botm=0.0)
    model = flopy.mfusg.MfUsg(
        modelname="flow",
        structured=False,
        model_ws=str(workspace),
        exe_name=str(executable),
    )
    metadata = {"node_id", "cell_map", "connectivity"}
    flopy.mfusg.MfUsgDisU(
        model, **{key: value for key, value in props.items() if key not in metadata}
    )
    flopy.mfusg.MfUsgBas(model, ibound=np.where(left | right, -1, 1), strt=initial)
    flopy.mfusg.MfUsgLpf(model, laytyp=0, hk=CONDUCTIVITY, vka=CONDUCTIVITY, ipakcb=53)
    flopy.mfusg.MfUsgSms(
        model,
        clin="cg",
        hclose=1e-8,
        hiclose=1e-8,
        rclosepcgu=1e-8,
        mxiter=200,
        iter1=1000,
    )
    flopy.modflow.ModflowOc(
        model,
        stress_period_data={(0, 0): ["save head", "save budget", "print budget"]},
    )
    model.write_input()
    _run_process([executable, "flow.nam"], workspace)
    reader = flopy.utils.HeadUFile(workspace / "flow.hds")
    times = reader.get_times()
    heads = np.concatenate(reader.get_data(totim=times[-1]))
    reader.close()
    incremental, _ = flopy.utils.MfusgListBudget(workspace / "flow.list").get_budget()
    final = incremental[-1]
    return dict(
        heads=heads,
        time=float(times[-1]),
        inflow=float(final["CONSTANT_HEAD_IN"]),
        outflow=float(final["CONSTANT_HEAD_OUT"]),
        budget=float(final["PERCENT_DISCREPANCY"]),
        outputs=["flow.hds", "flow.cbc", "flow.list"],
    )


def _run_mf6(grid, workspace, executable, initial, left, right, pairs, pumping=None):
    flopy = _flopy()

    arrays = grid_to_disv(grid)
    sim = flopy.mf6.MFSimulation(
        sim_name="flow", sim_ws=str(workspace), exe_name=str(executable)
    )
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(1, 1, 1)])
    flopy.mf6.ModflowIms(
        sim,
        complexity="COMPLEX",
        linear_acceleration="BICGSTAB",
        outer_dvclose=1e-9,
        inner_dvclose=1e-9,
        rcloserecord=1e-8,
        outer_maximum=200,
        inner_maximum=1000,
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname="flow", save_flows=True)
    flopy.mf6.ModflowGwfdisv(
        gwf,
        nlay=1,
        top=THICKNESS,
        botm=0.0,
        **{key: value for key, value in arrays.items() if key != "node_id"},
    )
    flopy.mf6.ModflowGwfic(gwf, strt=initial)
    # No XT3D here: this benchmark deliberately tests the ordinary two-point
    # conductance and detects mesh repairs that lose its orthogonality assumptions.
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=CONDUCTIVITY, save_flows=True)
    flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=[
            ((0, int(i)), LEFT_HEAD if left[i] else RIGHT_HEAD)
            for i in np.flatnonzero(left | right)
        ],
        save_flows=True,
    )
    if pairs:
        flopy.mf6.ModflowGwfhfb(
            gwf,
            stress_period_data=[
                ((0, row["cell_a"]), (0, row["cell_b"]), HYDRAULIC_CHARACTERISTIC)
                for row in pairs
            ],
        )
    if pumping is not None:
        flopy.mf6.ModflowGwfwel(
            gwf,
            stress_period_data=[
                ((0, int(i)), float(q)) for i, q in enumerate(pumping) if q != 0
            ],
            save_flows=True,
        )
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="flow.hds",
        budget_filerecord="flow.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("BUDGET", "ALL")],
    )
    sim.write_simulation(silent=True)
    _run_process([executable], workspace)
    reader = flopy.utils.HeadFile(workspace / "flow.hds")
    times = reader.get_times()
    heads = reader.get_data(totim=times[-1]).ravel()
    reader.close()
    budget = flopy.utils.CellBudgetFile(workspace / "flow.cbc", precision="double")
    chd = budget.get_data(text="CHD", totim=times[-1])[0]["q"]
    inflow = float(chd[chd > 0].sum())
    outflow = float(-chd[chd < 0].sum())
    well_outflow = 0.0
    if pumping is not None:
        well_q = budget.get_data(text="WEL", totim=times[-1])[0]["q"]
        well_outflow = float(-well_q[well_q < 0].sum())
    pair_fluxes = []
    if pairs:
        binary_grid = flopy.mf6.utils.MfGrdFile(workspace / "flow.disv.grb")
        flow_ja = budget.get_data(text="FLOW-JA-FACE", totim=times[-1])[0].ravel()
        for pair in pairs:
            a, b = pair["cell_a"], pair["cell_b"]
            positions = np.flatnonzero(
                binary_grid.ja[binary_grid.ia[a] : binary_grid.ia[a + 1]] == b
            )
            if len(positions) != 1:
                raise RuntimeError(
                    "Tracked HFB pair missing from actual solver connectivity."
                )
            pair_fluxes.append(
                dict(
                    cell_a=a,
                    cell_b=b,
                    q_into_a_m3_day=float(flow_ja[binary_grid.ia[a] + positions[0]]),
                )
            )
    budget.close()
    incremental, _ = flopy.utils.Mf6ListBudget(workspace / "flow.lst").get_budget()
    return dict(
        heads=heads,
        time=float(times[-1]),
        inflow=inflow,
        outflow=outflow,
        budget=float(incremental[-1]["PERCENT_DISCREPANCY"]),
        well_outflow=well_outflow,
        pair_fluxes=pair_fluxes,
        outputs=["flow.hds", "flow.cbc", "flow.lst", "mfsim.lst"],
    )


def closed_reference(spacing, parent, executable=None):
    """Independent structured DIS/HFB reference; no Vorflow geometry or tracking.

    Uniform extraction density covers the 64 m square. Grid spacings 2 m and
    1 m align exactly with all four HFB sides. Constant heads apply to the outer
    perimeter cell centres; their distance from the physical boundary shrinks
    under refinement and is included in the measured refinement difference.
    """
    flopy = _flopy()

    executable = Path(executable) if executable else executable_for("mf6")
    nrow, ncol = round(WIDTH / spacing), round(LENGTH / spacing)
    x = (np.arange(ncol) + 0.5) * spacing
    y = WIDTH - (np.arange(nrow) + 0.5) * spacing
    inside = (
        (y[:, None] > 18) & (y[:, None] < 82) & (x[None, :] > 68) & (x[None, :] < 132)
    )
    assert int(inside.sum()) * spacing**2 == 64**2
    workspace = (
        Path(parent) / f"mf6-closed-reference-h{spacing:g}-{uuid.uuid4().hex[:10]}"
    )
    workspace.mkdir(parents=True, exist_ok=False)
    sim = flopy.mf6.MFSimulation(
        sim_name="flow", sim_ws=str(workspace), exe_name=str(executable)
    )
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(1, 1, 1)])
    flopy.mf6.ModflowIms(
        sim,
        complexity="COMPLEX",
        linear_acceleration="BICGSTAB",
        outer_dvclose=1e-9,
        inner_dvclose=1e-9,
        rcloserecord=1e-8,
        outer_maximum=200,
        inner_maximum=2000,
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname="flow", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=nrow,
        ncol=ncol,
        delr=spacing,
        delc=spacing,
        top=THICKNESS,
        botm=0,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=18.123)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=CONDUCTIVITY, save_flows=True)
    boundary = {(row, col) for row in range(nrow) for col in (0, ncol - 1)}
    boundary |= {(row, col) for row in (0, nrow - 1) for col in range(ncol)}
    flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=[
            ((0, row, col), LEFT_HEAD) for row, col in sorted(boundary)
        ],
        save_flows=True,
    )
    flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=[
            ((0, int(row), int(col)), -CLOSED_PUMPING / inside.sum())
            for row, col in np.argwhere(inside)
        ],
        save_flows=True,
    )
    pairs = []
    for position in (68, 132):
        col = round(position / spacing)
        for row in np.flatnonzero((y > 18) & (y < 82)):
            pairs.append(
                ((0, int(row), col - 1), (0, int(row), col), HYDRAULIC_CHARACTERISTIC)
            )
    for position in (18, 82):
        row = round((WIDTH - position) / spacing)
        for col in np.flatnonzero((x > 68) & (x < 132)):
            pairs.append(
                ((0, row - 1, int(col)), (0, row, int(col)), HYDRAULIC_CHARACTERISTIC)
            )
    assert len(pairs) * spacing == 4 * 64
    flopy.mf6.ModflowGwfhfb(gwf, stress_period_data=pairs)
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="flow.hds",
        budget_filerecord="flow.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("BUDGET", "ALL")],
    )
    sim.write_simulation(silent=True)
    _run_process([executable], workspace)
    reader = flopy.utils.HeadFile(workspace / "flow.hds")
    total_time = reader.get_times()[-1]
    heads = reader.get_data(totim=total_time)[0]
    reader.close()
    budget = flopy.utils.CellBudgetFile(workspace / "flow.cbc", precision="double")
    chd = budget.get_data(text="CHD", totim=total_time)[0]["q"]
    wel = budget.get_data(text="WEL", totim=total_time)[0]["q"]
    budget.close()
    incremental, _ = flopy.utils.Mf6ListBudget(workspace / "flow.lst").get_budget()
    discrepancy = float(incremental[-1]["PERCENT_DISCREPANCY"])
    mean_head = float(heads[inside].mean())
    inflow, extraction = float(chd[chd > 0].sum()), float(-wel[wel < 0].sum())
    if not (
        np.isfinite(heads).all()
        and abs(total_time - 1) < 1e-10
        and abs(discrepancy) < BUDGET_ATOL
        and abs(inflow - CLOSED_PUMPING) / CLOSED_PUMPING < STRICT_FLOW_RTOL
        and abs(extraction - CLOSED_PUMPING) / CLOSED_PUMPING < STRICT_FLOW_RTOL
    ):
        raise RuntimeError("Independent closed-loop reference failed numerical checks.")
    result = dict(
        engine="mf6",
        problem="closed_hfb_reference",
        workspace=str(workspace),
        spacing_m=spacing,
        cells=nrow * ncol,
        hfb_pairs=len(pairs),
        mean_inside_head_m=mean_head,
        mean_inside_drawdown_m=LEFT_HEAD - mean_head,
        inflow_m3_day=inflow,
        well_outflow_m3_day=extraction,
        budget_discrepancy_percent=discrepancy,
        profile_x_m=x.tolist(),
        profile_heads_m=heads[np.abs(y - WIDTH / 2).argmin()].tolist(),
        x_centres_m=x.tolist(),
        y_centres_m=y.tolist(),
        heads_m=heads.ravel().tolist(),
    )
    (workspace / "evidence.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_closed_hfb(grid, report, parent, references, executable=None):
    """Force water across a closed HFB and compare drawdown to independent DIS.

    Q is prescribed and checks only mass balance. The area-weighted interior
    drawdown, compared to a separately refined structured model, tests hydraulic
    resistance. Incomplete feature coverage is rejected before assigning an HFB.
    """
    workspace = Path(parent) / f"mf6-closed-hfb-{uuid.uuid4().hex[:10]}"
    workspace.mkdir(parents=True, exist_ok=False)
    evidence = dict(
        engine="mf6", problem="closed_hfb", workspace=str(workspace), accepted=False
    )
    try:
        executable = Path(executable) if executable else executable_for("mf6")
        pairs = exact_hfb_pairs(grid, report, expected_length=4 * 64)
        inside_areas = shapely.area(
            shapely.intersection(grid.geometry.to_numpy(), box(68, 18, 132, 82))
        )
        if abs(float(inside_areas.sum()) - 64**2) > 1e-5:
            raise ValueError(
                "Closed extraction target has missing or overlapping cells."
            )
        q = -CLOSED_PUMPING * inside_areas / inside_areas.sum()
        x, y = grid.x.to_numpy(float), grid.y.to_numpy(float)
        boundary = (
            np.isclose(x, 0)
            | np.isclose(x, LENGTH)
            | np.isclose(y, 0)
            | np.isclose(y, WIDTH)
        )
        initial = np.where(boundary, LEFT_HEAD, 18.123)
        result = _run_mf6(
            grid,
            workspace,
            executable,
            initial,
            boundary,
            np.zeros(len(grid), bool),
            pairs,
            q,
        )
        mean_head = float(np.average(result["heads"], weights=inside_areas))
        drawdown = LEFT_HEAD - mean_head
        reference = references[-1]
        expected_drawdown = reference["mean_inside_drawdown_m"]
        # Sample the independent fine reference on the same side of the HFB.
        # A cross-barrier interpolation would smear the real head discontinuity.
        from scipy.spatial import cKDTree

        ref_x, ref_y = np.meshgrid(reference["x_centres_m"], reference["y_centres_m"])
        ref_xy = np.column_stack([ref_x.ravel(), ref_y.ravel()])
        ref_heads = np.asarray(reference["heads_m"])
        ref_inside = (
            (ref_xy[:, 0] > 68)
            & (ref_xy[:, 0] < 132)
            & (ref_xy[:, 1] > 18)
            & (ref_xy[:, 1] < 82)
        )
        centres_inside = (x > 68) & (x < 132) & (y > 18) & (y < 82)
        sampled_reference = np.empty(len(grid))
        for side in (False, True):
            ref_mask, cell_mask = ref_inside == side, centres_inside == side
            nearest = cKDTree(ref_xy[ref_mask]).query(
                np.column_stack([x[cell_mask], y[cell_mask]])
            )[1]
            sampled_reference[cell_mask] = ref_heads[ref_mask][nearest]
        max_head_error = float(np.abs(result["heads"] - sampled_reference).max())
        barrier_inflow = sum(
            row["q_into_a_m3_day"] * (1 if centres_inside[row["cell_a"]] else -1)
            for row in result["pair_fluxes"]
        )
        relative_error = abs(drawdown - expected_drawdown) / expected_drawdown
        refinement = (
            abs(references[-2]["mean_inside_drawdown_m"] - expected_drawdown)
            / expected_drawdown
        )
        checks = dict(
            normal_termination=True,
            expected_end_time=abs(result["time"] - 1) < 1e-10,
            finite_heads=bool(np.isfinite(result["heads"]).all()),
            water_budget=abs(result["budget"]) < BUDGET_ATOL,
            inflow_matches_extraction=abs(result["inflow"] - CLOSED_PUMPING)
            / CLOSED_PUMPING
            < STRICT_FLOW_RTOL,
            recorded_extraction=abs(result["well_outflow"] - CLOSED_PUMPING)
            / CLOSED_PUMPING
            < STRICT_FLOW_RTOL,
            reference_refinement=refinement <= CLOSED_REFINEMENT_RTOL,
            drawdown_accuracy=relative_error <= CLOSED_DRAWDOWN_RTOL,
            local_head_accuracy=max_head_error <= CLOSED_SAMPLE_HEAD_ATOL,
            actual_hfb_flow=abs(barrier_inflow - CLOSED_PUMPING) / CLOSED_PUMPING
            < STRICT_FLOW_RTOL,
        )
        evidence.update(
            accepted=all(checks.values()),
            checks=checks,
            hfb_pairs=pairs,
            mean_inside_head_m=mean_head,
            mean_inside_drawdown_m=drawdown,
            reference_mean_inside_drawdown_m=expected_drawdown,
            drawdown_relative_error=relative_error,
            drawdown_tolerance_relative=CLOSED_DRAWDOWN_RTOL,
            max_head_error_m=max_head_error,
            head_tolerance_m=CLOSED_SAMPLE_HEAD_ATOL,
            reference_refinement_relative_change=refinement,
            reference_workspace=reference["workspace"],
            inflow_m3_day=result["inflow"],
            well_outflow_m3_day=result["well_outflow"],
            expected_flow_m3_day=CLOSED_PUMPING,
            budget_discrepancy_percent=result["budget"],
            heads_m=result["heads"].tolist(),
            initial_heads_m=initial.tolist(),
            extraction_m3_day=q.tolist(),
            inside_areas_m2=inside_areas.tolist(),
            exact_heads_m=sampled_reference.tolist(),
            hfb_net_inflow_m3_day=barrier_inflow,
            pair_fluxes=result["pair_fluxes"],
        )
    except Exception as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
    (workspace / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    return evidence


def run_grid(
    grid,
    report,
    engine,
    parent,
    *,
    barrier=False,
    executable=None,
    strict_baseline=False,
):
    """Run one actual solver in a new workspace and return JSON-ready evidence.

    Geometry/export/coverage errors are recorded as failures. The unique workspace
    makes stale output reuse impossible; output sizes and modification times are
    retained, along with numerical results, exact heads and source-node mapping.
    """
    workspace = (
        Path(parent)
        / f"{engine}-{'hfb' if barrier else 'linear'}-{uuid.uuid4().hex[:10]}"
    )
    workspace.mkdir(parents=True, exist_ok=False)
    started = time.time()
    evidence = dict(
        engine=engine,
        problem="hfb" if barrier else "linear",
        workspace=str(workspace),
        started_utc=datetime.fromtimestamp(started, timezone.utc).isoformat(),
        cells=len(grid),
        accepted=False,
    )
    try:
        if engine == "mfusg":
            _mfusg_export()
        executable = Path(executable) if executable else executable_for(engine)
        evidence["executable"] = str(executable.resolve())
        evidence["executable_sha256"] = hashlib.sha256(
            executable.read_bytes()
        ).hexdigest()
        if barrier and engine != "mf6":
            raise ValueError(
                "The HFB benchmark uses MF6; MFUSG HFB is not implemented here."
            )
        pairs = exact_hfb_pairs(grid, report) if barrier else []
        evidence["hfb_pairs"] = pairs
        left, right, initial, exact, expected_flow = _boundary_arrays(grid, barrier)
        result = (
            _run_mfusg(grid, workspace, executable, initial, left, right)
            if engine == "mfusg"
            else _run_mf6(grid, workspace, executable, initial, left, right, pairs)
        )
        checks = assess_accuracy(
            result["heads"],
            exact,
            result["inflow"],
            result["outflow"],
            expected_flow,
            result["budget"],
            expected_end_time=abs(result["time"] - 1) < 1e-10,
            strict_baseline=strict_baseline,
        )
        evidence.update(checks)
        output_files = []
        for name in result["outputs"]:
            path = workspace / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty current-run output: {name}")
            output_files.append(
                dict(
                    name=name,
                    bytes=path.stat().st_size,
                    modified_utc=datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                )
            )
        evidence.update(
            total_time_days=result["time"],
            outputs=output_files,
            heads_m=result["heads"].tolist(),
            exact_heads_m=exact.tolist(),
            initial_heads_m=initial.tolist(),
        )
    except Exception as error:
        evidence.update(accepted=False, error=f"{type(error).__name__}: {error}")
    evidence["elapsed_seconds"] = time.time() - started
    (workspace / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    return evidence


def grid_evidence(grid, report, mg, vt):
    """Actual source elements and final rows for the interactive report."""
    raw = mg._element_data
    lookup = dict(zip(raw["node_tags"].tolist(), raw["node_xy"]))
    elements = []
    for block in raw["blocks"]:
        for tag, row in zip(block["tags"], block["connectivity"]):
            elements.append(
                dict(
                    tag=int(tag),
                    kind="quad"
                    if "Quadrilateral" in block["element_name"]
                    else "triangle",
                    ring=[lookup[int(n)].tolist() for n in row],
                )
            )
    summary = report["summary"].iloc[0]
    quality = vt.quality_report(
        cell_size=float(vt.cm.clean_lines.iloc[0].lc), domain=box(0, 0, LENGTH, WIDTH)
    )
    return _json_safe(
        dict(
            cells=[
                dict(
                    row=int(i),
                    node_id=int(row.node_id),
                    centre=[float(row.x), float(row.y)],
                    ring=np.asarray(row.geometry.exterior.coords).tolist(),
                    holes=[
                        np.asarray(ring.coords).tolist()
                        for ring in row.geometry.interiors
                    ],
                )
                for i, (_, row) in enumerate(grid.iterrows())
            ],
            original_elements=elements,
            mirrors=int(vt.n_barrier_mirrors),
            merge_diagnostics=vt.barrier_merge_diagnostics.to_dict("records"),
            matched_length_m=float(summary.matched_length),
            unmatched_length_m=float(summary.unmatched_length),
            crossing_cells=int(summary.crossing_cells),
            model_row_to_node_id=grid.node_id.astype(int).tolist(),
            target_closed=bool(summary.geometry.is_closed),
            source_report=mg.source_report.drop(columns="geometry").to_dict("records"),
            quality_summary=quality["summary"],
            cell_quality=quality["cells"].drop(columns="geometry").to_dict("records"),
            face_quality=quality["faces"].drop(columns="geometry").to_dict("records"),
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engines", nargs="+", choices=["mfusg", "mf6"], default=["mf6"]
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=list(benchmark_cases()),
        default=list(benchmark_cases()),
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["straddle", "quad"],
        default=["straddle", "quad"],
    )
    parser.add_argument(
        "--repairs",
        nargs="+",
        choices=["none", "mirror", "clip", "mirror_clip"],
        default=["none", "mirror", "clip", "mirror_clip"],
    )
    parser.add_argument(
        "--merge", choices=["none", "checked", "restore"], default="restore"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".bin" / "barrier-system-flow"
    )
    args = parser.parse_args(argv)
    set_verbosity(0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output.resolve() / f"run-{stamp}-{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    records = []
    provenance = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        checkout=str(ROOT),
        python=sys.version,
        packages={
            package: importlib.metadata.version(package)
            for package in ("vorflow", "flopy", "numpy", "shapely", "gmsh")
        },
        tolerances=dict(
            repaired_head_atol_m=HEAD_ATOL,
            repaired_flow_rtol=FLOW_RTOL,
            strict_head_atol_m=STRICT_HEAD_ATOL,
            strict_flow_rtol=STRICT_FLOW_RTOL,
            budget_percent=BUDGET_ATOL,
            closed_drawdown_rtol=CLOSED_DRAWDOWN_RTOL,
            closed_reference_refinement_rtol=CLOSED_REFINEMENT_RTOL,
        ),
        physics=dict(
            length_m=LENGTH,
            width_m=WIDTH,
            thickness_m=THICKNESS,
            conductivity_m_day=CONDUCTIVITY,
            heads_m=[LEFT_HEAD, RIGHT_HEAD],
            hfb_hydraulic_characteristic_day=HYDRAULIC_CHARACTERISTIC,
        ),
        limitations=[
            "Confined, homogeneous, isotropic, one-layer benchmark; no universal accuracy guarantee.",
            "Ordinary two-point flow; XT3D is deliberately disabled to expose nonorthogonal repairs.",
            "Analytical HFB accuracy is checked for a completely matched straight barrier spanning the domain.",
            "Closed square HFB is independently checked against refined structured DIS drawdown; other bends/loops are grid checks.",
        ],
        loaded_source_sha256=LOADED_SOURCE_HASHES,
    )
    references = []
    if "square" in args.cases and "mf6" in args.engines:
        try:
            references = [closed_reference(spacing, output) for spacing in (2.0, 1.0)]
        except Exception as error:
            provenance["closed_reference_error"] = f"{type(error).__name__}: {error}"
    for case_name in args.cases:
        case = benchmark_cases()[case_name]
        for layout in args.layouts:
            for repair in args.repairs:
                record = dict(
                    case=case_name,
                    layout=layout,
                    repair=repair,
                    merge=args.merge,
                    geometry=case,
                )
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        grid, report, mg, vt = generate_grid(
                            case, layout, repair, args.merge
                        )
                    record["warnings"] = list(
                        dict.fromkeys(str(row.message) for row in caught)
                    )
                    record["grid"] = grid_evidence(grid, report, mg, vt)
                    runs = [
                        run_grid(
                            grid,
                            report,
                            engine,
                            output,
                            strict_baseline=repair in {"none", "mirror"},
                        )
                        for engine in args.engines
                    ]
                    if case_name == "straight" and "mf6" in args.engines:
                        runs.append(
                            run_grid(
                                grid,
                                report,
                                "mf6",
                                output,
                                barrier=True,
                                strict_baseline=repair in {"none", "mirror"},
                            )
                        )
                    if case_name == "square" and "mf6" in args.engines:
                        runs.append(
                            run_closed_hfb(grid, report, output, references)
                            if references
                            else dict(
                                engine="mf6",
                                problem="closed_hfb",
                                accepted=False,
                                error=provenance["closed_reference_error"],
                            )
                        )
                    record["runs"] = runs
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                records.append(record)
                counts = f"{sum(run['accepted'] for run in record.get('runs', []))}/{len(record.get('runs', []))}"
                print(
                    f"{case_name} / {layout} / {repair}: {counts} accepted {record.get('error', '')}",
                    flush=True,
                )
                (output / "results.json").write_text(
                    json.dumps(
                        dict(
                            provenance=provenance,
                            records=records,
                            closed_references=references,
                        ),
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    accepted = sum(
        run["accepted"] for record in records for run in record.get("runs", [])
    )
    total = sum(len(record.get("runs", [])) for record in records)
    provenance["source_changed_during_run"] = source_hashes() != LOADED_SOURCE_HASHES
    (output / "results.json").write_text(
        json.dumps(
            dict(provenance=provenance, records=records, closed_references=references),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(output=str(output / "results.json"), accepted=accepted, total=total)
        )
    )
    return output


if __name__ == "__main__":
    main()
