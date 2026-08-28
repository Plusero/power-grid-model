# ruff: noqa: I001, PLC0415, PLR0915
"""Run the paper runtime benchmark using the PGM ternary-tree example code."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import scipy
import scipy.linalg

PGM_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = PGM_ROOT / "docs" / "examples"
NOTEBOOK_PATH = EXAMPLE_ROOT / "Synthetic MV Ternary Tree UQ Scaling Example.ipynb"
OUTPUT_ROOT = EXAMPLE_ROOT / "output" / "tsg_wls_uq_runtime"
TIMING_PATH = OUTPUT_ROOT / "ternary_tree_runtime.csv"
METADATA_PATH = OUTPUT_ROOT / "runtime_metadata.json"

NODE_COUNTS = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
GAIN_NODE_COUNTS = {500, 1_000, 2_000}
MC_SAMPLE_COUNT = 10


def load_notebook_helpers() -> dict[str, Any]:
    """Load definitions preceding the notebook's timing loop."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    sources: list[str] = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "timing_rows = []" in source:
            break
        sources.append(source)

    namespace: dict[str, Any] = {"__name__": "tsg_wls_uq_runtime_helpers"}
    exec(compile("\n\n".join(sources), str(NOTEBOOK_PATH), "exec"), namespace)  # noqa: S102
    return namespace


def elapsed_summary(call: Callable[[], tuple[Any, int | None]], repeats: int) -> tuple[dict[str, float], int | None]:
    """Warm up once, then return median/minimum/maximum wall time."""
    warmup_result, warmup_converged = call()
    del warmup_result
    samples: list[float] = []
    converged = warmup_converged
    for _ in range(repeats):
        start = perf_counter()
        result, converged = call()
        samples.append(perf_counter() - start)
        del result
    return (
        {
            "median time [s]": float(np.median(samples)),
            "minimum time [s]": float(np.min(samples)),
            "maximum time [s]": float(np.max(samples)),
        },
        converged,
    )


def checkpoint(rows: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_path = TIMING_PATH.with_suffix(".tmp")
    fieldnames = [
        "buses",
        "method",
        "median time [s]",
        "minimum time [s]",
        "maximum time [s]",
        "repeats",
        "Monte Carlo samples",
        "converged samples",
        "timing scope",
    ]
    with temporary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(TIMING_PATH)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-buses", type=int, default=1_000_000)
    parser.add_argument("--mc-max-buses", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()

    if str(EXAMPLE_ROOT.resolve()) not in sys.path:
        sys.path.insert(0, str(EXAMPLE_ROOT.resolve()))

    from gain_covariance import propagate_covariance
    from gain_matrix import build_gain_matrix
    from power_grid_model import (
        CalculationMethod,
        CalculationType,
        PowerGridModel,
    )
    from power_grid_model._core.power_grid_model_c.get_pgm_dll_path import (
        get_pgm_dll_path,
    )
    from power_grid_model.utils import create_state_estimation_monte_carlo_updates
    from power_grid_model.validation import assert_valid_input_data

    helpers = load_notebook_helpers()
    build_network = helpers["build_ternary_mv_network"]
    make_input = helpers["make_state_estimation_input"]
    calculate_method = helpers["calculate_method"]

    selected_counts = [count for count in NODE_COUNTS if count <= arguments.max_buses]
    rows: list[dict[str, Any]] = []
    checkpoint(rows)

    native_library = get_pgm_dll_path().resolve()
    try:
        native_library_label = native_library.relative_to(PGM_ROOT).as_posix()
    except ValueError:
        native_library_label = native_library.name

    metadata = {
        "timestamp timezone": "local system time",
        "platform": platform.platform(),
        "CPU": cpu_model(),
        "logical CPUs": os.cpu_count(),
        "Python": platform.python_version(),
        "NumPy": np.__version__,
        "SciPy": scipy.__version__,
        "PGM native library": native_library_label,
        "bus counts": selected_counts,
        "Monte Carlo samples": MC_SAMPLE_COUNT,
        "repeats": arguments.repeats,
        "BLAS thread environment": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    method_specs = {
        "Deterministic ILSE": (CalculationMethod.iterative_linear, False),
        "Deterministic NRSE": (CalculationMethod.newton_raphson, False),
        "Analytical UQ ILSE": (CalculationMethod.iterative_linear, True),
        "Analytical UQ NRSE": (CalculationMethod.newton_raphson, True),
    }

    for buses in selected_counts:
        print(f"Preparing {buses:,} buses ...", flush=True)
        base_input = build_network(buses)
        input_data = make_input(base_input)
        assert_valid_input_data(input_data, calculation_type=CalculationType.state_estimation, symmetric=True)

        for method_name, method_spec in method_specs.items():

            def deterministic_call(
                spec: tuple[Any, bool] = method_spec,
                data: dict[Any, np.ndarray] = input_data,
            ) -> tuple[Any, None]:
                return calculate_method(data, *spec), None

            timing, _ = elapsed_summary(deterministic_call, arguments.repeats)
            rows.append(
                {
                    "buses": buses,
                    "method": method_name,
                    **timing,
                    "repeats": arguments.repeats,
                    "Monte Carlo samples": "",
                    "converged samples": "",
                    "timing scope": "fresh model plus complete state-estimation call",
                }
            )
            checkpoint(rows)
            print(f"  {method_name}: {timing['median time [s]']:.6g} s", flush=True)

        if buses <= arguments.mc_max_buses:
            updates = create_state_estimation_monte_carlo_updates(input_data, MC_SAMPLE_COUNT, seed=2026)
            for method_name, calculation_method in (
                ("Monte Carlo ILSE", CalculationMethod.iterative_linear),
                ("Monte Carlo NRSE", CalculationMethod.newton_raphson),
            ):

                def monte_carlo_call(
                    method: Any = calculation_method,
                    data: dict[Any, np.ndarray] = input_data,
                    batch_updates: dict[Any, np.ndarray] = updates,
                ) -> tuple[Any, int]:
                    model = PowerGridModel(data, system_frequency=50.0)
                    result = model.calculate_state_estimation(
                        calculation_method=method,
                        update_data=batch_updates,
                        error_tolerance=helpers["SOLVER_ERROR_TOLERANCE"],
                        max_iterations=helpers["SOLVER_MAX_ITERATIONS"],
                        continue_on_batch_error=True,
                    )
                    failed_count = 0 if model.batch_error is None else len(model.batch_error.failed_scenarios)
                    return result, MC_SAMPLE_COUNT - failed_count

                timing, converged = elapsed_summary(monte_carlo_call, arguments.repeats)
                rows.append(
                    {
                        "buses": buses,
                        "method": method_name,
                        **timing,
                        "repeats": arguments.repeats,
                        "Monte Carlo samples": MC_SAMPLE_COUNT,
                        "converged samples": converged,
                        "timing scope": (
                            "fresh model plus 10-scenario batch state-estimation call; sample generation excluded"
                        ),
                    }
                )
                checkpoint(rows)
                print(f"  {method_name}: {timing['median time [s]']:.6g} s", flush=True)

        if buses in GAIN_NODE_COUNTS:
            for method_name, inverse in (
                ("Gain inverse NumPy", np.linalg.inv),
                ("Gain inverse SciPy", lambda matrix: scipy.linalg.inv(matrix, check_finite=False)),
            ):

                def gain_call(
                    inverse_function: Callable[[np.ndarray], np.ndarray] = inverse,
                    data: dict[Any, np.ndarray] = input_data,
                ) -> tuple[Any, None]:
                    model = PowerGridModel(data, system_frequency=50.0)
                    state_estimation_result = model.calculate_state_estimation(
                        calculation_method=CalculationMethod.newton_raphson,
                        error_tolerance=helpers["SOLVER_ERROR_TOLERANCE"],
                        max_iterations=helpers["SOLVER_MAX_ITERATIONS"],
                    )
                    gain_model = build_gain_matrix(
                        data,
                        state_estimation_result=state_estimation_result,
                        max_gain_matrix_bytes=None,
                    )
                    measurement_model = gain_model.measurement_model
                    raw_gain_inverse = inverse_function(gain_model.gain_matrix)
                    gain_inverse = 0.5 * (raw_gain_inverse + raw_gain_inverse.T)
                    result = propagate_covariance(
                        data,
                        measurement_model,
                        gain_inverse,
                        state_estimation_result,
                    )
                    return result, None

                timing, _ = elapsed_summary(gain_call, arguments.repeats)
                rows.append(
                    {
                        "buses": buses,
                        "method": method_name,
                        **timing,
                        "repeats": arguments.repeats,
                        "Monte Carlo samples": "",
                        "converged samples": "",
                        "timing scope": "NRSE, dense gain construction/inversion, and output covariance propagation",
                    }
                )
                checkpoint(rows)
                print(f"  {method_name}: {timing['median time [s]']:.6g} s", flush=True)


if __name__ == "__main__":
    main()
