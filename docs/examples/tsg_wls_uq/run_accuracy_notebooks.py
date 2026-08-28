# ruff: noqa: S603
"""Run the paper accuracy experiments from the existing PGM example notebooks.

The source notebooks are left unchanged. Their code cells are executed with the
paper sample counts and with outputs redirected to a dedicated directory. The
exported comparison tables also contain the nominal analytical estimates and
the clean power-flow values used for interval-score evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PGM_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = PGM_ROOT / "docs" / "examples"
OUTPUT_ROOT = EXAMPLE_ROOT / "output" / "tsg_wls_uq_accuracy"
PYTHON = Path(sys.executable).resolve()

NOTEBOOKS = {
    "cigre_mv": ("CIGRE MV State Estimation UQ Example.ipynb", "cigre_mv_radial"),
    "ieee33": ("IEEE33 State Estimation UQ Example.ipynb", "ieee33"),
    "mv_oberrhein": ("MV Oberrhein State Estimation UQ Example.ipynb", "mv_oberrhein"),
    "lv_schutterwald": (
        "LV Schutterwald State Estimation UQ Example.ipynb",
        "lv_schutterwald",
    ),
    "cigre_lv": ("CIGRE LV State Estimation UQ Example.ipynb", "cigre_lv"),
}


def patched_code(notebook_path: Path, output_slug: str) -> str:
    """Extract code cells and apply the paper-only execution parameters."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells: list[str] = []

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))

        # Timing-scaling panels are outside the accuracy experiment and can be
        # much more expensive than the requested 1000-sample baseline.
        if "monte_carlo_baseline_sample_count" in source:
            break

        source = re.sub(
            r"(?m)^monte_carlo_sample_count\s*=\s*[\d_]+\s*$",
            "monte_carlo_sample_count = 1000",
            source,
        )
        source = re.sub(
            r"(?m)^pandapower_sample_count\s*=\s*[\d_]+\s*$",
            "pandapower_sample_count = 1000",
            source,
        )
        source = re.sub(
            r"(?m)^monte_carlo_timing_sample_counts\s*=\s*\[[^\n]*\]\s*$",
            "monte_carlo_timing_sample_counts = []",
            source,
        )
        source = source.replace(
            f"docs/examples/output/{output_slug}",
            f"docs/examples/output/tsg_wls_uq_accuracy/{output_slug}",
        )
        source = source.replace(
            f'Path("output/{output_slug}")',
            f'Path("output/tsg_wls_uq_accuracy/{output_slug}")',
        )
        code_cells.append(source)

    preamble = "from IPython.display import display\n"
    code = preamble + "\n\n".join(code_cells) + "\n"

    # Add the nominal analytical estimate and clean, noise-free reference to
    # the existing comparison export. The notebooks use two layouts: the four
    # sensor-setting examples index analytical_results by (setting, method),
    # while LV Schutterwald uses (method). Derive the reference from the same
    # model input and output specifications used by the notebook.
    coverage_export = r"""
_coverage_input_data = base_input_data if "base_input_data" in globals() else input_data
_coverage_power_flow = PowerGridModel(
    _coverage_input_data, system_frequency=50.0
).calculate_power_flow(
    symmetric=True,
    calculation_method=CalculationMethod.newton_raphson,
    error_tolerance=1e-8,
    max_iterations=100,
)
_coverage_specs = {
    (spec["component"], int(spec["id"]), spec["terminal"], spec["quantity"]): spec
    for spec in output_specs
}
for _row_index, _row in comparison.iterrows():
    _spec = _coverage_specs[
        (_row["component"], int(_row["id"]), _row["terminal"], _row["quantity"])
    ]
    _coverage_result_key = (
        (_row["sensor setting"], "Analytical UQ ILSE")
        if "sensor setting" in comparison.columns
        else "Analytical UQ ILSE"
    )
    _coverage_nrse_key = (
        (_row["sensor setting"], "Analytical UQ NRSE")
        if "sensor setting" in comparison.columns
        else "Analytical UQ NRSE"
    )
    _coverage_truth = _coverage_power_flow[_spec["component_type"]][
        _spec["value_field"]
    ][_spec["index"]]
    comparison.loc[_row_index, "true value"] = float(_coverage_truth)
    for _method_name, _result_key in (
        ("Analytical UQ ILSE", _coverage_result_key),
        ("Analytical UQ NRSE", _coverage_nrse_key),
    ):
        _coverage_result = analytical_results[_result_key]
        comparison.loc[_row_index, f"{_method_name} estimate"] = float(
            _coverage_result[_spec["component_type"]][_spec["value_field"]][
                _spec["index"]
            ]
        )
"""
    export_marker = "comparison = pd.DataFrame(comparison_rows)\n"
    if export_marker not in code:
        raise ValueError(f"Could not locate comparison export in {notebook_path}")
    code = code.replace(export_marker, export_marker + coverage_export, 1)
    numeric_marker = 'numeric_columns = [\n    "Analytical UQ ILSE sigma",'
    numeric_replacement = """numeric_columns = [
    "true value",
    "Analytical UQ ILSE estimate",
    "Analytical UQ NRSE estimate",
    "Analytical UQ ILSE sigma","""
    if numeric_marker not in code:
        raise ValueError(f"Could not locate numeric comparison columns in {notebook_path}")
    return code.replace(numeric_marker, numeric_replacement, 1)


def run_notebook(key: str) -> None:
    notebook_name, output_slug = NOTEBOOKS[key]
    notebook_path = EXAMPLE_ROOT / notebook_name
    output_directory = OUTPUT_ROOT / output_slug
    output_directory.mkdir(parents=True, exist_ok=True)
    log_path = output_directory / "execution.log"

    with tempfile.TemporaryDirectory(prefix=f"tsg-wls-uq-{key}-") as temporary_directory:
        script_path = Path(temporary_directory) / f"{key}_accuracy.py"
        script_path.write_text(patched_code(notebook_path, output_slug), encoding="utf-8")
        matplotlib_config_directory = Path(temporary_directory) / "mplconfig"
        matplotlib_config_directory.mkdir()
        environment = os.environ.copy()
        environment.update(
            {
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(matplotlib_config_directory),
                "PYTHONUNBUFFERED": "1",
            }
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(
                [str(PYTHON), str(script_path)],
                cwd=PGM_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("networks", nargs="*", choices=NOTEBOOKS, default=list(NOTEBOOKS))
    arguments = parser.parse_args()
    for network in arguments.networks:
        print(f"Running {network} ...", flush=True)
        run_notebook(network)
        print(f"Completed {network}.", flush=True)


if __name__ == "__main__":
    main()
