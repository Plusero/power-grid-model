# ruff: noqa: E501, S603
"""Compute one-sigma interval scores from the five PGM accuracy examples.

The notebooks are used as the source of the network and sensor definitions but
are not modified.  This runner executes only their setup cells, replaces the
sensor readings by the clean power-flow values, and evaluates analytical,
Monte Carlo, and gain-inverse intervals on a common repeated-sampling batch.
Monte Carlo sigmas use an independent training batch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PGM_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = PGM_ROOT / "docs" / "examples"
OUTPUT_ROOT = EXAMPLE_ROOT / "output" / "tsg_wls_uq_accuracy"
PYTHON = Path(sys.executable).resolve()
SAMPLE_COUNT = 1_000
EVALUATION_SEED = 2_026
MONTE_CARLO_TRAINING_SEED = 2_027

NOTEBOOKS = {
    "cigre_mv": ("CIGRE MV State Estimation UQ Example.ipynb", "cigre_mv_radial"),
    "ieee33": ("IEEE33 State Estimation UQ Example.ipynb", "ieee33"),
    "mv_oberrhein": ("MV Oberrhein State Estimation UQ Example.ipynb", "mv_oberrhein"),
    "lv_schutterwald": ("LV Schutterwald State Estimation UQ Example.ipynb", "lv_schutterwald"),
    "cigre_lv": ("CIGRE LV State Estimation UQ Example.ipynb", "cigre_lv"),
}


def setup_code(notebook_path: Path, output_slug: str) -> str:
    """Extract cells through construction of the state-estimation input."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells: list[str] = []
    found_input = False
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "methods =" in source or "analytical_results" in source:
            break
        source = source.replace(
            f"docs/examples/output/{output_slug}",
            f"docs/examples/output/tsg_wls_uq_accuracy/{output_slug}",
        )
        source = source.replace(
            f'Path("output/{output_slug}")',
            f'Path("output/tsg_wls_uq_accuracy/{output_slug}")',
        )
        cells.append(source)
        if "state_estimation_inputs =" in source or "state_estimation_input =" in source:
            found_input = True
            break
    if not found_input:
        raise ValueError(f"Could not find state-estimation input in {notebook_path}")
    code = "from IPython.display import display\n" + "\n\n".join(cells) + "\n"
    compile(code, str(notebook_path), "exec")
    return code


def analysis_code() -> str:
    """Return the shared execution and interval-score implementation."""
    code = r"""
from math import erf, sqrt

_interval_alpha = 1.0 - erf(1.0 / sqrt(2.0))
_interval_sample_count = __SAMPLE_COUNT__
_interval_evaluation_seed = __EVALUATION_SEED__
_interval_mc_training_seed = __MC_TRAINING_SEED__
_base_methods = {
    "ILSE": CalculationMethod.iterative_linear,
    "NRSE": CalculationMethod.newton_raphson,
}
_interval_inputs = (
    {"voltage + current + power": state_estimation_inputs["voltage + current + power"]}
    if "state_estimation_inputs" in globals()
    else {"single": state_estimation_input}
)

def _copy_input(data):
    return {component: values.copy() for component, values in data.items()}

def _branch_lookup(data):
    result = {}
    for component in (ComponentType.line, ComponentType.transformer):
        if component in data:
            for index, ident in enumerate(data[component][AttributeType.id]):
                result[int(ident)] = (component, index)
    return result

def _truth_centered(data, truth):
    centered = _copy_input(data)
    nodes = truth[ComponentType.node]
    node_lookup = {int(ident): i for i, ident in enumerate(nodes[AttributeType.id])}
    branches = _branch_lookup(truth)
    for sensor_type, sensor in list(centered.items()):
        if sensor_type not in {
            ComponentType.sym_voltage_sensor,
            ComponentType.sym_power_sensor,
            ComponentType.sym_current_sensor,
        }:
            continue
        for row in range(len(sensor)):
            object_id = int(sensor[AttributeType.measured_object][row])
            if sensor_type == ComponentType.sym_voltage_sensor:
                node_index = node_lookup[object_id]
                sensor[AttributeType.u_measured][row] = nodes[AttributeType.u][node_index]
                angle = nodes[AttributeType.u_angle][node_index]
                if not __import__("numpy").isfinite(sensor[AttributeType.u_angle_measured][row]):
                    sensor[AttributeType.u_angle_measured][row] = __import__("numpy").nan
                else:
                    sensor[AttributeType.u_angle_measured][row] = angle
                continue
            component, branch_index = branches[object_id]
            terminal = int(sensor[AttributeType.measured_terminal_type][row])
            if terminal == int(MeasuredTerminalType.branch_from):
                terminal_name = "from"
            elif terminal == int(MeasuredTerminalType.branch_to):
                terminal_name = "to"
            else:
                raise ValueError(f"Unsupported measured terminal type: {terminal}")
            suffix = "_from" if terminal_name == "from" else "_to"
            p = truth[component][getattr(AttributeType, "p" + suffix)][branch_index]
            q = truth[component][getattr(AttributeType, "q" + suffix)][branch_index]
            if sensor_type == ComponentType.sym_power_sensor:
                sensor[AttributeType.p_measured][row] = p
                sensor[AttributeType.q_measured][row] = q
            else:
                if sensor_type == ComponentType.sym_current_sensor:
                    angle_type = int(sensor[AttributeType.angle_measurement_type][row])
                    if angle_type != int(AngleMeasurementType.global_angle):
                        raise ValueError("Interval runner requires global current-angle sensors")
                node_field = getattr(AttributeType, terminal_name + "_node")
                node_index = node_lookup[int(data[component][node_field][branch_index])]
                u = nodes[AttributeType.u][node_index]
                u_angle = nodes[AttributeType.u_angle][node_index]
                current = __import__("numpy").conj((p + 1j * q) / (__import__("numpy").sqrt(3.0) * u * __import__("numpy").exp(1j * u_angle)))
                sensor[AttributeType.i_measured][row] = abs(current)
                if __import__("numpy").isfinite(sensor[AttributeType.i_angle_measured][row]):
                    sensor[AttributeType.i_angle_measured][row] = __import__("numpy").angle(current)
    return centered

def _score_output(
    component_type,
    ident,
    terminal,
    quantity,
    unit,
    y_true,
    y_hat,
    interval_sigma,
    empirical_mc_sigma,
    method,
    center_source,
    sigma_source,
    setting,
    attempted_samples,
    method_converged_samples,
    sigma_calibration_samples,
):
    np = __import__("numpy")
    y_hat = np.asarray(y_hat, dtype=float)
    interval_sigma = np.broadcast_to(
        np.asarray(interval_sigma, dtype=float), y_hat.shape
    )
    finite = np.isfinite(y_hat) & np.isfinite(interval_sigma)
    if not np.all(finite) or not np.isfinite(empirical_mc_sigma):
        return []
    low, up = y_hat - interval_sigma, y_hat + interval_sigma
    covered = (low <= y_true) & (y_true <= up)
    width = up - low
    penalty = np.where(
        y_true < low, low - y_true, np.where(y_true > up, y_true - up, 0.0)
    )
    winkler = width + 2.0 * penalty / _interval_alpha
    mean_width = float(np.mean(width))
    mean_winkler = float(np.mean(winkler))
    standardized_width = (
        mean_width / empirical_mc_sigma if empirical_mc_sigma > 0.0 else float("nan")
    )
    standardized_winkler = (
        mean_winkler / empirical_mc_sigma if empirical_mc_sigma > 0.0 else float("nan")
    )
    return [{
        "sensor setting": setting,
        "method": method,
        "center source": center_source,
        "sigma source": sigma_source,
        "component": component_type.value,
        "id": int(ident),
        "terminal": terminal,
        "quantity": quantity,
        "unit": unit,
        "true value": float(y_true),
        "mean estimate": float(np.mean(y_hat)),
        "mean interval sigma": float(np.mean(interval_sigma)),
        "attempted samples": int(attempted_samples),
        "method converged samples": int(method_converged_samples),
        "common converged samples": int(len(y_hat)),
        "sigma calibration samples": int(sigma_calibration_samples),
        "MCS": float(np.mean(covered)),
        "MWS": mean_width,
        "MWIS": mean_winkler,
        "empirical MC sigma": float(empirical_mc_sigma),
        "standardized MWS": standardized_width,
        "standardized MWIS": standardized_winkler,
    }]

_truth = PowerGridModel(base_input_data, system_frequency=50.0).calculate_power_flow(
    symmetric=True, calculation_method=CalculationMethod.newton_raphson,
    error_tolerance=1e-8, max_iterations=50,
)
_all_rows = []
_np = __import__("numpy")
_fields = {
    ComponentType.node: [("voltage magnitude", "u", "u_sigma", "V"), ("active power injection", "p", "p_sigma", "W"), ("reactive power injection", "q", "q_sigma", "var")],
    ComponentType.line: [("current", "i_from", "i_from_sigma", "A"), ("active power", "p_from", "p_from_sigma", "W"), ("reactive power", "q_from", "q_from_sigma", "var"), ("current", "i_to", "i_to_sigma", "A"), ("active power", "p_to", "p_to_sigma", "W"), ("reactive power", "q_to", "q_to_sigma", "var")],
    ComponentType.transformer: [("current", "i_from", "i_from_sigma", "A"), ("active power", "p_from", "p_from_sigma", "W"), ("reactive power", "q_from", "q_from_sigma", "var"), ("current", "i_to", "i_to_sigma", "A"), ("active power", "p_to", "p_to_sigma", "W"), ("reactive power", "q_to", "q_to_sigma", "var")],
}
for _setting, _input in _interval_inputs.items():
    _centered = _truth_centered(_input, _truth)
    _evaluation_updates = create_state_estimation_monte_carlo_updates(
        _centered, _interval_sample_count, seed=_interval_evaluation_seed
    )
    _training_updates = create_state_estimation_monte_carlo_updates(
        _centered, _interval_sample_count, seed=_interval_mc_training_seed
    )
    _evaluation_masks = {}
    # Obtain the common converged-scenario set without retaining two large UQ
    # result batches. The UQ passes below are checked against these masks.
    for _base_method, _calculation_method in _base_methods.items():
        _model = PowerGridModel(_centered, system_frequency=50.0)
        _model.calculate_state_estimation(
            symmetric=True, calculation_method=_calculation_method,
            update_data=_evaluation_updates, error_tolerance=1e-8, max_iterations=50,
            output_component_types={ComponentType.node}, continue_on_batch_error=True,
        )
        _failed = set() if _model.batch_error is None else set(map(int, _model.batch_error.failed_scenarios))
        _mask = _np.array([i not in _failed for i in range(_interval_sample_count)], dtype=bool)
        _evaluation_masks[_base_method] = _mask
    _common = _evaluation_masks["ILSE"] & _evaluation_masks["NRSE"]
    if int(_common.sum()) < 2:
        raise RuntimeError("Fewer than two scenarios converged for both methods")

    _output_components = {component for component in _fields if component in _truth}
    # Compute the sensor-only gain inverse once at the clean-centered NRSE
    # solution. Its intervals below use the scenario-wise NRSE estimates as
    # centers, but this fixed reference-point sigma as their half-width.
    _gain_model_instance = PowerGridModel(_centered, system_frequency=50.0)
    _gain_center_result = _gain_model_instance.calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        error_tolerance=1e-8,
        max_iterations=50,
        output_component_types=_output_components,
    )
    _gain_model = build_gain_matrix(
        _centered,
        state_estimation_result=_gain_center_result,
        max_gain_matrix_bytes=None,
    )
    _gain_measurement_model = _gain_model.measurement_model
    _gain_inverse = _np.linalg.inv(_gain_model.gain_matrix)
    del _gain_model
    _gain_result = propagate_covariance(
        _centered,
        _gain_measurement_model,
        _gain_inverse,
        _gain_center_result,
    )
    del _gain_inverse, _gain_center_result

    for _base_method, _calculation_method in _base_methods.items():
        # Estimate the MC interval widths from an independent training batch.
        _training_model = PowerGridModel(_centered, system_frequency=50.0)
        _training_result = _training_model.calculate_state_estimation(
            symmetric=True,
            calculation_method=_calculation_method,
            update_data=_training_updates,
            error_tolerance=1e-8,
            max_iterations=50,
            output_component_types=_output_components,
            continue_on_batch_error=True,
        )
        _training_failed = (
            set()
            if _training_model.batch_error is None
            else set(map(int, _training_model.batch_error.failed_scenarios))
        )
        _training_mask = _np.array(
            [i not in _training_failed for i in range(_interval_sample_count)],
            dtype=bool,
        )
        _training_count = int(_training_mask.sum())
        if _training_count < 2:
            raise RuntimeError(f"Fewer than two MC training scenarios converged for {_base_method}")
        _mc_sigmas = {}
        for _component, _fields_for_component in _fields.items():
            if _component not in _training_result:
                continue
            for _, _value_field, _, _ in _fields_for_component:
                if _value_field in _training_result[_component].dtype.names:
                    _mc_sigmas[_component, _value_field] = _np.std(
                        _training_result[_component][_training_mask][_value_field],
                        axis=0,
                        ddof=1,
                    )
        del _training_result

        _model = PowerGridModel(_centered, system_frequency=50.0)
        _method_result = _model.calculate_state_estimation(
            symmetric=True, calculation_method=_calculation_method, calculate_uncertainty=True,
            update_data=_evaluation_updates, error_tolerance=1e-8, max_iterations=50,
            output_component_types=_output_components, continue_on_batch_error=True,
        )
        _failed = set() if _model.batch_error is None else set(map(int, _model.batch_error.failed_scenarios))
        _uq_mask = _np.array([i not in _failed for i in range(_interval_sample_count)], dtype=bool)
        if not _np.array_equal(_uq_mask, _evaluation_masks[_base_method]):
            raise RuntimeError(f"UQ changed the converged scenarios for {_base_method}")
        _analytical_method = f"Analytical UQ {_base_method}"
        _mc_method = f"Monte Carlo UQ {_base_method}"
        for _component, _fields_for_component in _fields.items():
            if _component not in _method_result or _component not in _truth:
                continue
            _ids = _truth[_component][AttributeType.id]
            for _index, _ident in enumerate(_ids):
                for _terminal in ("from", "to") if _component != ComponentType.node else ("—",):
                    _terminal_fields = []
                    for _q, _vf, _sf, _unit in _fields_for_component:
                        if _component != ComponentType.node and not _vf.endswith("_" + _terminal):
                            continue
                        _terminal_fields.append((_q, _vf, _sf, _unit))
                    for _q, _vf, _sf, _unit in _terminal_fields:
                        if (
                            _vf not in _method_result[_component].dtype.names
                            or _sf not in _method_result[_component].dtype.names
                            or (_component, _vf) not in _mc_sigmas
                        ):
                            continue
                        _y_true = float(_truth[_component][_index][_vf])
                        _y_hat = _method_result[_component][_common, _index][_vf]
                        _analytical_sigma = _method_result[_component][_common, _index][_sf]
                        _mc_sigma = float(_mc_sigmas[_component, _vf][_index])
                        _method_converged = int(_evaluation_masks[_base_method].sum())
                        _all_rows.extend(_score_output(
                            _component, _ident, _terminal, _q, _unit, _y_true,
                            _y_hat, _analytical_sigma, _mc_sigma,
                            _analytical_method, _base_method,
                            "scenario-specific analytical", _setting,
                            _interval_sample_count, _method_converged, 0,
                        ))
                        _all_rows.extend(_score_output(
                            _component, _ident, _terminal, _q, _unit, _y_true,
                            _y_hat, _mc_sigma, _mc_sigma,
                            _mc_method, _base_method,
                            "independent Monte Carlo", _setting,
                            _interval_sample_count, _method_converged, _training_count,
                        ))
                        if (
                            _base_method == "NRSE"
                            and _component in _gain_result
                            and _sf in _gain_result[_component].dtype.names
                        ):
                            _gain_sigma = float(_gain_result[_component][_index][_sf])
                            _all_rows.extend(_score_output(
                                _component, _ident, _terminal, _q, _unit, _y_true,
                                _y_hat, _gain_sigma, _mc_sigma,
                                "Gain inverse NumPy", "NRSE",
                                "clean reference-point gain inverse", _setting,
                                _interval_sample_count, _method_converged, 1,
                            ))
        del _method_result
    del _gain_result
_out = pd.DataFrame(_all_rows)
_out.to_csv(output_directory / f"{output_slug}_one_sigma_interval_scores.csv", index=False)
"""
    return (
        code.replace("__SAMPLE_COUNT__", str(SAMPLE_COUNT))
        .replace("__EVALUATION_SEED__", str(EVALUATION_SEED))
        .replace("__MC_TRAINING_SEED__", str(MONTE_CARLO_TRAINING_SEED))
    )


def run_network(key: str) -> None:
    notebook_name, output_slug = NOTEBOOKS[key]
    output_directory = OUTPUT_ROOT / output_slug
    output_directory.mkdir(parents=True, exist_ok=True)
    code = (
        setup_code(EXAMPLE_ROOT / notebook_name, output_slug) + f"\noutput_slug = {output_slug!r}\n" + analysis_code()
    )
    with tempfile.TemporaryDirectory(prefix=f"interval-{key}-") as temporary_directory:
        script = Path(temporary_directory) / f"{key}_interval.py"
        script.write_text(code, encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {"MPLBACKEND": "Agg", "MPLCONFIGDIR": str(Path(temporary_directory) / "mplconfig"), "PYTHONUNBUFFERED": "1"}
        )
        Path(env["MPLCONFIGDIR"]).mkdir()
        with (output_directory / "one_sigma_interval_execution.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [str(PYTHON), str(script)], cwd=PGM_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("networks", nargs="*", choices=NOTEBOOKS, default=list(NOTEBOOKS))
    args = parser.parse_args()
    for network in args.networks:
        print(f"Running {network} ...", flush=True)
        run_network(network)
        print(f"Completed {network}.", flush=True)


if __name__ == "__main__":
    main()
