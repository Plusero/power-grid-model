# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from power_grid_model import (
    AngleMeasurementType,
    AttributeType as AT,
    ComponentType as CT,
    DatasetType as DT,
    MeasuredTerminalType,
    initialize_array,
)

WORKFLOW_ROOT = Path(__file__).parents[2] / "docs" / "examples" / "tsg_wls_uq"
sys.path.insert(0, str(WORKFLOW_ROOT))

from build_matrix_sparsity import (  # noqa: E402
    BASE_POWER_3P,
    NODE_INJECTION_TERMINAL_TYPE,
    SparsityCase,
    aggregate_bus_injection_variances,
    assemble_matrices,
    build_sparsity_artifacts,
    combined_current_sensor_variance,
    current_component_variances,
    load_case,
    select_full_measurement_angle_references,
)

MAX_COMPLEX_SIGMA_ERROR = 1.0e-6
MAX_INJECTION_JACOBIAN_ERROR = 3.0e-6
MAX_INVERSE_RESIDUAL = 1.0e-12


@pytest.fixture(scope="module")
def cigre_case() -> SparsityCase:
    """Load the non-IEEE regression case only once."""
    return load_case("cigre-mv")


def test_cigre_mv_exercises_generalized_sparsity_builder(tmp_path: Path, cigre_case: SparsityCase) -> None:
    """A transformer-containing non-33-bus case must use only derived dimensions."""
    matrices, summary, validation = build_sparsity_artifacts(
        cigre_case,
        output_root=tmp_path,
        render_figure=False,
    )

    assert matrices["y_bus"].shape == (15, 15)
    assert matrices["augmented"].shape == (30, 30)
    assert matrices["real_measurement_matrix"].shape == (82, 30)
    np.testing.assert_array_equal(matrices["real_measurement_group_counts"], [8, 24, 24, 26])
    assert summary["matrix"].tolist() == [
        "real_measurement_matrix",
        "real_gain",
        "real_gain_inverse",
        "y_bus",
        "augmented",
        "augmented_ps",
        "augmented_inverse",
        "augmented_ps_inverse",
    ]
    assert validation["maximum_complex_sigma_error"] < MAX_COMPLEX_SIGMA_ERROR
    assert validation["maximum_injection_jacobian_error"] < MAX_INJECTION_JACOBIAN_ERROR
    assert validation["real_gain_inverse_relative_residual"] < MAX_INVERSE_RESIDUAL

    assert (tmp_path / "results" / "cigre_mv_wls_uq_matrices.npz").is_file()
    assert (tmp_path / "results" / "cigre_mv_wls_uq_matrix_sparsity.csv").is_file()
    assert (tmp_path / "results" / "cigre_mv_pgm_bus_order.csv").is_file()
    assert not (tmp_path / "figures").exists()


def test_unsupported_static_shunt_fails_explicitly(tmp_path: Path, cigre_case: SparsityCase) -> None:
    """An omitted component must not silently yield an incomplete Y-bus."""
    topology = {name: [row.copy() for row in rows] for name, rows in cigre_case.topology.items()}
    topology["shunt"] = [{"id": 99_999, "node": int(topology["node"][0]["id"]), "status": 1}]
    case_with_shunt = replace(cigre_case, topology=topology, validate_complex_sigmas=False)

    with pytest.raises(NotImplementedError, match="shunt"):
        build_sparsity_artifacts(case_with_shunt, output_root=tmp_path, render_figure=False)

    shunt = initialize_array(DT.input, CT.shunt, 1)
    inputs_with_shunt = cigre_case.inputs | {CT.shunt: shunt}
    case_with_input_shunt = replace(cigre_case, inputs=inputs_with_shunt, validate_complex_sigmas=False)
    with pytest.raises(NotImplementedError, match="shunt"):
        build_sparsity_artifacts(case_with_input_shunt, output_root=tmp_path, render_figure=False)


def test_bus_injection_aggregation_covers_all_supported_terminal_types() -> None:
    """Repeated appliances, direct sensors, and exact injections follow PGM rules."""
    topology: dict[str, list[dict[str, int | float | str]]] = {
        "node": [{"id": node_id, "u_rated": 20_000.0} for node_id in range(4)],
        "sym_load": [
            {"id": 20, "node": 1, "status": 1},
            {"id": 21, "node": 2, "status": 1},
            {"id": 22, "node": 2, "status": 1},
        ],
        "sym_gen": [{"id": 40, "node": 1, "status": 1}],
        "source": [{"id": 30, "node": 0, "status": 1}],
    }
    sensors = initialize_array(DT.input, CT.sym_power_sensor, 7)
    sensors[AT.id] = np.arange(100, 107)
    sensors[AT.measured_object] = [20, 20, 40, 30, 21, 1, 2]
    sensors[AT.measured_terminal_type] = [
        MeasuredTerminalType.load,
        MeasuredTerminalType.load,
        MeasuredTerminalType.generator,
        MeasuredTerminalType.source,
        MeasuredTerminalType.load,
        NODE_INJECTION_TERMINAL_TYPE,
        NODE_INJECTION_TERMINAL_TYPE,
    ]
    p_sigmas = np.asarray([10_000.0, 20_000.0, 30_000.0, 40_000.0, 12_000.0, 25_000.0, 35_000.0])
    q_sigmas = 0.5 * p_sigmas
    sensors[AT.p_sigma] = p_sigmas
    sensors[AT.q_sigma] = q_sigmas

    observed = aggregate_bus_injection_variances(topology, {CT.sym_power_sensor: sensors})

    def variance(sigma: float) -> float:
        return (sigma / BASE_POWER_3P) ** 2

    repeated_load = 1.0 / (1.0 / variance(10_000.0) + 1.0 / variance(20_000.0))
    bus_1_appliances = repeated_load + variance(30_000.0)
    expected_bus_1_p = 1.0 / (1.0 / bus_1_appliances + 1.0 / variance(25_000.0))
    expected_bus_1_q = 0.25 * expected_bus_1_p
    np.testing.assert_allclose(observed[0], (variance(40_000.0), variance(20_000.0)))
    np.testing.assert_allclose(observed[1], (expected_bus_1_p, expected_bus_1_q))
    np.testing.assert_allclose(observed[2], (variance(35_000.0), variance(17_500.0)))
    np.testing.assert_array_equal(observed[3], (0.0, 0.0))


def test_repeated_current_sensors_are_combined_by_cartesian_channel() -> None:
    """PGM combines repeated real and imaginary current variances separately."""
    sensors = initialize_array(DT.input, CT.sym_current_sensor, 2)
    sensors[AT.i_measured] = [80.0, 120.0]
    sensors[AT.i_sigma] = [4.0, 9.0]
    sensors[AT.i_angle_measured] = [0.2, 1.1]
    sensors[AT.i_angle_sigma] = [0.01, 0.03]
    sensor_rows = [sensor for sensor in sensors]

    component_variances = np.asarray(
        [current_component_variances(sensor, 20_000.0, BASE_POWER_3P) for sensor in sensor_rows]
    )
    expected = float(np.sum(1.0 / np.sum(1.0 / component_variances, axis=0)))
    total_variance_first = float(1.0 / np.sum(1.0 / np.sum(component_variances, axis=1)))

    observed = combined_current_sensor_variance(sensor_rows, 20_000.0)
    assert not np.isclose(expected, total_variance_first, rtol=1.0e-3)
    np.testing.assert_allclose(observed, expected)


def test_disabled_repeated_current_sensor_is_ignored(tmp_path: Path, cigre_case: SparsityCase) -> None:
    """Infinite-sigma duplicate and unsupported-angle sensors add no precision."""
    baseline, _, _, _ = assemble_matrices(cigre_case)
    current_sensors = cigre_case.inputs[CT.sym_current_sensor]
    disabled_sensors = np.repeat(current_sensors[:1], 2)
    maximum_id = max(int(np.max(component[AT.id])) for component in cigre_case.inputs.values() if component.size)
    disabled_sensors[AT.id] = [maximum_id + 1, maximum_id + 2]
    disabled_sensors[AT.i_sigma] = np.inf
    disabled_sensors[1][AT.measured_object] = cigre_case.inputs[CT.transformer][0][AT.id]
    disabled_sensors[1][AT.angle_measurement_type] = AngleMeasurementType.local_angle
    inputs = cigre_case.inputs | {CT.sym_current_sensor: np.concatenate((current_sensors, disabled_sensors))}

    observed, _, _ = build_sparsity_artifacts(
        replace(cigre_case, inputs=inputs, validate_complex_sigmas=False),
        output_root=tmp_path,
        render_figure=False,
    )
    np.testing.assert_allclose(observed["augmented"], baseline["augmented"])


def test_angle_reference_selection_uses_injection_row_connectivity() -> None:
    """The complete H, not the branch-only subset, determines angle components."""
    solver_bus_ids = np.asarray([3, 2, 1, 0])
    measurement_matrix = np.zeros((3, 2 * solver_bus_ids.size))
    measurement_matrix[0, [0, 1]] = [1.0, -1.0]
    measurement_matrix[1, [2, 3]] = [1.0, -1.0]
    measurement_matrix[2, [1, 2]] = [1.0, -1.0]
    source = initialize_array(DT.input, CT.source, 1)
    source[AT.node] = [0]
    source[AT.status] = [1]
    inputs = {CT.source: source}

    assert select_full_measurement_angle_references(measurement_matrix, solver_bus_ids, inputs) == (0,)

    anchored_matrix = np.vstack((measurement_matrix, np.zeros(2 * solver_bus_ids.size)))
    anchored_matrix[-1, 0] = 1.0
    assert select_full_measurement_angle_references(anchored_matrix, solver_bus_ids, inputs) == ()
