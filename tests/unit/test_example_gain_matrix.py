# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from power_grid_model import (
    AngleMeasurementType,
    AttributeType as AT,
    CalculationMethod,
    ComponentType as CT,
    DatasetType as DT,
    LoadGenType,
    MeasuredTerminalType,
    PowerGridModel,
    initialize_array,
)

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "examples"))
from gain_matrix import CsrMatrix, _line_terminal_admittances, build_gain_matrix, build_measurement_model


def _state_estimation_input() -> dict:
    node = initialize_array(DT.input, CT.node, 3)
    node[AT.id] = [0, 1, 2]
    node[AT.u_rated] = 20_000.0

    line = initialize_array(DT.input, CT.line, 2)
    line[AT.id] = [10, 11]
    line[AT.from_node] = [0, 1]
    line[AT.to_node] = [1, 2]
    line[AT.from_status] = 1
    line[AT.to_status] = 1
    line[AT.r1] = 0.04
    line[AT.x1] = 0.08
    line[AT.c1] = 0.0
    line[AT.tan1] = 0.0
    line[AT.i_n] = 1.0e9
    line[AT.r0] = 0.04
    line[AT.x0] = 0.08
    line[AT.c0] = 0.0
    line[AT.tan0] = 0.0

    load = initialize_array(DT.input, CT.sym_load, 2)
    load[AT.id] = [20, 21]
    load[AT.node] = [1, 2]
    load[AT.status] = 1
    load[AT.type] = LoadGenType.const_power
    load[AT.p_specified] = [1_000.0, 1_200.0]
    load[AT.q_specified] = [300.0, 400.0]

    source = initialize_array(DT.input, CT.source, 1)
    source[AT.id] = [30]
    source[AT.node] = [0]
    source[AT.status] = [1]
    source[AT.u_ref] = [1.0]
    if "u_ref_angle" in (source.dtype.names or ()):
        source[AT.u_ref_angle] = [0.0]

    base_input = {CT.node: node, CT.line: line, CT.sym_load: load, CT.source: source}
    power_flow = PowerGridModel(base_input, system_frequency=50.0).calculate_power_flow(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )

    voltage_sensor = initialize_array(DT.input, CT.sym_voltage_sensor, 2)
    voltage_sensor[AT.id] = [100, 101]
    voltage_sensor[AT.measured_object] = [0, 2]
    voltage_sensor[AT.u_sigma] = [30.0, 40.0]
    voltage_sensor[AT.u_measured] = power_flow[CT.node][AT.u][[0, 2]]
    voltage_sensor[AT.u_angle_measured] = [power_flow[CT.node][AT.u_angle][0], np.nan]

    power_sensor = initialize_array(DT.input, CT.sym_power_sensor, 2)
    power_sensor[AT.id] = [102, 103]
    power_sensor[AT.measured_object] = line[AT.id]
    power_sensor[AT.measured_terminal_type] = [
        MeasuredTerminalType.branch_to,
        MeasuredTerminalType.branch_from,
    ]
    power_sensor[AT.p_sigma] = 25_000.0
    power_sensor[AT.q_sigma] = 10_000.0
    power_sensor[AT.p_measured] = [power_flow[CT.line][AT.p_to][0], power_flow[CT.line][AT.p_from][1]]
    power_sensor[AT.q_measured] = [power_flow[CT.line][AT.q_to][0], power_flow[CT.line][AT.q_from][1]]

    return {**base_input, CT.sym_voltage_sensor: voltage_sensor, CT.sym_power_sensor: power_sensor}


def _add_global_current_sensor(
    input_data: dict,
    *,
    line_index: int = 0,
    terminal_type: MeasuredTerminalType = MeasuredTerminalType.branch_from,
) -> complex:
    power_flow = PowerGridModel(input_data, system_frequency=50.0).calculate_power_flow(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )
    line = input_data[CT.line][line_index]
    node_ids = input_data[CT.node][AT.id]
    node_index_by_id = {int(node_id): index for index, node_id in enumerate(node_ids)}
    if terminal_type == MeasuredTerminalType.branch_from:
        node_index = node_index_by_id[int(line[AT.from_node])]
        p_field, q_field = AT.p_from, AT.q_from
    else:
        node_index = node_index_by_id[int(line[AT.to_node])]
        p_field, q_field = AT.p_to, AT.q_to
    voltage = power_flow[CT.node][AT.u][node_index] * np.exp(1.0j * power_flow[CT.node][AT.u_angle][node_index])
    power = power_flow[CT.line][p_field][line_index] + 1.0j * power_flow[CT.line][q_field][line_index]
    current = np.conj(power / (np.sqrt(3.0) * voltage))

    current_sensor = initialize_array(DT.input, CT.sym_current_sensor, 1)
    current_sensor[AT.id] = [104]
    current_sensor[AT.measured_object] = [line[AT.id]]
    current_sensor[AT.measured_terminal_type] = [terminal_type]
    current_sensor[AT.angle_measurement_type] = [AngleMeasurementType.global_angle]
    current_sensor[AT.i_sigma] = [12.0]
    current_sensor[AT.i_angle_sigma] = [0.015]
    current_sensor[AT.i_measured] = [abs(current)]
    current_sensor[AT.i_angle_measured] = [np.angle(current)]
    input_data[CT.sym_current_sensor] = current_sensor
    return current


def _make_transformer(*, transformer_id: int = 40, from_node: int = 1, to_node: int = 2) -> np.ndarray:
    transformer = initialize_array(DT.input, CT.transformer, 1)
    transformer[AT.id] = [transformer_id]
    transformer[AT.from_node] = [from_node]
    transformer[AT.to_node] = [to_node]
    transformer[AT.from_status] = [1]
    transformer[AT.to_status] = [1]
    transformer[AT.u1] = [20_000.0]
    transformer[AT.u2] = [20_000.0]
    transformer[AT.sn] = [1.0e6]
    transformer[AT.uk] = [0.1]
    transformer[AT.pk] = [10_000.0]
    transformer[AT.i0] = [0.01]
    transformer[AT.p0] = [100.0]
    transformer[AT.winding_from] = [1]
    transformer[AT.winding_to] = [1]
    transformer[AT.clock] = [0]
    transformer[AT.tap_side] = [0]
    transformer[AT.tap_pos] = [0]
    transformer[AT.tap_min] = [-2]
    transformer[AT.tap_max] = [2]
    transformer[AT.tap_nom] = [0]
    transformer[AT.tap_size] = [100.0]
    return transformer


def _measurement_values(state_vector: np.ndarray, *, capacitance: float = 0.0, loss_tangent: float = 0.0) -> np.ndarray:
    node_count = 3
    voltage_magnitudes = state_vector[node_count:]
    voltage_angles = state_vector[:node_count]
    voltage = voltage_magnitudes * np.exp(1.0j * voltage_angles)
    base_admittance = 1.0e6 / 20_000.0**2
    series_admittance = 1.0 / (0.04 + 0.08j) / base_admittance
    shunt_admittance = 2.0 * np.pi * 50.0 * capacitance * (loss_tangent + 1.0j) / base_admittance
    self_admittance = series_admittance + 0.5 * shunt_admittance
    line_0_to_current = -series_admittance * voltage[0] + self_admittance * voltage[1]
    line_0_to_power = voltage[1] * np.conj(line_0_to_current)
    line_1_from_current = self_admittance * voltage[1] - series_admittance * voltage[2]
    line_1_from_power = voltage[1] * np.conj(line_1_from_current)
    return np.asarray(
        [
            voltage_magnitudes[0],
            voltage_angles[0],
            voltage_magnitudes[2],
            line_0_to_power.real,
            line_0_to_power.imag,
            line_1_from_power.real,
            line_1_from_power.imag,
        ]
    )


def test_analytical_measurement_matrix_matches_central_difference() -> None:
    gain_model = build_gain_matrix(_state_estimation_input())
    measurement_model = gain_model.measurement_model
    analytical = measurement_model.measurement_matrix.to_dense()
    numerical = np.empty_like(analytical)
    step = 1.0e-6
    for column in range(measurement_model.state_vector.size):
        plus = measurement_model.state_vector.copy()
        minus = measurement_model.state_vector.copy()
        plus[column] += step
        minus[column] -= step
        numerical[:, column] = (_measurement_values(plus) - _measurement_values(minus)) / (2.0 * step)
    np.testing.assert_allclose(analytical, numerical, rtol=2.0e-6, atol=2.0e-6)


def test_measurement_model_can_retain_angle_columns_for_later_rows() -> None:
    """Callers adding measurements later can defer angle-reference selection."""
    input_data = _state_estimation_input()
    input_data[CT.sym_voltage_sensor][AT.u_angle_measured] = np.nan

    reduced = build_gain_matrix(input_data).measurement_model
    full = build_measurement_model(input_data, retain_all_angle_columns=True)

    full_state_size = 2 * input_data[CT.node].size
    assert reduced.measurement_matrix.shape[1] == full_state_size - 1
    assert reduced.fixed_angle_reference_node_ids == (0,)
    assert full.measurement_matrix.shape[1] == full_state_size
    assert full.fixed_angle_reference_node_ids == ()


def test_line_shunt_jacobian_matches_central_difference() -> None:
    input_data = _state_estimation_input()
    capacitance = 2.0e-6
    loss_tangent = 0.01
    input_data[CT.line][AT.c1] = capacitance
    input_data[CT.line][AT.tan1] = loss_tangent
    measurement_model = build_gain_matrix(input_data).measurement_model
    analytical = measurement_model.measurement_matrix.to_dense()
    numerical = np.empty_like(analytical)
    step = 1.0e-6
    for column in range(measurement_model.state_vector.size):
        plus = measurement_model.state_vector.copy()
        minus = measurement_model.state_vector.copy()
        plus[column] += step
        minus[column] -= step
        numerical[:, column] = (
            _measurement_values(plus, capacitance=capacitance, loss_tangent=loss_tangent)
            - _measurement_values(minus, capacitance=capacitance, loss_tangent=loss_tangent)
        ) / (2.0 * step)
    np.testing.assert_allclose(analytical, numerical, rtol=2.0e-6, atol=2.0e-6)


def test_global_current_jacobian_weights_and_pgm_state_sigmas() -> None:
    input_data = _state_estimation_input()
    _add_global_current_sensor(input_data)
    current_sensor = input_data[CT.sym_current_sensor]
    current_sensor[AT.i_measured][0] += 5.0
    current_sensor[AT.i_angle_measured][0] += 0.04
    measured_current = current_sensor[AT.i_measured][0] * np.exp(1.0j * current_sensor[AT.i_angle_measured][0])
    result = PowerGridModel(input_data, system_frequency=50.0).calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        calculate_uncertainty=True,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )
    gain_model = build_gain_matrix(input_data, state_estimation_result=result)
    measurement_model = gain_model.measurement_model
    analytical = measurement_model.measurement_matrix.to_dense()[-2:]

    node_count = 3
    node_index_by_id = {int(node_id): index for index, node_id in enumerate(input_data[CT.node][AT.id])}
    _, _, y_ff, y_ft, _, _ = _line_terminal_admittances(input_data, 50.0, node_index_by_id)[10]

    def current_components(state_vector: np.ndarray) -> np.ndarray:
        voltage = state_vector[node_count:] * np.exp(1.0j * state_vector[:node_count])
        current = y_ff * voltage[0] + y_ft * voltage[1]
        return np.asarray([current.real, current.imag])

    numerical = np.empty_like(analytical)
    step = 1.0e-6
    for column in range(measurement_model.state_vector.size):
        plus = measurement_model.state_vector.copy()
        minus = measurement_model.state_vector.copy()
        plus[column] += step
        minus[column] -= step
        numerical[:, column] = (current_components(plus) - current_components(minus)) / (2.0 * step)
    np.testing.assert_allclose(analytical, numerical, rtol=2.0e-6, atol=2.0e-6)

    base_current = 1.0e6 / (np.sqrt(3.0) * 20_000.0)
    normalized_magnitude = abs(measured_current) / base_current
    normalized_sigma = 12.0 / base_current
    angle = np.angle(measured_current)
    angle_variance = 0.015**2
    magnitude_variance = normalized_sigma**2
    real_variance = (
        magnitude_variance * np.cos(angle) ** 2
        + normalized_magnitude**2 * angle_variance * np.sin(angle) ** 2
        + 0.5 * normalized_magnitude**2 * angle_variance**2 * np.cos(angle) ** 2
        + magnitude_variance * angle_variance * np.sin(angle) ** 2
    )
    imag_variance = (
        magnitude_variance * np.sin(angle) ** 2
        + normalized_magnitude**2 * angle_variance * np.cos(angle) ** 2
        + 0.5 * normalized_magnitude**2 * angle_variance**2 * np.sin(angle) ** 2
        + magnitude_variance * angle_variance * np.cos(angle) ** 2
    )
    np.testing.assert_allclose(
        measurement_model.weight_matrix.diagonal[-2:],
        [1.0 / real_variance, 1.0 / imag_variance],
    )

    covariance = np.linalg.inv(gain_model.gain_matrix)
    np.testing.assert_allclose(
        np.sqrt(np.diag(covariance)[:node_count]),
        result[CT.node][AT.u_angle_sigma],
        rtol=2.0e-6,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.sqrt(np.diag(covariance)[node_count:]),
        result[CT.node][AT.u_pu_sigma],
        rtol=2.0e-6,
        atol=1.0e-12,
    )


def test_transformer_terminal_power_and_current_sensors_match_pgm_state_sigmas() -> None:
    input_data = _state_estimation_input()
    input_data[CT.transformer] = _make_transformer(from_node=0, to_node=1)
    input_data[CT.transformer][AT.tap_pos] = [1]
    power_flow = PowerGridModel(input_data, system_frequency=50.0).calculate_power_flow(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )
    voltage_sensor = input_data[CT.sym_voltage_sensor]
    voltage_sensor[AT.u_measured] = power_flow[CT.node][AT.u][[0, 2]]
    voltage_sensor[AT.u_angle_measured][0] = power_flow[CT.node][AT.u_angle][0]
    line_power_sensor = input_data[CT.sym_power_sensor]
    line_power_sensor[AT.p_measured] = [
        power_flow[CT.line][AT.p_to][0],
        power_flow[CT.line][AT.p_from][1],
    ]
    line_power_sensor[AT.q_measured] = [
        power_flow[CT.line][AT.q_to][0],
        power_flow[CT.line][AT.q_from][1],
    ]

    power_sensor = initialize_array(DT.input, CT.sym_power_sensor, 3)
    power_sensor[:2] = line_power_sensor
    power_sensor[AT.id][2] = 105
    power_sensor[AT.measured_object][2] = 40
    power_sensor[AT.measured_terminal_type][2] = MeasuredTerminalType.branch_to
    power_sensor[AT.p_sigma][2] = 25_000.0
    power_sensor[AT.q_sigma][2] = 10_000.0
    power_sensor[AT.p_measured][2] = power_flow[CT.transformer][AT.p_to][0]
    power_sensor[AT.q_measured][2] = power_flow[CT.transformer][AT.q_to][0]
    input_data[CT.sym_power_sensor] = power_sensor

    from_voltage = power_flow[CT.node][AT.u][0] * np.exp(1.0j * power_flow[CT.node][AT.u_angle][0])
    from_power = power_flow[CT.transformer][AT.p_from][0] + 1.0j * power_flow[CT.transformer][AT.q_from][0]
    from_current = np.conj(from_power / (np.sqrt(3.0) * from_voltage))
    current_sensor = initialize_array(DT.input, CT.sym_current_sensor, 1)
    current_sensor[AT.id] = [104]
    current_sensor[AT.measured_object] = [40]
    current_sensor[AT.measured_terminal_type] = [MeasuredTerminalType.branch_from]
    current_sensor[AT.angle_measurement_type] = [AngleMeasurementType.global_angle]
    current_sensor[AT.i_sigma] = [12.0]
    current_sensor[AT.i_angle_sigma] = [0.015]
    current_sensor[AT.i_measured] = [abs(from_current)]
    current_sensor[AT.i_angle_measured] = [np.angle(from_current)]
    input_data[CT.sym_current_sensor] = current_sensor

    result = PowerGridModel(input_data, system_frequency=50.0).calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        calculate_uncertainty=True,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )
    gain_model = build_gain_matrix(input_data, state_estimation_result=result)
    assert sum("transformer 40" in label for label in gain_model.measurement_model.measurement_labels) == 4  # noqa: PLR2004
    covariance = np.linalg.inv(gain_model.gain_matrix)
    np.testing.assert_allclose(
        np.sqrt(np.diag(covariance)[:3]),
        result[CT.node][AT.u_angle_sigma],
        rtol=2.0e-6,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.sqrt(np.diag(covariance)[3:]),
        result[CT.node][AT.u_pu_sigma],
        rtol=2.0e-6,
        atol=1.0e-12,
    )


def test_gain_matrix_matches_dense_definition_and_pgm_voltage_sigma() -> None:
    input_data = _state_estimation_input()
    result = PowerGridModel(input_data, system_frequency=50.0).calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        calculate_uncertainty=True,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )
    gain_model = build_gain_matrix(input_data, state_estimation_result=result)
    measurement_model = gain_model.measurement_model
    dense_h = measurement_model.measurement_matrix.to_dense()
    dense_w = measurement_model.weight_matrix.to_dense()
    np.testing.assert_allclose(
        measurement_model.weight_matrix.diagonal,
        [(20_000 / 30) ** 2, (20_000 / 30) ** 2, (20_000 / 40) ** 2, 1_600, 10_000, 1_600, 10_000],
    )
    np.testing.assert_allclose(gain_model.gain_matrix, dense_h.T @ dense_w @ dense_h, rtol=1.0e-13, atol=1.0e-9)
    np.testing.assert_allclose(gain_model.gain_matrix, gain_model.gain_matrix.T, rtol=0.0, atol=1.0e-9)

    covariance = np.linalg.inv(gain_model.gain_matrix)
    angle_sigma = np.sqrt(np.diag(covariance)[:3])
    voltage_sigma = np.sqrt(np.diag(covariance)[3:])
    np.testing.assert_allclose(angle_sigma, result[CT.node][AT.u_angle_sigma], rtol=2.0e-6, atol=1.0e-12)
    np.testing.assert_allclose(voltage_sigma, result[CT.node][AT.u_pu_sigma], rtol=2.0e-6, atol=1.0e-12)


def test_gain_matrix_memory_guard_runs_before_state_estimation() -> None:
    node = initialize_array(DT.input, CT.node, 2)
    node[AT.id] = [0, 1]
    node[AT.u_rated] = 20_000.0
    with pytest.raises(MemoryError, match="allocation limit"):
        build_gain_matrix({CT.node: node}, max_gain_matrix_bytes=1)


def test_dense_validation_views_have_allocation_guards() -> None:
    measurement_model = build_gain_matrix(_state_estimation_input()).measurement_model
    with pytest.raises(MemoryError, match="measurement matrix"):
        measurement_model.measurement_matrix.to_dense(max_dense_matrix_bytes=1)
    with pytest.raises(MemoryError, match="weight matrix"):
        measurement_model.weight_matrix.to_dense(max_dense_matrix_bytes=1)

    impossible_matrix = CsrMatrix(
        data=np.empty(0),
        column_indices=np.empty(0, dtype=np.int64),
        row_pointers=np.asarray([0], dtype=np.int64),
        shape=(2**32, 2**32),
    )
    with pytest.raises(MemoryError, match="measurement matrix"):
        impossible_matrix.to_dense(max_dense_matrix_bytes=1)


def test_apparent_power_sigma_is_split_between_p_and_q() -> None:
    input_data = _state_estimation_input()
    power_sensor = input_data[CT.sym_power_sensor]
    power_sensor[AT.p_sigma] = np.nan
    power_sensor[AT.q_sigma] = np.nan
    power_sensor[AT.power_sigma] = 20_000.0 * np.sqrt(2.0)
    gain_model = build_gain_matrix(input_data)
    np.testing.assert_allclose(gain_model.measurement_model.weight_matrix.diagonal[-4:], 2_500.0)


@pytest.mark.parametrize("use_explicit_sigmas", [False, True])
def test_infinite_power_sigma_disables_sensor(use_explicit_sigmas: bool) -> None:
    input_data = _state_estimation_input()
    state_estimation_result = PowerGridModel(input_data, system_frequency=50.0).calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )
    power_sensor = input_data[CT.sym_power_sensor]
    power_sensor[AT.power_sigma][1] = 20_000.0
    if use_explicit_sigmas:
        power_sensor[AT.p_sigma][1] = np.inf
        power_sensor[AT.q_sigma][1] = np.inf
    else:
        power_sensor[AT.p_sigma][1] = np.nan
        power_sensor[AT.q_sigma][1] = np.nan
        power_sensor[AT.power_sigma][1] = np.inf
    measurement_model = build_gain_matrix(input_data, state_estimation_result=state_estimation_result).measurement_model
    assert len(measurement_model.measurement_labels) == 5  # noqa: PLR2004
    assert all("line 11" not in label for label in measurement_model.measurement_labels)


def test_source_angle_is_an_exact_reduced_state_reference() -> None:
    input_data = _state_estimation_input()
    input_data[CT.sym_voltage_sensor][AT.u_angle_measured] = np.nan
    result = PowerGridModel(input_data, system_frequency=50.0).calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        calculate_uncertainty=True,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )
    measurement_model = build_gain_matrix(input_data, state_estimation_result=result).measurement_model
    assert measurement_model.fixed_angle_reference_node_id == 0
    assert measurement_model.state_vector.size == 5  # noqa: PLR2004

    covariance = np.linalg.inv(
        measurement_model.measurement_matrix.weighted_gram(measurement_model.weight_matrix.diagonal)
    )
    reduced_sigma = np.sqrt(np.diag(covariance))
    angle_sigma = np.insert(reduced_sigma[:2], 0, 0.0)
    voltage_sigma = reduced_sigma[2:]
    np.testing.assert_allclose(angle_sigma, result[CT.node][AT.u_angle_sigma], rtol=2.0e-6, atol=1.0e-12)
    np.testing.assert_allclose(voltage_sigma, result[CT.node][AT.u_pu_sigma], rtol=2.0e-6, atol=1.0e-12)


def test_infinite_voltage_sigma_disables_sensor() -> None:
    input_data = _state_estimation_input()
    input_data[CT.sym_voltage_sensor][AT.u_sigma][1] = np.inf
    result = PowerGridModel(input_data, system_frequency=50.0).calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        calculate_uncertainty=True,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )
    measurement_model = build_gain_matrix(input_data, state_estimation_result=result).measurement_model
    assert len(measurement_model.measurement_labels) == 6  # noqa: PLR2004

    covariance = np.linalg.inv(
        measurement_model.measurement_matrix.weighted_gram(measurement_model.weight_matrix.diagonal)
    )
    angle_sigma = np.sqrt(np.diag(covariance)[:3])
    voltage_sigma = np.sqrt(np.diag(covariance)[3:])
    np.testing.assert_allclose(angle_sigma, result[CT.node][AT.u_angle_sigma], rtol=2.0e-6, atol=1.0e-12)
    np.testing.assert_allclose(voltage_sigma, result[CT.node][AT.u_pu_sigma], rtol=2.0e-6, atol=1.0e-12)


def test_open_terminal_admittance_matches_pgm_and_zero_terminal_rows_are_skipped() -> None:
    input_data = _state_estimation_input()
    state_estimation_result = PowerGridModel(input_data, system_frequency=50.0).calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )
    input_data[CT.line][AT.c1][0] = 2.0e-6
    input_data[CT.line][AT.tan1][0] = 0.01
    input_data[CT.line][AT.to_status][0] = 0

    node_index_by_id = {int(node_id): index for index, node_id in enumerate(input_data[CT.node][AT.id])}
    _, _, y_ff, y_ft, y_tf, y_tt = _line_terminal_admittances(input_data, 50.0, node_index_by_id)[10]
    base_admittance = 1.0e6 / 20_000.0**2
    series_admittance = 1.0 / (0.04 + 0.08j) / base_admittance
    shunt_admittance = 2.0 * np.pi * 50.0 * 2.0e-6 * (0.01 + 1.0j) / base_admittance
    expected_y_ff = 0.5 * shunt_admittance + 1.0 / (1.0 / series_admittance + 2.0 / shunt_admittance)
    np.testing.assert_allclose(y_ff, expected_y_ff, rtol=1.0e-14, atol=1.0e-14)
    assert y_ft == y_tf == y_tt == 0.0

    current_sensor = initialize_array(DT.input, CT.sym_current_sensor, 1)
    current_sensor[AT.id] = [104]
    current_sensor[AT.measured_object] = [10]
    current_sensor[AT.measured_terminal_type] = [MeasuredTerminalType.branch_from]
    current_sensor[AT.angle_measurement_type] = [AngleMeasurementType.global_angle]
    current_sensor[AT.i_sigma] = [12.0]
    current_sensor[AT.i_angle_sigma] = [0.015]
    current_sensor[AT.i_measured] = [0.0]
    current_sensor[AT.i_angle_measured] = [0.0]
    input_data[CT.sym_current_sensor] = current_sensor
    measurement_model = build_gain_matrix(
        input_data,
        state_estimation_result=state_estimation_result,
    ).measurement_model
    assert all("line 10" not in label for label in measurement_model.measurement_labels)

    input_data[CT.line][AT.from_status][0] = 0
    _, _, y_ff, y_ft, y_tf, y_tt = _line_terminal_admittances(input_data, 50.0, node_index_by_id)[10]
    assert y_ff == y_ft == y_tf == y_tt == 0.0


def test_multiple_measurement_components_get_independent_angle_references() -> None:
    input_data = _state_estimation_input()
    state_estimation_result = PowerGridModel(input_data, system_frequency=50.0).calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )
    input_data[CT.line] = input_data[CT.line][:1].copy()
    input_data[CT.sym_power_sensor] = input_data[CT.sym_power_sensor][:1].copy()
    input_data[CT.sym_voltage_sensor][AT.u_angle_measured] = np.nan
    input_data[CT.transformer] = _make_transformer()

    measurement_model = build_gain_matrix(
        input_data,
        state_estimation_result=state_estimation_result,
    ).measurement_model
    assert measurement_model.fixed_angle_reference_node_ids == (0, 2)
    assert measurement_model.fixed_angle_reference_node_id is None
    assert measurement_model.state_vector.size == 4  # noqa: PLR2004


def test_exact_zero_injection_is_ignored() -> None:
    input_data = _state_estimation_input()
    state_estimation_result = PowerGridModel(input_data, system_frequency=50.0).calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )
    baseline_model = build_gain_matrix(
        input_data,
        state_estimation_result=state_estimation_result,
    ).measurement_model
    input_data[CT.sym_load][AT.status][1] = 0
    zero_injection_model = build_gain_matrix(
        input_data,
        state_estimation_result=state_estimation_result,
    ).measurement_model
    np.testing.assert_array_equal(
        zero_injection_model.measurement_matrix.to_dense(),
        baseline_model.measurement_matrix.to_dense(),
    )
    np.testing.assert_array_equal(
        zero_injection_model.weight_matrix.diagonal,
        baseline_model.weight_matrix.diagonal,
    )
    assert zero_injection_model.measurement_labels == baseline_model.measurement_labels
