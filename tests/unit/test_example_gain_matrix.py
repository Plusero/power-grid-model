# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from power_grid_model import (
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
from gain_matrix import CsrMatrix, build_gain_matrix


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


def test_multiple_islands_are_rejected() -> None:
    input_data = _state_estimation_input()
    input_data[CT.line] = input_data[CT.line][:1].copy()
    input_data[CT.sym_power_sensor] = input_data[CT.sym_power_sensor][:1].copy()
    with pytest.raises(NotImplementedError, match="one connected network"):
        build_gain_matrix(input_data)


def test_exact_zero_injection_is_rejected() -> None:
    input_data = _state_estimation_input()
    input_data[CT.sym_load][AT.status][1] = 0
    with pytest.raises(NotImplementedError, match="exact zero-injection"):
        build_gain_matrix(input_data)
