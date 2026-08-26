# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from power_grid_model import (
    AttributeType as AT,
    BranchSide,
    CalculationMethod,
    ComponentType as CT,
    DatasetType as DT,
    LoadGenType,
    PowerGridModel,
    WindingType,
    initialize_array,
)

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "examples"))
from gain_covariance import propagate_covariance
from gain_matrix import build_gain_matrix

from .test_example_gain_matrix import _state_estimation_input

_NODE_SIGMA_FIELDS = (AT.u_pu_sigma, AT.u_sigma, AT.u_angle_sigma, AT.p_sigma, AT.q_sigma)
_BRANCH_SIGMA_FIELDS = (
    AT.p_from_sigma,
    AT.q_from_sigma,
    AT.i_from_sigma,
    AT.p_to_sigma,
    AT.q_to_sigma,
    AT.i_to_sigma,
)


def _analytical_nrse(input_data: dict) -> dict:
    return PowerGridModel(input_data, system_frequency=50.0).calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        calculate_uncertainty=True,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )


def _assert_propagated_sigmas_match(input_data: dict, branch_component: CT) -> None:
    analytical = _analytical_nrse(input_data)
    gain_model = build_gain_matrix(input_data, state_estimation_result=analytical)
    state_covariance = np.linalg.inv(gain_model.gain_matrix)
    propagated = propagate_covariance(
        input_data,
        gain_model.measurement_model,
        state_covariance,
        analytical,
    )
    for field in _NODE_SIGMA_FIELDS:
        np.testing.assert_allclose(propagated[CT.node][field], analytical[CT.node][field], rtol=3.0e-6, atol=1.0e-10)
    for field in _BRANCH_SIGMA_FIELDS:
        np.testing.assert_allclose(
            propagated[branch_component][field],
            analytical[branch_component][field],
            rtol=3.0e-6,
            atol=1.0e-8,
        )


def test_gain_covariance_matches_analytical_nrse_for_lines() -> None:
    _assert_propagated_sigmas_match(_state_estimation_input(), CT.line)


def test_gain_covariance_matches_with_reduced_angle_reference() -> None:
    input_data = _state_estimation_input()
    input_data[CT.sym_voltage_sensor][AT.u_angle_measured] = np.nan
    analytical = _analytical_nrse(input_data)
    gain_model = build_gain_matrix(input_data, state_estimation_result=analytical)
    assert gain_model.measurement_model.fixed_angle_reference_node_ids == (0,)

    propagated = propagate_covariance(
        input_data,
        gain_model.measurement_model,
        np.linalg.inv(gain_model.gain_matrix),
        analytical,
    )
    assert propagated[CT.node][AT.u_angle_sigma][0] == 0.0
    for field in _NODE_SIGMA_FIELDS:
        np.testing.assert_allclose(propagated[CT.node][field], analytical[CT.node][field], rtol=3.0e-6, atol=1.0e-10)
    for field in _BRANCH_SIGMA_FIELDS:
        np.testing.assert_allclose(propagated[CT.line][field], analytical[CT.line][field], rtol=3.0e-6, atol=1.0e-8)


def _transformer_state_estimation_input() -> dict:
    node = initialize_array(DT.input, CT.node, 2)
    node[AT.id] = [0, 1]
    node[AT.u_rated] = [10_500.0, 420.0]

    transformer = initialize_array(DT.input, CT.transformer, 1)
    transformer[AT.id] = [10]
    transformer[AT.from_node] = [0]
    transformer[AT.to_node] = [1]
    transformer[AT.from_status] = 1
    transformer[AT.to_status] = 1
    transformer[AT.u1] = [10_750.0]
    transformer[AT.u2] = [420.0]
    transformer[AT.sn] = [400_000.0]
    transformer[AT.uk] = [0.04]
    transformer[AT.pk] = [3_750.0]
    transformer[AT.i0] = [0.002]
    transformer[AT.p0] = [500.0]
    transformer[AT.winding_from] = WindingType.delta
    transformer[AT.winding_to] = WindingType.wye_n
    transformer[AT.clock] = [5]
    transformer[AT.tap_side] = BranchSide.from_side
    transformer[AT.tap_pos] = [1]
    transformer[AT.tap_min] = [-2]
    transformer[AT.tap_max] = [2]
    transformer[AT.tap_nom] = [0]
    transformer[AT.tap_size] = [250.0]

    source = initialize_array(DT.input, CT.source, 1)
    source[AT.id] = [20]
    source[AT.node] = [0]
    source[AT.status] = [1]
    source[AT.u_ref] = [1.02]
    source[AT.u_ref_angle] = [0.01]

    load = initialize_array(DT.input, CT.sym_load, 1)
    load[AT.id] = [21]
    load[AT.node] = [1]
    load[AT.status] = [1]
    load[AT.type] = LoadGenType.const_power
    load[AT.p_specified] = [80_000.0]
    load[AT.q_specified] = [20_000.0]

    base_input = {CT.node: node, CT.transformer: transformer, CT.source: source, CT.sym_load: load}
    power_flow = PowerGridModel(base_input, system_frequency=50.0).calculate_power_flow(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        error_tolerance=1.0e-10,
        max_iterations=100,
    )
    voltage_sensor = initialize_array(DT.input, CT.sym_voltage_sensor, 2)
    voltage_sensor[AT.id] = [100, 101]
    voltage_sensor[AT.measured_object] = [0, 1]
    voltage_sensor[AT.u_sigma] = [20.0, 2.0]
    voltage_sensor[AT.u_measured] = power_flow[CT.node][AT.u]
    voltage_sensor[AT.u_angle_measured] = power_flow[CT.node][AT.u_angle]
    return {**base_input, CT.sym_voltage_sensor: voltage_sensor}


def test_gain_covariance_matches_analytical_nrse_for_transformer() -> None:
    _assert_propagated_sigmas_match(_transformer_state_estimation_input(), CT.transformer)


def test_wrong_state_covariance_shape_is_rejected() -> None:
    input_data = _state_estimation_input()
    analytical = _analytical_nrse(input_data)
    gain_model = build_gain_matrix(input_data, state_estimation_result=analytical)
    with np.testing.assert_raises_regex(ValueError, "Expected state covariance shape"):
        propagate_covariance(
            input_data,
            gain_model.measurement_model,
            np.eye(2),
            analytical,
        )
