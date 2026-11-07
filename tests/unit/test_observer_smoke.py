# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from power_grid_model import CalculationMethod, ComponentType as CT, DatasetType as DT, PowerGridModel, initialize_array
from power_grid_model.errors import PowerGridError


def single_bus_observer_input():
    def rows(component, **fields):
        result = initialize_array(DT.input, component, 1)
        for name, value in fields.items():
            result[name] = value
        return result

    # Unsensed appliances keep injection free; this fixture has only one complex voltage measurement.
    return {
        CT.node: rows(CT.node, id=1, u_rated=100.0),
        CT.source: rows(CT.source, id=2, node=1, status=1, u_ref=1.0),
        CT.sym_load: rows(CT.sym_load, id=3, node=1, status=1, type=0, p_specified=100.0, q_specified=20.0),
        CT.sym_voltage_sensor: rows(
            CT.sym_voltage_sensor, id=4, measured_object=1, u_sigma=2.0, u_measured=100.0, u_angle_measured=0.2
        ),
        CT.voltage_uncertainty_observer: rows(CT.voltage_uncertainty_observer, id=100, observed_object=1, status=1),
    }


@pytest.mark.parametrize("method", [CalculationMethod.iterative_linear, CalculationMethod.newton_raphson])
def test_public_voltage_observer_matches_independent_one_measurement_oracle(method):
    data = single_bus_observer_input()
    model = PowerGridModel(data)
    ordinary = model.calculate_state_estimation(calculation_method=method, output_component_types=[CT.node])
    result = model.calculate_state_estimation(calculation_method=method)
    np.testing.assert_array_equal(result[CT.node], ordinary[CT.node])
    observer = result[CT.voltage_uncertainty_observer]
    assert observer["id"].tolist() == [100]
    assert observer["energized"].tolist() == [1]
    # ILSE splits proper complex variance between radial/tangential axes; NRSE uses real polar variances.
    expected_sigma = 2.0 / np.sqrt(2.0) if method == CalculationMethod.iterative_linear else 2.0
    np.testing.assert_allclose(observer["u_sigma"], expected_sigma, rtol=1e-10)
    np.testing.assert_allclose(observer["u_angle_sigma"], expected_sigma / 100.0, rtol=1e-10)


@pytest.mark.parametrize("method", [CalculationMethod.iterative_linear, CalculationMethod.newton_raphson])
def test_public_observer_metadata_and_empty_selection_preserve_estimates(method):
    data = single_bus_observer_input()
    model = PowerGridModel(data)
    ordinary = model.calculate_state_estimation(calculation_method=method, output_component_types=[CT.node])
    metadata = model.calculate_state_estimation(
        calculation_method=method,
        output_component_types={CT.node: None, CT.voltage_uncertainty_observer: ["id", "energized"]},
    )
    np.testing.assert_array_equal(metadata[CT.node], ordinary[CT.node])
    assert metadata[CT.voltage_uncertainty_observer]["id"].tolist() == [100]
    assert metadata[CT.voltage_uncertainty_observer]["energized"].tolist() == [1]
    empty = model.calculate_state_estimation(
        calculation_method=method,
        output_component_types={CT.node: None, CT.voltage_uncertainty_observer: []},
    )
    assert CT.voltage_uncertainty_observer not in empty
    np.testing.assert_array_equal(empty[CT.node], ordinary[CT.node])


@pytest.mark.parametrize("method", [CalculationMethod.iterative_linear, CalculationMethod.newton_raphson])
@pytest.mark.parametrize("selection", [None, ["id"]])
def test_public_observer_asymmetric_se_keeps_scalar_metadata_and_phase_sigmas(method, selection):
    model = PowerGridModel(single_bus_observer_input())
    result = model.calculate_state_estimation(
        calculation_method=method,
        symmetric=False,
        output_component_types={CT.voltage_uncertainty_observer: selection},
    )
    observer = result[CT.voltage_uncertainty_observer]
    assert observer["id"].tolist() == [100]
    if selection is None:
        assert observer["energized"].tolist() == [1]
        assert observer["u_sigma"].shape == (1, 3)
        assert np.isfinite(observer["u_sigma"]).all()
        assert np.isfinite(observer["u_angle_sigma"]).all()


@pytest.mark.parametrize("calculation", ["calculate_power_flow", "calculate_short_circuit"])
def test_public_observer_unsupported_calculation_is_argument_error(calculation):
    model = PowerGridModel(single_bus_observer_input())
    with pytest.raises(PowerGridError, match="Observer output requires"):
        getattr(model, calculation)(output_component_types={CT.voltage_uncertainty_observer: ["id"]})


def two_bus_observer_input():
    data = single_bus_observer_input()
    node = initialize_array(DT.input, CT.node, 2)
    node["id"] = [1, 5]
    node["u_rated"] = 100.0
    data[CT.node] = node
    data[CT.sym_load]["node"] = 5
    sensor = initialize_array(DT.input, CT.sym_voltage_sensor, 2)
    sensor["id"] = [4, 6]
    sensor["measured_object"] = [1, 5]
    sensor["u_sigma"] = [2.0, 3.0]
    sensor["u_measured"] = [100.0, 90.0]
    sensor["u_angle_measured"] = [0.2, 0.1]
    data[CT.sym_voltage_sensor] = sensor
    line = initialize_array(DT.input, CT.line, 1)
    for name, value in dict(
        id=7,
        from_node=1,
        to_node=5,
        from_status=1,
        to_status=1,
        r1=1.0,
        x1=2.0,
        c1=0.0,
        tan1=0.0,
        r0=1.0,
        x0=2.0,
        c0=0.0,
        tan0=0.0,
        i_n=100.0,
    ).items():
        line[name] = value
    data[CT.line] = line
    for component, ids in [(CT.power_uncertainty_observer, [101, 102]), (CT.current_uncertainty_observer, [103, 104])]:
        observer = initialize_array(DT.input, component, 2)
        observer["id"] = ids
        observer["observed_object"] = 7
        observer["status"] = 1
        observer["observed_terminal_type"] = [0, 1]
        data[component] = observer
    return data


@pytest.mark.parametrize("method", [CalculationMethod.iterative_linear, CalculationMethod.newton_raphson])
def test_public_branch_observers_match_independent_physical_jacobian(method):
    model = PowerGridModel(two_bus_observer_input())
    result = model.calculate_state_estimation(calculation_method=method)
    voltage = np.array([100.0 * np.exp(0.2j), 90.0 * np.exp(0.1j)])
    if method == CalculationMethod.iterative_linear:
        x = np.array([voltage[0].real, voltage[0].imag, voltage[1].real, voltage[1].imag])
    else:
        x = np.array([0.2, 100.0, 0.1, 90.0])

    def physical_outputs(coordinates, side):
        u = (
            coordinates[::2] + 1j * coordinates[1::2]
            if method == CalculationMethod.iterative_linear
            else coordinates[1::2] * np.exp(1j * coordinates[::2])
        )
        current = (u[side] - u[1 - side]) / (np.sqrt(3.0) * (1.0 + 2.0j))
        power = np.sqrt(3.0) * u[side] * current.conjugate()
        return np.array([power.real, power.imag, abs(current), np.angle(current)])

    # ILSE has proper rectangular deviations; NRSE reuses prepared voltage variance
    # for the real angle and per-unit magnitude rows, as the ordinary estimator does.
    variance = (
        np.repeat(np.array([2.0, 3.0]) ** 2, 2) / 2.0
        if method == CalculationMethod.iterative_linear
        else np.array([0.02, 2.0, 0.03, 3.0]) ** 2
    )
    for side in range(2):
        jacobian = np.empty((4, 4))
        for coordinate in range(4):
            step = np.zeros(4)
            delta = 1e-6 if method == CalculationMethod.newton_raphson and coordinate % 2 == 0 else 1e-4
            step[coordinate] = delta
            jacobian[:, coordinate] = (physical_outputs(x + step, side) - physical_outputs(x - step, side)) / (
                2 * delta
            )
        expected = np.sqrt((jacobian**2) @ variance)
        power = result[CT.power_uncertainty_observer][side]
        current = result[CT.current_uncertainty_observer][side]
        actual = [power["p_sigma"], power["q_sigma"], current["i_sigma"], current["i_angle_sigma"]]
        np.testing.assert_allclose(actual, expected, rtol=2e-8)
