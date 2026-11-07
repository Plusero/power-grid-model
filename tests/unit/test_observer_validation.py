# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from power_grid_model import CalculationType, ComponentType as CT, DatasetType as DT, initialize_array
from power_grid_model.enum import ObservedTerminalType as Terminal
from power_grid_model.validation import validate_batch_data, validate_input_data
from power_grid_model.validation._validation import validate_uncertainty_observer
from power_grid_model.validation.errors import InvalidEnumValueError, InvalidIdError, MissingValueError, NotBooleanError

OBSERVERS = [CT.voltage_uncertainty_observer, CT.power_uncertainty_observer, CT.current_uncertainty_observer]


def observer_data(component, status=1, target=1, terminal=Terminal.branch_from):
    data = initialize_array(DT.input, component, 1)
    data["id"] = 100
    data["observed_object"] = target
    data["status"] = status
    if component != CT.voltage_uncertainty_observer:
        data["observed_terminal_type"] = terminal
    return data


def node_data():
    nodes = initialize_array(DT.input, CT.node, 1)
    nodes["id"] = 1
    nodes["u_rated"] = 100
    return nodes


@pytest.mark.parametrize("calculation_type", [None, *CalculationType])
@pytest.mark.parametrize("component", OBSERVERS)
def test_observer_required_fields_for_every_calculation(component, calculation_type):
    data = initialize_array(DT.input, component, 1)
    errors = validate_input_data({component: data}, calculation_type=calculation_type)
    fields = {str(error.field) for error in errors if isinstance(error, MissingValueError)}
    assert {"id", "observed_object", "status"} <= fields
    if component != CT.voltage_uncertainty_observer:
        assert "observed_terminal_type" in fields


@pytest.mark.parametrize("calculation_type", [None, *CalculationType])
@pytest.mark.parametrize("status", [0, 1, 2, -1])
def test_observer_status_and_targets_for_every_calculation(calculation_type, status):
    data = {
        CT.node: node_data(),
        CT.voltage_uncertainty_observer: observer_data(CT.voltage_uncertainty_observer, status),
    }
    errors = validate_input_data(data, calculation_type=calculation_type)
    if status in (0, 1):
        assert errors is None
    else:
        assert any(isinstance(error, NotBooleanError) for error in errors)
    data[CT.voltage_uncertainty_observer]["observed_object"] = 999
    errors = validate_input_data(data, calculation_type=calculation_type)
    assert any(isinstance(error, InvalidIdError) for error in errors)


@pytest.mark.parametrize("component", OBSERVERS)
@pytest.mark.parametrize("target_type", [*CT])
@pytest.mark.parametrize("terminal", [*Terminal])
def test_observer_permitted_target_terminal_pairs(component, target_type, terminal):
    # Test observer predicates in isolation: unrelated electrical target fields need not be specified here.
    target = initialize_array(DT.input, target_type, 1)
    target["id"] = 1
    data = {target_type: target, component: observer_data(component, status=0, terminal=terminal)}
    finite_branch = target_type in (CT.line, CT.asym_line, CT.generic_branch, CT.transformer)
    if component == CT.voltage_uncertainty_observer:
        valid = target_type == CT.node
    else:
        valid = (
            (finite_branch and terminal in (Terminal.branch_from, Terminal.branch_to))
            or (
                target_type == CT.three_winding_transformer
                and terminal in (Terminal.branch3_1, Terminal.branch3_2, Terminal.branch3_3)
            )
            or (
                component == CT.power_uncertainty_observer
                and target_type == CT.node
                and terminal == Terminal.node_injection
            )
        )
    errors = validate_uncertainty_observer(data, component)
    assert bool(errors) != valid
    if not valid:
        assert any(isinstance(error, InvalidIdError) for error in errors)


@pytest.mark.parametrize("component", OBSERVERS[1:])
@pytest.mark.parametrize("terminal", [-1, 2, 3, 4, 5, 10])
def test_observer_invalid_terminal_codes(component, terminal):
    data = {CT.node: node_data(), component: observer_data(component, status=0, terminal=terminal)}
    errors = validate_uncertainty_observer(data, component)
    assert any(isinstance(error, InvalidEnumValueError) for error in errors)


@pytest.mark.parametrize("sparse", [False, True])
def test_observer_status_batch_positional_updates(sparse):
    component = CT.voltage_uncertainty_observer
    data = {CT.node: node_data(), component: observer_data(component)}
    update = initialize_array(DT.update, component, (2, 1))
    update["status"][:, 0] = [0, 1]
    batch = {"indptr": np.array([0, 1, 2], dtype=np.int64), "data": update.ravel()} if sparse else update
    assert validate_batch_data(data, {component: batch}) is None
    assert data[component]["status"][0] == 1
    update["status"][0, 0] = 2
    errors = validate_batch_data(data, {component: batch})
    assert set(errors) == {0}
    assert any(isinstance(error, NotBooleanError) for error in errors[0])
