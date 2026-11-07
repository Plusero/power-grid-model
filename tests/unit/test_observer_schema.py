# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from power_grid_model import CalculationType, ComponentType, DatasetType, PowerGridModel, initialize_array
from power_grid_model._core.data_handling import create_output_data
from power_grid_model.enum import ObservedTerminalType


@pytest.mark.parametrize(
    ("component", "sigmas"),
    [
        (ComponentType.voltage_uncertainty_observer, ("u_sigma", "u_angle_sigma")),
        (ComponentType.power_uncertainty_observer, ("p_sigma", "q_sigma")),
        (ComponentType.current_uncertainty_observer, ("i_sigma", "i_angle_sigma")),
    ],
)
@pytest.mark.parametrize("dataset_type", [DatasetType.sym_output, DatasetType.asym_output])
def test_observer_schema(component, sigmas, dataset_type):
    input_data = initialize_array(DatasetType.input, component, 1)
    expected = ("id", "observed_object", "status")
    if component != ComponentType.voltage_uncertainty_observer:
        expected += ("observed_terminal_type",)
    assert input_data.dtype.names == expected
    assert input_data["observed_object"][0] == np.iinfo(np.int32).min
    assert input_data["status"][0] == np.iinfo(np.int8).min
    update_data = initialize_array(DatasetType.update, component, 1)
    assert update_data.dtype.names == ("id", "status")
    output_data = initialize_array(dataset_type, component, 1)
    assert output_data.dtype.names == ("id", "energized", *sigmas)
    for sigma in sigmas:
        expected = np.dtype("float64") if dataset_type == DatasetType.sym_output else np.dtype(("float64", (3,)))
        assert output_data.dtype[sigma] == expected
        assert np.isnan(output_data[sigma][0]).all()


def test_observed_terminal_type_codes():
    assert {member.name: member.value for member in ObservedTerminalType} == {
        "branch_from": 0,
        "branch_to": 1,
        "branch3_1": 6,
        "branch3_2": 7,
        "branch3_3": 8,
        "node_injection": 9,
    }
    for code in (-128, -1, 2, 3, 4, 5, 10):
        with pytest.raises(ValueError, match="is not a valid ObservedTerminalType"):
            ObservedTerminalType(code)


@pytest.mark.parametrize("empty", [[], set()])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_empty_observer_selection_omits_buffers(empty, batch_size):
    component = ComponentType.current_uncertainty_observer
    result = create_output_data(
        {component: empty, ComponentType.node: ["id"]},
        DatasetType.sym_output,
        {component: 2, ComponentType.node: 1},
        batch_size > 1,
        batch_size,
    )
    assert component not in result
    assert set(result) == {ComponentType.node}


@pytest.mark.parametrize("calculation_type", [*CalculationType])
@pytest.mark.parametrize("symmetric", [True, False])
def test_observer_default_and_explicit_output_selection(calculation_type, symmetric):
    model = PowerGridModel.__new__(PowerGridModel)
    model._all_component_count = {ComponentType.node: 1, ComponentType.current_uncertainty_observer: 2}  # noqa: SLF001
    defaults = model._construct_output(  # noqa: SLF001
        None, calculation_type, symmetric=symmetric, is_batch=False, batch_size=1
    )
    assert (ComponentType.current_uncertainty_observer in defaults) == (
        calculation_type == CalculationType.state_estimation
    )
    explicit = model._construct_output(  # noqa: SLF001
        {ComponentType.current_uncertainty_observer: ["id"]},
        calculation_type,
        symmetric=symmetric,
        is_batch=False,
        batch_size=1,
    )
    assert set(explicit) == {ComponentType.current_uncertainty_observer}
    assert set(explicit[ComponentType.current_uncertainty_observer]) == {"id"}
