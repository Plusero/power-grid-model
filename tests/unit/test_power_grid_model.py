# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from copy import copy, deepcopy

import numpy as np
import pytest

from power_grid_model import (
    AngleMeasurementType,
    AttributeType as AT,
    BranchSide,
    ComponentAttributeFilterOptions,
    ComponentType as CT,
    DatasetType as DT,
    LoadGenType,
    MeasuredTerminalType,
    PowerGridModel,
    initialize_array,
)
from power_grid_model._core.utils import compatibility_convert_row_columnar_dataset
from power_grid_model.data_types import BatchDataset
from power_grid_model.errors import (
    InvalidCalculationMethod,
    IterationDiverge,
    PowerGridBatchError,
    PowerGridError,
)
from power_grid_model.utils import get_dataset_scenario
from power_grid_model.validation import assert_valid_input_data

from .utils import DATA_PATH, compare_result, import_case_data

"""
Testing network

source_1(1.0 p.u., 100.0 V) --internal_impedance(j10.0 ohm, sk=1000.0 VA, rx_ratio=0.0)--
-- node_0 (100.0 V) --load_2(const_i, -j5.0A, 0.0 W, 500.0 var)

u0 = 100.0 V - (j10.0 ohm * -j5.0 A) = 50.0 V

update_0:
    u_ref = 0.5 p.u. (50.0 V)
    q_specified = 100 var (-j1.0A)
u0 = 50.0 V - (j10.0 ohm * -j1.0 A) = 40.0 V

update_1:
    q_specified = 300 var (-j3.0A)
u0 = 100.0 V - (j10.0 ohm * -j3.0 A) = 70.0 V
"""


@pytest.fixture
def input_row():
    node = initialize_array(DT.input, CT.node, 1)
    node[AT.id] = 0
    node[AT.u_rated] = 100.0

    source = initialize_array(DT.input, CT.source, 1)
    source[AT.id] = 1
    source[AT.node] = 0
    source[AT.status] = 1
    source[AT.u_ref] = 1.0
    source[AT.sk] = 1000.0
    source[AT.rx_ratio] = 0.0

    sym_load = initialize_array(DT.input, CT.sym_load, 1)
    sym_load[AT.id] = 2
    sym_load[AT.node] = 0
    sym_load[AT.status] = 1
    sym_load[AT.type] = 2
    sym_load[AT.p_specified] = 0.0
    sym_load[AT.q_specified] = 500.0

    return {
        CT.node: node,
        CT.source: source,
        CT.sym_load: sym_load,
    }


@pytest.fixture
def input_col(input_row):
    return compatibility_convert_row_columnar_dataset(input_row, ComponentAttributeFilterOptions.relevant, DT.input)


@pytest.fixture(params=["input_row", "input_col"])
def input(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def sym_output():
    node = initialize_array(DT.sym_output, CT.node, 1)
    node[AT.id] = 0
    node[AT.u] = 50.0
    node[AT.u_pu] = 0.5
    node[AT.u_angle] = 0.0

    return {CT.node: node}


@pytest.fixture
def update_batch_row():
    source = initialize_array(DT.update, CT.source, 1)
    source[AT.id] = 1
    source[AT.u_ref] = 0.5

    sym_load = initialize_array(DT.update, CT.sym_load, 2)
    sym_load[AT.id] = [2, 2]
    sym_load[AT.q_specified] = [100.0, 300.0]

    return {
        CT.source: {
            "data": source,
            "indptr": np.array([0, 1, 1]),
        },
        CT.sym_load: {
            "data": sym_load,
            "indptr": np.array([0, 1, 2]),
        },
    }


@pytest.fixture
def update_batch_col(update_batch_row):
    return compatibility_convert_row_columnar_dataset(
        update_batch_row, ComponentAttributeFilterOptions.relevant, DT.update
    )


@pytest.fixture(params=["update_batch_row", "update_batch_col"])
def update_batch(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def sym_output_batch():
    node = initialize_array(DT.sym_output, CT.node, (2, 1))
    node[AT.id] = [[0], [0]]
    node[AT.u] = [[40.0], [70.0]]
    node[AT.u_pu] = [[0.4], [0.7]]
    node[AT.u_angle] = [[0.0], [0.0]]

    return {
        CT.node: node,
    }


@pytest.fixture
def model(input):
    return PowerGridModel(input)


@pytest.fixture
def empty_model():
    return PowerGridModel({})


def test_simple_power_flow(model: PowerGridModel, sym_output):
    result = model.calculate_power_flow()
    compare_result(result, sym_output, rtol=0.0, atol=1e-8)


def test_simple_permanent_update(model: PowerGridModel, update_batch, sym_output_batch):
    model.update(update_data=get_dataset_scenario(update_batch, 0))  # single permanent model update
    result = model.calculate_power_flow()
    expected_result = get_dataset_scenario(sym_output_batch, 0)
    compare_result(result, expected_result, rtol=0.0, atol=1e-8)


def test_update_error(model: PowerGridModel):
    load_update = initialize_array(DT.update, CT.sym_load, 1)
    load_update[AT.id] = 5
    update_data = {CT.sym_load: load_update}
    with pytest.raises(PowerGridError, match="The id cannot be found:"):
        model.update(update_data=update_data)
    update_data_col = compatibility_convert_row_columnar_dataset(
        update_data, ComponentAttributeFilterOptions.relevant, DT.update
    )
    with pytest.raises(PowerGridError, match="The id cannot be found:"):
        model.update(update_data=update_data_col)


def test_copy_model(model: PowerGridModel, sym_output):
    model_2 = copy(model)
    result = model_2.calculate_power_flow()
    compare_result(result, sym_output, rtol=0.0, atol=1e-8)


def test_deepcopy_model(model: PowerGridModel, empty_model: PowerGridModel, sym_output, update_batch, sym_output_batch):
    # list containing different models twice
    model_list = [model, empty_model, model, empty_model]

    new_model_list = deepcopy(model_list)

    # check if identities are as expected
    assert id(new_model_list[0]) != id(model_list[0])
    assert id(new_model_list[1]) != id(model_list[1])
    assert id(new_model_list[0]) != id(new_model_list[1])
    assert id(new_model_list[0]) == id(new_model_list[2])
    assert id(new_model_list[1]) == id(new_model_list[3])

    # check if the deepcopied objects are really independent from the original ones
    # by modifying the copies and seeing if the original one is impacted by this change
    new_model_list[0].update(update_data=get_dataset_scenario(update_batch, 0))

    new_expected_result = get_dataset_scenario(sym_output_batch, 0)
    new_result_0 = new_model_list[0].calculate_power_flow()
    compare_result(new_result_0, new_expected_result, rtol=0.0, atol=1e-8)
    # at index 0 and 2 should be the same objects, check if changing the object at index 0
    # and obtaining a power flow result is ident to the result at index 2
    new_result_2 = new_model_list[2].calculate_power_flow()
    compare_result(new_result_2, new_expected_result, rtol=0.0, atol=1e-8)

    result = model.calculate_power_flow()
    compare_result(result, sym_output, rtol=0.0, atol=1e-8)


def test_repr_and_str(model: PowerGridModel, empty_model: PowerGridModel):
    repr_empty_model_expected = "PowerGridModel (0 components)\n"
    assert repr_empty_model_expected == repr(empty_model)
    assert repr_empty_model_expected == str(empty_model)

    repr_model_expected = "PowerGridModel (3 components)\n  - node: 1\n  - source: 1\n  - sym_load: 1\n"
    assert repr_model_expected == repr(model)
    assert repr_model_expected == str(model)


def test_get_indexer(model: PowerGridModel):
    ids = np.array([2, 2])
    expected_indexer = np.array([0, 0])
    indexer = model.get_indexer(CT.sym_load, ids)
    np.testing.assert_allclose(expected_indexer, indexer)


def test_batch_power_flow(model: PowerGridModel, update_batch: BatchDataset, sym_output_batch):
    result = model.calculate_power_flow(update_data=update_batch)
    compare_result(result, sym_output_batch, rtol=0.0, atol=1e-8)


def test_construction_error(input):
    input[CT.sym_load][AT.id][0] = 0
    with pytest.raises(PowerGridError, match="Conflicting id detected:"):
        PowerGridModel(input)


def test_single_calculation_error(model: PowerGridModel):
    with pytest.raises(IterationDiverge, match="Iteration failed to converge after"):
        model.calculate_power_flow(max_iterations=1, error_tolerance=1e-100)
    with pytest.raises(InvalidCalculationMethod, match="The calculation method is invalid for this calculation!"):
        model.calculate_state_estimation(calculation_method="iterative_current")

    for calculation_method in ("linear", "newton_raphson", "iterative_current", "linear_current", "iterative_linear"):
        with pytest.raises(InvalidCalculationMethod):
            model.calculate_short_circuit(calculation_method=calculation_method)


@pytest.mark.parametrize(
    ("case_name", "calculation_method", "expected_node_count", "expected_line_count"),
    [
        ("single-line-load-il", "iterative_linear", 2, 1),
        ("1os2msr", "newton_raphson", 3, 2),
    ],
)
def test_state_estimation_uncertainty(
    case_name: str, calculation_method: str, expected_node_count: int, expected_line_count: int
):
    case_data = import_case_data(
        DATA_PATH / "state_estimation" / case_name, calculation_type="state_estimation", sym=True
    )
    model = PowerGridModel(case_data[DT.input], system_frequency=50.0)

    result_without_uncertainty = model.calculate_state_estimation(calculation_method=calculation_method)
    result = model.calculate_state_estimation(
        calculation_method=calculation_method,
        calculate_uncertainty=True,
    )

    assert result[CT.node].shape == (expected_node_count,)
    assert result[CT.line].shape == (expected_line_count,)

    sigma_to_value = {
        CT.node: {
            AT.u_pu_sigma: AT.u_pu,
            AT.u_sigma: AT.u,
            AT.u_angle_sigma: AT.u_angle,
            AT.p_sigma: AT.p,
            AT.q_sigma: AT.q,
        },
        CT.line: {
            AT.p_from_sigma: AT.p_from,
            AT.q_from_sigma: AT.q_from,
            AT.i_from_sigma: AT.i_from,
            AT.p_to_sigma: AT.p_to,
            AT.q_to_sigma: AT.q_to,
            AT.i_to_sigma: AT.i_to,
        },
    }

    for component, fields in sigma_to_value.items():
        result_fields = result[component].dtype.names
        assert result_fields is not None
        for sigma_field, value_field in fields.items():
            assert sigma_field in result_fields
            assert result[component][sigma_field].shape == result[component][value_field].shape
            assert np.all(np.isnan(result_without_uncertainty[component][sigma_field]))
            assert np.all(np.isfinite(result[component][sigma_field]))
            assert np.all(result[component][sigma_field] >= 0.0)


@pytest.mark.parametrize(
    ("case_name", "symmetric", "calculation_method", "expected_u_sigma"),
    [
        ("single-node-source-sym-voltage-sensor", True, "iterative_linear", 100.0 / np.sqrt(2.0)),
        ("single-node-source-asym-voltage-sensor", False, "iterative_linear", 100.0 / np.sqrt(2.0)),
        ("single-node-source-sym-voltage-sensor", True, "newton_raphson", 100.0),
        ("single-node-source-asym-voltage-sensor", False, "newton_raphson", 100.0),
    ],
)
def test_state_estimation_voltage_uncertainty_analytical(
    case_name: str, symmetric: bool, calculation_method: str, expected_u_sigma: float
):
    case_data = import_case_data(
        DATA_PATH / "state_estimation" / case_name,
        calculation_type="state_estimation",
        sym=symmetric,
    )
    result = PowerGridModel(case_data[DT.input], system_frequency=50.0).calculate_state_estimation(
        symmetric=symmetric,
        calculation_method=calculation_method,
        calculate_uncertainty=True,
    )

    node = result[CT.node]
    np.testing.assert_allclose(node[AT.u_sigma], expected_u_sigma)
    np.testing.assert_allclose(node[AT.u_pu_sigma], expected_u_sigma * node[AT.u_pu] / node[AT.u])
    expected_angle_sigma = (
        node[AT.u_pu_sigma] if calculation_method == "newton_raphson" else expected_u_sigma / node[AT.u]
    )
    np.testing.assert_allclose(node[AT.u_angle_sigma], expected_angle_sigma)


def test_newton_raphson_state_estimation_uncertainty_matches_numerical_sensitivity():
    case_data = import_case_data(
        DATA_PATH / "state_estimation" / "1os2msr",
        calculation_type="state_estimation",
        sym=True,
    )[DT.input]

    def calculate(input_data: dict, *, calculate_uncertainty: bool = False):
        return PowerGridModel(input_data, system_frequency=50.0).calculate_state_estimation(
            calculation_method="newton_raphson",
            calculate_uncertainty=calculate_uncertainty,
            error_tolerance=1e-12,
            max_iterations=50,
        )

    result = calculate(case_data, calculate_uncertainty=True)
    output_fields = {
        CT.node: [
            (AT.u, AT.u_sigma),
            (AT.u_angle, AT.u_angle_sigma),
            (AT.p, AT.p_sigma),
            (AT.q, AT.q_sigma),
        ],
        CT.line: [
            (AT.i_from, AT.i_from_sigma),
            (AT.p_from, AT.p_from_sigma),
            (AT.q_from, AT.q_from_sigma),
            (AT.i_to, AT.i_to_sigma),
            (AT.p_to, AT.p_to_sigma),
            (AT.q_to, AT.q_to_sigma),
        ],
    }
    numerical_variances = {
        (component, value_field): np.zeros_like(result[component][value_field], dtype=float)
        for component, fields in output_fields.items()
        for value_field, _ in fields
    }

    rated_voltage = {node[AT.id]: node[AT.u_rated] for node in case_data[CT.node]}
    perturbations = []
    for idx, sensor in enumerate(case_data[CT.sym_voltage_sensor]):
        magnitude_sigma = sensor[AT.u_sigma]
        perturbations.append((CT.sym_voltage_sensor, idx, AT.u_measured, magnitude_sigma))
        if np.isfinite(sensor[AT.u_angle_measured]):
            angle_sigma = magnitude_sigma / rated_voltage[sensor[AT.measured_object]]
            perturbations.append((CT.sym_voltage_sensor, idx, AT.u_angle_measured, angle_sigma))

    # A symmetric power_sigma describes the complex error; its independent P and Q components each carry half
    # the variance. Infinite-sigma channels are excluded from the estimator and from this sensitivity sum.
    for idx, sensor in enumerate(case_data[CT.sym_power_sensor]):
        component_sigma = sensor[AT.power_sigma] / np.sqrt(2.0)
        if np.isfinite(component_sigma):
            perturbations.append((CT.sym_power_sensor, idx, AT.p_measured, component_sigma))
            perturbations.append((CT.sym_power_sensor, idx, AT.q_measured, component_sigma))

    for sensor_type, sensor_idx, measurement_field, measurement_sigma in perturbations:
        step = 1e-3 * measurement_sigma
        plus_data = deepcopy(case_data)
        minus_data = deepcopy(case_data)
        plus_data[sensor_type][measurement_field][sensor_idx] += step
        minus_data[sensor_type][measurement_field][sensor_idx] -= step
        plus_result = calculate(plus_data)
        minus_result = calculate(minus_data)

        for component, fields in output_fields.items():
            for value_field, _ in fields:
                output_delta = plus_result[component][value_field] - minus_result[component][value_field]
                if value_field == AT.u_angle:
                    output_delta = np.angle(np.exp(1j * output_delta))
                sensitivity = output_delta / (2.0 * step)
                numerical_variances[component, value_field] += np.square(sensitivity * measurement_sigma)

    for component, fields in output_fields.items():
        for value_field, sigma_field in fields:
            numerical_sigma = np.sqrt(numerical_variances[component, value_field])
            np.testing.assert_allclose(result[component][sigma_field], numerical_sigma, rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("has_voltage_angles", [True, False])
def test_asymmetric_newton_raphson_state_estimation_uncertainty_matches_local_current_sensitivity(
    has_voltage_angles: bool,
):
    original_data = import_case_data(
        DATA_PATH / "state_estimation" / "dummy-test-line-into-itself",
        calculation_type="state_estimation",
        sym=False,
    )[DT.input]

    def calculate(input_data: dict, *, calculate_uncertainty: bool = False):
        return PowerGridModel(input_data, system_frequency=50.0).calculate_state_estimation(
            symmetric=False,
            calculation_method="newton_raphson",
            calculate_uncertainty=calculate_uncertainty,
            error_tolerance=1e-12,
            max_iterations=50,
        )

    # Turn an existing consistent state into a direct-gain fixture with one asymmetric local-frame current sensor.
    # The fixture is exactly determined, phase coupled, and has unequal phase currents, so numerical sensitivity catches
    # both inner phase-block transposition errors and applying the local-voltage normalization on the wrong side.
    target_line_id = 3
    target_result = calculate(original_data)
    target_line = target_result[CT.line][target_result[CT.line][AT.id] == target_line_id][0]
    input_data = deepcopy(original_data)
    input_data.pop(CT.sym_power_sensor)
    input_data.pop(CT.asym_power_sensor)
    current_sensor = initialize_array(DT.input, CT.asym_current_sensor, 1)
    current_sensor[AT.id] = 1001
    current_sensor[AT.measured_object] = target_line[AT.id]
    current_sensor[AT.measured_terminal_type] = MeasuredTerminalType.branch_from
    current_sensor[AT.angle_measurement_type] = AngleMeasurementType.local_angle
    current_sensor[AT.i_sigma] = 0.1
    current_sensor[AT.i_angle_sigma] = 2e-4
    current_sensor[AT.i_measured] = target_line[AT.i_from]
    current_sensor[AT.i_angle_measured] = np.arctan2(target_line[AT.q_from], target_line[AT.p_from])
    input_data[CT.asym_current_sensor] = current_sensor

    if not has_voltage_angles:
        # Put the magnitude sensor on the non-slack bus so the virtual-angle anchor differs from the reported slack
        # phase-A reference. This exercises both the low-rank deterministic-row correction and the Gamma projection.
        voltage_node_id = 2
        target_voltage_node = target_result[CT.node][target_result[CT.node][AT.id] == voltage_node_id][0]
        input_data[CT.asym_voltage_sensor][AT.measured_object] = voltage_node_id
        input_data[CT.asym_voltage_sensor][AT.u_measured] = target_voltage_node[AT.u]
        input_data[CT.asym_voltage_sensor][AT.u_angle_measured] = np.nan

    # Recalibrate the current measurement after the no-angle virtual rows have selected their deterministic phase
    # reference. Local current is invariant to common rotation, while asymmetric virtual rows also fix relative phase.
    calibrated_result = calculate(input_data)
    calibrated_line = calibrated_result[CT.line][calibrated_result[CT.line][AT.id] == target_line_id][0]
    input_data[CT.asym_current_sensor][AT.i_measured] = calibrated_line[AT.i_from]
    input_data[CT.asym_current_sensor][AT.i_angle_measured] = np.arctan2(
        calibrated_line[AT.q_from], calibrated_line[AT.p_from]
    )

    result = calculate(input_data, calculate_uncertainty=True)
    output_fields = {
        CT.node: [
            (AT.u, AT.u_sigma),
            (AT.u_angle, AT.u_angle_sigma),
            (AT.p, AT.p_sigma),
            (AT.q, AT.q_sigma),
        ],
        CT.line: [
            (AT.i_from, AT.i_from_sigma),
            (AT.p_from, AT.p_from_sigma),
            (AT.q_from, AT.q_from_sigma),
            (AT.i_to, AT.i_to_sigma),
            (AT.p_to, AT.p_to_sigma),
            (AT.q_to, AT.q_to_sigma),
        ],
    }
    numerical_variances = {
        (component, value_field): np.zeros_like(result[component][value_field], dtype=float)
        for component, fields in output_fields.items()
        for value_field, _ in fields
    }

    def accumulate_sensitivity(plus_data: dict, minus_data: dict, measurement_sigma: float, step: float):
        plus_result = calculate(plus_data)
        minus_result = calculate(minus_data)
        for component, fields in output_fields.items():
            for value_field, _ in fields:
                output_delta = plus_result[component][value_field] - minus_result[component][value_field]
                if value_field == AT.u_angle:
                    output_delta = np.angle(np.exp(1j * output_delta))
                sensitivity = output_delta / (2.0 * step)
                numerical_variances[component, value_field] += np.square(sensitivity * measurement_sigma)

    voltage_sensor = input_data[CT.asym_voltage_sensor][0]
    rated_voltage = {node[AT.id]: node[AT.u_rated] for node in input_data[CT.node]}[voltage_sensor[AT.measured_object]]
    voltage_sigmas = {AT.u_measured: voltage_sensor[AT.u_sigma]}
    if has_voltage_angles:
        voltage_sigmas[AT.u_angle_measured] = voltage_sensor[AT.u_sigma] / (rated_voltage / np.sqrt(3.0))
    for phase in range(3):
        for measurement_field, measurement_sigma in voltage_sigmas.items():
            step = 1e-3 * measurement_sigma
            plus_data = deepcopy(input_data)
            minus_data = deepcopy(input_data)
            plus_data[CT.asym_voltage_sensor][measurement_field][0, phase] += step
            minus_data[CT.asym_voltage_sensor][measurement_field][0, phase] -= step
            accumulate_sensitivity(plus_data, minus_data, measurement_sigma, step)

    # Current sensors are processed into independent local Cartesian channels. Perturb those channels through their
    # public polar representation and use the same second-order marginal variances as the input conversion.
    current_sensor = input_data[CT.asym_current_sensor][0]
    magnitude_variance = current_sensor[AT.i_sigma] ** 2
    angle_variance = current_sensor[AT.i_angle_sigma] ** 2
    for phase in range(3):
        magnitude = current_sensor[AT.i_measured][phase]
        angle = current_sensor[AT.i_angle_measured][phase]
        local_current = magnitude * np.exp(1j * angle)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        real_variance = (
            magnitude_variance * cos_angle**2
            + magnitude**2 * angle_variance * sin_angle**2
            + 0.5 * magnitude**2 * angle_variance**2 * cos_angle**2
            + magnitude_variance * angle_variance * sin_angle**2
        )
        imaginary_variance = (
            magnitude_variance * sin_angle**2
            + magnitude**2 * angle_variance * cos_angle**2
            + 0.5 * magnitude**2 * angle_variance**2 * sin_angle**2
            + magnitude_variance * angle_variance * cos_angle**2
        )

        for measurement_sigma, direction in (
            (np.sqrt(real_variance), 1.0),
            (np.sqrt(imaginary_variance), 1.0j),
        ):
            step = 1e-3 * measurement_sigma
            plus_data = deepcopy(input_data)
            minus_data = deepcopy(input_data)
            for perturbed_data, signed_step in ((plus_data, step), (minus_data, -step)):
                perturbed_current = local_current + direction * signed_step
                perturbed_data[CT.asym_current_sensor][AT.i_measured][0, phase] = np.abs(perturbed_current)
                perturbed_data[CT.asym_current_sensor][AT.i_angle_measured][0, phase] = np.angle(perturbed_current)
            accumulate_sensitivity(plus_data, minus_data, measurement_sigma, step)

    for component, fields in output_fields.items():
        for value_field, sigma_field in fields:
            numerical_sigma = np.sqrt(numerical_variances[component, value_field])
            finite = np.isfinite(result[component][sigma_field])
            assert np.any(finite)
            np.testing.assert_allclose(
                result[component][sigma_field][finite], numerical_sigma[finite], rtol=1e-5, atol=1e-12
            )


@pytest.mark.parametrize("calculation_method", ["iterative_linear", "newton_raphson"])
def test_state_estimation_uncertainty_without_angle_reference(calculation_method: str):
    case_data = import_case_data(
        DATA_PATH / "state_estimation" / "single-node-source-asym-voltage-sensor-no-angle",
        calculation_type="state_estimation",
        sym=False,
    )
    result = PowerGridModel(case_data[DT.input], system_frequency=50.0).calculate_state_estimation(
        symmetric=False,
        calculation_method=calculation_method,
        calculate_uncertainty=True,
    )

    node = result[CT.node][0]
    angle_sigma = node[AT.u_angle_sigma]
    assert angle_sigma[0] == 0.0
    if calculation_method == "newton_raphson":
        np.testing.assert_array_equal(angle_sigma, 0.0)
        return
    unreferenced_angle_sigma = node[AT.u_sigma] / node[AT.u]
    expected_referenced_sigma = np.sqrt(unreferenced_angle_sigma[1:] ** 2 + unreferenced_angle_sigma[0] ** 2)
    np.testing.assert_allclose(angle_sigma[1:], expected_referenced_sigma)


@pytest.mark.parametrize("calculation_method", ["iterative_linear", "newton_raphson"])
def test_state_estimation_uncertainty_for_ideal_link_supernode(calculation_method: str):
    case_data = import_case_data(
        DATA_PATH / "state_estimation" / "node-injection-with-injection-sensor-sym-sensors",
        calculation_type="state_estimation",
        sym=True,
    )
    result = PowerGridModel(case_data[DT.input], system_frequency=50.0).calculate_state_estimation(
        calculation_method=calculation_method,
        calculate_uncertainty=True,
    )

    assert np.all(np.isfinite(result[CT.node][AT.u_sigma]))
    assert np.all(np.isnan(result[CT.node][AT.p_sigma]))
    assert np.all(np.isnan(result[CT.node][AT.q_sigma]))

    link_sigma_fields = (
        AT.p_from_sigma,
        AT.q_from_sigma,
        AT.i_from_sigma,
        AT.p_to_sigma,
        AT.q_to_sigma,
        AT.i_to_sigma,
    )
    for field in link_sigma_fields:
        assert np.all(np.isnan(result[CT.link][field]))


@pytest.mark.parametrize("calculation_method", ["iterative_linear", "newton_raphson"])
def test_state_estimation_uncertainty_for_three_winding_transformer(calculation_method: str):
    case_data = import_case_data(
        DATA_PATH / "state_estimation" / "three_winding_transformer",
        calculation_type="state_estimation",
        sym=True,
    )
    model = PowerGridModel(case_data[DT.input], system_frequency=50.0)
    result_without_uncertainty = model.calculate_state_estimation(calculation_method=calculation_method)
    result = model.calculate_state_estimation(
        calculation_method=calculation_method,
        calculate_uncertainty=True,
    )

    component = CT.three_winding_transformer
    for field in (
        AT.p_1_sigma,
        AT.q_1_sigma,
        AT.i_1_sigma,
        AT.p_2_sigma,
        AT.q_2_sigma,
        AT.i_2_sigma,
        AT.p_3_sigma,
        AT.q_3_sigma,
        AT.i_3_sigma,
    ):
        assert np.all(np.isnan(result_without_uncertainty[component][field]))
        assert np.all(np.isfinite(result[component][field]))
        assert np.all(result[component][field] >= 0.0)


def test_batch_calculation_error(model: PowerGridModel, update_batch):
    # wrong id
    update_batch[CT.sym_load]["data"][AT.id][1] = 5
    # with error
    with pytest.raises(PowerGridBatchError) as e:
        model.calculate_power_flow(update_data=update_batch)
    error = e.value
    np.testing.assert_allclose(error.failed_scenarios, [1])
    np.testing.assert_allclose(error.succeeded_scenarios, [0])
    assert "The id cannot be found:" in error.error_messages[0]


def test_batch_calculation_error_continue(model: PowerGridModel, update_batch, sym_output_batch):
    # wrong id
    update_batch[CT.sym_load]["data"][AT.id][1] = 5
    result = model.calculate_power_flow(update_data=update_batch, continue_on_batch_error=True)
    # assert error
    error = model.batch_error
    assert error is not None
    np.testing.assert_allclose(error.failed_scenarios, [1])
    np.testing.assert_allclose(error.succeeded_scenarios, [0])
    assert "The id cannot be found:" in error.error_messages[0]
    # assert value result for scenario 0
    result = {CT.node: result[CT.node][error.succeeded_scenarios, :]}
    expected_result = {CT.node: sym_output_batch[CT.node][error.succeeded_scenarios, :]}
    compare_result(result, expected_result, rtol=0.0, atol=1e-8)
    # general error before the batch
    with pytest.raises(PowerGridError, match="The calculation method is invalid for this calculation!"):
        model.calculate_state_estimation(
            calculation_method="iterative_current",
            update_data={CT.source: initialize_array(DT.update, CT.source, shape=(5, 0))},
            continue_on_batch_error=True,
        )


def test_empty_input():
    node = initialize_array(DT.input, CT.node, 0)
    line = initialize_array(DT.input, CT.line, 0)
    sym_load = initialize_array(DT.input, CT.sym_load, 0)
    source = initialize_array(DT.input, CT.source, 0)

    input_data = {
        CT.node: node,
        CT.line: line,
        CT.sym_load: sym_load,
        CT.source: source,
    }

    assert_valid_input_data(input_data)
    model = PowerGridModel(input_data, system_frequency=50.0)

    result = model.calculate_power_flow()

    assert result == {}


@pytest.fixture
def input_sym_load_col(input_row):
    return compatibility_convert_row_columnar_dataset(
        input_row,
        {
            CT.node: None,
            CT.source: None,
            CT.sym_load: ComponentAttributeFilterOptions.relevant,
        },
        DT.input,
    )


@pytest.fixture(params=[pytest.param("input_row", id="input_row"), pytest.param("input_sym_load_col", id="input_col")])
def minimal_input(request):
    return request.getfixturevalue(request.param)


def update_sym_load_row():
    sym_load = initialize_array(DT.update, CT.sym_load, (2, 1))
    sym_load[AT.id] = [[2], [2]]
    sym_load[AT.q_specified] = [[100.0], [300.0]]
    return {CT.sym_load: sym_load}


def update_sym_load_row_optional_id():
    sym_load = initialize_array(DT.update, CT.sym_load, (2, 1))
    sym_load[AT.q_specified] = [[100.0], [300.0]]
    return {CT.sym_load: sym_load}


def update_sym_load_row_invalid_id():
    sym_load = initialize_array(DT.update, CT.sym_load, (2, 1))
    sym_load[AT.id] = [[2], [5]]
    sym_load[AT.q_specified] = [[100.0], [300.0]]
    return {CT.sym_load: sym_load}


def update_sym_load_col(update_sym_load_row):
    return compatibility_convert_row_columnar_dataset(
        update_sym_load_row, ComponentAttributeFilterOptions.relevant, DT.update
    )


def update_sym_load_sparse(update_data):
    return {
        CT.sym_load: {
            "data": update_data[CT.sym_load].reshape(-1),
            "indptr": np.array([0, 1, 2]),
        },
    }


@pytest.mark.parametrize(
    "minimal_update",
    [
        pytest.param(update_sym_load_row(), id="update_dense_row"),
        pytest.param(update_sym_load_col(update_sym_load_row()), id="update_dense_col"),
        pytest.param(update_sym_load_sparse(update_sym_load_row()), id="update_sparse_row"),
        pytest.param(update_sym_load_col(update_sym_load_sparse(update_sym_load_row())), id="update_sparse_col"),
    ],
)
def test_update_ids_batch(minimal_update, minimal_input):
    output_data = PowerGridModel(minimal_input).calculate_power_flow(update_data=minimal_update)
    np.testing.assert_almost_equal(output_data[CT.node][AT.u], np.array([[90.0], [70.0]]))


@pytest.mark.parametrize(
    "minimal_update",
    [
        pytest.param(update_sym_load_row_optional_id(), id="update_dense_row"),
        pytest.param(update_sym_load_col(update_sym_load_row_optional_id()), id="update_dense_col"),
        pytest.param(update_sym_load_sparse(update_sym_load_row_optional_id()), id="update_sparse_row"),
        pytest.param(
            update_sym_load_col(update_sym_load_sparse(update_sym_load_row_optional_id())), id="update_sparse_col"
        ),
    ],
)
def test_update_id_optional(minimal_update, minimal_input):
    output_data = PowerGridModel(minimal_input).calculate_power_flow(update_data=minimal_update)
    np.testing.assert_almost_equal(output_data[CT.node][AT.u], np.array([[90.0], [70.0]]))


def test_update_id_mixed(minimal_input):
    update_sym_load_no_id = initialize_array(DT.update, CT.sym_load, (3, 1))
    update_sym_load_no_id[AT.p_specified] = [[30e6], [15e5], [0]]

    update_source_indptr = np.array([0, 1, 1, 2])
    update_source = initialize_array(DT.update, CT.source, 2)
    update_source[AT.id] = 1
    update_source[AT.status] = 0

    update_batch = {
        CT.sym_load: update_sym_load_no_id,
        CT.source: {"indptr": update_source_indptr, "data": update_source},
    }

    _output_data = PowerGridModel(minimal_input).calculate_power_flow(update_data=update_batch)


@pytest.mark.parametrize(
    "minimal_update",
    [
        update_sym_load_row_invalid_id(),
        update_sym_load_col(update_sym_load_row_invalid_id()),
        update_sym_load_sparse(update_sym_load_row_invalid_id()),
        update_sym_load_col(update_sym_load_sparse(update_sym_load_row_invalid_id())),
    ],
)
def test_update_id_error(minimal_update, minimal_input):
    with pytest.raises(PowerGridBatchError) as e:
        PowerGridModel(minimal_input).calculate_power_flow(update_data=minimal_update)
    assert e.value.failed_scenarios == [1]
    assert "The id cannot be found: 5" in e.value.error_messages[0]


@pytest.fixture
def input_data__irrelevant_components_test():
    node = initialize_array(DT.input, CT.node, 2)
    node[AT.id] = np.array([1, 2])
    node[AT.u_rated] = [10000, 400]

    transformer = initialize_array(DT.input, CT.transformer, 1)
    transformer[AT.id] = [3]
    transformer[AT.from_node] = [1]
    transformer[AT.to_node] = [2]
    transformer[AT.from_status] = [1]
    transformer[AT.to_status] = [1]
    transformer[AT.u1] = [10000]
    transformer[AT.u2] = [400]
    transformer[AT.sn] = [100000]
    transformer[AT.uk] = [0.1]
    transformer[AT.pk] = [1000]
    transformer[AT.i0] = [1.0e-6]
    transformer[AT.p0] = [0.1]
    transformer[AT.winding_from] = [2]
    transformer[AT.winding_to] = [1]
    transformer[AT.clock] = [5]
    transformer[AT.tap_side] = [0]
    transformer[AT.tap_pos] = [3]
    transformer[AT.tap_min] = [-11]
    transformer[AT.tap_max] = [9]
    transformer[AT.tap_size] = [100]

    sym_load = initialize_array(DT.input, CT.sym_load, 1)
    sym_load[AT.id] = [4]
    sym_load[AT.node] = [2]
    sym_load[AT.status] = [1]
    sym_load[AT.type] = [LoadGenType.const_power]
    sym_load[AT.p_specified] = [1000.0]
    sym_load[AT.q_specified] = [200.0]

    source = initialize_array(DT.input, CT.source, 1)
    source[AT.id] = [5]
    source[AT.node] = [1]
    source[AT.status] = [1]
    source[AT.u_ref] = [1.0]

    sym_current_sensor = initialize_array(DT.input, CT.sym_current_sensor, 1)
    sym_current_sensor[AT.id] = [6]
    sym_current_sensor[AT.measured_object] = [3]
    sym_current_sensor[AT.measured_terminal_type] = [MeasuredTerminalType.branch_to]
    sym_current_sensor[AT.angle_measurement_type] = [AngleMeasurementType.local_angle]
    sym_current_sensor[AT.i_sigma] = [100]
    sym_current_sensor[AT.i_angle_sigma] = [0.1]
    sym_current_sensor[AT.i_measured] = [1000.0]
    sym_current_sensor[AT.i_angle_measured] = [0.2]

    sym_voltage_sensor = initialize_array(DT.input, CT.sym_voltage_sensor, 1)
    sym_voltage_sensor[AT.id] = [7]
    sym_voltage_sensor[AT.measured_object] = [1]
    sym_voltage_sensor[AT.u_sigma] = [1.0]
    sym_voltage_sensor[AT.u_measured] = [10000.0]

    return {
        CT.node: node,
        CT.transformer: transformer,
        CT.sym_load: sym_load,
        CT.source: source,
        CT.sym_current_sensor: sym_current_sensor,
        CT.sym_voltage_sensor: sym_voltage_sensor,
    }


@pytest.fixture
def input_data__voltage_regulator():
    voltage_regulator = initialize_array(DT.input, CT.voltage_regulator, 1)
    voltage_regulator[AT.id] = [8]
    voltage_regulator[AT.regulated_object] = [4]
    voltage_regulator[AT.status] = [1]
    voltage_regulator[AT.u_ref] = [1.05]
    return {CT.voltage_regulator: voltage_regulator}


@pytest.fixture
def input_data__transformer_tap_regulator():
    transformer_tap_regulator = initialize_array(DT.input, CT.transformer_tap_regulator, 1)
    transformer_tap_regulator[AT.id] = [8]
    transformer_tap_regulator[AT.regulated_object] = [3]
    transformer_tap_regulator[AT.status] = [1]
    transformer_tap_regulator[AT.control_side] = [BranchSide.to_side]
    transformer_tap_regulator[AT.u_set] = [400.0]
    transformer_tap_regulator[AT.u_band] = [20.0]
    return {CT.transformer_tap_regulator: transformer_tap_regulator}


@pytest.mark.parametrize("regulator_input", ["input_data__voltage_regulator", "input_data__transformer_tap_regulator"])
def test_irrelevant_components__power_flow(input_data__irrelevant_components_test, regulator_input, request):
    regulator = request.getfixturevalue(regulator_input)
    input_data = {**input_data__irrelevant_components_test, **regulator}
    model = PowerGridModel(input_data)
    result = model.calculate_power_flow()

    assert CT.transformer in result
    assert CT.sym_load in result
    assert CT.source in result
    assert CT.node in result
    assert CT.sym_voltage_sensor not in result
    assert CT.sym_current_sensor not in result
    if CT.voltage_regulator in regulator:
        assert CT.voltage_regulator in result
    else:
        assert CT.transformer_tap_regulator in result


@pytest.mark.parametrize("regulator_input", ["input_data__voltage_regulator", "input_data__transformer_tap_regulator"])
def test_irrelevant_components__state_estimation(input_data__irrelevant_components_test, regulator_input, request):
    regulator = request.getfixturevalue(regulator_input)
    input_data = {**input_data__irrelevant_components_test, **regulator}
    model = PowerGridModel(input_data)
    result = model.calculate_state_estimation()

    assert CT.transformer in result
    assert CT.sym_load in result
    assert CT.source in result
    assert CT.node in result
    assert CT.sym_voltage_sensor in result
    assert CT.sym_current_sensor in result
    assert CT.voltage_regulator not in result
    assert CT.transformer_tap_regulator not in result


@pytest.mark.parametrize("regulator_input", ["input_data__voltage_regulator", "input_data__transformer_tap_regulator"])
def test_irrelevant_components__short_circuit(input_data__irrelevant_components_test, regulator_input, request):
    regulator = request.getfixturevalue(regulator_input)
    input_data = {**input_data__irrelevant_components_test, **regulator}
    model = PowerGridModel(input_data)
    result = model.calculate_short_circuit()

    assert CT.transformer in result
    assert CT.sym_load in result
    assert CT.source in result
    assert CT.node in result
    assert CT.voltage_regulator not in result
    assert CT.transformer_tap_regulator not in result
    assert CT.transformer_tap_regulator not in result
    assert CT.sym_voltage_sensor not in result
    assert CT.sym_current_sensor not in result
