# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""Public observer acceptance for symmetric and asymmetric state estimation.

Sigmas are checked against independent references: finite differences of the
ordinary state-estimation outputs, physical formulas and dense inverses.
"""

from copy import deepcopy

import numpy as np
import pytest

from power_grid_model import (
    CalculationMethod as CM,
    ComponentType as CT,
    DatasetType as DT,
    PowerGridModel,
    initialize_array,
)
from power_grid_model.enum import LoadGenType, MeasuredTerminalType, ObservedTerminalType
from power_grid_model.errors import PowerGridError
from power_grid_model.utils import json_deserialize, json_serialize, msgpack_deserialize, msgpack_serialize

METHODS = [CM.iterative_linear, CM.newton_raphson]
FAMILIES = [CT.line, CT.asym_line, CT.generic_branch, CT.transformer, CT.three_winding_transformer]
OBSERVERS = [CT.voltage_uncertainty_observer, CT.power_uncertainty_observer, CT.current_uncertainty_observer]
SIGMAS = {
    CT.voltage_uncertainty_observer: ["u_sigma", "u_angle_sigma"],
    CT.power_uncertainty_observer: ["p_sigma", "q_sigma"],
    CT.current_uncertainty_observer: ["i_sigma", "i_angle_sigma"],
}
SYMMETRY = pytest.mark.parametrize("symmetric", [True, False], ids=["sym", "asym"])
# Every symmetric branch family; the asymmetric fixtures cover a coupled line
# and a three-winding transformer.
REFERENCE_CASES = [pytest.param(True, family, id=f"sym-{family.value}") for family in FAMILIES] + [
    pytest.param(False, family, id=f"asym-{family.value}") for family in [CT.line, CT.three_winding_transformer]
]


def rows(component, count=None, **fields):
    if count is None:
        count = max((np.shape(value)[0] if np.ndim(value) else 1 for value in fields.values()), default=1)
    result = initialize_array(DT.input, component, count)
    for name, value in fields.items():
        result[name] = value
    return result


def network(family=CT.line, relative=False):
    three = family == CT.three_winding_transformer
    n = 3 if three else 2
    rated = np.array([138e3, 69e3, 13.8e3] if three else [10e3, 10e3])
    common = dict(id=10, from_node=1, to_node=2, from_status=1, to_status=1)
    if family == CT.line:
        branch = dict(**common, r1=1.0, x1=2.0, c1=1e-8, tan1=0.01, r0=1.0, x0=2.0, c0=1e-8, tan0=0.01, i_n=1e3)
    elif family == CT.asym_line:
        branch = dict(
            **common,
            r_aa=1.0,
            r_bb=1.1,
            r_cc=1.2,
            r_ba=0.1,
            r_ca=0.05,
            r_cb=0.15,
            x_aa=2.0,
            x_bb=2.1,
            x_cc=2.2,
            x_ba=0.1,
            x_ca=0.05,
            x_cb=0.15,
            c_aa=1e-8,
            c_bb=1.1e-8,
            c_cc=1.2e-8,
            c_ba=0.0,
            c_ca=0.0,
            c_cb=0.0,
            i_n=1e3,
        )
    elif family == CT.generic_branch:
        branch = dict(**common, r1=1.0, x1=2.0, g1=1e-7, b1=1e-6, k=1.03, theta=0.15, sn=10e6)
    elif family == CT.transformer:
        branch = dict(
            **common,
            u1=10e3,
            u2=10e3,
            sn=10e6,
            uk=0.1,
            pk=1e4,
            i0=0.01,
            p0=1e3,
            winding_from=1,
            winding_to=2,
            clock=1,
            tap_side=0,
            tap_pos=1,
            tap_min=-2,
            tap_max=2,
            tap_nom=0,
            tap_size=100.0,
        )
    else:
        branch = dict(
            id=10,
            node_1=1,
            node_2=2,
            node_3=3,
            status_1=1,
            status_2=1,
            status_3=1,
            u1=rated[0],
            u2=rated[1],
            u3=rated[2],
            sn_1=60e6,
            sn_2=50e6,
            sn_3=10e6,
            uk_12=0.09,
            uk_13=0.03,
            uk_23=0.06,
            pk_12=5e4,
            pk_13=5e3,
            pk_23=1e4,
            i0=0.01,
            p0=1e3,
            winding_1=1,
            winding_2=2,
            winding_3=2,
            clock_12=11,
            clock_13=11,
            tap_side=2,
            tap_pos=0,
            tap_min=-2,
            tap_max=2,
            tap_nom=0,
            tap_size=100.0,
        )
    terminals = [6, 7, 8] if three else [0, 1]
    return {
        CT.node: rows(CT.node, id=np.arange(1, n + 1), u_rated=rated),
        family: rows(family, **branch),
        CT.source: rows(CT.source, id=20, node=1, status=1, u_ref=1.0),
        CT.sym_load: rows(
            CT.sym_load,
            id=np.arange(21, 20 + n),
            node=np.arange(2, n + 1),
            status=1,
            type=0,
            p_specified=1e5,
            q_specified=1e4,
        ),
        CT.sym_voltage_sensor: rows(
            CT.sym_voltage_sensor,
            id=np.arange(31, 31 + n),
            measured_object=np.arange(1, n + 1),
            u_sigma=rated * np.arange(1, n + 1) * 0.01,
            u_measured=rated * np.linspace(1.03, 0.97, n),
            u_angle_measured=np.full(n, np.nan) if relative else np.linspace(0.13, -0.08, n),
        ),
        CT.voltage_uncertainty_observer: rows(
            CT.voltage_uncertainty_observer, id=np.arange(101, 101 + n), observed_object=np.arange(1, n + 1), status=1
        ),
        CT.power_uncertainty_observer: rows(
            CT.power_uncertainty_observer,
            id=np.arange(111, 111 + n),
            observed_object=10,
            observed_terminal_type=terminals,
            status=1,
        ),
        CT.current_uncertainty_observer: rows(
            CT.current_uncertainty_observer,
            id=np.arange(121, 121 + n),
            observed_object=10,
            observed_terminal_type=terminals,
            status=1,
        ),
    }


def asym_network(three_winding=False):
    count = 3 if three_winding else 2
    rated = np.array([138e3, 69e3, 13.8e3] if three_winding else [10e3, 10e3])
    phase_base = rated / np.sqrt(3.0)
    magnitudes = phase_base[:, None] * np.array([1.03, 0.98, 1.06])[None, :]
    magnitudes *= np.linspace(1.0, 0.94, count)[:, None]
    angles = np.array([0.13, -2.01, 2.18])[None, :] + np.linspace(0.0, -0.2, count)[:, None]
    family = CT.three_winding_transformer if three_winding else CT.line
    if three_winding:
        # On the side-1 power base the three star legs have uk=[.054,.054,.306]
        # and r=[.0006,.0006,.0024]: positive passive leakage on every leg.
        branch = rows(
            family,
            id=10,
            node_1=1,
            node_2=2,
            node_3=3,
            status_1=1,
            status_2=1,
            status_3=1,
            u1=rated[0],
            u2=rated[1],
            u3=rated[2],
            sn_1=60e6,
            sn_2=50e6,
            sn_3=10e6,
            uk_12=0.09,
            uk_13=0.06,
            uk_23=0.06,
            pk_12=5e4,
            pk_13=5e3,
            pk_23=5e3,
            i0=0.01,
            p0=1e3,
            winding_1=1,
            winding_2=2,
            winding_3=2,
            clock_12=11,
            clock_13=11,
            tap_side=2,
            tap_pos=0,
            tap_min=-2,
            tap_max=2,
            tap_nom=0,
            tap_size=100.0,
        )
        terminals = [ObservedTerminalType.branch3_1, ObservedTerminalType.branch3_2, ObservedTerminalType.branch3_3]
    else:
        branch = rows(
            family,
            id=10,
            from_node=1,
            to_node=2,
            from_status=1,
            to_status=1,
            r1=1.0,
            x1=2.0,
            r0=2.0,
            x0=3.0,
            c1=0.0,
            c0=0.0,
            tan1=0.0,
            tan0=0.0,
            i_n=1000.0,
        )
        terminals = [ObservedTerminalType.branch_from, ObservedTerminalType.branch_to]
    return {
        CT.node: rows(CT.node, count, id=np.arange(1, count + 1), u_rated=rated),
        family: branch,
        # Unmeasured sources leave every physical node's injection free. Thus
        # complete voltage measurements give an independent diagonal state oracle.
        CT.source: rows(
            CT.source, count, id=np.arange(20, 20 + count), node=np.arange(1, count + 1), status=1, u_ref=1.0
        ),
        CT.asym_voltage_sensor: rows(
            CT.asym_voltage_sensor,
            count,
            id=np.arange(31, 31 + count),
            measured_object=np.arange(1, count + 1),
            u_sigma=phase_base * np.linspace(0.01, 0.02, count),
            u_measured=magnitudes,
            u_angle_measured=angles,
        ),
        CT.voltage_uncertainty_observer: rows(
            CT.voltage_uncertainty_observer,
            count,
            id=np.arange(101, 101 + count),
            observed_object=np.arange(1, count + 1),
            status=1,
        ),
        CT.power_uncertainty_observer: rows(
            CT.power_uncertainty_observer,
            2 * count,
            id=np.arange(111, 111 + 2 * count),
            observed_object=[10] * count + list(range(1, count + 1)),
            observed_terminal_type=terminals + [ObservedTerminalType.node_injection] * count,
            status=1,
        ),
        CT.current_uncertainty_observer: rows(
            CT.current_uncertainty_observer,
            count,
            id=np.arange(121, 121 + count),
            observed_object=10,
            observed_terminal_type=terminals,
            status=1,
        ),
    }


def observer_network(symmetric):
    return network() if symmetric else asym_network()


def calculate(data, method, symmetric=True, **kwargs):
    return PowerGridModel(data).calculate_state_estimation(
        symmetric=symmetric, calculation_method=method, error_tolerance=1e-11, max_iterations=100, **kwargs
    )


def physical_quantities(output, family, injection=False):
    nodes = output[CT.node]
    if injection:
        return np.stack([nodes["p"], nodes["q"]], axis=-1).ravel()
    branch = output[family][0]
    labels = ["1", "2", "3"] if family == CT.three_winding_transformer else ["from", "to"]
    result = []
    for index, label in enumerate(labels):
        power = branch[f"p_{label}"] + 1j * branch[f"q_{label}"]
        angle = nodes["u_angle"][index] - np.angle(power)
        result.append(np.stack([power.real, power.imag, branch[f"i_{label}"], angle], axis=-1))
    return np.ravel(result)


def check_family_reference(method, family, symmetric, injection=False):
    data = network(family) if symmetric else asym_network(family == CT.three_winding_transformer)
    n = len(data[CT.node])
    if injection and symmetric:
        data[CT.power_uncertainty_observer] = np.concatenate(
            [
                data[CT.power_uncertainty_observer],
                rows(
                    CT.power_uncertainty_observer,
                    id=np.arange(131, 131 + n),
                    observed_object=np.arange(1, n + 1),
                    observed_terminal_type=9,
                    status=1,
                ),
            ]
        )
    result = calculate(data, method, symmetric)
    ordinary = calculate(data, method, symmetric, output_component_types=[CT.node, family])
    np.testing.assert_array_equal(result[CT.node], ordinary[CT.node])
    np.testing.assert_array_equal(result[family], ordinary[family])
    # Perturb the voltage measurements: one coordinate pair per node (phase).
    sensor_type = CT.sym_voltage_sensor if symmetric else CT.asym_voltage_sensor
    sensor = data[sensor_type]
    shape = sensor["u_measured"].shape
    phases = 1 if symmetric else 3
    u = (sensor["u_measured"] * np.exp(1j * sensor["u_angle_measured"])).ravel()
    sigma = np.repeat(sensor["u_sigma"], phases)
    base = np.repeat(data[CT.node]["u_rated"] / (1.0 if symmetric else np.sqrt(3.0)), phases)
    if method == CM.iterative_linear:
        x = np.column_stack([u.real, u.imag]).ravel()
        variance = np.repeat(sigma**2 / 2.0, 2)
    else:
        x = np.column_stack([np.angle(u), abs(u)]).ravel()
        variance = np.column_stack([(sigma / base) ** 2, sigma**2]).ravel()
    columns = []
    for coordinate in range(len(x)):
        step = 1e-5 if method == CM.newton_raphson and coordinate % 2 == 0 else base[coordinate // 2] * 1e-5
        outputs = []
        for sign in [-1, 1]:
            state = x.copy()
            state[coordinate] += sign * step
            perturbed_u = (
                state[::2] + 1j * state[1::2]
                if method == CM.iterative_linear
                else state[1::2] * np.exp(1j * state[::2])
            )
            perturbed = deepcopy(data)
            perturbed[sensor_type]["u_measured"] = abs(perturbed_u).reshape(shape)
            perturbed[sensor_type]["u_angle_measured"] = np.angle(perturbed_u).reshape(shape)
            ordinary = calculate(perturbed, method, symmetric, output_component_types=[CT.node, family])
            outputs.append(physical_quantities(ordinary, family, injection))
        delta = outputs[1] - outputs[0]
        # Current angles use their continuous local branch, even across +/- pi.
        if not injection:
            delta[3::4] = np.angle(np.exp(1j * delta[3::4]))
        columns.append(delta / (2.0 * step))
    expected = np.sqrt(np.column_stack(columns) ** 2 @ variance)
    power = result[CT.power_uncertainty_observer]
    if injection:
        actual = np.stack([power["p_sigma"][-n:], power["q_sigma"][-n:]], axis=-1)
    else:
        current = result[CT.current_uncertainty_observer]
        terminals = len(current)
        actual = np.stack(
            [power["p_sigma"][:terminals], power["q_sigma"][:terminals], current["i_sigma"], current["i_angle_sigma"]],
            axis=-1,
        )
    np.testing.assert_allclose(actual, expected.reshape(actual.shape), rtol=2e-6)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(("symmetric", "family"), REFERENCE_CASES)
def test_branch_terminals_match_independent_ordinary_output_jacobian(method, symmetric, family):
    check_family_reference(method, family, symmetric)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(("symmetric", "family"), REFERENCE_CASES)
def test_node_injection_matches_independent_ordinary_output_jacobian(method, symmetric, family):
    check_family_reference(method, family, symmetric, injection=True)


@pytest.mark.parametrize("method", METHODS)
def test_voltage_observers_match_one_measurement_phase_units_and_preserve_estimates(method):
    data = asym_network()
    ordinary = calculate(data, method, symmetric=False, output_component_types=[CT.node, CT.line])
    result = calculate(data, method, symmetric=False)
    for component in [CT.node, CT.line]:
        np.testing.assert_array_equal(result[component], ordinary[component])
    voltage = result[CT.voltage_uncertainty_observer]
    sigma = data[CT.asym_voltage_sensor]["u_sigma"]
    factor = 1 / np.sqrt(2) if method == CM.iterative_linear else 1.0
    np.testing.assert_allclose(voltage["u_sigma"], np.broadcast_to((factor * sigma)[:, None], (2, 3)), rtol=1e-10)
    expected_angle = (
        factor * sigma[:, None] / data[CT.asym_voltage_sensor]["u_measured"]
        if method == CM.iterative_linear
        else np.broadcast_to((sigma / (data[CT.node]["u_rated"] / np.sqrt(3)))[:, None], (2, 3))
    )
    np.testing.assert_allclose(voltage["u_angle_sigma"], expected_angle, rtol=1e-10)
    for component in OBSERVERS:
        assert result[component]["id"].shape == (len(data[component]),)
        assert result[component]["energized"].shape == (len(data[component]),)
        for field in SIGMAS[component]:
            assert result[component][field].shape == (len(data[component]), 3)


@SYMMETRY
def test_magnitude_only_nrse_retains_inverse_virtual_phase_variance(symmetric):
    rated, sigma = 10000.0, 20.0
    voltage_base = rated if symmetric else rated / np.sqrt(3.0)
    sensor_type = CT.sym_voltage_sensor if symmetric else CT.asym_voltage_sensor
    data = {
        CT.node: rows(CT.node, id=1, u_rated=rated),
        CT.source: rows(CT.source, id=2, node=1, status=1, u_ref=1.0),
        sensor_type: rows(
            sensor_type, id=3, measured_object=1, u_sigma=sigma, u_measured=voltage_base, u_angle_measured=np.nan
        ),
        CT.voltage_uncertainty_observer: rows(CT.voltage_uncertainty_observer, id=100, observed_object=1, status=1),
    }
    ordinary = calculate(data, CM.newton_raphson, symmetric, output_component_types=[CT.node])
    result = calculate(data, CM.newton_raphson, symmetric)
    np.testing.assert_array_equal(result[CT.node], ordinary[CT.node])
    observer = result[CT.voltage_uncertainty_observer][0]
    np.testing.assert_allclose(observer["u_sigma"], sigma, rtol=1e-10)
    # The inverse retains one virtual-angle variance per phase. Reporting relative
    # to phase a removes its coordinate and gives Var(theta_b/c - theta_a)=2*sigma_pu^2.
    expected_angle = (
        [0.0] if symmetric else [0.0, np.sqrt(2.0) * sigma / voltage_base, np.sqrt(2.0) * sigma / voltage_base]
    )
    np.testing.assert_allclose(np.atleast_1d(observer["u_angle_sigma"]), expected_angle, rtol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_balanced_independent_phase_voltage_noise_maps_to_symmetric_positive_sequence(method):
    # Equal independent phase errors, mapped into a positive-sequence phasor,
    # give this specific one-bus reduction. No network-equivalence claim follows.
    rated, sigma, angle = 10000.0, 20.0, 0.13
    phase_voltage = rated / np.sqrt(3.0)
    common = {
        CT.node: rows(CT.node, id=1, u_rated=rated),
        CT.source: rows(CT.source, id=2, node=1, status=1, u_ref=1.0),
        CT.voltage_uncertainty_observer: rows(CT.voltage_uncertainty_observer, id=100, observed_object=1, status=1),
    }
    asymmetric = deepcopy(common)
    asymmetric[CT.asym_voltage_sensor] = rows(
        CT.asym_voltage_sensor,
        id=3,
        measured_object=1,
        u_sigma=sigma,
        u_measured=[[phase_voltage] * 3],
        u_angle_measured=[[angle, angle - 2 * np.pi / 3, angle + 2 * np.pi / 3]],
    )
    symmetric = deepcopy(common)
    symmetric[CT.sym_voltage_sensor] = rows(
        CT.sym_voltage_sensor,
        id=3,
        measured_object=1,
        u_sigma=sigma,
        u_measured=rated,
        u_angle_measured=angle,
    )
    phase = calculate(asymmetric, method, symmetric=False, output_component_types=[CT.voltage_uncertainty_observer])[
        CT.voltage_uncertainty_observer
    ][0]
    positive_sequence = PowerGridModel(symmetric).calculate_state_estimation(
        symmetric=True,
        calculation_method=method,
        output_component_types=[CT.voltage_uncertainty_observer],
    )[CT.voltage_uncertainty_observer][0]
    factor = 1 / np.sqrt(2) if method == CM.iterative_linear else 1.0
    np.testing.assert_allclose(phase["u_sigma"], sigma * factor, rtol=1e-10)
    np.testing.assert_allclose(phase["u_angle_sigma"], sigma * factor / phase_voltage, rtol=1e-10)
    # U1 = (Ua + a Ub + a^2 Uc)/3. Independent equal phase
    # radial/tangential variances sum with squared coefficient 1/9.
    mapped_line_voltage_sigma = np.sqrt(3) * np.sqrt(np.sum(phase["u_sigma"] ** 2) / 9)
    mapped_angle_sigma = np.sqrt(np.sum(phase["u_angle_sigma"] ** 2) / 9)
    np.testing.assert_allclose(mapped_line_voltage_sigma, positive_sequence["u_sigma"], rtol=1e-10)
    np.testing.assert_allclose(mapped_angle_sigma, positive_sequence["u_angle_sigma"], rtol=1e-10)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("wire_format", ["json", "msgpack"])
@SYMMETRY
def test_observer_row_column_and_wire_roundtrips_preserve_values_and_missingness(
    method, compact, wire_format, symmetric
):
    data = observer_network(symmetric)
    data[CT.current_uncertainty_observer]["status"][1] = 0
    row = calculate(data, method, symmetric, output_component_types=OBSERVERS)
    fields = {component: list(row[component].dtype.names) for component in OBSERVERS}
    column = calculate(data, method, symmetric, output_component_types=fields)
    for component in OBSERVERS:
        for field in fields[component]:
            np.testing.assert_array_equal(column[component][field], row[component][field])
    assert row[CT.current_uncertainty_observer]["energized"].tolist() == [1, 0]
    assert np.isnan(row[CT.current_uncertainty_observer]["i_sigma"][1]).all()
    serialize, deserialize = (
        (json_serialize, json_deserialize) if wire_format == "json" else (msgpack_serialize, msgpack_deserialize)
    )
    # Synthetic missing slots test the generic wire contract independently from
    # whether a particular solver/measurement can produce that invalid value.
    partial = {component: values.copy() for component, values in row.items()}
    for component in OBSERVERS:
        for field in SIGMAS[component]:
            partial[component][field][1] = np.nan
            if not symmetric:
                partial[component][field][0, 1] = np.nan
    partial_column = {
        component: {field: values[field] for field in fields[component]} for component, values in partial.items()
    }
    dataset_type = DT.sym_output if symmetric else DT.asym_output
    for dataset, expected in [(row, row), (column, row), (partial, partial), (partial_column, partial)]:
        wire = serialize(dataset, dataset_type=dataset_type, use_compact_list=compact)
        if wire_format == "json":
            assert "NaN" not in wire
            # Map serialization omits a wholly missing scalar; compact lists and
            # three-phase arrays keep the position as null.
            if expected is partial and (compact or not symmetric):
                assert "null" in wire
        restored = deserialize(wire)
        for component in OBSERVERS:
            for field in fields[component]:
                np.testing.assert_array_equal(restored[component][field], expected[component][field])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("three_winding", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("positional", [False, True])
@SYMMETRY
def test_topology_status_and_measurement_batches_match_independent_scenarios_and_restore_base(
    method, three_winding, sparse, positional, symmetric
):
    family = CT.three_winding_transformer if three_winding else CT.line
    data = network(family) if symmetric else asym_network(three_winding)
    sensor_type = CT.sym_voltage_sensor if symmetric else CT.asym_voltage_sensor
    count = len(data[CT.current_uncertainty_observer])
    topology = initialize_array(DT.update, family, (3, 1))
    topology["id"] = 10
    topology["status_3" if three_winding else "to_status"][:, 0] = [1, 0, 1]
    status = initialize_array(DT.update, CT.current_uncertainty_observer, (3, count))
    if not positional:
        status["id"] = data[CT.current_uncertainty_observer]["id"]
    status["status"] = 1
    status["status"][0, 0] = 0
    voltage = initialize_array(DT.update, sensor_type, (3, 1))
    voltage["id"] = 31
    voltage["u_sigma"][:, 0] = data[sensor_type]["u_sigma"][0] * np.array([1, 2, 1])
    dense = {family: topology, CT.current_uncertainty_observer: status, sensor_type: voltage}
    updates = {
        component: {"indptr": np.arange(0, values.size + 1, values.shape[1], dtype=np.int64), "data": values.ravel()}
        if sparse
        else values
        for component, values in dense.items()
    }
    selection = [CT.node, *OBSERVERS]
    options = dict(
        symmetric=symmetric,
        calculation_method=method,
        max_iterations=100,
        error_tolerance=1e-11,
        output_component_types=selection,
    )
    model = PowerGridModel(data)
    result = model.calculate_state_estimation(update_data=updates, **options)
    for scenario in range(3):
        independent = PowerGridModel(data)
        independent.update(update_data={component: values[scenario] for component, values in dense.items()})
        expected = independent.calculate_state_estimation(**options)
        for component in selection:
            for field in expected[component].dtype.names:
                np.testing.assert_array_equal(result[component][field][scenario], expected[component][field])
    # The disconnected terminal's observers are de-energized and missing.
    assert result[CT.power_uncertainty_observer]["energized"][1, count - 1] == 0
    for field in SIGMAS[CT.power_uncertainty_observer]:
        assert np.isnan(result[CT.power_uncertainty_observer][field][1, count - 1]).all()
    after = model.calculate_state_estimation(**options)
    baseline = calculate(data, method, symmetric, output_component_types=selection)
    for component in selection:
        for field in baseline[component].dtype.names:
            np.testing.assert_array_equal(after[component][field], baseline[component][field])


@pytest.mark.parametrize("method", METHODS)
def test_observer_batch_error_isolated_from_success_and_later_calls(method):
    data = network()
    model = PowerGridModel(data)
    update = initialize_array(DT.update, CT.current_uncertainty_observer, (3, 1))
    update["id"][:, 0] = [121, 9999, 122]
    update["status"][:, 0] = [0, 0, 0]
    result = model.calculate_state_estimation(
        calculation_method=method,
        update_data={CT.current_uncertainty_observer: update},
        continue_on_batch_error=True,
        output_component_types=OBSERVERS,
    )
    assert model.batch_error is not None
    np.testing.assert_array_equal(model.batch_error.failed_scenarios, [1])
    np.testing.assert_array_equal(model.batch_error.succeeded_scenarios, [0, 2])
    assert result[CT.current_uncertainty_observer]["energized"][0].tolist() == [0, 1]
    assert result[CT.current_uncertainty_observer]["energized"][2].tolist() == [1, 0]
    after = model.calculate_state_estimation(calculation_method=method, output_component_types=OBSERVERS)
    assert after[CT.current_uncertainty_observer]["energized"].tolist() == [1, 1]
    assert np.isfinite(after[CT.current_uncertainty_observer]["i_sigma"]).all()


@pytest.mark.parametrize("method", METHODS)
@SYMMETRY
def test_metadata_empty_disabled_and_duplicate_requests_keep_estimates_and_input_order(method, symmetric):
    data = observer_network(symmetric)
    data[CT.current_uncertainty_observer]["status"][1] = 0
    for index, component in enumerate(OBSERVERS):
        duplicate = np.tile(data[component][::-1], 2)
        duplicate["id"] = np.arange(1000 + 100 * index, 1000 + 100 * index + len(duplicate))
        data[component] = duplicate
    ordinary = calculate(data, method, symmetric, output_component_types=[CT.node])
    metadata = calculate(
        data,
        method,
        symmetric,
        output_component_types={CT.node: None, CT.voltage_uncertainty_observer: ["id", "energized"]},
    )
    empty = calculate(
        data, method, symmetric, output_component_types={CT.node: None, CT.voltage_uncertainty_observer: []}
    )
    result = calculate(data, method, symmetric, output_component_types=[CT.node, *OBSERVERS])
    for output in [metadata, empty, result]:
        np.testing.assert_array_equal(output[CT.node], ordinary[CT.node])
    assert CT.voltage_uncertainty_observer not in empty
    for component in OBSERVERS:
        observed = result[component]
        np.testing.assert_array_equal(observed["id"], data[component]["id"])
        half = len(observed) // 2
        for field in SIGMAS[component]:
            np.testing.assert_array_equal(observed[field][:half], observed[field][half:])
    current = result[CT.current_uncertainty_observer]
    disabled = data[CT.current_uncertainty_observer]["status"] == 0
    np.testing.assert_array_equal(current["energized"], ~disabled)
    for field in SIGMAS[CT.current_uncertainty_observer]:
        assert np.isnan(current[field][disabled]).all()
        assert np.isfinite(current[field][~disabled]).all()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("noisy", [False, True])
def test_public_injection_exact_zero_predicate_distinguishes_noisy_zero(method, noisy):
    data = network()
    data[CT.power_uncertainty_observer] = rows(
        CT.power_uncertainty_observer, id=140, observed_object=2, observed_terminal_type=9, status=1
    )
    if noisy:
        data[CT.sym_load]["p_specified"] = 0
        data[CT.sym_load]["q_specified"] = 0
        data[CT.sym_power_sensor] = rows(
            CT.sym_power_sensor,
            id=50,
            measured_object=21,
            measured_terminal_type=4,
            p_measured=0.0,
            q_measured=0.0,
            power_sigma=100.0,
        )
    else:
        del data[CT.sym_load]
    result = PowerGridModel(data).calculate_state_estimation(calculation_method=method, max_iterations=100)
    output = result[CT.power_uncertainty_observer]
    assert output["energized"].tolist() == [1]
    if noisy:
        assert np.isfinite(output["p_sigma"]).all()
        assert np.all(output["p_sigma"] > 0)
        assert np.isfinite(output["q_sigma"]).all()
        assert np.all(output["q_sigma"] > 0)
    else:
        np.testing.assert_array_equal(output["p_sigma"], [0.0])
        np.testing.assert_array_equal(output["q_sigma"], [0.0])


def fused_network(injection=True):
    data = network()
    data[CT.node] = rows(CT.node, id=[1, 2, 3], u_rated=10e3)
    data[CT.link] = rows(CT.link, id=40, from_node=2, to_node=3, from_status=1, to_status=0)
    data[CT.voltage_uncertainty_observer] = rows(
        CT.voltage_uncertainty_observer, id=[101, 102, 103], observed_object=[1, 2, 3], status=1
    )
    if injection:
        data[CT.power_uncertainty_observer] = rows(
            CT.power_uncertainty_observer, id=140, observed_object=2, observed_terminal_type=9, status=1
        )
    return data


@pytest.mark.parametrize("method", METHODS)
def test_fused_voltage_mapping_and_injection_rejection_follow_link_updates(method):
    data = fused_network()
    model = PowerGridModel(data)
    update = initialize_array(DT.update, CT.link, (3, 1))
    update["id"] = 40
    update["to_status"][:, 0] = [0, 1, 0]
    result = model.calculate_state_estimation(
        calculation_method=method,
        update_data={CT.link: update},
        continue_on_batch_error=True,
        output_component_types=OBSERVERS,
    )
    assert model.batch_error is not None
    np.testing.assert_array_equal(model.batch_error.failed_scenarios, [1])
    np.testing.assert_array_equal(model.batch_error.succeeded_scenarios, [0, 2])
    assert "fused" in model.batch_error.error_messages[0].lower()
    for field in ["p_sigma", "q_sigma"]:
        np.testing.assert_array_equal(
            result[CT.power_uncertainty_observer][field][0], result[CT.power_uncertainty_observer][field][2]
        )
    data[CT.link]["to_status"] = 1
    model = PowerGridModel(data)
    # Metadata performs no ambiguous per-node injection attribution.
    metadata = model.calculate_state_estimation(
        calculation_method=method,
        output_component_types={
            CT.voltage_uncertainty_observer: None,
            CT.power_uncertainty_observer: ["id", "energized"],
        },
    )
    voltage = metadata[CT.voltage_uncertainty_observer]
    np.testing.assert_array_equal(voltage["u_sigma"][1], voltage["u_sigma"][2])
    np.testing.assert_array_equal(voltage["u_angle_sigma"][1], voltage["u_angle_sigma"][2])
    assert metadata[CT.power_uncertainty_observer]["energized"].tolist() == [1]
    data[CT.power_uncertainty_observer]["status"] = 0
    disabled = PowerGridModel(data).calculate_state_estimation(
        calculation_method=method, output_component_types=OBSERVERS
    )
    assert disabled[CT.power_uncertainty_observer]["energized"].tolist() == [0]
    assert np.isnan(disabled[CT.power_uncertainty_observer]["p_sigma"]).all()


@pytest.mark.parametrize("method", METHODS)
def test_zero_current_returns_missing_current_sigmas_with_valid_signed_power(method):
    data = network()
    data[CT.line]["c1"] = 0
    data[CT.sym_voltage_sensor]["u_measured"] = 10e3
    data[CT.sym_voltage_sensor]["u_angle_measured"] = 0.13
    result = PowerGridModel(data).calculate_state_estimation(
        calculation_method=method, output_component_types=OBSERVERS
    )
    assert np.isnan(result[CT.current_uncertainty_observer]["i_sigma"]).all()
    assert np.isnan(result[CT.current_uncertainty_observer]["i_angle_sigma"]).all()
    assert np.isfinite(result[CT.power_uncertainty_observer]["p_sigma"]).all()
    assert np.isfinite(result[CT.power_uncertainty_observer]["q_sigma"]).all()


@pytest.mark.parametrize(
    ("component", "target", "terminal"),
    [
        (CT.voltage_uncertainty_observer, 10, None),
        (CT.power_uncertainty_observer, 20, 2),
        (CT.power_uncertainty_observer, 10, 6),
        (CT.current_uncertainty_observer, 2, 9),
        (CT.current_uncertainty_observer, 9999, 0),
    ],
)
def test_invalid_observer_targets_rejected_even_when_disabled(component, target, terminal):
    data = network()
    data[component]["status"] = 0
    data[component]["observed_object"][0] = target
    if terminal is not None:
        data[component]["observed_terminal_type"][0] = terminal
    with pytest.raises(PowerGridError):
        PowerGridModel(data)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("wire_format", ["json", "msgpack"])
def test_public_input_and_update_wire_roundtrips_execute_identically(method, compact, wire_format):
    data = network(CT.transformer)
    serialize, deserialize = (
        (json_serialize, json_deserialize) if wire_format == "json" else (msgpack_serialize, msgpack_deserialize)
    )
    input_wire = serialize(data, dataset_type=DT.input, use_compact_list=compact)
    restored = deserialize(input_wire)
    model = PowerGridModel(restored)
    update = initialize_array(DT.update, CT.current_uncertainty_observer, (2, 2))
    update["id"] = [121, 122]
    update["status"] = [[0, 1], [1, 0]]
    update_data = {CT.current_uncertainty_observer: update}
    restored_update = deserialize(serialize(update_data, dataset_type=DT.update, use_compact_list=compact))
    result = model.calculate_state_estimation(
        calculation_method=method, update_data=restored_update, output_component_types=OBSERVERS
    )
    expected = PowerGridModel(data).calculate_state_estimation(
        calculation_method=method, update_data=update_data, output_component_types=OBSERVERS
    )
    for component in OBSERVERS:
        for field in expected[component].dtype.names:
            np.testing.assert_array_equal(result[component][field], expected[component][field])


@pytest.mark.parametrize("method", METHODS)
def test_observer_cartesian_batch_status_and_measurement_updates(method):
    data = network()
    status = initialize_array(DT.update, CT.current_uncertainty_observer, (2, 2))
    status["id"] = [121, 122]
    status["status"] = [[1, 1], [0, 1]]
    voltage = initialize_array(DT.update, CT.sym_voltage_sensor, (2, 1))
    voltage["id"] = 31
    voltage["u_sigma"][:, 0] = [100.0, 200.0]
    result = PowerGridModel(data).calculate_state_estimation(
        calculation_method=method,
        update_data=[{CT.current_uncertainty_observer: status}, {CT.sym_voltage_sensor: voltage}],
        output_component_types=OBSERVERS,
    )
    assert result[CT.current_uncertainty_observer].shape == (4, 2)
    assert np.count_nonzero(result[CT.current_uncertainty_observer]["energized"][:, 0]) == len(status)
    for sigma in [100.0, 200.0]:
        scenario = deepcopy(data)
        scenario[CT.sym_voltage_sensor]["u_sigma"][0] = sigma
        single = PowerGridModel(scenario).calculate_state_estimation(
            calculation_method=method, output_component_types=OBSERVERS
        )
        expected = single[CT.current_uncertainty_observer]["i_sigma"][1]
        assert np.count_nonzero(np.isclose(result[CT.current_uncertainty_observer]["i_sigma"][:, 1], expected)) == len(
            status
        )


@pytest.mark.parametrize("method", METHODS)
@SYMMETRY
def test_zero_voltage_returns_missing_voltage_sigmas_only_in_that_phase(method, symmetric):
    data = {
        CT.node: rows(CT.node, id=1, u_rated=100.0),
        CT.source: rows(CT.source, id=2, node=1, status=1, u_ref=1.0),
        CT.voltage_uncertainty_observer: rows(CT.voltage_uncertainty_observer, id=100, observed_object=1, status=1),
    }
    if symmetric:
        data[CT.sym_load] = rows(CT.sym_load, id=3, node=1, status=1, type=0, p_specified=0.0, q_specified=0.0)
        data[CT.sym_voltage_sensor] = rows(
            CT.sym_voltage_sensor, id=4, measured_object=1, u_sigma=2.0, u_measured=0.0, u_angle_measured=0.0
        )
    else:
        data[CT.asym_voltage_sensor] = rows(
            CT.asym_voltage_sensor,
            id=3,
            measured_object=1,
            u_sigma=2.0,
            u_measured=[[50.0, 0.0, 60.0]],
            u_angle_measured=[[0.1, -2.0, 2.1]],
        )
    result = calculate(data, method, symmetric, output_component_types=[CT.voltage_uncertainty_observer])
    observer = result[CT.voltage_uncertainty_observer]
    assert observer["energized"].tolist() == [1]
    missing = [True] if symmetric else [False, True, False]
    for field in SIGMAS[CT.voltage_uncertainty_observer]:
        np.testing.assert_array_equal(np.isnan(np.atleast_1d(observer[field][0])), missing)


def separator_network(cycle):
    n = 4
    edges = [(1, 2), (2, 3), (3, 4), (4, 1)] if cycle else [(1, 4), (2, 4), (3, 4)]
    return {
        CT.node: rows(CT.node, id=np.arange(1, n + 1), u_rated=10000.0),
        CT.line: rows(
            CT.line,
            id=np.arange(10, 10 + len(edges)),
            from_node=[a for a, _ in edges],
            to_node=[b for _, b in edges],
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
            i_n=1000.0,
        ),
        CT.source: rows(CT.source, id=20, node=1, status=1, u_ref=1.0),
        CT.sym_load: rows(
            CT.sym_load, id=[21, 22, 23], node=[2, 3, 4], status=1, type=0, p_specified=1e5, q_specified=1e4
        ),
        CT.sym_voltage_sensor: rows(
            CT.sym_voltage_sensor,
            id=[31, 32, 33, 34],
            measured_object=[1, 2, 3, 4],
            u_sigma=100.0 + 10.0 * np.arange(n),
            u_measured=10000.0 + 50.0 * np.arange(n),
            u_angle_measured=0.1 - 0.03 * np.arange(n),
        ),
        CT.sym_power_sensor: rows(
            CT.sym_power_sensor,
            id=[41, 42, 43, 44],
            measured_object=[20, 21, 22, 23],
            measured_terminal_type=[2, 4, 4, 4],
            power_sigma=1e6,
            p_measured=1e5,
            q_measured=1e4,
        ),
        CT.power_uncertainty_observer: rows(
            CT.power_uncertainty_observer,
            id=[201, 202, 203, 204],
            observed_object=[1, 2, 3, 4],
            observed_terminal_type=9,
            status=1,
        ),
    }


def rectangular_covariance(complex_covariance):
    n = len(complex_covariance)
    result = np.empty((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            value = complex_covariance[i, j]
            result[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = (
                np.array([[value.real, -value.imag], [value.imag, value.real]]) / 2.0
            )
    return result


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cycle", [False, True])
def test_public_separator_exactness_and_augmented_path_approximation_against_dense_reference(method, cycle):
    data = separator_network(cycle)
    result = PowerGridModel(data).calculate_state_estimation(
        calculation_method=method,
        error_tolerance=1e-10,
        max_iterations=100,
        output_component_types=[CT.node, CT.power_uncertainty_observer],
    )
    voltage = result[CT.node]["u"] * np.exp(1j * result[CT.node]["u_angle"])
    n = len(voltage)
    admittance = np.zeros((n, n), dtype=complex)
    for branch in data[CT.line]:
        i, j = int(branch["from_node"]) - 1, int(branch["to_node"]) - 1
        y = 1.0 / (branch["r1"] + 1j * branch["x1"])
        admittance[i, i] += y
        admittance[j, j] += y
        admittance[i, j] -= y
        admittance[j, i] -= y

    def injections(u):
        power = u * (admittance @ u).conjugate()
        return np.column_stack([power.real, power.imag]).ravel()

    jacobian_rectangular = np.empty((2 * n, 2 * n))
    for k in range(2 * n):
        delta = np.zeros(n, dtype=complex)
        delta[k // 2] = 0.01 * (1 if k % 2 == 0 else 1j)
        jacobian_rectangular[:, k] = (injections(voltage + delta) - injections(voltage - delta)) / 0.02
    voltage_variance = data[CT.sym_voltage_sensor]["u_sigma"] ** 2
    # The prepared apparent-power sigma splits equally between P and Q.
    power_variance = data[CT.sym_power_sensor]["power_sigma"] ** 2 / 2.0
    voltage_base = data[CT.node]["u_rated"]
    power_base = 1e6
    voltage_variance_pu = voltage_variance / voltage_base**2
    power_variance_pu = power_variance / power_base**2
    alpha = min(np.min(voltage_variance_pu), np.min(2.0 * power_variance_pu))

    def reconstructed_state(augmented, state_size):
        # The native dispatcher fixture asserts the actual cycle ordering is
        # identical to physical node order: missing (0,2), first common middle 1.
        raw = np.linalg.inv(augmented)

        def block(i, j):
            rows = [*(state_size * i + np.arange(state_size)), *(state_size * (n + i) + np.arange(state_size))]
            columns = [*(state_size * j + np.arange(state_size)), *(state_size * (n + j) + np.arange(state_size))]
            return raw[np.ix_(rows, columns)]

        middle = block(1, 1)
        singular_values = np.linalg.svd(middle, compute_uv=False)
        conditioning_threshold = 1e-12
        assert singular_values[-1] / singular_values[0] >= conditioning_threshold
        completed_raw = block(0, 1) @ np.linalg.solve(middle, block(1, 2))
        return alpha * completed_raw[:state_size, :state_size]

    if method == CM.iterative_linear:
        # Frozen ILSE transforms P/Q to complex current with the documented
        # near-unit-voltage variance approximation; no nonlinear gain inverse.
        precision = (
            np.diag(1.0 / voltage_variance)
            + admittance.conjugate().T @ np.diag(data[CT.node]["u_rated"] ** 2 / (2.0 * power_variance)) @ admittance
        )
        complex_covariance = np.linalg.inv(precision)
        exact = rectangular_covariance(complex_covariance)
        completed = complex_covariance.copy()
        if cycle:
            y_pu = admittance * voltage_base[:, None] * voltage_base[None, :] / power_base
            augmented = np.block(
                [
                    [np.diag(alpha / voltage_variance_pu), y_pu.conjugate().T],
                    [y_pu, -np.diag(2.0 * power_variance_pu / alpha)],
                ]
            )
            completed[0, 2] = reconstructed_state(augmented, 1)[0, 0] * voltage_base[0] * voltage_base[2]
            completed[2, 0] = completed[0, 2].conjugate()
        approximate = rectangular_covariance(completed)
    else:
        transform = np.zeros((2 * n, 2 * n))
        for i, u in enumerate(voltage):
            transform[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [
                [-u.imag, np.cos(np.angle(u))],
                [u.real, np.sin(np.angle(u))],
            ]
        jacobian_polar = jacobian_rectangular @ transform
        polar_variance = np.column_stack([voltage_variance / data[CT.node]["u_rated"] ** 2, voltage_variance]).ravel()
        precision = (
            np.diag(1.0 / polar_variance)
            + jacobian_polar.T @ np.diag(1.0 / np.repeat(power_variance, 2)) @ jacobian_polar
        )
        exact = transform @ np.linalg.inv(precision) @ transform.T
        approximate = exact.copy()
        if cycle:
            coordinate_scale = np.column_stack([np.ones(n), voltage_base]).ravel()
            jacobian_pu = jacobian_polar * coordinate_scale[None, :] / power_base
            augmented = np.block(
                [
                    [np.diag(np.repeat(alpha / voltage_variance_pu, 2)), jacobian_pu.T],
                    [jacobian_pu, -np.diag(np.repeat(power_variance_pu / alpha, 2))],
                ]
            )
            state = reconstructed_state(augmented, 2)
            transform_pu = transform * coordinate_scale[None, :]
            path = transform_pu[0:2, 0:2] @ state @ transform_pu[4:6, 4:6].T
            approximate[0:2, 4:6] = path
            approximate[4:6, 0:2] = path.T
    exact_variance = np.diag(jacobian_rectangular @ exact @ jacobian_rectangular.T)
    expected_variance = np.diag(jacobian_rectangular @ approximate @ jacobian_rectangular.T)
    actual = np.column_stack(
        [result[CT.power_uncertainty_observer]["p_sigma"], result[CT.power_uncertainty_observer]["q_sigma"]]
    ).ravel()
    np.testing.assert_allclose(actual, np.sqrt(expected_variance), rtol=2e-7)
    if cycle:
        # The nonseparator result is checked as an approximation, with a
        # visible error against the exact dense covariance rather than equality.
        minimum_visible_error = 1e-8
        assert (
            np.linalg.norm(expected_variance - exact_variance) / np.linalg.norm(exact_variance) > minimum_visible_error
        )
    else:
        np.testing.assert_allclose(actual, np.sqrt(exact_variance), rtol=2e-7)


@pytest.mark.parametrize("method", METHODS)
def test_relative_current_angles_include_slack_cross_covariance_against_dense_reference(method):
    data = separator_network(cycle=False)
    data[CT.sym_voltage_sensor]["u_angle_measured"] = np.nan
    branches = data[CT.line]
    data[CT.current_uncertainty_observer] = rows(
        CT.current_uncertainty_observer,
        id=np.arange(301, 301 + 2 * len(branches)),
        observed_object=np.repeat(branches["id"], 2),
        observed_terminal_type=np.tile([0, 1], len(branches)),
        status=1,
    )
    data[CT.voltage_uncertainty_observer] = rows(
        CT.voltage_uncertainty_observer, id=[101, 102, 103, 104], observed_object=[1, 2, 3, 4], status=1
    )
    result = PowerGridModel(data).calculate_state_estimation(
        calculation_method=method,
        error_tolerance=1e-11,
        max_iterations=100,
        output_component_types=[CT.node, CT.current_uncertainty_observer, CT.voltage_uncertainty_observer],
    )
    voltage = result[CT.node]["u"] * np.exp(1j * result[CT.node]["u_angle"])
    n = len(voltage)
    admittance = np.zeros((n, n), dtype=complex)
    for branch in branches:
        i, j = int(branch["from_node"]) - 1, int(branch["to_node"]) - 1
        y = 1.0 / (branch["r1"] + 1j * branch["x1"])
        admittance[i, i] += y
        admittance[j, j] += y
        admittance[i, j] -= y
        admittance[j, i] -= y

    def injections(u):
        power = u * (admittance @ u).conjugate()
        return np.column_stack([power.real, power.imag]).ravel()

    def currents(u):
        values = []
        for branch in branches:
            i, j = int(branch["from_node"]) - 1, int(branch["to_node"]) - 1
            current = (u[i] - u[j]) / (np.sqrt(3.0) * (branch["r1"] + 1j * branch["x1"]))
            for value in [current, -current]:
                values.extend([abs(value), np.angle(value) - np.angle(u[0])])
        return np.array(values)

    current_jacobian = np.empty((4 * len(branches), 2 * n))
    power_jacobian = np.empty((2 * n, 2 * n))
    for coordinate in range(2 * n):
        delta = np.zeros(n, dtype=complex)
        delta[coordinate // 2] = 0.01 * (1 if coordinate % 2 == 0 else 1j)
        upper, lower = currents(voltage + delta), currents(voltage - delta)
        difference = upper - lower
        difference[1::2] = np.angle(np.exp(1j * difference[1::2]))
        current_jacobian[:, coordinate] = difference / 0.02
        power_jacobian[:, coordinate] = (injections(voltage + delta) - injections(voltage - delta)) / 0.02
    voltage_variance = data[CT.sym_voltage_sensor]["u_sigma"] ** 2
    power_variance = data[CT.sym_power_sensor]["power_sigma"] ** 2 / 2.0
    if method == CM.iterative_linear:
        precision = (
            np.diag(1.0 / voltage_variance)
            + admittance.conjugate().T @ np.diag(data[CT.node]["u_rated"] ** 2 / (2.0 * power_variance)) @ admittance
        )
        covariance = rectangular_covariance(np.linalg.inv(precision))
    else:
        transform = np.zeros((2 * n, 2 * n))
        for i, u in enumerate(voltage):
            transform[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [
                [-u.imag, np.cos(np.angle(u))],
                [u.real, np.sin(np.angle(u))],
            ]
        polar_power = power_jacobian @ transform
        weights = np.zeros(2 * n)
        weights[1::2] = 1.0 / voltage_variance
        gain = np.diag(weights) + polar_power.T @ np.diag(1.0 / np.repeat(power_variance, 2)) @ polar_power
        # Independent exact gauge constraint theta(source/slack node 1)=0.
        polar_covariance = np.zeros_like(gain)
        polar_covariance[1:, 1:] = np.linalg.inv(gain[1:, 1:])
        covariance = transform @ polar_covariance @ transform.T
    expected = np.sqrt(np.diag(current_jacobian @ covariance @ current_jacobian.T)).reshape(-1, 2)
    observed = result[CT.current_uncertainty_observer]
    actual = np.column_stack([observed["i_sigma"], observed["i_angle_sigma"]])
    np.testing.assert_allclose(actual, expected, rtol=3e-6)
    assert result[CT.voltage_uncertainty_observer]["u_angle_sigma"][0] == 0.0


# Both solvers have independent static absolute/relative reference oracles above.
# This mixed-reference topology-transition fixture uses NRSE.
@pytest.mark.parametrize("method", [CM.newton_raphson])
def test_topology_batch_changes_absolute_to_relative_reporting_frame(method):
    data = network()
    data[CT.node] = rows(CT.node, id=[1, 2, 3], u_rated=10000.0)
    data[CT.line] = rows(
        CT.line,
        id=[10, 11],
        from_node=[1, 2],
        to_node=[2, 3],
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
        i_n=1000.0,
    )
    data[CT.source] = rows(CT.source, id=[20, 22], node=[1, 3], status=1, u_ref=1.0)
    data[CT.sym_voltage_sensor] = rows(
        CT.sym_voltage_sensor,
        id=[31, 32, 33],
        measured_object=[1, 2, 3],
        u_sigma=100.0,
        u_measured=[10300.0, 9700.0, 10100.0],
        u_angle_measured=[0.0, np.nan, np.nan],
    )
    data[CT.sym_power_sensor] = rows(
        CT.sym_power_sensor,
        id=[41, 42],
        measured_object=[21, 22],
        measured_terminal_type=[4, 2],
        power_sigma=1e6,
        p_measured=[1e5, 1e5],
        q_measured=[1e4, 1e4],
    )
    data[CT.voltage_uncertainty_observer] = rows(
        CT.voltage_uncertainty_observer, id=[101, 102, 103], observed_object=[1, 2, 3], status=1
    )
    data[CT.current_uncertainty_observer]["observed_object"] = 11
    del data[CT.power_uncertainty_observer]
    model = PowerGridModel(data)
    update = initialize_array(DT.update, CT.line, (3, 1))
    update["id"] = 10
    update["to_status"][:, 0] = [1, 0, 1]
    result = model.calculate_state_estimation(
        calculation_method=method,
        max_iterations=100,
        update_data={CT.line: update},
        output_component_types=[CT.node, CT.voltage_uncertainty_observer, CT.current_uncertainty_observer],
    )
    sigma = result[CT.voltage_uncertainty_observer]["u_angle_sigma"]
    assert sigma[0, 2] > 0.0
    assert sigma[1, 2] == 0.0
    np.testing.assert_array_equal(sigma[0], sigma[2])
    np.testing.assert_array_equal(
        result[CT.current_uncertainty_observer]["i_angle_sigma"][0],
        result[CT.current_uncertainty_observer]["i_angle_sigma"][2],
    )
    assert np.isfinite(result[CT.current_uncertainty_observer]["i_angle_sigma"]).all()


@pytest.mark.parametrize("method", METHODS)
def test_multiple_islands_keep_interleaved_observer_requests_and_result_rows_aligned(method):
    first = network()
    second = deepcopy(first)
    identifier_fields = {"id", "node", "from_node", "to_node", "measured_object", "observed_object"}
    for values in second.values():
        for field in identifier_fields.intersection(values.dtype.names):
            values[field] += 1000
    second[CT.sym_voltage_sensor]["u_sigma"] *= 1.7
    second[CT.sym_voltage_sensor]["u_measured"] *= 0.98
    second[CT.sym_voltage_sensor]["u_angle_measured"] += 0.3
    combined = {component: np.concatenate([first[component], second[component]]) for component in first}
    permutation = [3, 0, 2, 1]
    for component in OBSERVERS:
        combined[component] = combined[component][permutation]
    actual = PowerGridModel(combined).calculate_state_estimation(
        calculation_method=method, output_component_types=OBSERVERS
    )
    independent = [
        PowerGridModel(data).calculate_state_estimation(calculation_method=method, output_component_types=OBSERVERS)
        for data in [first, second]
    ]
    for component in OBSERVERS:
        expected = np.concatenate([result[component] for result in independent])[permutation]
        for field in expected.dtype.names:
            np.testing.assert_allclose(actual[component][field], expected[field], rtol=1e-12)


def _loaded_ternary_network():
    """Reproduce the 100-bus runtime smoke input without external fixture files.

    Every nonroot bus has an unsensed load, so these injections must not be
    mistaken for exact zeros. Four voltage sensors and branch-to P/Q sensors
    provide the modeled noise. Only the root voltage has a physical angle.
    """
    count = 100
    children = np.arange(1, count)
    line_ids = np.arange(count, 2 * count - 1)
    load_scale = 1.0 + 0.15 * np.sin(0.37 * children)
    data = {
        CT.node: rows(CT.node, count, id=np.arange(count), u_rated=20_000.0),
        CT.line: rows(
            CT.line,
            count - 1,
            id=line_ids,
            from_node=(children - 1) // 3,
            to_node=children,
            from_status=1,
            to_status=1,
            r1=0.04,
            x1=0.08,
            c1=0.0,
            tan1=0.0,
            r0=0.04,
            x0=0.08,
            c0=0.0,
            tan0=0.0,
            i_n=1e9,
        ),
        CT.sym_load: rows(
            CT.sym_load,
            count - 1,
            id=np.arange(2 * count - 1, 3 * count - 2),
            node=children,
            status=1,
            type=LoadGenType.const_power,
            p_specified=1000.0 * load_scale,
            q_specified=300.0 * load_scale,
        ),
        CT.source: rows(CT.source, 1, id=3 * count - 2, node=0, status=1, u_ref=1.0, u_ref_angle=0.0),
    }
    power_flow = PowerGridModel(data, system_frequency=50.0).calculate_power_flow(
        symmetric=True,
        calculation_method=CM.newton_raphson,
        error_tolerance=1e-8,
        max_iterations=100,
    )
    measured_buses = np.linspace(0, count - 1, 4, dtype=int)
    angles = np.full(4, np.nan)
    angles[0] = power_flow[CT.node]["u_angle"][0]
    next_id = 3 * count - 1
    data[CT.sym_voltage_sensor] = rows(
        CT.sym_voltage_sensor,
        4,
        id=np.arange(next_id, next_id + 4),
        measured_object=measured_buses,
        u_sigma=30.0,
        u_measured=power_flow[CT.node]["u"][measured_buses],
        u_angle_measured=angles,
    )
    next_id += 4
    data[CT.sym_power_sensor] = rows(
        CT.sym_power_sensor,
        count - 1,
        id=np.arange(next_id, next_id + count - 1),
        measured_object=line_ids,
        measured_terminal_type=MeasuredTerminalType.branch_to,
        p_sigma=25_000.0,
        q_sigma=10_000.0,
        p_measured=power_flow[CT.line]["p_to"],
        q_measured=power_flow[CT.line]["q_to"],
    )
    next_id += count - 1
    branch_terminals = np.repeat([ObservedTerminalType.branch_from, ObservedTerminalType.branch_to], count - 1)
    observer_targets = (
        (CT.voltage_uncertainty_observer, np.arange(count), None),
        (
            CT.power_uncertainty_observer,
            np.concatenate([np.arange(count), line_ids, line_ids]),
            np.concatenate([np.full(count, ObservedTerminalType.node_injection), branch_terminals]),
        ),
        (CT.current_uncertainty_observer, np.tile(line_ids, 2), branch_terminals),
    )
    for component, targets, terminals in observer_targets:
        observer = rows(
            component,
            len(targets),
            id=np.arange(next_id, next_id + len(targets)),
            observed_object=targets,
            status=1,
        )
        if terminals is not None:
            observer["observed_terminal_type"] = terminals
        data[component] = observer
        next_id += len(targets)
    return data


def _nrse_voltage_sigma_reference(data, nodes):
    """Invert an independent physical measurement Jacobian at the returned state.

    Coordinates are interleaved angle in radians and voltage magnitude in per
    unit. Balanced three-phase total power in line-to-line voltage coordinates
    is S_to = U_to * conj((U_to - U_from) / Z); no sqrt(3) remains in this map.
    The root angle is a physical measurement, and all injections are unsensed,
    so this fixture requires neither virtual-angle correction nor constraints.
    """
    count = len(nodes)
    node_index = {node_id: index for index, node_id in enumerate(nodes["id"])}
    rated = data[CT.node]["u_rated"]
    phase = np.exp(1j * nodes["u_angle"])
    voltage = nodes["u"] * phase
    weighted_rows = []
    for sensor in data[CT.sym_voltage_sensor]:
        bus = node_index[sensor["measured_object"]]
        magnitude_row = np.zeros(2 * count)
        magnitude_row[2 * bus + 1] = rated[bus] / sensor["u_sigma"]
        weighted_rows.append(magnitude_row)
        if np.isfinite(sensor["u_angle_measured"]):
            angle_row = np.zeros(2 * count)
            # The ordinary NRSE model gives the physical angle standard
            # deviation u_sigma/u_rated, independently of magnitude noise.
            angle_row[2 * bus] = rated[bus] / sensor["u_sigma"]
            weighted_rows.append(angle_row)

    lines = {line["id"]: line for line in data[CT.line]}
    for sensor in data[CT.sym_power_sensor]:
        line = lines[sensor["measured_object"]]
        from_bus = node_index[line["from_node"]]
        to_bus = node_index[line["to_node"]]
        impedance = complex(line["r1"], line["x1"])
        current_ll = (voltage[to_bus] - voltage[from_bus]) / impedance
        power_derivative = np.zeros(2 * count, dtype=complex)
        for bus, sign in ((to_bus, 1.0), (from_bus, -1.0)):
            voltage_derivative = np.array([1j * voltage[bus], rated[bus] * phase[bus]])
            derivative = voltage[to_bus] * np.conj(sign * voltage_derivative / impedance)
            if bus == to_bus:
                derivative += voltage_derivative * current_ll.conjugate()
            power_derivative[2 * bus : 2 * bus + 2] = derivative
        weighted_rows.append(power_derivative.real / sensor["p_sigma"])
        weighted_rows.append(power_derivative.imag / sensor["q_sigma"])

    jacobian = np.asarray(weighted_rows)
    covariance = np.linalg.inv(jacobian.T @ jacobian)
    sigma = np.sqrt(np.diag(covariance).reshape(count, 2))
    # Convert only magnitude coordinates back to physical line-to-line volts.
    return sigma[:, 1] * rated, sigma[:, 0]


@pytest.mark.parametrize("method", [CM.iterative_linear, CM.newton_raphson])
def test_loaded_ternary_observers_are_finite_without_changing_estimates(method):
    data = _loaded_ternary_network()
    model = PowerGridModel(data, system_frequency=50.0)
    options = dict(symmetric=True, calculation_method=method, error_tolerance=1e-8, max_iterations=100, threading=1)
    ordinary = model.calculate_state_estimation(**options, output_component_types=[CT.node, CT.line])
    fields = {
        CT.voltage_uncertainty_observer: ("u_sigma", "u_angle_sigma"),
        CT.power_uncertainty_observer: ("p_sigma", "q_sigma"),
        CT.current_uncertainty_observer: ("i_sigma", "i_angle_sigma"),
    }
    result = model.calculate_state_estimation(**options, output_component_types=[CT.node, CT.line, *fields])
    for component in [CT.node, CT.line]:
        np.testing.assert_array_equal(result[component], ordinary[component])
        assert np.all(result[component]["energized"] == 1)
    finite_count = 0
    for component, attributes in fields.items():
        observer = result[component]
        np.testing.assert_array_equal(observer["id"], data[component]["id"])
        assert np.all(observer["energized"] == 1)
        for attribute in attributes:
            assert np.all(np.isfinite(observer[attribute])), (component, attribute)
            assert np.all(observer[attribute] >= 0.0), (component, attribute)
            finite_count += observer[attribute].size
    # 2 * (100 voltages + 100 injections + 198 branch powers + 198 currents).
    expected_sigma_count = 1192
    assert finite_count == expected_sigma_count
    # Four 30 V magnitude measurements constrain this almost flat radial network.
    # This broad physical bound catches a lost/noisy result without freezing a
    # floating-point golden value from a particular native build.
    assert 0.0 < result[CT.voltage_uncertainty_observer]["u_sigma"][0] < data[CT.sym_voltage_sensor]["u_sigma"][0]
    if method == CM.newton_raphson:
        magnitude_sigma, angle_sigma = _nrse_voltage_sigma_reference(data, result[CT.node])
        observer = result[CT.voltage_uncertainty_observer]
        np.testing.assert_allclose(observer["u_sigma"], magnitude_sigma, rtol=1e-7, atol=0.0)
        np.testing.assert_allclose(observer["u_angle_sigma"], angle_sigma, rtol=1e-7, atol=0.0)
