# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from power_grid_model import CalculationMethod, DatasetType, PowerGridModel, initialize_array
from power_grid_model._core.dataset_definitions import AttributeType as AT, ComponentType as CT
from power_grid_model._core.power_grid_meta import power_grid_meta_data
from power_grid_model.data_types import Dataset
from power_grid_model.utils import (
    LICENSE_TEXT,
    _make_test_case,
    create_state_estimation_monte_carlo_updates,
    get_component_batch_size,
    get_dataset_batch_size,
    get_dataset_scenario,
    json_deserialize_from_file,
    json_serialize_to_file,
    msgpack_deserialize_from_file,
    msgpack_serialize_to_file,
    self_test,
)

from .utils import DATA_PATH, import_case_data


def test_get_dataset_scenario():
    data = {
        "foo": np.array([["bar", "baz"], ["foobar", "foobaz"]]),
        "hi": {
            "data": np.array(["hello", "hey"]),
            "indptr": np.array([0, 0, 2]),
        },
    }
    result = get_dataset_scenario(data, 0)
    assert result.keys() == data.keys()
    np.testing.assert_array_equal(result["foo"], data["foo"][0])
    np.testing.assert_array_equal(result["hi"], data["hi"]["data"][0:0])

    result = get_dataset_scenario(data, 1)
    assert result.keys() == data.keys()
    np.testing.assert_array_equal(result["foo"], data["foo"][1])
    np.testing.assert_array_equal(result["hi"], data["hi"]["data"][0:2])

    with pytest.raises(IndexError):
        get_dataset_scenario(data, 2)


@pytest.mark.parametrize("n_samples", [0, -1, 1.5, True])
def test_create_state_estimation_monte_carlo_updates_rejects_invalid_sample_count(n_samples):
    with pytest.raises(ValueError, match="positive integer"):
        create_state_estimation_monte_carlo_updates({}, n_samples)


def test_create_state_estimation_monte_carlo_updates_requires_sensors():
    with pytest.raises(ValueError, match="no state-estimation sensors"):
        create_state_estimation_monte_carlo_updates({}, 2)


def test_create_state_estimation_monte_carlo_updates_samples_voltage_magnitude_and_angle():
    node = initialize_array(DatasetType.input, CT.node, 2)
    node[AT.id] = [1, 2]
    node[AT.u_rated] = [10_000.0, 20_000.0]

    sensor = initialize_array(DatasetType.input, CT.asym_voltage_sensor, 2)
    sensor[AT.id] = [11, 12]
    sensor[AT.measured_object] = [1, 2]
    sensor[AT.u_measured] = [[5_700.0, 5_800.0, 5_900.0], [11_400.0, 11_500.0, 11_600.0]]
    sensor[AT.u_angle_measured] = [[0.0, -2.0, 2.0], [np.nan, np.nan, np.nan]]
    sensor[AT.u_sigma] = [30.0, 60.0]

    n_samples = 3
    seed = 7
    updates = create_state_estimation_monte_carlo_updates(
        {CT.node: node, CT.asym_voltage_sensor: sensor}, n_samples, seed=seed
    )
    voltage_updates = updates[CT.asym_voltage_sensor]

    rng = np.random.default_rng(seed)
    expected_magnitude = rng.normal(
        loc=sensor[AT.u_measured],
        scale=sensor[AT.u_sigma][:, np.newaxis],
        size=(n_samples, *sensor[AT.u_measured].shape),
    )
    angle_sigma = sensor[AT.u_sigma] / (node[AT.u_rated] / np.sqrt(3.0))
    expected_angle = rng.normal(
        loc=sensor[AT.u_angle_measured],
        scale=angle_sigma[:, np.newaxis],
        size=(n_samples, *sensor[AT.u_angle_measured].shape),
    )

    np.testing.assert_allclose(voltage_updates[AT.u_measured], expected_magnitude)
    np.testing.assert_allclose(voltage_updates[AT.u_angle_measured], expected_angle, equal_nan=True)
    assert np.all(np.isnan(voltage_updates[AT.u_angle_measured][:, 1]))


def test_create_state_estimation_monte_carlo_updates_supports_columnar_voltage_data_without_angle():
    node = initialize_array(DatasetType.input, CT.node, 1)
    node[AT.id] = [1]
    node[AT.u_rated] = [10_000.0]
    sensor = initialize_array(DatasetType.input, CT.sym_voltage_sensor, 1)
    sensor[AT.measured_object] = [1]
    sensor[AT.u_measured] = [10_100.0]
    sensor[AT.u_sigma] = [20.0]
    columnar_input = {
        CT.node: {attribute: node[attribute] for attribute in node.dtype.names},
        CT.sym_voltage_sensor: {attribute: sensor[attribute] for attribute in sensor.dtype.names},
    }

    updates = create_state_estimation_monte_carlo_updates(columnar_input, 2, seed=3)

    voltage_updates = updates[CT.sym_voltage_sensor]
    assert voltage_updates.shape == (2, 1)
    assert np.all(np.isfinite(voltage_updates[AT.u_measured]))
    assert np.all(np.isnan(voltage_updates[AT.u_angle_measured]))


def test_create_state_estimation_monte_carlo_updates_requires_nodes_for_voltage_angles():
    sensor = initialize_array(DatasetType.input, CT.sym_voltage_sensor, 1)
    sensor[AT.u_measured] = [10_000.0]
    sensor[AT.u_angle_measured] = [0.0]
    sensor[AT.u_sigma] = [10.0]

    with pytest.raises(ValueError, match="requires node"):
        create_state_estimation_monte_carlo_updates({CT.sym_voltage_sensor: sensor}, 1)


def test_create_state_estimation_monte_carlo_updates_rejects_unknown_voltage_node():
    node = initialize_array(DatasetType.input, CT.node, 1)
    node[AT.id] = [1]
    node[AT.u_rated] = [10_000.0]
    sensor = initialize_array(DatasetType.input, CT.sym_voltage_sensor, 1)
    sensor[AT.measured_object] = [99]
    sensor[AT.u_measured] = [10_000.0]
    sensor[AT.u_angle_measured] = [0.0]
    sensor[AT.u_sigma] = [10.0]

    with pytest.raises(ValueError, match="unknown node 99"):
        create_state_estimation_monte_carlo_updates({CT.node: node, CT.sym_voltage_sensor: sensor}, 1)


def test_create_state_estimation_monte_carlo_updates_samples_power_sigma_variants():
    sym_sensor = initialize_array(DatasetType.input, CT.sym_power_sensor, 2)
    sym_sensor[AT.p_measured] = [100.0, 200.0]
    sym_sensor[AT.q_measured] = [10.0, 20.0]
    sym_sensor[AT.power_sigma] = [np.nan, 14.0]
    sym_sensor[AT.p_sigma] = [2.0, np.nan]
    sym_sensor[AT.q_sigma] = [3.0, np.nan]

    asym_sensor = initialize_array(DatasetType.input, CT.asym_power_sensor, 1)
    asym_sensor[AT.p_measured] = [[1.0, 2.0, 3.0]]
    asym_sensor[AT.q_measured] = [[4.0, 5.0, 6.0]]
    asym_sensor[AT.power_sigma] = [np.sqrt(2.0)]

    n_samples = 2
    seed = 9
    updates = create_state_estimation_monte_carlo_updates(
        {CT.sym_power_sensor: sym_sensor, CT.asym_power_sensor: asym_sensor}, n_samples, seed=seed
    )

    rng = np.random.default_rng(seed)
    expected_sym_p = rng.normal(sym_sensor[AT.p_measured], [2.0, 14.0 / np.sqrt(2.0)], (n_samples, 2))
    expected_sym_q = rng.normal(sym_sensor[AT.q_measured], [3.0, 14.0 / np.sqrt(2.0)], (n_samples, 2))
    expected_asym_p = rng.normal(asym_sensor[AT.p_measured], 1.0, (n_samples, 1, 3))
    expected_asym_q = rng.normal(asym_sensor[AT.q_measured], 1.0, (n_samples, 1, 3))

    np.testing.assert_allclose(updates[CT.sym_power_sensor][AT.p_measured], expected_sym_p)
    np.testing.assert_allclose(updates[CT.sym_power_sensor][AT.q_measured], expected_sym_q)
    np.testing.assert_allclose(updates[CT.asym_power_sensor][AT.p_measured], expected_asym_p)
    np.testing.assert_allclose(updates[CT.asym_power_sensor][AT.q_measured], expected_asym_q)


def test_create_state_estimation_monte_carlo_updates_samples_current_polar_coordinates():
    sensor = initialize_array(DatasetType.input, CT.asym_current_sensor, 1)
    sensor[AT.i_measured] = [[10.0, 20.0, 30.0]]
    sensor[AT.i_angle_measured] = [[0.1, 0.2, 0.3]]
    sensor[AT.i_sigma] = [0.5]
    sensor[AT.i_angle_sigma] = [0.01]

    n_samples = 2
    seed = 11
    updates = create_state_estimation_monte_carlo_updates({CT.asym_current_sensor: sensor}, n_samples, seed=seed)

    rng = np.random.default_rng(seed)
    expected_magnitude = rng.normal(sensor[AT.i_measured], 0.5, (n_samples, 1, 3))
    expected_angle = rng.normal(sensor[AT.i_angle_measured], 0.01, (n_samples, 1, 3))
    current_updates = updates[CT.asym_current_sensor]
    np.testing.assert_allclose(current_updates[AT.i_measured], expected_magnitude)
    np.testing.assert_allclose(current_updates[AT.i_angle_measured], expected_angle)


@pytest.mark.parametrize("calculation_method", [CalculationMethod.iterative_linear, CalculationMethod.newton_raphson])
def test_state_estimation_accepts_monte_carlo_updates(calculation_method):
    input_data = import_case_data(
        DATA_PATH / "state_estimation" / "1os2msr", calculation_type="state_estimation", sym=True
    )[DatasetType.input]
    n_samples = 4
    updates = create_state_estimation_monte_carlo_updates(input_data, n_samples, seed=42)

    result = PowerGridModel(input_data).calculate_state_estimation(
        calculation_method=calculation_method,
        update_data=updates,
    )

    assert result[CT.node].shape[0] == n_samples
    assert np.all(np.isfinite(result[CT.node][AT.u]))


def test_get_data_set_batch_size():
    line = initialize_array(DatasetType.update, CT.line, (3, 2))
    line[AT.id] = [[5, 6], [6, 7], [7, 5]]
    line[AT.from_status] = [[1, 1], [1, 1], [1, 1]]

    asym_load = initialize_array(DatasetType.update, CT.asym_load, (3, 2))
    asym_load[AT.id] = [[9, 10], [9, 10], [9, 10]]

    batch_data = {CT.line: line, CT.asym_load: asym_load}

    n_batch_size = 3

    assert get_dataset_batch_size(batch_data) == n_batch_size


def test_get_dataset_batch_size_sparse():
    data = {
        CT.node: {
            "data": np.zeros(shape=3, dtype=power_grid_meta_data[DatasetType.input][CT.node]),
            "indptr": np.array([0, 2, 3, 3]),
        },
        CT.sym_load: {
            "data": np.zeros(shape=2, dtype=power_grid_meta_data[DatasetType.input][CT.sym_load]),
            "indptr": np.array([0, 0, 1, 2]),
        },
        CT.asym_load: {
            "data": np.zeros(shape=4, dtype=power_grid_meta_data[DatasetType.input][CT.asym_load]),
            "indptr": np.array([0, 2, 3, 4]),
        },
    }

    n_batch_size = 3

    assert get_dataset_batch_size(data) == n_batch_size


def test_get_dataset_batch_size_mixed():
    line = initialize_array(DatasetType.update, CT.line, (3, 2))
    line[AT.id] = [[5, 6], [6, 7], [7, 5]]
    line[AT.from_status] = [[1, 1], [1, 1], [1, 1]]

    asym_load = initialize_array(DatasetType.update, CT.asym_load, (2, 2))
    asym_load[AT.id] = [[9, 10], [9, 10]]

    data_dense = {CT.line: line, CT.asym_load: asym_load}
    data_sparse = {
        CT.node: {
            "data": np.zeros(shape=3, dtype=power_grid_meta_data[DatasetType.input][CT.node]),
            "indptr": np.array([0, 2, 3, 3, 5]),
        },
        CT.sym_load: {
            "data": np.zeros(shape=2, dtype=power_grid_meta_data[DatasetType.input][CT.sym_load]),
            "indptr": np.array([0, 0, 1, 2]),
        },
        CT.asym_load: {
            "data": np.zeros(shape=4, dtype=power_grid_meta_data[DatasetType.input][CT.asym_load]),
            "indptr": np.array([0, 2, 3]),
        },
    }
    with pytest.raises(ValueError, match="Inconsistent number of batches in batch data"):
        get_dataset_batch_size(data_dense)
    with pytest.raises(ValueError, match="Inconsistent number of batches in batch data"):
        get_dataset_batch_size(data_sparse)


def test_get_component_batch_size():
    asym_load = initialize_array(DatasetType.update, CT.asym_load, (3, 2))
    asym_load[AT.id] = [[9, 10], [9, 10], [9, 10]]

    sym_load = {
        "data": np.zeros(shape=2, dtype=power_grid_meta_data[DatasetType.input][CT.sym_load]),
        "indptr": np.array([0, 0, 1, 2]),
    }

    asym_load_batch_size = 3
    sym_load_batch_size = 3
    assert get_component_batch_size(asym_load) == asym_load_batch_size
    assert get_component_batch_size(sym_load) == sym_load_batch_size


@patch("pathlib.Path.open", new_callable=mock_open)
@patch("power_grid_model.utils.json_deserialize")
def test_json_deserialize_from_file(deserialize_mock: MagicMock, open_mock: MagicMock):
    handle = open_mock()
    handle.read.return_value = '{"version": "1.0", "data": {"foo": [{"val": 123}]}, "bar": {"baz": 456}}'
    deserialize_mock.return_value = {"foo": [{"val": 123}]}
    assert json_deserialize_from_file(file_path=Path("output.json")) == deserialize_mock.return_value
    handle.read.assert_called_once()
    deserialize_mock.assert_called_once_with(handle.read.return_value, data_filter=None)


@patch("pathlib.Path.open", new_callable=mock_open)
@patch("power_grid_model.utils.json_serialize")
def test_json_serialize(serialize_mock: MagicMock, open_mock: MagicMock):
    serialize_mock.return_value = '{"version": "1.0", "data": {"foo": [{"val": 123}]}, "bar": {"baz": 456}}'
    data: Dataset = {}
    json_serialize_to_file(file_path=Path("output.json"), data=data, use_compact_list=False, indent=2)
    serialize_mock.assert_called_once_with(data=data, dataset_type=None, use_compact_list=False, indent=2)
    handle = open_mock()
    handle.write.assert_called_once_with(serialize_mock.return_value)


@patch("pathlib.Path.open", new_callable=mock_open)
@patch("power_grid_model.utils.msgpack_deserialize")
def test_msgpack_deserialize_from_file(deserialize_mock: MagicMock, open_mock: MagicMock):
    handle = open_mock()
    handle.read.return_value = b'{"version": "1.0", "data": {"foo": [{"val": 123}]}, "bar": {"baz": 456}}'
    deserialize_mock.return_value = {"foo": [{"val": 123}]}
    assert msgpack_deserialize_from_file(file_path=Path("output.msgpack")) == deserialize_mock.return_value
    handle.read.assert_called_once()
    deserialize_mock.assert_called_once_with(handle.read.return_value, data_filter=None)


@patch("pathlib.Path.open", new_callable=mock_open)
@patch("power_grid_model.utils.msgpack_serialize")
def test_msgpack_serialize(serialize_mock: MagicMock, open_mock: MagicMock):
    serialize_mock.return_value = b'{"version": "1.0", "data": {"foo": [{"val": 123}]}, "bar": {"baz": 456}}'
    data: Dataset = {}
    msgpack_serialize_to_file(file_path=Path("output.msgpack"), data=data, use_compact_list=False)
    serialize_mock.assert_called_once_with(data=data, dataset_type=None, use_compact_list=False)
    handle = open_mock()
    handle.write.assert_called_once_with(serialize_mock.return_value)


def test_self_test():
    self_test()


@pytest.mark.parametrize(
    ("output_dataset_type", "output_file_name", "update_data"),
    [
        (DatasetType.sym_output, "sym_output_batch.json", {"version": "1.0", "data": "update_data"}),
        (DatasetType.sym_output, "sym_output.json", None),
        (DatasetType.asym_output, "asym_output_batch.json", {"version": "1.0", "data": "update_data"}),
        (DatasetType.asym_output, "asym_output.json", None),
        (DatasetType.sc_output, "sc_output_batch.json", {"version": "1.0", "data": "update_data"}),
        (DatasetType.sc_output, "sc_output.json", None),
    ],
)
@patch.object(Path, "write_text", autospec=True)
@patch("power_grid_model.utils.json_serialize_to_file")
def test__make_test_case(
    serialize_to_file_mock: MagicMock, write_text_mock: MagicMock, output_dataset_type, output_file_name, update_data
):
    input_data: Dataset = {"version": "1.0", "data": "input_data"}
    output_data: Dataset = {"version": "1.0", "data": "output_data"}
    output_path = Path("test_path")
    params = {"param1": "value1", "param2": "value2"}
    write_update_call_count = 5
    serialize_update_call_count = 3
    write_call_count = 4
    serialize_call_count = 2

    _make_test_case(
        output_path=output_path,
        input_data=input_data,
        output_data=output_data,
        params=params,
        output_dataset_type=output_dataset_type,
        update_data=update_data,
    )

    serialize_to_file_mock.assert_any_call(
        file_path=output_path / "input.json", data=input_data, dataset_type=DatasetType.input
    )
    serialize_to_file_mock.assert_any_call(
        file_path=output_path / output_file_name, data=output_data, dataset_type=output_dataset_type
    )
    write_text_mock.assert_any_call(output_path / "params.json", data=json.dumps(params, indent=2), encoding="utf-8")
    for file_name in ["input.json.license", f"{output_file_name}.license", "params.json.license"]:
        write_text_mock.assert_any_call(output_path / file_name, data=LICENSE_TEXT, encoding="utf-8")
    if update_data is not None:
        write_text_mock.assert_any_call(output_path / "update_batch.json.license", data=LICENSE_TEXT, encoding="utf-8")
        serialize_to_file_mock.assert_any_call(
            file_path=output_path / "update_batch.json", data=update_data, dataset_type=DatasetType.update
        )
        assert write_text_mock.call_count == write_update_call_count
        assert serialize_to_file_mock.call_count == serialize_update_call_count
    else:
        assert write_text_mock.call_count == write_call_count
        assert serialize_to_file_mock.call_count == serialize_call_count
