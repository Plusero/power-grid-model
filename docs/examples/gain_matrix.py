# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Conventional dense gain-matrix helpers for the symmetric SE UQ example.

The public entry point, :func:`build_gain_matrix`, accepts the same row-based
input dataset as :meth:`power_grid_model.PowerGridModel.calculate_state_estimation`.
It obtains a converged Newton--Raphson operating point, analytically constructs
the real polar measurement Jacobian ``H`` and diagonal weight matrix ``W``, and
returns ``G = H.T @ W @ H``.

This example helper intentionally lives outside ``src/power_grid_model``. It
requires the connected line network, voltage sensors, and line-terminal power
sensors used by the synthetic scaling notebook. Exact zero-injection
constraints and other branch or sensor types require an augmented formulation
and are rejected explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from power_grid_model import (
    AttributeType,
    CalculationMethod,
    ComponentType,
    MeasuredTerminalType,
    PowerGridModel,
)

_BASE_POWER_3P = 1.0e6
_DEFAULT_MAX_GAIN_MATRIX_BYTES = 256 * 1024**2


def _check_dense_allocation(shape: tuple[int, int], max_dense_matrix_bytes: int | None, matrix_name: str) -> None:
    required_bytes = int(shape[0]) * int(shape[1]) * np.dtype(np.float64).itemsize
    if max_dense_matrix_bytes is not None and required_bytes > max_dense_matrix_bytes:
        raise MemoryError(
            f"Dense {matrix_name} needs {required_bytes / 1024**3:.2f} GiB, "
            f"above the {max_dense_matrix_bytes / 1024**3:.2f} GiB allocation limit."
        )


@dataclass(frozen=True)
class CsrMatrix:
    """Small NumPy-only CSR container used to avoid a SciPy dependency."""

    data: np.ndarray
    column_indices: np.ndarray
    row_pointers: np.ndarray
    shape: tuple[int, int]

    @property
    def nnz(self) -> int:
        """Return the number of explicitly stored entries."""
        return int(self.data.size)

    def to_dense(self, *, max_dense_matrix_bytes: int | None = _DEFAULT_MAX_GAIN_MATRIX_BYTES) -> np.ndarray:
        """Materialize the matrix with a small-case allocation guard."""
        _check_dense_allocation(self.shape, max_dense_matrix_bytes, "measurement matrix")
        dense = np.zeros(self.shape, dtype=float)
        row_indices = np.repeat(np.arange(self.shape[0]), np.diff(self.row_pointers))
        np.add.at(dense, (row_indices, self.column_indices), self.data)
        return dense

    def weighted_gram(
        self,
        weights: np.ndarray,
        *,
        max_dense_matrix_bytes: int | None = _DEFAULT_MAX_GAIN_MATRIX_BYTES,
    ) -> np.ndarray:
        """Return ``self.T @ diag(weights) @ self`` without dense H or W."""
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.shape[0],):
            raise ValueError(f"Expected {self.shape[0]} diagonal weights, got {weights.shape}.")

        _check_dense_allocation((self.shape[1], self.shape[1]), max_dense_matrix_bytes, "gain matrix")
        gain_matrix = np.zeros((self.shape[1], self.shape[1]), dtype=float)
        for row, weight in enumerate(weights):
            start = self.row_pointers[row]
            stop = self.row_pointers[row + 1]
            columns = self.column_indices[start:stop]
            values = self.data[start:stop]
            gain_matrix[np.ix_(columns, columns)] += weight * np.outer(values, values)
        return gain_matrix


@dataclass(frozen=True)
class DiagonalMatrix:
    """Diagonal matrix represented by its diagonal entries."""

    diagonal: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """Return the square matrix shape."""
        size = int(self.diagonal.size)
        return size, size

    def to_dense(self, *, max_dense_matrix_bytes: int | None = _DEFAULT_MAX_GAIN_MATRIX_BYTES) -> np.ndarray:
        """Materialize the matrix with a small-case allocation guard."""
        _check_dense_allocation(self.shape, max_dense_matrix_bytes, "weight matrix")
        return np.diag(self.diagonal)


@dataclass(frozen=True)
class MeasurementModel:
    """Analytical real-polar measurement model at a converged NRSE state."""

    measurement_matrix: CsrMatrix
    weight_matrix: DiagonalMatrix
    state_vector: np.ndarray
    node_ids: np.ndarray
    measurement_labels: tuple[str, ...]
    fixed_angle_reference_node_id: int | None


@dataclass(frozen=True)
class GainMatrixModel:
    """Measurement model and its conventional dense gain matrix."""

    measurement_model: MeasurementModel
    gain_matrix: np.ndarray


def _component(dataset: Mapping[Any, Any], component_type: ComponentType, *, required: bool = True) -> Any:
    for key in (component_type, component_type.value):
        if key in dataset:
            return dataset[key]
    if required:
        raise ValueError(f"Input data is missing required component {component_type.value!r}.")
    return None


def _field(component_data: Any, attribute: AttributeType) -> np.ndarray:
    if isinstance(component_data, Mapping):
        for key in (attribute, attribute.value):
            if key in component_data:
                return np.asarray(component_data[key])
        raise ValueError(f"Component data is missing required attribute {attribute.value!r}.")
    return np.asarray(component_data[attribute])


def _optional_field(component_data: Any, attribute: AttributeType, size: int) -> np.ndarray:
    try:
        return _field(component_data, attribute)
    except (KeyError, ValueError, IndexError):
        return np.full(size, np.nan)


def _operating_point(
    input_data: Mapping[Any, Any],
    *,
    system_frequency: float,
    error_tolerance: float,
    max_iterations: int,
    state_estimation_result: Mapping[Any, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if state_estimation_result is None:
        state_estimation_result = PowerGridModel(
            dict(input_data), system_frequency=system_frequency
        ).calculate_state_estimation(
            symmetric=True,
            calculation_method=CalculationMethod.newton_raphson,
            error_tolerance=error_tolerance,
            max_iterations=max_iterations,
        )

    input_nodes = _component(input_data, ComponentType.node)
    output_nodes = _component(state_estimation_result, ComponentType.node)
    node_ids = _field(input_nodes, AttributeType.id).astype(np.int64, copy=False)
    output_ids = _field(output_nodes, AttributeType.id).astype(np.int64, copy=False)
    output_by_id = {int(node_id): index for index, node_id in enumerate(output_ids)}
    try:
        output_indices = np.asarray([output_by_id[int(node_id)] for node_id in node_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"State-estimation output is missing node {error.args[0]}.") from error

    voltage_magnitudes = _field(output_nodes, AttributeType.u_pu)[output_indices].astype(float, copy=False)
    voltage_angles = _field(output_nodes, AttributeType.u_angle)[output_indices].astype(float, copy=False)
    if not np.all(np.isfinite(voltage_magnitudes)) or not np.all(np.isfinite(voltage_angles)):
        raise ValueError("The converged state-estimation operating point contains non-finite voltages.")
    return node_ids, voltage_magnitudes, voltage_angles


def _line_terminal_admittances(
    input_data: Mapping[Any, Any], system_frequency: float, node_index_by_id: Mapping[int, int]
) -> dict[int, tuple[int, int, complex, complex, complex, complex]]:
    lines = _component(input_data, ComponentType.line)
    nodes = _component(input_data, ComponentType.node)
    node_ids = _field(nodes, AttributeType.id).astype(np.int64, copy=False)
    rated_voltage_by_id = dict(
        zip(node_ids.tolist(), _field(nodes, AttributeType.u_rated).astype(float, copy=False).tolist(), strict=True)
    )

    line_ids = _field(lines, AttributeType.id).astype(np.int64, copy=False)
    from_nodes = _field(lines, AttributeType.from_node).astype(np.int64, copy=False)
    to_nodes = _field(lines, AttributeType.to_node).astype(np.int64, copy=False)
    from_status = _field(lines, AttributeType.from_status).astype(bool, copy=False)
    to_status = _field(lines, AttributeType.to_status).astype(bool, copy=False)
    resistance = _field(lines, AttributeType.r1).astype(float, copy=False)
    reactance = _field(lines, AttributeType.x1).astype(float, copy=False)
    capacitance = _field(lines, AttributeType.c1).astype(float, copy=False)
    loss_tangent = _field(lines, AttributeType.tan1).astype(float, copy=False)

    result: dict[int, tuple[int, int, complex, complex, complex, complex]] = {}
    for index, line_id in enumerate(line_ids):
        from_node_id = int(from_nodes[index])
        to_node_id = int(to_nodes[index])
        if not from_status[index] or not to_status[index]:
            raise NotImplementedError("The gain-matrix example supports only lines connected at both terminals.")
        try:
            from_index = node_index_by_id[from_node_id]
            to_index = node_index_by_id[to_node_id]
            from_rated_voltage = rated_voltage_by_id[from_node_id]
            to_rated_voltage = rated_voltage_by_id[to_node_id]
        except KeyError as error:
            raise ValueError(f"Line {int(line_id)} refers to unknown node {error.args[0]}.") from error
        if from_index == to_index:
            raise NotImplementedError("The gain-matrix example does not support lines connected to one node twice.")
        if not np.isclose(from_rated_voltage, to_rated_voltage):
            raise NotImplementedError("The gain-matrix example supports lines between equal rated voltages only.")

        series_impedance = resistance[index] + 1.0j * reactance[index]
        if series_impedance == 0.0:
            raise ValueError(f"Line {int(line_id)} has zero positive-sequence impedance.")
        base_admittance = _BASE_POWER_3P / from_rated_voltage**2
        series_admittance = 1.0 / series_impedance / base_admittance
        shunt_admittance = (
            2.0 * np.pi * system_frequency * capacitance[index] * (loss_tangent[index] + 1.0j) / base_admittance
        )
        self_admittance = series_admittance + 0.5 * shunt_admittance
        mutual_admittance = -series_admittance
        result[int(line_id)] = (
            from_index,
            to_index,
            self_admittance,
            mutual_admittance,
            mutual_admittance,
            self_admittance,
        )
    return result


def _power_sigmas(power_sensors: Any) -> tuple[np.ndarray, np.ndarray]:
    sensor_count = _field(power_sensors, AttributeType.id).size
    p_sigma = _optional_field(power_sensors, AttributeType.p_sigma, sensor_count).astype(float, copy=False)
    q_sigma = _optional_field(power_sensors, AttributeType.q_sigma, sensor_count).astype(float, copy=False)
    power_sigma = _optional_field(power_sensors, AttributeType.power_sigma, sensor_count).astype(float, copy=False)
    explicit = np.isfinite(p_sigma) & np.isfinite(q_sigma)
    component_sigma = power_sigma / np.sqrt(2.0)
    use_apparent_sigma = np.isnan(p_sigma)
    disabled_sigma = np.full(sensor_count, np.inf)
    return (
        np.where(explicit, p_sigma, np.where(use_apparent_sigma, component_sigma, disabled_sigma)),
        np.where(explicit, q_sigma, np.where(use_apparent_sigma, component_sigma, disabled_sigma)),
    )


def _validate_connected_network(
    node_count: int,
    line_admittances: Mapping[int, tuple[int, int, complex, complex, complex, complex]],
) -> None:
    """Reject multiple islands, whose angle references must be handled separately."""
    if node_count == 0:
        raise ValueError("The gain-matrix example requires at least one node.")
    neighbours: list[list[int]] = [[] for _ in range(node_count)]
    for from_index, to_index, *_ in line_admittances.values():
        neighbours[from_index].append(to_index)
        neighbours[to_index].append(from_index)

    visited = {0}
    pending = [0]
    while pending:
        node_index = pending.pop()
        for neighbour in neighbours[node_index]:
            if neighbour not in visited:
                visited.add(neighbour)
                pending.append(neighbour)
    if len(visited) != node_count:
        raise NotImplementedError(
            "The gain-matrix example supports one connected network only; each island needs a separate angle reference."
        )


def _validate_supported_input(input_data: Mapping[Any, Any]) -> None:
    unsupported_components = (
        ComponentType.asym_line,
        ComponentType.link,
        ComponentType.generic_branch,
        ComponentType.transformer,
        ComponentType.three_winding_transformer,
        ComponentType.asym_load,
        ComponentType.asym_gen,
        ComponentType.asym_voltage_sensor,
        ComponentType.asym_power_sensor,
        ComponentType.sym_current_sensor,
        ComponentType.asym_current_sensor,
    )
    for component_type in unsupported_components:
        component_data = _component(input_data, component_type, required=False)
        if component_data is not None and _field(component_data, AttributeType.id).size:
            raise NotImplementedError(f"The gain-matrix example does not support component {component_type.value!r}.")

    nodes = _component(input_data, ComponentType.node)
    node_ids = _field(nodes, AttributeType.id).astype(np.int64, copy=False)
    node_has_appliance = {int(node_id): False for node_id in node_ids}
    for component_type in (ComponentType.sym_load, ComponentType.sym_gen, ComponentType.source):
        component_data = _component(input_data, component_type, required=False)
        if component_data is None or not _field(component_data, AttributeType.id).size:
            continue
        component_nodes = _field(component_data, AttributeType.node).astype(np.int64, copy=False)
        component_status = _field(component_data, AttributeType.status).astype(bool, copy=False)
        for node_id, active in zip(component_nodes, component_status, strict=True):
            if active:
                if int(node_id) not in node_has_appliance:
                    raise ValueError(f"Component {component_type.value!r} refers to unknown node {int(node_id)}.")
                node_has_appliance[int(node_id)] = True
    exact_zero_injection_nodes = [node_id for node_id, has_appliance in node_has_appliance.items() if not has_appliance]
    if exact_zero_injection_nodes:
        raise NotImplementedError(
            "The conventional gain inverse does not support PGM's exact zero-injection constraints; "
            f"nodes without active appliances: {exact_zero_injection_nodes}."
        )


def _append_sparse_row(  # noqa: PLR0913, PLR0917
    rows: list[dict[int, float]],
    weights: list[float],
    labels: list[str],
    entries: Mapping[int, float],
    weight: float,
    label: str,
) -> None:
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"Measurement {label!r} has a non-positive or non-finite weight.")
    combined = {int(column): float(value) for column, value in entries.items() if value != 0.0}
    if not combined:
        raise ValueError(f"Measurement {label!r} has an empty analytical Jacobian row.")
    rows.append(combined)
    weights.append(float(weight))
    labels.append(label)


def _to_csr(rows: list[dict[int, float]], column_count: int) -> CsrMatrix:
    data: list[float] = []
    column_indices: list[int] = []
    row_pointers = [0]
    for row in rows:
        for column, value in sorted(row.items()):
            column_indices.append(column)
            data.append(value)
        row_pointers.append(len(data))
    return CsrMatrix(
        data=np.asarray(data, dtype=float),
        column_indices=np.asarray(column_indices, dtype=np.int64),
        row_pointers=np.asarray(row_pointers, dtype=np.int64),
        shape=(len(rows), column_count),
    )


def build_measurement_model(  # noqa: PLR0912, PLR0915
    input_data: Mapping[Any, Any],
    *,
    system_frequency: float = 50.0,
    error_tolerance: float = 1.0e-10,
    max_iterations: int = 100,
    state_estimation_result: Mapping[Any, Any] | None = None,
) -> MeasurementModel:
    """Build analytical ``H`` and diagonal ``W`` for the example's symmetric NRSE.

    With a physical angle measurement, the state ordering is
    ``[theta_0, ..., theta_(n-1), v_0, ..., v_(n-1)]``.  Without one, the
    active source's exact reference angle is removed from the angle block.
    Angles are in radians and magnitudes are per unit. Power measurements and
    their standard deviations are converted to the PGM three-phase 1 MVA base.
    """
    _validate_supported_input(input_data)
    nodes = _component(input_data, ComponentType.node)
    input_node_ids = _field(nodes, AttributeType.id).astype(np.int64, copy=False)
    input_node_index_by_id = {int(node_id): index for index, node_id in enumerate(input_node_ids)}
    line_admittances = _line_terminal_admittances(input_data, system_frequency, input_node_index_by_id)
    _validate_connected_network(input_node_ids.size, line_admittances)

    node_ids, voltage_magnitudes, voltage_angles = _operating_point(
        input_data,
        system_frequency=system_frequency,
        error_tolerance=error_tolerance,
        max_iterations=max_iterations,
        state_estimation_result=state_estimation_result,
    )
    node_count = node_ids.size
    node_index_by_id = {int(node_id): index for index, node_id in enumerate(node_ids)}
    if node_index_by_id != input_node_index_by_id:
        raise ValueError("State-estimation node ordering does not match the input data.")

    rows: list[dict[int, float]] = []
    weights: list[float] = []
    labels: list[str] = []

    voltage_sensors = _component(input_data, ComponentType.sym_voltage_sensor)
    sensor_node_ids = _field(voltage_sensors, AttributeType.measured_object).astype(np.int64, copy=False)
    voltage_sigmas = _field(voltage_sensors, AttributeType.u_sigma).astype(float, copy=False)
    voltage_angles_measured = _field(voltage_sensors, AttributeType.u_angle_measured).astype(float, copy=False)
    rated_voltage_by_id = dict(
        zip(
            _field(nodes, AttributeType.id).astype(np.int64, copy=False).tolist(),
            _field(nodes, AttributeType.u_rated).astype(float, copy=False).tolist(),
            strict=True,
        )
    )

    voltage_sensor_indices_by_node: dict[int, list[int]] = {}
    for sensor_index, node_id in enumerate(sensor_node_ids):
        voltage_sensor_indices_by_node.setdefault(int(node_id), []).append(sensor_index)

    has_angle_measurement = False
    for node_id, sensor_indices in voltage_sensor_indices_by_node.items():
        try:
            node_index = node_index_by_id[node_id]
            rated_voltage = rated_voltage_by_id[node_id]
        except KeyError as error:
            raise ValueError(f"Voltage sensor refers to unknown node {error.args[0]}.") from error
        normalized_sigmas = voltage_sigmas[sensor_indices] / rated_voltage
        invalid_sigmas = np.isnan(normalized_sigmas) | np.isneginf(normalized_sigmas) | (normalized_sigmas <= 0.0)
        if np.any(invalid_sigmas):
            raise ValueError(f"Voltage sensors at node {node_id} require positive u_sigma values.")
        finite_sigmas = normalized_sigmas[np.isfinite(normalized_sigmas)]
        if not finite_sigmas.size:
            continue
        combined_weight = float(np.sum(1.0 / np.square(finite_sigmas)))
        _append_sparse_row(
            rows,
            weights,
            labels,
            {node_count + node_index: 1.0},
            combined_weight,
            f"voltage magnitude at node {node_id}",
        )
        if np.all(np.isfinite(voltage_angles_measured[sensor_indices])):
            _append_sparse_row(
                rows,
                weights,
                labels,
                {node_index: 1.0},
                combined_weight,
                f"voltage angle at node {node_id}",
            )
            has_angle_measurement = True

    fixed_angle_reference_node_id: int | None = None
    if not has_angle_measurement:
        sources = _component(input_data, ComponentType.source, required=False)
        if sources is None:
            raise ValueError("An active source is required to define the exact angle reference.")
        source_nodes = _field(sources, AttributeType.node).astype(np.int64, copy=False)
        source_status = _field(sources, AttributeType.status).astype(bool, copy=False)
        active_source_nodes = [
            int(node_id) for node_id, active in zip(source_nodes, source_status, strict=True) if active
        ]
        if len(active_source_nodes) != 1:
            raise NotImplementedError(
                "Without a physical angle measurement, the gain-matrix example requires exactly one active source."
            )
        fixed_angle_reference_node_id = active_source_nodes[0]
        if fixed_angle_reference_node_id not in node_index_by_id:
            raise ValueError(f"Source refers to unknown node {fixed_angle_reference_node_id}.")

    power_sensors = _component(input_data, ComponentType.sym_power_sensor)
    sensor_ids = _field(power_sensors, AttributeType.id).astype(np.int64, copy=False)
    measured_objects = _field(power_sensors, AttributeType.measured_object).astype(np.int64, copy=False)
    terminal_types = _field(power_sensors, AttributeType.measured_terminal_type).astype(np.int64, copy=False)
    p_sigmas, q_sigmas = _power_sigmas(power_sensors)

    voltage_phasors = voltage_magnitudes * np.exp(1.0j * voltage_angles)
    for sensor_id, measured_object, terminal_type, p_sigma, q_sigma in zip(
        sensor_ids, measured_objects, terminal_types, p_sigmas, q_sigmas, strict=True
    ):
        try:
            from_index, to_index, y_ff, y_ft, y_tf, y_tt = line_admittances[int(measured_object)]
        except KeyError as error:
            raise NotImplementedError(
                f"Power sensor {int(sensor_id)} does not measure a supported line terminal."
            ) from error

        if terminal_type == int(MeasuredTerminalType.branch_from):
            local_index, remote_index = from_index, to_index
            self_admittance, mutual_admittance = y_ff, y_ft
            terminal_name = "from"
        elif terminal_type == int(MeasuredTerminalType.branch_to):
            local_index, remote_index = to_index, from_index
            self_admittance, mutual_admittance = y_tt, y_tf
            terminal_name = "to"
        else:
            raise NotImplementedError(
                f"Power sensor {int(sensor_id)} uses unsupported terminal type {int(terminal_type)}."
            )

        local_voltage = voltage_phasors[local_index]
        remote_voltage = voltage_phasors[remote_index]
        self_power = local_voltage * np.conj(self_admittance * local_voltage)
        mutual_power = local_voltage * np.conj(mutual_admittance * remote_voltage)
        local_angle_derivative = 1.0j * mutual_power
        remote_angle_derivative = -local_angle_derivative
        local_magnitude_derivative = (2.0 * self_power + mutual_power) / voltage_magnitudes[local_index]
        remote_magnitude_derivative = mutual_power / voltage_magnitudes[remote_index]

        complex_entries = {
            local_index: local_angle_derivative,
            remote_index: remote_angle_derivative,
            node_count + local_index: local_magnitude_derivative,
            node_count + remote_index: remote_magnitude_derivative,
        }
        p_entries = {column: float(np.real(value)) for column, value in complex_entries.items()}
        q_entries = {column: float(np.imag(value)) for column, value in complex_entries.items()}
        normalized_p_sigma = float(p_sigma) / _BASE_POWER_3P
        normalized_q_sigma = float(q_sigma) / _BASE_POWER_3P
        if np.isfinite(normalized_p_sigma):
            if normalized_p_sigma <= 0.0:
                raise ValueError(f"Power sensor {int(sensor_id)} requires a positive active-power sigma.")
            _append_sparse_row(
                rows,
                weights,
                labels,
                p_entries,
                1.0 / normalized_p_sigma**2,
                f"active power at line {int(measured_object)} {terminal_name} terminal",
            )
        elif not np.isposinf(normalized_p_sigma):
            raise ValueError(f"Power sensor {int(sensor_id)} has an invalid active-power sigma.")
        if np.isfinite(normalized_q_sigma):
            if normalized_q_sigma <= 0.0:
                raise ValueError(f"Power sensor {int(sensor_id)} requires a positive reactive-power sigma.")
            _append_sparse_row(
                rows,
                weights,
                labels,
                q_entries,
                1.0 / normalized_q_sigma**2,
                f"reactive power at line {int(measured_object)} {terminal_name} terminal",
            )
        elif not np.isposinf(normalized_q_sigma):
            raise ValueError(f"Power sensor {int(sensor_id)} has an invalid reactive-power sigma.")

    full_state_vector = np.concatenate((voltage_angles, voltage_magnitudes))
    if fixed_angle_reference_node_id is None:
        state_vector = full_state_vector
    else:
        reference_index = node_index_by_id[fixed_angle_reference_node_id]
        rows = [
            {
                column - int(column > reference_index): value
                for column, value in row.items()
                if column != reference_index
            }
            for row in rows
        ]
        state_vector = np.delete(full_state_vector, reference_index)

    measurement_matrix = _to_csr(rows, state_vector.size)
    weight_matrix = DiagonalMatrix(np.asarray(weights, dtype=float))
    return MeasurementModel(
        measurement_matrix=measurement_matrix,
        weight_matrix=weight_matrix,
        state_vector=state_vector,
        node_ids=node_ids.copy(),
        measurement_labels=tuple(labels),
        fixed_angle_reference_node_id=fixed_angle_reference_node_id,
    )


def build_gain_matrix(  # noqa: PLR0913
    input_data: Mapping[Any, Any],
    *,
    system_frequency: float = 50.0,
    error_tolerance: float = 1.0e-10,
    max_iterations: int = 100,
    state_estimation_result: Mapping[Any, Any] | None = None,
    max_gain_matrix_bytes: int | None = _DEFAULT_MAX_GAIN_MATRIX_BYTES,
) -> GainMatrixModel:
    """Build the conventional dense gain matrix ``G = H.T @ W @ H``.

    ``max_gain_matrix_bytes`` guards the allocation of dense ``G`` itself;
    NumPy inversion needs additional output and LAPACK workspace. Pass ``None``
    only when the caller has established a separate memory budget.
    """
    node_count = _field(_component(input_data, ComponentType.node), AttributeType.id).size
    gain_matrix_bytes = (2 * node_count) ** 2 * np.dtype(np.float64).itemsize
    if max_gain_matrix_bytes is not None and gain_matrix_bytes > max_gain_matrix_bytes:
        raise MemoryError(
            f"Dense gain matrix needs {gain_matrix_bytes / 1024**3:.2f} GiB, "
            f"above the {max_gain_matrix_bytes / 1024**3:.2f} GiB allocation limit."
        )
    measurement_model = build_measurement_model(
        input_data,
        system_frequency=system_frequency,
        error_tolerance=error_tolerance,
        max_iterations=max_iterations,
        state_estimation_result=state_estimation_result,
    )
    gain_matrix = measurement_model.measurement_matrix.weighted_gram(
        measurement_model.weight_matrix.diagonal,
        max_dense_matrix_bytes=max_gain_matrix_bytes,
    )
    return GainMatrixModel(measurement_model=measurement_model, gain_matrix=gain_matrix)
