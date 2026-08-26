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
models symmetric voltage sensors and branch-terminal power or global-angle
current sensors on lines and ordinary transformers. Exact
zero-injection constraints and unmeasured branch equations are deliberately
excluded from the conventional gain matrix; unsupported components remain
outside this small example helper.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from power_grid_model import (
    AngleMeasurementType,
    AttributeType,
    CalculationMethod,
    ComponentType,
    MeasuredTerminalType,
    PowerGridModel,
)

_BASE_POWER_3P = 1.0e6
_DEFAULT_MAX_GAIN_MATRIX_BYTES = 256 * 1024**2
_NUMERICAL_TOLERANCE = 1.0e-8
_TerminalAdmittance = tuple[int, int, complex, complex, complex, complex]


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
    fixed_angle_reference_node_ids: tuple[int, ...]

    @property
    def fixed_angle_reference_node_id(self) -> int | None:
        """Return the fixed node for the legacy zero-or-one-reference case."""
        if len(self.fixed_angle_reference_node_ids) == 1:
            return self.fixed_angle_reference_node_ids[0]
        return None


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


def _calc_param_y_sym(
    y_series: complex,
    y_shunt: complex,
    tap_ratio: complex,
    from_status: bool,
    to_status: bool,
) -> tuple[complex, complex, complex, complex]:
    """Translate PGM's ``Branch::calc_param_y_sym`` to Python."""
    if from_status and to_status:
        y_tt = y_series + 0.5 * y_shunt
        y_ff = y_tt / abs(tap_ratio) ** 2
        y_ft = -y_series / np.conj(tap_ratio)
        y_tf = -y_series / tap_ratio
        return y_ff, y_ft, y_tf, y_tt
    if from_status or to_status:
        if abs(y_shunt) < _NUMERICAL_TOLERANCE:
            connected_terminal_admittance = 0.0j
        else:
            connected_terminal_admittance = 0.5 * y_shunt + 1.0 / (1.0 / y_series + 2.0 / y_shunt)
        y_ff = connected_terminal_admittance / abs(tap_ratio) ** 2 if from_status else 0.0j
        y_tt = connected_terminal_admittance if to_status else 0.0j
        return y_ff, 0.0j, 0.0j, y_tt
    return 0.0j, 0.0j, 0.0j, 0.0j


def _line_terminal_admittances(
    input_data: Mapping[Any, Any], system_frequency: float, node_index_by_id: Mapping[int, int]
) -> dict[int, _TerminalAdmittance]:
    lines = _component(input_data, ComponentType.line, required=False)
    if lines is None:
        return {}
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

    result: dict[int, _TerminalAdmittance] = {}
    for index, line_id in enumerate(line_ids):
        from_node_id = int(from_nodes[index])
        to_node_id = int(to_nodes[index])
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
        y_ff, y_ft, y_tf, y_tt = _calc_param_y_sym(
            series_admittance,
            shunt_admittance,
            1.0 + 0.0j,
            bool(from_status[index]),
            bool(to_status[index]),
        )
        result[int(line_id)] = (
            from_index,
            to_index,
            y_ff,
            y_ft,
            y_tf,
            y_tt,
        )
    return result


def _tap_adjust_impedance(  # noqa: PLR0913, PLR0917
    tap_pos: int,
    tap_min: int,
    tap_max: int,
    tap_nom: int,
    nominal_value: float,
    value_at_min: float,
    value_at_max: float,
) -> float:
    if min(tap_nom, tap_max) <= tap_pos <= max(tap_nom, tap_max):
        if tap_max == tap_nom:
            return nominal_value
        return nominal_value + (tap_pos - tap_nom) * (value_at_max - nominal_value) / (tap_max - tap_nom)
    if tap_min == tap_nom:
        return nominal_value
    return nominal_value + (tap_pos - tap_nom) * (value_at_min - nominal_value) / (tap_min - tap_nom)


def _transformer_terminal_admittances(  # noqa: PLR0915
    input_data: Mapping[Any, Any], node_index_by_id: Mapping[int, int]
) -> dict[int, _TerminalAdmittance]:
    """Return ordinary-transformer positive-sequence terminal admittances."""
    transformers = _component(input_data, ComponentType.transformer, required=False)
    if transformers is None:
        return {}
    nodes = _component(input_data, ComponentType.node)
    node_ids = _field(nodes, AttributeType.id).astype(np.int64, copy=False)
    rated_voltage_by_id = dict(
        zip(node_ids.tolist(), _field(nodes, AttributeType.u_rated).astype(float, copy=False).tolist(), strict=True)
    )

    transformer_ids = _field(transformers, AttributeType.id).astype(np.int64, copy=False)
    from_nodes = _field(transformers, AttributeType.from_node).astype(np.int64, copy=False)
    to_nodes = _field(transformers, AttributeType.to_node).astype(np.int64, copy=False)
    from_status = _field(transformers, AttributeType.from_status).astype(bool, copy=False)
    to_status = _field(transformers, AttributeType.to_status).astype(bool, copy=False)
    u1_values = _field(transformers, AttributeType.u1).astype(float, copy=False)
    u2_values = _field(transformers, AttributeType.u2).astype(float, copy=False)
    rated_powers = _field(transformers, AttributeType.sn).astype(float, copy=False)
    uk_values = _field(transformers, AttributeType.uk).astype(float, copy=False)
    pk_values = _field(transformers, AttributeType.pk).astype(float, copy=False)
    i0_values = _field(transformers, AttributeType.i0).astype(float, copy=False)
    p0_values = _field(transformers, AttributeType.p0).astype(float, copy=False)
    clock_values = _field(transformers, AttributeType.clock).astype(np.int64, copy=False)
    tap_side_values = _field(transformers, AttributeType.tap_side).astype(np.int64, copy=False)
    tap_min_values = _field(transformers, AttributeType.tap_min).astype(np.int64, copy=False)
    tap_max_values = _field(transformers, AttributeType.tap_max).astype(np.int64, copy=False)
    tap_size_values = _field(transformers, AttributeType.tap_size).astype(float, copy=False)

    raw_tap_nom_values = _field(transformers, AttributeType.tap_nom)
    raw_tap_pos_values = _field(transformers, AttributeType.tap_pos)
    tap_nom_missing = np.iinfo(raw_tap_nom_values.dtype).min
    tap_pos_missing = np.iinfo(raw_tap_pos_values.dtype).min
    tap_nom_values = np.where(raw_tap_nom_values == tap_nom_missing, 0, raw_tap_nom_values).astype(np.int64, copy=False)
    tap_pos_values = np.where(raw_tap_pos_values == tap_pos_missing, tap_nom_values, raw_tap_pos_values).astype(
        np.int64, copy=False
    )
    uk_min_values = _optional_field(transformers, AttributeType.uk_min, transformer_ids.size).astype(float, copy=False)
    uk_max_values = _optional_field(transformers, AttributeType.uk_max, transformer_ids.size).astype(float, copy=False)
    pk_min_values = _optional_field(transformers, AttributeType.pk_min, transformer_ids.size).astype(float, copy=False)
    pk_max_values = _optional_field(transformers, AttributeType.pk_max, transformer_ids.size).astype(float, copy=False)

    result: dict[int, _TerminalAdmittance] = {}
    for index, transformer_id in enumerate(transformer_ids):
        from_node_id = int(from_nodes[index])
        to_node_id = int(to_nodes[index])
        try:
            from_index = node_index_by_id[from_node_id]
            to_index = node_index_by_id[to_node_id]
            from_rated_voltage = rated_voltage_by_id[from_node_id]
            to_rated_voltage = rated_voltage_by_id[to_node_id]
        except KeyError as error:
            raise ValueError(f"Transformer {int(transformer_id)} refers to unknown node {error.args[0]}.") from error
        if from_index == to_index:
            raise NotImplementedError(
                "The gain-matrix example does not support transformers connected to one node twice."
            )

        tap_min = int(tap_min_values[index])
        tap_max = int(tap_max_values[index])
        tap_nom = int(tap_nom_values[index])
        tap_pos = int(np.clip(tap_pos_values[index], min(tap_min, tap_max), max(tap_min, tap_max)))
        tap_direction = 1 if tap_max > tap_min else -1
        u1 = float(u1_values[index])
        u2 = float(u2_values[index])
        tap_offset = tap_direction * (tap_pos - tap_nom) * float(tap_size_values[index])
        if tap_side_values[index] == 0:
            u1 += tap_offset
        elif tap_side_values[index] == 1:
            u2 += tap_offset
        else:
            raise ValueError(f"Transformer {int(transformer_id)} has invalid tap_side {int(tap_side_values[index])}.")

        uk_nominal = float(uk_values[index])
        pk_nominal = float(pk_values[index])
        uk = _tap_adjust_impedance(
            tap_pos,
            tap_min,
            tap_max,
            tap_nom,
            uk_nominal,
            uk_nominal if np.isnan(uk_min_values[index]) else float(uk_min_values[index]),
            uk_nominal if np.isnan(uk_max_values[index]) else float(uk_max_values[index]),
        )
        pk = _tap_adjust_impedance(
            tap_pos,
            tap_min,
            tap_max,
            tap_nom,
            pk_nominal,
            pk_nominal if np.isnan(pk_min_values[index]) else float(pk_min_values[index]),
            pk_nominal if np.isnan(pk_max_values[index]) else float(pk_max_values[index]),
        )
        rated_power = float(rated_powers[index])
        base_admittance_to = _BASE_POWER_3P / to_rated_voltage**2
        series_impedance_abs = abs(uk) * u2**2 / rated_power
        series_resistance = pk * u2**2 / rated_power**2
        series_reactance_squared = series_impedance_abs**2 - series_resistance**2
        series_reactance = np.copysign(np.sqrt(max(series_reactance_squared, 0.0)), uk)
        series_admittance = 1.0 / complex(series_resistance, series_reactance) / base_admittance_to

        shunt_admittance_abs = float(i0_values[index]) * rated_power / u2**2
        shunt_conductance = float(p0_values[index]) / u2**2
        shunt_susceptance_squared = shunt_admittance_abs**2 - shunt_conductance**2
        shunt_admittance = (
            complex(
                shunt_conductance,
                -np.sqrt(max(shunt_susceptance_squared, 0.0)),
            )
            / base_admittance_to
        )
        nominal_ratio = from_rated_voltage / to_rated_voltage
        off_nominal_ratio = (u1 / u2) / nominal_ratio
        clock = int(clock_values[index]) % 12
        tap_ratio = off_nominal_ratio * np.exp(1.0j * clock * np.pi / 6.0)
        y_ff, y_ft, y_tf, y_tt = _calc_param_y_sym(
            series_admittance,
            shunt_admittance,
            tap_ratio,
            bool(from_status[index]),
            bool(to_status[index]),
        )
        result[int(transformer_id)] = (from_index, to_index, y_ff, y_ft, y_tf, y_tt)
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


def _decomposed_current_variances(
    magnitude: float,
    angle: float,
    magnitude_sigma: float,
    angle_sigma: float,
) -> tuple[float, float]:
    """Match PGM's second-order conversion from polar to Cartesian variance."""
    magnitude_variance = magnitude_sigma**2
    angle_variance = angle_sigma**2
    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    cos_angle_squared = cos_angle**2
    sin_angle_squared = sin_angle**2
    magnitude_squared = magnitude**2
    real_variance = (
        magnitude_variance * cos_angle_squared
        + magnitude_squared * angle_variance * sin_angle_squared
        + 0.5 * magnitude_squared * angle_variance**2 * cos_angle_squared
        + magnitude_variance * angle_variance * sin_angle_squared
    )
    imag_variance = (
        magnitude_variance * sin_angle_squared
        + magnitude_squared * angle_variance * cos_angle_squared
        + 0.5 * magnitude_squared * angle_variance**2 * sin_angle_squared
        + magnitude_variance * angle_variance * cos_angle_squared
    )
    return real_variance, imag_variance


def _validate_supported_input(input_data: Mapping[Any, Any]) -> None:
    unsupported_components = (
        ComponentType.asym_line,
        ComponentType.link,
        ComponentType.generic_branch,
        ComponentType.three_winding_transformer,
        ComponentType.asym_load,
        ComponentType.asym_gen,
        ComponentType.asym_voltage_sensor,
        ComponentType.asym_power_sensor,
        ComponentType.asym_current_sensor,
    )
    for component_type in unsupported_components:
        component_data = _component(input_data, component_type, required=False)
        if component_data is not None and _field(component_data, AttributeType.id).size:
            raise NotImplementedError(f"The gain-matrix example does not support component {component_type.value!r}.")


def _append_sparse_row(  # noqa: PLR0913, PLR0917
    rows: list[dict[int, float]],
    weights: list[float],
    labels: list[str],
    entries: Mapping[int, float],
    weight: float,
    label: str,
) -> bool:
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"Measurement {label!r} has a non-positive or non-finite weight.")
    combined = {int(column): float(value) for column, value in entries.items() if value != 0.0}
    if not combined:
        return False
    rows.append(combined)
    weights.append(float(weight))
    labels.append(label)
    return True


def _connect_angle_nodes(neighbours: list[set[int]], angle_derivatives: Mapping[int, complex]) -> tuple[int, ...]:
    measured_nodes = tuple(sorted(index for index, value in angle_derivatives.items() if value != 0.0))
    if measured_nodes:
        first_node = measured_nodes[0]
        for node_index in measured_nodes[1:]:
            neighbours[first_node].add(node_index)
            neighbours[node_index].add(first_node)
    return measured_nodes


def _select_fixed_angle_references(
    node_ids: np.ndarray,
    neighbours: list[set[int]],
    anchored_nodes: set[int],
    preferred_nodes: tuple[int, ...],
) -> tuple[int, ...]:
    fixed_indices: list[int] = []
    visited: set[int] = set()
    for initial_node in range(node_ids.size):
        if initial_node in visited:
            continue
        component: set[int] = set()
        pending = [initial_node]
        while pending:
            node_index = pending.pop()
            if node_index in component:
                continue
            component.add(node_index)
            pending.extend(neighbours[node_index] - component)
        visited.update(component)
        if component & anchored_nodes:
            continue
        preferred_node = next((node_index for node_index in preferred_nodes if node_index in component), None)
        fixed_indices.append(min(component) if preferred_node is None else preferred_node)
    return tuple(fixed_indices)


def _active_source_node_indices(input_data: Mapping[Any, Any], node_index_by_id: Mapping[int, int]) -> tuple[int, ...]:
    sources = _component(input_data, ComponentType.source, required=False)
    if sources is None:
        return ()
    source_nodes = _field(sources, AttributeType.node).astype(np.int64, copy=False)
    source_status = _field(sources, AttributeType.status).astype(bool, copy=False)
    active_source_indices: list[int] = []
    for node_id, active in zip(source_nodes, source_status, strict=True):
        if not active:
            continue
        try:
            node_index = node_index_by_id[int(node_id)]
        except KeyError as error:
            raise ValueError(f"Source refers to unknown node {error.args[0]}.") from error
        if node_index not in active_source_indices:
            active_source_indices.append(node_index)
    return tuple(active_source_indices)


def _select_line_terminal(
    line_admittance: tuple[int, int, complex, complex, complex, complex],
    terminal_type: int,
) -> tuple[int, int, complex, complex, str]:
    from_index, to_index, y_ff, y_ft, y_tf, y_tt = line_admittance
    if terminal_type == int(MeasuredTerminalType.branch_from):
        return from_index, to_index, y_ff, y_ft, "from"
    if terminal_type == int(MeasuredTerminalType.branch_to):
        return to_index, from_index, y_tt, y_tf, "to"
    raise NotImplementedError(f"Unsupported line-terminal type {terminal_type}.")


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

    The full state ordering is
    ``[theta_0, ..., theta_(n-1), v_0, ..., v_(n-1)]``. One angle is removed
    from every measurement component that has neither a physical voltage-angle
    measurement nor a global-angle current measurement. Angles are in radians
    and magnitudes are per unit. Power and current measurements and their
    standard deviations are converted to PGM's per-unit bases.

    PGM's exact zero-injection constraints and any unmeasured branch equations
    are not included in this conventional measurement-only gain matrix.
    """
    _validate_supported_input(input_data)
    nodes = _component(input_data, ComponentType.node)
    input_node_ids = _field(nodes, AttributeType.id).astype(np.int64, copy=False)
    if input_node_ids.size == 0:
        raise ValueError("The gain-matrix example requires at least one node.")
    input_node_index_by_id = {int(node_id): index for index, node_id in enumerate(input_node_ids)}
    line_admittances = _line_terminal_admittances(input_data, system_frequency, input_node_index_by_id)
    transformer_admittances = _transformer_terminal_admittances(input_data, input_node_index_by_id)
    duplicate_branch_ids = line_admittances.keys() & transformer_admittances.keys()
    if duplicate_branch_ids:
        raise ValueError(f"Line and transformer IDs overlap: {sorted(duplicate_branch_ids)}.")
    branch_admittances = line_admittances | transformer_admittances
    branch_type_by_id = {
        **dict.fromkeys(line_admittances, "line"),
        **dict.fromkeys(transformer_admittances, "transformer"),
    }

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
    angle_neighbours: list[set[int]] = [set() for _ in range(node_count)]
    anchored_angle_nodes: set[int] = set()

    rated_voltage_by_id = dict(
        zip(
            _field(nodes, AttributeType.id).astype(np.int64, copy=False).tolist(),
            _field(nodes, AttributeType.u_rated).astype(float, copy=False).tolist(),
            strict=True,
        )
    )

    voltage_sensors = _component(input_data, ComponentType.sym_voltage_sensor, required=False)
    if voltage_sensors is not None:
        sensor_node_ids = _field(voltage_sensors, AttributeType.measured_object).astype(np.int64, copy=False)
        voltage_sigmas = _field(voltage_sensors, AttributeType.u_sigma).astype(float, copy=False)
        voltage_angles_measured = _field(voltage_sensors, AttributeType.u_angle_measured).astype(float, copy=False)
        voltage_sensor_indices_by_node: dict[int, list[int]] = {}
        for sensor_index, node_id in enumerate(sensor_node_ids):
            voltage_sensor_indices_by_node.setdefault(int(node_id), []).append(sensor_index)

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
                anchored_angle_nodes.add(node_index)

    voltage_phasors = voltage_magnitudes * np.exp(1.0j * voltage_angles)
    power_sensors = _component(input_data, ComponentType.sym_power_sensor, required=False)
    if power_sensors is not None:
        sensor_ids = _field(power_sensors, AttributeType.id).astype(np.int64, copy=False)
        measured_objects = _field(power_sensors, AttributeType.measured_object).astype(np.int64, copy=False)
        terminal_types = _field(power_sensors, AttributeType.measured_terminal_type).astype(np.int64, copy=False)
        p_sigmas, q_sigmas = _power_sigmas(power_sensors)
        for sensor_id, measured_object, terminal_type, p_sigma, q_sigma in zip(
            sensor_ids, measured_objects, terminal_types, p_sigmas, q_sigmas, strict=True
        ):
            try:
                branch_admittance = branch_admittances[int(measured_object)]
            except KeyError as error:
                raise NotImplementedError(
                    f"Power sensor {int(sensor_id)} does not measure a supported branch terminal."
                ) from error
            try:
                local_index, remote_index, self_admittance, mutual_admittance, terminal_name = _select_line_terminal(
                    branch_admittance, int(terminal_type)
                )
            except NotImplementedError as error:
                raise NotImplementedError(
                    f"Power sensor {int(sensor_id)} uses unsupported terminal type {int(terminal_type)}."
                ) from error
            # PGM's NRSE processes branch sensors through off-diagonal branch
            # entries, so a half-connected line has no active sensor row even
            # though its connected-terminal equivalent shunt is non-zero.
            if mutual_admittance == 0.0:
                continue

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
            measured_component = branch_type_by_id[int(measured_object)]
            row_was_added = False
            if np.isfinite(normalized_p_sigma):
                if normalized_p_sigma <= 0.0:
                    raise ValueError(f"Power sensor {int(sensor_id)} requires a positive active-power sigma.")
                row_was_added |= _append_sparse_row(
                    rows,
                    weights,
                    labels,
                    p_entries,
                    1.0 / normalized_p_sigma**2,
                    f"active power at {measured_component} {int(measured_object)} {terminal_name} terminal",
                )
            elif not np.isposinf(normalized_p_sigma):
                raise ValueError(f"Power sensor {int(sensor_id)} has an invalid active-power sigma.")
            if np.isfinite(normalized_q_sigma):
                if normalized_q_sigma <= 0.0:
                    raise ValueError(f"Power sensor {int(sensor_id)} requires a positive reactive-power sigma.")
                row_was_added |= _append_sparse_row(
                    rows,
                    weights,
                    labels,
                    q_entries,
                    1.0 / normalized_q_sigma**2,
                    f"reactive power at {measured_component} {int(measured_object)} {terminal_name} terminal",
                )
            elif not np.isposinf(normalized_q_sigma):
                raise ValueError(f"Power sensor {int(sensor_id)} has an invalid reactive-power sigma.")
            if row_was_added:
                _connect_angle_nodes(
                    angle_neighbours,
                    {local_index: local_angle_derivative, remote_index: remote_angle_derivative},
                )

    current_sensors = _component(input_data, ComponentType.sym_current_sensor, required=False)
    if current_sensors is not None:
        sensor_ids = _field(current_sensors, AttributeType.id).astype(np.int64, copy=False)
        measured_objects = _field(current_sensors, AttributeType.measured_object).astype(np.int64, copy=False)
        terminal_types = _field(current_sensors, AttributeType.measured_terminal_type).astype(np.int64, copy=False)
        angle_types = _field(current_sensors, AttributeType.angle_measurement_type).astype(np.int64, copy=False)
        current_sigmas = _field(current_sensors, AttributeType.i_sigma).astype(float, copy=False)
        current_angle_sigmas = _field(current_sensors, AttributeType.i_angle_sigma).astype(float, copy=False)
        measured_magnitudes = _field(current_sensors, AttributeType.i_measured).astype(float, copy=False)
        measured_angles = _field(current_sensors, AttributeType.i_angle_measured).astype(float, copy=False)
        current_sensor_parameters = zip(
            sensor_ids,
            measured_objects,
            terminal_types,
            angle_types,
            current_sigmas,
            current_angle_sigmas,
            measured_magnitudes,
            measured_angles,
            strict=True,
        )
        for (
            sensor_id,
            measured_object,
            terminal_type,
            angle_type,
            current_sigma,
            current_angle_sigma,
            magnitude,
            angle,
        ) in current_sensor_parameters:
            if angle_type != int(AngleMeasurementType.global_angle):
                raise NotImplementedError(
                    f"Current sensor {int(sensor_id)} requires AngleMeasurementType.global_angle."
                )
            try:
                branch_admittance = branch_admittances[int(measured_object)]
            except KeyError as error:
                raise NotImplementedError(
                    f"Current sensor {int(sensor_id)} does not measure a supported branch terminal."
                ) from error
            try:
                local_index, remote_index, self_admittance, mutual_admittance, terminal_name = _select_line_terminal(
                    branch_admittance, int(terminal_type)
                )
            except NotImplementedError as error:
                raise NotImplementedError(
                    f"Current sensor {int(sensor_id)} uses unsupported terminal type {int(terminal_type)}."
                ) from error
            if mutual_admittance == 0.0:
                continue
            if not np.isfinite(magnitude) or not np.isfinite(angle):
                raise ValueError(
                    f"Current sensor {int(sensor_id)} requires finite measured magnitude and angle values."
                )
            if np.isposinf(current_sigma) or np.isposinf(current_angle_sigma):
                continue
            if not np.isfinite(current_sigma) or current_sigma <= 0.0:
                raise ValueError(f"Current sensor {int(sensor_id)} requires a positive current sigma.")
            if not np.isfinite(current_angle_sigma) or current_angle_sigma <= 0.0:
                raise ValueError(f"Current sensor {int(sensor_id)} requires a positive current-angle sigma.")

            local_node_id = int(node_ids[local_index])
            base_current = _BASE_POWER_3P / (np.sqrt(3.0) * rated_voltage_by_id[local_node_id])
            normalized_magnitude = float(magnitude) / base_current
            normalized_current_sigma = float(current_sigma) / base_current
            real_variance, imag_variance = _decomposed_current_variances(
                normalized_magnitude,
                float(angle),
                normalized_current_sigma,
                float(current_angle_sigma),
            )
            local_derivative = self_admittance * voltage_phasors[local_index]
            remote_derivative = mutual_admittance * voltage_phasors[remote_index]
            complex_entries = {
                local_index: 1.0j * local_derivative,
                remote_index: 1.0j * remote_derivative,
                node_count + local_index: local_derivative / voltage_magnitudes[local_index],
                node_count + remote_index: remote_derivative / voltage_magnitudes[remote_index],
            }
            real_entries = {column: float(np.real(value)) for column, value in complex_entries.items()}
            imag_entries = {column: float(np.imag(value)) for column, value in complex_entries.items()}
            measured_component = branch_type_by_id[int(measured_object)]
            real_row_added = _append_sparse_row(
                rows,
                weights,
                labels,
                real_entries,
                1.0 / real_variance,
                f"real current at {measured_component} {int(measured_object)} {terminal_name} terminal",
            )
            imag_row_added = _append_sparse_row(
                rows,
                weights,
                labels,
                imag_entries,
                1.0 / imag_variance,
                f"imaginary current at {measured_component} {int(measured_object)} {terminal_name} terminal",
            )
            if real_row_added or imag_row_added:
                measured_angle_nodes = _connect_angle_nodes(
                    angle_neighbours,
                    {local_index: complex_entries[local_index], remote_index: complex_entries[remote_index]},
                )
                anchored_angle_nodes.update(measured_angle_nodes)

    full_state_vector = np.concatenate((voltage_angles, voltage_magnitudes))
    fixed_reference_indices = _select_fixed_angle_references(
        node_ids,
        angle_neighbours,
        anchored_angle_nodes,
        _active_source_node_indices(input_data, node_index_by_id),
    )
    fixed_angle_reference_node_ids = tuple(int(node_ids[index]) for index in fixed_reference_indices)
    retained_columns = [column for column in range(full_state_vector.size) if column not in fixed_reference_indices]
    reduced_column_by_full_column = {column: reduced for reduced, column in enumerate(retained_columns)}
    rows = [
        {
            reduced_column_by_full_column[column]: value
            for column, value in row.items()
            if column in reduced_column_by_full_column
        }
        for row in rows
    ]
    state_vector = full_state_vector[retained_columns]

    measurement_matrix = _to_csr(rows, state_vector.size)
    weight_matrix = DiagonalMatrix(np.asarray(weights, dtype=float))
    return MeasurementModel(
        measurement_matrix=measurement_matrix,
        weight_matrix=weight_matrix,
        state_vector=state_vector,
        node_ids=node_ids.copy(),
        measurement_labels=tuple(labels),
        fixed_angle_reference_node_ids=fixed_angle_reference_node_ids,
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
    minimum_gain_matrix_bytes = node_count**2 * np.dtype(np.float64).itemsize
    if max_gain_matrix_bytes is not None and minimum_gain_matrix_bytes > max_gain_matrix_bytes:
        raise MemoryError(
            f"Dense gain matrix needs at least {minimum_gain_matrix_bytes / 1024**3:.2f} GiB, "
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
