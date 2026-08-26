# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Propagate a conventional symmetric NRSE state covariance to PGM outputs.

The public :func:`propagate_covariance` helper accepts the voltage-angle and
voltage-magnitude covariance produced by inversion of the conventional gain
matrix in :mod:`gain_matrix`.  It applies the same first-order polar-voltage
Jacobians used by analytical NRSE to node injections and two-terminal branch
flows.  The returned dataset has the same shape as a PGM symmetric
state-estimation result, which makes it suitable for the accuracy-comparison
notebooks.

This example module intentionally lives outside ``src/power_grid_model``.  It
supports the line and two-winding-transformer components used by the example
notebooks.  Exact zero-injection constraints are deliberately not included in
either the input state covariance or this propagation step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from gain_matrix import MeasurementModel

from power_grid_model import AttributeType, ComponentType

_BASE_POWER_3P = 1.0e6
_SQRT3 = np.sqrt(3.0)
_NUMERICAL_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class BranchAdmittance:
    """Symmetric per-unit two-terminal branch model in input order."""

    component_type: ComponentType
    component_id: int
    from_index: int
    to_index: int
    y_ff: complex
    y_ft: complex
    y_tf: complex
    y_tt: complex
    base_current_from: float
    base_current_to: float


def _component(dataset: Mapping[Any, Any], component_type: ComponentType, *, required: bool = True) -> Any:
    for key in (component_type, component_type.value):
        if key in dataset:
            return dataset[key]
    if required:
        raise ValueError(f"Dataset is missing required component {component_type.value!r}.")
    return None


def _field(component_data: Any, attribute: AttributeType) -> np.ndarray:
    if isinstance(component_data, Mapping):
        for key in (attribute, attribute.value):
            if key in component_data:
                return np.asarray(component_data[key])
        raise ValueError(f"Component data is missing required attribute {attribute.value!r}.")
    return np.asarray(component_data[attribute])


def _optional_float_field(component_data: Any, attribute: AttributeType, size: int, fallback: np.ndarray) -> np.ndarray:
    try:
        values = _field(component_data, attribute).astype(float, copy=False)
    except (KeyError, ValueError, IndexError):
        values = np.full(size, np.nan)
    return np.where(np.isfinite(values), values, fallback)


def _replace_na_integers(values: np.ndarray, fallback: np.ndarray | int) -> np.ndarray:
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.integer):
        return np.where(np.isfinite(values), values, fallback).astype(np.int64)
    missing = values == np.iinfo(values.dtype).min
    return np.where(missing, fallback, values).astype(np.int64)


def _terminal_admittances(
    y_series: complex,
    y_shunt: complex,
    tap_ratio: complex,
    from_status: bool,
    to_status: bool,
) -> tuple[complex, complex, complex, complex]:
    """Mirror PGM's symmetric two-terminal status and tap handling."""
    if from_status and to_status:
        y_tt = y_series + 0.5 * y_shunt
        y_ff = y_tt / abs(tap_ratio) ** 2
        y_ft = -y_series / np.conj(tap_ratio)
        y_tf = -y_series / tap_ratio
        return y_ff, y_ft, y_tf, y_tt

    y_ff = 0.0j
    y_tt = 0.0j
    if from_status or to_status:
        if abs(y_shunt) < _NUMERICAL_TOLERANCE:
            dangling_shunt = 0.0j
        else:
            dangling_shunt = 0.5 * y_shunt + 1.0 / (1.0 / y_series + 2.0 / y_shunt)
        if from_status:
            y_ff = dangling_shunt / abs(tap_ratio) ** 2
        if to_status:
            y_tt = dangling_shunt
    return y_ff, 0.0j, 0.0j, y_tt


def _line_admittances(
    input_data: Mapping[Any, Any],
    *,
    system_frequency: float,
    node_index_by_id: Mapping[int, int],
    rated_voltage_by_id: Mapping[int, float],
) -> list[BranchAdmittance]:
    lines = _component(input_data, ComponentType.line, required=False)
    if lines is None:
        return []

    line_ids = _field(lines, AttributeType.id).astype(np.int64, copy=False)
    from_nodes = _field(lines, AttributeType.from_node).astype(np.int64, copy=False)
    to_nodes = _field(lines, AttributeType.to_node).astype(np.int64, copy=False)
    from_statuses = _field(lines, AttributeType.from_status).astype(bool, copy=False)
    to_statuses = _field(lines, AttributeType.to_status).astype(bool, copy=False)
    resistance = _field(lines, AttributeType.r1).astype(float, copy=False)
    reactance = _field(lines, AttributeType.x1).astype(float, copy=False)
    capacitance = _field(lines, AttributeType.c1).astype(float, copy=False)
    loss_tangent = _field(lines, AttributeType.tan1).astype(float, copy=False)

    result: list[BranchAdmittance] = []
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
        if not np.isclose(from_rated_voltage, to_rated_voltage):
            raise NotImplementedError("The covariance example supports lines between equal rated voltages only.")

        series_impedance = resistance[index] + 1.0j * reactance[index]
        if series_impedance == 0.0:
            raise ValueError(f"Line {int(line_id)} has zero positive-sequence impedance.")
        base_admittance = _BASE_POWER_3P / from_rated_voltage**2
        y_series = 1.0 / series_impedance / base_admittance
        y_shunt = 2.0 * np.pi * system_frequency * capacitance[index] * (loss_tangent[index] + 1.0j) / base_admittance
        y_ff, y_ft, y_tf, y_tt = _terminal_admittances(
            y_series,
            y_shunt,
            1.0 + 0.0j,
            bool(from_statuses[index]),
            bool(to_statuses[index]),
        )
        base_current = _BASE_POWER_3P / (_SQRT3 * from_rated_voltage)
        result.append(
            BranchAdmittance(
                component_type=ComponentType.line,
                component_id=int(line_id),
                from_index=from_index,
                to_index=to_index,
                y_ff=y_ff,
                y_ft=y_ft,
                y_tf=y_tf,
                y_tt=y_tt,
                base_current_from=base_current,
                base_current_to=base_current,
            )
        )
    return result


def _tap_adjusted_value(  # noqa: PLR0913, PLR0917
    tap_pos: int,
    tap_min: int,
    tap_max: int,
    tap_nom: int,
    nominal_value: float,
    minimum_value: float,
    maximum_value: float,
) -> float:
    if min(tap_nom, tap_max) <= tap_pos <= max(tap_nom, tap_max):
        if tap_max == tap_nom:
            return nominal_value
        return nominal_value + (tap_pos - tap_nom) * (maximum_value - nominal_value) / (tap_max - tap_nom)
    if tap_min == tap_nom:
        return nominal_value
    return nominal_value + (tap_pos - tap_nom) * (minimum_value - nominal_value) / (tap_min - tap_nom)


def _transformer_admittances(  # noqa: PLR0915
    input_data: Mapping[Any, Any],
    *,
    node_index_by_id: Mapping[int, int],
    rated_voltage_by_id: Mapping[int, float],
) -> list[BranchAdmittance]:
    transformers = _component(input_data, ComponentType.transformer, required=False)
    if transformers is None:
        return []

    transformer_ids = _field(transformers, AttributeType.id).astype(np.int64, copy=False)
    size = transformer_ids.size
    from_nodes = _field(transformers, AttributeType.from_node).astype(np.int64, copy=False)
    to_nodes = _field(transformers, AttributeType.to_node).astype(np.int64, copy=False)
    from_statuses = _field(transformers, AttributeType.from_status).astype(bool, copy=False)
    to_statuses = _field(transformers, AttributeType.to_status).astype(bool, copy=False)
    u1_nominal = _field(transformers, AttributeType.u1).astype(float, copy=False)
    u2_nominal = _field(transformers, AttributeType.u2).astype(float, copy=False)
    rated_power = _field(transformers, AttributeType.sn).astype(float, copy=False)
    uk_nominal = _field(transformers, AttributeType.uk).astype(float, copy=False)
    pk_nominal = _field(transformers, AttributeType.pk).astype(float, copy=False)
    no_load_current = _field(transformers, AttributeType.i0).astype(float, copy=False)
    no_load_loss = _field(transformers, AttributeType.p0).astype(float, copy=False)
    clock = _field(transformers, AttributeType.clock).astype(np.int64, copy=False)
    tap_side = _field(transformers, AttributeType.tap_side).astype(np.int64, copy=False)
    tap_min = _replace_na_integers(_field(transformers, AttributeType.tap_min), 0)
    tap_max = _replace_na_integers(_field(transformers, AttributeType.tap_max), 0)
    tap_nom = _replace_na_integers(_field(transformers, AttributeType.tap_nom), 0)
    tap_pos = _replace_na_integers(_field(transformers, AttributeType.tap_pos), tap_nom)
    tap_size = _optional_float_field(transformers, AttributeType.tap_size, size, np.zeros(size))
    uk_min = _optional_float_field(transformers, AttributeType.uk_min, size, uk_nominal)
    uk_max = _optional_float_field(transformers, AttributeType.uk_max, size, uk_nominal)
    pk_min = _optional_float_field(transformers, AttributeType.pk_min, size, pk_nominal)
    pk_max = _optional_float_field(transformers, AttributeType.pk_max, size, pk_nominal)

    result: list[BranchAdmittance] = []
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

        direction = 1 if tap_max[index] > tap_min[index] else -1
        u1 = float(u1_nominal[index])
        u2 = float(u2_nominal[index])
        tap_offset = direction * (tap_pos[index] - tap_nom[index]) * tap_size[index]
        if int(tap_side[index]) == 0:
            u1 += tap_offset
        elif int(tap_side[index]) == 1:
            u2 += tap_offset
        else:
            raise ValueError(f"Transformer {int(transformer_id)} has invalid tap side {int(tap_side[index])}.")

        nominal_ratio = from_rated_voltage / to_rated_voltage
        off_nominal_ratio = (u1 / u2) / nominal_ratio
        uk = _tap_adjusted_value(
            int(tap_pos[index]),
            int(tap_min[index]),
            int(tap_max[index]),
            int(tap_nom[index]),
            float(uk_nominal[index]),
            float(uk_min[index]),
            float(uk_max[index]),
        )
        pk = _tap_adjusted_value(
            int(tap_pos[index]),
            int(tap_min[index]),
            int(tap_max[index]),
            int(tap_nom[index]),
            float(pk_nominal[index]),
            float(pk_min[index]),
            float(pk_max[index]),
        )

        z_abs = abs(uk) * u2**2 / rated_power[index]
        z_real = pk * u2**2 / rated_power[index] ** 2
        z_imag_squared = z_abs**2 - z_real**2
        z_imag = (1.0 if uk >= 0.0 else -1.0) * np.sqrt(max(0.0, z_imag_squared))
        z_series = z_real + 1.0j * z_imag
        base_current_to = _BASE_POWER_3P / (_SQRT3 * to_rated_voltage)
        base_admittance_to = base_current_to**2 / (_BASE_POWER_3P / 3.0)
        y_series = 1.0 / z_series / base_admittance_to

        y_shunt_abs = no_load_current[index] * rated_power[index] / u2**2
        y_shunt_real = no_load_loss[index] / u2**2
        y_shunt_imag = -np.sqrt(max(0.0, y_shunt_abs**2 - y_shunt_real**2))
        y_shunt = (y_shunt_real + 1.0j * y_shunt_imag) / base_admittance_to
        tap_ratio = off_nominal_ratio * np.exp(1.0j * (int(clock[index]) % 12) * np.pi / 6.0)
        y_ff, y_ft, y_tf, y_tt = _terminal_admittances(
            y_series,
            y_shunt,
            tap_ratio,
            bool(from_statuses[index]),
            bool(to_statuses[index]),
        )
        result.append(
            BranchAdmittance(
                component_type=ComponentType.transformer,
                component_id=int(transformer_id),
                from_index=from_index,
                to_index=to_index,
                y_ff=y_ff,
                y_ft=y_ft,
                y_tf=y_tf,
                y_tt=y_tt,
                base_current_from=_BASE_POWER_3P / (_SQRT3 * from_rated_voltage),
                base_current_to=base_current_to,
            )
        )
    return result


def build_branch_admittances(
    input_data: Mapping[Any, Any],
    *,
    node_ids: np.ndarray,
    system_frequency: float = 50.0,
) -> list[BranchAdmittance]:
    """Build status-aware line and transformer terminal admittances."""
    unsupported_components = (
        ComponentType.asym_line,
        ComponentType.link,
        ComponentType.generic_branch,
        ComponentType.three_winding_transformer,
        ComponentType.shunt,
    )
    for component_type in unsupported_components:
        component_data = _component(input_data, component_type, required=False)
        if component_data is not None and _field(component_data, AttributeType.id).size:
            raise NotImplementedError(f"The covariance example does not support component {component_type.value!r}.")
    nodes = _component(input_data, ComponentType.node)
    input_node_ids = _field(nodes, AttributeType.id).astype(np.int64, copy=False)
    if not np.array_equal(input_node_ids, node_ids):
        raise ValueError("Measurement-model node ordering does not match the input data.")
    node_index_by_id = {int(node_id): index for index, node_id in enumerate(input_node_ids)}
    rated_voltage_by_id = dict(
        zip(
            input_node_ids.tolist(),
            _field(nodes, AttributeType.u_rated).astype(float, copy=False).tolist(),
            strict=True,
        )
    )
    return [
        *_line_admittances(
            input_data,
            system_frequency=system_frequency,
            node_index_by_id=node_index_by_id,
            rated_voltage_by_id=rated_voltage_by_id,
        ),
        *_transformer_admittances(
            input_data,
            node_index_by_id=node_index_by_id,
            rated_voltage_by_id=rated_voltage_by_id,
        ),
    ]


def _fixed_reference_ids(measurement_model: MeasurementModel) -> tuple[int, ...]:
    reference_ids = getattr(measurement_model, "fixed_angle_reference_node_ids", None)
    if reference_ids is not None:
        return tuple(int(node_id) for node_id in reference_ids)
    reference_id = measurement_model.fixed_angle_reference_node_id
    return () if reference_id is None else (int(reference_id),)


def _validate_state_covariance(
    measurement_model: MeasurementModel, state_covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    state_covariance = np.asarray(state_covariance, dtype=float)
    node_count = measurement_model.node_ids.size
    reference_index_by_id = {int(node_id): index for index, node_id in enumerate(measurement_model.node_ids)}
    try:
        removed_indices = sorted(reference_index_by_id[node_id] for node_id in _fixed_reference_ids(measurement_model))
    except KeyError as error:
        raise ValueError(f"Unknown fixed angle reference node {error.args[0]}.") from error
    full_state_size = 2 * node_count
    retained = np.delete(np.arange(full_state_size), removed_indices)
    if state_covariance.shape != (retained.size, retained.size):
        raise ValueError(
            f"Expected state covariance shape {(retained.size, retained.size)}, got {state_covariance.shape}."
        )
    if not np.all(np.isfinite(state_covariance)):
        raise ValueError("State covariance must contain only finite values.")
    if not np.allclose(state_covariance, state_covariance.T, rtol=1.0e-8, atol=1.0e-12):
        raise ValueError("State covariance must be symmetric.")
    full_to_reduced = np.full(full_state_size, -1, dtype=np.int64)
    full_to_reduced[retained] = np.arange(retained.size)
    return 0.5 * (state_covariance + state_covariance.T), full_to_reduced


def _variance_to_sigma(variance: float, scale: float) -> float:
    tolerance = 1000.0 * np.finfo(float).eps * max(1.0, scale)
    if not np.isfinite(variance) or variance < -tolerance:
        return float("nan")
    return float(np.sqrt(max(0.0, variance)))


def _sigma_from_gradient(gradient: np.ndarray, covariance: np.ndarray) -> float:
    variance = float(gradient @ covariance @ gradient)
    scale = float(np.max(np.abs(covariance), initial=0.0) * np.dot(gradient, gradient))
    return _variance_to_sigma(variance, scale)


def _local_covariance(
    state_covariance: np.ndarray,
    node_indices: list[int],
    node_count: int,
    full_to_reduced: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    full_state_indices = np.asarray([*node_indices, *(node_count + index for index in node_indices)])
    reduced_state_indices = full_to_reduced[full_state_indices]
    retained = reduced_state_indices >= 0
    return state_covariance[np.ix_(reduced_state_indices[retained], reduced_state_indices[retained])], retained


def _branch_gradients(
    branch: BranchAdmittance,
    voltage: np.ndarray,
    voltage_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, complex, complex]:
    from_voltage = voltage[branch.from_index]
    to_voltage = voltage[branch.to_index]
    d_from_voltage = np.asarray([1.0j * from_voltage, 0.0j, voltage_direction[branch.from_index], 0.0j])
    d_to_voltage = np.asarray([0.0j, 1.0j * to_voltage, 0.0j, voltage_direction[branch.to_index]])
    from_current = branch.y_ff * from_voltage + branch.y_ft * to_voltage
    to_current = branch.y_tf * from_voltage + branch.y_tt * to_voltage
    d_from_current = branch.y_ff * d_from_voltage + branch.y_ft * d_to_voltage
    d_to_current = branch.y_tf * d_from_voltage + branch.y_tt * d_to_voltage
    return d_from_current, d_to_current, from_current, to_current


def _branch_side_covariance(  # noqa: PLR0913, PLR0917
    local_covariance: np.ndarray,
    terminal_voltage: complex,
    terminal_voltage_gradient: np.ndarray,
    current: complex,
    current_gradient: np.ndarray,
    base_current: float,
) -> np.ndarray:
    power_gradient = np.conj(current) * terminal_voltage_gradient + terminal_voltage * np.conj(current_gradient)
    if abs(current) == 0.0:
        current_magnitude_gradient = np.full(current_gradient.shape, np.nan)
    else:
        current_magnitude_gradient = np.real(np.conj(current) / abs(current) * current_gradient)
    jacobian = np.vstack(
        (
            np.real(power_gradient) * _BASE_POWER_3P,
            np.imag(power_gradient) * _BASE_POWER_3P,
            current_magnitude_gradient * base_current,
        )
    )
    return jacobian @ local_covariance @ jacobian.T


def _write_branch_sigmas(
    output: np.ndarray,
    index: int,
    from_covariance: np.ndarray,
    to_covariance: np.ndarray,
) -> None:
    sigma_fields = (
        (AttributeType.p_from_sigma, from_covariance[0, 0]),
        (AttributeType.q_from_sigma, from_covariance[1, 1]),
        (AttributeType.i_from_sigma, from_covariance[2, 2]),
        (AttributeType.p_to_sigma, to_covariance[0, 0]),
        (AttributeType.q_to_sigma, to_covariance[1, 1]),
        (AttributeType.i_to_sigma, to_covariance[2, 2]),
    )
    for field, variance in sigma_fields:
        scale = abs(float(variance))
        output[field][index] = _variance_to_sigma(float(variance), scale)


def propagate_covariance(  # noqa: PLR0915
    input_data: Mapping[Any, Any],
    measurement_model: MeasurementModel,
    state_covariance: np.ndarray,
    state_estimation_result: Mapping[Any, Any],
    *,
    system_frequency: float = 50.0,
) -> dict[Any, np.ndarray]:
    """Propagate ``G_inv`` to node and branch output sigmas.

    ``state_covariance`` must follow ``measurement_model.state_vector``.  The
    deterministic result supplies the accepted NRSE operating point and is
    copied before its sigma fields are populated.  Exact zero-injection
    constraints are intentionally absent: callers must pass the covariance of
    the sensor-only conventional gain matrix.
    """
    result = {component: values.copy() for component, values in state_estimation_result.items()}
    node_count = measurement_model.node_ids.size
    state_covariance, full_to_reduced = _validate_state_covariance(measurement_model, state_covariance)

    input_nodes = _component(input_data, ComponentType.node)
    rated_voltage = _field(input_nodes, AttributeType.u_rated).astype(float, copy=False)
    output_nodes = _component(state_estimation_result, ComponentType.node)
    output_node_ids = _field(output_nodes, AttributeType.id).astype(np.int64, copy=False)
    output_index_by_id = {int(node_id): index for index, node_id in enumerate(output_node_ids)}
    try:
        output_indices = np.asarray(
            [output_index_by_id[int(node_id)] for node_id in measurement_model.node_ids], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(f"State-estimation result is missing node {error.args[0]}.") from error
    voltage_magnitude = _field(output_nodes, AttributeType.u_pu)[output_indices].astype(float, copy=False)
    voltage_angle = _field(output_nodes, AttributeType.u_angle)[output_indices].astype(float, copy=False)
    voltage_direction = np.exp(1.0j * voltage_angle)
    voltage = voltage_magnitude * voltage_direction

    output_node = _component(result, ComponentType.node)
    covariance_diagonal = np.zeros(2 * node_count, dtype=float)
    retained = full_to_reduced >= 0
    covariance_diagonal[retained] = np.diag(state_covariance)[full_to_reduced[retained]]
    angle_sigma = np.sqrt(np.maximum(0.0, covariance_diagonal[:node_count]))
    voltage_pu_sigma = np.sqrt(np.maximum(0.0, covariance_diagonal[node_count:]))
    output_node[AttributeType.u_angle_sigma][output_indices] = angle_sigma
    output_node[AttributeType.u_pu_sigma][output_indices] = voltage_pu_sigma
    output_node[AttributeType.u_sigma][output_indices] = rated_voltage * voltage_pu_sigma

    branches = build_branch_admittances(
        input_data,
        node_ids=measurement_model.node_ids,
        system_frequency=system_frequency,
    )
    node_admittance_rows: list[dict[int, complex]] = [dict() for _ in range(node_count)]
    branches_by_component: dict[ComponentType, list[BranchAdmittance]] = {}
    for branch in branches:
        branches_by_component.setdefault(branch.component_type, []).append(branch)
        from_row = node_admittance_rows[branch.from_index]
        to_row = node_admittance_rows[branch.to_index]
        from_row[branch.from_index] = from_row.get(branch.from_index, 0.0j) + branch.y_ff
        from_row[branch.to_index] = from_row.get(branch.to_index, 0.0j) + branch.y_ft
        to_row[branch.from_index] = to_row.get(branch.from_index, 0.0j) + branch.y_tf
        to_row[branch.to_index] = to_row.get(branch.to_index, 0.0j) + branch.y_tt

    for node_index, admittance_row in enumerate(node_admittance_rows):
        dependency_indices = sorted(admittance_row)
        if not dependency_indices:
            output_node[AttributeType.p_sigma][output_indices[node_index]] = 0.0
            output_node[AttributeType.q_sigma][output_indices[node_index]] = 0.0
            continue
        local_covariance, local_retained = _local_covariance(
            state_covariance,
            dependency_indices,
            node_count,
            full_to_reduced,
        )
        current = sum(admittance_row[index] * voltage[index] for index in dependency_indices)
        power_gradient = np.empty(2 * len(dependency_indices), dtype=complex)
        for local_index, dependency_index in enumerate(dependency_indices):
            current_angle_gradient = admittance_row[dependency_index] * 1.0j * voltage[dependency_index]
            current_magnitude_gradient = admittance_row[dependency_index] * voltage_direction[dependency_index]
            voltage_angle_gradient = 1.0j * voltage[node_index] if dependency_index == node_index else 0.0j
            voltage_magnitude_gradient = voltage_direction[node_index] if dependency_index == node_index else 0.0j
            power_gradient[local_index] = voltage_angle_gradient * np.conj(current) + voltage[node_index] * np.conj(
                current_angle_gradient
            )
            power_gradient[len(dependency_indices) + local_index] = voltage_magnitude_gradient * np.conj(
                current
            ) + voltage[node_index] * np.conj(current_magnitude_gradient)
        p_gradient = np.real(power_gradient) * _BASE_POWER_3P
        q_gradient = np.imag(power_gradient) * _BASE_POWER_3P
        output_index = output_indices[node_index]
        output_node[AttributeType.p_sigma][output_index] = _sigma_from_gradient(
            p_gradient[local_retained], local_covariance
        )
        output_node[AttributeType.q_sigma][output_index] = _sigma_from_gradient(
            q_gradient[local_retained], local_covariance
        )

    for component_type, component_branches in branches_by_component.items():
        output_component = _component(result, component_type)
        output_ids = _field(output_component, AttributeType.id).astype(np.int64, copy=False)
        output_index_by_id = {int(component_id): index for index, component_id in enumerate(output_ids)}
        for branch in component_branches:
            try:
                output_index = output_index_by_id[branch.component_id]
            except KeyError as error:
                raise ValueError(
                    f"State-estimation result is missing {component_type.value} {branch.component_id}."
                ) from error
            local_covariance, local_retained = _local_covariance(
                state_covariance,
                [branch.from_index, branch.to_index],
                node_count,
                full_to_reduced,
            )
            d_from_current, d_to_current, from_current, to_current = _branch_gradients(
                branch,
                voltage,
                voltage_direction,
            )
            from_voltage_gradient = np.asarray(
                [
                    1.0j * voltage[branch.from_index],
                    0.0j,
                    voltage_direction[branch.from_index],
                    0.0j,
                ]
            )
            to_voltage_gradient = np.asarray(
                [
                    0.0j,
                    1.0j * voltage[branch.to_index],
                    0.0j,
                    voltage_direction[branch.to_index],
                ]
            )
            from_covariance = _branch_side_covariance(
                local_covariance,
                voltage[branch.from_index],
                from_voltage_gradient[local_retained],
                from_current,
                d_from_current[local_retained],
                branch.base_current_from,
            )
            to_covariance = _branch_side_covariance(
                local_covariance,
                voltage[branch.to_index],
                to_voltage_gradient[local_retained],
                to_current,
                d_to_current[local_retained],
                branch.base_current_to,
            )
            _write_branch_sigmas(output_component, output_index, from_covariance, to_covariance)

    return result
