# ruff: noqa: PLR0912, PLR0915, S102
"""Build WLS-UQ matrix sparsity artifacts for supported symmetric PGM cases.

The reusable API accepts a :class:`SparsityCase` containing PGM input data and
its row-oriented topology.  It derives every dimension and measurement count
from that case, mirrors PGM's radial reverse-DFS ordering, reconstructs the
complex ILSE augmented system, and builds the conventional real-polar
``H``, ``G = H.T @ W @ H``, and ``G^-1`` matrices.  The command-line adapters
load IEEE 33 or CIGRE MV from their public example notebooks; IEEE 33 remains
the default paper case.

The complex reconstruction currently supports one connected radial symmetric
network made from ordinary lines and two-winding transformers.  The real gain
supports the same branch measurements plus finite-variance PGM-style bus
injections aggregated from load, generator, source, and direct-node power
sensors.  Unsupported components or sensor terminals fail explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath

from power_grid_model import (
    AngleMeasurementType,
    AttributeType,
    CalculationMethod,
    CalculationType,
    ComponentType,
    DatasetType,
    MeasuredTerminalType,
    PowerGridModel,
    initialize_array,
)
from power_grid_model.validation import assert_valid_input_data

TopologyValue = int | float | str
NetworkTopology = dict[str, list[dict[str, TopologyValue]]]

WORKFLOW_ROOT = Path(__file__).resolve().parent
PGM_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = PGM_ROOT / "docs" / "examples"
sys.path.insert(0, str(EXAMPLE_ROOT))

from gain_matrix import (  # noqa: E402
    _line_terminal_admittances,
    _power_sigmas,
    _select_fixed_angle_references,
    _transformer_terminal_admittances,
    build_measurement_model,
)

BASE_POWER_3P = 1.0e6
FIGURE_SIZE = (7.16, 4.8)
PREVIEW_DPI = 300
STYLE = ["science", "ieee"]
DEFAULT_RELATIVE_THRESHOLD = 1.0e-12
MAX_RELATIVE_INVERSE_RESIDUAL = 1.0e-12
DEFAULT_ERROR_TOLERANCE = 1.0e-10
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_BUS_INJECTION_P_SIGMA = 25_000.0
DEFAULT_BUS_INJECTION_Q_SIGMA = 10_000.0
REAL_MEASUREMENT_GROUPS = (
    "voltage magnitude and angle",
    "branch current, real and imaginary",
    "branch active and reactive power",
    "bus-injection active and reactive power",
)
SUPPORTED_CASE_NAMES = ("ieee33", "cigre-mv")
NODE_INJECTION_TERMINAL_TYPE = 9  # MeasuredTerminalType.node without triggering its deprecation warning.


@dataclass(frozen=True)
class CaseExpectations:
    """Optional regression assertions kept outside the reusable builder."""

    pgm_bus_order: tuple[int, ...] | None = None
    real_measurement_group_counts: Mapping[str, int] | None = None
    real_measurement_shape: tuple[int, int] | None = None
    complex_sigma_rtol: float = 5.0e-8
    complex_sigma_atol: float = 1.0e-8


@dataclass(frozen=True)
class SparsityCase:
    """Network data, output identity, and validation policy for one build."""

    name: str
    artifact_prefix: str
    topology: NetworkTopology
    inputs: dict[ComponentType, np.ndarray]
    system_frequency: float = 50.0
    error_tolerance: float = DEFAULT_ERROR_TOLERANCE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    validate_real_sigmas: bool = False
    validate_complex_sigmas: bool = True
    expectations: CaseExpectations = field(default_factory=CaseExpectations)

    @property
    def figure_stem(self) -> str:
        """Return the deterministic case-qualified figure stem."""
        return f"{self.artifact_prefix}_wls_uq_matrix_sparsity"

    @property
    def matrix_archive_name(self) -> str:
        """Return the deterministic case-qualified matrix archive name."""
        return f"{self.artifact_prefix}_wls_uq_matrices.npz"

    @property
    def bus_order_name(self) -> str:
        """Return the deterministic case-qualified PGM-order CSV name."""
        return f"{self.artifact_prefix}_pgm_bus_order.csv"


IEEE33_EXPECTATIONS = CaseExpectations(
    pgm_bus_order=(
        21,
        20,
        19,
        18,
        24,
        23,
        22,
        32,
        31,
        30,
        29,
        28,
        27,
        26,
        25,
        17,
        16,
        15,
        14,
        13,
        12,
        11,
        10,
        9,
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
        0,
    ),
    real_measurement_group_counts=dict(zip(REAL_MEASUREMENT_GROUPS, (8, 64, 64, 64), strict=True)),
    real_measurement_shape=(200, 66),
    complex_sigma_rtol=2.0e-10,
    complex_sigma_atol=1.0e-10,
)

CIGRE_MV_EXPECTATIONS = CaseExpectations(
    pgm_bus_order=(14, 13, 12, 11, 10, 9, 7, 8, 6, 5, 4, 3, 2, 1, 0),
    real_measurement_group_counts=dict(zip(REAL_MEASUREMENT_GROUPS, (8, 24, 24, 26), strict=True)),
    real_measurement_shape=(82, 30),
)


def notebook_cell_source(notebook_path: Path, marker: str) -> str:
    """Return the unique code cell containing ``marker``."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    matches = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and marker in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one cell in {notebook_path.name} containing {marker!r}, found {len(matches)}")
    return matches[0]


def add_complete_load_power_sensors(  # noqa: PLR0913
    inputs: dict[ComponentType, np.ndarray],
    *,
    system_frequency: float = 50.0,
    error_tolerance: float = DEFAULT_ERROR_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    aggregate_p_sigma: float = DEFAULT_BUS_INJECTION_P_SIGMA,
    aggregate_q_sigma: float = DEFAULT_BUS_INJECTION_Q_SIGMA,
) -> dict[ComponentType, np.ndarray]:
    """Add one finite-variance power sensor per active load.

    Per-object sigmas are scaled so independent load sensors at the same bus
    aggregate to the requested bus-level P/Q sigmas.  This case-adapter helper
    deliberately rejects pre-existing load-terminal sensors instead of
    silently changing their effective precision.
    """
    augmented_inputs = {component: values.copy() for component, values in inputs.items()}
    loads = augmented_inputs.get(ComponentType.sym_load)
    if loads is None or not loads.size:
        return augmented_inputs
    active_loads = loads[loads[AttributeType.status].astype(bool)]
    if not active_loads.size:
        return augmented_inputs

    load_bus_ids = active_loads[AttributeType.node].astype(int)
    unique_bus_ids, loads_per_bus = np.unique(load_bus_ids, return_counts=True)
    count_by_bus = dict(zip(unique_bus_ids.tolist(), loads_per_bus.tolist(), strict=True))

    existing_power_sensors = augmented_inputs.get(ComponentType.sym_power_sensor)
    if existing_power_sensors is not None and existing_power_sensors.size:
        existing_terminals = existing_power_sensors[AttributeType.measured_terminal_type]
        if np.any(existing_terminals == MeasuredTerminalType.load):
            raise ValueError("Complete-load sensor augmentation requires no pre-existing load-terminal sensors")

    power_flow = PowerGridModel(augmented_inputs, system_frequency=system_frequency).calculate_power_flow(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        error_tolerance=error_tolerance,
        max_iterations=max_iterations,
    )
    load_output = power_flow[ComponentType.sym_load]
    output_by_id = {int(row[AttributeType.id]): row for row in load_output}
    load_ids = active_loads[AttributeType.id].astype(int)
    if not set(load_ids.tolist()).issubset(output_by_id):
        raise ValueError("Power-flow output is missing one or more active loads")

    load_sensors = initialize_array(DatasetType.input, ComponentType.sym_power_sensor, active_loads.size)
    maximum_id = max(int(np.max(values[AttributeType.id])) for values in augmented_inputs.values() if values.size)
    load_sensors[AttributeType.id] = np.arange(maximum_id + 1, maximum_id + 1 + active_loads.size)
    load_sensors[AttributeType.measured_object] = load_ids
    load_sensors[AttributeType.measured_terminal_type] = MeasuredTerminalType.load
    load_sensors[AttributeType.p_measured] = [float(output_by_id[load_id][AttributeType.p]) for load_id in load_ids]
    load_sensors[AttributeType.q_measured] = [float(output_by_id[load_id][AttributeType.q]) for load_id in load_ids]
    load_sensors[AttributeType.p_sigma] = [
        aggregate_p_sigma / np.sqrt(count_by_bus[int(bus_id)]) for bus_id in load_bus_ids
    ]
    load_sensors[AttributeType.q_sigma] = [
        aggregate_q_sigma / np.sqrt(count_by_bus[int(bus_id)]) for bus_id in load_bus_ids
    ]
    augmented_inputs[ComponentType.sym_power_sensor] = (
        load_sensors if existing_power_sensors is None else np.concatenate((existing_power_sensors, load_sensors))
    )
    assert_valid_input_data(
        augmented_inputs,
        calculation_type=CalculationType.state_estimation,
        symmetric=True,
    )
    return augmented_inputs


def load_notebook_case_inputs(notebook_path: Path) -> tuple[NetworkTopology, dict[ComponentType, np.ndarray]]:
    """Execute the trusted topology and sensor-construction cells of a PGM example."""
    namespace = {
        "AngleMeasurementType": AngleMeasurementType,
        "AttributeType": AttributeType,
        "CalculationMethod": CalculationMethod,
        "CalculationType": CalculationType,
        "ComponentType": ComponentType,
        "DatasetType": DatasetType,
        "MeasuredTerminalType": MeasuredTerminalType,
        "PowerGridModel": PowerGridModel,
        "assert_valid_input_data": assert_valid_input_data,
        "initialize_array": initialize_array,
        "np": np,
        "pd": pd,
    }
    for marker in ("workshop_model =", "def make_sensor_input"):
        source = notebook_cell_source(notebook_path, marker)
        exec(compile(source, str(notebook_path), "exec"), namespace)

    topology = namespace["workshop_model"]
    inputs = namespace["state_estimation_inputs"]["voltage + current + power"]
    return topology, inputs


def load_case(case_name: str) -> SparsityCase:
    """Load one of the reproducible notebook-backed command-line cases."""
    case_definitions = {
        "ieee33": ("IEEE 33", "IEEE33 State Estimation UQ Example.ipynb", "ieee33", IEEE33_EXPECTATIONS),
        "cigre-mv": (
            "CIGRE MV",
            "CIGRE MV State Estimation UQ Example.ipynb",
            "cigre_mv",
            CIGRE_MV_EXPECTATIONS,
        ),
    }
    try:
        name, notebook_name, artifact_prefix, expectations = case_definitions[case_name]
    except KeyError as error:
        raise ValueError(f"Unsupported case {case_name!r}; choose from {SUPPORTED_CASE_NAMES}") from error
    topology, inputs = load_notebook_case_inputs(EXAMPLE_ROOT / notebook_name)
    inputs = add_complete_load_power_sensors(inputs)
    return SparsityCase(
        name=name,
        artifact_prefix=artifact_prefix,
        topology=topology,
        inputs=inputs,
        validate_real_sigmas=case_name == "ieee33",
        validate_complex_sigmas=True,
        expectations=expectations,
    )


def _combined_power_sensor_variance(
    sensor_indices: list[int],
    p_sigmas: np.ndarray,
    q_sigmas: np.ndarray,
    base_power_3p: float,
) -> tuple[float, float] | None:
    """Combine repeated P/Q sensors using PGM's inverse-variance rule."""
    if not sensor_indices:
        return None
    positions = np.asarray(sensor_indices, dtype=int)
    normalized_p_variances = np.square(p_sigmas[positions] / base_power_3p)
    normalized_q_variances = np.square(q_sigmas[positions] / base_power_3p)
    p_inverse_variance = float(np.sum(np.where(np.isfinite(normalized_p_variances), 1.0 / normalized_p_variances, 0.0)))
    q_inverse_variance = float(np.sum(np.where(np.isfinite(normalized_q_variances), 1.0 / normalized_q_variances, 0.0)))
    if p_inverse_variance <= 0.0 or q_inverse_variance <= 0.0:
        return None
    return 1.0 / p_inverse_variance, 1.0 / q_inverse_variance


def aggregate_bus_injection_variances(
    topology: NetworkTopology,
    inputs: dict[ComponentType, np.ndarray],
    *,
    base_power_3p: float = BASE_POWER_3P,
) -> dict[int, tuple[float, float]]:
    """Mirror PGM's finite, direct, appliance, and exact bus-injection choices."""
    node_ids = [int(row["id"]) for row in topology["node"]]
    power_sensors = inputs.get(ComponentType.sym_power_sensor)
    sensors_by_terminal_object: dict[tuple[int, int], list[int]] = {}
    if power_sensors is not None and power_sensors.size:
        p_sigmas, q_sigmas = _power_sigmas(power_sensors)
        for index, sensor in enumerate(power_sensors):
            key = (
                int(sensor[AttributeType.measured_terminal_type]),
                int(sensor[AttributeType.measured_object]),
            )
            sensors_by_terminal_object.setdefault(key, []).append(index)
    else:
        p_sigmas = np.empty(0)
        q_sigmas = np.empty(0)

    appliance_definitions = (
        ("sym_load", MeasuredTerminalType.load),
        ("sym_gen", MeasuredTerminalType.generator),
        ("source", MeasuredTerminalType.source),
    )
    appliances_by_bus: dict[int, list[tuple[int, int]]] = {node_id: [] for node_id in node_ids}
    for component_name, terminal_type in appliance_definitions:
        for appliance in topology.get(component_name, []):
            if int(appliance["status"]):
                appliances_by_bus[int(appliance["node"])].append((int(terminal_type), int(appliance["id"])))

    result: dict[int, tuple[float, float]] = {}
    for bus_id in node_ids:
        appliance_variance = np.zeros(2, dtype=float)
        unmeasured_appliance_count = 0
        for key in appliances_by_bus[bus_id]:
            combined = _combined_power_sensor_variance(
                sensors_by_terminal_object.get(key, []), p_sigmas, q_sigmas, base_power_3p
            )
            if combined is None:
                unmeasured_appliance_count += 1
            else:
                appliance_variance += combined

        direct = _combined_power_sensor_variance(
            sensors_by_terminal_object.get((NODE_INJECTION_TERMINAL_TYPE, bus_id), []),
            p_sigmas,
            q_sigmas,
            base_power_3p,
        )
        if unmeasured_appliance_count:
            if direct is not None:
                result[bus_id] = direct
            continue
        if direct is None or np.any(appliance_variance == 0.0):
            result[bus_id] = (float(appliance_variance[0]), float(appliance_variance[1]))
            continue
        direct_variance = np.asarray(direct)
        combined_variance = 1.0 / (1.0 / appliance_variance + 1.0 / direct_variance)
        result[bus_id] = (float(combined_variance[0]), float(combined_variance[1]))
    return result


def current_component_variances(sensor: np.void, rated_voltage: float, base_power_3p: float) -> tuple[float, float]:
    """Mirror PGM's second-order polar-to-Cartesian current variances."""
    base_current = base_power_3p / (np.sqrt(3.0) * rated_voltage)
    magnitude = float(sensor[AttributeType.i_measured]) / base_current
    magnitude_variance = (float(sensor[AttributeType.i_sigma]) / base_current) ** 2
    angle = float(sensor[AttributeType.i_angle_measured])
    angle_variance = float(sensor[AttributeType.i_angle_sigma]) ** 2
    cosine_squared = np.cos(angle) ** 2
    sine_squared = np.sin(angle) ** 2
    magnitude_squared = magnitude**2

    real_variance = (
        magnitude_variance * cosine_squared
        + magnitude_squared * angle_variance * sine_squared
        + 0.5 * magnitude_squared * angle_variance**2 * cosine_squared
        + magnitude_variance * angle_variance * sine_squared
    )
    imaginary_variance = (
        magnitude_variance * sine_squared
        + magnitude_squared * angle_variance * cosine_squared
        + 0.5 * magnitude_squared * angle_variance**2 * sine_squared
        + magnitude_variance * angle_variance * cosine_squared
    )
    return float(real_variance), float(imaginary_variance)


def combined_current_sensor_variance(
    sensors: list[np.void],
    rated_voltage: float,
    *,
    base_power_3p: float = BASE_POWER_3P,
) -> float:
    """Combine repeated current sensors channel-wise, then sum both channels."""
    component_variances = np.asarray(
        [current_component_variances(sensor, rated_voltage, base_power_3p) for sensor in sensors]
    )
    if not component_variances.size or np.any(~np.isfinite(component_variances)) or np.any(component_variances <= 0.0):
        raise ValueError("Current sensors require positive finite Cartesian component variances")
    combined_component_variances = 1.0 / np.sum(1.0 / component_variances, axis=0)
    return float(np.sum(combined_component_variances))


def topology_branches(topology: NetworkTopology) -> dict[int, dict[str, TopologyValue]]:
    """Return ordinary line and two-winding-transformer rows keyed by ID."""
    branches: dict[int, dict[str, TopologyValue]] = {}
    for component_name in ("line", "transformer"):
        for branch in topology.get(component_name, []):
            branch_id = int(branch["id"])
            if branch_id in branches:
                raise ValueError(f"Line and transformer IDs overlap at {branch_id}")
            branches[branch_id] = branch
    return branches


def pgm_radial_bus_order(
    topology: NetworkTopology,
    *,
    expected_order: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Mirror PGM's source-rooted DFS reversal for a connected radial case."""
    input_bus_ids = [int(row["id"]) for row in topology["node"]]
    input_position = {bus_id: position for position, bus_id in enumerate(input_bus_ids)}
    active_sources = [row for row in topology.get("source", []) if int(row["status"])]
    if len(active_sources) != 1:
        raise NotImplementedError(
            f"The generalized sparsity builder currently requires one active source, found {len(active_sources)}"
        )

    adjacency: list[list[int]] = [[] for _ in input_bus_ids]
    active_edge_count = 0
    for component_name in ("line", "transformer"):
        for branch in topology.get(component_name, []):
            if not int(branch["from_status"]) or not int(branch["to_status"]):
                continue
            from_position = input_position[int(branch["from_node"])]
            to_position = input_position[int(branch["to_node"])]
            adjacency[from_position].append(to_position)
            adjacency[to_position].append(from_position)
            active_edge_count += 1
    if active_edge_count != len(input_bus_ids) - 1:
        raise NotImplementedError(
            "The generalized complex-matrix reconstruction currently requires a connected radial topology"
        )

    source_position = input_position[int(active_sources[0]["node"])]
    visited = np.zeros(len(input_bus_ids), dtype=bool)
    preorder: list[int] = []

    def visit(position: int) -> None:
        visited[position] = True
        preorder.append(position)
        for neighbour in adjacency[position]:
            if not visited[neighbour]:
                visit(neighbour)

    visit(source_position)
    if not np.all(visited):
        raise NotImplementedError("The generalized complex-matrix reconstruction does not support disconnected cases")

    solver_positions = np.array(preorder[::-1], dtype=int)
    solver_bus_ids = np.array([input_bus_ids[position] for position in solver_positions], dtype=int)
    if expected_order is not None:
        np.testing.assert_array_equal(solver_bus_ids, np.asarray(expected_order, dtype=int))

    remaining = set(range(len(input_bus_ids)))
    for position in solver_positions:
        future_neighbours = remaining.intersection(adjacency[position])
        if len(future_neighbours) > 1:
            raise ValueError("PGM's reversed radial DFS unexpectedly creates symbolic fill")
        remaining.remove(int(position))
    return solver_bus_ids


def assemble_matrices(case: SparsityCase) -> tuple[dict[str, np.ndarray], float, np.ndarray, np.ndarray]:
    """Reconstruct PGM's ordered complex ILSE matrices for a supported case."""
    topology = case.topology
    inputs = case.inputs
    unsupported_components = {
        "link": ComponentType.link,
        "generic_branch": ComponentType.generic_branch,
        "three_winding_transformer": ComponentType.three_winding_transformer,
        "shunt": ComponentType.shunt,
    }
    for unsupported_name, component_type in unsupported_components.items():
        component_input = inputs.get(component_type)
        if topology.get(unsupported_name) or (component_input is not None and component_input.size):
            raise NotImplementedError(f"Complex sparsity reconstruction does not support {unsupported_name!r}")

    nodes = topology["node"]
    input_bus_ids = np.asarray([int(row["id"]) for row in nodes], dtype=int)
    bus_index = {int(bus_id): index for index, bus_id in enumerate(input_bus_ids)}
    rated_voltage = {int(row["id"]): float(row["u_rated"]) for row in nodes}
    n_bus = input_bus_ids.size
    input_node_ids = inputs[ComponentType.node][AttributeType.id].astype(int)
    np.testing.assert_array_equal(input_node_ids, input_bus_ids)

    line_admittances = _line_terminal_admittances(inputs, case.system_frequency, bus_index)
    transformer_admittances = _transformer_terminal_admittances(inputs, bus_index)
    if line_admittances.keys() & transformer_admittances.keys():
        raise ValueError("Line and transformer IDs must be globally unique")
    branch_admittances = line_admittances | transformer_admittances
    branch_rows = topology_branches(topology)
    if set(branch_admittances) != set(branch_rows):
        raise ValueError("Topology rows and PGM input arrays contain different ordinary branches")

    y_bus_input = np.zeros((n_bus, n_bus), dtype=np.complex128)
    for from_index, to_index, y_ff, y_ft, y_tf, y_tt in branch_admittances.values():
        y_bus_input[from_index, from_index] += y_ff
        y_bus_input[from_index, to_index] += y_ft
        y_bus_input[to_index, from_index] += y_tf
        y_bus_input[to_index, to_index] += y_tt

    solver_bus_ids = pgm_radial_bus_order(
        topology,
        expected_order=case.expectations.pgm_bus_order,
    )
    solver_rank = {int(bus_id): rank for rank, bus_id in enumerate(solver_bus_ids)}
    solver_positions = np.asarray([bus_index[int(bus_id)] for bus_id in solver_bus_ids], dtype=int)

    def terminal_row(branch_id: int, terminal_type: int) -> tuple[np.ndarray, int, int, str]:
        try:
            from_index, to_index, y_ff, y_ft, y_tf, y_tt = branch_admittances[branch_id]
        except KeyError as error:
            raise NotImplementedError(f"Sensor refers to unsupported branch {branch_id}") from error
        row = np.zeros(n_bus, dtype=np.complex128)
        if terminal_type == int(MeasuredTerminalType.branch_from):
            row[from_index] = y_ff
            row[to_index] = y_ft
            return row, from_index, to_index, "from"
        if terminal_type == int(MeasuredTerminalType.branch_to):
            row[from_index] = y_tf
            row[to_index] = y_tt
            return row, to_index, from_index, "to"
        raise NotImplementedError(f"Unsupported ordinary-branch terminal type {terminal_type}")

    sortable_rows: list[tuple[tuple[int, int, int, int], np.ndarray, float, str]] = []
    voltage_sensors = inputs.get(ComponentType.sym_voltage_sensor)
    if voltage_sensors is not None:
        voltage_sensor_indices: dict[int, list[int]] = {}
        for index, sensor in enumerate(voltage_sensors):
            voltage_sensor_indices.setdefault(int(sensor[AttributeType.measured_object]), []).append(index)
        for bus_id, indices in voltage_sensor_indices.items():
            normalized_sigmas = np.asarray(
                [float(voltage_sensors[index][AttributeType.u_sigma]) / rated_voltage[bus_id] for index in indices]
            )
            finite_sigmas = normalized_sigmas[np.isfinite(normalized_sigmas)]
            if not finite_sigmas.size:
                continue
            if np.any(finite_sigmas <= 0.0):
                raise ValueError(f"Voltage sensors at bus {bus_id} require positive sigma values")
            variance = 1.0 / float(np.sum(1.0 / np.square(finite_sigmas)))
            row = np.zeros(n_bus, dtype=np.complex128)
            row[bus_index[bus_id]] = 1.0
            sortable_rows.append(
                ((0, solver_rank[bus_id], solver_rank[bus_id], bus_id), row, variance, "voltage phasor")
            )

    current_sensors = inputs.get(ComponentType.sym_current_sensor)
    if current_sensors is not None:
        grouped_current_sensors: dict[tuple[int, int], list[np.void]] = {}
        for sensor in current_sensors:
            magnitude_sigma = float(sensor[AttributeType.i_sigma])
            angle_sigma = float(sensor[AttributeType.i_angle_sigma])
            if np.isposinf(magnitude_sigma) or np.isposinf(angle_sigma):
                continue
            if int(sensor[AttributeType.angle_measurement_type]) != int(AngleMeasurementType.global_angle):
                raise NotImplementedError("Complex sparsity reconstruction supports only global-angle current sensors")
            magnitude = float(sensor[AttributeType.i_measured])
            angle = float(sensor[AttributeType.i_angle_measured])
            if not np.isfinite(magnitude) or not np.isfinite(angle):
                raise ValueError("Current sensors require finite measured magnitude and angle values")
            if not np.isfinite(magnitude_sigma) or magnitude_sigma <= 0.0:
                raise ValueError("Current sensors require positive finite magnitude sigma values")
            if not np.isfinite(angle_sigma) or angle_sigma <= 0.0:
                raise ValueError("Current sensors require positive finite angle sigma values")
            key = (
                int(sensor[AttributeType.measured_object]),
                int(sensor[AttributeType.measured_terminal_type]),
            )
            grouped_current_sensors.setdefault(key, []).append(sensor)
        for (branch_id, terminal_type), sensors in grouped_current_sensors.items():
            row, local_index, remote_index, terminal_name = terminal_row(branch_id, terminal_type)
            if row[remote_index] == 0.0:
                continue
            variance = combined_current_sensor_variance(
                sensors,
                rated_voltage[int(input_bus_ids[local_index])],
            )
            sortable_rows.append(
                (
                    (
                        1,
                        solver_rank[int(input_bus_ids[local_index])],
                        solver_rank[int(input_bus_ids[remote_index])],
                        branch_id,
                    ),
                    row,
                    variance,
                    f"{terminal_name} current phasor",
                )
            )

    power_sensors = inputs.get(ComponentType.sym_power_sensor)
    if power_sensors is not None:
        p_sigmas, q_sigmas = _power_sigmas(power_sensors)
        grouped_branch_power_indices: dict[tuple[int, int], list[int]] = {}
        injection_terminals = {
            int(MeasuredTerminalType.load),
            int(MeasuredTerminalType.generator),
            int(MeasuredTerminalType.source),
            NODE_INJECTION_TERMINAL_TYPE,
        }
        for index, sensor in enumerate(power_sensors):
            terminal_type = int(sensor[AttributeType.measured_terminal_type])
            if terminal_type in injection_terminals:
                continue
            if terminal_type not in {
                int(MeasuredTerminalType.branch_from),
                int(MeasuredTerminalType.branch_to),
            }:
                raise NotImplementedError(f"Unsupported power-sensor terminal type {terminal_type}")
            key = (int(sensor[AttributeType.measured_object]), terminal_type)
            grouped_branch_power_indices.setdefault(key, []).append(index)
        for (branch_id, terminal_type), indices in grouped_branch_power_indices.items():
            row, local_index, remote_index, terminal_name = terminal_row(branch_id, terminal_type)
            if row[remote_index] == 0.0:
                continue
            combined = _combined_power_sensor_variance(indices, p_sigmas, q_sigmas, BASE_POWER_3P)
            if combined is None:
                continue
            variance = combined[0] + combined[1]
            sortable_rows.append(
                (
                    (
                        2,
                        solver_rank[int(input_bus_ids[local_index])],
                        solver_rank[int(input_bus_ids[remote_index])],
                        branch_id,
                    ),
                    row,
                    variance,
                    f"{terminal_name} power as current",
                )
            )

    sortable_rows.sort(key=lambda value: value[0])
    if not sortable_rows:
        raise ValueError("The complex reconstruction found no active finite-variance direct measurements")
    complex_measurement_matrix_input = np.vstack([value[1] for value in sortable_rows])
    measurement_variances = np.asarray([value[2] for value in sortable_rows])
    measurement_row_groups = np.asarray([value[3] for value in sortable_rows])

    injection_component_variances = aggregate_bus_injection_variances(
        topology,
        inputs,
        base_power_3p=BASE_POWER_3P,
    )
    injection_variances = {
        bus_id: p_variance + q_variance for bus_id, (p_variance, q_variance) in injection_component_variances.items()
    }
    positive_variances = [
        *measurement_variances[np.isfinite(measurement_variances) & (measurement_variances > 0.0)].tolist(),
        *(variance for variance in injection_variances.values() if np.isfinite(variance) and variance > 0.0),
    ]
    if not positive_variances:
        raise ValueError("The complex reconstruction requires at least one positive finite measurement variance")
    variance_normalization = min(positive_variances)
    normalized_complex_weights = variance_normalization / measurement_variances
    complex_gain_input = complex_measurement_matrix_input.conj().T @ (
        normalized_complex_weights[:, np.newaxis] * complex_measurement_matrix_input
    )

    y_bus = y_bus_input[np.ix_(solver_positions, solver_positions)]
    complex_measurement_matrix = complex_measurement_matrix_input[:, solver_positions]
    complex_gain_block = complex_gain_input[np.ix_(solver_positions, solver_positions)]
    constraint_matrix = np.zeros_like(y_bus)
    auxiliary_matrix = -np.eye(n_bus, dtype=np.complex128)
    for bus_id, variance in injection_variances.items():
        solver_position = solver_rank[bus_id]
        constraint_matrix[solver_position] = y_bus[solver_position]
        auxiliary_matrix[solver_position, solver_position] = -variance / variance_normalization

    np.testing.assert_allclose(complex_gain_block, complex_gain_block.conj().T, rtol=1.0e-13, atol=1.0e-12)
    measured_injection_mask = np.asarray([int(bus_id) in injection_variances for bus_id in solver_bus_ids], dtype=bool)
    np.testing.assert_allclose(
        constraint_matrix[measured_injection_mask],
        y_bus[measured_injection_mask],
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_array_equal(
        constraint_matrix[~measured_injection_mask],
        np.zeros((int(np.count_nonzero(~measured_injection_mask)), n_bus), dtype=np.complex128),
    )

    augmented = np.block([[complex_gain_block, constraint_matrix.conj().T], [constraint_matrix, auxiliary_matrix]])
    np.testing.assert_allclose(augmented, augmented.conj().T, rtol=1.0e-13, atol=1.0e-12)
    shuffle = np.column_stack((np.arange(n_bus), np.arange(n_bus) + n_bus)).ravel()
    augmented_ps = augmented[np.ix_(shuffle, shuffle)]
    augmented_inverse = np.linalg.inv(augmented)
    augmented_ps_inverse = np.linalg.inv(augmented_ps)
    np.testing.assert_allclose(
        augmented_ps_inverse,
        augmented_inverse[np.ix_(shuffle, shuffle)],
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    matrices = {
        "complex_branch_measurement_matrix": complex_measurement_matrix,
        "y_bus": y_bus,
        "complex_gain_block": complex_gain_block,
        "constraint_matrix": constraint_matrix,
        "auxiliary_matrix": auxiliary_matrix,
        "augmented": augmented,
        "augmented_ps": augmented_ps,
        "augmented_inverse": augmented_inverse,
        "augmented_ps_inverse": augmented_ps_inverse,
        "normalized_complex_branch_weights": normalized_complex_weights,
        "complex_measurement_row_groups": measurement_row_groups,
        "measured_injection_mask": measured_injection_mask,
        "injection_variances": np.asarray([injection_variances.get(int(bus_id), np.nan) for bus_id in solver_bus_ids]),
        "injection_p_variances": np.asarray(
            [injection_component_variances.get(int(bus_id), (np.nan, np.nan))[0] for bus_id in solver_bus_ids]
        ),
        "injection_q_variances": np.asarray(
            [injection_component_variances.get(int(bus_id), (np.nan, np.nan))[1] for bus_id in solver_bus_ids]
        ),
    }
    return matrices, variance_normalization, shuffle, solver_bus_ids


def real_measurement_group(label: str) -> str:
    """Map the conventional real measurement labels to figure row groups."""
    if label.startswith("voltage "):
        return "voltage magnitude and angle"
    if " current at " in label:
        return "branch current, real and imaginary"
    if " power at " in label:
        return "branch active and reactive power"
    raise ValueError(f"Unexpected real measurement row label: {label}")


def real_measurement_sort_key(
    label: str,
    solver_rank: dict[int, int],
    branches: dict[int, dict[str, TopologyValue]],
) -> tuple[int, int, int, int, int]:
    """Sort real rows by group, PGM bus rank, component, and channel."""
    group = real_measurement_group(label)
    group_rank = REAL_MEASUREMENT_GROUPS.index(group)
    if label.startswith("voltage "):
        bus_id = int(label.rsplit(" ", maxsplit=1)[1])
        channel_rank = 0 if label.startswith("voltage magnitude") else 1
        return group_rank, solver_rank[bus_id], solver_rank[bus_id], bus_id, channel_rank

    words = label.split()
    try:
        component_word = "line" if "line" in words else "transformer"
        branch_id = int(words[words.index(component_word) + 1])
        branch = branches[branch_id]
    except (KeyError, ValueError) as error:
        raise ValueError(f"Could not extract an ordinary branch from measurement label: {label}") from error
    from_bus = int(branch["from_node"])
    to_bus = int(branch["to_node"])
    local_bus, remote_bus = (from_bus, to_bus) if "from terminal" in label else (to_bus, from_bus)
    if group == "branch current, real and imaginary":
        channel_rank = 0 if label.startswith("real current") else 1
        return group_rank, solver_rank[local_bus], solver_rank[remote_bus], branch_id, channel_rank
    channel_rank = 0 if label.startswith("active power") else 1
    return group_rank, solver_rank[local_bus], solver_rank[remote_bus], branch_id, channel_rank


def select_full_measurement_angle_references(
    measurement_matrix: np.ndarray,
    solver_bus_ids: np.ndarray,
    inputs: dict[ComponentType, np.ndarray],
) -> tuple[int, ...]:
    """Choose one source-preferred reference per unanchored angle component."""
    n_bus = solver_bus_ids.size
    angle_neighbours: list[set[int]] = [set() for _ in range(n_bus)]
    anchored_angle_nodes: set[int] = set()
    for row in measurement_matrix:
        derivatives = row[:n_bus]
        scale = float(np.max(np.abs(derivatives)))
        if scale == 0.0:
            continue
        active_nodes = np.flatnonzero(np.abs(derivatives) > DEFAULT_RELATIVE_THRESHOLD * scale).tolist()
        first_node = active_nodes[0]
        for node in active_nodes[1:]:
            angle_neighbours[first_node].add(node)
            angle_neighbours[node].add(first_node)
        if abs(float(np.sum(derivatives))) > DEFAULT_RELATIVE_THRESHOLD * float(np.sum(np.abs(derivatives))):
            anchored_angle_nodes.update(active_nodes)

    solver_rank = {int(bus_id): position for position, bus_id in enumerate(solver_bus_ids)}
    preferred_source_positions: list[int] = []
    sources = inputs.get(ComponentType.source)
    if sources is not None:
        for source in sources:
            if not bool(source[AttributeType.status]):
                continue
            try:
                position = solver_rank[int(source[AttributeType.node])]
            except KeyError as error:
                raise ValueError(f"Source refers to unknown node {error.args[0]}") from error
            if position not in preferred_source_positions:
                preferred_source_positions.append(position)

    reference_positions = _select_fixed_angle_references(
        solver_bus_ids,
        angle_neighbours,
        anchored_angle_nodes,
        tuple(preferred_source_positions),
    )
    return tuple(int(solver_bus_ids[position]) for position in reference_positions)


def build_real_gain_matrices(
    case: SparsityCase,
    complex_matrices: dict[str, np.ndarray],
    solver_bus_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Build the all-measurement real-polar H, gain, and UQ gain inverse."""
    topology = case.topology
    inputs = case.inputs
    result = PowerGridModel(inputs, system_frequency=case.system_frequency).calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod.newton_raphson,
        calculate_uncertainty=True,
        error_tolerance=case.error_tolerance,
        max_iterations=case.max_iterations,
    )

    branch_inputs = {component: values.copy() for component, values in inputs.items()}
    power_sensors = branch_inputs.get(ComponentType.sym_power_sensor)
    if power_sensors is not None:
        power_terminals = power_sensors[AttributeType.measured_terminal_type]
        branch_sensor_mask = (power_terminals == MeasuredTerminalType.branch_from) | (
            power_terminals == MeasuredTerminalType.branch_to
        )
        branch_inputs[ComponentType.sym_power_sensor] = power_sensors[branch_sensor_mask].copy()
    current_sensors = branch_inputs.get(ComponentType.sym_current_sensor)
    if current_sensors is not None:
        disabled_current_mask = np.isposinf(current_sensors[AttributeType.i_sigma]) | np.isposinf(
            current_sensors[AttributeType.i_angle_sigma]
        )
        branch_inputs[ComponentType.sym_current_sensor] = current_sensors[~disabled_current_mask].copy()

    branch_model = build_measurement_model(
        branch_inputs,
        system_frequency=case.system_frequency,
        error_tolerance=case.error_tolerance,
        max_iterations=case.max_iterations,
        state_estimation_result=result,
        retain_all_angle_columns=True,
    )

    input_position = {int(bus_id): position for position, bus_id in enumerate(branch_model.node_ids)}
    n_bus = solver_bus_ids.size
    solver_positions = np.asarray([input_position[int(bus_id)] for bus_id in solver_bus_ids], dtype=int)
    state_permutation = np.concatenate((solver_positions, n_bus + solver_positions))
    branch_h_full = branch_model.measurement_matrix.to_dense(max_dense_matrix_bytes=None)[:, state_permutation]
    branch_weights = branch_model.weight_matrix.diagonal.copy()
    branch_labels = np.array(branch_model.measurement_labels)
    branch_groups = np.array([real_measurement_group(label) for label in branch_model.measurement_labels])
    solver_rank = {int(bus_id): rank for rank, bus_id in enumerate(solver_bus_ids)}
    branches = topology_branches(topology)
    branch_row_order = np.array(
        sorted(
            range(branch_labels.size),
            key=lambda index: real_measurement_sort_key(str(branch_labels[index]), solver_rank, branches),
        ),
        dtype=int,
    )
    branch_h_full = branch_h_full[branch_row_order]
    branch_weights = branch_weights[branch_row_order]
    branch_labels = branch_labels[branch_row_order]
    branch_groups = branch_groups[branch_row_order]

    node_output = result[ComponentType.node]
    output_by_id = {int(row[AttributeType.id]): row for row in node_output}
    voltage_magnitudes = np.array([float(output_by_id[int(bus_id)][AttributeType.u_pu]) for bus_id in solver_bus_ids])
    voltage_angles = np.array([float(output_by_id[int(bus_id)][AttributeType.u_angle]) for bus_id in solver_bus_ids])
    voltage_phasors = voltage_magnitudes * np.exp(1.0j * voltage_angles)
    y_bus = complex_matrices["y_bus"]
    injection_component_variances = aggregate_bus_injection_variances(
        topology,
        inputs,
        base_power_3p=BASE_POWER_3P,
    )
    injection_rows: list[np.ndarray] = []
    injection_weights: list[float] = []
    injection_labels: list[str] = []
    injection_channels: list[tuple[int, int]] = []
    for bus_position, bus_id_value in enumerate(solver_bus_ids):
        bus_id = int(bus_id_value)
        if bus_id not in injection_component_variances:
            continue
        power_terms = voltage_phasors[bus_position] * np.conj(y_bus[bus_position] * voltage_phasors)
        bus_power = np.sum(power_terms)
        diagonal_power = power_terms[bus_position]

        angle_derivative = -1.0j * power_terms
        angle_derivative[bus_position] = 1.0j * (bus_power - diagonal_power)
        magnitude_derivative = power_terms / voltage_magnitudes
        magnitude_derivative[bus_position] = (bus_power + diagonal_power) / voltage_magnitudes[bus_position]
        complex_derivative = np.concatenate((angle_derivative, magnitude_derivative))
        p_variance, q_variance = injection_component_variances[bus_id]
        if np.isfinite(p_variance) and p_variance > 0.0:
            injection_rows.append(np.real(complex_derivative))
            injection_weights.append(1.0 / p_variance)
            injection_labels.append(f"active-power injection at node {bus_id}")
            injection_channels.append((bus_position, 0))
        if np.isfinite(q_variance) and q_variance > 0.0:
            injection_rows.append(np.imag(complex_derivative))
            injection_weights.append(1.0 / q_variance)
            injection_labels.append(f"reactive-power injection at node {bus_id}")
            injection_channels.append((bus_position, 1))

    full_state_order = 2 * n_bus
    injection_h_full = np.vstack(injection_rows) if injection_rows else np.empty((0, full_state_order))
    injection_groups = np.full(
        injection_h_full.shape[0],
        "bus-injection active and reactive power",
    )
    real_h_full = np.vstack((branch_h_full, injection_h_full))
    real_weights = np.concatenate((branch_weights, np.asarray(injection_weights)))
    real_labels = np.concatenate((branch_labels, np.asarray(injection_labels)))
    real_groups = np.concatenate((branch_groups, injection_groups))

    fixed_angle_reference_node_ids = select_full_measurement_angle_references(real_h_full, solver_bus_ids, inputs)
    fixed_angle_node_ids = set(fixed_angle_reference_node_ids)
    retained_angle_positions = np.asarray(
        [position for position, bus_id in enumerate(solver_bus_ids) if int(bus_id) not in fixed_angle_node_ids],
        dtype=int,
    )
    retained_state_columns = np.concatenate((retained_angle_positions, n_bus + np.arange(n_bus)))
    real_h = real_h_full[:, retained_state_columns]
    injection_h = injection_h_full[:, retained_state_columns]
    ordered_angle_bus_ids = [int(solver_bus_ids[position]) for position in retained_angle_positions]
    state_order = real_h.shape[1]

    observed_group_counts = {group: int(np.count_nonzero(real_groups == group)) for group in REAL_MEASUREMENT_GROUPS}
    expected_group_counts = case.expectations.real_measurement_group_counts
    if expected_group_counts is not None and observed_group_counts != dict(expected_group_counts):
        raise ValueError(
            f"Unexpected real measurement groups: {observed_group_counts}; expected {dict(expected_group_counts)}"
        )
    expected_shape = case.expectations.real_measurement_shape
    if expected_shape is not None and real_h.shape != expected_shape:
        raise ValueError(f"Unexpected real measurement shape {real_h.shape}; expected {expected_shape}")

    state = np.concatenate((voltage_angles[retained_angle_positions], voltage_magnitudes))

    def evaluate_injections(candidate_state: np.ndarray) -> np.ndarray:
        angle_count = retained_angle_positions.size
        candidate_angles = voltage_angles.copy()
        candidate_angles[retained_angle_positions] = candidate_state[:angle_count]
        candidate_magnitudes = candidate_state[angle_count:]
        candidate_phasors = candidate_magnitudes * np.exp(1.0j * candidate_angles)
        candidate_injections = candidate_phasors * np.conj(y_bus @ candidate_phasors)
        return np.asarray(
            [
                float(np.real(candidate_injections[position]))
                if component == 0
                else float(np.imag(candidate_injections[position]))
                for position, component in injection_channels
            ]
        )

    finite_difference_h = np.empty_like(injection_h)
    finite_difference_step = 1.0e-7
    for column in range(state.size):
        positive_state = state.copy()
        negative_state = state.copy()
        positive_state[column] += finite_difference_step
        negative_state[column] -= finite_difference_step
        finite_difference_h[:, column] = (evaluate_injections(positive_state) - evaluate_injections(negative_state)) / (
            2.0 * finite_difference_step
        )
    maximum_injection_jacobian_error = (
        float(np.max(np.abs(injection_h - finite_difference_h))) if injection_h.size else 0.0
    )
    if injection_h.size:
        np.testing.assert_allclose(injection_h, finite_difference_h, rtol=5.0e-7, atol=2.0e-7)

    real_gain = real_h.T @ (real_weights[:, np.newaxis] * real_h)
    np.testing.assert_allclose(real_gain, real_gain.T, rtol=1.0e-13, atol=1.0e-8)
    if np.linalg.matrix_rank(real_gain) != state_order:
        raise ValueError(f"The all-measurement {case.name} real WLS gain is singular")
    real_gain_inverse = np.linalg.inv(real_gain)
    real_gain_inverse = 0.5 * (real_gain_inverse + real_gain_inverse.T)
    absolute_inverse_residual = float(np.linalg.norm(real_gain @ real_gain_inverse - np.eye(state_order), ord=np.inf))
    relative_inverse_residual = absolute_inverse_residual / (
        np.linalg.norm(real_gain, ord=np.inf) * np.linalg.norm(real_gain_inverse, ord=np.inf)
    )
    if relative_inverse_residual > MAX_RELATIVE_INVERSE_RESIDUAL:
        raise ValueError(f"Real gain inverse has relative residual {relative_inverse_residual:.3e}")

    covariance_diagonal = np.diag(real_gain_inverse)
    angle_count = len(ordered_angle_bus_ids)
    predicted_angle_sigma = np.sqrt(np.maximum(covariance_diagonal[:angle_count], 0.0))
    predicted_magnitude_sigma = np.sqrt(np.maximum(covariance_diagonal[angle_count:], 0.0))
    reported_angle_sigma = np.array(
        [float(output_by_id[bus_id][AttributeType.u_angle_sigma]) for bus_id in ordered_angle_bus_ids]
    )
    reported_magnitude_sigma = np.array(
        [float(output_by_id[int(bus_id)][AttributeType.u_pu_sigma]) for bus_id in solver_bus_ids]
    )
    maximum_angle_sigma_error = float("nan")
    maximum_magnitude_sigma_error = float("nan")
    if case.validate_real_sigmas:
        np.testing.assert_allclose(predicted_angle_sigma, reported_angle_sigma, rtol=1.0e-9, atol=1.0e-12)
        np.testing.assert_allclose(predicted_magnitude_sigma, reported_magnitude_sigma, rtol=1.0e-9, atol=1.0e-12)
        maximum_angle_sigma_error = float(np.max(np.abs(predicted_angle_sigma - reported_angle_sigma)))
        maximum_magnitude_sigma_error = float(np.max(np.abs(predicted_magnitude_sigma - reported_magnitude_sigma)))

    real_matrices = {
        "real_measurement_matrix": real_h,
        "real_weight_diagonal": real_weights,
        "real_gain": real_gain,
        "real_gain_inverse": real_gain_inverse,
        "real_measurement_labels": real_labels,
        "real_measurement_row_groups": real_groups,
        "real_measurement_group_names": np.asarray(REAL_MEASUREMENT_GROUPS),
        "real_measurement_group_counts": np.asarray(
            [observed_group_counts[group] for group in REAL_MEASUREMENT_GROUPS], dtype=int
        ),
        "real_angle_state_count": np.asarray(angle_count),
        "fixed_angle_reference_node_ids": np.asarray(fixed_angle_reference_node_ids, dtype=int),
        "real_state_labels": np.array(
            [
                *(f"theta[{bus_id}]" for bus_id in ordered_angle_bus_ids),
                *(f"v[{int(bus_id)}]" for bus_id in solver_bus_ids),
            ]
        ),
    }
    validation = {
        "maximum_injection_jacobian_error": maximum_injection_jacobian_error,
        "real_gain_inverse_relative_residual": relative_inverse_residual,
        "maximum_angle_sigma_error": maximum_angle_sigma_error,
        "maximum_magnitude_sigma_error": maximum_magnitude_sigma_error,
    }
    return real_matrices, validation


def validate_complex_augmented_against_pgm(
    case: SparsityCase,
    matrices: dict[str, np.ndarray],
    variance_normalization: float,
    node_ids: np.ndarray,
) -> float:
    """Check the reconstructed state covariance diagonal against PGM output."""
    if not case.validate_complex_sigmas:
        return float("nan")
    result = PowerGridModel(case.inputs, system_frequency=case.system_frequency).calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod.iterative_linear,
        calculate_uncertainty=True,
        error_tolerance=case.error_tolerance,
        max_iterations=case.max_iterations,
    )
    node_output = result[ComponentType.node]
    output_by_id = {int(row[AttributeType.id]): float(row[AttributeType.u_sigma]) for row in node_output}
    rated_voltage = {int(row["id"]): float(row["u_rated"]) for row in case.topology["node"]}
    state_covariance = matrices["augmented_inverse"][: len(node_ids), : len(node_ids)]
    predicted = np.sqrt(np.maximum(0.5 * variance_normalization * np.real(np.diag(state_covariance)), 0.0)) * np.array(
        [rated_voltage[int(node_id)] for node_id in node_ids]
    )
    reported = np.array([output_by_id[int(node_id)] for node_id in node_ids])
    np.testing.assert_allclose(
        predicted,
        reported,
        rtol=case.expectations.complex_sigma_rtol,
        atol=case.expectations.complex_sigma_atol,
    )
    return float(np.max(np.abs(predicted - reported)))


def numerical_mask(matrix: np.ndarray, relative_threshold: float) -> tuple[np.ndarray, float]:
    """Return the relative-threshold numerical nonzero mask and absolute cutoff."""
    cutoff = relative_threshold * float(np.linalg.norm(matrix, ord=np.inf))
    return np.abs(matrix) > cutoff, cutoff


def draw_numerical_mask(axis: plt.Axes, mask: np.ndarray) -> None:
    """Draw all nonzero cells as one vector compound path."""
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for row, column in zip(*np.nonzero(mask), strict=True):
        left = float(column) - 0.5
        right = float(column) + 0.5
        top = float(row) - 0.5
        bottom = float(row) + 0.5
        vertices.extend(((left, top), (right, top), (right, bottom), (left, bottom), (left, top)))
        codes.extend(
            int(code) for code in (MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY)
        )

    path = MplPath(vertices, codes)
    axis.add_patch(PathPatch(path, facecolor="#3B0F70", edgecolor="none", linewidth=0.0, antialiased=False))
    axis.set_xlim(-0.5, mask.shape[1] - 0.5)
    axis.set_ylim(mask.shape[0] - 0.5, -0.5)
    axis.set_aspect("equal")


def build_figure(
    matrices: dict[str, np.ndarray],
    relative_threshold: float,
    output_root: Path,
    figure_stem: str,
) -> pd.DataFrame:
    """Render a case-independent eight-panel IEEE Transactions figure."""
    panel_specs = (
        ("real_measurement_matrix", r"(a) $\mathbf{H}_{\mathrm{r}}$", "real polar"),
        ("real_gain", r"(b) $\mathbf{G}_{\mathrm{r}}$", "real polar"),
        ("real_gain_inverse", r"(c) $\mathbf{G}_{\mathrm{r}}^{-1}$", "real polar"),
        ("y_bus", r"(d) $\mathbf{Y}_{\mathrm{bus}}$", "complex"),
        ("augmented", r"(e) $\mathbf{A}$", "complex"),
        ("augmented_ps", r"(f) $\mathbf{A}_{\mathrm{ps}}$", "complex"),
        ("augmented_inverse", r"(g) $\mathbf{A}^{-1}$", "complex"),
        ("augmented_ps_inverse", r"(h) $\mathbf{A}_{\mathrm{ps}}^{-1}$", "complex"),
    )
    summary_rows: list[dict[str, float | int | str]] = []
    angle_state_count = int(matrices["real_angle_state_count"])
    real_group_counts = matrices["real_measurement_group_counts"].astype(int)
    real_row_count = matrices["real_measurement_matrix"].shape[0]
    n_bus = matrices["y_bus"].shape[0]

    with plt.style.context(STYLE), plt.rc_context({"savefig.bbox": None}):
        figure = plt.figure(figsize=FIGURE_SIZE)
        grid = figure.add_gridspec(2, 5)
        axes = (
            figure.add_subplot(grid[:, 0]),
            figure.add_subplot(grid[0, 1]),
            figure.add_subplot(grid[0, 2]),
            figure.add_subplot(grid[0, 3]),
            figure.add_subplot(grid[0, 4]),
            figure.add_subplot(grid[1, 1]),
            figure.add_subplot(grid[1, 2]),
            figure.add_subplot(grid[1, 3]),
        )
        legend_axis = figure.add_subplot(grid[1, 4])
        for axis, (name, panel_label, domain) in zip(axes, panel_specs, strict=True):
            matrix = matrices[name]
            mask, cutoff = numerical_mask(matrix, relative_threshold)
            nonzeros = int(np.count_nonzero(mask))
            density = nonzeros / mask.size
            summary_rows.append(
                {
                    "matrix": name,
                    "domain": domain,
                    "rows": matrix.shape[0],
                    "columns": matrix.shape[1],
                    "absolute threshold": cutoff,
                    "nonzeros": nonzeros,
                    "density": density,
                }
            )

            draw_numerical_mask(axis, mask)
            axis.text(0.0, 1.04, panel_label, transform=axis.transAxes, ha="left", va="bottom")
            axis.text(
                1.0,
                1.04,
                rf"$\mathrm{{nnz}}={nonzeros}$ ({100.0 * density:.1f}\%)",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
            )
            if name == "real_measurement_matrix":
                axis.set_xlabel(r"State position $[\theta;|U|]$")
                axis.set_ylabel("Measurement row")
                if 0 < angle_state_count < matrix.shape[1]:
                    axis.axvline(angle_state_count - 0.5, color="#D55E00", linewidth=0.45)
                group_boundaries = np.unique(np.cumsum(real_group_counts)[:-1]) - 0.5
                for boundary in group_boundaries:
                    if 0.0 <= boundary < real_row_count - 0.5:
                        axis.axhline(boundary, color="#D55E00", linewidth=0.45)
            elif name in {"real_gain", "real_gain_inverse"}:
                axis.set_xlabel(r"State position $[\theta;|U|]$")
                axis.set_ylabel(r"State position $[\theta;|U|]$")
                if 0 < angle_state_count < matrix.shape[0]:
                    axis.axhline(angle_state_count - 0.5, color="#D55E00", linewidth=0.45)
                    axis.axvline(angle_state_count - 0.5, color="#D55E00", linewidth=0.45)
            elif name == "y_bus":
                axis.set_xlabel("PGM bus position")
                axis.set_ylabel("PGM bus position")
            else:
                axis.set_xlabel("Matrix position")
                axis.set_ylabel("Matrix position")
            last_column = matrix.shape[1] - 1
            last_row = matrix.shape[0] - 1
            axis.set_xticks(sorted({0, last_column // 2, last_column}))
            axis.set_yticks(sorted({0, last_row // 2, last_row}))
            axis.tick_params(labelsize=6)
            if name in {"augmented", "augmented_inverse"}:
                axis.axhline(n_bus - 0.5, color="#D55E00", linewidth=0.45)
                axis.axvline(n_bus - 0.5, color="#D55E00", linewidth=0.45)

        legend_axis.axis("off")
        legend_axis.legend(
            handles=[
                Patch(facecolor="#3B0F70", edgecolor="black", linewidth=0.3, label="Numerical nonzero"),
                Patch(facecolor="white", edgecolor="black", linewidth=0.3, label="Numerical zero"),
                Line2D([0], [0], color="#D55E00", linewidth=0.6, label="Block / row-group boundary"),
            ],
            loc="center",
            frameon=False,
            fontsize=6,
        )
        figure.tight_layout(pad=0.35, w_pad=0.45, h_pad=0.65)
        output_root.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_root / f"{figure_stem}.pdf")
        figure.savefig(output_root / f"{figure_stem}.png", dpi=PREVIEW_DPI)
        plt.close(figure)

    return pd.DataFrame(summary_rows)


def build_sparsity_artifacts(
    case: SparsityCase,
    *,
    relative_threshold: float = DEFAULT_RELATIVE_THRESHOLD,
    output_root: Path = WORKFLOW_ROOT,
    render_figure: bool = True,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, float]]:
    """Build, validate, and persist all artifacts for ``case``."""
    if not 0.0 <= relative_threshold < 1.0:
        raise ValueError("relative_threshold must lie in [0, 1)")
    if Path(case.artifact_prefix).name != case.artifact_prefix or case.artifact_prefix in {"", ".", ".."}:
        raise ValueError("artifact_prefix must be a non-empty filename prefix, not a path")
    if not np.isfinite(case.system_frequency) or case.system_frequency <= 0.0:
        raise ValueError("system_frequency must be positive and finite")
    if not np.isfinite(case.error_tolerance) or case.error_tolerance <= 0.0:
        raise ValueError("error_tolerance must be positive and finite")
    if case.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    complex_matrices, variance_normalization, shuffle, node_ids = assemble_matrices(case)
    real_matrices, real_validation = build_real_gain_matrices(case, complex_matrices, node_ids)
    matrices = complex_matrices | real_matrices
    matrices["pgm_bus_order"] = node_ids
    maximum_complex_sigma_error = validate_complex_augmented_against_pgm(
        case,
        matrices,
        variance_normalization,
        node_ids,
    )
    figure_root = output_root / "figures"
    results_root = output_root / "results"
    if render_figure:
        summary = build_figure(matrices, relative_threshold, figure_root, case.figure_stem)
    else:
        summary_rows = []
        for name, domain in (
            ("real_measurement_matrix", "real polar"),
            ("real_gain", "real polar"),
            ("real_gain_inverse", "real polar"),
            ("y_bus", "complex"),
            ("augmented", "complex"),
            ("augmented_ps", "complex"),
            ("augmented_inverse", "complex"),
            ("augmented_ps_inverse", "complex"),
        ):
            matrix = matrices[name]
            mask, cutoff = numerical_mask(matrix, relative_threshold)
            summary_rows.append(
                {
                    "matrix": name,
                    "domain": domain,
                    "rows": matrix.shape[0],
                    "columns": matrix.shape[1],
                    "absolute threshold": cutoff,
                    "nonzeros": int(np.count_nonzero(mask)),
                    "density": float(np.count_nonzero(mask) / mask.size),
                }
            )
        summary = pd.DataFrame(summary_rows)

    results_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        results_root / case.matrix_archive_name,
        **matrices,  # type: ignore[arg-type]
        case_name=np.asarray(case.name),
        artifact_prefix=np.asarray(case.artifact_prefix),
        perfect_shuffle=shuffle,
        variance_normalization=np.array(variance_normalization),
        system_frequency=np.asarray(case.system_frequency),
        base_power_3p=np.asarray(BASE_POWER_3P),
        error_tolerance=np.asarray(case.error_tolerance),
        max_iterations=np.asarray(case.max_iterations),
    )
    summary.to_csv(results_root / f"{case.figure_stem}.csv", index=False)
    pd.DataFrame({"pgm_position": np.arange(node_ids.size), "bus_id": node_ids}).to_csv(
        results_root / case.bus_order_name, index=False
    )
    validation = real_validation | {"maximum_complex_sigma_error": maximum_complex_sigma_error}
    return matrices, summary, validation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=SUPPORTED_CASE_NAMES,
        default="ieee33",
        help="Notebook-backed case to build (default: ieee33).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKFLOW_ROOT,
        help="Root receiving figures/ and results/ (default: this workflow directory).",
    )
    parser.add_argument(
        "--artifact-prefix",
        help="Override the deterministic case prefix for an exploratory build.",
    )
    parser.add_argument(
        "--relative-threshold",
        type=float,
        default=DEFAULT_RELATIVE_THRESHOLD,
        help="Relative cutoff tau in |M_ij| > tau ||M||_inf.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the selected notebook-backed case from the command line."""
    args = parse_args(argv)
    case = load_case(args.case)
    if args.artifact_prefix:
        case = replace(case, artifact_prefix=args.artifact_prefix)
    matrices, summary, validation = build_sparsity_artifacts(
        case,
        relative_threshold=args.relative_threshold,
        output_root=args.output_root,
    )
    print(summary.to_string(index=False, formatters={"density": "{:.6f}".format}))
    group_counts = dict(
        zip(
            matrices["real_measurement_group_names"].tolist(),
            matrices["real_measurement_group_counts"].astype(int).tolist(),
            strict=True,
        )
    )
    print(f"Case: {case.name}; real H shape: {matrices['real_measurement_matrix'].shape}; groups: {group_counts}")
    print(f"PGM bus order: {matrices['pgm_bus_order'].tolist()}")
    print(f"Maximum complex-ILSE voltage-sigma reconstruction error: {validation['maximum_complex_sigma_error']:.3e} V")
    print(
        "Maximum real gain-inverse sigma errors: "
        f"angle={validation['maximum_angle_sigma_error']:.3e} rad; "
        f"magnitude={validation['maximum_magnitude_sigma_error']:.3e} p.u."
    )
    print(f"Maximum injection-Jacobian finite-difference error: {validation['maximum_injection_jacobian_error']:.3e}")
    print(f"Real gain-inverse normwise relative residual: {validation['real_gain_inverse_relative_residual']:.3e}")
    print(f"Style: {STYLE}; figure size: {FIGURE_SIZE} in; PNG DPI: {PREVIEW_DPI}")


if __name__ == "__main__":
    main()
