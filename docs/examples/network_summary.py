# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Compact, consistent network statistics for the UQ example notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from power_grid_model import ComponentType

_LOAD_COMPONENTS = (ComponentType.sym_load, ComponentType.asym_load)
_GENERATOR_COMPONENTS = (ComponentType.sym_gen, ComponentType.asym_gen)
_APPLIANCE_COMPONENTS = (ComponentType.source, *_LOAD_COMPONENTS, *_GENERATOR_COMPONENTS)
_LINE_COMPONENTS = (ComponentType.line, ComponentType.asym_line)
_TRANSFORMER_COMPONENTS = (ComponentType.transformer, ComponentType.three_winding_transformer)
_OTHER_BRANCH_COMPONENTS = (ComponentType.link, ComponentType.generic_branch)
_BRANCH_COMPONENTS = (*_LINE_COMPONENTS, *_TRANSFORMER_COMPONENTS, *_OTHER_BRANCH_COMPONENTS)


def _component(input_data: Mapping[Any, np.ndarray], component: ComponentType) -> np.ndarray | None:
    """Return one component array from enum- or string-keyed input data."""
    if component in input_data:
        return input_data[component]
    if component.value in input_data:
        return input_data[component.value]
    return None


def _component_count(input_data: Mapping[Any, np.ndarray], components: tuple[ComponentType, ...]) -> int:
    return sum(len(values) for component in components if (values := _component(input_data, component)) is not None)


def _in_service_mask(values: np.ndarray) -> np.ndarray:
    """Return whether each component is connected at all required terminals."""
    names = values.dtype.names or ()
    if "status" in names:
        return np.asarray(values["status"], dtype=bool)
    terminal_status_fields = [
        field for field in ("from_status", "to_status", "status_1", "status_2", "status_3") if field in names
    ]
    if not terminal_status_fields:
        return np.ones(len(values), dtype=bool)
    status = np.ones(len(values), dtype=bool)
    for field in terminal_status_fields:
        status &= np.asarray(values[field], dtype=bool)
    return status


def _in_service_and_total(
    input_data: Mapping[Any, np.ndarray], components: tuple[ComponentType, ...]
) -> tuple[int, int]:
    in_service = 0
    total = 0
    for component in components:
        values = _component(input_data, component)
        if values is None:
            continue
        total += len(values)
        in_service += int(np.count_nonzero(_in_service_mask(values)))
    return in_service, total


def zero_injection_node_ids(input_data: Mapping[Any, np.ndarray]) -> np.ndarray:
    """Return buses with no connected source, load, or generator.

    This matches PGM's exact zero-injection definition for state estimation:
    shunts are modeled separately and do not make a bus an appliance-injection
    bus. A connected zero-valued load or generator still counts as an appliance.
    """
    nodes = _component(input_data, ComponentType.node)
    if nodes is None:
        raise KeyError("The input data has no node component")
    all_node_ids = np.asarray(nodes["id"], dtype=np.int64)
    appliance_node_ids: list[np.ndarray] = []
    for component in _APPLIANCE_COMPONENTS:
        values = _component(input_data, component)
        if values is None or "node" not in (values.dtype.names or ()):
            continue
        appliance_node_ids.append(np.asarray(values["node"][_in_service_mask(values)], dtype=np.int64))
    if not appliance_node_ids:
        return np.sort(all_node_ids)
    return np.setdiff1d(all_node_ids, np.concatenate(appliance_node_ids), assume_unique=False)


def network_statistics(input_data: Mapping[Any, np.ndarray]) -> dict[str, Any]:
    """Calculate validated component and topology counts for one PGM network."""
    nodes = _component(input_data, ComponentType.node)
    if nodes is None:
        raise KeyError("The input data has no node component")

    voltage_levels = np.asarray(nodes["u_rated"], dtype=float) / 1.0e3
    voltage_levels = np.unique(voltage_levels[np.isfinite(voltage_levels) & (voltage_levels > 0.0)])
    in_service_branches, total_branches = _in_service_and_total(input_data, _BRANCH_COMPONENTS)
    in_service_sources, total_sources = _in_service_and_total(input_data, (ComponentType.source,))
    in_service_loads, total_loads = _in_service_and_total(input_data, _LOAD_COMPONENTS)
    in_service_generators, total_generators = _in_service_and_total(input_data, _GENERATOR_COMPONENTS)
    in_service_shunts, total_shunts = _in_service_and_total(input_data, (ComponentType.shunt,))
    two_winding_transformers = _component_count(input_data, (ComponentType.transformer,))
    three_winding_transformers = _component_count(input_data, (ComponentType.three_winding_transformer,))

    return {
        "buses": len(nodes),
        "nominal_voltage_levels_kv": voltage_levels,
        "branch_elements": total_branches,
        "in_service_branch_elements": in_service_branches,
        "open_branch_elements": total_branches - in_service_branches,
        "lines": _component_count(input_data, _LINE_COMPONENTS),
        "transformers": two_winding_transformers + three_winding_transformers,
        "two_winding_transformers": two_winding_transformers,
        "three_winding_transformers": three_winding_transformers,
        "links_and_generic_branches": _component_count(input_data, _OTHER_BRANCH_COMPONENTS),
        "sources_in_service": in_service_sources,
        "sources_total": total_sources,
        "loads_in_service": in_service_loads,
        "loads_total": total_loads,
        "generators_in_service": in_service_generators,
        "generators_total": total_generators,
        "shunts_in_service": in_service_shunts,
        "shunts_total": total_shunts,
        "zero_injection_buses": len(zero_injection_node_ids(input_data)),
    }


def network_summary_rows(input_data: Mapping[Any, np.ndarray]) -> list[dict[str, Any]]:
    """Return reader-facing rows for a compact notebook summary table."""
    statistics = network_statistics(input_data)
    voltage_levels = ", ".join(f"{level:g}" for level in statistics["nominal_voltage_levels_kv"])
    return [
        {"Metric": "Buses", "Value": statistics["buses"]},
        {"Metric": "Nominal voltage levels [kV]", "Value": voltage_levels},
        {"Metric": "Branch elements", "Value": statistics["branch_elements"]},
        {"Metric": "In-service branch elements", "Value": statistics["in_service_branch_elements"]},
        {"Metric": "Open branch elements", "Value": statistics["open_branch_elements"]},
        {"Metric": "Lines", "Value": statistics["lines"]},
        {"Metric": "Transformers", "Value": statistics["transformers"]},
        {"Metric": "Two-winding transformers", "Value": statistics["two_winding_transformers"]},
        {"Metric": "Three-winding transformers", "Value": statistics["three_winding_transformers"]},
        {"Metric": "Links and generic branches", "Value": statistics["links_and_generic_branches"]},
        {
            "Metric": "Sources (in service / total)",
            "Value": f"{statistics['sources_in_service']} / {statistics['sources_total']}",
        },
        {
            "Metric": "Loads (in service / total)",
            "Value": f"{statistics['loads_in_service']} / {statistics['loads_total']}",
        },
        {
            "Metric": "Generators (in service / total)",
            "Value": f"{statistics['generators_in_service']} / {statistics['generators_total']}",
        },
        {
            "Metric": "Shunts (in service / total)",
            "Value": f"{statistics['shunts_in_service']} / {statistics['shunts_total']}",
        },
        {"Metric": "Zero-injection buses", "Value": statistics["zero_injection_buses"]},
    ]
