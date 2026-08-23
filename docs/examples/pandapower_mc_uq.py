# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Pandapower WLS Monte Carlo uncertainty quantification for PGM examples.

See ``pandapower_mc_uq.md`` for the native-network mapping and fallback logic.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandapower as pp
import pandas as pd
from network_summary import zero_injection_node_ids
from pandapower.estimation import estimate
from pandapower.toolbox import select_subnet
from pandapower.topology import connected_components, create_nxgraph

from power_grid_model import AttributeType, ComponentType, LoadGenType, MeasuredTerminalType

_MINIMUM_MONTE_CARLO_SAMPLES = 2


@dataclass(frozen=True)
class _MeasurementBinding:
    measurement_index: int
    sensor_type: ComponentType
    sensor_index: int
    attribute: AttributeType
    scale: float


@dataclass
class PandapowerModel:
    """Pandapower model plus the PGM-to-pandapower index mappings."""

    net: Any
    node_index: dict[int, int]
    line_index: dict[int, int]
    transformer_index: dict[int, int]
    transformer_side: dict[tuple[int, str], str]
    active_branch_ids: set[int]
    network_source: str


@dataclass
class PandapowerMonteCarloResult:
    """Monte Carlo outputs in PGM component order and SI units."""

    outputs: dict[ComponentType, dict[str, np.ndarray]]
    converged: np.ndarray
    seconds: float
    measurement_count: int
    omitted_current_angle_count: int
    omitted_out_of_service_measurement_count: int
    network_source: str
    failure_reason: str | None = None


def _component(input_data: dict[Any, Any], component_type: ComponentType) -> np.ndarray | None:
    if component_type in input_data:
        return input_data[component_type]
    if component_type.value in input_data:
        return input_data[component_type.value]
    return None


def _is_in_service(row: np.void) -> bool:
    names = row.dtype.names or ()
    if "status" in names:
        return bool(row["status"])
    terminal_statuses = [
        bool(row[field]) for field in ("from_status", "to_status", "status_1", "status_2", "status_3") if field in names
    ]
    return all(terminal_statuses) if terminal_statuses else True


def _terminal_status(row: np.void, field: str) -> bool:
    return bool(row[field]) if field in (row.dtype.names or ()) else True


def _build_nodes(model: PandapowerModel, input_data: dict[Any, Any]) -> None:
    nodes = _component(input_data, ComponentType.node)
    if nodes is None:
        raise KeyError("Pandapower conversion requires PGM node data")
    for position, row in enumerate(nodes):
        pgm_id = int(row[AttributeType.id])
        pp_index = pp.create_bus(model.net, vn_kv=float(row[AttributeType.u_rated]) / 1.0e3, index=position)
        model.node_index[pgm_id] = int(pp_index)


def _build_sources(model: PandapowerModel, input_data: dict[Any, Any]) -> None:
    sources = _component(input_data, ComponentType.source)
    if sources is None:
        return
    for row in sources:
        pp.create_ext_grid(
            model.net,
            bus=model.node_index[int(row[AttributeType.node])],
            vm_pu=float(row[AttributeType.u_ref]),
            va_degree=float(np.rad2deg(row[AttributeType.u_ref_angle])),
            in_service=_is_in_service(row),
        )


def _build_lines(model: PandapowerModel, input_data: dict[Any, Any], system_frequency: float) -> None:
    lines = _component(input_data, ComponentType.line)
    if lines is None:
        return
    for position, row in enumerate(lines):
        pgm_id = int(row[AttributeType.id])
        from_bus = model.node_index[int(row[AttributeType.from_node])]
        to_bus = model.node_index[int(row[AttributeType.to_node])]
        capacitance = float(row[AttributeType.c1])
        loss_tangent = float(row[AttributeType.tan1])
        from_status = _terminal_status(row, "from_status")
        to_status = _terminal_status(row, "to_status")
        pp_index = pp.create_line_from_parameters(
            model.net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=1.0,
            r_ohm_per_km=float(row[AttributeType.r1]),
            x_ohm_per_km=float(row[AttributeType.x1]),
            c_nf_per_km=capacitance * 1.0e9,
            g_us_per_km=2.0 * np.pi * system_frequency * capacitance * loss_tangent * 1.0e6,
            max_i_ka=float(row[AttributeType.i_n]) / 1.0e3,
            in_service=from_status and to_status,
            index=position,
        )
        model.line_index[pgm_id] = int(pp_index)
        if from_status and to_status:
            model.active_branch_ids.add(pgm_id)


def _build_transformers(model: PandapowerModel, input_data: dict[Any, Any]) -> None:
    transformers = _component(input_data, ComponentType.transformer)
    if transformers is None:
        return
    nodes = _component(input_data, ComponentType.node)
    if nodes is None:
        raise KeyError("Transformer conversion requires PGM node data")
    rated_voltage = {int(row[AttributeType.id]): float(row[AttributeType.u_rated]) for row in nodes}
    for position, row in enumerate(transformers):
        pgm_id = int(row[AttributeType.id])
        from_node = int(row[AttributeType.from_node])
        to_node = int(row[AttributeType.to_node])
        from_is_hv = rated_voltage[from_node] >= rated_voltage[to_node]
        hv_node, lv_node = (from_node, to_node) if from_is_hv else (to_node, from_node)
        from_side, to_side = ("hv", "lv") if from_is_hv else ("lv", "hv")
        model.transformer_side[pgm_id, "from"] = from_side
        model.transformer_side[pgm_id, "to"] = to_side
        nominal_power = float(row[AttributeType.sn])
        tap_size = float(row[AttributeType.tap_size])
        tap_side_value = int(row[AttributeType.tap_side])
        pgm_tap_side = "from" if tap_side_value == 0 else "to"
        pp_tap_side = model.transformer_side[pgm_id, pgm_tap_side]
        from_status = _terminal_status(row, "from_status")
        to_status = _terminal_status(row, "to_status")
        pp_index = pp.create_transformer_from_parameters(
            model.net,
            hv_bus=model.node_index[hv_node],
            lv_bus=model.node_index[lv_node],
            sn_mva=nominal_power / 1.0e6,
            vn_hv_kv=max(float(row[AttributeType.u1]), float(row[AttributeType.u2])) / 1.0e3,
            vn_lv_kv=min(float(row[AttributeType.u1]), float(row[AttributeType.u2])) / 1.0e3,
            vkr_percent=100.0 * float(row[AttributeType.pk]) / nominal_power,
            vk_percent=100.0 * float(row[AttributeType.uk]),
            pfe_kw=float(row[AttributeType.p0]) / 1.0e3,
            i0_percent=100.0 * float(row[AttributeType.i0]),
            shift_degree=30.0 * float(row[AttributeType.clock]),
            tap_side=pp_tap_side if tap_size else None,
            tap_neutral=float(row[AttributeType.tap_nom]) if tap_size else np.nan,
            tap_min=float(row[AttributeType.tap_min]) if tap_size else np.nan,
            tap_max=float(row[AttributeType.tap_max]) if tap_size else np.nan,
            tap_step_percent=100.0 * tap_size if tap_size else np.nan,
            tap_pos=float(row[AttributeType.tap_pos]) if tap_size else np.nan,
            in_service=from_status and to_status,
            index=position,
        )
        model.transformer_index[pgm_id] = int(pp_index)
        if from_status and to_status:
            model.active_branch_ids.add(pgm_id)


def _build_loads_and_generators(model: PandapowerModel, input_data: dict[Any, Any]) -> None:
    for component_type, creator in (
        (ComponentType.sym_load, pp.create_load),
        (ComponentType.sym_gen, pp.create_sgen),
    ):
        components = _component(input_data, component_type)
        if components is None:
            continue
        for row in components:
            load_gen_type = int(row[AttributeType.type])
            zip_parameters: dict[str, float] = {}
            if component_type == ComponentType.sym_load:
                zip_parameters = {
                    "const_z_p_percent": 100.0 if load_gen_type == int(LoadGenType.const_impedance) else 0.0,
                    "const_i_p_percent": 100.0 if load_gen_type == int(LoadGenType.const_current) else 0.0,
                    "const_z_q_percent": 100.0 if load_gen_type == int(LoadGenType.const_impedance) else 0.0,
                    "const_i_q_percent": 100.0 if load_gen_type == int(LoadGenType.const_current) else 0.0,
                }
            creator(
                model.net,
                bus=model.node_index[int(row[AttributeType.node])],
                p_mw=float(row[AttributeType.p_specified]) / 1.0e6,
                q_mvar=float(row[AttributeType.q_specified]) / 1.0e6,
                in_service=_is_in_service(row),
                **zip_parameters,
            )


def _build_links(model: PandapowerModel, input_data: dict[Any, Any]) -> None:
    links = _component(input_data, ComponentType.link)
    if links is None:
        return
    for row in links:
        pp.create_switch(
            model.net,
            bus=model.node_index[int(row[AttributeType.from_node])],
            element=model.node_index[int(row[AttributeType.to_node])],
            et="b",
            closed=_is_in_service(row),
        )


def _validate_native_value(
    name: str,
    pgm_value: float,
    pandapower_value: float,
    *,
    relative_tolerance: float = 1.0e-9,
    absolute_tolerance: float = 1.0e-12,
) -> None:
    if not np.isclose(
        pgm_value,
        pandapower_value,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        equal_nan=True,
    ):
        raise ValueError(f"Native pandapower {name} mismatch: PGM={pgm_value}, pandapower={pandapower_value}")


def _native_terminal_status(
    net: Any,
    *,
    element_type: str,
    element_index: int,
    bus_index: int,
    element_in_service: bool,
) -> bool:
    if not element_in_service:
        return False
    switches = net.switch[
        (net.switch["et"] == element_type) & (net.switch["element"] == element_index) & (net.switch["bus"] == bus_index)
    ]
    return bool(switches["closed"].all()) if len(switches) else True


def _map_native_pandapower_model(  # noqa: PLR0912, PLR0915
    input_data: dict[Any, Any], pandapower_network: Any
) -> PandapowerModel:
    net = copy.deepcopy(pandapower_network)
    model = PandapowerModel(
        net=net,
        node_index={},
        line_index={},
        transformer_index={},
        transformer_side={},
        active_branch_ids=set(),
        network_source="native pandapower network",
    )

    nodes = _component(input_data, ComponentType.node)
    lines = _component(input_data, ComponentType.line)
    transformers = _component(input_data, ComponentType.transformer)
    sources = _component(input_data, ComponentType.source)
    if nodes is None:
        raise KeyError("Native pandapower mapping requires PGM node data")
    component_counts = {
        "bus": (len(nodes), len(net.bus)),
        "line": (0 if lines is None else len(lines), len(net.line)),
        "trafo": (0 if transformers is None else len(transformers), len(net.trafo)),
        "ext_grid": (0 if sources is None else len(sources), len(net.ext_grid)),
    }
    mismatched_counts = {name: counts for name, counts in component_counts.items() if counts[0] != counts[1]}
    if mismatched_counts:
        raise ValueError(f"Native pandapower component-count mismatch (PGM, pandapower): {mismatched_counts}")

    native_bus_to_pgm: dict[int, int] = {}
    for pgm_node, (native_bus_index, native_bus) in zip(nodes, net.bus.iterrows(), strict=True):
        pgm_node_id = int(pgm_node[AttributeType.id])
        model.node_index[pgm_node_id] = int(native_bus_index)
        native_bus_to_pgm[int(native_bus_index)] = pgm_node_id
        _validate_native_value(
            f"bus {pgm_node_id} rated voltage",
            float(pgm_node[AttributeType.u_rated]),
            float(native_bus["vn_kv"]) * 1.0e3,
        )

    if sources is not None:
        for pgm_source, (_, native_source) in zip(sources, net.ext_grid.iterrows(), strict=True):
            pgm_source_node = int(pgm_source[AttributeType.node])
            native_source_node = native_bus_to_pgm[int(native_source["bus"])]
            if pgm_source_node != native_source_node:
                raise ValueError(
                    f"Native pandapower source-node mismatch: PGM={pgm_source_node}, pandapower={native_source_node}"
                )

    if lines is not None:
        for pgm_line, (native_line_index, native_line) in zip(lines, net.line.iterrows(), strict=True):
            pgm_line_id = int(pgm_line[AttributeType.id])
            model.line_index[pgm_line_id] = int(native_line_index)
            pgm_endpoints = (
                int(pgm_line[AttributeType.from_node]),
                int(pgm_line[AttributeType.to_node]),
            )
            native_endpoints = (
                native_bus_to_pgm[int(native_line["from_bus"])],
                native_bus_to_pgm[int(native_line["to_bus"])],
            )
            if pgm_endpoints != native_endpoints:
                raise ValueError(
                    f"Native pandapower line {pgm_line_id} endpoint mismatch: "
                    f"PGM={pgm_endpoints}, pandapower={native_endpoints}"
                )
            length = float(native_line["length_km"])
            _validate_native_value(
                f"line {pgm_line_id} resistance",
                float(pgm_line[AttributeType.r1]),
                float(native_line["r_ohm_per_km"]) * length,
            )
            _validate_native_value(
                f"line {pgm_line_id} reactance",
                float(pgm_line[AttributeType.x1]),
                float(native_line["x_ohm_per_km"]) * length,
            )
            _validate_native_value(
                f"line {pgm_line_id} capacitance",
                float(pgm_line[AttributeType.c1]),
                float(native_line["c_nf_per_km"]) * length * 1.0e-9,
            )
            native_from_status = _native_terminal_status(
                net,
                element_type="l",
                element_index=int(native_line_index),
                bus_index=int(native_line["from_bus"]),
                element_in_service=bool(native_line["in_service"]),
            )
            native_to_status = _native_terminal_status(
                net,
                element_type="l",
                element_index=int(native_line_index),
                bus_index=int(native_line["to_bus"]),
                element_in_service=bool(native_line["in_service"]),
            )
            pgm_status = (
                _terminal_status(pgm_line, "from_status"),
                _terminal_status(pgm_line, "to_status"),
            )
            native_status = (native_from_status, native_to_status)
            if pgm_status != native_status:
                raise ValueError(
                    f"Native pandapower line {pgm_line_id} terminal-status mismatch: "
                    f"PGM={pgm_status}, pandapower={native_status}"
                )
            if all(native_status):
                model.active_branch_ids.add(pgm_line_id)
            else:
                model.net.line.at[native_line_index, "in_service"] = False

    if transformers is not None:
        for pgm_transformer, (native_transformer_index, native_transformer) in zip(
            transformers, net.trafo.iterrows(), strict=True
        ):
            pgm_transformer_id = int(pgm_transformer[AttributeType.id])
            model.transformer_index[pgm_transformer_id] = int(native_transformer_index)
            native_hv_node = native_bus_to_pgm[int(native_transformer["hv_bus"])]
            native_lv_node = native_bus_to_pgm[int(native_transformer["lv_bus"])]
            pgm_from_node = int(pgm_transformer[AttributeType.from_node])
            pgm_to_node = int(pgm_transformer[AttributeType.to_node])
            if {pgm_from_node, pgm_to_node} != {native_hv_node, native_lv_node}:
                raise ValueError(
                    f"Native pandapower transformer {pgm_transformer_id} endpoint mismatch: "
                    f"PGM={(pgm_from_node, pgm_to_node)}, "
                    f"pandapower={(native_hv_node, native_lv_node)}"
                )
            model.transformer_side[pgm_transformer_id, "from"] = "hv" if pgm_from_node == native_hv_node else "lv"
            model.transformer_side[pgm_transformer_id, "to"] = "hv" if pgm_to_node == native_hv_node else "lv"
            _validate_native_value(
                f"transformer {pgm_transformer_id} HV voltage",
                max(
                    float(pgm_transformer[AttributeType.u1]),
                    float(pgm_transformer[AttributeType.u2]),
                ),
                float(native_transformer["vn_hv_kv"]) * 1.0e3,
            )
            _validate_native_value(
                f"transformer {pgm_transformer_id} LV voltage",
                min(
                    float(pgm_transformer[AttributeType.u1]),
                    float(pgm_transformer[AttributeType.u2]),
                ),
                float(native_transformer["vn_lv_kv"]) * 1.0e3,
            )
            native_hv_status = _native_terminal_status(
                net,
                element_type="t",
                element_index=int(native_transformer_index),
                bus_index=int(native_transformer["hv_bus"]),
                element_in_service=bool(native_transformer["in_service"]),
            )
            native_lv_status = _native_terminal_status(
                net,
                element_type="t",
                element_index=int(native_transformer_index),
                bus_index=int(native_transformer["lv_bus"]),
                element_in_service=bool(native_transformer["in_service"]),
            )
            native_status_by_side = {"hv": native_hv_status, "lv": native_lv_status}
            native_status = (
                native_status_by_side[model.transformer_side[pgm_transformer_id, "from"]],
                native_status_by_side[model.transformer_side[pgm_transformer_id, "to"]],
            )
            pgm_status = (
                _terminal_status(pgm_transformer, "from_status"),
                _terminal_status(pgm_transformer, "to_status"),
            )
            if pgm_status != native_status:
                raise ValueError(
                    f"Native pandapower transformer {pgm_transformer_id} terminal-status mismatch: "
                    f"PGM={pgm_status}, pandapower={native_status}"
                )
            if all(native_status):
                model.active_branch_ids.add(pgm_transformer_id)
            else:
                model.net.trafo.at[native_transformer_index, "in_service"] = False

    model.net.measurement = model.net.measurement.iloc[0:0].copy()
    return model


def build_pandapower_model(
    input_data: dict[Any, Any],
    *,
    pandapower_network: Any | None = None,
    system_frequency: float = 50.0,
) -> PandapowerModel:
    """Map a native pandapower network or convert symmetric PGM components as a fallback."""

    unsupported = (
        ComponentType.asym_load,
        ComponentType.asym_gen,
        ComponentType.asym_line,
        ComponentType.three_winding_transformer,
        ComponentType.generic_branch,
    )
    present_unsupported = [
        component.value
        for component in unsupported
        if (values := _component(input_data, component)) is not None and len(values)
    ]
    if present_unsupported:
        raise NotImplementedError(f"Pandapower UQ conversion does not support {present_unsupported}")

    if pandapower_network is not None:
        return _map_native_pandapower_model(input_data, pandapower_network)

    model = PandapowerModel(
        net=pp.create_empty_network(sn_mva=1.0, f_hz=system_frequency),
        node_index={},
        line_index={},
        transformer_index={},
        transformer_side={},
        active_branch_ids=set(),
        network_source="PGM-to-pandapower fallback conversion",
    )
    _build_nodes(model, input_data)
    _build_sources(model, input_data)
    _build_lines(model, input_data, system_frequency)
    _build_transformers(model, input_data)
    _build_loads_and_generators(model, input_data)
    _build_links(model, input_data)
    return model


def _branch_measurement_target(
    model: PandapowerModel, measured_object: int, terminal_type: int
) -> tuple[str, int, str]:
    terminal = "from" if terminal_type == int(MeasuredTerminalType.branch_from) else "to"
    if measured_object in model.line_index:
        return "line", model.line_index[measured_object], terminal
    if measured_object in model.transformer_index:
        return (
            "trafo",
            model.transformer_index[measured_object],
            model.transformer_side[measured_object, terminal],
        )
    raise KeyError(f"Unknown measured branch ID {measured_object}")


def _add_measurements(model: PandapowerModel, input_data: dict[Any, Any]) -> tuple[list[_MeasurementBinding], int, int]:
    bindings: list[_MeasurementBinding] = []
    omitted_current_angle_count = 0
    omitted_out_of_service_measurement_count = 0
    nodes = _component(input_data, ComponentType.node)
    if nodes is None:
        raise KeyError("Measurement conversion requires PGM node data")
    rated_voltage = {int(row[AttributeType.id]): float(row[AttributeType.u_rated]) for row in nodes}

    voltage_sensors = _component(input_data, ComponentType.sym_voltage_sensor)
    if voltage_sensors is not None:
        for sensor_index, sensor in enumerate(voltage_sensors):
            measured_object = int(sensor[AttributeType.measured_object])
            voltage_base = rated_voltage[measured_object]
            channels = [
                (
                    "v",
                    AttributeType.u_measured,
                    1.0 / voltage_base,
                    float(sensor[AttributeType.u_sigma]) / voltage_base,
                ),
            ]
            if np.isfinite(sensor[AttributeType.u_angle_measured]):
                angle_sigma = float(sensor[AttributeType.u_sigma]) / voltage_base
                channels.append(("va", AttributeType.u_angle_measured, 180.0 / np.pi, float(np.rad2deg(angle_sigma))))
            for measurement_type, attribute, scale, standard_deviation in channels:
                measurement_index = pp.create_measurement(
                    model.net,
                    measurement_type,
                    "bus",
                    float(sensor[attribute]) * scale,
                    standard_deviation,
                    model.node_index[measured_object],
                )
                bindings.append(
                    _MeasurementBinding(
                        measurement_index, ComponentType.sym_voltage_sensor, sensor_index, attribute, scale
                    )
                )

    power_sensors = _component(input_data, ComponentType.sym_power_sensor)
    if power_sensors is not None:
        for sensor_index, sensor in enumerate(power_sensors):
            measured_object = int(sensor[AttributeType.measured_object])
            if measured_object not in model.active_branch_ids:
                omitted_out_of_service_measurement_count += 2
                continue
            element_type, element, side = _branch_measurement_target(
                model,
                measured_object,
                int(sensor[AttributeType.measured_terminal_type]),
            )
            for measurement_type, attribute, sigma_attribute in (
                ("p", AttributeType.p_measured, AttributeType.p_sigma),
                ("q", AttributeType.q_measured, AttributeType.q_sigma),
            ):
                measurement_index = pp.create_measurement(
                    model.net,
                    measurement_type,
                    element_type,
                    float(sensor[attribute]) / 1.0e6,
                    float(sensor[sigma_attribute]) / 1.0e6,
                    element,
                    side=side,
                )
                bindings.append(
                    _MeasurementBinding(
                        measurement_index,
                        ComponentType.sym_power_sensor,
                        sensor_index,
                        attribute,
                        1.0e-6,
                    )
                )

    current_sensors = _component(input_data, ComponentType.sym_current_sensor)
    if current_sensors is not None:
        for sensor_index, sensor in enumerate(current_sensors):
            measured_object = int(sensor[AttributeType.measured_object])
            has_current_angle = int(np.isfinite(sensor[AttributeType.i_angle_measured]))
            omitted_current_angle_count += has_current_angle
            if measured_object not in model.active_branch_ids:
                omitted_out_of_service_measurement_count += 1
                continue
            element_type, element, side = _branch_measurement_target(
                model,
                measured_object,
                int(sensor[AttributeType.measured_terminal_type]),
            )
            measurement_index = pp.create_measurement(
                model.net,
                "i",
                element_type,
                float(sensor[AttributeType.i_measured]) / 1.0e3,
                float(sensor[AttributeType.i_sigma]) / 1.0e3,
                element,
                side=side,
            )
            bindings.append(
                _MeasurementBinding(
                    measurement_index,
                    ComponentType.sym_current_sensor,
                    sensor_index,
                    AttributeType.i_measured,
                    1.0e-3,
                )
            )
    return bindings, omitted_current_angle_count, omitted_out_of_service_measurement_count


def _allocate_outputs(input_data: dict[Any, Any], sample_count: int) -> dict[ComponentType, dict[str, np.ndarray]]:
    fields = {
        ComponentType.node: ("u", "u_angle", "p", "q"),
        ComponentType.line: ("i_from", "i_to", "p_from", "q_from", "p_to", "q_to"),
        ComponentType.transformer: ("i_from", "i_to", "p_from", "q_from", "p_to", "q_to"),
    }
    outputs = {}
    for component_type, component_fields in fields.items():
        components = _component(input_data, component_type)
        if components is None:
            continue
        outputs[component_type] = {
            field: np.full((sample_count, len(components)), np.nan) for field in component_fields
        }
    return outputs


def _estimation_subnets(model: PandapowerModel) -> list[Any]:
    graph = create_nxgraph(model.net, respect_switches=True)
    bus_components = [set(component) for component in connected_components(graph)]
    if len(bus_components) == 1:
        return [model.net]
    return [select_subnet(model.net, buses) for buses in bus_components]


def _merge_subnet_results(model: PandapowerModel, subnets: list[Any]) -> None:
    for result_table_name, element_table_name in (
        ("res_bus_est", "bus"),
        ("res_line_est", "line"),
        ("res_trafo_est", "trafo"),
    ):
        result_tables = [getattr(subnet, result_table_name) for subnet in subnets]
        combined = pd.concat(result_tables) if result_tables else pd.DataFrame()
        combined = combined[~combined.index.duplicated(keep="first")]
        setattr(model.net, result_table_name, combined.reindex(getattr(model.net, element_table_name).index))


def _copy_estimation_outputs(
    result: PandapowerMonteCarloResult,
    scenario: int,
    model: PandapowerModel,
    input_data: dict[Any, Any],
) -> None:
    nodes = _component(input_data, ComponentType.node)
    if nodes is None:
        raise KeyError("Output conversion requires PGM node data")
    bus_indices = [model.node_index[int(row[AttributeType.id])] for row in nodes]
    result.outputs[ComponentType.node]["u"][scenario] = model.net.res_bus_est.loc[
        bus_indices, "vm_pu"
    ].to_numpy() * np.asarray(nodes[AttributeType.u_rated], dtype=float)
    result.outputs[ComponentType.node]["u_angle"][scenario] = np.deg2rad(
        model.net.res_bus_est.loc[bus_indices, "va_degree"].to_numpy()
    )
    result.outputs[ComponentType.node]["p"][scenario] = (
        model.net.res_bus_est.loc[bus_indices, "p_mw"].to_numpy() * 1.0e6
    )
    result.outputs[ComponentType.node]["q"][scenario] = (
        model.net.res_bus_est.loc[bus_indices, "q_mvar"].to_numpy() * 1.0e6
    )

    component_specs = (
        (ComponentType.line, "res_line_est", model.line_index, {"from": "from", "to": "to"}),
        (
            ComponentType.transformer,
            "res_trafo_est",
            model.transformer_index,
            {"from": None, "to": None},
        ),
    )
    for component_type, result_table_name, index_mapping, side_mapping in component_specs:
        components = _component(input_data, component_type)
        if components is None:
            continue
        table = getattr(model.net, result_table_name)
        pp_indices = [index_mapping[int(row[AttributeType.id])] for row in components]
        for component_position, (component, pp_index) in enumerate(zip(components, pp_indices, strict=True)):
            component_id = int(component[AttributeType.id])
            if component_id not in model.active_branch_ids:
                continue
            for pgm_side in ("from", "to"):
                pp_side = (
                    model.transformer_side[component_id, pgm_side]
                    if component_type == ComponentType.transformer
                    else side_mapping[pgm_side]
                )
                result.outputs[component_type][f"i_{pgm_side}"][scenario, component_position] = (
                    table.at[pp_index, f"i_{pp_side}_ka"] * 1.0e3
                )
                result.outputs[component_type][f"p_{pgm_side}"][scenario, component_position] = (
                    table.at[pp_index, f"p_{pp_side}_mw"] * 1.0e6
                )
                result.outputs[component_type][f"q_{pgm_side}"][scenario, component_position] = (
                    table.at[pp_index, f"q_{pp_side}_mvar"] * 1.0e6
                )


def run_pandapower_monte_carlo(  # noqa: PLR0913
    input_data: dict[Any, Any],
    updates: dict[Any, Any],
    *,
    pandapower_network: Any | None = None,
    sample_count: int | None = None,
    tolerance: float = 1.0e-6,
    maximum_iterations: int = 50,
) -> PandapowerMonteCarloResult:
    """Run pandapower WLS for PGM sensor updates and return empirical outputs.

    Pandapower 3.5 WLS consumes current magnitudes but not current-angle
    measurements. The omitted channel count is returned so notebooks can make
    this estimator-model difference visible.
    """

    available_sample_count = next(iter(updates.values())).shape[0]
    if sample_count is None:
        sample_count = available_sample_count
    if sample_count < _MINIMUM_MONTE_CARLO_SAMPLES or sample_count > available_sample_count:
        raise ValueError("sample_count must be between 2 and the available update count")

    model = build_pandapower_model(input_data, pandapower_network=pandapower_network)
    bindings, omitted_current_angle_count, omitted_out_of_service_measurement_count = _add_measurements(
        model, input_data
    )
    result = PandapowerMonteCarloResult(
        outputs=_allocate_outputs(input_data, sample_count),
        converged=np.zeros(sample_count, dtype=bool),
        seconds=0.0,
        measurement_count=len(bindings),
        omitted_current_angle_count=omitted_current_angle_count,
        omitted_out_of_service_measurement_count=omitted_out_of_service_measurement_count,
        network_source=model.network_source,
    )
    current_sensors = _component(input_data, ComponentType.sym_current_sensor)
    power_sensors = _component(input_data, ComponentType.sym_power_sensor)
    if current_sensors is not None and len(current_sensors) and (power_sensors is None or not len(power_sensors)):
        result.failure_reason = (
            "Pandapower WLS does not consume current-angle measurements; after omitting that channel, "
            "the voltage-plus-current layout is under-observable."
        )
        return result
    zero_injection_buses = [model.node_index[int(node_id)] for node_id in zero_injection_node_ids(input_data)]
    subnets = _estimation_subnets(model)

    start = perf_counter()
    for scenario in range(sample_count):
        for binding in bindings:
            sensor_updates = _component(updates, binding.sensor_type)
            if sensor_updates is None:
                raise KeyError(f"Missing Monte Carlo updates for {binding.sensor_type.value}")
            model.net.measurement.at[binding.measurement_index, "value"] = (
                float(sensor_updates[binding.attribute][scenario, binding.sensor_index]) * binding.scale
            )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=pd.errors.SettingWithCopyWarning,
                    module=r"pandapower\.estimation\.ppc_conversion",
                )
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in cast",
                    category=RuntimeWarning,
                    module=r"pandapower\.estimation\.ppc_conversion",
                )
                successful = True
                for subnet in subnets:
                    measurement_indices = subnet.measurement.index
                    subnet.measurement.loc[measurement_indices, "value"] = model.net.measurement.loc[
                        measurement_indices, "value"
                    ]
                    subnet_zero_injection_buses = [bus for bus in zero_injection_buses if bus in subnet.bus.index]
                    algorithm = "wls_with_zero_constraint" if subnet_zero_injection_buses else "wls"
                    successful = successful and estimate(
                        subnet,
                        algorithm=algorithm,
                        init="slack",
                        tolerance=tolerance,
                        maximum_iterations=maximum_iterations,
                        zero_injection=subnet_zero_injection_buses,
                    )
        except (np.linalg.LinAlgError, UserWarning):
            successful = False
        if successful:
            result.converged[scenario] = True
            if len(subnets) > 1:
                _merge_subnet_results(model, subnets)
            _copy_estimation_outputs(result, scenario, model, input_data)
    result.seconds = perf_counter() - start
    if not np.any(result.converged):
        result.failure_reason = "No pandapower WLS Monte Carlo scenario converged."
    return result
