# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

# ruff: noqa: PLR2004

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from power_grid_model import AttributeType as AT, ComponentType as CT, DatasetType as DT, initialize_array

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "examples"))
from network_summary import network_statistics, zero_injection_node_ids


def test_network_statistics_and_zero_injection_definition() -> None:
    node = initialize_array(DT.input, CT.node, 4)
    node[AT.id] = [0, 1, 2, 3]
    node[AT.u_rated] = [20_000.0, 20_000.0, 400.0, 400.0]

    line = initialize_array(DT.input, CT.line, 2)
    line[AT.from_status] = [1, 1]
    line[AT.to_status] = [1, 0]

    transformer = initialize_array(DT.input, CT.transformer, 1)
    transformer[AT.from_status] = 1
    transformer[AT.to_status] = 1

    source = initialize_array(DT.input, CT.source, 1)
    source[AT.node] = 0
    source[AT.status] = 1

    load = initialize_array(DT.input, CT.sym_load, 2)
    load[AT.node] = [1, 2]
    load[AT.status] = [1, 0]

    shunt = initialize_array(DT.input, CT.shunt, 1)
    shunt[AT.node] = 2
    shunt[AT.status] = 1

    input_data = {
        CT.node: node,
        CT.line: line,
        CT.transformer: transformer,
        CT.source: source,
        CT.sym_load: load,
        CT.shunt: shunt,
    }

    np.testing.assert_array_equal(zero_injection_node_ids(input_data), [2, 3])
    statistics = network_statistics(input_data)
    assert statistics["buses"] == 4
    assert statistics["branch_elements"] == 3
    assert statistics["in_service_branch_elements"] == 2
    assert statistics["open_branch_elements"] == 1
    assert statistics["lines"] == 2
    assert statistics["transformers"] == 1
    assert statistics["loads_in_service"] == 1
    assert statistics["loads_total"] == 2
    assert statistics["zero_injection_buses"] == 2
    np.testing.assert_array_equal(statistics["nominal_voltage_levels_kv"], [0.4, 20.0])
