# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""Run symmetric state estimation and serialize quantity-specific observer outputs."""

import argparse

from power_grid_model import CalculationMethod, ComponentType as CT, DatasetType as DT, PowerGridModel, initialize_array
from power_grid_model.enum import ObservedTerminalType as Terminal
from power_grid_model.utils import json_serialize


def rows(component, **fields):
    result = initialize_array(
        DT.input, component, max(len(value) if isinstance(value, list) else 1 for value in fields.values())
    )
    for name, value in fields.items():
        result[name] = value
    return result


def example_input():
    """Ordinary voltage sensors supply the measurements; observers request outputs."""
    return {
        CT.node: rows(CT.node, id=[1, 2], u_rated=10e3),
        CT.line: rows(
            CT.line,
            id=10,
            from_node=1,
            to_node=2,
            from_status=1,
            to_status=1,
            r1=1.0,
            x1=2.0,
            c1=1e-8,
            tan1=0.0,
            r0=1.0,
            x0=2.0,
            c0=1e-8,
            tan0=0.0,
            i_n=1e3,
        ),
        CT.source: rows(CT.source, id=20, node=1, status=1, u_ref=1.0),
        CT.sym_load: rows(CT.sym_load, id=21, node=2, status=1, type=0, p_specified=1e5, q_specified=1e4),
        CT.sym_voltage_sensor: rows(
            CT.sym_voltage_sensor,
            id=[31, 32],
            measured_object=[1, 2],
            u_measured=[10300.0, 9700.0],
            u_angle_measured=[0.13, -0.08],
            u_sigma=[100.0, 200.0],
        ),
        CT.voltage_uncertainty_observer: rows(
            CT.voltage_uncertainty_observer, id=[101, 102], observed_object=[1, 2], status=1
        ),
        CT.power_uncertainty_observer: rows(
            CT.power_uncertainty_observer,
            id=[111, 112, 113],
            observed_object=[10, 10, 2],
            observed_terminal_type=[Terminal.branch_from, Terminal.branch_to, Terminal.node_injection],
            status=1,
        ),
        CT.current_uncertainty_observer: rows(
            CT.current_uncertainty_observer,
            id=[121, 122],
            observed_object=10,
            observed_terminal_type=[Terminal.branch_from, Terminal.branch_to],
            status=1,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["iterative_linear", "newton_raphson"], default="iterative_linear")
    args = parser.parse_args()
    model = PowerGridModel(example_input())
    result = model.calculate_state_estimation(
        symmetric=True,
        calculation_method=CalculationMethod[args.method],
        output_component_types=[
            CT.voltage_uncertainty_observer,
            CT.power_uncertainty_observer,
            CT.current_uncertainty_observer,
        ],
    )
    print(json_serialize(result, dataset_type=DT.sym_output, use_compact_list=True))


if __name__ == "__main__":
    main()
