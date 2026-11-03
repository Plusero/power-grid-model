# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Estimate a three-phase network and print phase V/W/var/A and radian uncertainty."""

import argparse

from power_grid_model import CalculationMethod, ComponentType as CT, DatasetType as DT, PowerGridModel, initialize_array
from power_grid_model.enum import ObservedTerminalType as Terminal
from power_grid_model.utils import json_serialize


def rows(component, count=1, **fields):
    result = initialize_array(DT.input, component, count)
    for field, values in fields.items():
        result[field] = values
    return result


def example_input():
    """Measurements determine the estimate; the three observer types request sigmas."""
    return {
        CT.node: rows(CT.node, 2, id=[1, 2], u_rated=10e3),
        CT.line: rows(
            CT.line,
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
        ),
        CT.source: rows(CT.source, id=20, node=1, status=1, u_ref=1.0),
        CT.asym_load: rows(
            CT.asym_load,
            id=21,
            node=2,
            status=1,
            type=0,
            p_specified=[[100e3, 90e3, 110e3]],
            q_specified=[[10e3, 12e3, 8e3]],
        ),
        CT.asym_voltage_sensor: rows(
            CT.asym_voltage_sensor,
            2,
            id=[31, 32],
            measured_object=[1, 2],
            u_measured=[[5950.0, 5700.0, 6100.0], [5750.0, 5500.0, 5850.0]],
            u_angle_measured=[[0.13, -2.01, 2.18], [-0.07, -2.21, 1.98]],
            u_sigma=[50.0, 80.0],
        ),
        CT.voltage_uncertainty_observer: rows(
            CT.voltage_uncertainty_observer,
            2,
            id=[101, 102],
            observed_object=[1, 2],
            status=1,
        ),
        CT.power_uncertainty_observer: rows(
            CT.power_uncertainty_observer,
            3,
            id=[111, 112, 113],
            observed_object=[10, 10, 2],
            observed_terminal_type=[Terminal.branch_from, Terminal.branch_to, Terminal.node_injection],
            status=1,
        ),
        CT.current_uncertainty_observer: rows(
            CT.current_uncertainty_observer,
            2,
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
    result = PowerGridModel(example_input()).calculate_state_estimation(
        symmetric=False,
        calculation_method=CalculationMethod[args.method],
        output_component_types=[
            CT.voltage_uncertainty_observer,
            CT.power_uncertainty_observer,
            CT.current_uncertainty_observer,
        ],
    )
    # Every sigma attribute is an a/b/c array. These are modeled uncertainties
    # under the prepared sensor weights, including ILSE's proper-complex model.
    print(json_serialize(result, dataset_type=DT.asym_output, use_compact_list=True))


if __name__ == "__main__":
    main()
