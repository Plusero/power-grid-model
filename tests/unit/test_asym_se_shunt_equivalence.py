# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""A coupled shunt and an electrically equivalent self-loop must fit the same state."""

from pathlib import Path

import numpy as np

from power_grid_model import PowerGridModel
from power_grid_model._core.utils import convert_batch_dataset_to_batch_list
from power_grid_model.utils import json_deserialize_from_file


def test_asymmetric_nrse_coupled_shunt_matches_equivalent_self_loop():
    case = Path(__file__).parents[1] / "data/state_estimation/dummy-test-line-into-itself-minimal"
    input_data = json_deserialize_from_file(case / "input.json")
    updates = convert_batch_dataset_to_batch_list(json_deserialize_from_file(case / "update_batch.json"))
    nodes = []
    for update in updates:
        model = PowerGridModel(input_data, system_frequency=50.0)
        model.update(update_data=update)
        result = model.calculate_state_estimation(
            symmetric=False, calculation_method="newton_raphson", error_tolerance=1e-12, max_iterations=100
        )
        nodes.append(result["node"])

    # The self-loop's two half-charging terminals and sensor weights are
    # electrically/statistically equivalent to the measured coupled shunt.
    for attribute, tolerance in (("u", 1e-8), ("u_angle", 1e-12), ("p", 1e-8), ("q", 1e-8)):
        np.testing.assert_allclose(nodes[0][attribute], nodes[1][attribute], rtol=0.0, atol=tolerance)
