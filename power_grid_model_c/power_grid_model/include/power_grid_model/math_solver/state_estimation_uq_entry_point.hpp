// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// Post-solve entry point for state-estimation uncertainty quantification, shared by ILSE and NRSE.

#include "measured_values.hpp"
#include "state_estimation_uq.hpp"
#include "state_estimation_uq_branch.hpp"
#include "state_estimation_uq_voltage.hpp"
#include "y_bus.hpp"

#include "../common/common.hpp"
#include "../common/exception.hpp"
#include "../common/three_phase_tensor.hpp"

#include <cstddef>
#include <vector>

namespace power_grid_model::math_solver::state_estimation_uq {

// Every requested quantity is invalid; quantities that were not requested stay not_requested.
template <symmetry_tag sym> StateEstimationUQResult<sym> invalid_results(StateEstimationUQRequest const& request) {
    StateEstimationUQResult<sym> result;
    result.voltage.resize(request.voltage.size());
    for (std::size_t index = 0; index != request.voltage.size(); ++index) {
        auto const& voltage_request = request.voltage[index];
        auto& output = result.voltage[index];
        for (std::size_t phase = 0; phase != output.magnitude.phase.size(); ++phase) {
            if (voltage_request.magnitude) {
                output.magnitude.phase[phase].raw_status = UQVarianceStatus::invalid;
            }
            if (voltage_request.angle) {
                output.angle.phase[phase].raw_status = UQVarianceStatus::invalid;
            }
        }
    }
    result.branch.resize(request.branch.size());
    for (std::size_t index = 0; index != request.branch.size(); ++index) {
        auto const& branch_request = request.branch[index];
        auto& output = result.branch[index];
        for (std::size_t phase = 0; phase != output.active_power.phase.size(); ++phase) {
            if (branch_request.active_power) {
                output.active_power.phase[phase].raw_status = UQVarianceStatus::invalid;
            }
            if (branch_request.reactive_power) {
                output.reactive_power.phase[phase].raw_status = UQVarianceStatus::invalid;
            }
            if (branch_request.current_magnitude) {
                output.current_magnitude.phase[phase].raw_status = UQVarianceStatus::invalid;
            }
            if (branch_request.current_angle) {
                output.current_angle.phase[phase].raw_status = UQVarianceStatus::invalid;
            }
        }
    }
    return result;
}

// Requested uncertainties for one island at the converged estimate `u`.
// `gain` and `factorization` are passed on to the Workspace constructor. If that constructor cannot factorize
// the matrix (SparseMatrixError), the estimate is unaffected and every requested uncertainty is invalid.
template <class Workspace, class Gain>
StateEstimationUQResult<typename Workspace::Symmetry>
calculate_results(YBus<typename Workspace::Symmetry> const& y_bus, std::vector<Gain> const& gain,
                  MeasuredValues<typename Workspace::Symmetry> const& measured_values,
                  ComplexValueVector<typename Workspace::Symmetry> const& u, StateEstimationUQRequest const& request,
                  typename Workspace::Factorization const* factorization = nullptr) {
    try {
        Workspace const workspace{
            y_bus, gain, measured_values.variance_normalization(), u, measured_values.has_angle(), factorization};
        auto result = calculate_voltage_uq_results(workspace, request);
        branch_results(workspace, y_bus, request, result);
        return result;
    } catch (SparseMatrixError const&) {
        return invalid_results<typename Workspace::Symmetry>(request);
    }
}

} // namespace power_grid_model::math_solver::state_estimation_uq
