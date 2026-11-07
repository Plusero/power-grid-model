// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// Shared entry point for state-estimation UQ, used by ILSE and NRSE.
// Calls the covariance and requested-result calculations.

#include "measured_values.hpp"
#include "state_estimation_uq.hpp"
#include "state_estimation_uq_workspace.hpp"
#include "y_bus.hpp"

#include "../common/common.hpp"
#include "../common/exception.hpp"
#include "../common/three_phase_tensor.hpp"

#include <cstddef>
#include <vector>

namespace power_grid_model::math_solver::state_estimation_uq {

// Every requested quantity is invalid, with NaN sigma and diagnostics.
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
    return result;
}

// UQ entry point after ordinary SE has produced the current math model's estimate.
// Construct one covariance workspace and calculate requested results.
// ILSE supplies existing factors; NRSE supplies the gain at the returned state.
// The estimate does not depend on the uncertainty: if the gain fails a numerical check
// (assembly, pivot, perturbed factors, or a nonfinite inverse), every requested quantity
// is reported invalid instead of discarding the estimate.
template <class Workspace, class Gain>
StateEstimationUQResult<typename Workspace::Symmetry>
calculate_results(YBus<typename Workspace::Symmetry> const& y_bus, std::vector<Gain> const& gain,
                  MeasuredValues<typename Workspace::Symmetry> const& measured_values,
                  ComplexValueVector<typename Workspace::Symmetry> const& u, StateEstimationUQRequest const& request,
                  Idx virtual_bus = -1, double virtual_weight = 0.0,
                  typename Workspace::Factorization const* factorization = nullptr) {
    // Stochastic Rbar = R/alpha, Wbar = alpha*W; covariance corrections restore one factor of alpha.
    double const alpha = measured_values.variance_normalization();
    try {
        Workspace const workspace{y_bus,       gain,           alpha,        u, measured_values.has_angle(),
                                  virtual_bus, virtual_weight, factorization};
        return calculate_voltage_uq_results(workspace, request);
    } catch (SparseMatrixError const&) {
        return invalid_results<typename Workspace::Symmetry>(request);
    }
}

} // namespace power_grid_model::math_solver::state_estimation_uq
