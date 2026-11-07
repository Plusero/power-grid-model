// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// Request, result, and context types for state-estimation uncertainty quantification (UQ).

#include "../common/common.hpp"
#include "../common/enum.hpp"
#include "../common/exception.hpp"

#include <array>
#include <limits>
#include <vector>

namespace power_grid_model::math_solver {

// Uncertainty requests use indices local to one island's math model.
// Branch and injection requests are declared here to keep the solver dispatch
// interface stable across UQ stages; validate_context rejects them until implemented.
struct VoltageUQRequest {
    Idx bus;
    bool magnitude;
    bool angle;
};
struct BranchUQRequest {
    Idx branch;
    BranchSide side;
    bool active_power;
    bool reactive_power;
    bool current_magnitude;
    bool current_angle;
};
struct InjectionUQRequest {
    Idx bus;
    bool active_power;
    bool reactive_power;
};
struct StateEstimationUQRequest {
    std::vector<VoltageUQRequest> voltage;
    std::vector<BranchUQRequest> branch;
    std::vector<InjectionUQRequest> injection;
    // Opt-in per-pair completion diagnostics.
    bool pair_diagnostics = false;
    bool empty() const { return voltage.empty() && branch.empty() && injection.empty(); }
};

// Per-phase uncertainty status and result values.
// Voltage magnitudes use per-unit (raw_variance in p.u.^2, sigma in p.u.);
// voltage angles use radians (raw_variance in rad^2, sigma in rad).
// Small negative variances within roundoff tolerance are clipped to sigma = 0
// with status clipped_roundoff; unrequested or invalid quantities keep NaN sigma.
enum class UQVarianceStatus : IntS { not_requested, valid, clipped_roundoff, invalid };
struct UQPhaseValue {
    double raw_variance = std::numeric_limits<double>::quiet_NaN();
    double absolute_scale = std::numeric_limits<double>::quiet_NaN();
    double sigma = std::numeric_limits<double>::quiet_NaN();
    UQVarianceStatus raw_status = UQVarianceStatus::not_requested;
    bool exact_zero_enforced = false;
};
template <symmetry_tag sym> struct UQValue {
    std::array<UQPhaseValue, is_symmetric_v<sym> ? 1 : 3> phase{};
};
template <symmetry_tag sym> struct VoltageUQResult {
    UQValue<sym> magnitude, angle;
};

template <symmetry_tag sym> struct BranchUQResult {
    UQValue<sym> active_power, reactive_power, current_magnitude, current_angle;
};
template <symmetry_tag sym> struct InjectionUQResult {
    UQValue<sym> active_power, reactive_power;
};

enum class UQPairRoute : IntS { stored, separator, augmented_path, mean_path, zero_target, invalid };
// Requested per-pair completion diagnostic. approximation_warning marks a kept
// completed pair whose middle solve residual or normalized correlation exceeds
// its threshold; it never changes route, value or validity.
struct UQPairDiagnostic {
    Idx first_bus = -1, second_bus = -1;
    UQPairRoute route = UQPairRoute::invalid;
    Idx middle_bus = -1;
    bool approximation_warning = false;
};
template <symmetry_tag sym> struct StateEstimationUQResult {
    std::vector<VoltageUQResult<sym>> voltage;
    std::vector<BranchUQResult<sym>> branch;
    std::vector<InjectionUQResult<sym>> injection;
    std::vector<UQPairDiagnostic> pairs;
};
// Non-owning request/result context passed to run_state_estimation for one island.
template <symmetry_tag sym> struct StateEstimationUQContext {
    StateEstimationUQRequest const& request;
    StateEstimationUQResult<sym>& result;
};

namespace state_estimation_uq {
// Reject unsupported request categories and out-of-range indices before solving.
template <symmetry_tag sym> void validate_context(StateEstimationUQContext<sym>* uq, Idx n_bus, Idx n_branch) {
    if (!uq) {
        return;
    }
    auto const& request = uq->request;
    for (auto const& item : request.voltage) {
        if (item.bus < 0 || item.bus >= n_bus) {
            throw InvalidArguments{"Invalid voltage uncertainty request"};
        }
    }
    for (auto const& item : request.branch) {
        if (item.branch < 0 || item.branch >= n_branch ||
            (item.side != BranchSide::from && item.side != BranchSide::to)) {
            throw InvalidArguments{"Invalid branch uncertainty request"};
        }
    }
    for (auto const& item : request.injection) {
        if (item.bus < 0 || item.bus >= n_bus) {
            throw InvalidArguments{"Invalid injection uncertainty request"};
        }
    }
}
} // namespace state_estimation_uq
} // namespace power_grid_model::math_solver
