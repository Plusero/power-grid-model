// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// Request, result, and context types for state-estimation uncertainty quantification (UQ), and variance status rule.

#include "../common/common.hpp"
#include "../common/enum.hpp"
#include "../common/exception.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace power_grid_model::math_solver {

// Uncertainty requests use bus and branch indices local to a single solved island's math model.
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
    // Also report how the covariance of each pair of buses behind the injection results was obtained.
    bool pair_diagnostics = false;
    bool empty() const { return voltage.empty() && branch.empty() && injection.empty(); }
};

// Classification of a scalar variance computation:
// - not_requested:    quantity was not requested by the caller (sigma and raw_variance remain NaN).
// - valid:            finite nonnegative variance (sigma = sqrt(raw_variance)).
// - clipped_roundoff: small negative variance within [-round_rtol * absolute_scale, 0) caused by
//                     floating-point cancellation. Clipped to sigma = 0 while preserving raw_variance. Mainly for angle
//                     variances.
// - invalid:          undefined quantity, failed factorization, nonfinite variance, or negative variance
//                     beyond roundoff tolerance (sigma remains NaN).
enum class UQVarianceStatus : IntS { not_requested, valid, clipped_roundoff, invalid };

// Per-phase uncertainty output and diagnostic values.
// Units match the math solver's internal per-unit / radian convention:
// - Voltage magnitude: raw_variance and absolute_scale in p.u.^2, sigma in p.u.
// - Voltage angle:     raw_variance and absolute_scale in rad^2, sigma in rad.
// - Branch power and current magnitude in p.u., branch current angle in rad, at the requested side.
// - Nodal power injection in p.u.
// Angle uncertainty is expressed in the same reference frame as the state-estimation (SE) output:
// - Absolute frame when the island has at least one prepared voltage-angle measurement.
// - Otherwise relative to the phase-a angle of the slack bus. That angle itself has zero variance;
//   phases b and c of the slack bus do not.
struct UQPhaseValue {
    double raw_variance = std::numeric_limits<double>::quiet_NaN();
    double absolute_scale = std::numeric_limits<double>::quiet_NaN();
    double sigma = std::numeric_limits<double>::quiet_NaN();
    UQVarianceStatus raw_status = UQVarianceStatus::not_requested;
    // Set for an injection at an exact zero-injection constraint. Its sigma is then 0, whatever the computed
    // variance and its status are.
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

// How the covariance of a pair of buses was obtained, see state_estimation_uq_completion.hpp.
enum class UQPairRoute : IntS { stored, separator, augmented_path, mean_path, zero_target, invalid };
// Diagnostic of one completed pair. approximation_warning marks a kept pair whose middle solve residual or
// normalized correlation exceeds its limit; it never changes the route, the value or the validity.
struct UQPairDiagnostic {
    Idx first_bus = -1;
    Idx second_bus = -1;
    UQPairRoute route = UQPairRoute::invalid;
    Idx middle_bus = -1;
    bool approximation_warning = false;
};

// UQ results for one island.
// voltage[k], branch[k] and injection[k] answer the request rows of the same index.
// With pair_diagnostics, pairs lists each pair of buses behind the injection results once.
template <symmetry_tag sym> struct StateEstimationUQResult {
    std::vector<VoltageUQResult<sym>> voltage;
    std::vector<BranchUQResult<sym>> branch;
    std::vector<InjectionUQResult<sym>> injection;
    std::vector<UQPairDiagnostic> pairs;
};

// References to the caller's request and result for one island, passed through run_state_estimation().
// The caller owns both objects and keeps them alive during the call.
template <symmetry_tag sym> struct StateEstimationUQContext {
    StateEstimationUQRequest const& request;
    StateEstimationUQResult<sym>& result;
};

namespace state_estimation_uq {
// Reject out-of-range bus and branch indices and unknown branch sides before state estimation starts.
template <symmetry_tag sym> void validate_context(StateEstimationUQContext<sym>* uq, Idx n_bus, Idx n_branch) {
    if (!uq) {
        return;
    }
    for (auto const& item : uq->request.voltage) {
        if (item.bus < 0 || item.bus >= n_bus) {
            throw InvalidArguments{"Invalid voltage uncertainty request"};
        }
    }
    for (auto const& item : uq->request.branch) {
        if (item.branch < 0 || item.branch >= n_branch ||
            (item.side != BranchSide::from && item.side != BranchSide::to)) {
            throw InvalidArguments{"Invalid branch uncertainty request"};
        }
    }
    for (auto const& item : uq->request.injection) {
        if (item.bus < 0 || item.bus >= n_bus) {
            throw InvalidArguments{"Invalid injection uncertainty request"};
        }
    }
}

// A negative variance within round_rtol times the sum of its term magnitudes is treated as cancellation roundoff.
inline constexpr double round_rtol = 1000.0 * std::numeric_limits<double>::epsilon();

// A variance accumulated from additive terms, together with the sum of their magnitudes. That sum is the scale against
// which a negative result is judged.
class RealSum {
  public:
    void add(double term) {
        value_ += term;
        absolute_scale_ += std::abs(term);
    }
    double value() const { return value_; }
    double absolute_scale() const { return absolute_scale_; }

  private:
    double value_ = 0.0;
    double absolute_scale_ = 0.0;
};

// Classify a variance as described at UQVarianceStatus. `defined` is false when the quantity does not exist,
// such as an angle at a zero voltage or current. The variance and its scale are stored for every status.
inline UQPhaseValue finalize_variance(double variance, double absolute_scale, bool defined) {
    UQPhaseValue result;
    result.raw_variance = variance;
    result.absolute_scale = absolute_scale;
    result.raw_status = UQVarianceStatus::invalid;
    double const tolerance = round_rtol * absolute_scale;
    if (!defined || !std::isfinite(variance) || !std::isfinite(absolute_scale) || variance < -tolerance) {
        return result;
    }
    result.raw_status = variance < 0.0 ? UQVarianceStatus::clipped_roundoff : UQVarianceStatus::valid;
    result.sigma = std::sqrt(std::max(0.0, variance));
    return result;
}
inline UQPhaseValue finalize_variance(RealSum const& sum, bool defined) {
    return finalize_variance(sum.value(), sum.absolute_scale(), defined);
}
} // namespace state_estimation_uq
} // namespace power_grid_model::math_solver
