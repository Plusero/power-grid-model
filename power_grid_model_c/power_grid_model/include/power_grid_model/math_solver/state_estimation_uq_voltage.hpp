// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// Voltage magnitude and angle uncertainty of the state-estimation output for one island math model.

#include "state_estimation_uq.hpp"

#include "../common/common.hpp"

#include <cmath>
#include <complex>
#include <cstddef>

namespace power_grid_model::math_solver::state_estimation_uq {

// The functions below take a SparseCovarianceWorkspace. For ILSE its state is the complex voltage U with
// proper-complex covariance V, from which angle and magnitude variances are derived. For NRSE its state is the
// voltage angle and magnitude, so their variances are read directly.

// Variance Var(theta_i) of one voltage angle, before it is made slack-relative.
template <class Workspace> double absolute_angle_variance(Workspace const& workspace, Idx bus, int phase) {
    if constexpr (Workspace::uses_complex_state) {
        // Re(V_ii) / (2 |U_i|^2)
        return workspace.state_variance(bus, phase) / (2.0 * std::norm(workspace.phase_voltage(bus, phase)));
    } else {
        return workspace.state_variance(bus, phase);
    }
}

// Covariance Cov(theta_i, theta_s) of one voltage angle with the phase-a angle of the slack bus.
template <class Workspace> double angle_covariance_with_slack(Workspace const& workspace, Idx bus, int phase) {
    if constexpr (Workspace::uses_complex_state) {
        // Re(V_is / (U_i conj(U_s))) / 2
        DoubleComplex const voltage = workspace.phase_voltage(bus, phase);
        DoubleComplex const slack_voltage = workspace.phase_voltage(workspace.slack_bus(), 0);
        return (workspace.slack_covariance(bus, phase) / (voltage * std::conj(slack_voltage))).real() / 2.0;
    } else {
        return workspace.slack_covariance(bus, phase);
    }
}

// Uncertainty of one voltage angle as SE reports it. Without a voltage angle measurement,
// SE reports every angle relative to the phase-a angle of the slack bus.
// The variance of such a slack-relative angle is
// Var(theta_i - theta_s) = Var(theta_i) + Var(theta_s) - 2 Cov(theta_i,theta_s)
// The variance of the reference angle itself is exactly zero.
template <class Workspace>
UQPhaseValue angle_uncertainty(Workspace const& workspace, Idx bus, int phase, bool defined) {
    Idx const slack_bus = workspace.slack_bus();
    RealSum sum;
    if (!workspace.report_relative_angle()) {
        sum.add(absolute_angle_variance(workspace, bus, phase));
    } else if (bus != slack_bus || phase != 0) {
        sum.add(absolute_angle_variance(workspace, bus, phase));            // Var(theta_i)
        sum.add(absolute_angle_variance(workspace, slack_bus, 0));          // Var(theta_s)
        sum.add(-2.0 * angle_covariance_with_slack(workspace, bus, phase)); // -2 Cov(theta_i, theta_s)
    }
    return finalize_variance(sum, defined);
}

// Uncertainty of one voltage magnitude, which does not depend on the angle reference.
template <class Workspace>
UQPhaseValue magnitude_uncertainty(Workspace const& workspace, Idx bus, int phase, bool defined) {
    RealSum sum;
    if constexpr (Workspace::uses_complex_state) {
        // Re(V_ii) / 2
        sum.add(workspace.state_variance(bus, phase) / 2.0);
    } else {
        sum.add(workspace.state_variance(bus, phase + Workspace::phase_count));
    }
    return finalize_variance(sum, defined);
}

// Voltage uncertainty for all requested buses; result rows follow the request order.
template <class Workspace>
StateEstimationUQResult<typename Workspace::Symmetry>
calculate_voltage_uq_results(Workspace const& workspace, StateEstimationUQRequest const& request) {
    StateEstimationUQResult<typename Workspace::Symmetry> result;
    result.voltage.resize(request.voltage.size());
    for (std::size_t index = 0; index != request.voltage.size(); ++index) {
        auto const& voltage_request = request.voltage[index];
        auto& output = result.voltage[index];
        for (int phase = 0; phase != Workspace::phase_count; ++phase) {
            // Magnitude and angle are not defined at a zero or nonfinite voltage.
            double const magnitude = std::abs(workspace.phase_voltage(voltage_request.bus, phase));
            bool const defined = std::isfinite(magnitude) && magnitude > 0.0;
            if (voltage_request.magnitude) {
                output.magnitude.phase[phase] = magnitude_uncertainty(workspace, voltage_request.bus, phase, defined);
            }
            if (voltage_request.angle) {
                output.angle.phase[phase] = angle_uncertainty(workspace, voltage_request.bus, phase, defined);
            }
        }
    }
    return result;
}

} // namespace power_grid_model::math_solver::state_estimation_uq
