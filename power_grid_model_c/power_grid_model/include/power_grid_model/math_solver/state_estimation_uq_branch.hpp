// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// Branch power and current uncertainty of the state-estimation output for one island math model.

#include "state_estimation_uq.hpp"
#include "state_estimation_uq_voltage.hpp"
#include "state_estimation_uq_workspace.hpp"
#include "y_bus.hpp"

#include "../calculation_parameters.hpp"
#include "../common/common.hpp"
#include "../common/enum.hpp"
#include "../common/three_phase_tensor.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>

namespace power_grid_model::math_solver::state_estimation_uq {

namespace branch_detail {

inline DoubleComplex admittance_entry(DoubleComplex const& y, int /*row*/, int /*col*/) { return y; }
inline DoubleComplex admittance_entry(ComplexTensor<asymmetric_t> const& y, int row, int col) { return y(row, col); }

// The current of one phase at one side of a branch is I = sum_k c_k U_k over the phases k of both buses.
// coefficient[bus][phase] holds c_k, with bus 0 the from bus and bus 1 the to bus.
template <int phases> using Coefficients = std::array<std::array<DoubleComplex, phases>, 2>;

// Derivative of one real output with respect to the rectangular voltages [Re U, Im U] of both buses.
template <int phases> using Derivative = std::array<Eigen::Matrix<double, 2 * phases, 1>, 2>;

template <int phases> Derivative<phases> zero_derivative() {
    Derivative<phases> result;
    result[0].setZero();
    result[1].setZero();
    return result;
}

// Derivatives dS/dRe U_k and dS/dIm U_k of the power S = U_t conj(I) at the terminal voltage U_t:
//   dS/dRe U_k = U_t conj(c_k)   and   dS/dIm U_k = -j U_t conj(c_k),
// plus conj(I) and j conj(I) for the terminal's own voltage. There the two terms are combined as
// 2 Re(U_t) conj(c_t) + others and 2 Im(U_t) conj(c_t) + j others, with others = sum_{k != t} conj(c_k U_k).
// This keeps the result exact where U_t conj(c_t) and conj(c_t U_t) would otherwise cancel in rounding, as at
// a branch that is open at the other side.
inline std::array<DoubleComplex, 2> power_derivative(DoubleComplex terminal_voltage, DoubleComplex coefficient,
                                                     bool own_voltage, DoubleComplex others) {
    DoubleComplex const conj_c = std::conj(coefficient);
    if (own_voltage) {
        return {2.0 * terminal_voltage.real() * conj_c + others,
                2.0 * terminal_voltage.imag() * conj_c + DoubleComplex{-others.imag(), others.real()}};
    }
    DoubleComplex const w = terminal_voltage * conj_c;
    return {w, DoubleComplex{w.imag(), -w.real()}};
}

} // namespace branch_detail

// Requested branch uncertainties. Request rows are validated by validate_context before the estimate.
//
// Every output is a real function of the voltages of the two buses of the branch, so its first-order
// variance is d C d^T with d its derivative and C the covariance of those voltages. Power and current
// magnitude do not depend on the angle reference. A current angle that state estimation reports relative to
// the slack phase-a angle theta_s has variance Var(angle) + Var(theta_s) - 2 Cov(angle, theta_s).
//
// An output is invalid at a side that is not connected to a bus. A current whose squared modulus is not a
// normal number has neither magnitude derivative nor angle.
template <class Workspace, symmetry_tag sym>
void branch_results(Workspace const& workspace, YBus<sym> const& y_bus, StateEstimationUQRequest const& request,
                    StateEstimationUQResult<sym>& result) {
    using branch_detail::Derivative;
    constexpr int phases = Workspace::phase_count;
    UQPhaseValue const undefined = finalize_variance(RealSum{}, false);
    result.branch.resize(request.branch.size());
    for (std::size_t index = 0; index != request.branch.size(); ++index) {
        auto const& item = request.branch[index];
        auto& output = result.branch[index];
        auto const& buses = y_bus.math_topology().branch_bus_idx[item.branch];
        auto const& param = y_bus.math_model_param().branch_param[item.branch];
        int const terminal = item.side == BranchSide::from ? 0 : 1;
        bool const relative_angle = item.current_angle && workspace.report_relative_angle();

        // Covariance of the rectangular voltages of the connected buses, and with the slack angle if needed.
        std::array<std::array<typename Workspace::RectangularMatrix, 2>, 2> covariance;
        std::array<typename Workspace::RectangularVector, 2> slack_covariance;
        for (int i = 0; i != 2; ++i) {
            if (buses[i] < 0) {
                continue;
            }
            for (int j = 0; j != 2; ++j) {
                if (buses[j] >= 0) {
                    covariance[i][j] = workspace.rectangular_covariance(buses[i], buses[j]);
                }
            }
            if (relative_angle) {
                slack_covariance[i] = workspace.rectangular_slack_covariance(buses[i]);
            }
        }
        auto const variance = [&](Derivative<phases> const& derivative, bool defined, bool subtract_slack_angle) {
            RealSum sum;
            for (int i = 0; i != 2; ++i) {
                for (int j = 0; j != 2; ++j) {
                    if (buses[i] < 0 || buses[j] < 0) {
                        continue;
                    }
                    for (int p = 0; p != 2 * phases; ++p) {
                        for (int q = 0; q != 2 * phases; ++q) {
                            sum.add(derivative[i](p) * covariance[i][j](p, q) * derivative[j](q));
                        }
                    }
                }
            }
            if (subtract_slack_angle) {
                sum.add(absolute_angle_variance(workspace, workspace.slack_bus(), 0));
                for (int i = 0; i != 2; ++i) {
                    for (int p = 0; buses[i] >= 0 && p != 2 * phases; ++p) {
                        sum.add(-2.0 * derivative[i](p) * slack_covariance[i](p));
                    }
                }
            }
            return finalize_variance(sum, defined);
        };

        for (int phase = 0; phase != phases; ++phase) {
            if (buses[terminal] < 0) {
                if (item.active_power) {
                    output.active_power.phase[phase] = undefined;
                }
                if (item.reactive_power) {
                    output.reactive_power.phase[phase] = undefined;
                }
                if (item.current_magnitude) {
                    output.current_magnitude.phase[phase] = undefined;
                }
                if (item.current_angle) {
                    output.current_angle.phase[phase] = undefined;
                }
                continue;
            }
            // I = sum_k c_k U_k, and others = sum of conj(c_k U_k) over all voltages but the terminal's own.
            branch_detail::Coefficients<phases> coefficient{};
            DoubleComplex current{};
            DoubleComplex others{};
            for (int i = 0; i != 2; ++i) {
                auto const& admittance =
                    terminal == 0 ? (i == 0 ? param.yff() : param.yft()) : (i == 0 ? param.ytf() : param.ytt());
                for (int p = 0; p != phases; ++p) {
                    coefficient[i][p] = branch_detail::admittance_entry(admittance, phase, p);
                    if (buses[i] < 0) {
                        continue;
                    }
                    DoubleComplex const term = coefficient[i][p] * workspace.phase_voltage(buses[i], p);
                    current += term;
                    if (i != terminal || p != phase) {
                        others += std::conj(term);
                    }
                }
            }

            if (item.active_power || item.reactive_power) {
                DoubleComplex const voltage = workspace.phase_voltage(buses[terminal], phase);
                auto active = branch_detail::zero_derivative<phases>();
                auto reactive = branch_detail::zero_derivative<phases>();
                for (int i = 0; i != 2; ++i) {
                    for (int p = 0; buses[i] >= 0 && p != phases; ++p) {
                        auto const [d_real, d_imag] = branch_detail::power_derivative(
                            voltage, coefficient[i][p], i == terminal && p == phase, others);
                        active[i](p) = d_real.real();
                        active[i](phases + p) = d_imag.real();
                        reactive[i](p) = d_real.imag();
                        reactive[i](phases + p) = d_imag.imag();
                    }
                }
                if (item.active_power) {
                    output.active_power.phase[phase] = variance(active, true, false);
                }
                if (item.reactive_power) {
                    output.reactive_power.phase[phase] = variance(reactive, true, false);
                }
            }

            if (item.current_magnitude || item.current_angle) {
                bool const has_direction = std::norm(current) >= std::numeric_limits<double>::min();
                auto magnitude = branch_detail::zero_derivative<phases>();
                auto angle = branch_detail::zero_derivative<phases>();
                if (has_direction) {
                    // d|I| = Re(conj(I) dI) / |I| and d(arg I) = Im(dI / I), with dI/dRe U_k = c_k and
                    // dI/dIm U_k = j c_k.
                    DoubleComplex const direction = std::conj(current) / std::abs(current);
                    for (int i = 0; i != 2; ++i) {
                        for (int p = 0; buses[i] >= 0 && p != phases; ++p) {
                            DoubleComplex const d_magnitude = direction * coefficient[i][p];
                            DoubleComplex const d_angle = coefficient[i][p] / current;
                            magnitude[i](p) = d_magnitude.real();
                            magnitude[i](phases + p) = -d_magnitude.imag();
                            angle[i](p) = d_angle.imag();
                            angle[i](phases + p) = d_angle.real();
                        }
                    }
                }
                if (item.current_magnitude) {
                    output.current_magnitude.phase[phase] = variance(magnitude, has_direction, false);
                }
                if (item.current_angle) {
                    output.current_angle.phase[phase] = variance(angle, has_direction, relative_angle);
                }
            }
        }
    }
}

} // namespace power_grid_model::math_solver::state_estimation_uq
