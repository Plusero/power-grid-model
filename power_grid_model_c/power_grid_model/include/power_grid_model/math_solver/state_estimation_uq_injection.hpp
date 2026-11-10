// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// Nodal power injection uncertainty of the state-estimation output for one island math model.

#include "measured_values.hpp"
#include "state_estimation_uq.hpp"
#include "state_estimation_uq_branch.hpp"
#include "state_estimation_uq_completion.hpp"
#include "y_bus.hpp"

#include "../common/common.hpp"
#include "../common/three_phase_tensor.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

namespace power_grid_model::math_solver::state_estimation_uq {

namespace injection_detail {

inline double phase_entry(double value, int /*phase*/) { return value; }
inline double phase_entry(RealValue<asymmetric_t> const& value, int phase) { return value(phase); }

// An exact zero-injection constraint: P and Q of the phase are measured as zero with zero variance.
template <symmetry_tag sym> bool exact_zero(MeasuredValues<sym> const& measured, Idx bus, int phase) {
    if (!measured.has_bus_injection(bus)) {
        return false;
    }
    auto const& injection = measured.bus_injection(bus);
    return phase_entry(injection.real_component.value, phase) == 0.0 &&
           phase_entry(injection.imag_component.value, phase) == 0.0 &&
           phase_entry(injection.real_component.variance, phase) == 0.0 &&
           phase_entry(injection.imag_component.variance, phase) == 0.0;
}

} // namespace injection_detail

// Requested injection uncertainties. Request rows are validated by validate_context before the estimate.
//
// The power injected at bus i is S_i = U_i conj(sum_k Y_ik U_k) over the bus itself and its neighbours k in
// the admittance matrix. Its first-order variance is d C d^T over the rectangular voltages of all of them,
// with the same derivative as the branch power. Two neighbours of i need not be on the LU sparsity pattern;
// CovarianceCompletion gives their covariance. Power does not depend on the angle reference.
//
// A result is invalid if the covariance of one of the bus pairs is. At an exact zero-injection constraint
// sigma is 0 and exact_zero_enforced is set, while the computed variance and its status are kept.
//
// With request.pair_diagnostics, result.pairs lists every pair of buses behind the results once, in
// ascending order.
template <class Workspace, symmetry_tag sym>
void injection_results(Workspace const& workspace, YBus<sym> const& y_bus, MeasuredValues<sym> const& measured,
                       StateEstimationUQRequest const& request, StateEstimationUQResult<sym>& result) {
    constexpr int phases = Workspace::phase_count;
    using Derivative = typename Workspace::RectangularVector;
    result.injection.resize(request.injection.size());
    if (request.injection.empty()) {
        return;
    }

    // The neighbours of every bus in the admittance matrix are also its neighbours in the augmented matrix.
    Idx const n_bus = y_bus.size();
    auto const& row_indptr = y_bus.row_indptr();
    auto const& col_indices = y_bus.col_indices();
    completion_detail::Graph neighbours;
    neighbours.offsets.reserve(static_cast<std::size_t>(n_bus + 1));
    neighbours.targets.reserve(col_indices.size());
    for (Idx bus = 0; bus != n_bus; ++bus) {
        for (Idx k = row_indptr[bus]; k != row_indptr[bus + 1]; ++k) {
            if (col_indices[k] != bus) {
                neighbours.targets.push_back(col_indices[k]);
            }
        }
        neighbours.offsets.push_back(static_cast<Idx>(neighbours.targets.size()));
    }
    CovarianceCompletion<Workspace> completion{
        workspace, std::move(neighbours), completion_detail::undirected(n_bus, y_bus.math_topology().branch_bus_idx),
        request.pair_diagnostics};

    std::vector<std::pair<Idx, Idx>> pairs;
    std::vector<Derivative> active;
    std::vector<Derivative> reactive;
    for (std::size_t index = 0; index != request.injection.size(); ++index) {
        auto const& item = request.injection[index];
        auto& output = result.injection[index];
        // The bus itself and its neighbours, in ascending order.
        Idx const first = row_indptr[item.bus];
        auto const count = static_cast<std::size_t>(row_indptr[item.bus + 1] - first);
        auto const bus_at = [&](std::size_t a) { return col_indices[first + static_cast<Idx>(a)]; };
        active.assign(count, Derivative::Zero());
        reactive.assign(count, Derivative::Zero());

        for (int phase = 0; phase != phases; ++phase) {
            // I = sum_k c_k U_k, and others = sum of conj(c_k U_k) over all voltages but the bus's own.
            DoubleComplex others{};
            for (std::size_t a = 0; a != count; ++a) {
                for (int q = 0; q != phases; ++q) {
                    if (bus_at(a) == item.bus && q == phase) {
                        continue;
                    }
                    DoubleComplex const coefficient =
                        branch_detail::admittance_entry(y_bus.admittance()[first + static_cast<Idx>(a)], phase, q);
                    others += std::conj(coefficient * workspace.phase_voltage(bus_at(a), q));
                }
            }
            DoubleComplex const voltage = workspace.phase_voltage(item.bus, phase);
            for (std::size_t a = 0; a != count; ++a) {
                for (int q = 0; q != phases; ++q) {
                    DoubleComplex const coefficient =
                        branch_detail::admittance_entry(y_bus.admittance()[first + static_cast<Idx>(a)], phase, q);
                    auto const [d_real, d_imag] = branch_detail::power_derivative(
                        voltage, coefficient, bus_at(a) == item.bus && q == phase, others);
                    active[a](q) = d_real.real();
                    active[a](phases + q) = d_imag.real();
                    reactive[a](q) = d_real.imag();
                    reactive[a](phases + q) = d_imag.imag();
                }
            }

            // d C d^T over all pairs of the neighbourhood; a pair a < b counts twice.
            RealSum active_sum;
            RealSum reactive_sum;
            bool valid = true;
            auto const add = [&](std::size_t a, std::size_t b, auto const& block, double weight) {
                valid = valid && block.valid;
                for (int p = 0; p != 2 * phases; ++p) {
                    for (int q = 0; q != 2 * phases; ++q) {
                        active_sum.add(weight * active[a](p) * block.value(p, q) * active[b](q));
                        reactive_sum.add(weight * reactive[a](p) * block.value(p, q) * reactive[b](q));
                    }
                }
            };
            for (std::size_t a = 0; a != count; ++a) {
                add(a, a, completion.diagonal(bus_at(a)), 1.0);
                for (std::size_t b = a + 1; b != count; ++b) {
                    add(a, b, completion.block(bus_at(a), bus_at(b)), 2.0);
                }
            }

            bool const zero = injection_detail::exact_zero(measured, item.bus, phase);
            auto const finalize = [zero, valid](RealSum const& sum) {
                UQPhaseValue value = finalize_variance(sum, valid);
                if (zero) {
                    value.sigma = 0.0;
                    value.exact_zero_enforced = true;
                }
                return value;
            };
            if (item.active_power) {
                output.active_power.phase[phase] = finalize(active_sum);
            }
            if (item.reactive_power) {
                output.reactive_power.phase[phase] = finalize(reactive_sum);
            }
        }
        if (request.pair_diagnostics) {
            for (std::size_t a = 0; a != count; ++a) {
                for (std::size_t b = a + 1; b != count; ++b) {
                    pairs.emplace_back(bus_at(a), bus_at(b));
                }
            }
        }
    }
    std::ranges::sort(pairs);
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    result.pairs.reserve(pairs.size());
    for (auto const& [a, b] : pairs) {
        result.pairs.push_back(completion.pair(a, b).diagnostic);
    }
}

} // namespace power_grid_model::math_solver::state_estimation_uq
