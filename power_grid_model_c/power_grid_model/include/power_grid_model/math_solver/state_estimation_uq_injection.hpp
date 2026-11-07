// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "state_estimation_uq_branch.hpp"
#include "state_estimation_uq_completion.hpp"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace power_grid_model::math_solver::state_estimation_uq {
namespace injection_detail {

template <int phases> struct Neighbor {
    Idx bus;
    std::array<DoubleComplex, phases> admittance{};
    std::array<double, 2 * phases> active{}, reactive{};
};

// Actual per-phase nodal power, S_ip = U_ip conj(sum_kq Y_ip,kq U_kq).
// Differentiate all neighboring phases, including the terminal's voltage term.
template <int phases>
void gradients(Neighbor<phases>& neighbor, Idx terminal, int output_phase, DoubleComplex voltage,
               DoubleComplex others) {
    for (int p = 0; p != phases; ++p) {
        auto const [x, y] =
            power_axes(voltage, neighbor.admittance[p], neighbor.bus == terminal && p == output_phase, others);
        neighbor.active[p] = x.real();
        neighbor.active[phases + p] = y.real();
        neighbor.reactive[p] = x.imag();
        neighbor.reactive[phases + p] = y.imag();
    }
}

template <int phases, class Block>
void add_block(Neighbor<phases> const& left, Neighbor<phases> const& right, Block const& block, bool cross,
               RealSum& active, RealSum& reactive) {
    auto entry = [&block](int row, int col) {
        auto const& value = block.value;
        if constexpr (Eigen::NumTraits<typename std::remove_cvref_t<decltype(value)>::Scalar>::IsComplex) {
            DoubleComplex const v = value(row % phases, col % phases);
            return (row / phases == col / phases ? v.real() : (row < phases ? -v.imag() : v.imag())) / 2.0;
        } else {
            return value(row, col);
        }
    };
    auto add = [&](int row, int col, double value) {
        active.add_product(left.active[row], value, right.active[col]);
        reactive.add_product(left.reactive[row], value, right.reactive[col]);
    };
    for (int row = 0; row < 2 * phases; ++row)
        for (int col = 0; col < 2 * phases; ++col) {
            double const value = entry(row, col);
            if (cross)
                add(row, col, 2.0 * value);
            else if (row == col)
                add(row, col, value);
            else if (row < col) {
                double const mirrored = entry(col, row);
                if (value == mirrored)
                    add(row, col, 2.0 * value);
                else {
                    add(row, col, value);
                    add(col, row, mirrored);
                }
            }
        }
}

template <class Value> decltype(auto) phase(Value const& value, int p) {
    if constexpr (std::is_convertible_v<Value, DoubleComplex>)
        return value;
    else
        return value(p);
}

template <class Measurements> bool exact_zero(Measurements const& measured, Idx bus, int phase) {
    if (!measured.has_bus_injection(bus))
        return false;
    auto const& injection = measured.bus_injection(bus);
    return injection_detail::phase(injection.real_component.value, phase) == 0 &&
           injection_detail::phase(injection.imag_component.value, phase) == 0 &&
           injection_detail::phase(injection.real_component.variance, phase) == 0 &&
           injection_detail::phase(injection.imag_component.variance, phase) == 0;
}
inline void enforce_zero(UQPhaseValue& value) {
    value.sigma = 0;
    value.exact_zero_enforced = true;
}
} // namespace injection_detail

// Called after calculate_voltage_uq_results and branch_results on the same returned-state
// workspace. Original Y-bus support is the conservative unfactorized augmented
// graph; only topology branches define physical fallback edges. LU fill is used
// solely by CovarianceCompletion as original raw/donor availability.
//
// Requested pair diagnostics contain each distinct closed-neighborhood pair
// once, in canonical bus order. The caller supplies fresh injection/pair result
// storage and the physical-block cache shared with branch_results.
template <class Workspace, class YBusType, class Measurements, symmetry_tag sym>
void injection_results(Workspace const& workspace, YBusType const& y_bus, Measurements const& measured,
                       StateEstimationUQRequest const& request, StateEstimationUQResult<sym>& result,
                       PhysicalBlocks<Workspace>& stored) {
    constexpr int phases = is_symmetric_v<sym> ? 1 : 3;
    result.injection.resize(request.injection.size());
    if (request.injection.empty())
        return;

    // The Y-bus rows without their own bus are the augmented graph.
    Idx const n = y_bus.size();
    completion_detail::Graph augmented_graph;
    augmented_graph.offsets.reserve(static_cast<std::size_t>(n + 1));
    augmented_graph.targets.reserve(y_bus.col_indices().size());
    for (Idx bus = 0; bus < n; ++bus) {
        for (Idx p = y_bus.row_indptr()[bus]; p < y_bus.row_indptr()[bus + 1]; ++p)
            if (Idx const other = y_bus.col_indices()[p]; other != bus)
                augmented_graph.targets.push_back(other);
        augmented_graph.offsets.push_back(static_cast<Idx>(augmented_graph.targets.size()));
    }
    auto physical_graph = completion_detail::undirected(y_bus.size(), y_bus.math_topology().branch_bus_idx);
    CovarianceCompletion completion{workspace, std::move(augmented_graph), std::move(physical_graph), stored,
                                    request.pair_diagnostics};
    std::vector<std::pair<Idx, Idx>> visited; // Only for requested diagnostics.
    std::vector<injection_detail::Neighbor<phases>> neighbors;
    for (std::size_t index = 0; index < request.injection.size(); ++index) {
        auto const& item = request.injection[index];
        auto& output = result.injection[index];
        for (int phase = 0; phase != phases; ++phase) {
            neighbors.clear();
            ComplexSum current, others;
            DoubleComplex const voltage = workspace.phase_voltage(item.bus, phase);
            bool defined = is_finite(voltage);
            for (Idx p = y_bus.row_indptr()[item.bus]; p < y_bus.row_indptr()[item.bus + 1]; ++p) {
                Idx const bus = y_bus.col_indices()[p];
                auto& neighbor = neighbors.emplace_back();
                neighbor.bus = bus;
                for (int q = 0; q != phases; ++q) {
                    DoubleComplex const admittance = branch_detail::coefficient(y_bus.admittance()[p], phase, q);
                    DoubleComplex const u = workspace.phase_voltage(bus, q);
                    neighbor.admittance[q] = admittance;
                    ComplexSum term;
                    term.add_product(admittance, u);
                    current.add(term.value());
                    if (bus != item.bus || q != phase)
                        others.add(std::conj(term.value()));
                    defined = defined && is_finite(admittance) && is_finite(u);
                }
            }
            defined = defined && is_finite(current.value());
            for (auto& neighbor : neighbors)
                injection_detail::gradients(neighbor, item.bus, phase, voltage, others.value());

            RealSum active, reactive;
            bool valid = defined;
            for (std::size_t i = 0; i < neighbors.size(); ++i) {
                auto const& diagonal = completion.diagonal(neighbors[i].bus);
                valid = valid && diagonal.valid;
                injection_detail::add_block(neighbors[i], neighbors[i], diagonal, false, active, reactive);
                for (std::size_t j = i + 1; j < neighbors.size(); ++j) {
                    auto const& block = completion.block(neighbors[i].bus, neighbors[j].bus);
                    valid = valid && block.valid;
                    injection_detail::add_block(neighbors[i], neighbors[j], block, true, active, reactive);
                }
            }
            bool const zero = injection_detail::exact_zero(measured, item.bus, phase);
            if (item.active_power) {
                output.active_power.phase[phase] = finalize_variance(active, valid);
                if (zero)
                    injection_detail::enforce_zero(output.active_power.phase[phase]);
            }
            if (item.reactive_power) {
                output.reactive_power.phase[phase] = finalize_variance(reactive, valid);
                if (zero)
                    injection_detail::enforce_zero(output.reactive_power.phase[phase]);
            }
        }
        if (request.pair_diagnostics)
            for (std::size_t i = 0; i < neighbors.size(); ++i)
                for (std::size_t j = i + 1; j < neighbors.size(); ++j)
                    visited.emplace_back(neighbors[i].bus, neighbors[j].bus);
    }
    std::ranges::sort(visited);
    visited.erase(std::unique(visited.begin(), visited.end()), visited.end());
    result.pairs.reserve(visited.size());
    for (auto const& [a, b] : visited)
        result.pairs.push_back(completion.pair(a, b).diagnostic);
}
} // namespace power_grid_model::math_solver::state_estimation_uq
