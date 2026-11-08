// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "state_estimation_uq_workspace.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <optional>
#include <type_traits>
#include <vector>

namespace power_grid_model::math_solver::state_estimation_uq {

namespace branch_detail {
template <int phases> using PhaseRow = std::array<double, 4 * phases>;

template <class Value> DoubleComplex coefficient(Value const& value, int p, int q) {
    if constexpr (std::is_convertible_v<Value, DoubleComplex>)
        return value;
    else
        return value(p, q);
}
template <class Workspace>
constexpr bool uses_complex_state = Eigen::NumTraits<typename Workspace::StateMatrix::Scalar>::IsComplex;

// The same real-coordinate map covers proper-complex ILSE (including the
// power output pseudo-covariance) and NRSE physical rectangular covariance.
// Coordinates per bus are [Re U_abc, Im U_abc]. The terminal's own phase is
// regrouped through power_axes to preserve charging-current cancellation.
template <int phases>
std::array<PhaseRow<phases>, 2> power_rows(std::array<std::array<DoubleComplex, phases>, 2> const& c,
                                           DoubleComplex voltage, DoubleComplex others, int terminal,
                                           int output_phase) {
    std::array<PhaseRow<phases>, 2> rows{};
    for (int bus = 0; bus != 2; ++bus) {
        for (int p = 0; p != phases; ++p) {
            auto const [x, y] = power_axes(voltage, c[bus][p], bus == terminal && p == output_phase, others);
            rows[0][2 * phases * bus + p] = x.real();
            rows[0][2 * phases * bus + phases + p] = y.real();
            rows[1][2 * phases * bus + p] = x.imag();
            rows[1][2 * phases * bus + phases + p] = y.imag();
        }
    }
    return rows;
}

// Stored workspace blocks already carry a canonical physical view.
// Reference blocks use the same inexpensive projection-size limit.
template <class Workspace>
auto physical_pair(Workspace const& workspace, Idx first, Idx second, typename Workspace::PhysicalBlock forward) {
    if (first == second || forward.symmetry_projected) {
        return forward;
    }
    auto const reverse = workspace.physical_block(second, first);
    FrobeniusNorm skew;
    for (int r = 0; r != forward.value.rows(); ++r) {
        for (int c = 0; c != forward.value.cols(); ++c) {
            auto const a = forward.value(r, c);
            auto const b = conjugate(reverse.value(c, r));
            skew.add(a - b);
            forward.value(r, c) = average(a, b);
        }
    }
    double const residual = skew.value();
    double const scale = average(forward.construction_scale, reverse.construction_scale);
    double const ratio =
        scale > 0.0 ? (residual / scale) / 2.0 : (residual == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    forward.valid = forward.valid && reverse.valid && ratio <= symmetry_rtol;
    forward.construction_scale = scale;
    forward.symmetry_projected = forward.valid;
    return forward;
}
template <class Workspace> auto physical_pair(Workspace const& workspace, Idx first, Idx second) {
    return physical_pair(workspace, first, second, workspace.physical_block(first, second));
}
// A branch contraction sums all terminal entries, so a negative contribution from one bus's
// block can be masked by the others. Accept a Hermitian physical diagonal block only if
// lambda_min(D) >= -round_rtol * construction_scale.
template <class Block> bool positive_semidefinite(Block const& block) {
    using Matrix = std::remove_cvref_t<decltype(block.value)>;
    double const tolerance = round_rtol * block.construction_scale;
    double smallest{};
    if constexpr (Matrix::RowsAtCompileTime == 1) {
        smallest = std::real(block.value(0, 0));
    } else if constexpr (Matrix::RowsAtCompileTime == 2) {
        // Closed form for the real symmetric NRSE block [[x, y], [y, z]].
        double const x = block.value(0, 0);
        double const y = block.value(1, 0);
        double const z = block.value(1, 1);
        double const half_difference = (x - z) / 2.0;
        smallest = ((x + z) / 2.0) - std::sqrt((half_difference * half_difference) + (y * y));
    } else {
        Eigen::SelfAdjointEigenSolver<Matrix> eigen{block.value, Eigen::EigenvaluesOnly};
        if (eigen.info() != Eigen::Success) {
            return false;
        }
        smallest = eigen.eigenvalues()(0);
    }
    return std::isfinite(smallest) && smallest >= -tolerance;
}

// Canonical (first <= second) terminal-pair blocks at their LU positions, read
// through a per-call PhysicalBlocks cache. Each block is derived once and shared
// by every quantity, both terminals, all branches at a bus and, when the cache is
// shared, the injection helper. The reverse orientation is returned as the
// adjoint, which SparseCovarianceWorkspace's physical_pair gives exactly.
// Diagonal blocks must also pass positive_semidefinite, evaluated once per bus.
template <class Workspace> class TerminalBlocks {
  public:
    using Block = typename Workspace::PhysicalBlock;

    explicit TerminalBlocks(PhysicalBlocks<Workspace>& stored)
        : stored_{stored}, semidefinite_(stored.workspace().augmented_matrix.size()) {}

    Block operator()(Idx first, Idx second) {
        if (first > second) {
            auto result = (*this)(second, first);
            result.value = result.value.adjoint().eval();
            return result;
        }
        auto const& workspace = stored_.workspace();
        auto const index = workspace.find_block_index(first, second);
        if (!index) {
            // No stored inverse block for a pair outside the LU pattern: undefined.
            Block missing;
            missing.valid = false;
            return missing;
        }
        auto block = physical_pair(workspace, first, second, stored_.stored(*index, first, second));
        if (first == second) {
            auto& semidefinite = semidefinite_[*index];
            if (!semidefinite) {
                semidefinite = positive_semidefinite(block);
            }
            block.valid = block.valid && *semidefinite;
        }
        return block;
    }

  private:
    PhysicalBlocks<Workspace>& stored_;
    std::vector<std::optional<bool>> semidefinite_;
};

template <class Workspace> using TerminalCovariance = std::array<std::array<typename Workspace::PhysicalBlock, 2>, 2>;

// ILSE slack-relative current-angle terms: the slack angle variance and, for a
// real row h, Cov(h dU, angle U_s) = Re(j h V_is / conj(U_s)) / 2 per terminal,
// with V_is = alpha times the slack selector column, as double products.
template <int phases, class Workspace>
void slack_terms(Workspace const& workspace, std::array<Idx, 2> const& buses, PhaseRow<phases> const& row,
                 SlackAngle const& slack, bool& defined, RealSum& sum) {
    defined = defined && slack.defined;
    sum.add(slack.variance_term);
    for (int i = 0; i != 2; ++i)
        if (buses[i] >= 0)
            for (int p = 0; p != phases; ++p) {
                DoubleComplex const h{row[2 * phases * i + p], -row[2 * phases * i + phases + p]};
                sum.add(imag_part_of_product(workspace.variance_normalization, workspace.slack_column[buses[i]](p), h,
                                             std::conj(slack.inverse_voltage)));
            }
}

template <int phases, class Workspace>
double terminal_entry(TerminalCovariance<Workspace> const& blocks, int p, int q) {
    constexpr int coordinates = 2 * phases;
    auto const& value = blocks[p / coordinates][q / coordinates].value;
    int const a = p % coordinates, b = q % coordinates;
    if constexpr (uses_complex_state<Workspace>) {
        DoubleComplex const v = value(a % phases, b % phases);
        return (a / phases == b / phases ? v.real() : (a < phases ? -v.imag() : v.imag())) / 2.0;
    } else {
        return value(a, b);
    }
}

// Contract every endpoint, phase and real/imaginary cross term. Mirrored
// equal entries are one doubled product, preserving the existing scale rule.
template <int phases, class Workspace>
UQPhaseValue contract(Workspace const& workspace, std::array<Idx, 2> const& buses,
                      TerminalCovariance<Workspace> const& blocks, PhaseRow<phases> const& row, bool defined,
                      SlackAngle const* slack = nullptr) {
    constexpr int coordinates = 2 * phases;
    RealSum sum;
    for (int i = 0; i != 2; ++i)
        for (int j = 0; j != 2; ++j)
            if (buses[i] >= 0 && buses[j] >= 0)
                defined = defined && blocks[i][j].valid;
    for (int p = 0; p != 2 * coordinates; ++p) {
        if (buses[p / coordinates] < 0)
            continue;
        sum.add_product(row[p], terminal_entry<phases, Workspace>(blocks, p, p), row[p]);
        for (int q = p + 1; q != 2 * coordinates; ++q) {
            if (buses[q / coordinates] < 0)
                continue;
            double const forward = terminal_entry<phases, Workspace>(blocks, p, q);
            double const backward = terminal_entry<phases, Workspace>(blocks, q, p);
            if (forward == backward)
                sum.add_product(row[p], 2.0 * forward, row[q]);
            else {
                sum.add_product(row[p], forward, row[q]);
                sum.add_product(row[q], backward, row[p]);
            }
        }
    }
    if constexpr (uses_complex_state<Workspace>)
        if (slack != nullptr)
            slack_terms<phases>(workspace, buses, row, *slack, defined, sum);
    return finalize_variance(sum, defined);
}
} // namespace branch_detail

// Request indices and sides are validated by validate_context before the workspace is
// built. Append only branch results; voltage and injection results stay owned by
// their respective helpers. All contractions use at most two terminal buses.
// The caller supplies fresh result storage and may share stored blocks with injection_results.
template <class Workspace, symmetry_tag sym>
void branch_results(Workspace const& workspace, YBus<sym> const& y_bus, StateEstimationUQRequest const& request,
                    StateEstimationUQResult<sym>& result, PhysicalBlocks<Workspace>& stored) {
    constexpr int phases = is_symmetric_v<sym> ? 1 : 3;
    result.branch.resize(request.branch.size());
    branch_detail::TerminalBlocks<Workspace> blocks{stored};
    std::optional<SlackAngle> slack;
    if constexpr (branch_detail::uses_complex_state<Workspace>)
        if (workspace.report_relative_angle && std::ranges::any_of(request.branch, &BranchUQRequest::current_angle))
            slack = slack_angle(workspace);
    for (std::size_t index = 0; index != request.branch.size(); ++index) {
        auto const& item = request.branch[index];
        auto& output = result.branch[index];
        auto const& endpoints = y_bus.math_topology().branch_bus_idx[item.branch];
        std::array<Idx, 2> const buses{endpoints[0], endpoints[1]};
        int const terminal = item.side == BranchSide::from ? 0 : 1;
        auto const& param = y_bus.math_model_param().branch_param[item.branch];
        branch_detail::TerminalCovariance<Workspace> terminals{};
        for (int i = 0; i != 2; ++i)
            for (int j = 0; j != 2; ++j)
                if (buses[i] >= 0 && buses[j] >= 0)
                    terminals[i][j] = blocks(buses[i], buses[j]);
        for (int phase = 0; phase != phases; ++phase) {
            std::array<std::array<DoubleComplex, phases>, 2> c{};
            bool defined = buses[terminal] >= 0;
            DoubleComplex voltage{};
            if (defined)
                voltage = workspace.phase_voltage(buses[terminal], phase);
            defined = defined && is_finite(voltage);
            ComplexSum current_sum, others_sum;
            for (int i = 0; i != 2; ++i)
                for (int p = 0; p != phases; ++p) {
                    c[i][p] = branch_detail::coefficient(terminal == 0 ? (i == 0 ? param.yff() : param.yft())
                                                                       : (i == 0 ? param.ytf() : param.ytt()),
                                                         phase, p);
                    if (buses[i] < 0)
                        continue;
                    DoubleComplex const u = workspace.phase_voltage(buses[i], p);
                    ComplexSum term;
                    term.add_product(c[i][p], u);
                    current_sum.add(term.value());
                    if (i != terminal || p != phase)
                        others_sum.add(std::conj(term.value()));
                    defined = defined && is_finite(u) && is_finite(c[i][p]);
                }
            DoubleComplex const current = current_sum.value();
            defined = defined && is_finite(current);
            if (item.active_power || item.reactive_power) {
                auto const rows = branch_detail::power_rows<phases>(c, voltage, others_sum.value(), terminal, phase);
                if (item.active_power)
                    output.active_power.phase[phase] =
                        branch_detail::contract<phases>(workspace, buses, terminals, rows[0], defined);
                if (item.reactive_power)
                    output.reactive_power.phase[phase] =
                        branch_detail::contract<phases>(workspace, buses, terminals, rows[1], defined);
            }
            // A current whose squared modulus is not a normal double has no
            // accurate direction or reciprocal: undefined, as at zero current.
            bool const nonzero = defined && ((current.real() * current.real()) + (current.imag() * current.imag())) >=
                                                std::numeric_limits<double>::min();
            if (item.current_magnitude || item.current_angle) {
                branch_detail::PhaseRow<phases> magnitude{}, angle{};
                if (nonzero) {
                    DoubleComplex const direction = std::conj(current) / modulus(current);
                    DoubleComplex const inverse = reciprocal(current);
                    for (int i = 0; i != 2; ++i)
                        for (int p = 0; p != phases; ++p)
                            for (int a = 0; a != 2; ++a) {
                                DoubleComplex const delta =
                                    a == 0 ? c[i][p] : DoubleComplex{-c[i][p].imag(), c[i][p].real()};
                                magnitude[2 * phases * i + a * phases + p] = std::real(direction * delta);
                                angle[2 * phases * i + a * phases + p] = std::imag(delta * inverse);
                            }
                }
                if (item.current_magnitude)
                    output.current_magnitude.phase[phase] =
                        branch_detail::contract<phases>(workspace, buses, terminals, magnitude, nonzero);
                if (item.current_angle)
                    output.current_angle.phase[phase] = branch_detail::contract<phases>(
                        workspace, buses, terminals, angle, nonzero, slack ? &*slack : nullptr);
            }
        }
    }
}

} // namespace power_grid_model::math_solver::state_estimation_uq
