// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// State covariance of one island math model from selected blocks of the inverse augmented gain matrix.

#include "sparse_lu_solver.hpp"
#include "y_bus.hpp"

#include "../common/common.hpp"
#include "../common/three_phase_tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

namespace power_grid_model::math_solver::state_estimation_uq {

// Entries and blocks of the state covariance C_x from the inverse of the augmented state-estimation matrix.
// Abar_ps is the matrix that state estimation assembles, with perfect shuffling (one block per bus) and variance
// normalization by alpha. The inverse Zbar = Abar_ps^-1, C_x = alpha * (state rows and columns of Zbar).
// Each bus block starts with the StateDim state coordinates of the bus, which `coordinate` indexes below:
// - ILSE: U (symmetric) or U_a, U_b, U_c (asymmetric), the complex phase voltages
// - NRSE: theta, |U| (symmetric) or theta_a, theta_b, theta_c, |U_a|, |U_b|, |U_c| (asymmetric), in rad and p.u.
template <class Gain, class Unknown, int StateDim = Gain::RowsAtCompileTime / 2> class SparseCovarianceWorkspace {
  public:
    using Scalar = typename Gain::Scalar;
    // True for the complex voltage state of ILSE; false for the real polar state of NRSE.
    static constexpr bool uses_complex_state = std::is_same_v<Scalar, DoubleComplex>;
    static constexpr int phase_count = uses_complex_state ? StateDim : StateDim / 2;
    static_assert(phase_count == 1 || phase_count == 3);
    using Symmetry = std::conditional_t<phase_count == 1, symmetric_t, asymmetric_t>;
    using StateMatrix = Eigen::Matrix<Scalar, StateDim, StateDim>;
    // A whole block of Zbar: the state coordinates of a bus followed by its multipliers.
    using RawMatrix = Eigen::Matrix<Scalar, Gain::RowsAtCompileTime, Gain::ColsAtCompileTime>;
    // Covariance of the rectangular voltages x_i = [Re U_i, Im U_i] of buses, real parts of all phases first.
    using RectangularMatrix = Eigen::Matrix<double, 2 * phase_count, 2 * phase_count>;
    using RectangularVector = Eigen::Matrix<double, 2 * phase_count, 1>;
    using Solver = SparseLUSolver<Gain, Unknown, Unknown>;

    // LU factors of ILSE, whose augmented matrix does not depend on the state and needs no second factorization.
    struct Factorization {
        Solver& solver;
        typename Solver::BlockPermArray const& permutations;
    };

    // `gain` is the augmented matrix Abar_ps at the estimate `u`. If factorization has been done, `gain` holds its
    // LU factors and `factorization` gives the solver and block permutations that produced them.
    // `absolute_angle` is MeasuredValues::has_angle(), which decides the angle reference of the SE output.
    // Throws SparseMatrixError if the matrix cannot be factorized without pivot perturbation.
    // Only the uncertainty fails then, not the state estimation: calculate_results() catches the error and reports
    // the requested uncertainties as invalid, and the SE results are returned as usual.
    SparseCovarianceWorkspace(YBus<Symmetry> const& y_bus, std::vector<Gain> gain, double variance_normalization,
                              ComplexValueVector<Symmetry> const& u, bool absolute_angle,
                              Factorization const* factorization = nullptr)
        : inverse_{std::move(gain)},
          variance_normalization_{variance_normalization},
          slack_bus_{y_bus.math_topology().slack_bus},
          report_relative_angle_{!absolute_angle},
          row_indptr_{y_bus.row_indptr_lu()},
          col_indices_{y_bus.col_indices_lu()},
          lu_diag_{y_bus.lu_diag()},
          u_{u} {
        Solver owned_solver{y_bus.row_indptr_lu(), y_bus.col_indices_lu(), y_bus.lu_diag()};
        Solver& solver = factorization ? factorization->solver : owned_solver;
        typename Solver::BlockPermArray owned_perm;
        if (!factorization) {
            owned_perm.resize(u_.size());
            solver.prefactorize(inverse_, owned_perm, false);
        }
        auto const& perm = factorization ? factorization->permutations : owned_perm;

        // Relative angles need Cov(theta_i, theta_s), from the slack column of Zbar, which is in general not on the
        // LU pattern. Solve for it now, because the selective inverse below overwrites the LU factors.
        if (report_relative_angle_) {
            solve_slack_column(solver, perm);
        }
        // Replace the factors in place by the blocks of Zbar on the LU sparsity pattern.
        solver.inplace_selective_inverse_with_prefactorized_matrix(inverse_, perm);
    }

    Idx slack_bus() const { return slack_bus_; }
    // True when SE reports angles relative to phase a of the slack bus.
    bool report_relative_angle() const { return report_relative_angle_; }

    DoubleComplex phase_voltage(Idx bus, int phase) const {
        if constexpr (phase_count == 1) {
            return u_[bus];
        } else {
            return u_[bus](phase);
        }
    }

    // The factor alpha in C_x = alpha * (state rows and columns of Zbar).
    double variance_normalization() const { return variance_normalization_; }

    // Variance C_x,ii(coordinate, coordinate) of the given state coordinate of `bus`.
    double state_variance(Idx bus, int coordinate) const {
        return variance_normalization_ * std::real(inverse_[lu_diag_[bus]](coordinate, coordinate));
    }

    // Covariance C_x,is(coordinate, 0) between the given state coordinate of `bus` and the first state coordinate of
    // the slack bus, which is its phase-a voltage for ILSE and its phase-a angle for NRSE.
    // Only available for relative angles.
    Scalar slack_covariance(Idx bus, int coordinate) const {
        return variance_normalization_ * slack_column_[bus](coordinate);
    }

    // Covariance block C_x,ij of the state coordinates of two buses. The pair must be on the LU sparsity
    // pattern, which holds for a bus with itself and for the two buses of a branch.
    StateMatrix state_covariance(Idx row, Idx col) const {
        auto const index = find_block_index(row, col);
        assert(index.has_value());
        return variance_normalization_ * inverse_[*index].matrix().template topLeftCorner<StateDim, StateDim>();
    }

    // The blocks of Zbar themselves, for the completion of pairs that are not on the LU sparsity pattern.
    Idx block_count() const { return static_cast<Idx>(inverse_.size()); }
    std::optional<Idx> find_block_index(Idx row, Idx col) const {
        auto const first = col_indices_.begin() + row_indptr_[row];
        auto const last = col_indices_.begin() + row_indptr_[row + 1];
        auto const found = std::lower_bound(first, last, col);
        if (found == last || *found != col) {
            return std::nullopt;
        }
        return static_cast<Idx>(found - col_indices_.begin());
    }
    Gain const& inverse_block_at(Idx index) const { return inverse_[index]; }

    // Power and current are functions of the rectangular voltages, whatever the state. The two functions
    // below give their covariance for either state, so that code which uses them needs no case distinction.

    // Cov(x_row, x_col). The bus pair must be on the LU sparsity pattern.
    RectangularMatrix rectangular_covariance(Idx row, Idx col) const {
        return rectangular_covariance(row, col, state_covariance(row, col));
    }
    // The same for a given state covariance block C_ij of the two buses.
    RectangularMatrix rectangular_covariance(Idx row, Idx col, StateMatrix const& state) const {
        if constexpr (uses_complex_state) {
            // A proper complex dU with V = E[dU dU^H] has Cov(Re, Re) = Cov(Im, Im) = Re(V) / 2 and
            // Cov(Im, Re) = -Cov(Re, Im) = Im(V) / 2.
            RectangularMatrix result;
            result.template topLeftCorner<phase_count, phase_count>() = state.real() / 2.0;
            result.template topRightCorner<phase_count, phase_count>() = -state.imag() / 2.0;
            result.template bottomLeftCorner<phase_count, phase_count>() = state.imag() / 2.0;
            result.template bottomRightCorner<phase_count, phase_count>() = state.real() / 2.0;
            return result;
        } else {
            return polar_to_rectangular(row) * state * polar_to_rectangular(col).transpose();
        }
    }

    // Cov(x_bus, theta_s), with theta_s the phase-a angle of the slack bus. Only available for relative angles.
    RectangularVector rectangular_slack_covariance(Idx bus) const {
        RectangularVector result;
        if constexpr (uses_complex_state) {
            // theta_s = Im(dU_s / U_s), so with c = V_is / conj(U_s):
            // Cov(Re dU_i, theta_s) = -Im(c) / 2 and Cov(Im dU_i, theta_s) = Re(c) / 2.
            DoubleComplex const slack_voltage = phase_voltage(slack_bus_, 0);
            for (int phase = 0; phase != phase_count; ++phase) {
                DoubleComplex const c = slack_covariance(bus, phase) / std::conj(slack_voltage);
                result(phase) = -c.imag() / 2.0;
                result(phase_count + phase) = c.real() / 2.0;
            }
            return result;
        } else {
            for (int coordinate = 0; coordinate != StateDim; ++coordinate) {
                result(coordinate) = slack_covariance(bus, coordinate);
            }
            return polar_to_rectangular(bus) * result;
        }
    }

  private:
    using StateColumn = Eigen::Matrix<Scalar, StateDim, 1>;

    // Derivative of x_i with respect to the polar state [theta_i, |U_i|] of the same bus.
    RectangularMatrix polar_to_rectangular(Idx bus) const {
        RectangularMatrix result = RectangularMatrix::Zero();
        for (int phase = 0; phase != phase_count; ++phase) {
            DoubleComplex const voltage = phase_voltage(bus, phase);
            DoubleComplex const direction = std::polar(1.0, std::arg(voltage));
            result(phase, phase) = -voltage.imag();
            result(phase, phase_count + phase) = direction.real();
            result(phase_count + phase, phase) = voltage.real();
            result(phase_count + phase, phase_count + phase) = direction.imag();
        }
        return result;
    }

    // Solve Abar_ps * t = e for the first state coordinate of the slack bus and keep the state rows of t.
    void solve_slack_column(Solver& solver, typename Solver::BlockPermArray const& perm) {
        std::vector<Unknown> rhs(u_.size());
        std::vector<Unknown> solution(u_.size());
        rhs[slack_bus_](0) = 1.0;
        solver.solve_with_prefactorized_matrix(inverse_, perm, rhs, solution);
        slack_column_.resize(u_.size());
        for (Idx bus = 0; bus != static_cast<Idx>(u_.size()); ++bus) {
            slack_column_[bus] = solution[bus].matrix().template topRows<StateDim>();
        }
    }

    std::vector<Gain> inverse_; // LU factors during construction, afterwards blocks of Zbar.
    double variance_normalization_;
    Idx slack_bus_;
    bool report_relative_angle_;
    IdxVector row_indptr_;
    IdxVector col_indices_;
    IdxVector lu_diag_;
    ComplexValueVector<Symmetry> u_;
    std::vector<StateColumn> slack_column_; // state rows of the slack column of Zbar. Empty for absolute angles.
};

} // namespace power_grid_model::math_solver::state_estimation_uq
