// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// Sparse augmented-inverse workspace and voltage uncertainty calculations for one island math model.

#include "sparse_lu_solver.hpp"
#include "state_estimation_uq.hpp"
#include "state_estimation_uq_numerics.hpp"
#include "y_bus.hpp"

#include "../common/common.hpp"
#include "../common/three_phase_tensor.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

namespace power_grid_model::math_solver::state_estimation_uq {

// Indices of the additive terms C_k that form a corrected state covariance block C_ij.
namespace correction_term {
inline constexpr std::size_t normalized_source = 0;
inline constexpr std::size_t virtual_noise = 1;
inline constexpr std::size_t row_reference = 2;
inline constexpr std::size_t col_reference = 3;
inline constexpr std::size_t slack_reference = 4;
inline constexpr std::size_t count = 5;
} // namespace correction_term

// Accumulates the construction scale S_C = sum_k ||C_k||_F across the covariance correction terms.
class CorrectionNorms {
  public:
    void add(std::size_t k, double x) { norms_[k].add(x); }
    void add(std::size_t k, DoubleComplex x) { norms_[k].add(x); }
    double sum() const {
        double result = 0.0;
        for (auto const& norm : norms_) {
            result += norm.value();
        }
        return result;
    }

  private:
    std::array<FrobeniusNorm, correction_term::count> norms_{};
};

// A covariance block with its value, construction scale and validity. symmetry_projected marks a
// block whose Hermitian symmetry has already been checked and projected.
template <class Matrix> struct CovarianceBlock {
    Matrix value = Matrix::Zero();
    double construction_scale = 0.0;
    bool valid = true;
    bool symmetry_projected = false;
};

template <class Matrix> std::array<Matrix, correction_term::count> zero_terms() {
    std::array<Matrix, correction_term::count> result;
    for (auto& term : result) {
        term.setZero();
    }
    return result;
}

// Covariance workspace for one island and one UQ request.
//
// Notation used throughout this file:
//   Abar   variance-normalized augmented WLS (KKT) matrix [[Gbar, Q^H], [Q, -Rbar]], bus-interleaved
//   B      Abar^-1, with bus blocks B_ij
//   q      state rows of a bus block: complex U (ILSE) or [theta, |U|] per phase (NRSE)
//   Z_ij   B_ij[q,q], the raw state corner of B_ij
//
// The constructor factorizes Abar (or reuses ILSE's factors) and overwrites `augmented_matrix` with the
// blocks B_ij on the LU fill pattern (block Takahashi selective inverse). `calculate_state_covariance_block`
// then returns corrected state covariance blocks without modifying B.
template <class Gain, class Unknown, int StateDim = Gain::RowsAtCompileTime / 2> class SparseCovarianceWorkspace {
  public:
    using Scalar = typename Gain::Scalar;
    // True for complex-voltage ILSE; false for real polar [theta, |U|] NRSE.
    static constexpr bool uses_complex_state = std::is_same_v<Scalar, DoubleComplex>;
    static constexpr int phase_count = uses_complex_state ? StateDim : StateDim / 2;
    static_assert(phase_count == 1 || phase_count == 3);
    using Symmetry = std::conditional_t<phase_count == 1, symmetric_t, asymmetric_t>;
    // Full augmented bus block (state coordinates q followed by auxiliary multipliers).
    using RawMatrix = Eigen::Matrix<Scalar, Gain::RowsAtCompileTime, Gain::ColsAtCompileTime>;
    using StateMatrix = Eigen::Matrix<Scalar, StateDim, StateDim>;
    // State rows q of one bus in a single inverse column of Abar.
    using StateColumn = Eigen::Matrix<Scalar, StateDim, 1>;
    // State rows T_i of the NRSE virtual-angle inverse columns (one column per phase).
    using VirtualColumnBlock = Eigen::Matrix<Scalar, StateDim, phase_count>;
    using RealStateMatrix = Eigen::Matrix<double, StateDim, StateDim>;
    // The state coordinates of one bus mapped to physical coordinates: complex voltages for ILSE,
    // rectangular [Re U, Im U] for NRSE.
    using PhysicalMatrix = StateMatrix;
    // Corrected covariance block C_ij in value, with roundoff bookkeeping over the correction terms C_k
    // (correction_term): terms[k] are the rounded C_k, term_scale(r,c) = sum_k |C_k(r,c)|,
    // construction_scale = sum_k ||C_k||_F, and has_nonzero_term(r,c) marks a term that is nonzero in
    // exact arithmetic. valid is false when a finiteness or symmetry check fails.
    struct StateCovarianceBlock : CovarianceBlock<StateMatrix> {
        std::array<StateMatrix, correction_term::count> terms = zero_terms<StateMatrix>();
        RealStateMatrix term_scale = RealStateMatrix::Zero();
        Eigen::Matrix<bool, StateDim, StateDim> has_nonzero_term = Eigen::Matrix<bool, StateDim, StateDim>::Zero();
    };
    using PhysicalBlock = CovarianceBlock<PhysicalMatrix>;
    using Solver = SparseLUSolver<Gain, Unknown, Unknown>;

    // Pre-existing LU factorization and block permutations from ILSE.
    // The workspace copies the factor blocks into `augmented_matrix` before destructive selective inversion.
    struct Factorization {
        Solver& solver;
        typename Solver::BlockPermArray const& permutations;
    };

    SparseCovarianceWorkspace(YBus<Symmetry> const& y_bus, std::vector<Gain> gain, double alpha,
                              ComplexValueVector<Symmetry> const& u, bool absolute_angle, Idx virtual_angle_bus = -1,
                              double virtual_angle_weight = 0.0, Factorization const* factorization = nullptr)
        : augmented_matrix{std::move(gain)},
          variance_normalization{alpha},
          slack_bus{y_bus.math_topology().slack_bus},
          report_relative_angle{!absolute_angle},
          row_indptr_{y_bus.row_indptr_lu()},
          col_indices_{y_bus.col_indices_lu()},
          lu_diag_{y_bus.lu_diag()},
          u_{u},
          virtual_angle_bus_{virtual_angle_bus},
          virtual_angle_weight_{virtual_angle_weight} {
        // NRSE inserts a virtual angle reference if and only if no voltage-angle sensor is present
        // (has_angle() is false; global-angle current sensors are not counted), which is also when
        // angles are reported relative to the slack bus.
        assert(uses_complex_state || report_relative_angle == (virtual_angle_bus_ >= 0));

        // 1-2. Reuse ILSE's LU factors, or validate, symmetrize, and factorize NRSE's returned-state gain.
        // Perturbed ILSE factors end in SparseMatrixError, at the latest in the selective inverse (step 5).
        Solver owned_solver{row_indptr_, col_indices_, lu_diag_};
        Solver& solver = factorization ? factorization->solver : owned_solver;
        typename Solver::BlockPermArray owned_perm;
        if (!factorization) {
            validate_assembly(y_bus.lu_transpose_entry());
            owned_perm.resize(u_.size());
            solver.prefactorize(augmented_matrix, owned_perm, false);
        }
        auto const& perm = factorization ? factorization->permutations : owned_perm;

        // 3. Verify that all equilibrated diagonal LU pivot blocks are well-conditioned.
        validate_pivots();

        // 4. Solve required inverse columns while `augmented_matrix` still holds the LU factors:
        //    first the NRSE virtual-angle columns T, then the slack phase-a column for relative angles.
        if constexpr (!uses_complex_state) {
            if (virtual_angle_bus_ >= 0) {
                solve_virtual_columns(solver, perm);
            }
        }
        if (report_relative_angle) {
            // When NRSE's virtual angle is on the slack bus, its phase-a column is already the slack column.
            if (!uses_complex_state && slack_bus == virtual_angle_bus_) {
                slack_column.resize(u_.size());
                for (std::size_t bus = 0; bus != u_.size(); ++bus) {
                    slack_column[bus] = virtual_columns_[bus].col(0);
                }
            } else {
                slack_column = solve_inverse_column(solver, perm, slack_bus);
            }
        }

        // 5. Overwrite `augmented_matrix` with the selective inverse blocks B_ij on the filled LU pattern.
        solver.inplace_selective_inverse_with_prefactorized_matrix(augmented_matrix, perm);
        for (auto const& block : augmented_matrix) {
            if (!block.isFinite().all()) {
                throw SparseMatrixError{};
            }
        }

        // 6. NRSE: the polar phase needs atan2 and sincos, so form each bus's rectangular map once;
        // for relative angles also one reference cross per bus, shared by all of its blocks.
        if constexpr (!uses_complex_state) {
            jacobians_.reserve(u_.size());
            for (Idx bus = 0; bus != static_cast<Idx>(u_.size()); ++bus) {
                jacobians_.push_back(rectangular_jacobian(bus));
            }
            if (report_relative_angle) {
                reference_crosses_.reserve(u_.size());
                for (Idx bus = 0; bus != static_cast<Idx>(u_.size()); ++bus) {
                    reference_crosses_.push_back(compute_reference_crosses(bus));
                }
            }
        }
    }

    // Compute the corrected state covariance block C_ij for bus pair (row, col), with C_ji = C_ij^H.
    // Checks Hermitian symmetry of the raw state corner Z_ij = (B_ij[q,q] + B_ji[q,q]^H) / 2,
    // and applies reference corrections and variance normalization alpha.
    StateCovarianceBlock calculate_state_covariance_block(Idx row, Idx col) const {
        if (row > col) {
            auto result = calculate_state_covariance_block(col, row);
            result.value = result.value.adjoint().eval();
            for (auto& term : result.terms) {
                term = term.adjoint().eval();
            }
            result.term_scale.transposeInPlace();
            result.has_nonzero_term.transposeInPlace();
            return result;
        }
        // The filled LU sparsity pattern is structurally symmetric, so both (row, col) and (col, row) exist.
        auto const& forward_block = inverse_block(row, col);
        auto const& reverse_block = inverse_block(col, row);
        // Read only the q corner without modifying the stored inverse blocks.
        // Accumulate the unnormalized orientation sum 2*Z_ij = B_ij[q,q] + B_ji[q,q]^H;
        // correct_covariance applies any reference corrections and scales by alpha * 0.5.
        StateMatrix source;
        FrobeniusNorm skew;
        FrobeniusNorm forward_norm;
        FrobeniusNorm reverse_norm;
        for (int r = 0; r != StateDim; ++r) {
            for (int c = 0; c != StateDim; ++c) {
                Scalar const forward = forward_block(r, c);
                Scalar const reverse = conjugate(reverse_block(c, r));
                // Relative NRSE sets the slack phase-a row and column to zero algebraically.
                if (!slack_angle_projected(row, col, r, c)) {
                    skew.add(forward - reverse);
                    forward_norm.add(forward);
                    reverse_norm.add(reverse);
                }
                source(r, c) = forward + reverse;
            }
        }
        // Positive alpha cancels from the relative skew ratio. Allow symmetry_rtol * scale on diagonal
        // blocks and 2 * symmetry_rtol * scale (i.e., symmetry_rtol times the sum of orientation norms) off-diagonal.
        double const skew_norm = skew.value();
        double const scale = average(forward_norm.value(), reverse_norm.value());
        if (!std::isfinite(scale) || skew_norm > symmetry_rtol * scale * (row == col ? 1.0 : 2.0)) {
            StateCovarianceBlock invalid;
            invalid.valid = false;
            return invalid;
        }
        auto result = correct_covariance(row, col, source);
        result.symmetry_projected = result.valid;
        return result;
    }

    // The block (row, col) in physical coordinates, J_row C J_col^H; see physical_from_state.
    PhysicalBlock physical_block(Idx row, Idx col) const {
        if (row > col) {
            auto result = physical_block(col, row);
            result.value = result.value.adjoint().eval();
            return result;
        }
        return physical_from_state(row, col, calculate_state_covariance_block(row, col));
    }
    // Map the state block of buses (row <= col) to physical coordinates. The result is valid only
    // when the state block is valid and maps to finite values.
    PhysicalBlock physical_from_state(Idx row, Idx col, StateCovarianceBlock const& state) const {
        PhysicalBlock result;
        result.valid = state.valid;
        result.symmetry_projected = state.symmetry_projected;
        if constexpr (uses_complex_state) {
            result.value = state.value;
            result.construction_scale = state.construction_scale;
        } else {
            map_physical(jacobians_[row], jacobians_[col], row == col, state, result);
            result.valid = result.valid && result.value.allFinite() && std::isfinite(result.construction_scale);
        }
        return result;
    }

    // NRSE: the map of bus's polar state [theta, |U|] to rectangular [Re U, Im U].
    PhysicalMatrix rectangular_jacobian(Idx bus) const
        requires(!uses_complex_state)
    {
        PhysicalMatrix result = PhysicalMatrix::Zero();
        for (int phase = 0; phase != phase_count; ++phase) {
            DoubleComplex const voltage = phase_voltage(bus, phase);
            DoubleComplex const direction = std::polar(1.0, std::arg(voltage));
            result(phase, phase) = -voltage.imag();
            result(phase, phase + phase_count) = direction.real();
            result(phase + phase_count, phase) = voltage.real();
            result(phase + phase_count, phase + phase_count) = direction.imag();
        }
        return result;
    }

    // Position of the stored block (row, col) on the LU pattern, if the pair is stored.
    std::optional<Idx> find_block_index(Idx row, Idx col) const {
        auto const first = col_indices_.begin() + row_indptr_[row];
        auto const last = col_indices_.begin() + row_indptr_[row + 1];
        auto const found = std::lower_bound(first, last, col);
        if (found == last || *found != col) {
            return std::nullopt;
        }
        return static_cast<Idx>(found - col_indices_.begin());
    }

    DoubleComplex phase_voltage(Idx bus, int phase) const {
        if constexpr (phase_count == 1) {
            return u_[bus];
        } else {
            return u_[bus](phase);
        }
    }

    // Accessible to the voltage uncertainty functions below and the dense-reference unit tests.
    std::vector<Gain> augmented_matrix; // LU factor storage, overwritten in-place by selective inverse blocks B_ij
    double variance_normalization;      // shared variance scale alpha
    Idx slack_bus;
    bool report_relative_angle;
    std::vector<StateColumn> slack_column; // per bus i: state rows of B's column for slack phase a

  private:
    // ----- Covariance correction -----

    Idx block_index(Idx row, Idx col) const {
        if (row == col) {
            return lu_diag_[row];
        }
        return detail::find_sparse_entry(col_indices_, col, row_indptr_[row], row_indptr_[row + 1]);
    }
    Gain const& inverse_block(Idx row, Idx col) const { return augmented_matrix[block_index(row, col)]; }

    // Apply variance normalization alpha and (for NRSE) virtual-angle and slack-reference corrections
    // to `source = 2 * Z_ij`:
    // - ILSE: V_ij = alpha * Z_ij.
    // - NRSE: C0_ij = alpha * (Z_ij - T_i * W_v * T_j^T) with W_v = w * I, and for relative angles
    //         Cr_ij = C0_ij - r_i * d^T - d * r_j^T + c * d * d^T.
    StateCovarianceBlock correct_covariance(Idx row, Idx col, StateMatrix const& source) const {
        StateCovarianceBlock result;
        CorrectionNorms norms;
        if constexpr (uses_complex_state) {
            double const alpha = variance_normalization;
            for (int r = 0; r != StateDim; ++r) {
                for (int c = 0; c != StateDim; ++c) {
                    DoubleComplex const term = alpha * source(r, c) * 0.5;
                    result.value(r, c) = term;
                    result.terms[correction_term::normalized_source](r, c) = term;
                    result.term_scale(r, c) = modulus(term);
                    norms.add(correction_term::normalized_source, term);
                }
            }
        } else {
            for (int r = 0; r != StateDim; ++r) {
                for (int c = 0; c != StateDim; ++c) {
                    if (row == col && r > c) {
                        // A diagonal block is symmetric by construction.
                        mirror_entry(result, norms, r, c);
                    } else if (!slack_angle_projected(row, col, r, c)) {
                        // The known projected row/column is algebraically exact and stays zero.
                        correct_entry(result, norms, row, col, r, c, source(r, c));
                    }
                }
            }
        }
        result.construction_scale = norms.sum();
        result.valid = result.value.allFinite() && std::isfinite(result.construction_scale);
        return result;
    }

    // Compute entry (r, c) of the corrected NRSE state covariance block for bus pair (row, col).
    // `source` is the doubled orientation sum 2 * Z_ij(r, c), so each unnormalized correction is
    // accumulated with a factor of 2 and scaled by alpha * 0.5 at the end.
    // For relative angles, d selects the angle coordinates [0, phase_count), r_i = alpha * rho_i
    // is the bus-to-slack-phase-a cross-covariance vector, and c = alpha * rho_s[0] is the slack
    // phase-a angle variance in C0.
    void correct_entry(StateCovarianceBlock& result, CorrectionNorms& norms, Idx row, Idx col, int r, int c,
                       double source) const {
        double const alpha = variance_normalization;
        RealSum sum;
        std::array<double, correction_term::count> doubled_terms{};
        sum.add(source);
        doubled_terms[correction_term::normalized_source] = source;
        if (report_relative_angle) {
            double const factor = -2.0 * virtual_angle_weight_;
            for (int phase = 0; phase != phase_count; ++phase) {
                sum.add_product(factor, virtual_columns_[row](r, phase), virtual_columns_[col](c, phase));
                doubled_terms[correction_term::virtual_noise] +=
                    factor * virtual_columns_[row](r, phase) * virtual_columns_[col](c, phase);
            }
            if (c < phase_count) {
                doubled_terms[correction_term::row_reference] = -2.0 * reference_cross(row, r);
                sum.add(doubled_terms[correction_term::row_reference]);
            }
            if (r < phase_count) {
                doubled_terms[correction_term::col_reference] = -2.0 * reference_cross(col, c);
                sum.add(doubled_terms[correction_term::col_reference]);
            }
            if (r < phase_count && c < phase_count) {
                doubled_terms[correction_term::slack_reference] = 2.0 * reference_cross(slack_bus, 0);
                sum.add(doubled_terms[correction_term::slack_reference]);
            }
        }
        result.value(r, c) = alpha * sum.value() * 0.5;
        double term_scale = 0.0;
        for (std::size_t k = 0; k != correction_term::count; ++k) {
            double const term = alpha * doubled_terms[k] * 0.5;
            result.terms[k](r, c) = term;
            term_scale += std::abs(term);
            norms.add(k, term);
        }
        result.term_scale(r, c) = term_scale;
        result.has_nonzero_term(r, c) = sum.has_nonzero_term();
    }

    // Mirror upper-triangle entry (c, r) to lower-triangle entry (r, c) on a symmetric NRSE diagonal block.
    static void mirror_entry(StateCovarianceBlock& result, CorrectionNorms& norms, int r, int c) {
        result.value(r, c) = result.value(c, r);
        result.term_scale(r, c) = result.term_scale(c, r);
        result.has_nonzero_term(r, c) = result.has_nonzero_term(c, r);
        for (std::size_t k = 0; k != correction_term::count; ++k) {
            result.terms[k](r, c) = result.terms[k](c, r);
            norms.add(k, result.terms[k](r, c));
        }
    }

    // True when (row, r) or (col, c) is the slack phase-a angle in relative NRSE (identically zero).
    bool slack_angle_projected(Idx row, Idx col, int r, int c) const {
        return !uses_complex_state && report_relative_angle &&
               ((row == slack_bus && r == 0) || (col == slack_bus && c == 0));
    }

    // Unnormalized bus-to-slack-phase-a cross covariance rho_i[k] = (C0_{i,s} e_0)[k] / alpha.
    double reference_cross(Idx bus, int coordinate) const { return reference_crosses_[bus][coordinate]; }

    // J_row (C J_col^T) in ordinary double; a diagonal block is symmetric by
    // construction. The construction scale sums the Frobenius norms of the
    // mapped correction terms.
    static void map_physical(PhysicalMatrix const& left, PhysicalMatrix const& right, bool diagonal,
                             StateCovarianceBlock const& state, PhysicalBlock& result) {
        for (int r = 0; r != StateDim; ++r) {
            for (int c = 0; c != StateDim; ++c) {
                if (diagonal && r > c) {
                    result.value(r, c) = result.value(c, r);
                    continue;
                }
                RealSum sum;
                for (int a = 0; a != StateDim; ++a) {
                    for (int b = 0; b != StateDim; ++b) {
                        sum.add_product(left(r, a), state.value(a, b), right(c, b));
                    }
                }
                result.value(r, c) = sum.value();
            }
        }
        CorrectionNorms norms;
        for (std::size_t k = 0; k != state.terms.size(); ++k) {
            if (state.terms[k].isZero(0)) {
                continue;
            }
            for (int r = 0; r != StateDim; ++r) {
                for (int c = 0; c != StateDim; ++c) {
                    RealSum sum;
                    for (int a = 0; a != StateDim; ++a)
                        for (int b = 0; b != StateDim; ++b)
                            sum.add_product(left(r, a), state.terms[k](a, b), right(c, b));
                    norms.add(k, sum.value());
                }
            }
        }
        result.construction_scale = norms.sum();
    }

    // ----- Workspace construction helpers (in constructor execution order) -----

    // Verify that the assembled NRSE matrix Abar is finite and Hermitian within relative Frobenius
    // tolerance 1e-12, then symmetrize any accepted assembly roundoff.
    void validate_assembly(IdxVector const& transpose) {
        FrobeniusNorm norm;
        FrobeniusNorm skew;
        for (Idx i = 0; i != static_cast<Idx>(augmented_matrix.size()); ++i) {
            for (int r = 0; r != RawMatrix::RowsAtCompileTime; ++r) {
                for (int c = 0; c != RawMatrix::ColsAtCompileTime; ++c) {
                    Scalar const entry = augmented_matrix[i](r, c);
                    norm.add(entry);
                    skew.add(entry - conjugate(augmented_matrix[transpose[i]](c, r)));
                }
            }
        }
        double const skew_norm = skew.value();
        if (!std::isfinite(norm.value()) || skew_norm > 1e-12 * norm.value()) {
            throw SparseMatrixError{};
        }
        if (skew_norm != 0.0) {
            canonicalize_assembly(transpose);
        }
    }

    // Symmetrize Abar_ij and Abar_ji^H in place using overflow-safe entrywise midpoints.
    void canonicalize_assembly(IdxVector const& transpose) {
        for (Idx index = 0; index != static_cast<Idx>(augmented_matrix.size()); ++index) {
            Idx const reverse_index = transpose[index];
            if (index > reverse_index) {
                continue;
            }
            for (int r = 0; r != RawMatrix::RowsAtCompileTime; ++r) {
                for (int c = 0; c != RawMatrix::ColsAtCompileTime; ++c) {
                    if (index == reverse_index && r > c) {
                        continue;
                    }
                    Scalar const forward = augmented_matrix[index](r, c);
                    Scalar const reverse = conjugate(augmented_matrix[reverse_index](c, r));
                    Scalar const value = average(forward, reverse);
                    augmented_matrix[index](r, c) = value;
                    augmented_matrix[reverse_index](c, r) = conjugate(value);
                }
            }
        }
    }

    // Reconstruct every diagonal pivot block P = L_ii * U_ii, equilibrate its rows and columns by
    // their maximum moduli, and require sigma_min(P_eq) / sigma_max(P_eq) >= 1e-12.
    void validate_pivots() const {
        for (auto const index : lu_diag_) {
            RawMatrix const packed = augmented_matrix[index].matrix();
            RawMatrix lower = RawMatrix::Identity(), upper = RawMatrix::Zero();
            lower.template triangularView<Eigen::StrictlyLower>() = packed;
            upper.template triangularView<Eigen::Upper>() = packed;
            RawMatrix pivot = lower * upper;
            // Equilibrate rows and then columns by their infinity norms.
            auto const equilibrate = [](auto&& line) { line /= line.cwiseAbs().maxCoeff(); };
            for (int i = 0; i != pivot.rows(); ++i) {
                equilibrate(pivot.row(i));
            }
            for (int i = 0; i != pivot.cols(); ++i) {
                equilibrate(pivot.col(i));
            }
            // An inf (inf/inf) or NaN entry, or a zero row/column (0/0), leaves a NaN; JacobiSVD must not see it.
            if (!pivot.allFinite()) {
                throw SparseMatrixError{};
            }
            if (pivot_condition_certified(pivot)) {
                continue;
            }
            // After equilibration, every column has unit infinity norm, so sigma_max >= 1.
            Eigen::JacobiSVD<RawMatrix> const svd{pivot};
            auto const& values = svd.singularValues();
            if (values(values.size() - 1) / values(0) < 1e-12) {
                throw SparseMatrixError{};
            }
        }
    }

    // Skips JacobiSVD when the computed inverse X certifies sigma_min / sigma_max >= 1e-12 with a 2x margin:
    //   r = ||I - P X||_F + 16u ||P||_F ||X||_F < 0.5  and  (1 - r) / (||P||_F ||X||_F) >= 2e-12.
    // Proof: ||P^-1||_2 <= ||X||_2 / (1 - r) and ||.||_2 <= ||.||_F. `false` only means "not certified".
    static bool pivot_condition_certified(RawMatrix const& pivot) {
        // The 16u residual bound applies to 2x2 and 4x4 symmetric blocks; larger three-phase blocks
        // (6x6 and 12x12) proceed directly to JacobiSVD.
        if constexpr (RawMatrix::RowsAtCompileTime > 4) {
            return false;
        } else {
            RawMatrix const inverse = pivot.inverse();
            double const norm_product = pivot.norm() * inverse.norm();
            double const residual_bound =
                (RawMatrix::Identity() - pivot * inverse).norm() + 16.0 * unit_roundoff * norm_product;
            return residual_bound < 0.5 && (1.0 - residual_bound) / norm_product >= 2e-12;
        }
    }

    // Solve the NRSE virtual-angle inverse columns T_i[:, p] = (Abar^-1 * e_(v, theta_p))_i[q]
    // for each phase p at `virtual_angle_bus_`.
    void solve_virtual_columns(Solver& solver, typename Solver::BlockPermArray const& perm) {
        virtual_columns_.resize(u_.size(), VirtualColumnBlock::Zero());
        for (int phase = 0; phase != phase_count; ++phase) {
            auto const column = solve_inverse_column(solver, perm, virtual_angle_bus_, phase);
            for (std::size_t bus = 0; bus != u_.size(); ++bus) {
                virtual_columns_[bus].col(phase) = column[bus];
            }
        }
    }

    // Solve Abar * t = e_(bus, coordinate) using the prefactorized LU factors and extract each bus's state rows q.
    std::vector<StateColumn> solve_inverse_column(Solver& solver, typename Solver::BlockPermArray const& perm, Idx bus,
                                                  int coordinate = 0) {
        std::vector<Unknown> rhs(u_.size()), solution(u_.size());
        rhs[bus](coordinate) = 1.0;
        solver.solve_with_prefactorized_matrix(augmented_matrix, perm, rhs, solution);
        std::vector<StateColumn> result(u_.size());
        for (Idx i = 0; i != static_cast<Idx>(u_.size()); ++i) {
            if (!solution[i].isFinite().all()) {
                throw SparseMatrixError{};
            }
            result[i] = solution[i].matrix().template topRows<StateDim>();
        }
        return result;
    }

    // Compute bus `bus`'s unnormalized cross covariance with slack phase a in C0:
    //   rho_i = (C0_{i,s} * e_0) / alpha = slack_column[i] - w * sum_p T_i[:, p] * T_s[0, p].
    std::array<double, StateDim> compute_reference_crosses(Idx bus) const {
        std::array<double, StateDim> result{};
        for (int i = 0; i != StateDim; ++i) {
            RealSum cross;
            cross.add(slack_column[bus](i));
            for (int phase = 0; phase != phase_count; ++phase) {
                cross.add_product(-virtual_angle_weight_, virtual_columns_[slack_bus](0, phase),
                                  virtual_columns_[bus](i, phase));
            }
            result[i] = cross.value();
        }
        return result;
    }

    IdxVector row_indptr_, col_indices_, lu_diag_;
    ComplexValueVector<Symmetry> u_;
    Idx virtual_angle_bus_;
    double virtual_angle_weight_;                     // normalized virtual-angle weight w = alpha / var(U_v)
    std::vector<VirtualColumnBlock> virtual_columns_; // NRSE: per-bus virtual-angle columns T_i
    std::vector<PhysicalMatrix> jacobians_;           // NRSE: per-bus rectangular_jacobian(i)
    std::vector<std::array<double, StateDim>> reference_crosses_; // relative NRSE: per-bus cross vector rho_i
};

// Per-call cache of stored canonical (row <= col) physical blocks by LU position,
// shared by the branch and injection helpers of one calculation so each block is
// derived once. The workspace must not change while the cache is in use.
template <class Workspace> class PhysicalBlocks {
  public:
    using Block = typename Workspace::PhysicalBlock;

    explicit PhysicalBlocks(Workspace const& workspace)
        : workspace_{workspace}, blocks_(workspace.augmented_matrix.size()) {}

    Workspace const& workspace() const { return workspace_; }
    // index must be the LU position of (row, col), row <= col.
    Block const& stored(Idx index, Idx row, Idx col) {
        auto& block = blocks_[index];
        if (!block) {
            block = workspace_.physical_block(row, col);
        }
        return *block;
    }

  private:
    Workspace const& workspace_;
    std::vector<std::optional<Block>> blocks_;
};

// ----- Voltage uncertainty from corrected state covariance -----

// Cached slack phase-a data for relative ILSE angle variances: 1 / U_s and
// its angle-variance contribution C_{theta,ss} = Re(V_ss) * |1 / U_s|^2 / 2.
struct SlackAngle {
    DoubleComplex inverse_voltage{};
    RealSum variance_term;
    bool defined = false;
};
template <class Workspace> SlackAngle slack_angle(Workspace const& workspace) {
    auto const slack_covariance = workspace.calculate_state_covariance_block(workspace.slack_bus, workspace.slack_bus);
    DoubleComplex const voltage = workspace.phase_voltage(workspace.slack_bus, 0);
    SlackAngle result;
    bool const nonzero = is_finite(voltage) && voltage != DoubleComplex{};
    // C_{theta,ss} is a single-term variance: reject a negative one here, since a relative sum could mask it.
    result.defined = slack_covariance.valid && nonzero && slack_covariance.value(0, 0).real() >= 0.0;
    if (nonzero) {
        result.inverse_voltage = reciprocal(voltage);
        result.variance_term = half_square(slack_covariance.value(0, 0).real(), result.inverse_voltage);
    }
    return result;
}

// Compute ILSE voltage-angle variance under the proper complex Gaussian model:
//   C_{theta,ij} = Re(V_ij / (U_i * conj(U_j))) / 2.
// Absolute mode returns C_{theta,ii}; slack-relative mode returns C_{theta,ii} + C_{theta,ss} - 2 * C_{theta,is}
// with s = slack phase a (identically zero at slack phase a itself).
template <class Workspace>
UQPhaseValue ilse_angle_variance(Workspace const& workspace, std::optional<SlackAngle> const& slack, Idx bus, int phase,
                                 double variance_ii, DoubleComplex inverse_voltage, bool defined) {
    RealSum sum;
    if (!slack) {
        sum.add(half_square(variance_ii, inverse_voltage)); // C_{theta,ii}
        return finalize_variance(sum, defined);
    }
    if (bus != workspace.slack_bus || phase != 0) {
        sum.add(half_square(variance_ii, inverse_voltage)); // C_{theta,ii}
        sum.add(slack->variance_term);                      // C_{theta,ss}
        // -2 * C_{theta,is} = -Re(V_is * (1 / U_i) * conj(1 / U_s)) with V_is = alpha * slack_column[i].
        sum.add(real_part_of_product(-workspace.variance_normalization, workspace.slack_column[bus](phase),
                                     inverse_voltage, std::conj(slack->inverse_voltage)));
    }
    // As for C_{theta,ss}, a negative single-term C_{theta,ii} must not be masked by the relative sum.
    return finalize_variance(sum, defined && slack->defined && variance_ii >= 0.0);
}

// Extract an NRSE polar coordinate variance C_ii(coordinate, coordinate) with its roundoff scale.
template <class Block> UQPhaseValue corrected_diagonal_variance(Block const& block, int coordinate, bool defined) {
    double const scale = block.term_scale(coordinate, coordinate);
    return finalize_variance(block.value(coordinate, coordinate), scale,
                             defined && !underflowed(block.has_nonzero_term(coordinate, coordinate), scale));
}

// Evaluate all requested bus voltage magnitude and angle uncertainties from `workspace`.
// For a nonzero phasor U, first-order polar perturbations are d|U| = Re(conj(U) * dU) / |U|
// and dtheta = Im(dU / U).
template <class Workspace>
StateEstimationUQResult<typename Workspace::Symmetry>
calculate_voltage_uq_results(Workspace const& workspace, StateEstimationUQRequest const& request) {
    StateEstimationUQResult<typename Workspace::Symmetry> result;
    result.voltage.resize(request.voltage.size());
    constexpr bool uses_complex_state = Workspace::uses_complex_state;
    constexpr int phase_count = Workspace::phase_count;
    std::optional<SlackAngle> slack;
    if constexpr (uses_complex_state) {
        if (workspace.report_relative_angle && std::ranges::any_of(request.voltage, &VoltageUQRequest::angle)) {
            slack = slack_angle(workspace);
        }
    }
    for (std::size_t index = 0; index != request.voltage.size(); ++index) {
        auto const& voltage_request = request.voltage[index];
        auto& output = result.voltage[index];
        auto const state_covariance =
            workspace.calculate_state_covariance_block(voltage_request.bus, voltage_request.bus);
        for (int phase = 0; phase != phase_count; ++phase) {
            DoubleComplex const voltage = workspace.phase_voltage(voltage_request.bus, phase);
            bool const nonzero = is_finite(voltage) && voltage != DoubleComplex{};
            bool const defined = state_covariance.valid && nonzero;
            if constexpr (uses_complex_state) {
                // Proper-complex ILSE: Var(|U_i|) = Re(V_ii) / 2 (diagonal entries of V_ii are real by construction).
                DoubleComplex const variance_ii = state_covariance.value(phase, phase);
                if (voltage_request.magnitude) {
                    output.magnitude.phase[phase] = variance({variance_ii.real() / 2.0}, defined);
                }
                if (voltage_request.angle) {
                    DoubleComplex const inverse_voltage = nonzero ? reciprocal(voltage) : DoubleComplex{};
                    output.angle.phase[phase] = ilse_angle_variance(workspace, slack, voltage_request.bus, phase,
                                                                    variance_ii.real(), inverse_voltage, defined);
                }
            } else {
                // NRSE state ordering per bus: angles [0, phase_count), magnitudes [phase_count, 2 * phase_count).
                if (voltage_request.magnitude) {
                    output.magnitude.phase[phase] =
                        corrected_diagonal_variance(state_covariance, phase + phase_count, defined);
                }
                if (voltage_request.angle) {
                    output.angle.phase[phase] = corrected_diagonal_variance(state_covariance, phase, defined);
                }
            }
        }
    }
    return result;
}

} // namespace power_grid_model::math_solver::state_estimation_uq
