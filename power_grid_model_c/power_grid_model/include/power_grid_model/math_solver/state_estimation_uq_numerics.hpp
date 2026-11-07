// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// Scalar arithmetic helpers, accumulators, and variance validity decisions for state-estimation UQ.

#include "state_estimation_uq.hpp"

#include "../common/common.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <type_traits>

namespace power_grid_model::math_solver::state_estimation_uq {

// ----- Tolerances -----

// Unit roundoff u = eps / 2, used in the fast pivot-conditioning certificate.
inline constexpr double unit_roundoff = std::numeric_limits<double>::epsilon() / 2;
// Relative tolerance for clipping roundoff-negative variances and checking marginal eigenvalues.
inline constexpr double round_rtol = 1000.0 * std::numeric_limits<double>::epsilon();
// Relative Frobenius tolerance for the Hermitian symmetry check on raw state covariance blocks.
inline double const symmetry_rtol = std::sqrt(std::numeric_limits<double>::epsilon());

// ----- Scalar helpers -----

inline bool is_finite(double value) { return std::isfinite(value); }
inline bool is_finite(DoubleComplex value) { return std::isfinite(value.real()) && std::isfinite(value.imag()); }

// Complex modulus |value| via plain sum of squares (exact when imag == 0).
inline double modulus(DoubleComplex value) {
    if (value.imag() == 0.0) {
        return std::abs(value.real());
    }
    return std::sqrt((value.real() * value.real()) + (value.imag() * value.imag()));
}

template <class T> T conjugate(T x) {
    if constexpr (std::is_same_v<T, DoubleComplex>) {
        return std::conj(x);
    } else {
        return x;
    }
}

// Overflow-safe midpoint (a + b) / 2.
inline double average(double first, double second) { return std::midpoint(first, second); }
inline DoubleComplex average(DoubleComplex first, DoubleComplex second) {
    return {std::midpoint(first.real(), second.real()), std::midpoint(first.imag(), second.imag())};
}

// Complex reciprocal 1 / u = conj(u) / |u|^2 in binary64 (u must be finite and nonzero).
inline DoubleComplex reciprocal(DoubleComplex u) {
    double const squared = (u.real() * u.real()) + (u.imag() * u.imag());
    return {u.real() / squared, -u.imag() / squared};
}

// ----- Accumulators -----

// True when a sum contained at least one term that was nonzero in exact arithmetic, but the sum of
// term magnitudes fell below the smallest normal double (total subnormal underflow). Partial
// underflow alongside a normal-magnitude term is ordinary rounding and is not flagged.
inline bool underflowed(bool has_nonzero_term, double scale) {
    return has_nonzero_term && !(scale >= std::numeric_limits<double>::min());
}

// Accumulates a binary64 sum together with its absolute scale (sum_k |t_k|) and whether any
// term was nonzero in exact arithmetic (a product counts when all of its factors are nonzero),
// allowing total underflow to be distinguished from an exact zero.
class RealSum {
  public:
    void add(double term) {
        value_ += term;
        absolute_scale_ += std::abs(term);
        has_nonzero_term_ = has_nonzero_term_ || term != 0.0;
    }
    void add_product(double a, double b) {
        add(a * b);
        has_nonzero_term_ = has_nonzero_term_ || (a != 0.0 && b != 0.0);
    }
    void add_product(double a, double b, double c) {
        add(a * b * c);
        has_nonzero_term_ = has_nonzero_term_ || (a != 0.0 && b != 0.0 && c != 0.0);
    }
    void add_product(double a, double b, double c, double d) {
        add(a * b * c * d);
        has_nonzero_term_ = has_nonzero_term_ || (a != 0.0 && b != 0.0 && c != 0.0 && d != 0.0);
    }
    // Append a pre-accumulated inner sum as a single term; propagate its nonzero flag only if
    // the inner sum itself suffered total underflow (as opposed to normal cancellation).
    void add(RealSum const& term) {
        add(term.value());
        has_nonzero_term_ = has_nonzero_term_ || underflowed(term.has_nonzero_term_, term.absolute_scale_);
    }
    double value() const { return value_; }
    double absolute_scale() const { return absolute_scale_; }
    bool has_nonzero_term() const { return has_nonzero_term_; }

  private:
    double value_ = 0.0;
    double absolute_scale_ = 0.0;
    bool has_nonzero_term_ = false;
};

// Ordinary complex accumulation of componentwise products.
class ComplexSum {
  public:
    void add(DoubleComplex term) {
        real_.add(term.real());
        imag_.add(term.imag());
    }
    void add_product(DoubleComplex a, DoubleComplex b) {
        real_.add_product(a.real(), b.real());
        real_.add_product(-a.imag(), b.imag());
        imag_.add_product(a.real(), b.imag());
        imag_.add_product(a.imag(), b.real());
    }
    DoubleComplex value() const { return {real_.value(), imag_.value()}; }

  private:
    RealSum real_;
    RealSum imag_;
};
template <class Scalar>
using Accumulator = std::conditional_t<Eigen::NumTraits<Scalar>::IsComplex, ComplexSum, RealSum>;

// Accumulate weight * |x|^2 / 2 as real products.
inline RealSum half_square(double weight, DoubleComplex x) {
    RealSum result;
    result.add_product(0.5, weight, x.real(), x.real());
    result.add_product(0.5, weight, x.imag(), x.imag());
    return result;
}
// Accumulate factor * Re(x * y * z) as four real products.
inline RealSum real_part_of_product(double factor, DoubleComplex x, DoubleComplex y, DoubleComplex z) {
    RealSum result;
    result.add_product(factor, x.real(), y.real(), z.real());
    result.add_product(-factor, x.real(), y.imag(), z.imag());
    result.add_product(-factor, x.imag(), y.real(), z.imag());
    result.add_product(-factor, x.imag(), y.imag(), z.real());
    return result;
}
// factor * Im(x * y * z) as a sum of real products.
inline RealSum imag_part_of_product(double factor, DoubleComplex x, DoubleComplex y, DoubleComplex z) {
    RealSum result;
    result.add_product(factor, x.real(), y.real(), z.imag());
    result.add_product(factor, x.real(), y.imag(), z.real());
    result.add_product(factor, x.imag(), y.real(), z.real());
    result.add_product(-factor, x.imag(), y.imag(), z.imag());
    return result;
}

// Rectangular derivatives of S = U_t conj(I), I = sum_k c_k U_k, with respect to
// (Re U_k, Im U_k), as the complex pair (real axis, imaginary axis): w + conj(I)
// and -j w + j conj(I) with w = U_t conj(c_k), the conj(I) terms only at k = t.
// At the terminal the self term is regrouped exactly: U_t conj(c_t) +
// conj(c_t U_t) = 2 Re(U_t) conj(c_t) and -j U_t conj(c_t) + j conj(c_t U_t) =
// 2 Im(U_t) conj(c_t). So the two never cancel in rounding, as they would at a
// branch open at one end, where they differ by the small charging current.
// others is sum_{k != t} conj(c_k U_k).
inline std::array<DoubleComplex, 2> power_axes(DoubleComplex voltage, DoubleComplex c_k, bool terminal,
                                               DoubleComplex others) {
    DoubleComplex const conj_c = std::conj(c_k);
    if (terminal) {
        auto axis = [&](double component, DoubleComplex offset) {
            double const doubled = 2.0 * component;
            return DoubleComplex{doubled * conj_c.real(), doubled * conj_c.imag()} + offset;
        };
        return {axis(voltage.real(), others), axis(voltage.imag(), {-others.imag(), others.real()})};
    }
    ComplexSum product;
    product.add_product(voltage, conj_c);
    DoubleComplex const w = product.value();
    return {w, DoubleComplex{w.imag(), -w.real()}};
}

// Frobenius norm ||X||_F = sqrt(sum_ij |X_ij|^2) accumulated as a plain sum of squares.
class FrobeniusNorm {
  public:
    void add(double x) { squares_ += x * x; }
    void add(DoubleComplex x) {
        add(x.real());
        add(x.imag());
    }
    double value() const { return std::sqrt(squares_); }

  private:
    double squares_ = 0.0;
};

template <class Derived> double frobenius_norm(Eigen::MatrixBase<Derived> const& matrix) {
    FrobeniusNorm result;
    for (Eigen::Index r = 0; r != matrix.rows(); ++r) {
        for (Eigen::Index c = 0; c != matrix.cols(); ++c) {
            result.add(matrix(r, c));
        }
    }
    return result.value();
}

// ----- Variance decision -----

// Classify a scalar variance against its roundoff tolerance = round_rtol * absolute_scale:
// - invalid:          !defined, nonfinite variance/scale, or variance < -tolerance (sigma = NaN);
// - clipped_roundoff: -tolerance <= variance < 0 (sigma = 0);
// - valid:            variance >= 0 (sigma = sqrt(variance)).
// raw_variance and absolute_scale are always preserved in the returned UQPhaseValue.
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

// Finalize a RealSum variance accumulator, rejecting total subnormal underflow of nonzero terms.
inline UQPhaseValue finalize_variance(RealSum const& sum, bool defined) {
    bool const lost = underflowed(sum.has_nonzero_term(), sum.absolute_scale());
    return finalize_variance(sum.value(), sum.absolute_scale(), defined && !lost);
}

// Sum a list of variance contributions and finalize the result.
inline UQPhaseValue variance(std::initializer_list<double> terms, bool defined = true) {
    RealSum sum;
    for (auto const term : terms) {
        sum.add(term);
    }
    return finalize_variance(sum, defined);
}

} // namespace power_grid_model::math_solver::state_estimation_uq
