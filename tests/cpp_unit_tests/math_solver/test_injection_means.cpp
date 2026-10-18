// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0
#include <doctest/doctest.h>
#include <power_grid_model/math_solver/injection_means.hpp>

using power_grid_model::injection_experiment::CovarianceMeans;

TEST_CASE("Original-pattern scalar and block covariance means") {
    using ScalarMeans = CovarianceMeans<std::complex<double>, 1>;
    using RealMeans = CovarianceMeans<double, 2>;
    std::vector<int> rows{0, 2, 5, 7}, cols{0, 1, 0, 1, 2, 1, 2};
    std::vector<std::array<int, 2>> edges{{0, 1}, {1, 2}, {0, 1}};
    Eigen::Matrix3cd v = Eigen::Matrix3cd::Identity();
    v(0, 1) = v(1, 0) = 0.8;
    v(1, 2) = v(2, 1) = 0.6;
    auto scalar = [&](int j, int m) -> ScalarMeans::Block { return ScalarMeans::Block::Constant(v(j, m)); };
    ScalarMeans means{3, rows, cols, edges, scalar};
    CHECK(std::abs(means.estimate(0, 2, 4).value(0, 0) - 0.7) < 1e-14);
    CHECK(std::abs(means.estimate(0, 2, 5).value(0, 0) - 0.7) < 1e-14);
    CHECK(std::abs(means.estimate(0, 2, 6).value(0, 0) - 0.48) < 1e-14);
    CHECK_THROWS_AS(means.estimate(0, 1, 6), std::runtime_error);
    v(0, 1) = std::complex<double>{0.0, 0.8};
    v(1, 0) = std::conj(v(0, 1));
    v(1, 2) = std::complex<double>{0.3, 0.4};
    v(2, 1) = std::conj(v(1, 2));
    ScalarMeans complex_means{3, rows, cols, edges, scalar};
    auto real = [&](int j, int m) -> Eigen::Matrix2d {
        Eigen::Matrix2d result;
        result << v(j, m).real(), -v(j, m).imag(), v(j, m).imag(), v(j, m).real();
        return 0.5 * result;
    };
    RealMeans block_means{3, rows, cols, edges, real};
    for (int policy : {4, 5, 6}) {
        auto const value = complex_means.estimate(0, 2, policy).value(0, 0);
        Eigen::Matrix2d expected;
        expected << value.real(), -value.imag(), value.imag(), value.real();
        CHECK((block_means.estimate(0, 2, policy).value - 0.5 * expected).norm() < 1e-14);
    }
    CHECK(std::abs(complex_means.estimate(0, 2, 6).value(0, 0) - v(0, 1) * v(1, 2)) < 1e-14);
    CHECK(std::abs(complex_means.estimate(0, 2, 5).value(0, 0) - 0.15) < 1e-14);
}

TEST_CASE("Covariance means singular support and invalid donors") {
    using Means = CovarianceMeans<double, 2>;
    std::vector<int> rows{0, 2, 5, 7}, cols{0, 1, 0, 1, 2, 1, 2};
    std::vector<std::array<int, 2>> edges{{0, 1}, {1, 2}};
    double cross = 0.5, variance = 1.0;
    bool invalid_support = false;
    auto block = [&](int j, int m) -> Eigen::Matrix2d {
        Eigen::Matrix2d r = Eigen::Matrix2d::Zero();
        r(0, 0) = j == m ? variance : cross;
        if (invalid_support && j != m) {
            r(1, 0) = 0.1;
        }
        return r;
    };
    Means means{3, rows, cols, edges, block};
    CHECK(means.estimate(0, 2, 6).value(0, 0) == doctest::Approx(0.25));
    CHECK(means.estimate(0, 2, 6).value(1, 1) == 0.0);
    cross = 1.1;
    CHECK_THROWS_AS((Means{3, rows, cols, edges, block}), std::runtime_error);
    cross = 0.0;
    variance = 0.0;
    Means zero{3, rows, cols, edges, block};
    CHECK(zero.estimate(0, 2, 6).value.norm() == 0.0);
    variance = -1e-15;
    Means roundoff{3, rows, cols, edges, block};
    CHECK(roundoff.estimate(0, 2, 6).value.norm() == 0.0);
    variance = -1;
    CHECK_THROWS_AS((Means{3, rows, cols, edges, block}), std::runtime_error);
    variance = 1;
    invalid_support = true;
    CHECK_THROWS_AS((Means{3, rows, cols, edges, block}), std::runtime_error);
}
