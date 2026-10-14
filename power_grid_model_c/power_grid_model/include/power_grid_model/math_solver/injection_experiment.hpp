// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#ifndef PGM_INJECTION_SCHEME
#define PGM_INJECTION_SCHEME 0
#endif
namespace power_grid_model::injection_experiment {
inline constexpr int scheme = PGM_INJECTION_SCHEME;
static_assert(scheme >= 0 && scheme < 4);
inline constexpr std::array names{"full", "zero", "positive", "bound"};
inline bool enabled() { return std::getenv("PGM_INJECTION_DIAGNOSTICS") != nullptr; }
// The symmetric principal root fixes the standardized rectangular axes and also supports singular marginals.
struct PrincipalRoot {
    Eigen::Matrix2d value;
    Eigen::Vector2d eigenvalues;
    double tolerance;
    int clipped;
};
inline PrincipalRoot principal_root(Eigen::Matrix2d const& covariance) {
    if (!covariance.allFinite()) {
        throw std::runtime_error{"Nonfinite NRSE marginal covariance"};
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen{0.5 * (covariance + covariance.transpose())};
    double const tolerance =
        1000.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, covariance.cwiseAbs().maxCoeff());
    if (eigen.info() != Eigen::Success || eigen.eigenvalues().minCoeff() < -tolerance) {
        throw std::runtime_error{"Materially indefinite NRSE marginal covariance"};
    }
    return {eigen.eigenvectors() * eigen.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal() *
                eigen.eigenvectors().transpose(),
            eigen.eigenvalues(), tolerance, static_cast<int>((eigen.eigenvalues().array() < 0.0).count())};
}
template <typename Derived> nlohmann::json real_matrix(Eigen::MatrixBase<Derived> const& matrix) {
    auto result = nlohmann::json::array();
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
            result.push_back(matrix(row, col));
        }
    }
    return result;
}
inline void write(nlohmann::json const& record) {
    auto const* path = std::getenv("PGM_INJECTION_DIAGNOSTICS");
    if (path == nullptr) {
        return;
    }
    std::ofstream stream{path, std::ios::app};
    if (!(stream << record.dump() << '\n')) {
        throw std::runtime_error{"Cannot write injection diagnostics"};
    }
}
} // namespace power_grid_model::injection_experiment
