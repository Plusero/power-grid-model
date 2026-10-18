// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "injection_experiment.hpp"
#include <complex>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

namespace power_grid_model::injection_experiment {
inline constexpr double correlation_zero_tolerance = 1e-12;
inline constexpr double rank_rtol = 1000.0 * std::numeric_limits<double>::epsilon();
inline constexpr double correlation_validation_tolerance = 1e-6;

// One instance per energized math island and returned state. Only the original
// selected inverse is visited; cached replacements can never become donors.
template <typename Scalar, int Dim> class CovarianceMeans {
  public:
    using Block = Eigen::Matrix<Scalar, Dim, Dim>;
    using Index = Eigen::Index;
    using Pair = std::pair<Index, Index>;
    struct Estimate {
        Block value = Block::Zero();
        nlohmann::json detail;
    };

    template <typename Rows, typename Cols, typename Branches, typename Covariance>
    CovarianceMeans(Index n, Rows const& rows, Cols const& cols, Branches const& branches, Covariance covariance)
        : roots_(n), whiteners_(n), ranks_(n), adjacency_(n), local_sum_(n, Block::Zero()), local_count_(n) {
        for (Index j = 0; j < n; ++j) {
            Block const marginal = covariance(j, j);
            if (!marginal.allFinite()) {
                throw std::runtime_error{"Nonfinite averaging marginal"};
            }
            Eigen::SelfAdjointEigenSolver<Block> eigen{(marginal + marginal.adjoint()).eval() * 0.5};
            double const tolerance = rank_rtol * std::max(1.0, marginal.cwiseAbs().maxCoeff());
            if (eigen.info() != Eigen::Success || eigen.eigenvalues().minCoeff() < -tolerance ||
                (marginal - marginal.adjoint()).norm() > tolerance) {
                throw std::runtime_error{"Invalid averaging marginal"};
            }
            auto const values = eigen.eigenvalues().cwiseMax(0.0).eval();
            auto inverse_roots = values.eval();
            for (int d = 0; d < Dim; ++d) {
                bool const supported = values(d) > rank_rtol * values.maxCoeff();
                inverse_roots(d) = supported ? 1.0 / std::sqrt(values(d)) : 0.0;
                ranks_[j] += static_cast<int>(supported);
            }
            roots_[j] = eigen.eigenvectors() * values.cwiseSqrt().asDiagonal() * eigen.eigenvectors().adjoint();
            whiteners_[j] = eigen.eigenvectors() * inverse_roots.asDiagonal() * eigen.eigenvectors().adjoint();
            if (enabled()) {
                marginal_details_.push_back({{"bus", j},
                                             {"rank", ranks_[j]},
                                             {"clipped", (eigen.eigenvalues().array() < 0.0).count()},
                                             {"eigenvalue_tolerance", tolerance}});
            }
        }
        for (auto const& branch : branches) {
            Index j = branch[0], m = branch[1];
            if (j < 0 || m < 0 || j == m) {
                continue;
            }
            adjacency_[j].insert(m);
            adjacency_[m].insert(j);
        }
        for (Index j = 0; j < n; ++j) {
            for (auto idx = rows[j]; idx < rows[j + 1]; ++idx) {
                Index const m = cols[idx];
                if (j >= m) {
                    continue;
                }
                Block const raw = covariance(j, m);
                if (!raw.allFinite()) {
                    throw std::runtime_error{"Nonfinite averaging donor"};
                }
                Block const normalized = whiteners_[j] * raw * whiteners_[m].adjoint();
                Block const restored = roots_[j] * normalized * roots_[m].adjoint();
                double const tolerance = rank_rtol * std::max({1.0, raw.norm(), roots_[j].norm() * roots_[m].norm()});
                if ((restored - raw).norm() > tolerance) {
                    throw std::runtime_error{"Averaging donor outside marginal support"};
                }
                if (ranks_[j] == 0 || ranks_[m] == 0) {
                    continue;
                }
                double const norm = Eigen::JacobiSVD<Block>{normalized}.singularValues().maxCoeff();
                if (norm > 1.0 + correlation_validation_tolerance) {
                    throw std::runtime_error{"Averaging donor correlation exceeds one"};
                }
                donors_.emplace(Pair{j, m}, raw);
                Block const oriented = (normalized + normalized.adjoint()) * 0.5;
                global_ += oriented;
                if (adjacency_[j].contains(m) && norm > correlation_zero_tolerance) {
                    local_donors_.emplace(Pair{j, m}, oriented);
                    local_sum_[j] += oriented;
                    local_sum_[m] += oriented;
                    ++local_count_[j];
                    ++local_count_[m];
                }
            }
        }
        if (!donors_.empty()) {
            global_ /= static_cast<double>(donors_.size());
        }
    }

    Estimate const& estimate(Index j, Index m, int policy) {
        if (j >= m || policy < 4 || policy > 6) {
            throw std::runtime_error{"Expected canonical missing pair and averaging policy"};
        }
        Pair const pair{j, m};
        if (auto it = cache_.find({j, m, policy}); it != cache_.end()) {
            return it->second;
        }
        if (donors_.contains(pair)) {
            throw std::runtime_error{"Cannot replace an available donor"};
        }
        Estimate result;
        std::string source = "zero_target";
        std::vector<std::string> fallback;
        std::vector<Index> intermediates;
        auto paths = nlohmann::json::array();
        auto local_ids = nlohmann::json::array();
        auto count = local_count_[j] + local_count_[m];
        Block sum = local_sum_[j] + local_sum_[m];
        if (auto it = local_donors_.find(pair); it != local_donors_.end()) {
            --count;
            sum -= it->second;
        }
        if (ranks_[j] != 0 && ranks_[m] != 0) {
            if (policy == 6) {
                for (Index k : adjacency_[j]) {
                    if (k == j || k == m || !adjacency_[m].contains(k) || ranks_[k] == 0 ||
                        !donors_.contains(std::minmax(j, k)) || !donors_.contains(std::minmax(k, m))) {
                        continue;
                    }
                    Block const prediction = donor(j, k) * whiteners_[k] * whiteners_[k] * donor(k, m);
                    result.value += prediction;
                    intermediates.push_back(k);
                    if (enabled()) {
                        paths.push_back({{"k", k}, {"estimate", encode(prediction)}});
                    }
                }
                if (!intermediates.empty()) {
                    result.value /= static_cast<double>(intermediates.size());
                    source = "path";
                } else {
                    fallback.push_back("path_to_local");
                }
            }
            if (source != "path") {
                Block coefficient = global_;
                source = "global";
                if (policy >= 5 && count > 0) {
                    coefficient = sum / static_cast<double>(count);
                    source = "local";
                } else {
                    if (policy >= 5) {
                        fallback.push_back("local_to_global");
                    }
                    if (donors_.empty()) {
                        fallback.push_back("global_to_zero");
                        source = "zero";
                    }
                }
                result.value = roots_[j] * coefficient * roots_[m].adjoint();
            }
        }
        if (enabled()) {
            for (Index endpoint : {j, m}) {
                for (Index k : adjacency_[endpoint]) {
                    Pair const edge = std::minmax(endpoint, k);
                    if (local_donors_.contains(edge)) {
                        local_ids.push_back({edge.first, edge.second});
                    }
                }
            }
            result.detail = {{"j", j},
                             {"m", m},
                             {"source", source},
                             {"fallback", fallback},
                             {"global_count", donors_.size()},
                             {"local_count", count},
                             {"path_count", intermediates.size()},
                             {"intermediates", intermediates},
                             {"paths", paths},
                             {"local_donors", local_ids},
                             {"replacement", encode(result.value)}};
        }
        return cache_.emplace(std::tuple{j, m, policy}, std::move(result)).first->second;
    }

    nlohmann::json diagnostics() const {
        auto edges = nlohmann::json::array(), donors = nlohmann::json::array(), replacements = nlohmann::json::array();
        for (Index j = 0; j < static_cast<Index>(adjacency_.size()); ++j) {
            for (Index m : adjacency_[j]) {
                if (j < m) {
                    edges.push_back({j, m});
                }
            }
        }
        for (auto const& [pair, value] : donors_) {
            donors.push_back({pair.first, pair.second});
        }
        for (auto const& [pair, result] : cache_) {
            replacements.push_back(result.detail);
        }
        return {{"physical_edges", edges},
                {"global_donors", donors},
                {"global_count", donors_.size()},
                {"global_coefficient", encode(global_)},
                {"marginals", marginal_details_},
                {"pairs", replacements},
                {"correlation_zero_tolerance", correlation_zero_tolerance},
                {"rank_rtol", rank_rtol},
                {"correlation_validation_tolerance", correlation_validation_tolerance}};
    }

  private:
    static nlohmann::json encode(Block const& value) {
        if constexpr (Dim == 1) {
            return {std::real(value(0, 0)), std::imag(value(0, 0))};
        } else {
            return real_matrix(value);
        }
    }
    Block donor(Index j, Index m) const {
        Block value = donors_.at(std::minmax(j, m));
        if (j < m) {
            return value;
        }
        return value.adjoint();
    }
    std::vector<Block> roots_, whiteners_;
    std::vector<int> ranks_;
    std::vector<std::set<Index>> adjacency_;
    std::vector<Block> local_sum_;
    std::vector<Index> local_count_;
    std::map<Pair, Block> donors_, local_donors_;
    std::map<std::tuple<Index, Index, int>, Estimate> cache_;
    Block global_ = Block::Zero();
    nlohmann::json marginal_details_ = nlohmann::json::array();
};
} // namespace power_grid_model::injection_experiment
