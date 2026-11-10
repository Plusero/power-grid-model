// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// Covariance of bus pairs that the selective inverse does not provide.

#include "state_estimation_uq.hpp"
#include "state_estimation_uq_workspace.hpp"

#include "../common/common.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace power_grid_model::math_solver::state_estimation_uq {
namespace completion_detail {

// A middle block of the augmented path is skipped below this ratio of its smallest to largest singular value.
inline constexpr double middle_condition_limit = 1e-12;
// Diagnostics warn above this relative residual of a middle solve, and above this normalized correlation.
inline constexpr double solve_residual_limit = 1e-12;
inline constexpr double correlation_limit = 1.0 + 1e-6;

// Insert-only open-addressing table of completed pairs by nonnegative key, with linear probing at most half
// full. A std::unordered_map node allocation per missing pair was a visible share of large injection
// requests.
template <class Value> class PairTable {
  public:
    Value const* find(Idx key) const {
        if (slots_.empty()) {
            return nullptr;
        }
        for (std::size_t slot = hash(key);; slot = (slot + 1) & (slots_.size() - 1)) {
            Idx const index = slots_[slot];
            if (index < 0) {
                return nullptr;
            }
            if (keys_[index] == key) {
                return &at(index);
            }
        }
    }
    // Requires key to be absent.
    Value const& insert(Idx key, Value const& value) {
        if (2 * (keys_.size() + 1) > slots_.size()) {
            grow();
        }
        auto const index = static_cast<Idx>(keys_.size());
        if (index % chunk == 0) {
            chunks_.push_back(std::make_unique<Value[]>(chunk));
        }
        keys_.push_back(key);
        Value& stored = chunks_.back()[index % chunk];
        stored = value;
        place(index);
        return stored;
    }

  private:
    static constexpr Idx chunk = 256;

    Value const& at(Idx index) const { return chunks_[index / chunk][index % chunk]; }
    std::size_t hash(Idx key) const {
        return static_cast<std::size_t>((static_cast<std::uint64_t>(key) * 0x9E3779B97F4A7C15ULL) >> 17) &
               (slots_.size() - 1);
    }
    void place(Idx index) {
        std::size_t slot = hash(keys_[index]);
        while (slots_[slot] >= 0) {
            slot = (slot + 1) & (slots_.size() - 1);
        }
        slots_[slot] = index;
    }
    void grow() {
        slots_.assign(std::max<std::size_t>(64, 2 * slots_.size()), -1);
        for (Idx index = 0; index < static_cast<Idx>(keys_.size()); ++index) {
            place(index);
        }
    }

    std::vector<Idx> slots_;
    std::vector<Idx> keys_;
    // Fixed-size chunks keep returned references stable; a std::deque node holds only a few results, so it
    // allocated every few insertions.
    std::vector<std::unique_ptr<Value[]>> chunks_;
};

// Compressed rows of sorted, unique bus indices.
struct Graph {
    std::vector<Idx> offsets{0};
    std::vector<Idx> targets;

    Idx size() const { return static_cast<Idx>(offsets.size()) - 1; }
    std::span<Idx const> row(Idx bus) const {
        return {targets.data() + offsets[bus], static_cast<std::size_t>(offsets[bus + 1] - offsets[bus])};
    }
};

// Sorted and deduplicated rows, as given.
inline Graph compress(std::vector<std::vector<Idx>> rows) {
    Graph result;
    for (auto& row : rows) {
        std::ranges::sort(row);
        row.erase(std::unique(row.begin(), row.end()), row.end());
        result.targets.insert(result.targets.end(), row.begin(), row.end());
        result.offsets.push_back(static_cast<Idx>(result.targets.size()));
    }
    return result;
}

// Undirected graph on n buses from (a, b) edges; out-of-range buses and self loops are dropped.
inline Graph undirected(Idx n, std::span<std::array<Idx, 2> const> edges) {
    auto const valid = [n](auto const& edge) {
        return edge[0] >= 0 && edge[1] >= 0 && edge[0] < n && edge[1] < n && edge[0] != edge[1];
    };
    Graph result;
    result.offsets.assign(static_cast<std::size_t>(n + 1), 0);
    for (auto const& edge : edges) {
        if (valid(edge)) {
            ++result.offsets[edge[0] + 1];
            ++result.offsets[edge[1] + 1];
        }
    }
    std::partial_sum(result.offsets.begin(), result.offsets.end(), result.offsets.begin());
    result.targets.resize(static_cast<std::size_t>(result.offsets.back()));
    std::vector<Idx> next(result.offsets.begin(), result.offsets.end() - 1);
    for (auto const& edge : edges) {
        if (valid(edge)) {
            result.targets[next[edge[0]]++] = edge[1];
            result.targets[next[edge[1]]++] = edge[0];
        }
    }
    // Sort each row and drop parallel edges.
    Graph unique;
    for (Idx bus = 0; bus < n; ++bus) {
        auto const first = result.targets.begin() + result.offsets[bus];
        auto const last = result.targets.begin() + result.offsets[bus + 1];
        std::sort(first, last);
        unique.targets.insert(unique.targets.end(), first, std::unique(first, last));
        unique.offsets.push_back(static_cast<Idx>(unique.targets.size()));
    }
    return unique;
}

// Tells whether a bus separates two others in a graph, from the low links of a depth-first search.
// A row may contain its own bus, which the search ignores.
class Separators {
  public:
    explicit Separators(std::vector<std::vector<Idx>> graph) : Separators{compress(std::move(graph))} {}
    explicit Separators(Graph graph)
        : graph_{std::move(graph)},
          tin_(graph_.size(), -1),
          end_(graph_.size()),
          low_(graph_.size()),
          component_(graph_.size()),
          parent_(graph_.size(), -1) {
        order_.reserve(graph_.size());
        for (Idx i = 0; i < size(); ++i) {
            if (tin_[i] < 0) {
                visit(i, i);
            }
        }
        // Children of each bus in the search tree, in increasing discovery time.
        children_.offsets.assign(graph_.size() + 1, 0);
        for (Idx const bus : order_) {
            if (parent_[bus] >= 0) {
                ++children_.offsets[parent_[bus] + 1];
            }
        }
        std::partial_sum(children_.offsets.begin(), children_.offsets.end(), children_.offsets.begin());
        children_.targets.resize(static_cast<std::size_t>(children_.offsets.back()));
        std::vector<Idx> next(children_.offsets.begin(), children_.offsets.end() - 1);
        for (Idx const bus : order_) {
            if (parent_[bus] >= 0) {
                children_.targets[next[parent_[bus]]++] = bus;
            }
        }
    }

    Idx size() const { return graph_.size(); }
    std::span<Idx const> neighbors(Idx j) const { return graph_.row(j); }
    // True if every path from j to m passes through k.
    bool separates(Idx j, Idx m, Idx k) const {
        return j != k && m != k && component_[j] == component_[m] && component_[j] == component_[k] &&
               part(j, k) != part(m, k);
    }

  private:
    void visit(Idx root, Idx component) {
        struct Frame {
            Idx bus;
            Idx parent;
            std::size_t next{};
        };
        std::vector<Frame> stack{{root, -1}};
        tin_[root] = low_[root] = clock_++;
        component_[root] = component;
        order_.push_back(root);
        while (!stack.empty()) {
            auto& frame = stack.back();
            Idx const v = frame.bus;
            auto const row = graph_.row(v);
            if (frame.next == row.size()) {
                end_[v] = clock_;
                if (frame.parent >= 0) {
                    low_[frame.parent] = std::min(low_[frame.parent], low_[v]);
                }
                stack.pop_back();
                continue;
            }
            Idx const w = row[frame.next++];
            if (w == frame.parent || w == v) {
                continue;
            }
            if (tin_[w] >= 0) {
                low_[v] = std::min(low_[v], tin_[w]);
            } else {
                parent_[w] = v;
                order_.push_back(w);
                tin_[w] = low_[w] = clock_++;
                component_[w] = component;
                stack.push_back({w, v});
            }
        }
    }
    // The child subtree of k that holds v and is cut off when k is removed, or -1.
    Idx part(Idx v, Idx k) const {
        auto const children = children_.row(k);
        auto it = std::upper_bound(children.begin(), children.end(), tin_[v],
                                   [&](Idx time, Idx child) { return time < tin_[child]; });
        if (it == children.begin()) {
            return -1;
        }
        Idx const child = *--it;
        return tin_[v] < end_[child] && low_[child] >= tin_[k] ? child : -1;
    }

    Graph graph_;
    std::vector<Idx> tin_;
    std::vector<Idx> end_;
    std::vector<Idx> low_;
    std::vector<Idx> component_;
    std::vector<Idx> parent_;
    std::vector<Idx> order_;
    Graph children_;
    Idx clock_{};
};

} // namespace completion_detail

// The selective inverse gives the blocks of Zbar = Abar_ps^-1 on the LU sparsity pattern only. Nodal injection
// uncertainty also needs the covariance of two buses j and m that are both neighbours of a third bus but
// not on that pattern. This class provides the covariance of the rectangular voltages of any such pair.
//
// Augmented path. For a common neighbour k of j and m in the graph of Abar_ps,
//     Zbar_jm ~ Zbar_jk Zbar_kk^-1 Zbar_km
// with the whole blocks, state coordinates and multipliers; only the state corner of the product is formed.
// This is exact if k separates j and m in that graph, and an approximation otherwise. The common neighbours
// are tried in ascending order, and one whose block Zbar_kk is worse conditioned than middle_condition_limit is
// skipped.
//
// Mean path. If every common neighbour was skipped, the result is the mean of C_jk C_kk^+ C_km over the
// common neighbours k in the branch topology, with C the covariance of the rectangular voltages and C_kk^+
// its pseudo-inverse on the support of C_kk.
//
// A pair is invalid if a block that its route needs is missing or not finite, or if the covariance of one of
// its buses is not positive semidefinite within roundoff. An invalid pair is not retried on another route.
//
// Everything is computed on first use and kept for the lifetime of the object. `Workspace` supplies
// find_block_index, inverse_block_at, block_count, variance_normalization and rectangular_covariance.
template <class Workspace> class CovarianceCompletion {
  public:
    using Matrix = typename Workspace::RectangularMatrix;
    using StateMatrix = typename Workspace::StateMatrix;
    using RawMatrix = typename Workspace::RawMatrix;
    using Scalar = typename RawMatrix::Scalar;

    // Covariance of the rectangular voltages of two buses. `value` is meaningful only if `valid`.
    struct Block {
        Matrix value = Matrix::Zero();
        bool valid = false;
    };
    struct Result {
        Block block;
        UQPairDiagnostic diagnostic{};
    };

    // `augmented_graph` holds the neighbours of every bus in Abar_ps; `physical_graph` the branch topology.
    // The approximation warning of a pair is computed only with `diagnostics`.
    CovarianceCompletion(Workspace const& workspace, std::vector<std::vector<Idx>> augmented_graph,
                         std::vector<std::vector<Idx>> const& physical_graph, bool diagnostics = true)
        : CovarianceCompletion{workspace, completion_detail::compress(std::move(augmented_graph)),
                               symmetric(physical_graph), diagnostics} {}
    // Compressed form; physical_graph must be symmetric without self loops.
    CovarianceCompletion(Workspace const& workspace, completion_detail::Graph augmented_graph,
                         completion_detail::Graph physical_graph, bool diagnostics = true)
        : workspace_{workspace},
          diagnostics_{diagnostics},
          separators_{std::move(augmented_graph)},
          adjacency_{std::move(physical_graph)},
          marginals_(static_cast<std::size_t>(adjacency_.size())),
          stored_(static_cast<std::size_t>(workspace.block_count())) {}

    // Covariance of one bus with itself, with negative roundoff eigenvalues set to zero.
    Block const& diagonal(Idx bus) { return marginal(bus).covariance; }

    // Covariance and diagnostic of the pair j < m.
    Result pair(Idx j, Idx m) {
        if (auto const index = workspace_.find_block_index(j, m)) {
            Result result;
            result.block = stored(*index, j, m);
            result.diagnostic = {.first_bus = j,
                                 .second_bus = m,
                                 .route = result.block.valid ? UQPairRoute::stored : UQPairRoute::invalid};
            return result;
        }
        return completed(j, m);
    }
    // Covariance of the pair j < m alone.
    Block const& block(Idx j, Idx m) {
        if (auto const index = workspace_.find_block_index(j, m)) {
            return stored(*index, j, m);
        }
        return completed(j, m).block;
    }

  private:
    static constexpr int raw_size = RawMatrix::RowsAtCompileTime;
    static constexpr int state_size = StateMatrix::RowsAtCompileTime;
    using StateColumns = Eigen::Matrix<Scalar, raw_size, state_size>;

    enum class MiddleStatus : std::uint8_t { nonfinite, singular, conditioned };
    enum class SolveStatus : std::uint8_t { nonfinite, residual, accurate };

    // h = Zbar_kk^-1 Zbar_km for one right bus m. Only the state columns are kept.
    struct Solve {
        Idx right_bus{};
        SolveStatus status{};
        StateColumns x = StateColumns::Zero();
    };
    // Conditioning and factorization of Zbar_kk depend only on k, and h only on (k, m). The missing pairs through
    // one middle are requested together, so the most recent middle is kept with its solves.
    struct Middle {
        Idx bus{-1};
        MiddleStatus status{};
        RawMatrix block = RawMatrix::Zero();
        Eigen::PartialPivLU<RawMatrix> lu;
        std::vector<Solve> solves;
    };
    // Covariance of one bus with its inverse root on the support, W with W^T W = C^+.
    struct Marginal {
        Block covariance;
        Matrix whitener = Matrix::Zero();
        bool valid{};
        int rank{};
    };

    static completion_detail::Graph symmetric(std::vector<std::vector<Idx>> const& rows) {
        std::vector<std::array<Idx, 2>> edges;
        for (Idx i = 0; i < static_cast<Idx>(rows.size()); ++i) {
            for (Idx const j : rows[i]) {
                edges.push_back({i, j});
            }
        }
        return completion_detail::undirected(static_cast<Idx>(rows.size()), edges);
    }

    Idx size() const { return static_cast<Idx>(marginals_.size()); }

    // ----- Covariance of single buses and of pairs on the LU sparsity pattern -----

    Marginal const& marginal(Idx bus) {
        auto& result = marginals_[bus];
        if (!result) {
            result = make_marginal(workspace_.rectangular_covariance(bus, bus));
        }
        return *result;
    }
    static Marginal make_marginal(Matrix const& value) {
        Marginal result;
        result.covariance.value = value;
        if (!value.allFinite()) {
            return result;
        }
        Matrix const symmetric = (value + value.transpose()) / 2.0;
        Eigen::SelfAdjointEigenSolver<Matrix> const eigen{symmetric};
        if (eigen.info() != Eigen::Success || eigen.eigenvalues().minCoeff() < -round_rtol * symmetric.norm()) {
            return result;
        }
        // An eigenvalue supports the covariance if it is not negligible against the largest one.
        auto const values = eigen.eigenvalues().cwiseMax(0.0).eval();
        auto inverse_roots = values;
        for (int d = 0; d != values.size(); ++d) {
            bool const supported = values[d] > round_rtol * values.maxCoeff();
            inverse_roots[d] = supported ? 1.0 / std::sqrt(values[d]) : 0.0;
            result.rank += supported ? 1 : 0;
        }
        auto const& vectors = eigen.eigenvectors();
        result.whitener = vectors * inverse_roots.asDiagonal() * vectors.transpose();
        // Keep the symmetric block itself when no eigenvalue was raised to zero.
        result.covariance.value = eigen.eigenvalues().minCoeff() >= 0.0
                                      ? symmetric
                                      : Matrix{vectors * values.asDiagonal() * vectors.transpose()};
        result.valid = result.whitener.allFinite() && result.covariance.value.allFinite();
        result.covariance.valid = result.valid;
        return result;
    }

    // A covariance of the pair (j, m) is usable if both buses have a valid covariance and the correlation
    // W_j C_jm W_m^T, with W the inverse roots, is finite. With diagnostics, `warning` is set when the
    // spectral norm of that correlation exceeds correlation_limit.
    bool consistent(Idx j, Idx m, Matrix const& value, bool* warning = nullptr) {
        auto const& a = marginal(j);
        auto const& b = marginal(m);
        if (!a.valid || !b.valid || !value.allFinite()) {
            return false;
        }
        Matrix const correlation = a.whitener * value * b.whitener.transpose();
        if (!correlation.allFinite()) {
            return false;
        }
        if (diagnostics_ && warning != nullptr) {
            Eigen::JacobiSVD<Matrix> const singular{correlation};
            if (singular.info() != Eigen::Success || !singular.singularValues().allFinite() ||
                singular.singularValues().maxCoeff() > completion_detail::correlation_limit) {
                *warning = true;
            }
        }
        return true;
    }

    // Pair i < j at its position `index` on the LU sparsity pattern.
    Block const& stored(Idx index, Idx i, Idx j) {
        auto& result = stored_[index];
        if (!result) {
            Block block;
            block.value = workspace_.rectangular_covariance(i, j);
            block.valid = consistent(i, j, block.value);
            result = block;
        }
        return *result;
    }
    // Valid covariance of a pair on the pattern, in the requested orientation.
    std::optional<Matrix> donor(Idx j, Idx m) {
        auto const [i, k] = std::minmax(j, m);
        auto const index = workspace_.find_block_index(i, k);
        if (!index) {
            return std::nullopt;
        }
        auto const& block = stored(*index, i, k);
        if (!block.valid) {
            return std::nullopt;
        }
        if (j < m) {
            return block.value;
        }
        return Matrix{block.value.transpose()};
    }

    // ----- Middle block of the augmented path -----

    Middle& middle(Idx k, RawMatrix const& block) {
        if (middle_.bus == k) {
            return middle_;
        }
        middle_.bus = k;
        middle_.solves.clear();
        Eigen::JacobiSVD<RawMatrix> const decomposition{block};
        auto const& singular = decomposition.singularValues();
        if (decomposition.info() != Eigen::Success || !singular.allFinite()) {
            middle_.status = MiddleStatus::nonfinite;
        } else if (singular.maxCoeff() == 0.0 ||
                   singular.minCoeff() / singular.maxCoeff() < completion_detail::middle_condition_limit) {
            middle_.status = MiddleStatus::singular;
        } else {
            middle_.status = MiddleStatus::conditioned;
            middle_.block = block;
            middle_.lu.compute(middle_.block);
        }
        return middle_;
    }
    // Forward and back substitution with the partial-pivoting factors. The fixed-size loops avoid Eigen's
    // blocked triangular kernels, which dominated these tiny solves.
    template <int columns>
    static Eigen::Matrix<Scalar, raw_size, columns> substitute(Eigen::PartialPivLU<RawMatrix> const& lu,
                                                               Eigen::Matrix<Scalar, raw_size, columns> const& right) {
        Eigen::Matrix<Scalar, raw_size, columns> x = lu.permutationP() * right;
        auto const& factors = lu.matrixLU();
        for (int c = 0; c != columns; ++c) {
            for (int i = 1; i != raw_size; ++i) {
                for (int k = 0; k != i; ++k) {
                    x(i, c) -= factors(i, k) * x(k, c);
                }
            }
            for (int i = raw_size - 1; i >= 0; --i) {
                for (int k = i + 1; k != raw_size; ++k) {
                    x(i, c) -= factors(i, k) * x(k, c);
                }
                x(i, c) /= factors(i, i);
            }
        }
        return x;
    }
    // Solve with one step of iterative refinement.
    template <int columns>
    static Eigen::Matrix<Scalar, raw_size, columns>
    refined_solve(Middle const& middle, Eigen::Matrix<Scalar, raw_size, columns> const& right) {
        Eigen::Matrix<Scalar, raw_size, columns> x = substitute<columns>(middle.lu, right);
        Eigen::Matrix<Scalar, raw_size, columns> const residual = right - middle.block * x;
        x += substitute<columns>(middle.lu, residual);
        return x;
    }
    // Requires a conditioned middle.
    Solve const& solve(Middle& middle, Idx m, RawMatrix const& right) {
        for (auto const& cached : middle.solves) {
            if (cached.right_bus == m) {
                return cached;
            }
        }
        auto& result = middle.solves.emplace_back();
        result.right_bus = m;
        result.x = refined_solve<state_size>(middle, StateColumns{right.template leftCols<state_size>()});
        if (!result.x.allFinite()) {
            result.status = SolveStatus::nonfinite;
            return result;
        }
        result.status = SolveStatus::accurate;
        if (!diagnostics_) {
            return result;
        }
        // The residual warning looks at all columns, solved with the same arithmetic.
        RawMatrix const full = refined_solve<raw_size>(middle, right);
        RawMatrix const residual = right - middle.block * full;
        double const denominator = middle.block.norm() * full.norm() + right.norm();
        if (!full.allFinite() || !residual.allFinite() || !std::isfinite(denominator) ||
            residual.norm() > completion_detail::solve_residual_limit * (denominator == 0.0 ? 1.0 : denominator)) {
            result.status = SolveStatus::residual;
        }
        return result;
    }

    // ----- Pairs that are not on the LU sparsity pattern -----

    Result const& completed(Idx j, Idx m) {
        auto const key = j * size() + m;
        if (auto const* found = completed_.find(key)) {
            return *found;
        }
        Result result;
        result.diagnostic.first_bus = j;
        result.diagnostic.second_bus = m;
        if (!augmented_path(j, m, result)) {
            mean_path(j, m, result);
        }
        return completed_.insert(key, result);
    }

    // Returns false if every common neighbour was skipped, so that the mean path is tried. Returns true with
    // an invalid result if there is no common neighbour or a needed block is missing or not finite.
    bool augmented_path(Idx j, Idx m, Result& result) {
        bool has_candidate = false;
        auto const first = separators_.neighbors(j);
        auto const second = separators_.neighbors(m);
        auto first_it = first.begin();
        auto second_it = second.begin();
        while (first_it != first.end() && second_it != second.end()) {
            if (*first_it < *second_it) {
                ++first_it;
                continue;
            }
            if (*second_it < *first_it) {
                ++second_it;
                continue;
            }
            Idx const k = *first_it;
            ++first_it;
            ++second_it;
            if (k == j || k == m) {
                continue;
            }
            has_candidate = true;
            auto const left = workspace_.find_block_index(j, k);
            auto const center = workspace_.find_block_index(k, k);
            auto const right = workspace_.find_block_index(k, m);
            if (!left || !center || !right) {
                return true;
            }
            auto& conditioned = middle(k, workspace_.inverse_block_at(*center).matrix());
            if (conditioned.status == MiddleStatus::nonfinite) {
                return true;
            }
            if (conditioned.status == MiddleStatus::singular) {
                continue;
            }
            auto const& solved = solve(conditioned, m, workspace_.inverse_block_at(*right).matrix());
            if (solved.status == SolveStatus::nonfinite) {
                return true;
            }
            bool warning = solved.status == SolveStatus::residual;
            // Only the state corner of Zbar_jk h is formed.
            RawMatrix const& left_block = workspace_.inverse_block_at(*left).matrix();
            StateMatrix const corner = left_block.template topRows<state_size>() * solved.x;
            Matrix const value = workspace_.rectangular_covariance(j, m, workspace_.variance_normalization() * corner);
            if (!consistent(j, m, value, &warning)) {
                return true;
            }
            result.block = {value, true};
            result.diagnostic.middle_bus = k;
            result.diagnostic.route =
                separators_.separates(j, m, k) ? UQPairRoute::separator : UQPairRoute::augmented_path;
            result.diagnostic.approximation_warning = warning;
            return true;
        }
        return !has_candidate;
    }

    void mean_path(Idx j, Idx m, Result& result) {
        if (!marginal(j).valid || !marginal(m).valid) {
            return;
        }
        Matrix sum = Matrix::Zero();
        Idx count{};
        for (Idx const k : adjacency_.row(j)) {
            if (!std::ranges::binary_search(adjacency_.row(m), k)) {
                continue;
            }
            auto const& center = marginal(k);
            if (!center.valid) {
                return;
            }
            if (center.rank == 0) {
                continue;
            }
            auto const left = donor(j, k);
            auto const right = donor(k, m);
            if (!left || !right) {
                return;
            }
            Matrix const prediction = *left * center.whitener * center.whitener.transpose() * *right;
            if (!prediction.allFinite()) {
                return;
            }
            sum += prediction;
            ++count;
        }
        if (count == 0) {
            return;
        }
        Matrix const mean = sum / static_cast<double>(count);
        bool warning = false;
        if (!consistent(j, m, mean, &warning)) {
            return;
        }
        result.block = {mean, true};
        bool const zero_target = (marginal(j).rank == 0 || marginal(m).rank == 0) && mean.isZero(0.0);
        result.diagnostic.route = zero_target ? UQPairRoute::zero_target : UQPairRoute::mean_path;
        result.diagnostic.approximation_warning = warning;
    }

    Workspace const& workspace_;
    bool diagnostics_;
    completion_detail::Separators separators_;
    completion_detail::Graph adjacency_; // sorted, symmetric branch topology
    std::vector<std::optional<Marginal>> marginals_;
    std::vector<std::optional<Block>> stored_;       // by position on the LU sparsity pattern
    completion_detail::PairTable<Result> completed_; // pairs off the pattern, by key j * size + m
    Middle middle_;
};

} // namespace power_grid_model::math_solver::state_estimation_uq
