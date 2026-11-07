// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include "state_estimation_uq_branch.hpp"
#include "state_estimation_uq_workspace.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace power_grid_model::math_solver::state_estimation_uq {
namespace completion_detail {
inline constexpr double round_tolerance = round_rtol;

// The product of square factors in double; false if an entry is nonfinite.
template <class Matrix> bool product(std::initializer_list<Matrix> factors, Matrix& target) {
    Matrix result = Matrix::Identity();
    for (auto const& factor : factors) {
        Matrix next;
        for (Eigen::Index r = 0; r < result.rows(); ++r)
            for (Eigen::Index c = 0; c < result.cols(); ++c) {
                Accumulator<typename Matrix::Scalar> sum;
                for (Eigen::Index k = 0; k < result.cols(); ++k)
                    sum.add_product(result(r, k), factor(k, c));
                next(r, c) = sum.value();
            }
        result = next;
    }
    target = result;
    return result.allFinite();
}

// Insert-only open-addressing table of completed pairs by nonnegative key, with
// linear probing at most half full. A std::unordered_map node allocation per
// missing pair was a visible share of large injection requests.
template <class Value> class PairTable {
  public:
    Value const* find(Idx key) const {
        if (slots_.empty())
            return nullptr;
        for (std::size_t slot = hash(key);; slot = (slot + 1) & (slots_.size() - 1)) {
            Idx const index = slots_[slot];
            if (index < 0)
                return nullptr;
            if (keys_[index] == key)
                return &at(index);
        }
    }
    // Requires key to be absent.
    Value const& insert(Idx key, Value const& value) {
        if (2 * (keys_.size() + 1) > slots_.size())
            grow();
        auto const index = static_cast<Idx>(keys_.size());
        if (index % chunk == 0)
            chunks_.push_back(std::make_unique<Value[]>(chunk));
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
        while (slots_[slot] >= 0)
            slot = (slot + 1) & (slots_.size() - 1);
        slots_[slot] = index;
    }
    void grow() {
        slots_.assign(std::max<std::size_t>(64, 2 * slots_.size()), -1);
        for (Idx index = 0; index < static_cast<Idx>(keys_.size()); ++index)
            place(index);
    }
    std::vector<Idx> slots_;
    std::vector<Idx> keys_;
    // Fixed-size chunks keep returned references stable; a std::deque node holds
    // only a few results, so it allocated every few insertions.
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
// Undirected graph on n buses from (a, b) edges; out-of-range buses and self
// loops are dropped.
inline Graph undirected(Idx n, std::span<std::array<Idx, 2> const> edges) {
    auto valid = [n](auto const& edge) {
        return edge[0] >= 0 && edge[1] >= 0 && edge[0] < n && edge[1] < n && edge[0] != edge[1];
    };
    Graph result;
    result.offsets.assign(static_cast<std::size_t>(n + 1), 0);
    for (auto const& edge : edges)
        if (valid(edge)) {
            ++result.offsets[edge[0] + 1];
            ++result.offsets[edge[1] + 1];
        }
    std::partial_sum(result.offsets.begin(), result.offsets.end(), result.offsets.begin());
    result.targets.resize(static_cast<std::size_t>(result.offsets.back()));
    std::vector<Idx> next(result.offsets.begin(), result.offsets.end() - 1);
    for (auto const& edge : edges)
        if (valid(edge)) {
            result.targets[next[edge[0]]++] = edge[1];
            result.targets[next[edge[1]]++] = edge[0];
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

// DFS low links on the unfactorized augmented bus graph. Structural zeros must
// be present in this graph; fill belongs only to the workspace's raw pattern.
// A row may contain its own bus, which the traversal ignores.
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
        for (Idx i = 0; i < size(); ++i)
            if (tin_[i] < 0)
                visit(i, i);
        // DFS tree children of each bus, in increasing discovery time.
        children_.offsets.assign(graph_.size() + 1, 0);
        for (Idx bus : order_)
            if (parent_[bus] >= 0)
                ++children_.offsets[parent_[bus] + 1];
        std::partial_sum(children_.offsets.begin(), children_.offsets.end(), children_.offsets.begin());
        children_.targets.resize(static_cast<std::size_t>(children_.offsets.back()));
        std::vector<Idx> next(children_.offsets.begin(), children_.offsets.end() - 1);
        for (Idx bus : order_)
            if (parent_[bus] >= 0)
                children_.targets[next[parent_[bus]]++] = bus;
    }
    Idx size() const { return graph_.size(); }
    std::span<Idx const> neighbors(Idx j) const { return graph_.row(j); }
    bool separates(Idx j, Idx m, Idx k) const {
        return j != k && m != k && component_[j] == component_[m] && component_[j] == component_[k] &&
               part(j, k) != part(m, k);
    }

  private:
    void visit(Idx root, Idx component) {
        struct Frame {
            Idx bus, parent;
            std::size_t next{};
        };
        std::vector<Frame> stack{{root, -1}};
        tin_[root] = low_[root] = clock_++;
        component_[root] = component;
        order_.push_back(root);
        while (!stack.empty()) {
            auto& frame = stack.back();
            Idx v = frame.bus;
            auto const row = graph_.row(v);
            if (frame.next == row.size()) {
                end_[v] = clock_;
                if (frame.parent >= 0)
                    low_[frame.parent] = std::min(low_[frame.parent], low_[v]);
                stack.pop_back();
                continue;
            }
            Idx w = row[frame.next++];
            if (w == frame.parent || w == v)
                continue;
            if (tin_[w] >= 0)
                low_[v] = std::min(low_[v], tin_[w]);
            else {
                parent_[w] = v;
                order_.push_back(w);
                tin_[w] = low_[w] = clock_++;
                component_[w] = component;
                stack.push_back({w, v});
            }
        }
    }
    Idx part(Idx v, Idx k) const {
        auto const children = children_.row(k);
        auto it = std::upper_bound(children.begin(), children.end(), tin_[v],
                                   [&](Idx time, Idx child) { return time < tin_[child]; });
        if (it == children.begin())
            return -1;
        Idx child = *--it;
        return tin_[v] < end_[child] && low_[child] >= tin_[k] ? child : -1;
    }
    Graph graph_;
    std::vector<Idx> tin_, end_, low_, component_, parent_, order_;
    Graph children_;
    Idx clock_{};
};
} // namespace completion_detail

// One frozen donor snapshot per island, derived lazily: a marginal, stored
// pair or completion is computed on first use and cached for this call only.
// Workspace is immutable and supplies normalized/reference-corrected physical
// blocks, so laziness cannot change any result. A completion is never promoted
// to a raw or physical original donor.
//
// All arithmetic is double. The middle solve is a partial-pivoting LU with one
// step of iterative refinement using an ordinary double residual; a nonfinite
// result is invalid. The approximation warning is computed only with requested
// diagnostics, so values never depend on it.
template <class Workspace> class CovarianceCompletion {
  public:
    using Matrix = typename Workspace::PhysicalMatrix;
    using RawMatrix = typename Workspace::RawMatrix;
    using PhysicalBlock = typename Workspace::PhysicalBlock;
    using Scalar = typename Matrix::Scalar;
    using Pair = std::pair<Idx, Idx>;
    struct Result {
        PhysicalBlock block{Matrix::Zero(), 0.0, false};
        UQPairDiagnostic diagnostic{};
    };

    // Shares caller-owned physical blocks already derived by branch_results.
    CovarianceCompletion(Workspace const& workspace, std::vector<std::vector<Idx>> augmented_graph,
                         std::vector<std::vector<Idx>> physical_graph, PhysicalBlocks<Workspace>& stored,
                         bool diagnostics = true)
        : CovarianceCompletion{workspace, completion_detail::compress(std::move(augmented_graph)),
                               symmetric(physical_graph), stored, diagnostics} {}
    // Compressed form; physical_graph must be symmetric without self loops.
    CovarianceCompletion(Workspace const& workspace, completion_detail::Graph augmented_graph,
                         completion_detail::Graph physical_graph, PhysicalBlocks<Workspace>& stored,
                         bool diagnostics = true)
        : workspace_{workspace},
          diagnostics_{diagnostics},
          stored_{stored},
          separators_{std::move(augmented_graph)},
          adjacency_{std::move(physical_graph)},
          marginals_(static_cast<std::size_t>(adjacency_.size())),
          originals_(workspace.augmented_matrix.size()) {}

    // Original diagonal after Hermitian validation and admissible negative
    // clipping. Invalid input values remain inspectable with valid=false.
    PhysicalBlock const& diagonal(Idx bus) { return marginal(bus).covariance; }

    // j < m.
    Result pair(Idx j, Idx m) {
        Result result;
        result.diagnostic.first_bus = j;
        result.diagnostic.second_bus = m;
        if (auto const index = workspace_.find_block_index(j, m)) {
            result.block = original(*index, j, m);
            result.diagnostic.route = result.block.valid ? UQPairRoute::stored : UQPairRoute::invalid;
            return result;
        }
        return completed(j, m);
    }
    // The pair's block alone, without copying its diagnostic; j < m.
    PhysicalBlock const& block(Idx j, Idx m) {
        if (auto const index = workspace_.find_block_index(j, m))
            return original(*index, j, m);
        return completed(j, m).block;
    }

  private:
    static constexpr int raw_size = RawMatrix::RowsAtCompileTime;
    static constexpr int state_size = Matrix::ColsAtCompileTime;

    // Symmetric closure of a physical graph given as rows.
    static completion_detail::Graph symmetric(std::vector<std::vector<Idx>> const& rows) {
        std::vector<std::array<Idx, 2>> edges;
        for (Idx i = 0; i < static_cast<Idx>(rows.size()); ++i)
            for (Idx j : rows[i])
                edges.push_back({i, j});
        return completion_detail::undirected(static_cast<Idx>(rows.size()), edges);
    }
    // Originally missing pair, completed once.
    Result const& completed(Idx j, Idx m) {
        auto const key = j * size() + m;
        if (auto const* found = completed_.find(key))
            return *found;
        Result result;
        result.diagnostic.first_bus = j;
        result.diagnostic.second_bus = m;
        complete(j, m, result);
        return completed_.insert(key, result);
    }
    enum class MiddleStatus : std::uint8_t { nonfinite, singular, conditioned };
    enum class SolveStatus : std::uint8_t { nonfinite, residual, accurate };
    // x solves B_kk x = B_km. Only the state columns are solved, all columns
    // with requested diagnostics.
    struct Solve {
        Idx right_bus{};
        SolveStatus status{};
        RawMatrix x = RawMatrix::Zero();
    };
    // Conditioning and factorization of a raw middle block B_kk depend only on k,
    // and h=B_kk^-1 B_km only on (k, m). Missing pairs through one middle are
    // requested together, so the most recent middle is kept with its solves.
    // A miss recomputes with unchanged arithmetic.
    struct Middle {
        Idx bus{-1};
        MiddleStatus status{};
        RawMatrix block = RawMatrix::Zero();
        Eigen::PartialPivLU<RawMatrix> lu;
        std::vector<Solve> solves;
    };
    Middle& middle(Idx k, RawMatrix const& block) {
        if (middle_.bus == k)
            return middle_;
        middle_.bus = k;
        middle_.solves.clear();
        Eigen::JacobiSVD<RawMatrix> decomposition{block};
        auto singular = decomposition.singularValues().eval();
        if (decomposition.info() != Eigen::Success || !singular.allFinite())
            middle_.status = MiddleStatus::nonfinite;
        else if (singular.maxCoeff() == 0 || singular.minCoeff() / singular.maxCoeff() < 1e-12)
            middle_.status = MiddleStatus::singular;
        else {
            middle_.status = MiddleStatus::conditioned;
            middle_.block = block;
            middle_.lu.compute(middle_.block);
        }
        return middle_;
    }
    // Forward and back substitution with the partial-pivoting factors; the
    // fixed-size form avoids Eigen's blocked triangular kernels, which dominated
    // these tiny solves.
    template <int columns>
    static Eigen::Matrix<Scalar, raw_size, columns> substitute(Eigen::PartialPivLU<RawMatrix> const& lu,
                                                               Eigen::Matrix<Scalar, raw_size, columns> const& right) {
        Eigen::Matrix<Scalar, raw_size, columns> x = (lu.permutationP() * right).eval();
        auto const& factors = lu.matrixLU();
        for (int c = 0; c != columns; ++c) {
            for (int i = 1; i != raw_size; ++i)
                for (int k = 0; k != i; ++k)
                    x(i, c) -= factors(i, k) * x(k, c);
            for (int i = raw_size - 1; i >= 0; --i) {
                for (int k = i + 1; k != raw_size; ++k)
                    x(i, c) -= factors(i, k) * x(k, c);
                x(i, c) /= factors(i, i);
            }
        }
        return x;
    }
    // One ordinary-double residual/solve refinement, rounded into x. This is
    // iterative refinement of the linear solve, not rounding compensation.
    template <int columns>
    static Eigen::Matrix<Scalar, raw_size, columns>
    refined_solve(Middle const& middle, Eigen::Matrix<Scalar, raw_size, columns> const& right) {
        auto x = substitute<columns>(middle.lu, right);
        auto const residual = linear_residual<columns>(middle.block, x, right);
        x += substitute<columns>(middle.lu, residual);
        return x;
    }
    template <int columns>
    static Eigen::Matrix<Scalar, raw_size, columns>
    linear_residual(RawMatrix const& block, Eigen::Matrix<Scalar, raw_size, columns> const& x,
                    Eigen::Matrix<Scalar, raw_size, columns> const& right) {
        Eigen::Matrix<Scalar, raw_size, columns> residual;
        for (int c = 0; c != columns; ++c)
            for (int i = 0; i != raw_size; ++i) {
                Accumulator<Scalar> sum;
                sum.add(right(i, c));
                for (int k = 0; k != raw_size; ++k)
                    sum.add_product(-block(i, k), x(k, c));
                residual(i, c) = sum.value();
            }
        return residual;
    }
    // Requires a conditioned middle.
    Solve const& solve(Middle& middle, Idx m, RawMatrix const& right) {
        using namespace completion_detail;
        for (auto const& cached : middle.solves)
            if (cached.right_bus == m)
                return cached;
        auto& result = middle.solves.emplace_back();
        result.right_bus = m;
        // The reconstruction reads only the state columns of h. Requested
        // diagnostics also solve the other columns, with the same arithmetic per
        // column, for the residual warning alone.
        using StateColumns = Eigen::Matrix<Scalar, raw_size, state_size>;
        StateColumns const columns = refined_solve<state_size>(middle, right.template leftCols<state_size>().eval());
        if (!columns.allFinite()) {
            result.status = SolveStatus::nonfinite;
            return result;
        }
        result.x.template leftCols<state_size>() = columns;
        result.status = SolveStatus::accurate;
        if (!diagnostics_)
            return result;
        RawMatrix const full = refined_solve<raw_size>(middle, right);
        if (!full.allFinite()) {
            result.status = SolveStatus::residual;
            return result;
        }
        RawMatrix const residual = linear_residual<raw_size>(middle.block, full, right);
        double const denominator = frobenius_norm(middle.block) * frobenius_norm(full) + frobenius_norm(right);
        if (!residual.allFinite() || !std::isfinite(denominator) ||
            frobenius_norm(residual) > 1e-12 * (denominator == 0 ? 1.0 : denominator))
            result.status = SolveStatus::residual;
        return result;
    }
    // Only the state corner of B_jk h is formed.
    static Matrix reconstruct_corner(RawMatrix const& left, Solve const& solved) {
        Matrix corner;
        for (int r = 0; r != state_size; ++r)
            for (int c = 0; c != state_size; ++c) {
                Accumulator<Scalar> sum;
                for (int k = 0; k != raw_size; ++k)
                    sum.add_product(left(r, k), solved.x(k, c));
                corner(r, c) = sum.value();
            }
        return corner;
    }
    struct Marginal {
        PhysicalBlock covariance{Matrix::Zero(), 0.0, false};
        Matrix root = Matrix::Zero(), whitener = Matrix::Zero();
        bool valid{};
        int rank{};
    };
    Idx size() const { return static_cast<Idx>(marginals_.size()); }
    Marginal const& marginal(Idx bus) {
        auto& result = marginals_[bus];
        if (!result) {
            auto const index = workspace_.find_block_index(bus, bus);
            result = make_marginal(index ? stored_.stored(*index, bus, bus) : workspace_.physical_block(bus, bus));
        }
        return *result;
    }
    // Stored canonical pair i < j at its LU position.
    PhysicalBlock const& original(Idx index, Idx i, Idx j) {
        auto& result = originals_[index];
        if (!result)
            result = make_original(index, i, j);
        return *result;
    }
    // Reference blocks use the same bounded symmetry rule; production stored
    // views are already canonical.
    PhysicalBlock make_original(Idx index, Idx i, Idx j) {
        auto block = branch_detail::physical_pair(workspace_, i, j, stored_.stored(index, i, j));
        block.valid = block.valid && consistent(i, j, block.value);
        return block;
    }
    static Marginal make_marginal(PhysicalBlock const& block) {
        Marginal result;
        result.covariance = block;
        result.covariance.valid = false;
        if (!block.valid)
            return result;
        double const tolerance = completion_detail::round_tolerance * block.construction_scale;
        if (!block.symmetry_projected &&
            frobenius_norm((block.value - block.value.adjoint()).eval()) > symmetry_rtol * block.construction_scale)
            return result;
        Matrix hermitian;
        for (Eigen::Index r = 0; r != hermitian.rows(); ++r)
            for (Eigen::Index c = 0; c != hermitian.cols(); ++c)
                hermitian(r, c) = average(block.value(r, c), conjugate(block.value(c, r)));
        Eigen::SelfAdjointEigenSolver<Matrix> eigen{hermitian};
        if (eigen.info() != Eigen::Success || eigen.eigenvalues().minCoeff() < -tolerance)
            return result;
        auto values = eigen.eigenvalues().cwiseMax(0.).eval(), roots = values.eval(), inverses = values.eval();
        for (int d = 0; d < Matrix::RowsAtCompileTime; ++d) {
            roots[d] = std::sqrt(values[d]);
            bool supported = values[d] > completion_detail::round_tolerance * values.maxCoeff();
            inverses[d] = supported ? 1 / roots[d] : 0;
            result.rank += supported;
        }
        auto const& vectors = eigen.eigenvectors();
        auto reconstruct = [&vectors](auto const& diagonal, Matrix& target) {
            target = vectors * diagonal.template cast<Scalar>().asDiagonal() * vectors.adjoint();
            return target.allFinite();
        };
        result.valid = reconstruct(roots, result.root) && reconstruct(inverses, result.whitener);
        if (eigen.eigenvalues().minCoeff() >= 0) {
            // Keep the Hermitian block when no eigenvalue needs clipping.
            result.covariance.value = hermitian;
        } else {
            result.valid = result.valid && reconstruct(values, result.covariance.value);
        }
        result.covariance.valid = result.valid;
        return result;
    }
    // Validity of a pair against its endpoint marginals: the whitened
    // coefficient and the restored pair must be finite. With requested
    // diagnostics, warning (if given) is set when the normalized correlation
    // spectral norm exceeds 1 + 1e-6 or cannot be computed.
    bool consistent(Idx j, Idx m, Matrix const& value, bool* warning = nullptr) {
        auto const& a = marginal(j);
        auto const& b = marginal(m);
        if (!a.valid || !b.valid)
            return false;
        Matrix coefficient;
        Matrix restored;
        if (!completion_detail::product<Matrix>({a.whitener, value, b.whitener.adjoint()}, coefficient) ||
            !completion_detail::product<Matrix>({a.root, coefficient, b.root.adjoint()}, restored))
            return false;
        if (!diagnostics_ || warning == nullptr)
            return true;
        // Diagnostics never change validity: a failed check only warns.
        Eigen::JacobiSVD<Matrix> singular{coefficient};
        if (singular.info() != Eigen::Success || !singular.singularValues().allFinite() ||
            singular.singularValues().maxCoeff() > 1 + 1e-6)
            *warning = true;
        return true;
    }
    // Valid stored pair, in the requested orientation.
    std::optional<Matrix> donor(Idx j, Idx m) {
        auto const [i, k] = std::minmax(j, m);
        auto const index = workspace_.find_block_index(i, k);
        if (!index)
            return std::nullopt;
        auto const& stored = original(*index, i, k);
        if (!stored.valid)
            return std::nullopt;
        if (j < m)
            return stored.value;
        return stored.value.adjoint();
    }
    void complete(Idx j, Idx m, Result& result) {
        using namespace completion_detail;
        bool warning = false;
        bool candidate{};
        auto const& first = separators_.neighbors(j);
        auto const& second = separators_.neighbors(m);
        auto a_bus = first.begin(), b_bus = second.begin();
        while (a_bus != first.end() && b_bus != second.end()) {
            if (*a_bus < *b_bus) {
                ++a_bus;
                continue;
            }
            if (*b_bus < *a_bus) {
                ++b_bus;
                continue;
            }
            Idx const k = *a_bus;
            ++a_bus;
            ++b_bus;
            if (k == j || k == m)
                continue;
            candidate = true;
            auto const a = workspace_.find_block_index(j, k), b = workspace_.find_block_index(k, k),
                       c = workspace_.find_block_index(k, m);
            if (!a || !b || !c)
                return;
            auto const& b_block = workspace_.augmented_matrix[*b];
            auto const& c_block = workspace_.augmented_matrix[*c];
            RawMatrix const left = workspace_.augmented_matrix[*a].matrix();
            auto& conditioning = middle(k, b_block.matrix());
            if (conditioning.status == MiddleStatus::nonfinite)
                return;
            if (conditioning.status == MiddleStatus::singular)
                continue;
            auto const& solved = solve(conditioning, m, c_block.matrix());
            if (solved.status == SolveStatus::nonfinite)
                return;
            warning = solved.status == SolveStatus::residual;
            // Only the state corner of B_jk h enters the correction; the multiplier
            // entries of the product are not formed.
            RawMatrix reconstructed = RawMatrix::Zero();
            reconstructed.template topLeftCorner<state_size, state_size>() = reconstruct_corner(left, solved);
            PhysicalBlock block;
            block = workspace_.physical_from_state(j, m, workspace_.correct_raw(j, m, reconstructed));
            block.valid = block.valid && consistent(j, m, block.value, &warning);
            if (!block.valid)
                return;
            result.block = block;
            result.diagnostic.middle_bus = k;
            result.diagnostic.route =
                separators_.separates(j, m, k) ? UQPairRoute::separator : UQPairRoute::augmented_path;
            result.diagnostic.approximation_warning = warning;
            return;
        }
        if (!candidate || !marginal(j).valid || !marginal(m).valid)
            return;
        // Accumulate donor paths in their fixed order using ordinary double sums.
        std::array<Accumulator<Scalar>, static_cast<std::size_t>(Matrix::SizeAtCompileTime)> paths{};
        double path_scale = 0.0;
        Idx count{};
        for (Idx k : adjacency_.row(j)) {
            if (!std::ranges::binary_search(adjacency_.row(m), k))
                continue;
            auto const& middle = marginal(k);
            if (!middle.valid)
                return;
            if (!middle.rank)
                continue;
            auto const left = donor(j, k), right = donor(k, m);
            if (!left || !right)
                return;
            Matrix prediction;
            if (!product<Matrix>({*left, middle.whitener, middle.whitener.adjoint(), *right}, prediction))
                return;
            for (Eigen::Index i = 0; i != prediction.size(); ++i)
                paths[static_cast<std::size_t>(i)].add(prediction(i));
            path_scale += frobenius_norm(prediction);
            ++count;
        }
        if (!count)
            return;
        Matrix mean;
        for (Eigen::Index i = 0; i != mean.size(); ++i)
            mean(i) = paths[static_cast<std::size_t>(i)].value() / static_cast<double>(count);
        double const scale = path_scale / static_cast<double>(count);
        result.block.value = mean;
        result.block.construction_scale = scale;
        result.block.valid = std::isfinite(scale) && consistent(j, m, mean, &warning);
        if (!result.block.valid)
            return;
        result.diagnostic.route = (!marginal(j).rank || !marginal(m).rank) && mean.isZero(0) ? UQPairRoute::zero_target
                                                                                             : UQPairRoute::mean_path;
        result.diagnostic.approximation_warning = warning;
    }
    Workspace const& workspace_;
    bool diagnostics_;
    PhysicalBlocks<Workspace>& stored_;
    completion_detail::Separators separators_;
    completion_detail::Graph adjacency_; // Sorted, symmetric physical topology.
    std::vector<std::optional<Marginal>> marginals_;
    std::vector<std::optional<PhysicalBlock>> originals_; // Indexed by LU position.
    // Originally missing pairs, keyed j * size + m.
    completion_detail::PairTable<Result> completed_;
    Middle middle_;
};
} // namespace power_grid_model::math_solver::state_estimation_uq
