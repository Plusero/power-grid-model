// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#include <power_grid_model/math_solver/state_estimation_uq_completion.hpp>

#include <Eigen/LU>
#include <doctest/doctest.h>

#include <complex>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <vector>

namespace power_grid_model::math_solver {
namespace {
using namespace state_estimation_uq;

// Stand-in for the covariance workspace with freely chosen blocks. `raw` holds the blocks of the inverse
// augmented matrix on the pattern, of size N with D state coordinates. `physical` holds the covariance of the
// rectangular voltages that the workspace would return for a pair on the pattern; it can be set
// independently of `raw`. A real state is taken as rectangular itself, a complex one by its real form.
template <class S, int N, int D> struct Fixture {
    static constexpr bool is_complex = Eigen::NumTraits<S>::IsComplex;
    static constexpr int rectangular_size = is_complex ? 2 * D : D;
    using RawMatrix = Eigen::Matrix<S, N, N>;
    using StateMatrix = Eigen::Matrix<S, D, D>;
    using RectangularMatrix = Eigen::Matrix<double, rectangular_size, rectangular_size>;
    struct Raw {
        RawMatrix value;
        RawMatrix const& matrix() const { return value; }
    };
    std::vector<Raw> augmented_matrix; // by block index, as in the workspace
    std::map<std::pair<Idx, Idx>, Idx> block_index;
    std::map<std::pair<Idx, Idx>, RectangularMatrix> physical;
    double correction = 1;
    mutable Idx physical_calls = 0;

    void set(Idx i, Idx j, RectangularMatrix const& value) {
        physical[{i, j}] = value;
        physical[{j, i}] = value.transpose();
    }
    // The raw block (i, j), created on first use.
    Raw& raw(Idx i, Idx j) {
        auto const [it, inserted] = block_index.try_emplace({i, j}, static_cast<Idx>(augmented_matrix.size()));
        if (inserted) {
            augmented_matrix.emplace_back();
        }
        return augmented_matrix[it->second];
    }
    // The block leaves the pattern; its slot stays but is unreachable.
    void erase(Idx i, Idx j) { block_index.erase({i, j}); }

    Idx block_count() const { return static_cast<Idx>(augmented_matrix.size()); }
    Raw const& inverse_block_at(Idx index) const { return augmented_matrix[index]; }
    std::optional<Idx> find_block_index(Idx i, Idx j) const {
        auto it = block_index.find({i, j});
        return it == block_index.end() ? std::nullopt : std::optional<Idx>{it->second};
    }
    double variance_normalization() const { return correction; }
    RectangularMatrix rectangular_covariance(Idx i, Idx j) const {
        ++physical_calls;
        return physical.at({i, j});
    }
    RectangularMatrix rectangular_covariance(Idx /*i*/, Idx /*j*/, StateMatrix const& state) const {
        if constexpr (is_complex) {
            RectangularMatrix result;
            result << state.real() / 2.0, -state.imag() / 2.0, state.imag() / 2.0, state.real() / 2.0;
            return result;
        } else {
            return state;
        }
    }
    // The covariance that belongs to a raw block.
    RectangularMatrix from_raw(RawMatrix const& block) const {
        return rectangular_covariance(0, 0, StateMatrix{correction * block.template topLeftCorner<D, D>()});
    }
    void identity(Idx n, double scale = 1) {
        for (Idx i = 0; i < n; ++i) {
            raw(i, i).value = RawMatrix::Identity();
            set(i, i, RectangularMatrix::Identity() * scale);
        }
    }
    void donor(Idx i, Idx j, double coefficient) {
        raw(i, j).value = RawMatrix::Identity() * coefficient;
        raw(j, i).value = RawMatrix::Identity() * coefficient;
        set(i, j, RectangularMatrix::Identity() * coefficient);
    }
};
using RealFixture = Fixture<double, 4, 2>;
using Graph = std::vector<std::vector<Idx>>;
Graph chain() { return {{1}, {0, 2}, {1, 3}, {2}}; }

template <class S, int N, int D> void dense_separator() {
    Fixture<S, N, D> fixture;
    Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic> information =
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>::Zero(4 * N, 4 * N);
    Graph const graph{{1, 2}, {0}, {0, 3}, {2}};
    std::array<Idx, 4> const path{1, 0, 2, 3};
    for (int i = 0; i < 4; ++i) {
        information.block(i * N, i * N, N, N).setIdentity();
        information.block(i * N, i * N, N, N) *= 4;
        if (i < 3) {
            Eigen::Matrix<S, N, N> edge;
            for (int r = 0; r < N; ++r)
                for (int c = 0; c < N; ++c) {
                    double real = .08 * (r + 1) - .03 * (c + 1) + .05 * i;
                    if constexpr (Eigen::NumTraits<S>::IsComplex)
                        edge(r, c) = S{real, .03 * (r - c + 1)};
                    else
                        edge(r, c) = real;
                }
            information.block(path[i] * N, path[i + 1] * N, N, N) = edge;
            information.block(path[i + 1] * N, path[i] * N, N, N) = edge.adjoint();
        }
    }
    auto inverse = information.inverse().eval(); // Independent dense reference, test only.
    fixture.correction = 0.37;
    for (Idx i = 0; i < 4; ++i)
        for (Idx j = 0; j < 4; ++j)
            if (i == j || std::ranges::find(graph[i], j) != graph[i].end() ||
                std::minmax(i, j) == std::pair<Idx, Idx>{1, 2}) {
                fixture.raw(i, j).value = inverse.block(i * N, j * N, N, N);
                fixture.physical[{i, j}] = fixture.from_raw(fixture.raw(i, j).value);
            }
    // Natural elimination of bus 0 joins its remaining neighbors 1 and 2.
    REQUIRE(std::ranges::find(graph[0], 1) != graph[0].end());
    REQUIRE(std::ranges::find(graph[0], 2) != graph[0].end());
    REQUIRE(std::ranges::find(graph[1], 2) == graph[1].end());
    CovarianceCompletion completion{fixture, graph, graph};
    auto const& result = completion.pair(0, 3);
    CHECK(result.block.valid);
    CHECK(result.diagnostic.route == UQPairRoute::separator);
    CHECK(result.diagnostic.middle_bus == 2); // Common middle in the unfactorized graph.
    Eigen::Matrix<S, N, N> const exact = inverse.block(0, 3 * N, N, N);
    CHECK((result.block.value - fixture.from_raw(exact)).norm() < 1e-15);
    // A repeated query returns the cached completion without another attempt.
    CHECK(completion.pair(0, 3).block.value == result.block.value);
    CHECK(fixture.augmented_matrix.size() == 12);
    // The state corners alone do not give the pair: the multipliers take part in the path.
    auto left = fixture.raw(0, 2).value.template topLeftCorner<D, D>().eval();
    auto middle = fixture.raw(2, 2).value.template topLeftCorner<D, D>().eval();
    auto right = fixture.raw(2, 3).value.template topLeftCorner<D, D>().eval();
    CHECK((left * middle.inverse() * right - inverse.block(0, 3 * N, D, D)).norm() > 1e-9);
}

TEST_CASE("UQ structural articulation labels agree with independent deletion traversal") {
    for (Graph const graph :
         {chain(), Graph{{1, 2}, {0, 3}, {0, 3}, {1, 2}}, Graph{{1}, {0}, {3}, {2}}, Graph{{1}, {0, 2, 3}, {1}, {1}}}) {
        completion_detail::Separators separators{graph};
        auto reachable = [&](Idx source, Idx target, Idx removed) {
            std::set<Idx> visited{source};
            std::vector<Idx> queue{source};
            for (std::size_t cursor = 0; cursor < queue.size(); ++cursor)
                for (Idx next : graph[queue[cursor]])
                    if (next != removed && visited.insert(next).second)
                        queue.push_back(next);
            return visited.contains(target);
        };
        for (Idx j = 0; j < 4; ++j)
            for (Idx m = j + 1; m < 4; ++m)
                for (Idx k = 0; k < 4; ++k)
                    CHECK(separators.separates(j, m, k) ==
                          (j != k && m != k && reachable(j, m, -1) && !reachable(j, m, k)));
    }
    Graph radial(100000);
    for (Idx i = 1; i < 100000; ++i) {
        radial[i - 1].push_back(i);
        radial[i].push_back(i - 1);
    }
    completion_detail::Separators deep{std::move(radial)};
    CHECK(deep.separates(0, 99999, 50000));
}

TEST_CASE("UQ articulation labels match deletion reachability on seeded meshed graphs") {
    std::mt19937 engine{204191};
    for (int trial = 0; trial < 40; ++trial) {
        Idx const n = 3 + trial % 8;
        Graph graph(static_cast<std::size_t>(n));
        for (Idx i = 0; i < n; ++i)
            for (Idx j = i + 1; j < n; ++j)
                if (engine() % 4 == 0) {
                    graph[i].push_back(j);
                    graph[j].push_back(i);
                }
        completion_detail::Separators separators{graph};
        auto reachable = [&](Idx source, Idx target, Idx removed) {
            std::set<Idx> visited{source};
            std::vector<Idx> queue{source};
            for (std::size_t cursor = 0; cursor < queue.size(); ++cursor)
                for (Idx next : graph[queue[cursor]])
                    if (next != removed && visited.insert(next).second)
                        queue.push_back(next);
            return visited.contains(target);
        };
        for (Idx j = 0; j < n; ++j)
            for (Idx m = j + 1; m < n; ++m)
                for (Idx k = 0; k < n; ++k)
                    CHECK(separators.separates(j, m, k) ==
                          (j != k && m != k && reachable(j, m, -1) && !reachable(j, m, k)));
    }
}

TEST_CASE("UQ complete augmented separator agrees with independent dense inverses") {
    dense_separator<std::complex<double>, 2, 1>();
    dense_separator<double, 4, 2>();
}

Graph diamond() { return {{2, 1}, {3, 0}, {3, 0}, {2, 1}}; }
RealFixture diamond_fixture() {
    RealFixture f;
    f.identity(4);
    f.donor(0, 1, .2);
    f.donor(1, 3, .3);
    f.donor(0, 2, .4);
    f.donor(2, 3, .5);
    return f;
}

TEST_CASE("UQ completion derives marginals and stored pairs only on first use") {
    auto f = diamond_fixture();
    CovarianceCompletion c{f, diamond(), diamond()};
    CHECK(f.physical_calls == 0);
    // The pair and the covariance of its two buses.
    CHECK(c.pair(0, 1).diagnostic.route == UQPairRoute::stored);
    CHECK(f.physical_calls == 3);
    CHECK(c.pair(0, 1).block.value(0, 0) == doctest::Approx(.2));
    CHECK(c.diagonal(1).valid);
    CHECK(f.physical_calls == 3);
    // The augmented path adds only the covariance of the new bus.
    CHECK(c.pair(0, 3).block.valid);
    CHECK(f.physical_calls == 4);
    CHECK(c.pair(0, 3).block.valid);
    CHECK(f.physical_calls == 4);
}

TEST_CASE("UQ ascending common middles retain first conditioned nonseparator") {
    auto f = diamond_fixture();
    f.raw(2, 2).value.setConstant(std::numeric_limits<double>::quiet_NaN());
    CovarianceCompletion c{f, diamond(), diamond()};
    auto const& r = c.pair(0, 3);
    CHECK(r.block.valid);
    CHECK(r.diagnostic.route == UQPairRoute::augmented_path);
    CHECK(r.diagnostic.middle_bus == 1);
    CHECK(r.block.value(0, 0) == doctest::Approx(.06));
    CHECK_FALSE(r.diagnostic.approximation_warning); // Nonseparation alone does not warn.
    CHECK(c.pair(0, 3).diagnostic.middle_bus == r.diagnostic.middle_bus);
    CHECK(f.augmented_matrix.size() == 12);
}

TEST_CASE("UQ conditioning alone advances and threshold equality passes") {
    for (double ratio : {0., 1e-13, 1e-12}) {
        auto f = diamond_fixture();
        f.raw(1, 1).value(3, 3) = ratio;
        CovarianceCompletion c{f, diamond(), diamond()};
        auto const& r = c.pair(0, 3);
        CHECK(r.block.valid);
        CHECK(r.diagnostic.middle_bus == (ratio == 1e-12 ? 1 : 2));
    }
}

TEST_CASE("UQ missing or nonfinite prerequisites stop without fallback") {
    for (int failure = 0; failure < 4; ++failure) {
        auto f = diamond_fixture();
        if (failure == 0)
            f.erase(0, 1);
        if (failure == 1)
            f.raw(1, 1).value(0, 0) = std::numeric_limits<double>::quiet_NaN();
        if (failure == 2)
            f.raw(1, 3).value(0, 0) = std::numeric_limits<double>::infinity();
        if (failure == 3)
            f.raw(0, 1).value(0, 0) = std::numeric_limits<double>::quiet_NaN();
        CovarianceCompletion c{f, diamond(), diamond()};
        CHECK_FALSE(c.pair(0, 3).block.valid);
        CHECK(c.pair(0, 3).diagnostic.route == UQPairRoute::invalid);
        CHECK(c.pair(0, 3).diagnostic.middle_bus == -1);
    }
    auto f = diamond_fixture();
    CovarianceCompletion no_common{f, chain(), diamond()};
    CHECK_FALSE(no_common.pair(0, 3).block.valid); // Fill donors do not create adjacency.
}

TEST_CASE("UQ overflowing arithmetic stops without fallback") {
    for (int failure = 0; failure < 2; ++failure) {
        auto f = diamond_fixture();
        if (failure == 0)
            f.raw(1, 1).value *= 1e-310; // The middle solve overflows.
        if (failure == 1) {
            f.raw(0, 1).value *= 1e308; // The reconstruction overflows.
            f.raw(1, 3).value *= 100;
        }
        CovarianceCompletion c{f, diamond(), diamond()};
        CHECK_FALSE(c.pair(0, 3).block.valid);
        CHECK(c.pair(0, 3).diagnostic.route == UQPairRoute::invalid);
        CHECK(c.pair(0, 3).diagnostic.middle_bus == -1);
    }
}

TEST_CASE("UQ an overflowing normalization stops without another middle") {
    auto f = diamond_fixture();
    f.raw(0, 1).value *= 1e4;
    f.correction = 1e308;
    CovarianceCompletion c{f, diamond(), diamond()};
    CHECK_FALSE(c.pair(0, 3).block.valid);
    CHECK(c.pair(0, 3).diagnostic.route == UQPairRoute::invalid);
    CHECK(c.pair(0, 3).diagnostic.middle_bus == -1);
}

TEST_CASE("UQ finite correlation and unsupported augmented blocks are retained") {
    for (bool zero : {false, true}) {
        auto f = diamond_fixture();
        f.raw(0, 1).value *= 100;
        if (zero)
            f.set(0, 0, RealFixture::RectangularMatrix::Zero());
        CovarianceCompletion c{f, diamond(), diamond()};
        auto const& r = c.pair(0, 3);
        CHECK(r.block.valid);
        CHECK(r.diagnostic.route == UQPairRoute::augmented_path);
        CHECK(r.diagnostic.middle_bus == 1);
        CHECK(r.block.value(0, 0) == doctest::Approx(6.));
        // Only the correlation threshold warns; the unsupported zero marginal does not.
        CHECK(r.diagnostic.approximation_warning == !zero);
    }
}

TEST_CASE("UQ solve residual warning retains the conditioned candidate") {
    // A conditioned middle 1e-10 I with a multiplier entry 1e300 of B_13: the
    // diagnostic multiplier column of h overflows, the state columns do not.
    auto f = diamond_fixture();
    f.raw(1, 1).value *= 1e-10;
    f.raw(0, 1).value *= 1e-10;
    f.raw(1, 3).value(3, 3) = 1e300;
    for (bool const diagnostics : {false, true}) {
        CovarianceCompletion<RealFixture> c{f, diamond(), diamond(), diagnostics};
        auto const& r = c.pair(0, 3);
        REQUIRE(r.block.valid);
        CHECK(r.diagnostic.route == UQPairRoute::augmented_path);
        CHECK(r.diagnostic.middle_bus == 1);
        CHECK(r.block.value(0, 0) == doctest::Approx(.06));
        CHECK(r.diagnostic.approximation_warning == diagnostics);
    }
}

TEST_CASE("UQ terminal mean uses original physical paths including zero products") {
    auto f = diamond_fixture();
    f.raw(1, 1).value.setZero();
    f.raw(2, 2).value.setZero();
    f.set(0, 1, RealFixture::RectangularMatrix::Zero());
    CovarianceCompletion c{f, diamond(), diamond()};
    auto const& r = c.pair(0, 3);
    CHECK(r.block.valid);
    CHECK(r.diagnostic.route == UQPairRoute::mean_path);
    CHECK(r.diagnostic.middle_bus == -1);
    CHECK(r.block.value(0, 0) == doctest::Approx(.1)); // Mean of the zero and the .4 * .5 path.
    CHECK(f.augmented_matrix.size() == 12);
    auto const& other = c.pair(1, 2);
    CHECK(other.block.valid);
    CHECK(other.diagnostic.route == UQPairRoute::augmented_path);
    CHECK(other.block.value(0, 0) == doctest::Approx(.08)); // Completed pair never becomes donor.
}

TEST_CASE("UQ terminal empty invalid and unsupported middle pools remain invalid") {
    for (int failure = 0; failure < 4; ++failure) {
        auto f = diamond_fixture();
        f.raw(1, 1).value.setZero();
        f.raw(2, 2).value.setZero();
        Graph physical = diamond();
        if (failure == 0)
            physical = Graph(4);
        if (failure == 1)
            f.erase(0, 1);
        if (failure == 2)
            f.set(1, 1, -RealFixture::RectangularMatrix::Identity());
        if (failure == 3) {
            f.set(1, 1, RealFixture::RectangularMatrix::Zero());
            f.set(2, 2, RealFixture::RectangularMatrix::Zero());
        }
        CovarianceCompletion c{f, diamond(), physical};
        CHECK_FALSE(c.pair(0, 3).block.valid);
        CHECK(c.pair(0, 3).diagnostic.route == UQPairRoute::invalid);
    }
}

TEST_CASE("UQ terminal overflowing products are invalid") {
    auto f = diamond_fixture();
    f.raw(1, 1).value.setZero();
    f.raw(2, 2).value.setZero();
    f.set(0, 1, RealFixture::RectangularMatrix::Identity() * 1e200);
    f.set(1, 3, RealFixture::RectangularMatrix::Identity() * 1e200);
    CovarianceCompletion c{f, diamond(), diamond()};
    CHECK_FALSE(c.pair(0, 3).block.valid);
    CHECK(c.pair(0, 3).diagnostic.route == UQPairRoute::invalid);
}

TEST_CASE("UQ terminal rank zero middle exclusion and algebraic zero target") {
    auto f = diamond_fixture();
    f.raw(1, 1).value.setZero();
    f.raw(2, 2).value.setZero();
    f.set(1, 1, RealFixture::RectangularMatrix::Zero());
    CovarianceCompletion supported{f, diamond(), diamond()};
    CHECK(supported.pair(0, 3).diagnostic.route == UQPairRoute::mean_path);
    CHECK(supported.pair(0, 3).block.value(0, 0) == doctest::Approx(.2)); // Only the path through 2.
    f.set(0, 0, RealFixture::RectangularMatrix::Zero());
    f.set(0, 2, RealFixture::RectangularMatrix::Zero());
    CovarianceCompletion zero{f, diamond(), diamond()};
    CHECK(zero.pair(0, 3).block.valid);
    CHECK(zero.pair(0, 3).diagnostic.route == UQPairRoute::zero_target);
    f.set(0, 2, RealFixture::RectangularMatrix::Identity() * .4);
    CovarianceCompletion nonzero{f, diamond(), diamond()};
    CHECK(nonzero.pair(0, 3).block.valid);
    CHECK(nonzero.pair(0, 3).diagnostic.route == UQPairRoute::mean_path);
    CHECK(nonzero.pair(0, 3).block.value(0, 0) == doctest::Approx(.2)); // Retained outside the target support.
    f.set(0, 2, RealFixture::RectangularMatrix::Identity() * 1e-15);
    CovarianceCompletion tiny{f, diamond(), diamond()};
    CHECK(tiny.pair(0, 3).block.valid);
    CHECK(tiny.pair(0, 3).diagnostic.route == UQPairRoute::mean_path);
    CHECK(tiny.pair(0, 3).block.value(0, 0) != 0);
}

TEST_CASE("UQ terminal finite correlation diagnostics retain the mean") {
    auto f = diamond_fixture();
    f.raw(1, 1).value.setZero();
    f.raw(2, 2).value.setZero();
    f.set(0, 1, RealFixture::RectangularMatrix::Identity() * 2);
    f.set(1, 3, RealFixture::RectangularMatrix::Identity() * 2);
    f.set(0, 2, RealFixture::RectangularMatrix::Identity() * 2);
    f.set(2, 3, RealFixture::RectangularMatrix::Identity() * -2);
    CovarianceCompletion cancel{f, diamond(), diamond()};
    CHECK(cancel.pair(0, 3).block.valid);
    CHECK(cancel.pair(0, 3).block.value.isZero());
    f.set(2, 3, RealFixture::RectangularMatrix::Identity() * 2);
    CovarianceCompletion bound{f, diamond(), diamond()};
    CHECK(bound.pair(0, 3).block.valid);
    CHECK(bound.pair(0, 3).diagnostic.route == UQPairRoute::mean_path);
    CHECK(bound.pair(0, 3).block.value(0, 0) == 4);
    CHECK(bound.pair(0, 3).diagnostic.approximation_warning);
    CHECK_FALSE(cancel.pair(0, 3).diagnostic.approximation_warning);
}

template <class S, int N, int D> void dense_nonseparator() {
    Fixture<S, N, D> f;
    using Dense = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
    Dense a = Dense::Identity(4 * N, 4 * N) * 4.;
    auto graph = diamond();
    for (Idx j = 0; j < 4; ++j)
        for (Idx k : graph[j])
            a.block(j * N, k * N, N, N) = Eigen::Matrix<S, N, N>::Identity() * .3;
    Dense inverse = a.inverse();
    for (Idx j = 0; j < 4; ++j)
        for (Idx k = 0; k < 4; ++k)
            if (j == k || std::ranges::find(graph[j], k) != graph[j].end()) {
                f.raw(j, k).value = inverse.block(j * N, k * N, N, N);
                f.physical[{j, k}] = f.from_raw(f.raw(j, k).value);
            }
    CovarianceCompletion c{f, graph, graph};
    auto const& r = c.pair(0, 3);
    Eigen::Matrix<S, N, N> const expected =
        inverse.block(0, N, N, N) * inverse.block(N, N, N, N).inverse() * inverse.block(N, 3 * N, N, N);
    Eigen::Matrix<S, N, N> const exact = inverse.block(0, 3 * N, N, N);
    CHECK(r.block.valid);
    CHECK(r.diagnostic.route == UQPairRoute::augmented_path);
    CHECK((r.block.value - f.from_raw(expected)).norm() < 1e-15);
    CHECK((r.block.value - f.from_raw(exact)).norm() > 1e-4);
}
TEST_CASE("UQ nonseparator arithmetic agrees with dense formula and has explicit approximation error") {
    dense_nonseparator<std::complex<double>, 2, 1>();
    dense_nonseparator<double, 4, 2>();
}

TEST_CASE("UQ physical marginal units principal roots and strict support boundary") {
    for (double scale : {1e-18, 1., 1e18}) {
        auto f = diamond_fixture();
        f.identity(4, scale);
        f.raw(1, 1).value.setZero();
        f.raw(2, 2).value.setZero();
        for (Idx j : {1, 2}) {
            f.set(0, j, RealFixture::RectangularMatrix::Identity() * (.25 * scale));
            f.set(j, 3, RealFixture::RectangularMatrix::Identity() * (.5 * scale));
        }
        CovarianceCompletion c{f, diamond(), diamond()};
        CHECK(c.pair(0, 3).block.valid);
        CHECK(c.pair(0, 3).block.value(0, 0) / scale == doctest::Approx(.125));
    }
    auto f = diamond_fixture();
    auto d = RealFixture::RectangularMatrix::Identity().eval();
    d(1, 1) = static_cast<double>(round_rtol);
    f.set(0, 0, d);
    f.raw(0, 0).value.setZero();
    f.raw(3, 3).value.setZero();
    CovarianceCompletion boundary{f, diamond(), diamond()};
    CHECK(boundary.diagonal(0).valid);
    CHECK(boundary.diagonal(0).value(1, 1) == d(1, 1));
    CHECK(boundary.pair(0, 1).block.valid);
    // The terminal mean of pair (1, 2) through middles 0 and 3: the boundary
    // direction of marginal 0 is unsupported, so its path contributes zero there.
    auto const& mean = boundary.pair(1, 2);
    REQUIRE(mean.block.valid);
    CHECK(mean.diagnostic.route == UQPairRoute::mean_path);
    CHECK(mean.block.value(0, 0) == doctest::Approx((.2 * .4 + .3 * .5) / 2));
    CHECK(mean.block.value(1, 1) == doctest::Approx(.3 * .5 / 2));
    d(1, 1) = std::nextafter(d(1, 1), 1.);
    f.set(0, 0, d);
    auto cross = RealFixture::RectangularMatrix::Zero().eval();
    cross(0, 0) = .2;
    cross(1, 1) = .2 * std::sqrt(d(1, 1));
    f.set(0, 1, cross);
    CovarianceCompletion supported{f, diamond(), diamond()};
    CHECK(supported.pair(0, 1).block.valid);
    CHECK(supported.pair(0, 1).diagnostic.route == UQPairRoute::stored);
    f.set(0, 0, -RealFixture::RectangularMatrix::Identity());
    CovarianceCompletion invalid{f, diamond(), diamond()};
    CHECK_FALSE(invalid.pair(0, 3).block.valid);
    CHECK(invalid.pair(0, 3).diagnostic.route == UQPairRoute::invalid);
}
TEST_CASE("UQ completion symmetrizes a bus covariance without relaxing negative eigenvalues") {
    RealFixture fixture;
    fixture.identity(2);
    fixture.donor(0, 1, 0.25);
    fixture.physical[{0, 0}](0, 1) = 1e-10;
    std::vector<std::vector<Idx>> const graph{{1}, {0}};
    CovarianceCompletion corrected{fixture, graph, graph};
    CHECK(corrected.diagonal(0).valid);
    CHECK(corrected.pair(0, 1).block.valid);
    CHECK(corrected.diagonal(0).value(0, 1) == corrected.diagonal(0).value(1, 0));

    fixture.physical[{0, 0}](0, 0) = -1e-9;
    CovarianceCompletion negative{fixture, graph, graph};
    CHECK_FALSE(negative.diagonal(0).valid);
    CHECK_FALSE(negative.pair(0, 1).block.valid);
}

TEST_CASE("UQ completion reuses middles without changing results") {
    // Two adjacent star middles, 0 over {1, 2, 5} and 5 over {0, 3, 4}.
    constexpr int N = 4;
    Graph const graph{{1, 2, 5}, {0}, {0}, {5}, {5}, {0, 3, 4}};
    Eigen::MatrixXd information = Eigen::MatrixXd::Identity(6 * N, 6 * N) * 4;
    for (Idx i = 0; i < 6; ++i)
        for (Idx j : graph[i])
            if (i < j)
                for (int r = 0; r < N; ++r)
                    for (int c = 0; c < N; ++c)
                        information(i * N + r, j * N + c) = information(j * N + c, i * N + r) =
                            .07 * (r + 1) - .04 * (c + 1) + .03 * (i + j);
    Eigen::MatrixXd const inverse = information.inverse(); // Independent dense reference, test only.
    RealFixture fixture;
    fixture.correction = 0.41;
    for (Idx i = 0; i < 6; ++i)
        for (Idx j = 0; j < 6; ++j)
            if (i == j || std::ranges::find(graph[i], j) != graph[i].end()) {
                fixture.raw(i, j).value = inverse.block(i * N, j * N, N, N);
                fixture.physical[{i, j}] = fixture.from_raw(fixture.raw(i, j).value);
            }
    // Interleave the middles so the most recent middle is replaced and recomputed.
    std::vector<std::pair<Idx, Idx>> const pairs{{1, 2}, {0, 3}, {1, 5}, {3, 4}, {2, 5}, {0, 4}, {1, 2}};
    CovarianceCompletion together{fixture, graph, graph};
    for (auto const& [j, m] : pairs) {
        auto const result = together.pair(j, m);
        CovarianceCompletion fresh{fixture, graph, graph};
        auto const expected = fresh.pair(j, m);
        REQUIRE(expected.block.valid);
        CHECK(result.block.valid);
        CHECK((result.block.value.array() == expected.block.value.array()).all());
        CHECK(result.diagnostic.route == expected.diagnostic.route);
        CHECK(result.diagnostic.middle_bus == expected.diagnostic.middle_bus);
        CHECK(result.diagnostic.approximation_warning == expected.diagnostic.approximation_warning);
        CHECK((result.block.value - fixture.correction * inverse.block(j * N, m * N, 2, 2)).norm() < 1e-14);
    }
}

TEST_CASE("UQ pair table keeps every completed pair at a stable address") {
    completion_detail::PairTable<std::pair<Idx, double>> table;
    CHECK(table.find(0) == nullptr);
    std::mt19937_64 engine{77};
    std::vector<Idx> keys;
    std::set<Idx> unique;
    while (unique.size() < 3000) {
        Idx const key = static_cast<Idx>(engine() % 1000000000);
        if (unique.insert(key).second)
            keys.push_back(key);
    }
    std::vector<std::pair<Idx, double> const*> addresses;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        CHECK(table.find(keys[i]) == nullptr);
        addresses.push_back(&table.insert(keys[i], {keys[i], 0.5 * static_cast<double>(i)}));
    }
    for (std::size_t i = 0; i < keys.size(); ++i) {
        auto const* found = table.find(keys[i]);
        REQUIRE(found != nullptr);
        CHECK(found == addresses[i]);
        CHECK(found->first == keys[i]);
        CHECK(found->second == 0.5 * static_cast<double>(i));
    }
    for (Idx key = 0; key < 2000; ++key)
        if (!unique.contains(key))
            CHECK(table.find(key) == nullptr);
}

TEST_CASE("UQ compressed graphs keep sorted unique rows and separators ignore own buses") {
    auto const rows = completion_detail::compress({{2, 1, 2}, {}, {0, 0}});
    REQUIRE(rows.size() == 3);
    CHECK(std::ranges::equal(rows.row(0), std::vector<Idx>{1, 2}));
    CHECK(rows.row(1).empty());
    CHECK(std::ranges::equal(rows.row(2), std::vector<Idx>{0}));
    std::vector<std::array<Idx, 2>> const edges{{0, 1}, {1, 0}, {2, 2}, {1, 3}, {-1, 0}, {0, 4}};
    auto const physical = completion_detail::undirected(4, edges);
    REQUIRE(physical.size() == 4);
    CHECK(std::ranges::equal(physical.row(0), std::vector<Idx>{1}));
    CHECK(std::ranges::equal(physical.row(1), std::vector<Idx>{0, 3}));
    CHECK(physical.row(2).empty());
    CHECK(std::ranges::equal(physical.row(3), std::vector<Idx>{1}));

    std::mt19937 engine{4242};
    for (int trial = 0; trial < 30; ++trial) {
        Idx const n = 3 + trial % 7;
        Graph graph(n), with_own(n);
        for (Idx i = 0; i < n; ++i)
            for (Idx j = i + 1; j < n; ++j)
                if (engine() % 3 == 0) {
                    graph[i].push_back(j);
                    graph[j].push_back(i);
                }
        for (Idx i = 0; i < n; ++i) {
            with_own[i] = graph[i];
            if (engine() % 2 == 0)
                with_own[i].push_back(i);
        }
        completion_detail::Separators const plain{graph}, own{with_own};
        for (Idx j = 0; j < n; ++j)
            for (Idx m = 0; m < n; ++m)
                for (Idx k = 0; k < n; ++k)
                    CHECK(plain.separates(j, m, k) == own.separates(j, m, k));
    }
}
} // namespace
} // namespace power_grid_model::math_solver
