// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
// SPDX-License-Identifier: MPL-2.0

#include <power_grid_model/math_solver/state_estimation_uq_injection.hpp>

#include <Eigen/LU>
#include <doctest/doctest.h>

#include <map>
#include <set>

namespace power_grid_model::math_solver {
namespace {
using namespace state_estimation_uq;
using Pair = std::pair<Idx, Idx>;

template <bool complex> struct CoupledCovariance {
    using Scalar = std::conditional_t<complex, DoubleComplex, double>;
    static constexpr int states = complex ? 3 : 6;
    static constexpr int raw_size = 2 * states;
    using RawMatrix = Eigen::Matrix<Scalar, raw_size, raw_size>;
    using StateMatrix = Eigen::Matrix<Scalar, states, states>;
    using PhysicalMatrix = StateMatrix;
    using StateBlock = CovarianceBlock<StateMatrix>;
    using PhysicalBlock = StateBlock;
    struct Raw {
        RawMatrix value;
        auto const& matrix() const { return value; }
    };
    std::vector<Raw> augmented_matrix; // Indexed by position in col_indices, as in the workspace.
    std::map<Pair, PhysicalBlock> override_physical;
    IdxVector row_indptr{0}, col_indices;
    ComplexValueVector<asymmetric_t> u;
    std::vector<Eigen::Matrix<Scalar, states, 1>> slack_column;
    double variance_normalization = .7;
    Idx slack_bus = 2;
    bool report_relative_angle = false;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> inverse;

    DoubleComplex phase_voltage(Idx bus, int phase) const { return u[bus](phase); }

    explicit CoupledCovariance(int buses = 3) {
        for (int bus = 0; bus != buses; ++bus) {
            ComplexValue<asymmetric_t> voltage;
            for (int p = 0; p != 3; ++p)
                voltage(p) = std::polar(.8 + .12 * bus + .09 * p, .2 * bus - 2.0 * p);
            u.push_back(voltage);
        }
        // A full augmented block precision on 0--1--2, with phase/state-multiplier
        // coupling. The missing endpoint pair has the exact raw separator rule.
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> information =
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(buses * raw_size, buses * raw_size) * 6;
        for (int bus = 0; bus != buses - 1; ++bus) {
            RawMatrix edge;
            for (int r = 0; r != raw_size; ++r)
                for (int c = 0; c != raw_size; ++c) {
                    double const real = .025 * (r + 1) - .018 * (c + 1) + .03 * bus;
                    if constexpr (complex)
                        edge(r, c) = {real, .013 * (r - c + 1)};
                    else
                        edge(r, c) = real;
                }
            information.block(bus * raw_size, (bus + 1) * raw_size, raw_size, raw_size) = edge;
            information.block((bus + 1) * raw_size, bus * raw_size, raw_size, raw_size) = edge.adjoint();
        }
        inverse = information.inverse();
        for (Idx i = 0; i != buses; ++i) {
            for (Idx j = 0; j != buses; ++j)
                if (std::abs(i - j) <= 1) {
                    augmented_matrix.push_back({inverse.block(i * raw_size, j * raw_size, raw_size, raw_size)});
                    col_indices.push_back(j);
                }
            row_indptr.push_back(static_cast<Idx>(col_indices.size()));
            slack_column.push_back(inverse.template block<states, 1>(i * raw_size, slack_bus * raw_size));
        }
    }
    std::optional<Idx> find_block_index(Idx i, Idx j) const {
        for (Idx p = row_indptr[i]; p != row_indptr[i + 1]; ++p)
            if (col_indices[p] == j)
                return p;
        return std::nullopt;
    }
    Raw& raw(Idx i, Idx j) { return augmented_matrix[*find_block_index(i, j)]; }
    Raw const& raw(Idx i, Idx j) const { return augmented_matrix[*find_block_index(i, j)]; }
    StateBlock correct_raw(Idx, Idx, RawMatrix const& value) const {
        StateMatrix const physical = variance_normalization * value.template topLeftCorner<states, states>();
        return {physical, physical.norm(), true};
    }
    StateBlock calculate_state_covariance_block(Idx i, Idx j) const { return correct_raw(i, j, raw(i, j).value); }
    PhysicalBlock physical_from_state(Idx, Idx, StateBlock const& value) const { return value; }
    PhysicalBlock physical_block(Idx i, Idx j) const {
        if (auto const it = override_physical.find({i, j}); it != override_physical.end())
            return it->second;
        auto const index = find_block_index(i, j);
        return index ? correct_raw(i, j, augmented_matrix[*index].value)
                     : PhysicalBlock{StateMatrix::Zero(), 0., false};
    }
    Eigen::Matrix<double, 18, 18> rectangular() const {
        Eigen::Matrix<double, 18, 18> result;
        for (int i = 0; i != 3; ++i)
            for (int j = 0; j != 3; ++j) {
                StateMatrix const block =
                    variance_normalization * inverse.template block<states, states>(i * raw_size, j * raw_size);
                if constexpr (complex) {
                    result.template block<3, 3>(6 * i, 6 * j) = block.real() / 2.;
                    result.template block<3, 3>(6 * i, 6 * j + 3) = -block.imag() / 2.;
                    result.template block<3, 3>(6 * i + 3, 6 * j) = block.imag() / 2.;
                    result.template block<3, 3>(6 * i + 3, 6 * j + 3) = block.real() / 2.;
                } else
                    result.template block<6, 6>(6 * i, 6 * j) = block;
            }
        return result;
    }
};

MathModelTopology coupled_topology() {
    MathModelTopology topology;
    topology.slack_bus = 2;
    topology.phase_shift.resize(3, 0.);
    topology.branch_bus_idx = {{0, 1}, {1, 2}};
    topology.shunts_per_bus = {from_sparse, {0, 0, 1, 1}};
    return topology;
}
MathModelParam<asymmetric_t> coupled_parameters() {
    MathModelParam<asymmetric_t> parameters;
    parameters.branch_param.resize(2);
    for (int branch = 0; branch != 2; ++branch)
        for (int side = 0; side != 4; ++side)
            for (int r = 0; r != 3; ++r)
                for (int c = 0; c != 3; ++c)
                    parameters.branch_param[branch].value[side](r, c) =
                        DoubleComplex{(side == 0 || side == 3 ? 1. : -1.) * (r == c ? 1.8 + .2 * r : .13 * (r + c + 1)),
                                      .11 * (side + 1) - (r == c ? 2.4 : .09 * (r + c + 1))};
    ComplexTensor<asymmetric_t> shunt;
    shunt << DoubleComplex{.2, -.1}, DoubleComplex{.07, .02}, DoubleComplex{-.04, .03}, DoubleComplex{.07, .02},
        DoubleComplex{.3, -.2}, DoubleComplex{.05, -.06}, DoubleComplex{-.04, .03}, DoubleComplex{.05, -.06},
        DoubleComplex{.4, -.3};
    parameters.shunt_param.push_back(shunt);
    return parameters;
}

struct CoupledMeasurements {
    bool constrained = false;
    PowerSensorCalcParam<asymmetric_t> injection{
        {RealValue<asymmetric_t>{0., 0., 1.}, RealValue<asymmetric_t>{0., .2, 0.}},
        {RealValue<asymmetric_t>{0., 0., 0.}, RealValue<asymmetric_t>{0., .3, 0.}}};
    bool has_bus_injection(Idx) const { return constrained; }
    auto const& bus_injection(Idx) const { return injection; }
};

template <class Function> auto finite_difference(ComplexValueVector<asymmetric_t> const& u, Function&& evaluate) {
    Eigen::Matrix<double, 1, 18> row;
    constexpr double step = 1e-6;
    for (int bus = 0; bus != 3; ++bus)
        for (int a = 0; a != 2; ++a)
            for (int phase = 0; phase != 3; ++phase) {
                auto plus = u, minus = u;
                DoubleComplex const delta = a == 0 ? DoubleComplex{step, 0.} : DoubleComplex{0., step};
                plus[bus](phase) += delta;
                minus[bus](phase) -= delta;
                row(6 * bus + 3 * a + phase) = (evaluate(plus) - evaluate(minus)) / (2 * step);
            }
    return row;
}

template <bool complex> void check_coupled_propagation(bool relative) {
    CoupledCovariance<complex> covariance;
    covariance.report_relative_angle = relative;
    auto const topology = coupled_topology();
    auto const parameters = coupled_parameters();
    YBus<asymmetric_t> const y_bus{topology, parameters};
    auto const dense = covariance.rectangular();
    for (BranchSide side : {BranchSide::from, BranchSide::to}) {
        int const terminal = side == BranchSide::from ? 0 : 1;
        StateEstimationUQRequest request;
        request.branch = {{0, side, true, true, true, true}};
        StateEstimationUQResult<asymmetric_t> result;
        PhysicalBlocks stored{covariance};
        branch_results(covariance, y_bus, request, result, stored);
        for (int phase = 0; phase != 3; ++phase) {
            auto const& output = result.branch[0];
            std::array<UQPhaseValue, 4> const values{
                output.active_power.phase[phase], output.reactive_power.phase[phase],
                output.current_magnitude.phase[phase], output.current_angle.phase[phase]};
            for (int quantity = 0; quantity != 4; ++quantity) {
                auto evaluate = [&](auto const& u) {
                    DoubleComplex current{};
                    for (int p = 0; p != 3; ++p) {
                        auto const& param = parameters.branch_param[0];
                        current += (terminal == 0 ? param.yff()(phase, p) : param.ytf()(phase, p)) * u[0](p);
                        current += (terminal == 0 ? param.yft()(phase, p) : param.ytt()(phase, p)) * u[1](p);
                    }
                    DoubleComplex const power = u[terminal](phase) * std::conj(current);
                    return quantity == 0   ? power.real()
                           : quantity == 1 ? power.imag()
                           : quantity == 2 ? std::abs(current)
                                           : std::arg(current) - (relative ? std::arg(u[2](0)) : 0.);
                };
                auto const row = finite_difference(covariance.u, evaluate);
                double const expected = (row * dense * row.transpose())(0, 0);
                CHECK(values[quantity].raw_status == UQVarianceStatus::valid);
                CHECK(values[quantity].raw_variance == doctest::Approx(expected).epsilon(2e-8));
            }
        }
    }
    StateEstimationUQRequest request;
    request.injection = {{1, true, true}};
    request.pair_diagnostics = true;
    StateEstimationUQResult<asymmetric_t> result;
    CoupledMeasurements measured;
    PhysicalBlocks result_stored{covariance};
    injection_results(covariance, y_bus, measured, request, result, result_stored);
    REQUIRE(result.pairs.size() == 3);
    auto const pair = std::ranges::find_if(
        result.pairs, [](auto const& item) { return item.first_bus == 0 && item.second_bus == 2; });
    REQUIRE(pair != result.pairs.end());
    CHECK(pair->route == UQPairRoute::separator);
    CHECK(pair->middle_bus == 1);
    for (int phase = 0; phase != 3; ++phase)
        for (bool reactive : {false, true}) {
            auto evaluate = [&](auto const& u) {
                DoubleComplex current{};
                for (Idx p = y_bus.row_indptr()[1]; p != y_bus.row_indptr()[2]; ++p)
                    for (int q = 0; q != 3; ++q)
                        current += y_bus.admittance()[p](phase, q) * u[y_bus.col_indices()[p]](q);
                auto const power = u[1](phase) * std::conj(current);
                return reactive ? power.imag() : power.real();
            };
            auto const row = finite_difference(covariance.u, evaluate);
            double const expected = (row * dense * row.transpose())(0, 0);
            auto const& value = reactive ? result.injection[0].reactive_power.phase[phase]
                                         : result.injection[0].active_power.phase[phase];
            CHECK(value.raw_status == UQVarianceStatus::valid);
            CHECK(value.raw_variance == doctest::Approx(expected).epsilon(2e-8));
        }
    measured.constrained = true;
    auto const before = result.injection[0];
    StateEstimationUQResult<asymmetric_t> constrained;
    PhysicalBlocks constrained_stored{covariance};
    injection_results(covariance, y_bus, measured, request, constrained, constrained_stored);
    for (int phase = 0; phase != 3; ++phase) {
        auto const& value = constrained.injection[0].active_power.phase[phase];
        CHECK(value.exact_zero_enforced == (phase == 0));
        CHECK(value.raw_variance == before.active_power.phase[phase].raw_variance);
        CHECK(value.raw_status == before.active_power.phase[phase].raw_status);
        if (phase == 0)
            CHECK(value.sigma == 0.);
        else
            CHECK(value.sigma > 0.);
    }
}

TEST_CASE("Asymmetric UQ coupled endpoint and neighborhood contractions match independent real Jacobians") {
    check_coupled_propagation<true>(false);
    check_coupled_propagation<true>(true);
    check_coupled_propagation<false>(false);
}

template <bool complex> void check_singular_support_completion() {
    CoupledCovariance<complex> covariance;
    using Matrix = typename CoupledCovariance<complex>::StateMatrix;
    using Scalar = typename Matrix::Scalar;
    constexpr int states = Matrix::RowsAtCompileTime;
    std::array<Eigen::Matrix<Scalar, states, 1>, 3> vectors;
    for (int bus = 0; bus != 3; ++bus) {
        for (int p = 0; p != states; ++p) {
            if constexpr (complex)
                vectors[bus](p) = DoubleComplex{.3 + .2 * p, .1 * (p - bus)};
            else
                vectors[bus](p) = .3 + .2 * p - .1 * bus;
        }
        vectors[bus].normalize();
    }
    auto set = [&](Idx i, Idx j, double coefficient) {
        Matrix const value = coefficient * vectors[i] * vectors[j].adjoint();
        covariance.override_physical[{i, j}] = {value, value.norm(), true};
        covariance.override_physical[{j, i}] = {value.adjoint(), value.norm(), true};
    };
    for (Idx bus = 0; bus != 3; ++bus)
        set(bus, bus, 1.);
    set(0, 1, .25);
    set(1, 2, -.4);
    covariance.raw(1, 1).value.setZero(); // Conditioning alone enters mean_path.
    std::vector<std::vector<Idx>> const graph{{1}, {0, 2}, {1}};
    PhysicalBlocks completion_stored{covariance};
    CovarianceCompletion completion{covariance, graph, graph, completion_stored};
    auto const result = completion.pair(0, 2);
    REQUIRE(result.block.valid);
    CHECK(result.diagnostic.route == UQPairRoute::mean_path);
    CHECK(result.diagnostic.middle_bus == -1);
    Matrix const expected = -.1 * vectors[0] * vectors[2].adjoint();
    CHECK((result.block.value - expected).norm() < 2e-15);
    CHECK(covariance.augmented_matrix.size() == 7); // A completed pair never becomes an original donor.

    // Invalid raw inputs terminate before the fallback, even with valid donors.
    covariance.raw(1, 1).value(0, 0) = std::numeric_limits<double>::quiet_NaN();
    PhysicalBlocks invalid_stored{covariance};
    CovarianceCompletion invalid{covariance, graph, graph, invalid_stored};
    auto const rejected = invalid.pair(0, 2);
    CHECK_FALSE(rejected.block.valid);
    CHECK(rejected.diagnostic.route == UQPairRoute::invalid);
    CHECK(rejected.diagnostic.middle_bus == -1);
}

TEST_CASE("Asymmetric completion retains coupled singular phase support and terminal invalid routing") {
    check_singular_support_completion<true>();
    check_singular_support_completion<false>();
}

template <bool complex> void check_mesh_approximation() {
    using Fixture = CoupledCovariance<complex>;
    using Matrix = typename Fixture::StateMatrix;
    using Raw = typename Fixture::RawMatrix;
    Fixture covariance{4};
    auto information = covariance.inverse.inverse().eval();
    // Close the physical/precision cycle 0--1--2--3--0. Pair 0--2 is
    // nonadjacent, and natural-order elimination fills only pair 1--3.
    // Thus 0--2 is truly missing from the selected inverse's filled pattern.
    information.block(0, 3 * Fixture::raw_size, Fixture::raw_size, Fixture::raw_size) = Raw::Identity() * .9;
    information.block(3 * Fixture::raw_size, 0, Fixture::raw_size, Fixture::raw_size) = Raw::Identity() * .9;
    covariance.inverse = information.inverse();
    // Add the pairs 0--3 and 1--3 to the pattern, each with its raw block.
    std::set<Pair> pattern{{0, 3}, {3, 0}, {1, 3}, {3, 1}};
    for (Idx i = 0; i != 4; ++i)
        for (Idx j = 0; j != 4; ++j)
            if (std::abs(i - j) <= 1)
                pattern.insert({i, j});
    covariance.augmented_matrix.clear();
    covariance.row_indptr = {0};
    covariance.col_indices.clear();
    for (Idx i = 0; i != 4; ++i) {
        for (Idx j = 0; j != 4; ++j)
            if (pattern.contains({i, j})) {
                covariance.augmented_matrix.push_back({covariance.inverse.block(
                    i * Fixture::raw_size, j * Fixture::raw_size, Fixture::raw_size, Fixture::raw_size)});
                covariance.col_indices.push_back(j);
                covariance.override_physical[{i, j}] =
                    covariance.correct_raw(i, j, covariance.augmented_matrix.back().value);
            }
        covariance.row_indptr.push_back(static_cast<Idx>(covariance.col_indices.size()));
    }
    REQUIRE_FALSE(covariance.find_block_index(0, 2).has_value());
    REQUIRE(covariance.find_block_index(1, 3).has_value());
    std::vector<std::vector<Idx>> const graph{{1, 3}, {0, 2}, {1, 3}, {0, 2}};
    Matrix const exact = covariance.variance_normalization *
                         covariance.inverse.template block<Fixture::states, Fixture::states>(0, 2 * Fixture::raw_size);
    PhysicalBlocks augmented_stored{covariance};
    CovarianceCompletion augmented{covariance, graph, graph, augmented_stored};
    auto const approximation = augmented.pair(0, 2);
    REQUIRE(approximation.block.valid);
    CHECK(approximation.diagnostic.route == UQPairRoute::augmented_path);
    CHECK(approximation.diagnostic.middle_bus == 1); // First conditioned candidate in ascending order.
    double const augmented_error = (approximation.block.value - exact).norm();
    INFO("augmented_path four-cycle sparse-versus-Full error = ", augmented_error);
    CHECK(augmented_error > 1e-4);

    // Both augmented candidates must fail conditioning before mean_path. The
    // physical donor pool remains the original full-rank covariance blocks.
    covariance.raw(1, 1).value.setZero();
    covariance.raw(3, 3).value.setZero();
    PhysicalBlocks mean_stored{covariance};
    CovarianceCompletion mean{covariance, graph, graph, mean_stored};
    auto const fallback = mean.pair(0, 2);
    REQUIRE(fallback.block.valid);
    CHECK(fallback.diagnostic.route == UQPairRoute::mean_path);
    CHECK(fallback.diagnostic.middle_bus == -1);
    // Equal-weight mean over both original paths, D_0k D_kk^-1 D_k2 for full-rank middles.
    auto const& physical = covariance.override_physical;
    Matrix const expected_mean =
        (physical.at({0, 1}).value * physical.at({1, 1}).value.inverse() * physical.at({1, 2}).value +
         physical.at({0, 3}).value * physical.at({3, 3}).value.inverse() * physical.at({3, 2}).value) /
        2.;
    CHECK((fallback.block.value - expected_mean).norm() < 1e-12);
    double const mean_error = (fallback.block.value - exact).norm();
    INFO("mean_path four-cycle sparse-versus-Full error = ", mean_error);
    CHECK(mean_error > 1e-4);
}

TEST_CASE("Asymmetric nonseparator and mean completion validity does not imply exact covariance") {
    check_mesh_approximation<true>();
    check_mesh_approximation<false>();
}

template <bool complex> void check_full_middle_threshold() {
    using Fixture = CoupledCovariance<complex>;
    using Raw = typename Fixture::RawMatrix;
    std::vector<std::vector<Idx>> const graph{{1}, {0, 2}, {1}};
    for (double ratio : {1e-12 * (1. - 1e-6), 1e-12, 1e-12 * (1. + 1e-6)}) {
        Fixture covariance;
        covariance.raw(1, 1).value.setIdentity();
        // A multiplier phase alone controls the FULL augmented condition test.
        covariance.raw(1, 1).value(Fixture::raw_size - 1, Fixture::raw_size - 1) = ratio;
        for (Pair pair : {Pair{0, 1}, Pair{1, 0}, Pair{1, 2}, Pair{2, 1}})
            covariance.raw(pair.first, pair.second).value = Raw::Identity() * .1;
        PhysicalBlocks completion_stored{covariance};
        CovarianceCompletion completion{covariance, graph, graph, completion_stored};
        auto const result = completion.pair(0, 2);
        REQUIRE(result.block.valid);
        CHECK(result.diagnostic.route == (ratio < 1e-12 ? UQPairRoute::mean_path : UQPairRoute::separator));
        CHECK(result.diagnostic.middle_bus == (ratio < 1e-12 ? -1 : 1));
    }
}

TEST_CASE("Asymmetric completion conditioning includes every multiplier phase at the inherited threshold") {
    check_full_middle_threshold<true>();
    check_full_middle_threshold<false>();
}

TEST_CASE("Asymmetric zero-current singularity and exact-zero enforcement remain phase local") {
    CoupledCovariance<true> covariance;
    covariance.u[1] = covariance.u[0];
    auto const topology = coupled_topology();
    auto parameters = coupled_parameters();
    for (int p = 0; p != 3; ++p) {
        parameters.branch_param[0].yff()(0, p) = p == 0 ? 1. : 0.;
        parameters.branch_param[0].yft()(0, p) = p == 0 ? -1. : 0.;
    }
    YBus<asymmetric_t> const y_bus{topology, parameters};
    StateEstimationUQRequest request;
    request.branch = {{0, BranchSide::from, true, true, true, true}};
    StateEstimationUQResult<asymmetric_t> result;
    PhysicalBlocks stored{covariance};
    branch_results(covariance, y_bus, request, result, stored);
    auto const& branch = result.branch[0];
    CHECK(branch.active_power.phase[0].raw_status == UQVarianceStatus::valid);
    CHECK(branch.active_power.phase[0].sigma > 0.);
    CHECK(branch.reactive_power.phase[0].sigma > 0.);
    CHECK(branch.current_magnitude.phase[0].raw_status == UQVarianceStatus::invalid);
    CHECK(branch.current_angle.phase[0].raw_status == UQVarianceStatus::invalid);
    for (int p : {1, 2}) {
        CHECK(branch.current_magnitude.phase[p].raw_status == UQVarianceStatus::valid);
        CHECK(branch.current_angle.phase[p].raw_status == UQVarianceStatus::valid);
    }
    CoupledMeasurements measured;
    measured.constrained = true;
    request.injection = {{1, true, true}};
    auto& invalid = covariance.raw(1, 1).value;
    invalid(0, 0) = std::numeric_limits<double>::quiet_NaN();
    PhysicalBlocks result_stored{covariance};
    injection_results(covariance, y_bus, measured, request, result, result_stored);
    for (int p = 0; p != 3; ++p) {
        auto const& output = result.injection[0].active_power.phase[p];
        CHECK(output.raw_status == UQVarianceStatus::invalid);
        CHECK(output.exact_zero_enforced == (p == 0));
        if (p == 0)
            CHECK(output.sigma == 0.);
        else
            CHECK(std::isnan(output.sigma));
    }
}

} // namespace
} // namespace power_grid_model::math_solver
