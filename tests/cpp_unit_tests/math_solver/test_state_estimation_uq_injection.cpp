// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/LU>
#include <complex>
#include <doctest/doctest.h>
#include <map>
#include <power_grid_model/math_solver/state_estimation_uq_injection.hpp>
#include <set>

namespace power_grid_model::math_solver {
namespace {
using namespace state_estimation_uq;
using Complex = std::complex<double>;
using Pair = std::pair<Idx, Idx>;
using Pattern = std::set<Pair>;
using DenseComplex = Eigen::MatrixXcd;
using DenseReal = Eigen::MatrixXd;

struct InjectionYBus {
    MathModelTopology topology;
    IdxVector rows{0}, columns;
    std::vector<Complex> values;
    InjectionYBus(DenseComplex const& y, Pattern const& edges) {
        for (auto [a, b] : edges)
            topology.branch_bus_idx.push_back({a, b});
        for (Idx i = 0; i < y.rows(); ++i) {
            for (Idx j = 0; j < y.cols(); ++j)
                if (i == j || edges.contains(std::minmax(i, j))) {
                    columns.push_back(j);
                    values.push_back(y(i, j));
                }
            rows.push_back(static_cast<Idx>(columns.size()));
        }
    }
    Idx size() const { return static_cast<Idx>(rows.size() - 1); }
    auto const& row_indptr() const { return rows; }
    auto const& col_indices() const { return columns; }
    auto const& admittance() const { return values; }
    auto const& math_topology() const { return topology; }
};
struct InjectionMeasurements {
    std::map<Idx, PowerSensorCalcParam<symmetric_t>> values;
    bool has_bus_injection(Idx bus) const { return values.contains(bus); }
    auto const& bus_injection(Idx bus) const { return values.at(bus); }
    void set(Idx bus, double p, double q, double pvar, double qvar) { values[bus] = {{p, pvar}, {q, qvar}}; }
};

template <class S, int N, int D> struct InjectionWorkspace {
    using RawMatrix = Eigen::Matrix<S, N, N>;
    using StateMatrix = Eigen::Matrix<S, D, D>;
    using PhysicalMatrix = std::conditional_t<D == 1, Eigen::Matrix<Complex, 1, 1>, Eigen::Matrix2d>;
    using StateBlock = CovarianceBlock<StateMatrix>;
    using PhysicalBlock = CovarianceBlock<PhysicalMatrix>;
    struct Raw {
        RawMatrix value;
        RawMatrix const& matrix() const { return value; }
    };
    std::vector<Raw> augmented_matrix; // Indexed by block index, as in the workspace.
    std::map<Pair, Idx> block_index;
    std::map<Pair, PhysicalBlock> override_physical;
    std::vector<Complex> u;
    IdxVector row_indptr{0}, col_indices;
    double normalization = 1;
    explicit InjectionWorkspace(std::vector<Complex> voltage) : u{std::move(voltage)} {}
    std::optional<Idx> find_block_index(Idx i, Idx j) const {
        auto it = block_index.find({i, j});
        return it == block_index.end() ? std::nullopt : std::optional<Idx>{it->second};
    }
    DoubleComplex phase_voltage(Idx bus, int) const { return u[bus]; }
    StateBlock correct_raw(Idx, Idx, RawMatrix const& block) const {
        StateMatrix value = normalization * block.template topLeftCorner<D, D>();
        return {value, frobenius_norm(value), true};
    }
    Eigen::Matrix2d jacobian(Idx bus) const {
        Complex phase = std::polar(1., std::arg(u[bus]));
        Eigen::Matrix2d result;
        result << -u[bus].imag(), phase.real(), u[bus].real(), phase.imag();
        return result;
    }
    PhysicalBlock physical_from_state(Idx i, Idx j, StateBlock const& state) const {
        PhysicalMatrix value;
        if constexpr (D == 1)
            value = state.value;
        else
            value = jacobian(i) * state.value * jacobian(j).transpose();
        return {value, frobenius_norm(value), state.valid};
    }
    PhysicalBlock physical_block(Idx i, Idx j) const {
        if (auto it = override_physical.find({i, j}); it != override_physical.end())
            return it->second;
        auto index = find_block_index(i, j);
        if (!index)
            return {PhysicalMatrix::Zero(), 0, false};
        return physical_from_state(i, j, correct_raw(i, j, augmented_matrix[*index].matrix()));
    }
    void pattern() {
        row_indptr = {0};
        col_indices.clear();
        for (Idx i = 0; i < static_cast<Idx>(u.size()); ++i) {
            for (Idx j = 0; j < static_cast<Idx>(u.size()); ++j)
                if (block_index.contains({i, j}))
                    col_indices.push_back(j);
            row_indptr.push_back(static_cast<Idx>(col_indices.size()));
        }
    }
    template <class Dense> void set_inverse(Dense const& inverse, Pattern const& missing = {}) {
        for (Idx i = 0; i < static_cast<Idx>(u.size()); ++i)
            for (Idx j = 0; j < static_cast<Idx>(u.size()); ++j)
                if (!missing.contains(std::minmax(i, j))) {
                    auto const [it, inserted] =
                        block_index.try_emplace({i, j}, static_cast<Idx>(augmented_matrix.size()));
                    if (inserted)
                        augmented_matrix.emplace_back();
                    augmented_matrix[it->second].value = inverse.block(i * N, j * N, N, N);
                }
        pattern();
    }
    void physical(Idx i, Idx j, PhysicalMatrix const& value, double scale = -1) {
        if (scale < 0)
            scale = frobenius_norm(value);
        override_physical[{i, j}] = {value, scale, true};
        override_physical[{j, i}] = {value.adjoint(), scale, true};
    }
};
using ILSEWorkspace = InjectionWorkspace<Complex, 2, 1>;
using NRSEWorkspace = InjectionWorkspace<double, 4, 2>;

template <class S, int N> auto reference_inverse(Idx n, Pattern const& edges) {
    Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic> information =
        Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>::Identity(n * N, n * N) * 5;
    for (auto [a, b] : edges) {
        Eigen::Matrix<S, N, N> block;
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                double value = .07 * (r + 1) - .04 * (c + 1) + .02 * a;
                if constexpr (Eigen::NumTraits<S>::IsComplex)
                    block(r, c) = {value, .02 * (r - c + 1)};
                else
                    block(r, c) = value;
            }
        information.block(a * N, b * N, N, N) = block;
        information.block(b * N, a * N, N, N) = block.adjoint();
    }
    return information.inverse().eval();
}

template <class Workspace, class Dense>
DenseReal rectangular_reference(Workspace const& workspace, Dense const& inverse) {
    constexpr int N = Workspace::RawMatrix::RowsAtCompileTime;
    constexpr int D = Workspace::StateMatrix::RowsAtCompileTime;
    Idx n = static_cast<Idx>(workspace.u.size());
    DenseReal result(2 * n, 2 * n);
    for (Idx i = 0; i < n; ++i)
        for (Idx j = 0; j < n; ++j) {
            if constexpr (D == 1) {
                Complex value = workspace.normalization * inverse(i * N, j * N);
                result.block<2, 2>(2 * i, 2 * j) << value.real() / 2, -value.imag() / 2, value.imag() / 2,
                    value.real() / 2;
            } else {
                Eigen::Matrix2d state = workspace.normalization * inverse.template block<2, 2>(i * N, j * N);
                result.block<2, 2>(2 * i, 2 * j) = workspace.jacobian(i) * state * workspace.jacobian(j).transpose();
            }
        }
    return result;
}

Eigen::RowVectorXd differentiate(DenseComplex const& y, std::vector<Complex> const& u, Idx bus, bool reactive) {
    Eigen::RowVectorXd result(2 * static_cast<Eigen::Index>(u.size()));
    auto output = [&](std::vector<Complex> const& state) {
        Complex current{};
        for (std::size_t j = 0; j < state.size(); ++j)
            current += y(bus, j) * state[j];
        Complex power = state[bus] * std::conj(current);
        return reactive ? power.imag() : power.real();
    };
    double const h = 1e-6;
    for (std::size_t j = 0; j < u.size(); ++j)
        for (int axis = 0; axis < 2; ++axis) {
            auto plus = u, minus = u;
            Complex delta = axis == 0 ? Complex{h} : Complex{0, h};
            plus[j] += delta;
            minus[j] -= delta;
            result[2 * j + axis] = (output(plus) - output(minus)) / (2 * h);
        }
    return result;
}
StateEstimationUQRequest injections(std::initializer_list<Idx> buses) {
    StateEstimationUQRequest request;
    for (Idx bus : buses)
        request.injection.push_back({bus, true, true});
    request.pair_diagnostics = true;
    return request;
}

DenseComplex network_y(Idx n, Pattern const& edges) {
    DenseComplex y = DenseComplex::Zero(n, n);
    for (auto [a, b] : edges) {
        Complex line{.4 + .1 * a, -1.2 - .1 * b};
        y(a, a) += line;
        y(b, b) += line;
        y(a, b) -= line;
        y(b, a) -= line;
    }
    for (Idx i = 0; i < n; ++i)
        y(i, i) += Complex{.03 * (i + 1), .07}; // Actual diagonal shunts.
    return y;
}

template <class S, int N, int D> void dense_maps(bool missing) {
    Pattern const edges{{0, 1}, {1, 2}, {2, 3}};
    auto inverse = reference_inverse<S, N>(4, edges);
    InjectionWorkspace<S, N, D> workspace{{{1., .1}, {.97, -.2}, {1.03, .15}, {.95, .3}}};
    workspace.normalization = .37;
    workspace.set_inverse(inverse, missing ? Pattern{{0, 2}, {1, 3}, {0, 3}} : Pattern{});
    auto y = network_y(4, edges);
    InjectionYBus ybus{y, edges};
    auto covariance = rectangular_reference(workspace, inverse);
    StateEstimationUQResult<symmetric_t> result;
    result.voltage.resize(1);
    result.voltage[0].magnitude.phase[0].sigma = 123;
    result.branch.resize(1);
    result.branch[0].active_power.phase[0].sigma = 456;
    PhysicalBlocks result_stored{workspace};
    injection_results(workspace, ybus, InjectionMeasurements{}, injections({1, 2}), result, result_stored);
    // Physical blocks shared with another helper, also once filled, give identical results.
    PhysicalBlocks stored{workspace};
    for (int repeat = 0; repeat < 2; ++repeat) {
        StateEstimationUQResult<symmetric_t> shared;
        injection_results(workspace, ybus, InjectionMeasurements{}, injections({1, 2}), shared, stored);
        REQUIRE(shared.injection.size() == 2);
        for (int i = 0; i < 2; ++i)
            for (auto const& [x, z] : {std::pair{result.injection[i].active_power, shared.injection[i].active_power},
                                       {result.injection[i].reactive_power, shared.injection[i].reactive_power}}) {
                CHECK(x.phase[0].raw_status == z.phase[0].raw_status);
                CHECK(x.phase[0].raw_variance == z.phase[0].raw_variance);
                CHECK(x.phase[0].absolute_scale == z.phase[0].absolute_scale);
            }
        CHECK(shared.pairs.size() == result.pairs.size());
    }
    for (int i = 0; i < 2; ++i)
        for (bool reactive : {false, true}) {
            auto gradient = differentiate(y, workspace.u, i + 1, reactive);
            double expected = (gradient * covariance * gradient.transpose())(0, 0);
            auto const& value =
                reactive ? result.injection[i].reactive_power.phase[0] : result.injection[i].active_power.phase[0];
            CHECK(value.raw_status == UQVarianceStatus::valid);
            CHECK(static_cast<double>(value.raw_variance) == doctest::Approx(expected).epsilon(2e-9));
            CHECK(static_cast<double>(value.sigma * value.sigma) == doctest::Approx(expected).epsilon(2e-9));
        }
    CHECK(result.voltage[0].magnitude.phase[0].sigma == 123);
    CHECK(result.branch[0].active_power.phase[0].sigma == 456);
    CHECK(result.pairs.size() == 5);
    for (auto const& pair : result.pairs) {
        CHECK(pair.first_bus < pair.second_bus);
        if (pair.second_bus - pair.first_bus == 2)
            CHECK(pair.route == (missing ? UQPairRoute::separator : UQPairRoute::stored));
    }
}

TEST_CASE("UQ injection real maps match independently differentiated dense covariance") {
    dense_maps<Complex, 2, 1>(false);
    dense_maps<double, 4, 2>(false);
    dense_maps<Complex, 2, 1>(true);
    dense_maps<double, 4, 2>(true);
}

TEST_CASE("UQ proper voltage still gives nonproper power perturbations") {
    ILSEWorkspace workspace{{Complex{1}}};
    Eigen::Matrix2cd inverse = Eigen::Matrix2cd::Identity();
    workspace.set_inverse(inverse);
    DenseComplex y(1, 1);
    y(0, 0) = {1, 2};
    InjectionYBus ybus{y, {}};
    StateEstimationUQResult<symmetric_t> result;
    PhysicalBlocks result_stored{workspace};
    injection_results(workspace, ybus, InjectionMeasurements{}, injections({0}), result, result_stored);
    CHECK(result.injection[0].active_power.phase[0].raw_variance == 2);
    CHECK(result.injection[0].reactive_power.phase[0].raw_variance == 8);
}

template <class S, int N, int D> void approximation_maps(bool terminal_mean_path) {
    Pattern edges{{0, 1}, {1, 2}, {2, 3}, {0, 3}};
    auto inverse = reference_inverse<S, N>(4, edges);
    InjectionWorkspace<S, N, D> workspace{{{1, .1}, {.97, -.2}, {1.03, .15}, {.95, .3}}};
    workspace.normalization = .37;
    workspace.set_inverse(inverse, {{1, 3}});
    if (terminal_mean_path)
        for (Idx k : {0, 2})
            // Only auxiliary coordinates change: corrected physical marginals
            // and original physical donors remain those of the dense reference.
            workspace.augmented_matrix[*workspace.find_block_index(k, k)].value.row(N - 1).setZero();
    auto const original_raw = workspace.augmented_matrix;
    auto const original_index = workspace.block_index;
    auto const original_rows = workspace.row_indptr, original_columns = workspace.col_indices;
    auto y = network_y(4, edges);
    InjectionYBus ybus{y, edges};
    auto actual = rectangular_reference(workspace, inverse), completed = actual;
    Eigen::Matrix2d prediction = Eigen::Matrix2d::Zero();
    if (terminal_mean_path) {
        // Independent corrected physical path average, using both original
        // common-neighbor paths with equal weights after conditioning exhaustion.
        if constexpr (D == 1) {
            Complex physical{};
            for (Idx k : {0, 2})
                physical += workspace.normalization * inverse(1 * N, k * N) * inverse(k * N, 3 * N) /
                            inverse(k * N, k * N) / 2.;
            prediction << physical.real() / 2, -physical.imag() / 2, physical.imag() / 2, physical.real() / 2;
        } else {
            for (Idx k : {0, 2})
                prediction += actual.template block<2, 2>(2, 2 * k) *
                              actual.template block<2, 2>(2 * k, 2 * k).inverse() *
                              actual.template block<2, 2>(2 * k, 6) / 2;
        }
    } else {
        // Ascending common middles are {0, 2}. Form the complete raw augmented
        // product through the first middle before any state/physical correction.
        Eigen::Matrix<S, N, N> raw = inverse.template block<N, N>(N, 0) * inverse.template block<N, N>(0, 0).inverse() *
                                     inverse.template block<N, N>(0, 3 * N);
        auto reference = inverse;
        reference.template block<N, N>(N, 3 * N) = raw;
        reference.template block<N, N>(3 * N, N) = raw.adjoint();
        prediction = rectangular_reference(workspace, reference).template block<2, 2>(2, 6);
    }
    completed.template block<2, 2>(2, 6) = prediction;
    completed.template block<2, 2>(6, 2) = prediction.transpose();
    StateEstimationUQResult<symmetric_t> result;
    PhysicalBlocks result_stored{workspace};
    injection_results(workspace, ybus, InjectionMeasurements{}, injections({0}), result, result_stored);
    bool saw = false;
    for (auto const& diagnostic : result.pairs)
        if (diagnostic.first_bus == 1 && diagnostic.second_bus == 3) {
            saw = true;
            CHECK(diagnostic.route == (terminal_mean_path ? UQPairRoute::mean_path : UQPairRoute::augmented_path));
            CHECK(diagnostic.middle_bus == (terminal_mean_path ? -1 : 0));
        }
    CHECK(saw);
    auto p = differentiate(y, workspace.u, 0, false);
    double expected = (p * completed * p.transpose())(0, 0), full = (p * actual * p.transpose())(0, 0);
    CHECK(result.injection[0].active_power.phase[0].raw_status == UQVarianceStatus::valid);
    CHECK(static_cast<double>(result.injection[0].active_power.phase[0].raw_variance) ==
          doctest::Approx(expected).epsilon(2e-9));
    CHECK(std::abs(expected - full) > 1e-9); // Assess approximation separately from exact checks.
    CHECK(workspace.row_indptr == original_rows);
    CHECK(workspace.col_indices == original_columns);
    CHECK(workspace.augmented_matrix.size() == original_raw.size());
    for (auto const& [pair, index] : original_index)
        CHECK((workspace.augmented_matrix[index].matrix() - original_raw[index].matrix()).norm() == 0);
}
TEST_CASE("UQ injection retains first conditioned nonseparator and terminal physical mean path") {
    approximation_maps<Complex, 2, 1>(false);
    approximation_maps<double, 4, 2>(false);
    approximation_maps<Complex, 2, 1>(true);
    approximation_maps<double, 4, 2>(true);
}

TEST_CASE("UQ exact-zero injection reports zero while preserving raw validity and values") {
    ILSEWorkspace workspace{{Complex{1}}};
    workspace.set_inverse(Eigen::Matrix2cd::Identity());
    DenseComplex y(1, 1);
    y(0, 0) = {1, 2};
    InjectionYBus ybus{y, {}};
    for (auto const parameters : std::array<std::array<double, 4>, 5>{
             {{0, 0, 0, 0}, {0, 0, .1, 0}, {0, 0, 0, .1}, {1e-200, 0, 0, 0}, {0, 1e-200, 0, 0}}}) {
        InjectionMeasurements measured;
        measured.set(0, parameters[0], parameters[1], parameters[2], parameters[3]);
        StateEstimationUQResult<symmetric_t> result;
        PhysicalBlocks result_stored{workspace};
        injection_results(workspace, ybus, measured, injections({0}), result, result_stored);
        bool force = parameters == std::array<double, 4>{0, 0, 0, 0};
        for (bool reactive : {false, true}) {
            auto const& value =
                reactive ? result.injection[0].reactive_power.phase[0] : result.injection[0].active_power.phase[0];
            CHECK(value.exact_zero_enforced == force);
            CHECK(value.raw_status == UQVarianceStatus::valid);
            CHECK(value.raw_variance == (reactive ? 8 : 2));
            CHECK((value.sigma == 0) == force);
        }
    }
    workspace.physical(0, 0, ILSEWorkspace::PhysicalMatrix::Constant(Complex{-1}));
    InjectionMeasurements measured;
    measured.set(0, 0, 0, 0, 0);
    StateEstimationUQResult<symmetric_t> invalid;
    PhysicalBlocks invalid_stored{workspace};
    injection_results(workspace, ybus, measured, injections({0}), invalid, invalid_stored);
    auto const& value = invalid.injection[0].active_power.phase[0];
    CHECK(value.raw_status == UQVarianceStatus::invalid);
    CHECK(value.raw_variance == -2);
    CHECK(value.absolute_scale == 2);
    CHECK(value.exact_zero_enforced);
    CHECK(value.sigma == 0);
}

TEST_CASE("UQ injection signed power maps remain defined at zero voltage and current") {
    ILSEWorkspace workspace{{Complex{}, Complex{1}}};
    workspace.set_inverse(Eigen::Matrix4cd::Identity());
    DenseComplex y(2, 2);
    y << Complex{1}, Complex{-1}, Complex{-1}, Complex{1};
    InjectionYBus ybus{y, {{0, 1}}};
    StateEstimationUQResult<symmetric_t> zero_voltage;
    PhysicalBlocks zero_voltage_stored{workspace};
    injection_results(workspace, ybus, InjectionMeasurements{}, injections({0}), zero_voltage, zero_voltage_stored);
    CHECK(zero_voltage.injection[0].active_power.phase[0].raw_status == UQVarianceStatus::valid);
    CHECK(zero_voltage.injection[0].active_power.phase[0].raw_variance == .5);
    workspace.u[0] = 1;
    StateEstimationUQResult<symmetric_t> zero_current;
    PhysicalBlocks zero_current_stored{workspace};
    injection_results(workspace, ybus, InjectionMeasurements{}, injections({0}), zero_current, zero_current_stored);
    CHECK(zero_current.injection[0].active_power.phase[0].raw_status == UQVarianceStatus::valid);
    CHECK(zero_current.injection[0].active_power.phase[0].raw_variance == 1);
    CHECK_FALSE(zero_current.injection[0].active_power.phase[0].exact_zero_enforced);
}

TEST_CASE("UQ streamed injection contraction uses rounded double cancellation and rejects overflow") {
    NRSEWorkspace workspace{{Complex{1}, Complex{0, 1}}};
    workspace.set_inverse(Eigen::Matrix<double, 8, 8>::Identity());
    Eigen::Matrix2d marginal = Eigen::Matrix2d::Zero(), cross = Eigen::Matrix2d::Zero();
    marginal(0, 0) = 1e20;
    marginal(1, 1) = 1;
    cross(0, 0) = 1e20;
    workspace.physical(0, 0, marginal);
    workspace.physical(1, 1, marginal);
    workspace.physical(0, 1, cross);
    DenseComplex y = DenseComplex::Zero(2, 2);
    y(0, 0) = .5;
    y(0, 1) = -1;
    InjectionYBus ybus{y, {{0, 1}}};
    StateEstimationUQResult<symmetric_t> result;
    PhysicalBlocks result_stored{workspace};
    injection_results(workspace, ybus, InjectionMeasurements{}, injections({0}), result, result_stored);
    auto const& value = result.injection[0].active_power.phase[0];
    CHECK(value.raw_status == UQVarianceStatus::valid);
    // The unit term is absorbed before the 1e20 terms cancel in double.
    CHECK(value.raw_variance == 0);
    CHECK(value.sigma == 0);
    CHECK(value.absolute_scale == 4e20);
    ILSEWorkspace huge{{Complex{1e200}}};
    huge.set_inverse((Eigen::Matrix2cd::Identity() * 1e200).eval());
    DenseComplex admittance(1, 1);
    admittance(0, 0) = 1e100;
    InjectionYBus huge_y{admittance, {}};
    StateEstimationUQResult<symmetric_t> output;
    PhysicalBlocks output_stored{huge};
    injection_results(huge, huge_y, InjectionMeasurements{}, injections({0}), output, output_stored);
    CHECK(output.injection[0].active_power.phase[0].raw_status == UQVarianceStatus::invalid);
    CHECK(std::isnan(output.injection[0].active_power.phase[0].sigma));
}

TEST_CASE("UQ pair-consistent covariance can still give invalid injection variance") {
    NRSEWorkspace workspace{{Complex{1}, Complex{1}, Complex{1}}};
    workspace.set_inverse(Eigen::Matrix<double, 12, 12>::Identity());
    for (Idx i = 0; i < 3; ++i)
        workspace.physical(i, i, Eigen::Matrix2d::Identity());
    workspace.physical(0, 1, Eigen::Matrix2d::Identity() * .9);
    workspace.physical(0, 2, Eigen::Matrix2d::Identity() * .9);
    workspace.physical(1, 2, -Eigen::Matrix2d::Identity() * .9);
    DenseComplex y = DenseComplex::Zero(3, 3);
    y(0, 0) = -1.5;
    y(0, 1) = 1;
    y(0, 2) = 1;
    InjectionYBus ybus{y, {{0, 1}, {0, 2}, {1, 2}}};
    InjectionMeasurements noisy;
    noisy.set(0, 0, 0, .1, .1);
    StateEstimationUQResult<symmetric_t> result;
    PhysicalBlocks result_stored{workspace};
    injection_results(workspace, ybus, noisy, injections({0}), result, result_stored);
    auto const& raw = result.injection[0].active_power.phase[0];
    CHECK(raw.raw_status == UQVarianceStatus::invalid);
    CHECK(static_cast<double>(raw.raw_variance) == doctest::Approx(-2.4));
    CHECK(std::isnan(raw.sigma));
    CHECK_FALSE(raw.exact_zero_enforced);
    for (auto const& diagnostic : result.pairs)
        CHECK(diagnostic.route == UQPairRoute::stored);
    InjectionMeasurements exact;
    exact.set(0, 0, 0, 0, 0);
    StateEstimationUQResult<symmetric_t> enforced;
    PhysicalBlocks enforced_stored{workspace};
    injection_results(workspace, ybus, exact, injections({0}), enforced, enforced_stored);
    auto const& forced = enforced.injection[0].active_power.phase[0];
    CHECK(forced.raw_status == UQVarianceStatus::invalid);
    CHECK(forced.raw_variance == raw.raw_variance);
    CHECK(forced.absolute_scale == raw.absolute_scale);
    CHECK(forced.sigma == 0);
    CHECK(forced.exact_zero_enforced);
}

TEST_CASE("UQ injection clips only scale-relative negative roundoff") {
    NRSEWorkspace workspace{{Complex{1}, Complex{1}}};
    workspace.set_inverse(Eigen::Matrix<double, 8, 8>::Identity());
    for (Idx i = 0; i < 2; ++i)
        workspace.physical(i, i, Eigen::Matrix2d::Identity());
    workspace.physical(0, 1, Eigen::Matrix2d::Identity() * (1 + std::numeric_limits<double>::epsilon()));
    DenseComplex y = DenseComplex::Zero(2, 2);
    y(0, 0) = -1;
    y(0, 1) = 1;
    InjectionYBus ybus{y, {{0, 1}}};
    StateEstimationUQResult<symmetric_t> result;
    PhysicalBlocks result_stored{workspace};
    injection_results(workspace, ybus, InjectionMeasurements{}, injections({0}), result, result_stored);
    auto const& value = result.injection[0].active_power.phase[0];
    CHECK(value.raw_status == UQVarianceStatus::clipped_roundoff);
    CHECK(value.raw_variance < 0);
    CHECK(value.absolute_scale > 0);
    CHECK(value.sigma == 0);
    CHECK_FALSE(value.exact_zero_enforced);
}

TEST_CASE("UQ injection diagonal accessor clips only allowed physical roundoff") {
    NRSEWorkspace workspace{{Complex{1}}};
    workspace.set_inverse(Eigen::Matrix4d::Identity());
    Eigen::Matrix2d marginal = Eigen::Matrix2d::Identity();
    marginal(0, 0) = -1e-15;
    workspace.physical(0, 0, marginal, 1);
    std::vector<std::vector<Idx>> const one_bus{{}};
    PhysicalBlocks completion_stored{workspace};
    CovarianceCompletion completion{workspace, one_bus, one_bus, completion_stored};
    CHECK(completion.diagonal(0).valid);
    CHECK(completion.diagonal(0).value(0, 0) == 0);
    CHECK(completion.diagonal(0).value(1, 1) == 1);
    CHECK(completion.diagonal(0).construction_scale == 1);
    marginal(0, 0) = -1e-5;
    workspace.physical(0, 0, marginal, 1);
    PhysicalBlocks invalid_stored{workspace};
    CovarianceCompletion invalid{workspace, one_bus, one_bus, invalid_stored};
    CHECK_FALSE(invalid.diagonal(0).valid);
    CHECK(invalid.diagonal(0).value(0, 0) == -1e-5);
}
} // namespace
} // namespace power_grid_model::math_solver
