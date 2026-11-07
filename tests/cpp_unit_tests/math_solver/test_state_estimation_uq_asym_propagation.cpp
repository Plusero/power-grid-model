// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
// SPDX-License-Identifier: MPL-2.0

#include <power_grid_model/math_solver/state_estimation_uq_branch.hpp>

#include <Eigen/LU>
#include <doctest/doctest.h>

#include <map>

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
    std::map<Pair, Raw> augmented_matrix;
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
                    augmented_matrix[{i, j}].value = inverse.block(i * raw_size, j * raw_size, raw_size, raw_size);
                    col_indices.push_back(j);
                }
            row_indptr.push_back(static_cast<Idx>(col_indices.size()));
            slack_column.push_back(inverse.template block<states, 1>(i * raw_size, slack_bus * raw_size));
        }
    }
    Raw const* augmented_matrix_block(Idx i, Idx j) const {
        auto const it = augmented_matrix.find({i, j});
        return it == augmented_matrix.end() ? nullptr : &it->second;
    }
    std::optional<Idx> find_block_index(Idx i, Idx j) const {
        for (Idx p = row_indptr[i]; p != row_indptr[i + 1]; ++p)
            if (col_indices[p] == j)
                return p;
        return std::nullopt;
    }
    StateBlock correct_raw(Idx, Idx, RawMatrix const& value) const {
        StateMatrix const physical = variance_normalization * value.template topLeftCorner<states, states>();
        return {physical, physical.norm(), true};
    }
    StateBlock calculate_state_covariance_block(Idx i, Idx j) const {
        return correct_raw(i, j, augmented_matrix.at({i, j}).value);
    }
    PhysicalBlock physical_from_state(Idx, Idx, StateBlock const& value) const { return value; }
    PhysicalBlock physical_block(Idx i, Idx j) const {
        if (auto const it = override_physical.find({i, j}); it != override_physical.end())
            return it->second;
        auto const* block = augmented_matrix_block(i, j);
        return block ? correct_raw(i, j, block->value) : PhysicalBlock{StateMatrix::Zero(), 0., false};
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
}

TEST_CASE("Asymmetric UQ coupled endpoint contractions match independent real Jacobians") {
    check_coupled_propagation<true>(false);
    check_coupled_propagation<true>(true);
    check_coupled_propagation<false>(false);
}

TEST_CASE("Asymmetric zero-current singularity remains phase local") {
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
}

} // namespace
} // namespace power_grid_model::math_solver
