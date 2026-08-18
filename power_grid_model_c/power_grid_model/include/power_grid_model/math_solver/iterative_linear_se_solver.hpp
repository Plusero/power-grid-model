// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// iterative linear state estimation solver

#include "block_matrix.hpp"
#include "common_solver_functions.hpp"
#include "measured_values.hpp"
#include "observability.hpp"
#include "sparse_lu_solver.hpp"
#include "y_bus.hpp"

#include "../calculation_parameters.hpp"
#include "../common/common.hpp"
#include "../common/enum.hpp"
#include "../common/exception.hpp"
#include "../common/logging.hpp"
#include "../common/statistics.hpp"
#include "../common/three_phase_tensor.hpp"
#include "../common/timer.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace power_grid_model::math_solver {

// hide implementation in inside namespace
namespace iterative_linear_se {

// block class for the unknown vector and/or right-hand side in state estimation equation
template <symmetry_tag sym> struct ILSEUnknown : public Block<DoubleComplex, sym, false, 2> {
    template <int r, int c> using GetterType = Block<DoubleComplex, sym, false, 2>::template GetterType<r, c>;

    // eigen expression
    using Block<DoubleComplex, sym, false, 2>::Block;
    using Block<DoubleComplex, sym, false, 2>::operator=;

    GetterType<0, 0> u() { return this->template get_val<0, 0>(); }
    GetterType<1, 0> phi() { return this->template get_val<1, 0>(); }

    GetterType<0, 0> eta() { return this->template get_val<0, 0>(); }
    GetterType<1, 0> tau() { return this->template get_val<1, 0>(); }
};

// block class for the right hand side in state estimation equation
template <symmetry_tag sym> using ILSERhs = ILSEUnknown<sym>;

// class of 2*2 (6*6) se gain block
// [
//    [G, QH]
//    [Q, R ]
// ]
template <symmetry_tag sym> class ILSEGainBlock : public Block<DoubleComplex, sym, true, 2> {
  public:
    template <int r, int c> using GetterType = Block<DoubleComplex, sym, true, 2>::template GetterType<r, c>;

    // eigen expression
    using Block<DoubleComplex, sym, true, 2>::Block;
    using Block<DoubleComplex, sym, true, 2>::operator=;

    GetterType<0, 0> g() { return this->template get_val<0, 0>(); }
    GetterType<0, 1> qh() { return this->template get_val<0, 1>(); }
    GetterType<1, 0> q() { return this->template get_val<1, 0>(); }
    GetterType<1, 1> r() { return this->template get_val<1, 1>(); }
};

template <symmetry_tag sym_type> class IterativeLinearSESolver {
  public:
    using sym = sym_type;

    static constexpr auto is_iterative = true;
    static constexpr auto has_global_current_sensor_implemented =
        true; // TODO(figueroa1395): for testing purposes; remove after NRSE has global current sensor implemented
    static constexpr auto is_NRSE_solver = false; // for testing purposes only

  private:
    // block size 2 for symmetric, 6 for asym
    static constexpr Idx bsr_block_size_ = is_symmetric_v<sym> ? 2 : 6;
    static constexpr int n_phase_ = is_symmetric_v<sym> ? 1 : 3;

    using PhaseVector = Eigen::Matrix<DoubleComplex, n_phase_, 1>;
    using PhaseMatrix = Eigen::Matrix<DoubleComplex, n_phase_, n_phase_>;
    using TerminalStateMatrix = Eigen::Matrix<DoubleComplex, 2 * n_phase_, 2 * n_phase_>;
    using TerminalJacobian = Eigen::Matrix<DoubleComplex, n_phase_, 2 * n_phase_>;

  public:
    IterativeLinearSESolver(YBus<sym> const& y_bus, MathModelTopology const& topo)
        : n_bus_{y_bus.size()},
          math_topo_{topo},
          data_gain_(y_bus.nnz_lu()),
          x_rhs_(y_bus.size()),
          sparse_solver_{y_bus.row_indptr_lu(), y_bus.col_indices_lu(), y_bus.lu_diag()},
          perm_(y_bus.size()) {}

    SolverOutput<sym> run_state_estimation(YBus<sym> const& y_bus, StateEstimationInput<sym> const& input,
                                           double err_tol, Idx max_iter, Logger& log) {
        return run_state_estimation(y_bus, input, err_tol, max_iter, false, log);
    }

    SolverOutput<sym> run_state_estimation(YBus<sym> const& y_bus, StateEstimationInput<sym> const& input,
                                           double err_tol, Idx max_iter, bool calculate_uncertainty, Logger& log) {
        // prepare
        Timer main_timer;
        Timer sub_timer;
        SolverOutput<sym> output;
        output.u.resize(n_bus_);
        output.bus_injection.resize(n_bus_);
        double max_dev = std::numeric_limits<double>::max();

        main_timer = Timer{log, LogEvent::math_solver};

        // preprocess measured value
        sub_timer = Timer{log, LogEvent::preprocess_measured_value};
        MeasuredValues<sym> const measured_values{y_bus.math_topology(), input};
        auto const observability_result =
            observability::observability_check(measured_values, y_bus.math_topology(), y_bus.y_bus_structure());

        // prepare matrix
        sub_timer = Timer{log, LogEvent::prepare_matrix_including_prefactorization};
        prepare_matrix(y_bus, measured_values);
        // prefactorize
        sparse_solver_.prefactorize(data_gain_, perm_, observability_result.use_perturbation());

        // initialize voltage with initial angle
        sub_timer = Timer{log, LogEvent::initialize_voltages}; // TODO(mgovers): make scoped subtimers
        RealValue<sym> const mean_angle_shift = measured_values.mean_angle_shift();
        auto const& topo = math_topo_.get();
        for (Idx bus = 0; bus != n_bus_; ++bus) {
            output.u[bus] = exp(1.0i * (mean_angle_shift + topo.phase_shift[bus]));
        }

        // loop to iterate
        Idx num_iter = 0;
        while (max_dev > err_tol || num_iter == 0) {
            if (num_iter++ == max_iter) {
                throw IterationDiverge{max_iter, max_dev, err_tol};
            }
            sub_timer = Timer{log, LogEvent::calculate_rhs};
            prepare_rhs(y_bus, measured_values, output.u);
            // solve with prefactorization
            sub_timer = Timer{log, LogEvent::solve_sparse_linear_equation_prefactorized};
            sparse_solver_.solve_with_prefactorized_matrix(data_gain_, perm_, x_rhs_, x_rhs_);
            sub_timer = Timer{log, LogEvent::iterate_unknown};
            max_dev = iterate_unknown(output.u, measured_values.has_angle());
        }

        // calculate math result
        sub_timer = Timer{log, LogEvent::calculate_math_result};
        detail::calculate_se_result<sym>(y_bus, measured_values, output);

        if (calculate_uncertainty) {
            calculate_state_estimation_uncertainty(y_bus, measured_values, output);
        }

        // Manually stop timers to avoid "Max number of iterations" to be included in the timing.
        sub_timer.stop();
        main_timer.stop();

        log.log(LogEvent::max_num_iter, num_iter);

        return output;
    }

  private:
    // array selection function pointer
    static constexpr std::array has_branch_power_{&MeasuredValues<sym>::has_branch_from_power,
                                                  &MeasuredValues<sym>::has_branch_to_power};
    static constexpr std::array branch_power_{&MeasuredValues<sym>::branch_from_power,
                                              &MeasuredValues<sym>::branch_to_power};
    static constexpr std::array has_branch_current_{&MeasuredValues<sym>::has_branch_from_current,
                                                    &MeasuredValues<sym>::has_branch_to_current};
    static constexpr std::array branch_current_{&MeasuredValues<sym>::branch_from_current,
                                                &MeasuredValues<sym>::branch_to_current};

    Idx n_bus_;
    // shared topo data
    std::reference_wrapper<MathModelTopology const> math_topo_;

    // data for gain matrix
    std::vector<ILSEGainBlock<sym>> data_gain_;
    // unknown and rhs
    std::vector<ILSERhs<sym>> x_rhs_;
    // solver
    SparseLUSolver<ILSEGainBlock<sym>, ILSERhs<sym>, ILSEUnknown<sym>> sparse_solver_;
    SparseLUSolver<ILSEGainBlock<sym>, ILSERhs<sym>, ILSEUnknown<sym>>::BlockPermArray perm_;

    static auto diagonal_inverse(RealValue<sym> const& value) {
        return ComplexDiagonalTensor<sym>{static_cast<ComplexValue<sym>>(RealValue<sym>{1.0} / value)};
    }

    void prepare_matrix(YBus<sym> const& y_bus, MeasuredValues<sym> const& measured_value) {
        MathModelParam<sym> const& param = y_bus.math_model_param();
        IdxVector const& row_indptr = y_bus.row_indptr_lu();
        IdxVector const& col_indices = y_bus.col_indices_lu();

        // loop data index, all rows and columns
        for (Idx row = 0; row != n_bus_; ++row) {
            for (Idx data_idx_lu = row_indptr[row]; data_idx_lu != row_indptr[row + 1]; ++data_idx_lu) {
                Idx const col = col_indices[data_idx_lu];
                // get a reference and reset block to zero
                ILSEGainBlock<sym>& block = data_gain_[data_idx_lu];
                block.clear();
                // get data idx of y bus,
                // skip for a fill-in
                Idx const data_idx = y_bus.map_lu_y_bus()[data_idx_lu];
                if (data_idx == -1) {
                    continue;
                }
                // fill block with voltage measurement, only diagonal
                if ((row == col) && measured_value.has_voltage(row)) {
                    // G += 1.0 / variance
                    // for 3x3 tensor, fill diagonal
                    block.g() += ComplexTensor<sym>{1.0 / measured_value.voltage_var(row)};
                }
                // fill block with branch, shunt measurement
                for (Idx element_idx = y_bus.y_bus_entry_indptr()[data_idx];
                     element_idx != y_bus.y_bus_entry_indptr()[data_idx + 1]; ++element_idx) {
                    Idx const obj = y_bus.y_bus_element()[element_idx].idx;
                    YBusElementType const type = y_bus.y_bus_element()[element_idx].element_type;
                    // shunt
                    if (type == YBusElementType::shunt) {
                        if (measured_value.has_shunt(obj)) {
                            // G += (-Ys)^H * (variance^-1) * (-Ys) // NOSONAR(S125)
                            auto const& shunt_power = measured_value.shunt_power(obj);
                            auto const current = power_to_global_current_measurement(shunt_power);
                            block.g() += dot(hermitian_transpose(param.shunt_param[obj]),
                                             diagonal_inverse(current.variance), param.shunt_param[obj]);
                        }
                    }
                    // branch
                    else {
                        auto const add_branch_measurement = [&block, &param, obj, type](
                                                                IntS measured_side,
                                                                IndependentComplexRandVar<sym> const& branch_current) {
                            // branch from- and to-side index at 0, and 1 position
                            IntS const b0 = std::to_underlying(type) / 2;
                            IntS const b1 = std::to_underlying(type) % 2;

                            // G += Y{side, b0}^H * (variance^-1) * Y{side, b1} // NOSONAR(S125)
                            block.g() += dot(hermitian_transpose(param.branch_param[obj].value[measured_side * 2 + b0]),
                                             diagonal_inverse(branch_current.variance),
                                             param.branch_param[obj].value[measured_side * 2 + b1]);
                        };
                        // measured at from-side: 0, to-side: 1
                        for (IntS const measured_side : std::array<IntS, 2>{0, 1}) {
                            // has measurement
                            if (std::invoke(has_branch_power_[measured_side], measured_value, obj)) {
                                auto const& branch_power =
                                    std::invoke(branch_power_[measured_side], measured_value, obj);
                                add_branch_measurement(measured_side,
                                                       power_to_global_current_measurement(branch_power));
                            }
                            if (std::invoke(has_branch_current_[measured_side], measured_value, obj)) {
                                auto const& branch_current =
                                    std::invoke(branch_current_[measured_side], measured_value, obj);
                                add_branch_measurement(measured_side,
                                                       current_to_global_current_measurement(branch_current));
                            }
                        }
                    }
                }
                // fill block with injection measurement
                // injection measurement exist
                if (measured_value.has_bus_injection(row)) {
                    // Q_ij = Y_bus_ij // NOSONAR(S125)
                    block.q() = y_bus.admittance()[data_idx];
                    // R_ii = -variance, only diagonal
                    if (row == col) {
                        // assign variance to diagonal of 3x3 tensor, for asym
                        auto const& injection = measured_value.bus_injection(row);
                        block.r() = ComplexTensor<sym>{
                            static_cast<ComplexValue<sym>>(-power_to_global_current_measurement(injection).variance)};
                    }
                }
                // injection measurement not exist
                else {
                    // Q_ij = 0
                    // R_ii = -1.0, only diagonal
                    // assign -1.0 to diagonal of 3x3 tensor, for asym
                    if (row == col) {
                        block.r() = ComplexTensor<sym>{-1.0};
                    }
                }
            }
        }

        // loop all transpose entry for QH
        // assign the hermitian transpose of the transpose entry of Q
        for (Idx data_idx_lu = 0; data_idx_lu != y_bus.nnz_lu(); ++data_idx_lu) {
            // skip for fill-in
            if (y_bus.map_lu_y_bus()[data_idx_lu] == -1) {
                continue;
            }
            Idx const data_idx_tranpose = y_bus.lu_transpose_entry()[data_idx_lu];
            data_gain_[data_idx_lu].qh() = hermitian_transpose(data_gain_[data_idx_tranpose].q());
        }
    }

    void prepare_rhs(YBus<sym> const& y_bus, MeasuredValues<sym> const& measured_value,
                     ComplexValueVector<sym> const& current_u) {
        MathModelParam<sym> const& param = y_bus.math_model_param();
        std::vector<BranchIdx> const branch_bus_idx = y_bus.math_topology().branch_bus_idx;
        // get generated (measured/estimated) voltage phasor
        // with current result voltage angle
        ComplexValueVector<sym> u = linearize_measurements(current_u, measured_value);

        // loop all bus to fill rhs
        for (Idx bus = 0; bus != n_bus_; ++bus) {
            Idx const data_idx = y_bus.bus_entry()[bus];
            // reset rhs block to fill values
            ILSERhs<sym>& rhs_block = x_rhs_[bus];
            rhs_block.clear();
            // fill block with voltage measurement
            if (measured_value.has_voltage(bus)) {
                // eta += u / variance
                rhs_block.eta() += u[bus] / measured_value.voltage_var(bus);
            }
            // fill block with branch, shunt measurement, need to convert to current
            for (Idx element_idx = y_bus.y_bus_entry_indptr()[data_idx];
                 element_idx != y_bus.y_bus_entry_indptr()[data_idx + 1]; ++element_idx) {
                Idx const obj = y_bus.y_bus_element()[element_idx].idx;
                YBusElementType const type = y_bus.y_bus_element()[element_idx].element_type;
                // shunt
                if (type == YBusElementType::shunt) {
                    if (measured_value.has_shunt(obj)) {
                        PowerSensorCalcParam<sym> const& shunt_power = measured_value.shunt_power(obj);
                        auto const current = power_to_global_current_measurement(shunt_power, u[bus]);
                        // eta += (-Ys)^H * (variance^-1) * i_shunt
                        rhs_block.eta() -= dot(hermitian_transpose(param.shunt_param[obj]),
                                               diagonal_inverse(current.variance), current.value);
                    }
                }
                // branch
                else {
                    auto const add_branch_measurement = [&rhs_block, &param, obj,
                                                         type](IntS measured_side,
                                                               IndependentComplexRandVar<sym> const& branch_current) {
                        // branch is either ff or tt
                        IntS const b = std::to_underlying(type) / 2;
                        // eta += Y{side, b}^H * (variance^-1) * i_branch_{f, t}
                        rhs_block.eta() +=
                            dot(hermitian_transpose(param.branch_param[obj].value[measured_side * 2 + b]),
                                diagonal_inverse(branch_current.variance), branch_current.value);
                    };
                    // measured at from-side: 0, to-side: 1
                    for (IntS const measured_side : std::array<IntS, 2>{0, 1}) {
                        // the current needs to be calculated with the voltage of the measured bus side
                        // NOTE: not the bus that is currently being processed!
                        Idx const measured_bus = branch_bus_idx[obj][measured_side];

                        // has measurement
                        if (std::invoke(has_branch_power_[measured_side], measured_value, obj)) {
                            auto const& branch_power = std::invoke(branch_power_[measured_side], measured_value, obj);
                            add_branch_measurement(measured_side,
                                                   power_to_global_current_measurement(branch_power, u[measured_bus]));
                        }
                        if (std::invoke(has_branch_current_[measured_side], measured_value, obj)) {
                            auto const& branch_current =
                                std::invoke(branch_current_[measured_side], measured_value, obj);
                            add_branch_measurement(
                                measured_side, current_to_global_current_measurement(branch_current, u[measured_bus]));
                        }
                    }
                }
            }
            // fill block with injection measurement, need to convert to current
            if (measured_value.has_bus_injection(bus)) {
                rhs_block.tau() = power_to_global_current_measurement(measured_value.bus_injection(bus), u[bus]).value;
            }
        }
    }

    double iterate_unknown(ComplexValueVector<sym>& u, bool has_angle) {
        double max_dev = 0.0;
        // phase shift anti offset of slack bus, phase a
        // if no angle measurement is present
        DoubleComplex const angle_offset = [&]() -> DoubleComplex {
            if (has_angle) {
                return 1.0;
            }
            auto const& voltage = x_rhs_[math_topo_.get().slack_bus].u();
            auto const& voltage_a = [&voltage]() -> auto const& {
                if constexpr (is_symmetric_v<sym>) {
                    return voltage;
                } else {
                    return voltage(0);
                }
            }();
            return cabs(voltage_a) / voltage_a;
        }();

        for (Idx bus = 0; bus != n_bus_; ++bus) {
            // phase offset to calculated voltage as normalized
            ComplexValue<sym> const u_normalized = x_rhs_[bus].u() * angle_offset;
            // get dev of last iteration, get max
            double const dev = max_val(cabs(u_normalized - u[bus]));
            max_dev = std::max(dev, max_dev);
            // assign
            u[bus] = u_normalized;
        }
        return max_dev;
    }

    auto linearize_measurements(ComplexValueVector<sym> const& current_u,
                                MeasuredValues<sym> const& measured_values) const {
        return measured_values.combine_voltage_iteration_with_measurements(current_u);
    }

    static PhaseVector as_phase_vector(ComplexValue<sym> const& value) {
        PhaseVector result;
        if constexpr (is_symmetric_v<sym>) {
            result(0) = value;
        } else {
            result = value.matrix();
        }
        return result;
    }

    static PhaseMatrix as_phase_matrix(ComplexTensor<sym> const& value) {
        PhaseMatrix result;
        if constexpr (is_symmetric_v<sym>) {
            result(0, 0) = value;
        } else {
            result = value.matrix();
        }
        return result;
    }

    static void set_phase_value(RealValue<sym>& value, Idx phase, double phase_value) {
        if constexpr (is_symmetric_v<sym>) {
            (void)phase;
            value = phase_value;
        } else {
            value(phase) = phase_value;
        }
    }

    static void set_eta_phase(ILSERhs<sym>& value, Idx phase, DoubleComplex phase_value) {
        if constexpr (is_symmetric_v<sym>) {
            (void)phase;
            value.eta() = phase_value;
        } else {
            value.eta()(phase) = phase_value;
        }
    }

    static void add_eta_phase(ILSERhs<sym>& value, Idx phase, DoubleComplex phase_value) {
        if constexpr (is_symmetric_v<sym>) {
            (void)phase;
            value.eta() += phase_value;
        } else {
            value.eta()(phase) += phase_value;
        }
    }

    static DoubleComplex get_eta_phase(ILSERhs<sym>& value, Idx phase) {
        if constexpr (is_symmetric_v<sym>) {
            (void)phase;
            return value.eta();
        } else {
            return value.eta()(phase);
        }
    }

    static DoubleComplex get_u_phase(ILSERhs<sym>& value, Idx phase) {
        if constexpr (is_symmetric_v<sym>) {
            (void)phase;
            return value.u();
        } else {
            return value.u()(phase);
        }
    }

    static double variance_to_sigma(double variance, double scale) {
        double const tolerance = 1000.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, scale);
        if (!std::isfinite(variance) || variance < -tolerance) {
            return nan;
        }
        return std::sqrt(std::max(0.0, variance));
    }

    static Idx find_lu_entry(YBus<sym> const& y_bus, Idx row, Idx col) {
        for (Idx idx = y_bus.row_indptr_lu()[row]; idx != y_bus.row_indptr_lu()[row + 1]; ++idx) {
            if (y_bus.col_indices_lu()[idx] == col) {
                return idx;
            }
        }
        throw SparseMatrixError{};
    }

    PhaseMatrix selected_voltage_covariance(YBus<sym> const& y_bus, Idx row, Idx col, double variance_normalization) {
        if (row == disconnected || col == disconnected) {
            return PhaseMatrix::Zero();
        }

        Idx const idx = find_lu_entry(y_bus, row, col);
        PhaseMatrix result;
        if constexpr (is_symmetric_v<sym>) {
            result(0, 0) = data_gain_[idx].g();
        } else {
            result = data_gain_[idx].g().matrix();
        }
        return variance_normalization * result;
    }

    void calculate_bus_injection_uncertainty(YBus<sym> const& y_bus, double variance_normalization,
                                             SolverOutput<sym>& output) {
        std::vector<ILSERhs<sym>> rhs_h(n_bus_);
        std::vector<ILSERhs<sym>> rhs_k(n_bus_);
        std::vector<ILSERhs<sym>> solution_h(n_bus_);
        std::vector<ILSERhs<sym>> solution_k(n_bus_);

        for (Idx bus = 0; bus != n_bus_; ++bus) {
            PhaseVector current = PhaseVector::Zero();
            for (Idx idx = y_bus.row_indptr()[bus]; idx != y_bus.row_indptr()[bus + 1]; ++idx) {
                current +=
                    as_phase_matrix(y_bus.admittance()[idx]) * as_phase_vector(output.u[y_bus.col_indices()[idx]]);
            }
            PhaseVector const voltage = as_phase_vector(output.u[bus]);

            for (Idx phase = 0; phase != n_phase_; ++phase) {
                std::ranges::for_each(rhs_h, [](auto& value) { value.clear(); });
                std::ranges::for_each(rhs_k, [](auto& value) { value.clear(); });

                // h = conj(I_i,k) * T_i,k, so h^H has I_i,k at the selected state entry.
                set_eta_phase(rhs_h[bus], phase, current(phase));

                // k_j,q = U_i,k * conj(Y_i,j[k,q]); the solve needs the column k^T.
                for (Idx idx = y_bus.row_indptr()[bus]; idx != y_bus.row_indptr()[bus + 1]; ++idx) {
                    Idx const col = y_bus.col_indices()[idx];
                    PhaseMatrix const admittance = as_phase_matrix(y_bus.admittance()[idx]);
                    for (Idx col_phase = 0; col_phase != n_phase_; ++col_phase) {
                        add_eta_phase(rhs_k[col], col_phase, voltage(phase) * std::conj(admittance(phase, col_phase)));
                    }
                }

                sparse_solver_.solve_with_prefactorized_matrix(data_gain_, perm_, rhs_h, solution_h);
                sparse_solver_.solve_with_prefactorized_matrix(data_gain_, perm_, rhs_k, solution_k);

                DoubleComplex const h = std::conj(current(phase));
                DoubleComplex const h_v_hh = h * get_u_phase(solution_h[bus], phase);
                DoubleComplex const h_v_kt = h * get_u_phase(solution_k[bus], phase);
                DoubleComplex k_conj_v_kt{};
                for (Idx col = 0; col != n_bus_; ++col) {
                    for (Idx col_phase = 0; col_phase != n_phase_; ++col_phase) {
                        k_conj_v_kt +=
                            std::conj(get_eta_phase(rhs_k[col], col_phase)) * get_u_phase(solution_k[col], col_phase);
                    }
                }

                double const covariance = std::real(h_v_hh + std::conj(k_conj_v_kt));
                double const pseudo_covariance = 2.0 * std::real(h_v_kt);
                double const p_variance = 0.5 * variance_normalization * (covariance + pseudo_covariance);
                double const q_variance = 0.5 * variance_normalization * (covariance - pseudo_covariance);
                double const scale =
                    0.5 * variance_normalization * (std::abs(covariance) + std::abs(pseudo_covariance));
                set_phase_value(output.bus_uncertainty[bus].p_sigma, phase, variance_to_sigma(p_variance, scale));
                set_phase_value(output.bus_uncertainty[bus].q_sigma, phase, variance_to_sigma(q_variance, scale));
            }
        }
    }

    std::vector<PhaseVector> calculate_reference_voltage_covariance(bool has_angle_measurement) {
        if (has_angle_measurement) {
            return {};
        }

        std::vector<ILSERhs<sym>> rhs(n_bus_);
        std::vector<ILSERhs<sym>> solution(n_bus_);
        std::ranges::for_each(rhs, [](auto& value) { value.clear(); });
        set_eta_phase(rhs[math_topo_.get().slack_bus], 0, 1.0);
        sparse_solver_.solve_with_prefactorized_matrix(data_gain_, perm_, rhs, solution);

        std::vector<PhaseVector> covariance_column(n_bus_);
        for (Idx bus = 0; bus != n_bus_; ++bus) {
            for (Idx phase = 0; phase != n_phase_; ++phase) {
                covariance_column[bus](phase) = get_u_phase(solution[bus], phase);
            }
        }
        return covariance_column;
    }

    void calculate_voltage_uncertainty(YBus<sym> const& y_bus, double variance_normalization,
                                       std::vector<PhaseVector> const& reference_covariance,
                                       SolverOutput<sym>& output) {
        bool const uses_slack_angle_reference = !reference_covariance.empty();
        Idx const slack_bus = math_topo_.get().slack_bus;
        PhaseVector const slack_voltage = as_phase_vector(output.u[slack_bus]);
        double const slack_voltage_abs = std::abs(slack_voltage(0));
        PhaseMatrix const slack_covariance =
            selected_voltage_covariance(y_bus, slack_bus, slack_bus, variance_normalization);
        double const slack_angle_variance =
            uses_slack_angle_reference && slack_voltage_abs > 0.0
                ? 0.5 * std::real(slack_covariance(0, 0)) / (slack_voltage_abs * slack_voltage_abs)
                : nan;

        for (Idx bus = 0; bus != n_bus_; ++bus) {
            PhaseMatrix const covariance = selected_voltage_covariance(y_bus, bus, bus, variance_normalization);
            PhaseVector const voltage = as_phase_vector(output.u[bus]);
            for (Idx phase = 0; phase != n_phase_; ++phase) {
                double const voltage_abs = std::abs(voltage(phase));
                if (voltage_abs == 0.0) {
                    continue;
                }
                double const diagonal = std::real(covariance(phase, phase));
                double const magnitude_sigma = variance_to_sigma(0.5 * diagonal, 0.5 * std::abs(diagonal));
                set_phase_value(output.bus_uncertainty[bus].u_sigma, phase, magnitude_sigma);

                if (!uses_slack_angle_reference) {
                    set_phase_value(output.bus_uncertainty[bus].u_angle_sigma, phase, magnitude_sigma / voltage_abs);
                    continue;
                }
                if (bus == slack_bus && phase == 0) {
                    set_phase_value(output.bus_uncertainty[bus].u_angle_sigma, phase, 0.0);
                    continue;
                }
                if (slack_voltage_abs == 0.0) {
                    continue;
                }

                DoubleComplex const voltage_direction = voltage(phase) / voltage_abs;
                DoubleComplex const slack_voltage_direction = slack_voltage(0) / slack_voltage_abs;
                DoubleComplex const covariance_with_slack = variance_normalization * reference_covariance[bus](phase);
                double const angle_variance =
                    0.5 * diagonal / (voltage_abs * voltage_abs) + slack_angle_variance -
                    std::real(std::conj(voltage_direction) * covariance_with_slack * slack_voltage_direction) /
                        (voltage_abs * slack_voltage_abs);
                double const angle_scale = 0.5 * std::abs(diagonal) / (voltage_abs * voltage_abs) +
                                           std::abs(slack_angle_variance) +
                                           std::abs(covariance_with_slack) / (voltage_abs * slack_voltage_abs);
                set_phase_value(output.bus_uncertainty[bus].u_angle_sigma, phase,
                                variance_to_sigma(angle_variance, angle_scale));
            }
        }
    }

    void calculate_branch_side_uncertainty(TerminalStateMatrix const& covariance, TerminalJacobian const& current_jac,
                                           PhaseVector const& terminal_voltage, PhaseVector const& current,
                                           RealValue<sym>& p_sigma, RealValue<sym>& q_sigma, RealValue<sym>& i_sigma,
                                           Idx terminal_side) {
        auto const current_covariance = current_jac * covariance * current_jac.adjoint();

        TerminalJacobian h = TerminalJacobian::Zero();
        h.template middleCols<n_phase_>(terminal_side * n_phase_) = current.conjugate().asDiagonal();
        TerminalJacobian const k = terminal_voltage.asDiagonal() * current_jac.conjugate();
        auto const power_covariance = h * covariance * h.adjoint() + k * covariance.conjugate() * k.adjoint();
        auto const power_pseudo_covariance =
            h * covariance * k.transpose() + k * covariance.conjugate() * h.transpose();

        for (Idx phase = 0; phase != n_phase_; ++phase) {
            double const current_diagonal = std::real(current_covariance(phase, phase));
            if (std::abs(current(phase)) > 0.0) {
                set_phase_value(i_sigma, phase,
                                variance_to_sigma(0.5 * current_diagonal, 0.5 * std::abs(current_diagonal)));
            }

            double const power_diagonal = std::real(power_covariance(phase, phase));
            double const power_pseudo_diagonal = std::real(power_pseudo_covariance(phase, phase));
            double const scale = 0.5 * (std::abs(power_diagonal) + std::abs(power_pseudo_diagonal));
            set_phase_value(p_sigma, phase, variance_to_sigma(0.5 * (power_diagonal + power_pseudo_diagonal), scale));
            set_phase_value(q_sigma, phase, variance_to_sigma(0.5 * (power_diagonal - power_pseudo_diagonal), scale));
        }
    }

    void calculate_branch_uncertainty(YBus<sym> const& y_bus, double variance_normalization,
                                      SolverOutput<sym>& output) {
        auto const& branch_bus_idx = y_bus.math_topology().branch_bus_idx;
        auto const& branch_param = y_bus.math_model_param().branch_param;
        for (Idx branch = 0; branch != std::ssize(branch_bus_idx); ++branch) {
            auto const [from, to] = branch_bus_idx[branch];

            TerminalStateMatrix covariance = TerminalStateMatrix::Zero();
            covariance.template block<n_phase_, n_phase_>(0, 0) =
                selected_voltage_covariance(y_bus, from, from, variance_normalization);
            covariance.template block<n_phase_, n_phase_>(0, n_phase_) =
                selected_voltage_covariance(y_bus, from, to, variance_normalization);
            covariance.template block<n_phase_, n_phase_>(n_phase_, 0) =
                selected_voltage_covariance(y_bus, to, from, variance_normalization);
            covariance.template block<n_phase_, n_phase_>(n_phase_, n_phase_) =
                selected_voltage_covariance(y_bus, to, to, variance_normalization);

            PhaseVector const from_voltage =
                from == disconnected ? PhaseVector::Zero() : as_phase_vector(output.u[from]);
            PhaseVector const to_voltage = to == disconnected ? PhaseVector::Zero() : as_phase_vector(output.u[to]);
            auto& branch_output = output.branch[branch];

            TerminalJacobian from_current_jac;
            from_current_jac.template leftCols<n_phase_>() = as_phase_matrix(branch_param[branch].yff());
            from_current_jac.template rightCols<n_phase_>() = as_phase_matrix(branch_param[branch].yft());
            calculate_branch_side_uncertainty(covariance, from_current_jac, from_voltage,
                                              as_phase_vector(branch_output.i_f), branch_output.p_f_sigma,
                                              branch_output.q_f_sigma, branch_output.i_f_sigma, 0);

            TerminalJacobian to_current_jac;
            to_current_jac.template leftCols<n_phase_>() = as_phase_matrix(branch_param[branch].ytf());
            to_current_jac.template rightCols<n_phase_>() = as_phase_matrix(branch_param[branch].ytt());
            calculate_branch_side_uncertainty(covariance, to_current_jac, to_voltage,
                                              as_phase_vector(branch_output.i_t), branch_output.p_t_sigma,
                                              branch_output.q_t_sigma, branch_output.i_t_sigma, 1);
        }
    }

    void calculate_state_estimation_uncertainty(YBus<sym> const& y_bus, MeasuredValues<sym> const& measured_values,
                                                SolverOutput<sym>& output) {
        // This propagation adopts PGM's circular/proper effective complex-error model. The selected inverse
        // supplies Cov(U); voltage/current magnitudes and P/Q are first-order marginals at the final IL point.
        if (sparse_solver_.has_pivot_perturbation()) {
            throw SparseMatrixError{};
        }

        output.bus_uncertainty.resize(n_bus_);
        double const variance_normalization = measured_values.variance_normalization();

        // Injection Jacobians span closed bus neighborhoods and need covariance blocks outside the selected
        // inverse pattern. Obtain their marginal quadratic forms with prefactorized solves first.
        calculate_bus_injection_uncertainty(y_bus, variance_normalization, output);

        // Without an angle measurement, IL reports every phase relative to the slack phase-a angle. Preserve
        // the matching cross-covariances before the destructive selected-inverse sweep.
        auto const reference_covariance = calculate_reference_voltage_covariance(measured_values.has_angle());

        // The sweep is destructive, so it must follow every solve that uses the prefactorized augmented matrix.
        sparse_solver_.inplace_selective_inverse_with_prefactorized_matrix(data_gain_, perm_);
        calculate_voltage_uncertainty(y_bus, variance_normalization, reference_covariance, output);
        calculate_branch_uncertainty(y_bus, variance_normalization, output);
    }

    // The variance is not scaled as an approximation under the assumptions of:
    // - linearization of obtaining the current from power measurements
    // - voltages are ~1pu
    // - power sensor variances are often an approximation dominated by heuristics in the first place
    // See also https://github.com/PowerGridModel/power-grid-model/pull/951#issuecomment-2805154436
    IndependentComplexRandVar<sym>
    power_to_global_current_measurement(PowerSensorCalcParam<sym> const& power_measurement,
                                        ComplexValue<sym> const& voltage) const {
        auto measurement = static_cast<IndependentComplexRandVar<sym>>(power_measurement);
        measurement.value = conj(measurement.value / voltage);
        return measurement;
    }

    // Overload when the voltage is not present: the value can't be determined, but the variance assumption still holds.
    // The variance is not scaled as an approximation under the assumptions of:
    // - linearization of obtaining the current from power measurements
    // - voltages are ~1pu
    // - power sensor variances are often an approximation dominated by heuristics in the first place
    // See also https://github.com/PowerGridModel/power-grid-model/pull/951#issuecomment-2805154436
    IndependentComplexRandVar<sym>
    power_to_global_current_measurement(PowerSensorCalcParam<sym> const& power_measurement) const {
        auto measurement = static_cast<IndependentComplexRandVar<sym>>(power_measurement);
        measurement.value = ComplexValue<sym>{nan};
        return measurement;
    }

    IndependentComplexRandVar<sym>
    current_to_global_current_measurement(CurrentSensorCalcParam<sym> const& current_measurement,
                                          ComplexValue<sym> const& voltage) const {
        using statistics::scale;

        auto measurement = static_cast<IndependentComplexRandVar<sym>>(current_measurement.measurement);

        switch (current_measurement.angle_measurement_type) {
        case AngleMeasurementType::global_angle:
            return measurement; // no offset
        case AngleMeasurementType::local_angle:
            return scale(conj(measurement),
                         phase_shift(voltage)); // offset with the phase shift
        default:
            throw MissingCaseForEnumError{"AngleMeasurementType", current_measurement.angle_measurement_type};
        }
    }

    // Overload when the voltage is not present: the value can't be determined, but the variance assumption still holds.
    IndependentComplexRandVar<sym>
    current_to_global_current_measurement(CurrentSensorCalcParam<sym> const& current_measurement) const {
        auto measurement = static_cast<IndependentComplexRandVar<sym>>(current_measurement.measurement);
        measurement.value = ComplexValue<sym>{nan};
        return measurement;
    }
};

} // namespace iterative_linear_se

using iterative_linear_se::IterativeLinearSESolver;

} // namespace power_grid_model::math_solver
