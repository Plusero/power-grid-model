// SPDX-FileCopyrightText: Contributors to the Power Grid Model project
// <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#include <power_grid_model/auxiliary/meta_data_gen.hpp>
#include <power_grid_model/main_model.hpp>
#include <power_grid_model/math_solver/math_solver.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <vector>

namespace power_grid_model {
namespace {
using namespace math_solver;
struct RecordedUQ {
    bool requested = false;
    StateEstimationUQRequest request;
    StateEstimationUQResult<symmetric_t> result;
    ComplexValueVector<symmetric_t> voltage;
    MathModelTopology topology;
};
std::vector<RecordedUQ> recorded;

// Existing dispatcher injection permits observing the real synchronous call
// without changing production objects or adding a public diagnostics schema.
template <symmetry_tag Sym> class RecordingSolver : public MathSolverBase<Sym> {
  public:
    explicit RecordingSolver(std::shared_ptr<MathModelTopology const> const& topology) : solver_{topology} {}
    SolverOutput<Sym> run_power_flow(PowerFlowInput<Sym> const& input, double tolerance, Idx iterations, bool cache_run,
                                     Logger& logger, CalculationMethod method, YBus<Sym> const& y_bus) override {
        return solver_.run_power_flow(input, tolerance, iterations, cache_run, logger, method, y_bus);
    }
    ShortCircuitSolverOutput<Sym> run_short_circuit(ShortCircuitInput const& input, Logger& logger,
                                                    CalculationMethod method, YBus<Sym> const& y_bus) override {
        return solver_.run_short_circuit(input, logger, method, y_bus);
    }
    void clear_solver() override { solver_.clear_solver(); }
    void parameters_changed(bool changed) override { solver_.parameters_changed(changed); }
    MathSolverBase<Sym>* clone() const override { return new RecordingSolver{*this}; }
    SolverOutput<Sym> run_state_estimation(StateEstimationInput<Sym> const& input, double tolerance, Idx iterations,
                                           Logger& logger, CalculationMethod method, YBus<Sym> const& y_bus,
                                           StateEstimationUQContext<Sym>* uq = nullptr) override {
        // Opt in to pair diagnostics so the actual completion routes are observable.
        std::optional<StateEstimationUQRequest> diagnosed;
        std::optional<StateEstimationUQContext<Sym>> context;
        if (uq) {
            diagnosed = uq->request;
            diagnosed->pair_diagnostics = true;
            context.emplace(*diagnosed, uq->result);
        }
        auto output = solver_.run_state_estimation(input, tolerance, iterations, logger, method, y_bus,
                                                   context ? &*context : nullptr);
        if constexpr (is_symmetric_v<Sym>) {
            RecordedUQ item;
            item.requested = uq != nullptr;
            if (uq) {
                item.request = uq->request;
                item.result = uq->result;
            }
            item.voltage = output.u;
            item.topology = y_bus.math_topology();
            recorded.push_back(std::move(item));
        }
        return output;
    }

  private:
    MathSolver<Sym> solver_;
};

struct PublicNetwork {
    std::vector<NodeInput> nodes;
    std::vector<LineInput> lines;
    std::vector<SourceInput> sources;
    std::vector<SymLoadGenInput> loads;
    std::vector<SymVoltageSensorInput> voltage_sensors;
    std::vector<SymPowerSensorInput> power_sensors;
    std::vector<VoltageUncertaintyObserverInput> voltage_observers;
    std::vector<PowerUncertaintyObserverInput> power_observers;
    ConstDataset data{false, 1, "input", meta_data::meta_data_gen::meta_data};
    explicit PublicNetwork(bool cycle, bool include_observers = true) {
        for (ID i = 0; i < 4; ++i) {
            nodes.push_back({.id = i + 1, .u_rated = 10000.0});
            voltage_sensors.push_back({.id = 31 + i,
                                       .measured_object = i + 1,
                                       .u_sigma = 100.0 + 10 * i,
                                       .u_measured = 10000.0 + 50 * i,
                                       .u_angle_measured = 0.1 - 0.03 * i});
            for (ID duplicate = 0; duplicate < 3; ++duplicate) {
                voltage_observers.push_back({.id = 100 + 3 * i + duplicate, .observed_object = i + 1, .status = 1});
                power_observers.push_back({.id = 200 + 3 * i + duplicate,
                                           .observed_object = i + 1,
                                           .status = 1,
                                           .observed_terminal_type = ObservedTerminalType::node_injection});
            }
        }
        std::vector<std::array<ID, 2>> const edges =
            cycle ? std::vector<std::array<ID, 2>>{{1, 2}, {2, 3}, {3, 4}, {4, 1}}
                  : std::vector<std::array<ID, 2>>{{1, 4}, {2, 4}, {3, 4}};
        for (auto const& edge : edges) {
            lines.push_back({.id = 10 + static_cast<ID>(lines.size()),
                             .from_node = edge[0],
                             .to_node = edge[1],
                             .from_status = 1,
                             .to_status = 1,
                             .r1 = 1.0,
                             .x1 = 2.0,
                             .c1 = 0.0,
                             .tan1 = 0.0,
                             .r0 = 1.0,
                             .x0 = 2.0,
                             .c0 = 0.0,
                             .tan0 = 0.0,
                             .i_n = 1000.0});
        }
        sources.push_back({.id = 20, .node = 1, .status = 1, .u_ref = 1.0});
        power_sensors.push_back({.id = 41,
                                 .measured_object = 20,
                                 .measured_terminal_type = MeasuredTerminalType::source,
                                 .power_sigma = 1e6,
                                 .p_measured = 1e5,
                                 .q_measured = 1e4});
        for (ID i = 0; i < 3; ++i) {
            loads.push_back({.id = 21 + i,
                             .node = i + 2,
                             .status = 1,
                             .type = LoadGenType::const_pq,
                             .p_specified = 1e5,
                             .q_specified = 1e4});
            power_sensors.push_back({.id = 42 + i,
                                     .measured_object = 21 + i,
                                     .measured_terminal_type = MeasuredTerminalType::load,
                                     .power_sigma = 1e6,
                                     .p_measured = 1e5,
                                     .q_measured = 1e4});
        }
        auto add = [&](char const* name, auto const& entries) {
            data.add_buffer(name, static_cast<Idx>(entries.size()), static_cast<Idx>(entries.size()), nullptr,
                            entries.data());
        };
        add("node", nodes);
        add("line", lines);
        add("source", sources);
        add("sym_load", loads);
        add("sym_voltage_sensor", voltage_sensors);
        add("sym_power_sensor", power_sensors);
        if (include_observers) {
            add("voltage_uncertainty_observer", voltage_observers);
            add("power_uncertainty_observer", power_observers);
        }
    }
};
} // namespace

TEST_CASE("Public observer demand deduplicates and records actual separator or "
          "augmented approximation routes") {
    MathSolverDispatcher const dispatcher{math_solver_tag<RecordingSolver>{}};
    for (bool const cycle : {false, true}) {
        for (auto const method : {CalculationMethod::iterative_linear, CalculationMethod::newton_raphson}) {
            PublicNetwork fixture{cycle};
            MainModel model{50.0, fixture.data, dispatcher};
            MainModelOptions options;
            options.calculation_type = CalculationType::state_estimation;
            options.calculation_method = method;
            options.err_tol = 1e-10;
            options.max_iter = 100;
            ConstDataset updates{false, 1, "update", meta_data::meta_data_gen::meta_data};
            std::vector<VoltageUncertaintyObserverOutput> voltage(12);
            std::vector<PowerUncertaintyObserverOutput> power(12);
            MutableDataset output{false, 1, "sym_output", meta_data::meta_data_gen::meta_data};
            std::vector<ID> metadata(12);
            output.add_buffer("voltage_uncertainty_observer", 12, 12, nullptr, nullptr);
            output.add_attribute_buffer("voltage_uncertainty_observer", "id", metadata.data());
            recorded.clear();
            model.calculate(options, output, updates);
            REQUIRE(recorded.size() == 1);
            CHECK_FALSE(recorded.back().requested);
            auto const baseline = recorded.back().voltage;
            // Row buffers now demand all voltage sigmas, but no injection pairs.
            MutableDataset voltage_output{false, 1, "sym_output", meta_data::meta_data_gen::meta_data};
            voltage_output.add_buffer("voltage_uncertainty_observer", 12, 12, nullptr, voltage.data());
            recorded.clear();
            model.calculate(options, voltage_output, updates);
            REQUIRE(recorded.size() == 1);
            CHECK(recorded.back().voltage == baseline);
            REQUIRE(recorded.back().requested);
            CHECK(recorded.back().request.voltage.size() == 4);
            MutableDataset full{false, 1, "sym_output", meta_data::meta_data_gen::meta_data};
            full.add_buffer("voltage_uncertainty_observer", 12, 12, nullptr, voltage.data());
            full.add_buffer("power_uncertainty_observer", 12, 12, nullptr, power.data());
            recorded.clear();
            model.calculate(options, full, updates);
            REQUIRE(recorded.size() == 1);
            auto const& capture = recorded.back();
            CHECK(capture.voltage == baseline);
            CHECK(capture.request.voltage.size() == 4);
            CHECK(capture.request.injection.size() == 4);
            std::string provenance;
            for (auto const& edge : capture.topology.branch_bus_idx) {
                provenance += " (" + std::to_string(edge[0]) + "," + std::to_string(edge[1]) + ")";
            }
            for (auto const& pair : capture.result.pairs) {
                provenance += " pair=" + std::to_string(pair.first_bus) + "," + std::to_string(pair.second_bus) +
                              ":route=" + std::to_string(static_cast<int>(pair.route)) +
                              ":middle=" + std::to_string(pair.middle_bus);
            }
            INFO("cycle=" << cycle << " method=" << static_cast<int>(method)
                          << " mathematical branches/routes:" << provenance);
            auto const desired = cycle ? UQPairRoute::augmented_path : UQPairRoute::separator;
            CHECK(std::ranges::any_of(capture.result.pairs, [&](auto const& pair) { return pair.route == desired; }));
            // Record the actual mathematical topology rather than inferring a
            // separator from the physical graph before production reordering.
            REQUIRE(capture.topology.n_bus() == 4);
            if (cycle) {
                // Pin the mathematical order used by the independent public
                // Python augmented-block oracle, including both terminal ends.
                REQUIRE(capture.topology.branch_bus_idx.size() == 4);
                for (Idx branch = 0; branch < 4; ++branch) {
                    CHECK(capture.topology.branch_bus_idx[branch][0] == branch);
                    CHECK(capture.topology.branch_bus_idx[branch][1] == (branch + 1) % 4);
                }
                auto const pair = std::ranges::find_if(
                    capture.result.pairs, [](auto const& item) { return item.first_bus == 0 && item.second_bus == 2; });
                REQUIRE(pair != capture.result.pairs.end());
                CHECK(pair->route == UQPairRoute::augmented_path);
                CHECK(pair->middle_bus == 1);
            }
            for (std::size_t i = 0; i < 4; ++i) {
                for (std::size_t duplicate = 1; duplicate < 3; ++duplicate) {
                    CHECK(power[3 * i].p_sigma == power[3 * i + duplicate].p_sigma);
                    CHECK(power[3 * i].q_sigma == power[3 * i + duplicate].q_sigma);
                }
            }
            std::vector<double> core_sigmas;
            for (auto const& item : capture.result.injection) {
                core_sigmas.push_back(item.active_power.phase[0].sigma * base_power<symmetric_t>);
            }
            std::vector<double> public_sigmas;
            for (std::size_t i = 0; i < 4; ++i) {
                public_sigmas.push_back(power[3 * i].p_sigma);
            }
            std::ranges::sort(core_sigmas);
            std::ranges::sort(public_sigmas);
            for (std::size_t i = 0; i < 4; ++i) {
                CHECK(public_sigmas[i] == doctest::Approx(core_sigmas[i]).epsilon(1e-12));
            }
        }
    }
}

TEST_CASE("Public absent disabled and storage-free observer output perform no UQ") {
    MathSolverDispatcher const dispatcher{math_solver_tag<RecordingSolver>{}};
    for (auto const method : {CalculationMethod::iterative_linear, CalculationMethod::newton_raphson}) {
        MainModelOptions options;
        options.calculation_type = CalculationType::state_estimation;
        options.calculation_method = method;
        options.err_tol = 1e-10;
        options.max_iter = 100;
        ConstDataset updates{false, 1, "update", meta_data::meta_data_gen::meta_data};
        PublicNetwork absent{false, false};
        MainModel absent_model{50.0, absent.data, dispatcher};
        MutableDataset empty_output{false, 1, "sym_output", meta_data::meta_data_gen::meta_data};
        recorded.clear();
        absent_model.calculate(options, empty_output, updates);
        REQUIRE(recorded.size() == 1);
        CHECK_FALSE(recorded.back().requested);
        auto const baseline = recorded.back().voltage;
        PublicNetwork disabled{false};
        for (auto& observer : disabled.voltage_observers) {
            observer.status = 0;
        }
        for (auto& observer : disabled.power_observers) {
            observer.status = 0;
        }
        MainModel disabled_model{50.0, disabled.data, dispatcher};
        std::vector<VoltageUncertaintyObserverOutput> voltage(12);
        std::vector<PowerUncertaintyObserverOutput> power(12);
        MutableDataset full{false, 1, "sym_output", meta_data::meta_data_gen::meta_data};
        full.add_buffer("voltage_uncertainty_observer", 12, 12, nullptr, voltage.data());
        full.add_buffer("power_uncertainty_observer", 12, 12, nullptr, power.data());
        recorded.clear();
        disabled_model.calculate(options, full, updates);
        REQUIRE(recorded.size() == 1);
        CHECK_FALSE(recorded.back().requested);
        CHECK(recorded.back().voltage == baseline);
        for (std::size_t i = 0; i < voltage.size(); ++i) {
            CHECK(voltage[i].energized == 0);
            CHECK(std::isnan(voltage[i].u_sigma));
            CHECK(power[i].energized == 0);
            CHECK(std::isnan(power[i].p_sigma));
        }
        PublicNetwork enabled{false};
        MainModel enabled_model{50.0, enabled.data, dispatcher};
        MutableDataset no_storage{false, 1, "sym_output", meta_data::meta_data_gen::meta_data};
        no_storage.add_buffer("voltage_uncertainty_observer", 12, 12, nullptr, nullptr);
        no_storage.add_buffer("power_uncertainty_observer", 12, 12, nullptr, nullptr);
        recorded.clear();
        enabled_model.calculate(options, no_storage, updates);
        REQUIRE(recorded.size() == 1);
        CHECK_FALSE(recorded.back().requested);
        CHECK(recorded.back().voltage == baseline);
    }
}
} // namespace power_grid_model
