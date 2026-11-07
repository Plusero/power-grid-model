// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// Connects uncertainty observers to the state-estimation uncertainty calculation of the math solvers: which
// sigmas the output dataset asks for, which rows of each island's request they need, and the conversion of the
// per-unit results to observer output.

#include "container_queries.hpp"
#include "state_queries.hpp"

#include "../auxiliary/dataset.hpp"
#include "../component/current_uncertainty_observer.hpp"
#include "../component/power_uncertainty_observer.hpp"
#include "../component/voltage_uncertainty_observer.hpp"
#include "../main_model_fwd.hpp"
#include "../math_solver/state_estimation_uq.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <string_view>
#include <tuple>
#include <vector>

namespace power_grid_model::main_core {

namespace observer_uncertainty {

// The two sigma attributes of an observer type, in the order of the arguments of its get_output().
template <class ObserverType> inline constexpr std::array<std::string_view, 2> sigma_attributes{};
template <>
inline constexpr std::array<std::string_view, 2> sigma_attributes<VoltageUncertaintyObserver>{"u_sigma",
                                                                                              "u_angle_sigma"};
template <>
inline constexpr std::array<std::string_view, 2> sigma_attributes<PowerUncertaintyObserver>{"p_sigma", "q_sigma"};
template <>
inline constexpr std::array<std::string_view, 2> sigma_attributes<CurrentUncertaintyObserver>{"i_sigma",
                                                                                              "i_angle_sigma"};

// What the output dataset asks of one observer type.
struct OutputSelection {
    bool requested = false;      // the component has output storage
    std::array<bool, 2> sigma{}; // which of its sigma attributes have output storage
};

// A component that is registered with a count but without storage is not requested. A columnar component
// requests the attributes it has storage for, which may be none of the sigmas.
template <class ObserverType> OutputSelection output_selection(MutableDataset const& data) {
    Idx const index = data.find_component(ObserverType::name, false);
    if (index < 0 || data.get_component_info(index).total_elements == 0) {
        return {};
    }
    if (data.is_row_based(index)) {
        return {.requested = true, .sigma = {true, true}};
    }
    OutputSelection result;
    for (auto const& attribute : data.get_buffer(index).attributes) {
        if (attribute.data == nullptr) {
            continue;
        }
        result.requested = true;
        for (std::size_t i = 0; i != 2; ++i) {
            result.sigma[i] = result.sigma[i] || attribute.meta_attribute->name == sigma_attributes<ObserverType>[i];
        }
    }
    return result;
}

inline void validate_output_mode(MutableDataset const& data, MainModelOptions const& options) {
    bool const requested = output_selection<VoltageUncertaintyObserver>(data).requested ||
                           output_selection<PowerUncertaintyObserver>(data).requested ||
                           output_selection<CurrentUncertaintyObserver>(data).requested;
    bool const supported = options.calculation_type == CalculationType::state_estimation &&
                           (options.calculation_method == CalculationMethod::default_method ||
                            options.calculation_method == CalculationMethod::iterative_linear ||
                            options.calculation_method == CalculationMethod::newton_raphson);
    if (requested && !supported) {
        throw InvalidArguments{"Observer output requires iterative-linear or Newton-Raphson state estimation"};
    }
}

// Standard not-found/wrong-type errors, also for disabled observers. The
// constructors admit node injection only for power observers.
template <class Components, class ObserverType>
void validate_target(Components const& components, ObserverType const& observer) {
    ID const target = observer.observed_object();
    if constexpr (std::same_as<ObserverType, VoltageUncertaintyObserver>) {
        (void)get_component<Node>(components, target);
    } else {
        switch (observer.observed_terminal_type()) {
        case ObservedTerminalType::branch_from:
        case ObservedTerminalType::branch_to:
            (void)get_component<Branch>(components, target);
            break;
        case ObservedTerminalType::branch3_1:
        case ObservedTerminalType::branch3_2:
        case ObservedTerminalType::branch3_3:
            (void)get_component<Branch3>(components, target);
            break;
        case ObservedTerminalType::node_injection:
            (void)get_component<Node>(components, target);
            break;
        }
    }
}

// The pair of results an observer reports.
enum class Quantity { voltage, branch_power, branch_current, injection };

// Where the values of one observer come from.
struct Binding {
    Quantity quantity = Quantity::voltage;
    Idx island = -1;
    Idx math_index = -1; // bus or branch in the island's math model
    BranchSide side = BranchSide::from;
    bool energized = false;
    std::array<double, 2> scale{1.0, 1.0}; // from per unit or radian to the unit of each sigma attribute
    Idx result_index = -1;                 // row of the island's result; -1 if the observer requests nothing
};

template <class ObserverType> struct ObserverSet {
    OutputSelection selection;
    std::vector<Binding> bindings; // one per observer, in component order; empty if not requested
};

// The uncertainty work of one calculation.
template <symmetry_tag sym> struct Scenario {
    std::vector<math_solver::StateEstimationUQWork<sym>> work; // per island; empty if no sigma is needed
    std::tuple<ObserverSet<VoltageUncertaintyObserver>, ObserverSet<PowerUncertaintyObserver>,
               ObserverSet<CurrentUncertaintyObserver>>
        sets;

    explicit Scenario(MutableDataset const& data)
        : sets{ObserverSet<VoltageUncertaintyObserver>{output_selection<VoltageUncertaintyObserver>(data), {}},
               ObserverSet<PowerUncertaintyObserver>{output_selection<PowerUncertaintyObserver>(data), {}},
               ObserverSet<CurrentUncertaintyObserver>{output_selection<CurrentUncertaintyObserver>(data), {}}} {}

    template <class ObserverType> ObserverSet<ObserverType>& of() { return std::get<ObserverSet<ObserverType>>(sets); }
    template <class ObserverType> ObserverSet<ObserverType> const& of() const {
        return std::get<ObserverSet<ObserverType>>(sets);
    }
};

template <class Components> Binding locate_node(MainModelState<Components> const& state, ID node, Quantity quantity) {
    auto const math_id = get_user_node_math_id(state, get_component_sequence_idx<Node>(state.components, node));
    return {.quantity = quantity, .island = math_id.group, .math_index = math_id.pos, .energized = math_id.group >= 0};
}

// The observed node, with the voltage base of the calculation.
template <symmetry_tag sym, class Components>
Binding locate(MainModelState<Components> const& state, VoltageUncertaintyObserver const& observer) {
    Binding result = locate_node(state, observer.observed_object(), Quantity::voltage);
    result.scale[0] = u_scale<sym> * get_component<Node>(state.components, observer.observed_object()).u_rated();
    return result;
}

// The observed branch terminal or node injection of a power or current observer. Power is scaled with the
// base power of the calculation, and current with the base current of the terminal.
template <symmetry_tag sym, class Components, class ObserverType>
Binding locate(MainModelState<Components> const& state, ObserverType const& observer) {
    constexpr bool is_power = std::same_as<ObserverType, PowerUncertaintyObserver>;
    ID const object = observer.observed_object();
    Binding result{.quantity = is_power ? Quantity::branch_power : Quantity::branch_current};
    double base_i = 1.0;
    switch (observer.observed_terminal_type()) {
    case ObservedTerminalType::node_injection:
        result = locate_node(state, object, Quantity::injection);
        break;
    case ObservedTerminalType::branch_from:
    case ObservedTerminalType::branch_to: {
        auto const& branch = get_component<Branch>(state.components, object);
        auto const math_id = get_math_id<Branch>(state, get_component_sequence_idx<Branch>(state.components, object));
        result.side =
            observer.observed_terminal_type() == ObservedTerminalType::branch_from ? BranchSide::from : BranchSide::to;
        result.island = math_id.group;
        result.math_index = math_id.pos;
        result.energized = math_id.group >= 0 && branch.status(result.side);
        base_i = result.side == BranchSide::from ? branch.base_i_from() : branch.base_i_to();
        break;
    }
    case ObservedTerminalType::branch3_1:
    case ObservedTerminalType::branch3_2:
    case ObservedTerminalType::branch3_3: {
        // Each terminal of a three-winding transformer is the from side of its own math branch.
        Idx const terminal =
            static_cast<Idx>(observer.observed_terminal_type()) - static_cast<Idx>(ObservedTerminalType::branch3_1);
        auto const& branch = get_component<Branch3>(state.components, object);
        auto const math_id = get_math_id<Branch3>(state, get_component_sequence_idx<Branch3>(state.components, object));
        result.island = math_id.group;
        result.math_index = math_id.pos[terminal];
        result.energized =
            math_id.group >= 0 && math_id.pos[terminal] >= 0 && branch.status(static_cast<Branch3Side>(terminal));
        base_i = std::array{branch.base_i_1(), branch.base_i_2(), branch.base_i_3()}[terminal];
        break;
    }
    }
    result.scale = is_power ? std::array{base_power<sym>, base_power<sym>} : std::array{base_i, 1.0};
    return result;
}

// For each kind of request row of one island: the row of a bus or of a branch side, or -1 while there is none.
struct RequestRows {
    std::vector<Idx> voltage;
    std::vector<Idx> branch; // indexed by 2 * branch + side
    std::vector<Idx> injection;
};

template <class Row> Idx request_row(std::vector<Row>& rows, std::vector<Idx>& row_of, Idx key, Row const& empty) {
    if (key >= std::ssize(row_of)) {
        row_of.resize(static_cast<std::size_t>(key) + 1, -1);
    }
    if (row_of[key] < 0) {
        row_of[key] = std::ssize(rows);
        rows.push_back(empty);
    }
    return row_of[key];
}

// Adds the sigmas an observer needs to the request of its island and returns the row of their results.
// Observers of the same bus or branch side share a row, so nothing is calculated twice.
inline Idx add_request(math_solver::StateEstimationUQRequest& request, RequestRows& row_of, Binding const& binding,
                       std::array<bool, 2> sigma) {
    Idx const index = binding.math_index;
    Idx row = -1;
    switch (binding.quantity) {
    case Quantity::voltage: {
        row = request_row(request.voltage, row_of.voltage, index, {.bus = index, .magnitude = false, .angle = false});
        auto& item = request.voltage[row];
        item.magnitude = item.magnitude || sigma[0];
        item.angle = item.angle || sigma[1];
        break;
    }
    case Quantity::injection: {
        row = request_row(request.injection, row_of.injection, index,
                          {.bus = index, .active_power = false, .reactive_power = false});
        auto& item = request.injection[row];
        item.active_power = item.active_power || sigma[0];
        item.reactive_power = item.reactive_power || sigma[1];
        break;
    }
    case Quantity::branch_power:
    case Quantity::branch_current: {
        row = request_row(request.branch, row_of.branch, 2 * index + static_cast<Idx>(binding.side),
                          {.branch = index,
                           .side = binding.side,
                           .active_power = false,
                           .reactive_power = false,
                           .current_magnitude = false,
                           .current_angle = false});
        auto& item = request.branch[row];
        if (binding.quantity == Quantity::branch_power) {
            item.active_power = item.active_power || sigma[0];
            item.reactive_power = item.reactive_power || sigma[1];
        } else {
            item.current_magnitude = item.current_magnitude || sigma[0];
            item.current_angle = item.current_angle || sigma[1];
        }
        break;
    }
    }
    return row;
}

// The two results of a binding, in the order of the sigma attributes.
template <symmetry_tag sym>
std::array<math_solver::UQValue<sym> const*, 2> result_values(math_solver::StateEstimationUQResult<sym> const& result,
                                                              Binding const& binding) {
    if (binding.quantity == Quantity::voltage) {
        auto const& row = result.voltage[binding.result_index];
        return {&row.magnitude, &row.angle};
    }
    if (binding.quantity == Quantity::injection) {
        auto const& row = result.injection[binding.result_index];
        return {&row.active_power, &row.reactive_power};
    }
    auto const& row = result.branch[binding.result_index];
    if (binding.quantity == Quantity::branch_power) {
        return {&row.active_power, &row.reactive_power};
    }
    return {&row.current_magnitude, &row.current_angle};
}

inline double& phase_entry(double& value, std::size_t /*phase*/) { return value; }
inline double& phase_entry(RealValue<asymmetric_t>& value, std::size_t phase) { return value(static_cast<Idx>(phase)); }

} // namespace observer_uncertainty

template <class Components> void validate_observer_targets(Components const& components) {
    auto validate = [&]<class ObserverType>() {
        if constexpr (common::component_container_c<Components, ObserverType>) {
            for (auto const& observer : get_component_citer<ObserverType>(components)) {
                observer_uncertainty::validate_target(components, observer);
            }
        }
    };
    validate.template operator()<VoltageUncertaintyObserver>();
    validate.template operator()<PowerUncertaintyObserver>();
    validate.template operator()<CurrentUncertaintyObserver>();
}

// Binds every observer of a requested type to its place in the math model and builds the uncertainty request
// of each island. To be called once per scenario object, after the topology of the calculation is prepared.
template <symmetry_tag sym, class Components>
void prepare_observer_uncertainty(MainModelState<Components> const& state,
                                  observer_uncertainty::Scenario<sym>& scenario, Idx n_islands) {
    using namespace observer_uncertainty;
    std::vector<RequestRows> row_of; // per island

    // The bus of a node that closed links fuse with other nodes carries the injection of all of them, which is
    // not what an injection observer of one node asks for. Counted per calculation, as link statuses change.
    std::vector<Idx> nodes_per_topo_node;
    auto const is_fused = [&](ID node) {
        auto const& coupling = state.reduced_topology->topo_node_coup.coupling.user_nodes_to_topo_nodes;
        if (nodes_per_topo_node.empty()) {
            nodes_per_topo_node.resize(state.reduced_topology->topo_node_coup.topo_nodes.size(), 0);
            for (auto const& topo_node : coupling) {
                if (topo_node.group >= 0) {
                    ++nodes_per_topo_node[topo_node.group];
                }
            }
        }
        return nodes_per_topo_node[coupling[get_component_sequence_idx<Node>(state.components, node)].group] != 1;
    };

    auto const prepare = [&]<class ObserverType>(ObserverSet<ObserverType>& set) {
        if constexpr (common::component_container_c<Components, ObserverType>) {
            if (!set.selection.requested) {
                return;
            }
            bool const any_sigma = set.selection.sigma[0] || set.selection.sigma[1];
            set.bindings.reserve(get_component_size<ObserverType>(state.components));
            for (auto const& observer : get_component_citer<ObserverType>(state.components)) {
                Binding binding = locate<sym>(state, observer);
                binding.energized = binding.energized && observer.status();
                if (binding.energized && any_sigma) {
                    if (binding.quantity == Quantity::injection && is_fused(observer.observed_object())) {
                        throw InvalidArguments{
                            "Node injection observers do not support nodes fused by closed ideal links"};
                    }
                    if (scenario.work.empty()) {
                        scenario.work.resize(n_islands);
                        row_of.resize(n_islands);
                    }
                    binding.result_index = add_request(scenario.work[binding.island].request, row_of[binding.island],
                                                       binding, set.selection.sigma);
                }
                set.bindings.push_back(binding);
            }
        }
    };
    std::apply([&](auto&... set) { (prepare(set), ...); }, scenario.sets);
}

// Writes the output of all observers of one type. A sigma that was not requested or has no valid result is NaN.
template <class ObserverType, symmetry_tag sym, class Components, non_owning_view_c Outputs>
void output_observer_uncertainty(MainModelState<Components> const& state,
                                 observer_uncertainty::Scenario<sym> const& scenario, Outputs output) {
    using namespace observer_uncertainty;
    auto const& set = scenario.template of<ObserverType>();
    // Native callers may register a component count without any output storage.
    if (!set.selection.requested) {
        return;
    }
    Idx row = 0;
    for (auto const& observer : get_component_citer<ObserverType>(state.components)) {
        auto const& binding = set.bindings[row];
        std::array<RealValue<sym>, 2> sigma{RealValue<sym>{nan}, RealValue<sym>{nan}};
        if (binding.result_index >= 0) {
            auto const values = result_values(scenario.work[binding.island].result, binding);
            for (std::size_t i = 0; i != 2; ++i) {
                for (std::size_t phase = 0; phase != values[i]->phase.size(); ++phase) {
                    phase_entry(sigma[i], phase) = binding.scale[i] * values[i]->phase[phase].sigma;
                }
            }
        }
        output[row++] = observer.template get_output<sym>(binding.energized, sigma[0], sigma[1]);
    }
}
} // namespace power_grid_model::main_core
