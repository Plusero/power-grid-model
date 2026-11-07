// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "container_queries.hpp"
#include "state_queries.hpp"

#include "../auxiliary/dataset.hpp"
#include "../component/current_uncertainty_observer.hpp"
#include "../component/power_uncertainty_observer.hpp"
#include "../component/voltage_uncertainty_observer.hpp"
#include "../main_model_fwd.hpp"
#include "../math_solver/state_estimation_uq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace power_grid_model::main_core {

namespace observer_uncertainty {
struct OutputSelection {
    bool requested = false;
    bool first = false;
    bool second = false;
};

// Inspect registered storage, not inferred meaningful attributes: metadata-only
// columns are real requests, whereas a known count without storage is not.
inline OutputSelection output_selection(MutableDataset const& data, std::string_view component, std::string_view first,
                                        std::string_view second) {
    Idx const index = data.find_component(component, false);
    if (index < 0 || data.get_component_info(index).total_elements == 0) {
        return {};
    }
    auto const& buffer = data.get_buffer(index);
    if (data.is_row_based(index)) {
        return {true, true, true};
    }
    OutputSelection result;
    for (auto const& attribute : buffer.attributes) {
        if (attribute.data == nullptr) {
            continue;
        }
        result.requested = true;
        result.first = result.first || attribute.meta_attribute->name == first;
        result.second = result.second || attribute.meta_attribute->name == second;
    }
    return result;
}

inline void validate_output_mode(MutableDataset const& data, MainModelOptions const& options) {
    bool const requested =
        output_selection(data, VoltageUncertaintyObserver::name, "u_sigma", "u_angle_sigma").requested ||
        output_selection(data, PowerUncertaintyObserver::name, "p_sigma", "q_sigma").requested ||
        output_selection(data, CurrentUncertaintyObserver::name, "i_sigma", "i_angle_sigma").requested;
    if (requested && (options.calculation_type != CalculationType::state_estimation ||
                      (options.calculation_method != CalculationMethod::default_method &&
                       options.calculation_method != CalculationMethod::iterative_linear &&
                       options.calculation_method != CalculationMethod::newton_raphson))) {
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

enum class Quantity { voltage, branch, injection };
struct Binding {
    Quantity quantity = Quantity::voltage;
    Idx island = -1;
    Idx math_index = -1;
    BranchSide side = BranchSide::from;
    Idx result_index = -1;
    bool energized = false;
    bool first = false;
    bool second = false;
    double magnitude_scale = 1.0;
};

template <symmetry_tag sym> struct Scenario {
    std::vector<math_solver::StateEstimationUQWork<sym>> work;
    std::vector<Binding> voltage, power, current;
    OutputSelection voltage_selection, power_selection, current_selection;

    explicit Scenario(MutableDataset const& data)
        : voltage_selection{output_selection(data, VoltageUncertaintyObserver::name, "u_sigma", "u_angle_sigma")},
          power_selection{output_selection(data, PowerUncertaintyObserver::name, "p_sigma", "q_sigma")},
          current_selection{output_selection(data, CurrentUncertaintyObserver::name, "i_sigma", "i_angle_sigma")} {}
};

// One double product; overflow, or a positive product that underflows to zero,
// gives a missing value. A subnormal product is kept.
inline double physical_sigma(math_solver::UQPhaseValue const& value, double base) {
    if (!std::isfinite(base) || base < 0.0 || !std::isfinite(value.sigma) || value.sigma < 0.0) {
        return nan;
    }
    double const result = base * value.sigma;
    if (!std::isfinite(result) || (base > 0.0 && value.sigma > 0.0 && result == 0.0)) {
        return nan;
    }
    return result;
}

template <symmetry_tag sym = symmetric_t, class Components, class ObserverType>
Binding map_target(MainModelState<Components> const& state, ObserverType const& observer, OutputSelection selection) {
    Binding result;
    result.first = selection.first;
    result.second = selection.second;
    auto map_node = [&] {
        Idx const user_node = get_component_sequence_idx<Node>(state.components, observer.observed_object());
        auto const math_id = get_user_node_math_id(state, user_node);
        result.island = math_id.group;
        result.math_index = math_id.pos;
        result.energized = observer.status() && math_id.group >= 0;
        if constexpr (std::same_as<ObserverType, VoltageUncertaintyObserver>) {
            result.magnitude_scale =
                u_scale<sym> * get_component<Node>(state.components, observer.observed_object()).u_rated();
        }
    };
    if constexpr (std::same_as<ObserverType, VoltageUncertaintyObserver>) {
        map_node();
    } else {
        switch (observer.observed_terminal_type()) {
        case ObservedTerminalType::node_injection:
            result.quantity = Quantity::injection;
            map_node();
            break;
        case ObservedTerminalType::branch_from:
        case ObservedTerminalType::branch_to: {
            result.quantity = Quantity::branch;
            result.side = observer.observed_terminal_type() == ObservedTerminalType::branch_from ? BranchSide::from
                                                                                                 : BranchSide::to;
            auto const& branch = get_component<Branch>(state.components, observer.observed_object());
            Idx const sequence = get_component_sequence_idx<Branch>(state.components, observer.observed_object());
            auto const math_id = get_math_id<Branch>(state, sequence);
            result.island = math_id.group;
            result.math_index = math_id.pos;
            result.energized = observer.status() && math_id.group >= 0 && branch.status(result.side);
            result.magnitude_scale = result.side == BranchSide::from ? branch.base_i_from() : branch.base_i_to();
            break;
        }
        case ObservedTerminalType::branch3_1:
        case ObservedTerminalType::branch3_2:
        case ObservedTerminalType::branch3_3: {
            result.quantity = Quantity::branch;
            Idx const terminal =
                static_cast<Idx>(observer.observed_terminal_type()) - static_cast<Idx>(ObservedTerminalType::branch3_1);
            auto const& branch = get_component<Branch3>(state.components, observer.observed_object());
            Idx const sequence = get_component_sequence_idx<Branch3>(state.components, observer.observed_object());
            auto const math_id = get_math_id<Branch3>(state, sequence);
            result.island = math_id.group;
            result.math_index = math_id.pos[terminal];
            result.side = BranchSide::from;
            result.energized = observer.status() && math_id.group >= 0 && math_id.pos[terminal] >= 0 &&
                               branch.status(static_cast<Branch3Side>(terminal));
            std::array<double, 3> const bases{branch.base_i_1(), branch.base_i_2(), branch.base_i_3()};
            result.magnitude_scale = bases[terminal];
            break;
        }
        }
        if constexpr (std::same_as<ObserverType, PowerUncertaintyObserver>) {
            result.magnitude_scale = base_power<sym>;
        }
    }
    return result;
}

inline Idx branch_rank(Idx branch, BranchSide side) { return 2 * branch + static_cast<Idx>(side); }

// Sort by a dense nonnegative integer rank and merge duplicates; the result maps
// each rank to its item's final position (-1 if absent). Ranks are math indices
// of one island, so bucketing is linear and replaces a comparison sort that cost
// several percent of a large UQ calculation. Merges are commutative.
template <class Request, class Rank, class Merge>
std::vector<Idx> normalize(std::vector<Request>& items, Rank rank, Merge merge) {
    Idx size = 0;
    for (auto const& item : items) {
        size = std::max(size, rank(item) + 1);
    }
    std::vector<Idx> slot(size, -1);
    std::vector<Request> unique;
    unique.reserve(items.size());
    for (auto const& item : items) {
        auto& index = slot[rank(item)];
        if (index < 0) {
            index = static_cast<Idx>(unique.size());
            unique.push_back(item);
        } else {
            merge(unique[index], item);
        }
    }
    items.clear();
    for (Idx& index : slot) {
        if (index >= 0) {
            items.push_back(unique[index]);
            index = static_cast<Idx>(items.size()) - 1;
        }
    }
    return slot;
}
// Positions by rank, indexed by Quantity.
inline std::array<std::vector<Idx>, 3> normalize(math_solver::StateEstimationUQRequest& request) {
    std::array<std::vector<Idx>, 3> positions;
    positions[static_cast<std::size_t>(Quantity::voltage)] = normalize(
        request.voltage, [](auto const& item) { return item.bus; },
        [](auto& target, auto const& source) {
            target.magnitude = target.magnitude || source.magnitude;
            target.angle = target.angle || source.angle;
        });
    positions[static_cast<std::size_t>(Quantity::branch)] = normalize(
        request.branch, [](auto const& item) { return branch_rank(item.branch, item.side); },
        [](auto& target, auto const& source) {
            target.active_power = target.active_power || source.active_power;
            target.reactive_power = target.reactive_power || source.reactive_power;
            target.current_magnitude = target.current_magnitude || source.current_magnitude;
            target.current_angle = target.current_angle || source.current_angle;
        });
    positions[static_cast<std::size_t>(Quantity::injection)] = normalize(
        request.injection, [](auto const& item) { return item.bus; },
        [](auto& target, auto const& source) {
            target.active_power = target.active_power || source.active_power;
            target.reactive_power = target.reactive_power || source.reactive_power;
        });
    return positions;
}
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

// Populate the fresh per-calculation scenario after topology preparation.
template <symmetry_tag sym, class Components>
void prepare_observer_uncertainty(MainModelState<Components> const& state,
                                  observer_uncertainty::Scenario<sym>& scenario, Idx n_islands) {
    using namespace observer_uncertainty;
    // Count physical nodes per reduced node, rather than per mathematical index.
    // Rebuild every scenario because link updates can change fusion.
    std::vector<Idx> physical_node_counts;
    auto prepare = [&]<class ObserverType>(OutputSelection selection, std::vector<Binding>& bindings) {
        if (!selection.requested) {
            return;
        }
        if constexpr (common::component_container_c<Components, ObserverType>) {
            bindings.reserve(get_component_size<ObserverType>(state.components));
            for (auto const& observer : get_component_citer<ObserverType>(state.components)) {
                auto binding = map_target<sym>(state, observer, selection);
                if (binding.energized && (binding.first || binding.second)) {
                    if (binding.quantity == Quantity::injection) {
                        if (physical_node_counts.empty()) {
                            physical_node_counts.resize(state.reduced_topology->topo_node_coup.topo_nodes.size(), 0);
                            for (auto const& coupling :
                                 state.reduced_topology->topo_node_coup.coupling.user_nodes_to_topo_nodes) {
                                if (coupling.group >= 0) {
                                    ++physical_node_counts[coupling.group];
                                }
                            }
                        }
                        Idx const user_node =
                            get_component_sequence_idx<Node>(state.components, observer.observed_object());
                        Idx const topo_node =
                            state.reduced_topology->topo_node_coup.coupling.user_nodes_to_topo_nodes[user_node].group;
                        if (physical_node_counts[topo_node] != 1) {
                            throw InvalidArguments{
                                "Node injection observers do not support nodes fused by closed ideal links"};
                        }
                    }
                    if (scenario.work.empty()) {
                        scenario.work.resize(n_islands);
                    }
                    auto& request = scenario.work[binding.island].request;
                    if constexpr (std::same_as<ObserverType, VoltageUncertaintyObserver>) {
                        request.voltage.push_back({binding.math_index, binding.first, binding.second});
                    } else if constexpr (std::same_as<ObserverType, PowerUncertaintyObserver>) {
                        if (binding.quantity == Quantity::injection) {
                            request.injection.push_back({binding.math_index, binding.first, binding.second});
                        } else {
                            request.branch.push_back(
                                {binding.math_index, binding.side, binding.first, binding.second, false, false});
                        }
                    } else {
                        request.branch.push_back(
                            {binding.math_index, binding.side, false, false, binding.first, binding.second});
                    }
                }
                bindings.push_back(binding);
            }
        }
    };
    prepare.template operator()<VoltageUncertaintyObserver>(scenario.voltage_selection, scenario.voltage);
    prepare.template operator()<PowerUncertaintyObserver>(scenario.power_selection, scenario.power);
    prepare.template operator()<CurrentUncertaintyObserver>(scenario.current_selection, scenario.current);
    std::vector<std::array<std::vector<Idx>, 3>> positions;
    positions.reserve(scenario.work.size());
    for (auto& work : scenario.work) {
        positions.push_back(normalize(work.request));
    }
    // Every bound binding added its own request above.
    for (auto* bindings : {&scenario.voltage, &scenario.power, &scenario.current}) {
        for (auto& binding : *bindings) {
            if (binding.energized && (binding.first || binding.second)) {
                Idx const rank = binding.quantity == Quantity::branch ? branch_rank(binding.math_index, binding.side)
                                                                      : binding.math_index;
                binding.result_index = positions[binding.island][static_cast<std::size_t>(binding.quantity)][rank];
            }
        }
    }
}

template <class ObserverType, symmetry_tag sym, class Components, non_owning_view_c Outputs>
void output_observer_uncertainty(MainModelState<Components> const& state,
                                 observer_uncertainty::Scenario<sym> const& scenario, Outputs output) {
    using namespace observer_uncertainty;
    bool const requested = [&] {
        if constexpr (std::same_as<ObserverType, VoltageUncertaintyObserver>) {
            return scenario.voltage_selection.requested;
        } else if constexpr (std::same_as<ObserverType, PowerUncertaintyObserver>) {
            return scenario.power_selection.requested;
        } else {
            return scenario.current_selection.requested;
        }
    }();
    // Native callers may register a component count without any output storage.
    if (!requested) {
        return;
    }
    auto const& bindings = [&]() -> std::vector<Binding> const& {
        if constexpr (std::same_as<ObserverType, VoltageUncertaintyObserver>) {
            return scenario.voltage;
        } else if constexpr (std::same_as<ObserverType, PowerUncertaintyObserver>) {
            return scenario.power;
        } else {
            return scenario.current;
        }
    }();
    Idx row = 0;
    for (auto const& observer : get_component_citer<ObserverType>(state.components)) {
        auto const& binding = bindings[row];
        RealValue<sym> first{nan}, second{nan};
        constexpr int phases = is_symmetric_v<sym> ? 1 : 3;
        for (int phase = 0; phase != phases; ++phase) {
            auto phase_value = [phase](RealValue<sym>& value) -> double& {
                if constexpr (is_symmetric_v<sym>)
                    return value;
                else
                    return value(phase);
            };
            auto& first_phase = phase_value(first);
            auto& second_phase = phase_value(second);
            if (binding.result_index >= 0) {
                auto const& result = scenario.work[binding.island].result;
                if constexpr (std::same_as<ObserverType, VoltageUncertaintyObserver>) {
                    auto const& item = result.voltage[binding.result_index];
                    if (binding.first) {
                        first_phase = physical_sigma(item.magnitude.phase[phase], binding.magnitude_scale);
                    }
                    if (binding.second) {
                        second_phase = physical_sigma(item.angle.phase[phase], 1.0);
                    }
                } else if constexpr (std::same_as<ObserverType, PowerUncertaintyObserver>) {
                    if (binding.quantity == Quantity::injection) {
                        auto const& item = result.injection[binding.result_index];
                        if (binding.first) {
                            first_phase = physical_sigma(item.active_power.phase[phase], binding.magnitude_scale);
                        }
                        if (binding.second) {
                            second_phase = physical_sigma(item.reactive_power.phase[phase], binding.magnitude_scale);
                        }
                    } else {
                        auto const& item = result.branch[binding.result_index];
                        if (binding.first) {
                            first_phase = physical_sigma(item.active_power.phase[phase], binding.magnitude_scale);
                        }
                        if (binding.second) {
                            second_phase = physical_sigma(item.reactive_power.phase[phase], binding.magnitude_scale);
                        }
                    }
                } else {
                    auto const& item = result.branch[binding.result_index];
                    if (binding.first) {
                        first_phase = physical_sigma(item.current_magnitude.phase[phase], binding.magnitude_scale);
                    }
                    if (binding.second) {
                        second_phase = physical_sigma(item.current_angle.phase[phase], 1.0);
                    }
                }
            }
        }
        output[row++] = observer.template get_output<sym>(binding.energized, first, second);
    }
}
} // namespace power_grid_model::main_core
