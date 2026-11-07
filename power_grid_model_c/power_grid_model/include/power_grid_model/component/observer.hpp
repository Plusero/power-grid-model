// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "base.hpp"
#include "component.hpp"

#include "../common/exception.hpp"

#include <cassert>

namespace power_grid_model {

// Observers request results without adding measurements or electrical connections.
class Observer : public Base {
  public:
    using InputType = ObserverInput;
    using UpdateType = ObserverUpdate;
    static constexpr char const* name = "observer";

    ID observed_object() const { return observed_object_; }
    bool status() const { return status_; }
    bool energized(bool is_connected_to_source) const final { return status_ && is_connected_to_source; }
    ComponentType math_model_type() const final { return ComponentType::observer; }

    UpdateChange update(ObserverUpdate const& update_data) {
        assert(update_data.id == id() || is_nan(update_data.id));
        if (!is_nan(update_data.status)) {
            validate_status(update_data.status);
            status_ = update_data.status != 0;
        }
        return {.topo = false, .param = false};
    }

    ObserverUpdate inverse(ObserverUpdate update_data) const {
        assert(update_data.id == id() || is_nan(update_data.id));
        if (!is_nan(update_data.status)) {
            update_data.status = status_to_int(status_);
        }
        return update_data;
    }

  protected:
    explicit Observer(ObserverInput const& input) : Base{input}, observed_object_{input.observed_object} {
        if (is_nan(input.id) || is_nan(input.observed_object)) {
            throw InvalidArguments{"Observer requires id and observed_object"};
        }
        validate_status(input.status);
        status_ = input.status != 0;
    }

    static void validate_terminal(ObservedTerminalType terminal, bool allow_node_injection) {
        switch (terminal) {
        case ObservedTerminalType::branch_from:
        case ObservedTerminalType::branch_to:
        case ObservedTerminalType::branch3_1:
        case ObservedTerminalType::branch3_2:
        case ObservedTerminalType::branch3_3:
            return;
        case ObservedTerminalType::node_injection:
            if (allow_node_injection) {
                return;
            }
            break;
        default:
            break;
        }
        throw InvalidArguments{"Invalid observed terminal type"};
    }

  private:
    static void validate_status(IntS status) {
        if (status != 0 && status != 1) {
            throw InvalidArguments{"Observer status must be 0 or 1"};
        }
    }

    ID observed_object_;
    bool status_;
};

} // namespace power_grid_model
