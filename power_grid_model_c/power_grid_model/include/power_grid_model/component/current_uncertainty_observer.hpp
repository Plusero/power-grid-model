// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "observer.hpp"

namespace power_grid_model {

class CurrentUncertaintyObserver : public Observer {
  public:
    static constexpr char const* name = "current_uncertainty_observer";
    using InputType = CurrentUncertaintyObserverInput;
    template <symmetry_tag sym>
    using OutputType =
        std::conditional_t<is_symmetric_v<sym>, CurrentUncertaintyObserverOutput, AsymCurrentUncertaintyObserverOutput>;

    explicit CurrentUncertaintyObserver(InputType const& input)
        : Observer{input}, observed_terminal_type_{input.observed_terminal_type} {
        validate_terminal(observed_terminal_type_, false);
    }

    ObservedTerminalType observed_terminal_type() const { return observed_terminal_type_; }

    // The adapter supplies already converted physical sigmas; numerical UQ stays outside components.
    template <symmetry_tag sym = symmetric_t>
    OutputType<sym> get_output(bool is_connected_to_source, RealValue<sym> const& i_sigma,
                               RealValue<sym> const& i_angle_sigma) const {
        if (!energized(is_connected_to_source)) {
            return get_null_output<sym>();
        }
        return {.id = id(), .energized = 1, .i_sigma = i_sigma, .i_angle_sigma = i_angle_sigma};
    }

    template <symmetry_tag sym> OutputType<sym> get_null_output() const { return {.id = id(), .energized = 0}; }

  private:
    ObservedTerminalType observed_terminal_type_;
};

} // namespace power_grid_model
