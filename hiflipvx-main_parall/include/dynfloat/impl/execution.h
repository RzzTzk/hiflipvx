#pragma once
#include "../deps.h"

namespace dynfloat {
enum class strategy : std::int8_t {
    fast          = 0, // relative error <~ 10^-2
    refined_fast  = 1, // relative error <~ 10^-3
    refined       = 2, // relative error <~ 10^-4
    refined_exact = 3, // relative error <~ 10^-5
    exact         = 4, // relative error <~ 10^-6
};

constexpr auto
enum_name(const strategy s) noexcept -> const char* {
    switch (s) {
        case strategy::fast:
            return "fast";
        case strategy::refined_fast:
            return "refined_fast";
        case strategy::refined:
            return "refined";
        case strategy::refined_exact:
            return "refined_exact";
        case strategy::exact:
            return "exact";
    }
    return "";
}

enum class rounding_behavior : std::int8_t {
    to_zero      = 0,
    nearest_even = 1,
};

constexpr auto
enum_name(const rounding_behavior s) noexcept -> const char* {
    switch (s) {
        case rounding_behavior::to_zero:
            return "to_zero";
        case rounding_behavior::nearest_even:
            return "nearest_even";
    }
    return "";
}

enum class special_values : std::int8_t {
    zero                = 0,
    zero_and_saturation = 1,
};

constexpr auto
enum_name(const special_values s) noexcept -> const char* {
    switch (s) {
        case special_values::zero:
            return "zero";
        case special_values::zero_and_saturation:
            return "zero_and_saturation";
    }
    return "";
}

enum class expand_behavior : std::int8_t {
    expand    = 0,
    no_expand = 1,
};

constexpr auto
enum_name(const expand_behavior s) noexcept -> const char* {
    switch (s) {
        case expand_behavior::expand:
            return "expand";
        case expand_behavior::no_expand:
            return "no_expand";
    }
    return "";
}

template<strategy str_,
         rounding_behavior rnd_,
         special_values spv_,
         expand_behavior exp_       = expand_behavior::expand,
         bool native_fallback_impl_ = false>
struct execution {
    static constexpr auto str = str_;
    static constexpr auto rnd = rnd_;
    static constexpr auto spv = spv_;
    static constexpr auto exp = exp_;
    static constexpr auto nfi = native_fallback_impl_;

    constexpr auto operator==(const strategy other_str) const {
        return str == other_str;
    }

    constexpr auto operator==(const rounding_behavior other_rnd) const {
        return rnd == other_rnd;
    }

    constexpr auto operator==(const special_values other_spv) const {
        return spv == other_spv;
    }

    constexpr auto operator==(const expand_behavior other_exp) const {
        return exp == other_exp;
    }
};

template<strategy new_str, typename execution_>
using set_execution_strategy = execution<new_str, execution_::rnd, execution_::spv>;

template<typename execution_>
using make_exact_execution = execution<strategy::exact, execution_::rnd, execution_::spv>;

template<typename execution_>
using set_saturation = execution<execution_::str, execution_::rnd, special_values::zero_and_saturation>;

template<typename execution_, expand_behavior value>
using set_expansion = execution<execution_::str, execution_::rnd, execution_::spv, value>;

enum class sign_behavior : std::int8_t {
    positive = 0,
    negative = 1,
    unknown  = 2,
};

enum class zero_behavior : std::int8_t {
    non_zero = 0,
    unknown  = 1,
};

template<sign_behavior sign_behavior_, zero_behavior zero_behavior_>
struct assumptions_tuple {
    static constexpr auto sign_behavior = sign_behavior_;
    static constexpr auto zero_behavior = zero_behavior_;
};

using default_assumptions = assumptions_tuple<sign_behavior::unknown, zero_behavior::unknown>;
using positive_asm        = assumptions_tuple<sign_behavior::positive, zero_behavior::unknown>;
using positive_nz_asm     = assumptions_tuple<sign_behavior::positive, zero_behavior::non_zero>;
using negative_nz_asm     = assumptions_tuple<sign_behavior::negative, zero_behavior::non_zero>;

namespace modify_asm {
template<typename asm1>
using negate = assumptions_tuple<asm1::sign_behavior == sign_behavior::positive   ? sign_behavior::negative
                                 : asm1::sign_behavior == sign_behavior::negative ? sign_behavior::positive
                                                                                  : sign_behavior::unknown,
                                 asm1::zero_behavior>;

template<typename tup_>
using reciprocal = tup_;

template<typename asm1, typename asm2>
using mul = assumptions_tuple<
    asm1::sign_behavior == sign_behavior::unknown || asm2::sign_behavior == sign_behavior::unknown ? sign_behavior::unknown
    : asm1::sign_behavior == asm2::sign_behavior                                                   ? sign_behavior::positive
                                                                                                   : sign_behavior::negative,
    asm1::zero_behavior == zero_behavior::unknown || asm2::zero_behavior == zero_behavior::unknown ? zero_behavior::unknown
                                                                                                   : zero_behavior::non_zero>;

} // namespace modify_asm

} // namespace dynfloat
