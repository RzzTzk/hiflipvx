#pragma once
#include "make_dfloat.h"

namespace std {
template<std::int64_t exp_bits_, std::int64_t man_bits_>
class numeric_limits<dynfloat::dfloat<exp_bits_, man_bits_>> // NOLINT
{
public:

    using float_type                        = dynfloat::dfloat<exp_bits_, man_bits_>;
    static constexpr auto is_specialized    = true;
    static constexpr auto is_signed         = true;
    static constexpr auto is_integer        = false;
    static constexpr auto is_exact          = false;
    static constexpr auto has_infinity      = false;
    static constexpr auto has_quiet_NaN     = false;
    static constexpr auto has_signaling_NaN = false;
    static constexpr auto has_denorm        = std::denorm_absent;
    static constexpr auto has_denorm_loss   = false;
    static constexpr auto round_style       = std::round_toward_zero;
    static constexpr auto is_iec559         = false;
    static constexpr auto is_bounded        = true;
    static constexpr auto is_modulo         = false;
    static constexpr auto digits            = static_cast<int>(float_type::man_bits);
    static constexpr auto radix             = 2;
    static constexpr auto min_exponent      = -static_cast<int>(float_type::zero_exp);
    static constexpr auto max_exponent      = static_cast<int>(float_type::exp_mask);
    static constexpr auto traps             = std::numeric_limits<typename float_type ::uint_type>::traps;
    static constexpr auto tinyness_before   = false;

    static constexpr auto min() noexcept -> float_type {
        return float_type::from_int_repr(0, 1);
    }

    static constexpr auto lowest() noexcept -> float_type {
        return float_type::from_int_repr(1, 1);
    }

    static constexpr auto max() noexcept -> float_type {
        return float_type::from_components(0, float_type::exp_mask, float_type::man_mask);
    }

    static constexpr auto epsilon() noexcept -> float_type {
        return min();
    }

    static constexpr auto round_error() noexcept -> float_type {
        return dynfloat::make_dfloat<float_type>(1u, 2u);
    }
};
} // namespace std