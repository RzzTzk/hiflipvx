#pragma once
#include "dfloat.h"

namespace dynfloat {
template<std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
template<typename sign_t_, typename exp_t_, typename man_t_>
DYNFLOAT_FORCE_INLINE constexpr auto
dfloat<exponent_bits_, mantissa_bits_>::from_components(sign_t_ s, exp_t_ e, man_t_ m) noexcept -> dfloat<exponent_bits_, mantissa_bits_> {
#if defined(__GNUC__) // False positive on GCC
#pragma GCC diagnostic ignored "-Wconversion"
#endif
    return {
        static_cast<uint_type>(static_cast<uint_type>(s) & sign_mask),
        static_cast<uint_type>(static_cast<uint_type>(e) & exp_mask),
        static_cast<uint_type>(static_cast<uint_type>(m) & man_mask),
    };
}

template<std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
template<typename sign_t_, typename exp_t_, typename man_t_>
DYNFLOAT_FORCE_INLINE constexpr auto
dfloat<exponent_bits_, mantissa_bits_>::from_components_saturate(sign_t_ s, exp_t_ e, man_t_ m) noexcept
    -> dfloat<exponent_bits_, mantissa_bits_> {
    if (e > max_exp)
        return saturated_copy(s);
    return from_components(s, e, m);
}
} // namespace dynfloat
