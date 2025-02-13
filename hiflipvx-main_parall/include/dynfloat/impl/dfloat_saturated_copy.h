#pragma once
#include "dfloat.h"

namespace dynfloat {
template<std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
dfloat<exponent_bits_, mantissa_bits_>::saturated_copy(bool sign) noexcept -> dfloat<exponent_bits_, mantissa_bits_> {
    return from_int_repr(sign, static_cast<uint_repr_type>(-1));
}
} // namespace dynfloat
