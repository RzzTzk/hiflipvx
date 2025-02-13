#pragma once
#include "dfloat.h"

namespace dynfloat {
template<std::int64_t exponent_bits_, std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
dfloat<exponent_bits_, mantissa_bits_>::dman() const noexcept -> typename dfloat<exponent_bits_, mantissa_bits_>::dman_type {
    return static_cast<dman_type>(static_cast<dman_type>(unsigned_man()) | dman_type{1} << man_bits);
}
} // namespace dynfloat
