
#pragma once
#include "dfloat.h"

namespace dynfloat {

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE dfloat<exp_bits_, man_bits_>::operator float() const noexcept {
    const auto int_repr = static_cast<std_f32>(*this).int_repr();
    const auto res      = *reinterpret_cast<const float*>(&int_repr); // NOLINT
    return unsigned_sign() ? -res : res;
}

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE dfloat<exp_bits_, man_bits_>::operator double() const noexcept {
    const auto int_repr = static_cast<std_f64>(*this).int_repr();
    const auto res      = *reinterpret_cast<const double*>(&int_repr); // NOLINT
    return unsigned_sign() ? -res : res;
}

} // namespace dynfloat
