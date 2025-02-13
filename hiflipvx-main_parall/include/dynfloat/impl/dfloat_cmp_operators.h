#pragma once
#include "dfloat.h"

namespace dynfloat {
template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator<(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> bool {
    const bool lhs_eq_rhs = lhs.int_repr() == rhs.int_repr();
    const bool lhs_lt_rhs = lhs.int_repr() < rhs.int_repr();
    const bool res = lhs.unsigned_sign() == rhs.unsigned_sign() ? lhs_lt_rhs ^ rhs.bool_sign() : lhs.unsigned_sign() > rhs.unsigned_sign();
    return res && !lhs_eq_rhs;
}

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator<=(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> bool {
    return lhs == rhs || lhs < rhs;
}

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator==(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> bool {
    const auto both_zeros = lhs.is_zero() && rhs.is_zero();
    return (lhs.int_repr() == rhs.int_repr() && (lhs.unsigned_sign() == rhs.unsigned_sign())) || both_zeros;
}

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator>=(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> bool {
    return !(lhs < rhs);
}

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator>(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> bool {
    return !(lhs <= rhs);
}
} // namespace dynfloat
