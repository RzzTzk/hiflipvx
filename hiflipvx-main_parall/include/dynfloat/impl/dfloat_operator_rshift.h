#pragma once
#include "dfloat.h"

namespace dynfloat {
template<std::int64_t exp_bits_,
         std::int64_t man_bits_,
         typename integer_    //,
      //   std::enable_if_t<std::is_integral<integer_>::value, bool>
      >    
    //template<std::int64_t exp_bits_, std::int64_t man_bits_, typename integer_, std::enable_if_t<std::is_integral<integer_>::value, bool>>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator>>(dfloat<exp_bits_, man_bits_> lhs, integer_ rhs) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t     = dfloat<exp_bits_, man_bits_>;
    using int_type  = typename res_t::int_type;
    using uint_type = typename res_t::uint_type;

    // Calculate mask. The mask removes all bits in mantissa and exponent if shift is too large
    const auto signed_exp   = static_cast<int_type>(lhs.unsigned_exp());
    const auto res_non_zero = rhs < signed_exp;
    const auto mask         = utils::make_mask<utils::max(res_t::man_bits, res_t::exp_bits)>(res_non_zero);

    // Apply mask
    const auto res_exp = (lhs.unsigned_exp() - static_cast<uint_type>(rhs)) & mask;
    const auto res_man = lhs.unsigned_man() & mask;

    return res_t::from_components(lhs.unsigned_sign(), res_exp, res_man);
}

template<std::int64_t exp_bits_,
         std::int64_t man_bits_,
         typename integer_/*,
         std::enable_if_t<std::is_integral<integer_>::value, bool> = true*/>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator>>(dfloat<exp_bits_, man_bits_> lhs, integer_ rhs) noexcept -> dfloat<exp_bits_, man_bits_>;

} // namespace dynfloat
