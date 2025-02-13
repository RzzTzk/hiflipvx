#pragma once
#include "dfloat_operator_rshift.h"

namespace dynfloat {

template<typename execution_,
         typename asm1,
         std::int64_t exp_bits_,
         std::int64_t man_bits_,
         typename integer_/*,
         std::enable_if_t<std::is_integral<integer_>::value, bool>*/>
DYNFLOAT_FORCE_INLINE static constexpr auto
lshift(dfloat<exp_bits_, man_bits_> lhs, integer_ rhs) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t         = dfloat<exp_bits_, man_bits_>;
    using int_type      = typename res_t::int_type;
    using exp_add_type  = std::make_signed_t<std::common_type_t<int_type, std::make_signed_t<integer_>>>;
    using exp_add_utype = std::make_unsigned_t<std::common_type_t<int_type, std::make_signed_t<integer_>>>;
    const auto is_zero  = lhs.is_zero(asm1::zero_behavior);
    const auto mask     = utils::make_mask<res_t::exp_bits>(!is_zero);
    const auto imm_exp  = static_cast<exp_add_utype>(utils::add_saturate<execution_{} == special_values::zero_and_saturation>(
        static_cast<exp_add_type>(lhs.unsigned_exp()), static_cast<exp_add_type>(rhs), 0, res_t::max_exp));

    const auto new_exp = mask & imm_exp;

    return dfloat<exp_bits_, man_bits_>::from_components_saturate(lhs.unsigned_sign(), new_exp, lhs.unsigned_man());
}

 template<typename execution_, typename asm1 = default_assumptions, typename shifted_type_, typename shift_type_>
 DYNFLOAT_FORCE_INLINE static constexpr auto
 safe_lshift(const shifted_type_ val, const shift_type_ shift_amount) noexcept -> auto {
     const auto signed_shift = static_cast<std::make_signed_t<shift_type_>>(shift_amount);
     return signed_shift < 0 ? val >> (-signed_shift) : dynfloat::lshift<execution_, asm1>(val, signed_shift);
 }

} // namespace dynfloat
