#pragma once
#include "dfloat.h"

namespace dynfloat {
template<std::int64_t exp_bits_, std::int64_t man_bits_>
template<std::int64_t other_exp_bits_, std::int64_t other_man_bits_>
DYNFLOAT_FORCE_INLINE constexpr dfloat<exp_bits_, man_bits_>::operator dfloat<other_exp_bits_, other_man_bits_>() const noexcept {
    using res_t                   = dfloat<other_exp_bits_, other_man_bits_>;
    using other_uint_type         = typename res_t::uint_type;
    constexpr auto man_shift      = res_t::man_bits - man_bits;
    constexpr auto exp_correction = res_t::zero_exp - zero_exp;
    // TODO: fixme
    if (exp_correction >= 0) {
        return is_zero() ? res_t{}
                         : res_t::from_components(
                               unsigned_sign(), signed_exp() + exp_correction,
                               utils::safe_lshift(static_cast<std::common_type_t<other_uint_type, uint_type>>(unsigned_man()), man_shift));
    } else {
        return is_zero() ? res_t{}
                         : res_t::from_components(
                               unsigned_sign(), utils::non_overflowing_sub(signed_exp(), -exp_correction),
                               utils::safe_lshift(static_cast<std::common_type_t<other_uint_type, uint_type>>(unsigned_man()), man_shift));
    }
}

template<typename other_dfloat_, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
dfloat_cast(dfloat<exp_bits_, man_bits_> x) noexcept -> other_dfloat_ {
    if (asm1::zero_behavior == zero_behavior::non_zero) {
        using this_t                  = dfloat<exp_bits_, man_bits_>;
        using this_uint_type          = typename dfloat<exp_bits_, man_bits_>::uint_type;
        using res_t                   = other_dfloat_;
        using other_uint_type         = typename res_t::uint_type;
        constexpr auto man_shift      = res_t::man_bits - this_t::man_bits;
        constexpr auto exp_correction = res_t::zero_exp - this_t::zero_exp;
        if (exp_correction >= 0) {
            return res_t::from_components(
                x.unsigned_sign(), x.signed_exp() + exp_correction,
                utils::safe_lshift(static_cast<std::common_type_t<other_uint_type, this_uint_type>>(x.unsigned_man()), man_shift));
        } else {
            return res_t::from_components(
                x.unsigned_sign(), utils::non_overflowing_sub(x.signed_exp(), -exp_correction),
                utils::safe_lshift(static_cast<std::common_type_t<other_uint_type, this_uint_type>>(x.unsigned_man()), man_shift));
        }
    }
    return static_cast<other_dfloat_>(x);
}

} // namespace dynfloat
