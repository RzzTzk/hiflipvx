#pragma once
#include "dfloat.h"
#include "dfloat_from_components.h"

namespace dynfloat {

/******************************************************************************************************************************************/

template<typename dfloat_, typename integer_, std::enable_if_t<is_dfloat_v<dfloat_>, bool> = true>
DYNFLOAT_FORCE_INLINE static constexpr auto
make_dfloat(integer_ i) noexcept -> dfloat_;

template<std::int64_t exp_bits_,
         std::int64_t man_bits_,
         typename integer_,
         std::enable_if_t<utils::is_unsigned_integer<integer_>, bool> = true>
DYNFLOAT_FORCE_INLINE static constexpr auto
make_dfloat(integer_ i) noexcept -> dfloat<exp_bits_, man_bits_>;

template<std::int64_t exp_bits_,
         std::int64_t man_bits_,
         typename integer_,
         std::enable_if_t<utils::is_signed_integer<integer_>, bool> = true>
DYNFLOAT_FORCE_INLINE static constexpr auto
make_dfloat(integer_ i) noexcept -> dfloat<exp_bits_, man_bits_>;

template<typename dfloat_, typename execution_, typename num_, typename denom_, std::enable_if_t<is_dfloat_v<dfloat_>, bool> = true>
DYNFLOAT_FORCE_INLINE static constexpr auto
make_dfloat(num_ num, denom_ denom) noexcept -> dfloat_;

template<std::int64_t exp_bits_, std::int64_t man_bits_, typename execution_, typename num_, typename denom_>
DYNFLOAT_FORCE_INLINE static constexpr auto
make_dfloat(num_ num, denom_ denom) noexcept -> dfloat<exp_bits_, man_bits_>;

/******************************************************************************************************************************************/

template<typename dfloat_, typename integer_, std::enable_if_t<is_dfloat_v<dfloat_>, bool>>
DYNFLOAT_FORCE_INLINE static constexpr auto
make_dfloat(integer_ i) noexcept -> dfloat_ {
    return dynfloat::make_dfloat<dfloat_::exp_bits, dfloat_::man_bits>(i);
}

template<std::int64_t exp_bits_, std::int64_t man_bits_, typename integer_, std::enable_if_t<utils::is_unsigned_integer<integer_>, bool>>
DYNFLOAT_FORCE_INLINE static constexpr auto
make_dfloat(const integer_ i) noexcept -> dfloat<exp_bits_, man_bits_> {
    using float_type        = dfloat<exp_bits_, man_bits_>;
    using uint_type         = typename float_type::uint_type;
    const auto bit_width    = utils::bit_width(i);
    const auto res_exp      = bit_width ? float_type::zero_exp + bit_width - 1 : 0;
    const auto shift_amount = static_cast<std::int8_t>(float_type::dman_bits - bit_width);
    const auto res_man      = static_cast<uint_type>(utils::safe_lshift(static_cast<std::uint64_t>(i), shift_amount));
    return float_type::from_components(0, res_exp, res_man);
}

template<std::int64_t exp_bits_, std::int64_t man_bits_, typename integer_, std::enable_if_t<utils::is_signed_integer<integer_>, bool>>
DYNFLOAT_FORCE_INLINE static constexpr auto
make_dfloat(const integer_ i) noexcept -> dfloat<exp_bits_, man_bits_> {
    using float_type        = dfloat<exp_bits_, man_bits_>;
    using uint_type         = typename float_type::uint_type;
    using uinteger          = std::make_unsigned_t<integer_>;
    const auto res_exp_mask = static_cast<uint_type>(utils::make_mask<exp_bits_>(i != 0));
    const auto sign         = i < 0;
    const auto abs_i        = static_cast<integer_>(sign ? -i : i);
    const auto bit_width    = utils::bit_width(static_cast<uinteger>(abs_i));
    const auto res_exp      = (bit_width ? float_type::zero_exp + bit_width - 1 : 0) & res_exp_mask;
    const auto shift_amount = static_cast<std::int8_t>(float_type::dman_bits - bit_width);
    const auto res_man      = static_cast<uint_type>(utils::safe_lshift(static_cast<std::uint64_t>(abs_i), shift_amount));
    return float_type::from_components(sign, res_exp, res_man);
}

template<typename dfloat_, typename execution_, typename num_, typename denom_, std::enable_if_t<is_dfloat_v<dfloat_>, bool>>
DYNFLOAT_FORCE_INLINE static constexpr auto
make_dfloat(const num_ num, const denom_ denom) noexcept -> dfloat_ {
    return dynfloat::make_dfloat<dfloat_::exp_bits, dfloat_::man_bits, execution_>(num, denom);
}

template<std::int64_t exp_bits_, std::int64_t man_bits_, typename execution_, typename num_, typename denom_>
DYNFLOAT_FORCE_INLINE static constexpr auto
make_dfloat(const num_ num, const denom_ denom) noexcept -> dfloat<exp_bits_, man_bits_> {
    return dynfloat::div<execution_>(dynfloat::make_dfloat<exp_bits_, man_bits_>(num), dynfloat::make_dfloat<exp_bits_, man_bits_>(denom));
}

template<std::int64_t scale, typename dfloat_>
DYNFLOAT_FORCE_INLINE static constexpr auto
from_ufixed(const std::uint64_t fixed) noexcept -> dfloat_ {
    using float_type        = dfloat_;
    using uint_type         = typename float_type::uint_type;
    const auto bit_width    = utils::bit_width(fixed);
    const auto res_exp      = bit_width ? float_type::zero_exp + bit_width - 1 - scale : 0;
    const auto shift_amount = static_cast<std::int8_t>(float_type::dman_bits - bit_width);
    const auto res_man      = static_cast<uint_type>(utils::safe_lshift(fixed, shift_amount));
    return float_type::from_components(0, res_exp, res_man);
}

template<std::int64_t scale, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
from_ufixed(const std::uint64_t fixed) noexcept -> dfloat<exp_bits_, man_bits_> {
    using float_type = dfloat<exp_bits_, man_bits_>;
    return from_ufixed<scale, float_type>(fixed);
}

template<std::int64_t scale, typename dfloat_>
DYNFLOAT_FORCE_INLINE static constexpr auto
from_fixed(const std::int64_t fixed) noexcept -> dfloat_ {
    return fixed < 0 ? -from_ufixed<scale, dfloat_>(-fixed) : from_ufixed<scale, dfloat_>(fixed);
}

template<std::int64_t scale, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
from_fixed(const std::int64_t fixed) noexcept -> dfloat<exp_bits_, man_bits_> {
    using float_type = dfloat<exp_bits_, man_bits_>;
    return from_fixed<scale, float_type>(fixed);
}

} // namespace dynfloat
