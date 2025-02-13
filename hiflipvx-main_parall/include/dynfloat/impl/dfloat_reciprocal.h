#pragma once
//#include "dfloat.h"
#include "dfloat_ctor.h"

namespace dynfloat {
namespace details_reciprocal {
template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
trail_subtraction(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t     = dfloat<exp_bits_, man_bits_>;
    using uint_type = typename res_t::uint_type;
    using int_type  = typename res_t::int_type;

    if (x.is_zero())
        return x.saturated_copy(x.unsigned_sign());
    constexpr auto one = make_dfloat<res_t>(1u);
    const auto res_exp = res_t::zero_exp + one.signed_exp() - x.signed_exp();
    if (x.unsigned_man() == 0)
        return res_t::from_components(x.unsigned_sign(), res_exp, 0);

    auto nom                     = uint_type{1} << res_t::man_bits;
    auto inv_man                 = uint_type{};
    const auto denom             = x.dman();
    constexpr auto trail_sub_msk = utils::make_mask<res_t::dman_bits + 1>(true);
    for (std::ptrdiff_t i = 0; i < res_t::man_bits + 2; ++i) {
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
#pragma HLS UNROLL
#endif
        const auto trail_sub        = nom - denom;
        const auto neg              = static_cast<int_type>(trail_sub) < 0;
        const auto next_neg_inv_man = inv_man << 1;
        const auto next_neg_nom     = nom << 1;
        const auto next_pos_inv_man = inv_man << 1 | 1;
        const auto next_pos_nom     = trail_sub << 1;
        inv_man                     = (neg ? next_neg_inv_man : next_pos_inv_man) & trail_sub_msk;
        nom                         = (neg ? next_neg_nom : next_pos_nom) & trail_sub_msk;
    }

    return res_t::from_components(x.unsigned_sign(), res_exp - 1, inv_man);
}

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
reciprocal(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t       = dfloat<exp_bits_, man_bits_>;
    using res_7man_t  = dfloat<exp_bits_, std::min(static_cast<int64_t>(man_bits_), static_cast<int64_t>(7LL))>;
    using res_11man_t = dfloat<exp_bits_, std::min(static_cast<int64_t>(man_bits_), static_cast<int64_t>(11LL))>;
    using res_17man_t = dfloat<exp_bits_, std::min(static_cast<int64_t>(man_bits_), static_cast<int64_t>(17LL))>;

    switch (static_cast<strategy>(execution_::str)) {
        case strategy::fast: {
            using int_repr_type = typename res_t::int_repr_type;

            constexpr auto factor       = static_cast<int_repr_type>(2.0 * utils::int_repr_approx_factor(res_t::exp_bits, res_t::man_bits));
            const auto integer_repr     = x.int_repr();
            const auto res_integer_repr = factor - integer_repr;
            return res_t::from_int_repr(x.unsigned_sign(), static_cast<int_repr_type>(res_integer_repr));
        }
        case strategy::refined_fast: {
            return static_cast<res_t>(details_reciprocal::trail_subtraction<execution_>(static_cast<res_7man_t>(x)));
        }
        case strategy::refined: {
            return static_cast<res_t>(details_reciprocal::trail_subtraction<execution_>(static_cast<res_11man_t>(x)));
        }
        case strategy::refined_exact: {
            return static_cast<res_t>(details_reciprocal::trail_subtraction<execution_>(static_cast<res_17man_t>(x)));
        }
        case strategy::exact: {
            return details_reciprocal::trail_subtraction<execution_>(x);
        }
    }
    return {};
}
} // namespace details_reciprocal

template<typename execution_, typename asm1, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
reciprocal(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    return details_reciprocal::reciprocal<execution_>(x);
}
} // namespace dynfloat
