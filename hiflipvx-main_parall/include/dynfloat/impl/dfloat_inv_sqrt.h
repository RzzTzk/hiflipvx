#pragma once
#include "dfloat.h"

namespace dynfloat {
namespace details_inv_sqrt {
constexpr auto
iter_count(const strategy m) {
    switch (m) {
        case strategy::fast:
            return 0;
        case strategy::refined_fast:
        case strategy::refined:
            return 1;
        case strategy::refined_exact:
        case strategy::exact:
            return 2;
    }
    return 2;
}

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
inv_sqrt(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t         = dfloat<exp_bits_, man_bits_>;
    using int_repr_type = typename dfloat<exp_bits_, man_bits_>::int_repr_type;

    constexpr auto ofs      = static_cast<int_repr_type>(utils::int_repr_sqrt_approx_factor(res_t::exp_bits, res_t::man_bits) * 1.5);
    const auto integer_repr = x.int_repr() >> 1;
    const auto imm_res      = ofs - integer_repr;
    auto res                = res_t::from_int_repr(0, static_cast<int_repr_type>(imm_res));

    // Newton-Raphson
    constexpr auto iteration_count = iter_count(execution_::str);

    for (std::ptrdiff_t it = 0; it < iteration_count; ++it) {
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
#pragma HLS UNROLL
#endif
        using exact_execution         = make_exact_execution<execution_>;
        constexpr auto one_point_five = static_cast<res_t>(1.5f);
        const auto four_mul           = tree::mul<exact_execution, positive_asm>(res, res, res, x);
        const auto shifted_four_mul   = res_t::from_components(0, four_mul.unsigned_exp() - 1, four_mul.unsigned_man());
        const auto res_mul_1p5        = mul<exact_execution, positive_asm, positive_asm>(res, one_point_five);
        res                           = sub<exact_execution, positive_asm, positive_asm>(res_mul_1p5, shifted_four_mul);
    }
    return res;
}
} // namespace details_inv_sqrt

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
inv_sqrt(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t = dfloat<exp_bits_, man_bits_>;
    using imm_t = expand_to_dsp_opt<res_t>;
    return static_cast<res_t>(details_inv_sqrt::inv_sqrt<execution_>(static_cast<imm_t>(x)));
}

} // namespace dynfloat
