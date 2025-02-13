#pragma once
#include "dfloat_div.h"

namespace dynfloat {
namespace tanh_details {
template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
preconditioned_tanh(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t           = dfloat<exp_bits_, man_bits_>;
    using exact_execution = make_exact_execution<execution_>;
    constexpr auto _0_01  = static_cast<res_t>(0.01);
    constexpr auto _0_15  = static_cast<res_t>(0.15);
    constexpr auto one    = static_cast<res_t>(1.);

    switch (static_cast<strategy>(execution_::str)) {
        case strategy::fast: {
            if (std::abs(x) < _0_15)
                return x;

            const auto exp_pls_x = exp<execution_>(x);
            const auto exp_mns_x = exp<execution_>(-x);
            const auto nom       = sub<exact_execution, positive_nz_asm, positive_nz_asm>(exp_pls_x, exp_mns_x);
            const auto dnm       = add<exact_execution, positive_nz_asm, positive_nz_asm>(exp_pls_x, exp_mns_x);
            const auto res       = dynfloat::div<execution_>(nom, dnm);
            return res;
        }
        case strategy::refined:
        case strategy::refined_exact:
        case strategy::refined_fast:
        case strategy::exact: {
            if (std::abs(x) < _0_01)
                return x;
            const auto exp2x = dynfloat::exp<execution_>(res_t::from_components(x.unsigned_sign(), x.unsigned_exp() + 1, x.unsigned_man()));
            const auto exp2x_pls_1             = dynfloat::add<exact_execution, positive_nz_asm, positive_nz_asm>(exp2x, one);
            const auto recip_exp2x_pls_1       = dynfloat::reciprocal<execution_>(exp2x_pls_1);
            const auto twice_recip_exp2x_pls_1 = dynfloat::lshift<exact_execution, positive_nz_asm>(recip_exp2x_pls_1, 1);
            const auto res = dynfloat::sub<exact_execution, positive_nz_asm, positive_nz_asm>(one, twice_recip_exp2x_pls_1);
            return res;
        }
    }
    return {};
}
} // namespace tanh_details

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
tanh(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t = dfloat<exp_bits_, man_bits_>;
    using imm_t = expand_to_dsp_opt<res_t>;

    const auto imm_x = static_cast<imm_t>(x);
    constexpr strategy capped_str =
        imm_t::man_bits < std_f32::man_bits ? utils::min(strategy::refined_exact, execution_::str) : execution_::str;
    using exec = set_saturation<set_execution_strategy<capped_str, execution_>>;
    return static_cast<res_t>(tanh_details::preconditioned_tanh<exec>(imm_x));
}
} // namespace dynfloat
