#pragma once
#include "dfloat_div.h"

namespace dynfloat {
// NOLINTBEGIN
namespace details_exp {

template<std::int64_t scale, std::int64_t iterations, typename int_type_>
void
cordic_rotation_hyperbolic_fixed(int_type_& x_n,
                                 int_type_& y_n,
                                 int_type_& z_n,
                                 const int_type_ x0,
                                 const int_type_ y0,
                                 const int_type_ z0) {
    constexpr auto scale_value = static_cast<double>(1LL << scale);

    int_type_ x = x0, y = y0, z = z0;
    for (auto i_index = 1; i_index < iterations; ++i_index) {
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
#pragma HLS UNROLL
#endif
        const auto i     = constants::cordic_hyper_indexes[i_index];                              // NOLINT
        const auto angle = static_cast<int_type_>(constants::cordic_hyper_tanh[i] * scale_value); // NOLINT

        const auto z_pos       = z >= 0;
        const auto z_pos_new_x = x + (y >> i);
        const auto z_pos_new_y = y + (x >> i);
        const auto z_pos_new_z = z - angle;
        const auto z_neg_new_x = x - (y >> i);
        const auto z_neg_new_y = y - (x >> i);
        const auto z_neg_new_z = z + angle;
        x                      = z_pos ? z_pos_new_x : z_neg_new_x;
        y                      = z_pos ? z_pos_new_y : z_neg_new_y;
        z                      = z_pos ? z_pos_new_z : z_neg_new_z;
    }

    x_n = x;
    y_n = y;
    z_n = z;
}

template<typename execution_, std::int64_t iterations_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
cordic_exp(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t                = dfloat<exp_bits_, man_bits_>;
    using exact_execution      = make_exact_execution<execution_>;
    using int_type             = typename res_t::int_type;
    constexpr auto _1_over_ln2 = static_cast<res_t>(1. / constants::ln2);
    constexpr auto ln2         = static_cast<res_t>(constants::ln2);
    constexpr auto scale       = res_t::dman_bits;
    constexpr auto scale_value = static_cast<double>(1LL << scale);
    constexpr auto r_mask      = utils::make_mask<scale>(true);
    constexpr auto x0          = static_cast<int_type>(constants::cordic_hyper_exp_init * scale_value);
    constexpr auto y0          = x0;

    const auto x_over_ln2 = mul<exact_execution>(x, _1_over_ln2);
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
#pragma HLS bind_op variable = x_over_ln2 op = mul impl = fabric
#endif

    const auto floored_int = floor_int(x_over_ln2);
    const auto q           = floor<exact_execution>(x_over_ln2);
    const auto q_ln2       = mul<exact_execution>(q, ln2);
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
#pragma HLS bind_op variable = q_ln2 op = mul impl = fabric
#endif

    const auto r = dynfloat::to_fixed<scale>(sub<exact_execution>(x, q_ln2)) & r_mask;

    int_type cordic_fixed_res{}, y_n{}, z_n{};
    details_exp::cordic_rotation_hyperbolic_fixed<scale, iterations_>(cordic_fixed_res, y_n, z_n, x0, y0, static_cast<int_type>(r));
    const auto cordic_res = dynfloat::from_fixed<scale, res_t>(cordic_fixed_res);
    return lshift<exact_execution>(cordic_res, floored_int);
}

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
exp(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t                = dfloat<exp_bits_, man_bits_>;
    using exact_execution      = make_exact_execution<execution_>;
    constexpr auto _1_over_ln2 = static_cast<res_t>(1. / constants::ln2);
    switch (static_cast<strategy>(execution_::str)) {
        case strategy::fast: {
            // Algorithm:
            // res = e^x;
            // log2(res) = x * log2(e);
            // res.int_repr() / L - (B - sigma) ~ x * log2(e);   cheap logarithmic space transformation, see also
            //                                                   https://en.wikipedia.org/wiki/Fast_inverse_square_root
            // res.int_repr() ~ x * log2(e) * L + (B-sigma) * L;
            // res.int_repr() ~ x * c + approx_factor;
            // res ~ from_int_repr(x * c + approx_factor);

            // Same algorithm but with different derivation (see: SCHRAUDOLPH, Nicol N. A fast, compact approximation
            // of the exponential function. Neural Computation, 1999, 11. Jg., Nr. 4, S. 853-862.):
            //
            // Per float format definition: from_int_repr( (2^man_bits) * (y + zero_exp) ) = 2^y;       | (0)
            // with  e^y = e^(ln(2)/ln(2) * y) = (e^ln(2))^(y/ln(2)) = 2^(y/ln(2));                     | (1)
            // e^y = 2^(y/ln(2)) = from_int_repr(  (2^man_bits)/ln(2) * y + zero_exp * (2^man_bits) );  | (1 in 0)
            // e^y = from_int_repr( a * y + b );
            // since transformation of a * y + b to integer is inaccurate:
            // e^y ~ from_int_repr( a * y + b - c ); where c is a correction parameter
            // Note: b - c is the approx_factor from the first algorithm

            constexpr auto c_dbl = constants::log2_e * static_cast<double>(uint64_t{1} << res_t::man_bits); // log2(e) * L
            constexpr auto c     = static_cast<res_t>(c_dbl);
            constexpr auto ofs   = static_cast<res_t>(utils::int_repr_exponent_approx_factor(res_t::exp_bits, res_t::man_bits));

            const auto mul_res = mul<exact_execution>(c, x);
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
#pragma HLS bind_op variable = mul_res op = mul impl = fabric
#endif
            const auto value        = add<exact_execution>(mul_res, ofs);
            const auto value_as_int = static_cast<typename res_t::int_repr_type>(std::abs(value));
            const auto res          = res_t::from_int_repr(0, value_as_int);
            return res;
        }
        case strategy::refined_fast:
        case strategy::refined:
        case strategy::refined_exact:
        case strategy::exact: {
            // if (res_t::man_bits > pltfrm_dsp_opt::man_bits && execution_{} == strategy::exact)
            //{
            //     return details_exp::cordic_exp<execution_, res_t::dman_bits>(x);
            // }

            // if (res_t::man_bits > pltfrm_dsp_opt::man_bits && execution_{} == strategy::refined_exact)
            //{
            //     return details_exp::cordic_exp<execution_, 17>(x);
            // }

            const auto mul_res = mul<exact_execution>(x, _1_over_ln2);

#if defined(DYNFLOAT_XILINX_SYNTHESIS)
#pragma HLS bind_op variable = mul_res op = mul impl = fabric
#endif
            return dynfloat::pow2<execution_>(mul_res);
        }
    }
    return {};
}

template<typename execution_, typename res_t>
constexpr auto
cap_strategy() -> strategy {
    strategy capped_str = execution_::str;

    if (res_t::man_bits < std_f32::man_bits)
        capped_str = utils::min(execution_::str, strategy::refined_exact);

    if (res_t::man_bits < std_f32::man_bits / 2)
        capped_str = utils::min(execution_::str, strategy::refined_fast);

    return capped_str;
}

} // namespace details_exp

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
exp(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t       = dfloat<exp_bits_, man_bits_>;
    using round_asm_t = assumptions_tuple<sign_behavior::positive, zero_behavior::non_zero>;

    constexpr auto capped_str = details_exp::cap_strategy<execution_, res_t>();
    if (res_t::man_bits > pltfrm_dsp_opt::man_bits && execution_{} == strategy::exact) {
        if (execution_::nfi) {
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
            if (man_bits_ <= std_f32::man_bits) {
                return static_cast<res_t>(std::exp(static_cast<float>(static_cast<std_f32>(x))));
            }
#endif
        }

        using imm_t        = dfloat<exp_bits_, man_bits_>;
        const auto exp_res = details_exp::exp<set_execution_strategy<capped_str, execution_>>(static_cast<imm_t>(x));

        return dynfloat::round<res_t, execution_::rnd, round_asm_t>(exp_res);
    }
    using imm_t        = expand_to_dsp_opt<res_t>;
    const auto exp_res = details_exp::exp<set_execution_strategy<capped_str, execution_>>(static_cast<imm_t>(x));
    return dynfloat::round<res_t, execution_::rnd, round_asm_t>(exp_res);
}

// NOLINTEND
} // namespace dynfloat
