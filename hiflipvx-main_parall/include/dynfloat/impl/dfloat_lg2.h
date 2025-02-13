#pragma once
#include "dfloat_abs.h"

namespace dynfloat {
namespace details_lg2 {

template<std::int64_t scale, std::int64_t iterations, typename int_type_>
DYNFLOAT_FORCE_INLINE static constexpr auto
limited_lg2(int_type_ x) noexcept -> int_type_ {
    const auto ln_x            = dynfloat::fixed_ln_limited_range<scale, iterations>(x);
    constexpr auto _1_over_ln2 = static_cast<int_type_>(1. / constants::ln2 * (1LL << scale));
    const auto log2_x          = (static_cast<std::int64_t>(ln_x) * _1_over_ln2) >> scale;
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
#pragma HLS bind_op variable = log2_x op = mul impl = fabric
#endif
    return static_cast<int_type_>(log2_x);
}

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
lg2_lut_2x8(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t            = dfloat<exp_bits_, man_bits_>;
    using uint_type        = typename res_t::uint_type;
    using double_dman_type = typename res_t::double_dman_type;

    const auto initial_res       = make_dfloat<res_t>(x.signed_exp() - res_t::zero_exp);
    const auto initial_res_fixed = (x.signed_exp() - res_t::zero_exp) << res_t::man_bits;
    if (x.unsigned_man() == 0)
        return initial_res;

    const auto r          = x.unsigned_man();
    constexpr auto tbl_sz = 8;

    // NOLINTNEXTLINE
    constexpr uint_type a_s[tbl_sz] = {
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0017067457708698282))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.019340600967335274))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.04811707826863744))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.08385994276835246))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.12393477887969936))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.16663155940727387))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.21081574455202814))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.25572239069596586))),
    };

    // NOLINTNEXTLINE
    constexpr uint_type b_s[tbl_sz] = {
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3587716217349053))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2155749016710637))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.0996951583964674))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.0039936457174563))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.923620457641637))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.8551650511786876))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.7961590651397346))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.7447718770693366))),
    };

    const auto segment_idx = x.unsigned_man() >> (res_t::man_bits - 3);
    const auto a           = a_s[segment_idx];
    const auto b           = b_s[segment_idx];
    const auto b_r         = (static_cast<double_dman_type>(b) * r) >> res_t::man_bits;

    const auto tree_add = a + b_r + initial_res_fixed;
    return dynfloat::from_fixed<res_t::man_bits, res_t>(tree_add);
}

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
lg2_lut_3x8(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t    = dfloat<exp_bits_, man_bits_>;
    using int_type = typename res_t::int_type;
    using imm_t    = std::make_signed_t<typename res_t::double_dman_type>;

    const auto initial_res       = make_dfloat<res_t>(x.signed_exp() - res_t::zero_exp);
    const auto initial_res_fixed = (x.signed_exp() - res_t::zero_exp) << res_t::man_bits; // NOLINT
    if (x.unsigned_man() == 0)
        return initial_res;

    const auto r          = x.unsigned_man();
    constexpr auto tbl_sz = 8;

    // NOLINTNEXTLINE
    constexpr int_type a_s[tbl_sz] = {
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(4.02606804303815e-05))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0020023313985430766))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.007730326121782566))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.017443956229814095))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.030768948071633995))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.047161298436623866))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.06607219586810663))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.08700747094955603))),
    };

    // NOLINTNEXTLINE
    constexpr int_type b_s[tbl_sz] = {
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4387627534926364))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4076295142175308))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.361663229112927))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3096891031831783))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2562455370539958))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2036749533515412))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1531550099553738))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.105231097739591))),
    };

    // NOLINTNEXTLINE
    constexpr int_type c_s[tbl_sz] = {
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.6399284437289972))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.5121454706378361))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.4191488331995976))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.34936618950599874))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.29566670592459543))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.25346172506007125))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.21968981219408779))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.1922449421360028))),
    };

    const auto segment_idx = x.unsigned_man() >> (res_t::man_bits - 3);
    const auto a           = a_s[segment_idx];
    const auto b           = b_s[segment_idx];
    const auto c           = c_s[segment_idx];
    const auto r2          = (static_cast<imm_t>(r) * r) >> res_t::man_bits;
    const auto b_r         = (static_cast<imm_t>(b) * r) >> res_t::man_bits;
    const auto c_r2        = (static_cast<imm_t>(c) * r2) >> res_t::man_bits;

    const auto tree_add = a + b_r + c_r2 + initial_res_fixed;
    return dynfloat::from_fixed<res_t::man_bits, res_t>(tree_add);
}

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
lg2_lut_3x32(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t    = dfloat<exp_bits_, man_bits_>;
    using int_type = typename res_t::int_type;
    using imm_t    = std::make_signed_t<typename res_t::double_dman_type>;

    const auto initial_res       = make_dfloat<res_t>(x.signed_exp() - res_t::zero_exp);
    const auto initial_res_fixed = (x.signed_exp() - res_t::zero_exp) << res_t::man_bits; // NOLINT
    if (x.unsigned_man() == 0)
        return initial_res;

    const auto r          = x.unsigned_man();
    constexpr auto tbl_sz = 32;

    // NOLINTNEXTLINE
    constexpr int_type a_s[tbl_sz] = {
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(7.052214097092932e-07))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(4.166003417936659e-05))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.00018881244294910449))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0004916326604647026))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0009861173722534033))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.001697727399662563))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0026436868751780196))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.003834789037837183))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.00527681884336495))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.006971676578440444))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.008918267166381))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.011113205129975157))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.013551373999828087))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.016226370419179403))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.019130856643773342))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.022256840065686845))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.025595894488027682))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.02913933479855757))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.03287835429864572))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.036804132107697285))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.040907916504664854))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.045181088985970064))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.049615212800631525))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.05420206906354963))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.05893368287807732))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.06380234149915509))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.06880060611922545))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.07392131858864559))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.07915760415340856))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.08450287102965603))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.08995079715800397))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.09549534546003496))),
    };

    // NOLINTNEXTLINE
    constexpr int_type b_s[tbl_sz] = {
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4424230385310066))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.439869869885475))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4351843897129442))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4287341089510335))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4208262922127535))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4117185833745403))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4016275814932513))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3907357983269861))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3791973304271252))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3671425045687755))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3546816988514365))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3419084986174994))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.328902313093522))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3157305528252294))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3024504478829328))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.289110571120176))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2757521182327878))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2624099866179108))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2491136872056643))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2358881170910934))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2227542159025688))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2097295246405793))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1968286626080271))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1840637352001977))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1714446833456122))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1589795834417487))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1466749052754608))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1345357341567137))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1225659625058029))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1107684553821286))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.0991452154509338))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.087697462544888))),
    };

    // NOLINTNEXTLINE
    constexpr int_type c_s[tbl_sz] = {
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.6993938097320207))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.6582582370043859))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.6206479975021573))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.5861714637748392))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.5544899191441743))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.5253092054677211))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.49837286960850236))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.4734565089640057))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.4503630821385798))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.42891900100772773))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.4089708588002168))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.39038267859585574))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.37303358976679135))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.35681585791996895))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.3416332081991129))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.32739939306588894))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.3140369646540364))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.30147621893721066))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.28965428484067957))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2785143359026847))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2680049060870999))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2580792942361825))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.24869504432695066))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2398134907009819))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.23139935910427312))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.22342041589806172))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2158471588734301))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2086525441474123))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.20181174434395643))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.19530193416537145))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.18910211120272002))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.18319290160608864))),
    };

    const auto segment_idx = x.unsigned_man() >> (res_t::man_bits - 5);
    const auto a           = a_s[segment_idx];
    const auto b           = b_s[segment_idx];
    const auto c           = c_s[segment_idx];
    const auto r2          = (static_cast<imm_t>(r) * r) >> res_t::man_bits;
    const auto b_r         = (static_cast<imm_t>(b) * r) >> res_t::man_bits;
    const auto c_r2        = (static_cast<imm_t>(c) * r2) >> res_t::man_bits;

    const auto tree_add = a + b_r + c_r2 + initial_res_fixed;
    return dynfloat::from_fixed<res_t::man_bits, res_t>(tree_add);
}

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
lg2_lut_3x64(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t    = dfloat<exp_bits_, man_bits_>;
    using int_type = typename res_t::int_type;
    using imm_t    = std::make_signed_t<typename res_t::double_dman_type>;

    const auto initial_res       = make_dfloat<res_t>(x.signed_exp() - res_t::zero_exp);
    const auto initial_res_fixed = (x.signed_exp() - res_t::zero_exp) << res_t::man_bits; // NOLINT
    if (x.unsigned_man() == 0)
        return initial_res;

    const auto r          = x.unsigned_man();
    constexpr auto tbl_sz = 64;

    // NOLINTNEXTLINE
    constexpr int_type a_s[tbl_sz] = {
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(8.991389914515466e-08))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(5.4823790400573835e-06))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(2.56560941162276e-05))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(6.889585427921618e-05))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.00014234902728049725))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.00025215770922448755))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.00040357524222534025))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0006010690584057343))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.000848411549423389))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0011487604360738146))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0015047299151227378))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0019184536993286372))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.002391640916847848))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0029256257173670974))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.003521411324101109))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.004179709178870894))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0049009737432683664))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.00568543345846706))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0065331182995436166))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.007443884301309822))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.00841743540561879))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.00945334292266864))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.010551062867605765))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.011709951409130781))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.012929278637955122))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.01420824082910599))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.015545971378287504))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.016941550522489024))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.01839401401909413))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.01990236084881758))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.021465560080599744))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.023082556974088675))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0247522783866998))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.026473637582455467))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.028245538490409672))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.03006687946515285))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.03193655663477557))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0338534668078978))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.03581651008664721))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.037824592110951016))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.03987662605914011))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.041971534373828945))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.04410825024353926))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.04628571894181732))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.048502898922038185))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.05075876276316649))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.05305229799148492))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.055382507763624744))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.05774841140760145))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.06014904489234141))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.06258346118374902))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.06505073052733223))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.06754994066272957))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.07008019694262657))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.07264062242120417))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.07523035791382426))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.07784856194894019))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.08049441070124885))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.08316709792870824))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.08586583484111543))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.08858984992798469))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0913383888566841))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.0941107035909412))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(0.09690607302494625))),
    };

    // NOLINTNEXTLINE
    constexpr int_type b_s[tbl_sz] = {
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4426258352376227))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4419556582218203))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4406730265055607))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4388328614366905))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4364852831764843))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4336760691533996))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4304470640153681))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.426836546741086))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.422879559854124))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4186082050514415))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4140519090343844))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.409237662831032))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.4041902375479083))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3989323790902972))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3934849841113532))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3878672591856684))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3820968649722545))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3761900468867907))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.370161753721959))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3640257454094353))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3577946910040168))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.351480257884873))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3450931930112802))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.338643397058604))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3321399920414478))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3255913831318367))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3190053151173515))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3123889241446705))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.3057487850351208))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2990909547008869))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2924210118815154))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2857440937277715))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2790649293274328))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2723878704932758))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2657169202054774))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2590557586208888))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2524077671685063))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2457760506538307))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.239163457700215))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2325725995569883))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2260058675385608))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.21946544901175))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2129533423344583))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2064713704387486))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.2000211936694996))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1936043214341225))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1872221231324147))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1808758381498592))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1745665853104583))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.168295371385284))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1620630993384111))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1558705755455776))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1497185168273063))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1436075569304194))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.137538252311515))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1315110878781525))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1255264819728836))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.119584791314992))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1136863153815284))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1078313005582459))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.1020199440126817))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.0962523971811606))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.0905287910842931))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(1.084849195434697))),
    };

    // NOLINTNEXTLINE
    constexpr int_type c_s[tbl_sz] = {
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.710225392800993))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.688704129095683))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.66814646218333))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.6484957129093232))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.6296993090500962))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.6117084333111222))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.5944777060370257))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.5779648987527253))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.5621306751903035))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.54693835677881))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.5323537100823614))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.5183447537215216))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.5048815829508726))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.49193620988285147))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.4794824179441548))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.467495629011637))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.4559527820732683))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.44483222221060714))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.43411359895685564))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.4237777731325423))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.41380673135711277))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.4041835074917799))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.3948921104172314))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.38591745758185425))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.3772453137131606))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.3688622343904626))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.360755513834647))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.35291313681113934))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.34532373401304994))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.337976540886757))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.33086135939720407))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.3239685226560596))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.31728886203814))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.310813676700036))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.30453470521320014))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2984440992290729))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2925343988970326))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2867985100438091))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.28122968286834293))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.27582149206705253))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2705678182835527))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2654628308041538))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.26050097140887374))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2556769391480884))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.25098567626241675))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.24642235485498531))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2419823645316228))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.23766130067497215))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.23345495359535562))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.22935929820960155))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.22537048445133223))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.22148482817829063))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2176988027117659))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.21400903080416356))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.2104122770788308))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.20690544098761166))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.20348555009468328))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.20014975378512645))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.1968953173322916))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.19371961625120093))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.19062013105758524))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.1875944422192788))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.18464023660844386))),
        static_cast<int_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<std_f64>(-0.18175528003389196))),
    };

    const auto segment_idx = x.unsigned_man() >> (res_t::man_bits - 6);
    const auto a           = a_s[segment_idx];
    const auto b           = b_s[segment_idx];
    const auto c           = c_s[segment_idx];
    const auto r2          = (static_cast<imm_t>(r) * r) >> res_t::man_bits;
    const auto b_r         = (static_cast<imm_t>(b) * r) >> res_t::man_bits;
    const auto c_r2        = (static_cast<imm_t>(c) * r2) >> res_t::man_bits;

    const auto tree_add = a + b_r + c_r2 + initial_res_fixed;
    return dynfloat::from_fixed<res_t::man_bits, res_t>(tree_add);
}

template<typename execution_, std::int64_t iterations_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
lg2_cordic(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t                  = dfloat<exp_bits_, man_bits_>;
    const auto initial_res_fixed = (x.signed_exp() - res_t::zero_exp) << res_t::man_bits; // NOLINT
    const auto lg2_dman          = details_lg2::limited_lg2<res_t::man_bits, iterations_>(static_cast<std::int64_t>(x.dman()));
    return dynfloat::from_fixed<res_t::man_bits, res_t>(lg2_dman + initial_res_fixed);
}

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
lg2(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t              = dfloat<exp_bits_, man_bits_>;
    using dsp_imm_t          = expand_to_dsp_opt<res_t>;
    using expanded_dsp_imm_t = expand_for_rounding<expand_for_rounding<dsp_imm_t, execution_>, execution_>;
    switch (static_cast<strategy>(execution_::str)) {
        case strategy::fast: {
            constexpr auto offset = dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.0435));
            return dynfloat::from_fixed<res_t::man_bits, res_t>(x.int_repr() - (res_t::zero_exp << res_t::man_bits) + offset);
        }
        case strategy::refined_fast: {
            return dynfloat::round<res_t, execution_::rnd>(details_lg2::lg2_lut_2x8<execution_>(static_cast<dsp_imm_t>(x)));
        }
        case strategy::refined: {
            return dynfloat::round<res_t, execution_::rnd>(details_lg2::lg2_lut_3x8<execution_>(static_cast<dsp_imm_t>(x)));
        }
        case strategy::refined_exact: {
            return dynfloat::round<res_t, execution_::rnd>(details_lg2::lg2_lut_3x32<execution_>(static_cast<dsp_imm_t>(x)));
        }
        case strategy::exact: {
            if (res_t::man_bits < std_f32::man_bits) {
                return dynfloat::round<res_t, execution_::rnd>(
                    details_lg2::lg2_lut_3x32<execution_>(static_cast<max_dfloat<std_f32, res_t>>(x)));
            }
            return dynfloat::round<res_t, execution_::rnd>(details_lg2::lg2_lut_3x64<execution_>(static_cast<expanded_dsp_imm_t>(x)));
        }
    }
    return {};
}

template<typename execution_, typename res_t>
constexpr auto
cap_strategy() -> strategy {
    strategy capped_str = execution_::str;

    if (res_t::man_bits < std_f32::man_bits / 2)
        capped_str = utils::min(execution_::str, strategy::refined_exact);

    return capped_str;
}
} // namespace details_lg2

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
lg2(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t               = dfloat<exp_bits_, man_bits_>;
    constexpr auto capped_str = details_lg2::cap_strategy<execution_, res_t>();
    return details_lg2::lg2<set_execution_strategy<capped_str, execution_>>(x);
}

// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
} // namespace dynfloat
