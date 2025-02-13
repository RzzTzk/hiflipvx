#pragma once
#include "dfloat_div.h"


namespace dynfloat {
namespace details_pow2 {
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
pow2_lut_2x4(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t             = dfloat<exp_bits_, man_bits_>;
    using exact_execution   = make_exact_execution<execution_>;
    using uint_type         = typename res_t::uint_type;
    constexpr auto r_mask   = utils::make_mask<res_t::man_bits>(true);
    const auto x_fixed      = dynfloat::to_fixed<res_t::man_bits>(x);
    const auto x_fixed_frac = x_fixed & r_mask;
    const auto r            = x_fixed_frac;
    const auto shift        = x_fixed >> man_bits_;
    constexpr auto tbl_sz   = 8;

    // NOLINTNEXTLINE
    constexpr uint_type a_s[tbl_sz] = {
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9973164238582539))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.961121868719488))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.8755275074651101))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.723135489657039))),
    };

    // NOLINTNEXTLINE
    constexpr uint_type b_s[tbl_sz] = {
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7564499396957813))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.8995756660180798))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0697818010499083))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.2721921513543648))),
    };
    const auto segment_idx = (r >> (res_t::man_bits - 2)) & 0x3;
    const auto a           = a_s[segment_idx];
    const auto b           = b_s[segment_idx];
    const auto f_res       = a + ((r * b) >> res_t::man_bits);
    static_assert(res_t::man_bits < 32, "TODO: fixed mul for 64");

    const auto res = dynfloat::from_fixed<res_t::man_bits, res_t>(f_res);
    return static_cast<res_t>(std::abs(dynfloat::safe_lshift<exact_execution>(res, shift)));
}

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
pow2_lut_2x8(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t             = dfloat<exp_bits_, man_bits_>;
    using exact_execution   = make_exact_execution<execution_>;
    using uint_type         = typename res_t::uint_type;
    constexpr auto r_mask   = utils::make_mask<res_t::man_bits>(true);
    const auto x_fixed      = dynfloat::to_fixed<res_t::man_bits>(x);
    const auto x_fixed_frac = x_fixed & r_mask;
    const auto r            = x_fixed_frac;
    const auto shift        = x_fixed >> man_bits_;
    constexpr auto tbl_sz   = 8;

    // NOLINTNEXTLINE
    constexpr uint_type a_s[tbl_sz] = {
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9993522527855269))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9911143255489808))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.973198860661514))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9439215561871441))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9013725004740403))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.8433891305168771))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7675261434800701))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6710220313727954))),
    };

    // NOLINTNEXTLINE
    constexpr uint_type b_s[tbl_sz] = {
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7239712624322996))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7894962667502928))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.8609517912611373))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9388745939570133))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0238500135840454))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.1165163665766926))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.217569741948658))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.3277692301584705))),
    };
    const auto segment_idx = (r >> (res_t::man_bits - 3)) & 0x7;
    const auto a           = a_s[segment_idx];
    const auto b           = b_s[segment_idx];
    const auto f_res       = a + ((r * b) >> res_t::man_bits);
    static_assert(res_t::man_bits < 32, "TODO: fixed mul for 64");

    const auto res = dynfloat::from_fixed<res_t::man_bits, res_t>(f_res);
    return static_cast<res_t>(std::abs(dynfloat::safe_lshift<exact_execution>(res, shift)));
}

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
pow2_lut_2x32(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t             = dfloat<exp_bits_, man_bits_>;
    using exact_execution   = make_exact_execution<execution_>;
    using uint_type         = typename res_t::uint_type;
    constexpr auto r_mask   = utils::make_mask<res_t::man_bits>(true);
    const auto x_fixed      = dynfloat::to_fixed<res_t::man_bits>(x);
    const auto x_fixed_frac = x_fixed & r_mask;
    const auto r            = x_fixed_frac;
    const auto shift        = x_fixed >> man_bits_;
    constexpr auto tbl_sz   = 32;

    // NOLINTNEXTLINE
    constexpr uint_type a_s[tbl_sz] = {
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9999605603925803))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9994803873174745))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9984997190924216))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9969968671176951))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9949494329365257))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9923342875467343))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9891275501467863))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9853045663012323))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9808398855104625))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9757072381690703))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9698795118968389))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.963328727225841))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9560260126269673))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9479415788585172))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.93904469261897))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9293036494861323))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9186857461237039))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9071572517364664))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.894683378753905))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.8812282527232547))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.866754881390148))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.8512251229464589))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.8345996534240592))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.8168379332108675))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7978981726681021))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7777372968242919))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7563109091229268))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7335732541976223))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7094771796518997))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6839740968166494))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6570139404576707))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6285451274087015))),
    };

    // NOLINTNEXTLINE
    constexpr uint_type b_s[tbl_sz] = {
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7007032540152466))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7160466588818752))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7317260405996808))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7477487561044927))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7641223234279566))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7808544252254572))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.7979529123802186))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.8154258076878879))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.8332813096201825))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.8515277961721014))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.870173828792878))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.8892281564030279))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9086997194995513))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9285976543507712))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9489312972834358))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.969710189063024))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.9909440793708453))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0126429313781542))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.034816926421695))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0574764687798546))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0806321905557317))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.1042949566647784))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.1284758699332795))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.1531862763076113))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.1784377701784465))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.2042421998201327))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.2306116729508259))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.2575585624128633))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.2850955119789003))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.313235442284077))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.341991556888433))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.3713773484727065))),
    };
    const auto segment_idx = (r >> (res_t::man_bits - 5)) & 0x1F;
    const auto a           = a_s[segment_idx];
    const auto b           = b_s[segment_idx];
    const auto f_res       = a + ((r * b) >> res_t::man_bits);
    static_assert(res_t::man_bits < 32, "TODO: fixed mul for 64");

    const auto res = dynfloat::from_fixed<res_t::man_bits, res_t>(f_res);
    return static_cast<res_t>(std::abs(dynfloat::safe_lshift<exact_execution>(res, shift)));
}

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
pow2_lut_3x32(const dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t             = dfloat<exp_bits_, man_bits_>;
    using exact_execution   = make_exact_execution<execution_>;
    using uint_type         = typename res_t::uint_type;
    constexpr auto r_mask   = utils::make_mask<res_t::man_bits>(true);
    const auto x_fixed      = dynfloat::to_fixed<res_t::man_bits>(x);
    const auto x_fixed_frac = x_fixed & r_mask;
    const auto r            = x_fixed_frac;
    const auto shift        = x_fixed >> man_bits_;
    constexpr auto tbl_sz   = 32;

    // NOLINTNEXTLINE
    constexpr uint_type a_s[tbl_sz] = {
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0000000854814315))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.000005468678831))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0000269072979286))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0000759476601937))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0001648715433005))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0003067280537463))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0005153666576811))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0008054714038144))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0011925963854775))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.001693202470232))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.002324695352936))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.003105464958395))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0040549262539322))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0051935615077667))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0065429640419197))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0081258835183107))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0099662728416803))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0120893366766808))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0145215816795456))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0172908684781703))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0204264654530562))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0239591043766154))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0279210380036545))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0323460996037))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0372697645844369))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.04272921417504))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0487634013422564))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0554131189348936))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0627210701056526))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.070731941211534))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.079492477072165))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(1.0890515588699259))),
    };

    // NOLINTNEXTLINE
    constexpr uint_type b_s[tbl_sz] = {
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6931143891469729))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6927814910878531))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6921016758878835))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6910599101732586))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6896406685338832))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6878279191893739))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6856051092460618))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6829551495758466))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6798603992588852))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.676302649617142))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6722631077982868))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6677223799421341))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6626604538664225))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.657056681263299))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.650889759454742))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6441377126522809))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6367778726507822))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6287868590879384))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6201405590985587))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6108141064276538))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.6007818600492669))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.5900173821054864))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.5784934153547283))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.5661818599068056))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.5530537494461782))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.5390792267244251))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.5242275184345999))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.5084669094412675))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.4917647162432104))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.47408725976970345))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.45539983754343893))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.43566669486301635))),
    };

    // NOLINTNEXTLINE
    constexpr uint_type c_s[tbl_sz] = {
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.24284442860471245))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.24816202969270762))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.2535960710993237))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.2591491025214907))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.2648237295238971))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.27062261467878557))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.27654847890383394))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.2826041026649193))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.2887923273418096))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.29511605649875605))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.3015782573130821))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.3081819619160342))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.31493026882918684))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.3218263444318836))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.3288734244544287))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.33607481544549955))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.34343389637982114))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.350954120222184))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.35863901553869937))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.36649218817109386))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.37451732290978157))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.38271818522855483))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.3910986230657443))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.39966256861976035))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.4084140401829188))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.4173571440379078))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.42649607638853126))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.4358351253212902))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.4453786728103921))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.45513119680455816))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.46509727327941164))),
        static_cast<uint_type>(dynfloat::to_fixed<res_t::man_bits>(static_cast<res_t>(0.47528157844914176))),
    };

    const auto segment_idx = (r >> (res_t::man_bits - 5)) & 0x1F;
    const auto a           = a_s[segment_idx];
    const auto b           = b_s[segment_idx];
    const auto c           = c_s[segment_idx];
    const auto r2          = (r * r) >> res_t::man_bits;
    const auto b_r         = (r * b) >> res_t::man_bits;
    const auto c_r2        = (c * r2) >> res_t::man_bits;
    const auto f_res       = a + b_r + c_r2;
    static_assert(res_t::man_bits < 32, "TODO: fixed mul for 64");

    const auto res = dynfloat::from_fixed<res_t::man_bits, res_t>(f_res);
    return static_cast<res_t>(std::abs(dynfloat::safe_lshift<exact_execution>(res, shift)));
}

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
pow2(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t           = dfloat<exp_bits_, man_bits_>;
    using exact_execution = make_exact_execution<execution_>;
    switch (static_cast<strategy>(execution_::str)) {
        case strategy::fast: {
            constexpr auto c_dbl = static_cast<double>(uint64_t{1} << res_t::man_bits); // log2(e) * L
            constexpr auto c     = static_cast<res_t>(c_dbl);
            constexpr auto ofs   = static_cast<res_t>(utils::int_repr_exponent_approx_factor(res_t::exp_bits, res_t::man_bits));

            const auto value        = add<exact_execution>(mul<exact_execution>(c, x), ofs);
            const auto value_as_int = static_cast<typename res_t::int_repr_type>(std::abs(value));
            const auto res          = res_t::from_int_repr(0, value_as_int);
            return res;
        }
        case strategy::refined_fast: {
            return details_pow2::pow2_lut_2x4<execution_>(x);
        }
        case strategy::refined: {
            return details_pow2::pow2_lut_2x8<execution_>(x);
        }
        case strategy::refined_exact: {
            return details_pow2::pow2_lut_2x32<execution_>(x);
        }
        case strategy::exact: {
            return details_pow2::pow2_lut_3x32<execution_>(x);
        }
    }
    return {};
}

template<typename execution_, typename df_>
constexpr auto
cap_strategy() -> strategy {
    strategy capped_str = execution_::str;
    if (df_::man_bits < std_f32::man_bits)
        capped_str = utils::min(execution_::str, strategy::refined_exact);
    return capped_str;
}
} // namespace details_pow2

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
pow2(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_> {
    using res_t               = dfloat<exp_bits_, man_bits_>;
    using imm_t               = expand_to_dsp_opt<res_t>;
    constexpr auto capped_str = details_pow2::cap_strategy<execution_, res_t>();
    constexpr auto one        = dynfloat::make_dfloat<res_t>(1);
    if (x.is_zero())
        return one;
    const auto exp_res = details_pow2::pow2<set_execution_strategy<capped_str, execution_>>(static_cast<imm_t>(x));

    return dynfloat::round<res_t, execution_::rnd, assumptions_tuple<sign_behavior::positive, zero_behavior::non_zero>>(exp_res);
}

// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
} // namespace dynfloat
