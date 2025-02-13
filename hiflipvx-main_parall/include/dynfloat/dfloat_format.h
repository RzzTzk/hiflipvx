#pragma once
//#include "dynfloat.h"
//#include "fmt/format.h"
//#include "fmt/ranges.h"
//
//template<std::int64_t exp_bits_, std::int64_t man_bits_>
//struct fmt::formatter<dynfloat::dfloat<exp_bits_, man_bits_>>
//{
//    template<typename ParseContext>
//    constexpr auto parse(ParseContext& ctx)
//    {
//        return ctx.begin();
//    }
//
//    template<typename FormatContext>
//    auto format(const dynfloat::dfloat<exp_bits_, man_bits_> in, FormatContext& ctx)
//    {
//        return fmt::format_to(
//            ctx.out(),
//            "dfloat<{},{}>({},{},{})",
//            exp_bits_,
//            man_bits_,
//            in.unsigned_sign(),
//            in.unsigned_exp(),
//            in.unsigned_man());
//    }
//};
