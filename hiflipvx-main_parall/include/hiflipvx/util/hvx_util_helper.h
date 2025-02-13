/**
 *  Copyright <2024> <Lester Kalms>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
 * “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
 * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Additional restriction: The Software and its derivatives may not be used for, or in support of, any military purposes.
 *
 * @file    hvx_util_helper.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_HELPER_H_
#define HVX_UTIL_HELPER_H_

#include "hvx_util_interface.h"
#include "hvx_util_math_dfixed.h"
#include "hvx_util_dfloat.h"
#include "hvx_util_weights_bias.h"
#include "hvx_util_window.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_UTIL_HELPER_H_

///*!
// * @brief
// */
// struct hvx_ap_limits {
//    int64_t min_;
//    int64_t max_;
//};
//
///*!
// * @brief
// */
// struct hvx_ap_data {
//    int64_t min_; // minimum possible value
//    int64_t max_; // maximum possible value
//    int64_t bit_; // minimum number of bits required
//};
//
///*!
// * @brief
// */
// template<typename OprType, typename DstType>
// struct OprIntType {
//    const DstType dst_min_;
//    const DstType dst_max_;
//    const uint8_t opr_bit_;
//    const uint8_t dst_bit_;
//    const OprType opr_type_;
//    const DstType dst_type_;
//};
//
//// NOLINTNEXTLINE
// #def ine  HIFL IP_NEEDED_DATA_ TYPE(SRC1_TYPE, SRC2_TYPE, NEEDED_BITS) \
//    std::conditional_t< \
//        ((std::numeric_limits<SRC1_TYPE>::is_signed) || (std::numeric_limits<SRC2_TYPE>::is_signed)), \
//        std::conditional_t<((NEEDED_BITS) > 32), int64_t, \
//                           std::conditional_t<((NEEDED_BITS) > 16), int32_t, std::conditional_t<((NEEDED_BITS) > 8), int16_t, int8_t>>>,
//                            \
//        std::conditional_t< \
//            ((NEEDED_BITS) > 32), uint64_t, \
// std::conditional_t<((NEEDED_BITS) > 16), uint32_t, std::conditional_t<((NEEDED_BITS) >
//            8),
//             uint16_t, uint8_t>>>>
//
//// NOLINTNEXTLINE
// #def ine HIFLIP_NEEDED_S IGNED_TYPE(NEEDED_BITS)        \
//    std::conditional_t<((NEEDED_BITS) > 32), int64_t, \
//                       std::conditional_t<((NEEDED_BITS) > 16), int32_t, std::conditional_t<((NEEDED_BITS) > 8), int16_t, int8_t>>>
//
//// NOLINTNEXTLINE
// #def ine HIFLIP_NEEDED_U NSIGNED_TYPE(NEEDED_BITS)       \
//    std::conditional_t<((NEEDED_BITS) > 32), uint64_t, \
//                       std::conditional_t<((NEEDED_BITS) > 16), uint32_t, std::conditional_t<((NEEDED_BITS) > 8), uint16_t, uint8_t>>>
//
//// NOLINTNEXTLINE
// #define HIFLIP_GET_TYPE(DATA) decltype((DATA).max_)
//
///*!
// * @brief
// */
// template<int64_t VALUE>
// HVX_FORCE_INLINE constexpr auto
// GetMsb() noexcept -> int64_t {
//    int64_t MSB = 0;
//    for (int64_t i = 0; i < 64; i++) {
//        if ((VALUE & static_cast<int64_t>(1) << i) != 0)
//            MSB = i;
//    }
//    return MSB;
//}
//
///*!
// * @brief
// */
// template<int64_t VALUE>
// HVX_FORCE_INLINE constexpr auto
// GetNeededBitsSigned() noexcept -> int64_t {
//    constexpr int64_t ABS_VAL = hvx::Abs(VALUE) - 1;
//    return (VALUE < 0) ? (hvx::GetMsb<ABS_VAL>() + 2) : (hvx::GetMsb<VALUE>() + 2);
//}
//
///*!
// * @brief
// */
// template<int64_t VALUE>
// HVX_FORCE_INLINE constexpr auto
// GetNeededBitsUnsigned() noexcept -> int64_t {
//    return hvx::GetMsb<VALUE>() + 1;
//}
//
///*!
// * @brief
// */
// template<bool IS_SIGNED, int64_t VALUE>
// HVX_FORCE_INLINE constexpr auto
// GetNeededBits() noexcept -> int64_t {
//    constexpr int64_t bits_signed   = hvx::GetNeededBitsSigned<VALUE>();
//    constexpr int64_t bits_unsigned = hvx::GetNeededBitsUnsigned<VALUE>();
//    return (IS_SIGNED) ? (bits_signed) : (bits_unsigned);
//}
//
///*!
// * @brief
// */
// template<bool IS_SIGNED, int64_t VAL1, int64_t VAL2>
// HVX_FORCE_INLINE constexpr auto
// GetNeededBitsMax() noexcept -> int64_t {
//    constexpr int64_t bits_min_val = hvx::GetNeededBits<IS_SIGNED, VAL1>();
//    constexpr int64_t bits_max_val = hvx::GetNeededBits<IS_SIGNED, VAL2>();
//    return hvx::Max<int64_t>(bits_min_val, bits_max_val);
//}
//
////
// template<typename TYPE>
// HVX_FORCE_INLINE constexpr auto
// GetNumericLimits() noexcept -> hvx_ap_limits {
//     constexpr auto min = static_cast<int64_t>(std::numeric_limits<TYPE>::lowest());
//     constexpr auto max = static_cast<int64_t>(std::numeric_limits<TYPE>::max());
//     return hvx_ap_limits{min, max};
// }
//
// template<typename A_TYPE, typename B_TYPE>
// HVX_FORCE_INLINE constexpr auto
// IsSigned() noexcept -> bool {
//     return std::numeric_limits<A_TYPE>::is_signed || std::numeric_limits<B_TYPE>::is_signed;
// }
//
// template<typename Type>
// HVX_FORCE_INLINE constexpr auto
// IsSigned() noexcept -> bool {
//     return std::numeric_limits<Type>::is_signed;
// }
//
////
// template<int64_t A, int64_t B>
// HVX_FORCE_INLINE constexpr auto
// CheckOverflowAdd() noexcept -> int64_t {
//     constexpr bool oflow = (B > 0) && (A > static_cast<int64_t>(static_cast<uint64_t>(INT64_MAX) - static_cast<uint64_t>(B)));
//     constexpr bool uflow = (B < 0) && (A < static_cast<int64_t>(static_cast<uint64_t>(INT64_MIN) - static_cast<uint64_t>(B)));
//     static_assert(!oflow || !uflow, "Possible overflow/underflow!");
//     return A + B;
// }
//
////
// template<bool is_signed, int64_t MIN_A, int64_t MAX_A, int64_t MIN_B, int64_t MAX_B>
// HVX_FORCE_INLINE constexpr auto
// GetNeededBitsAdd() noexcept -> hvx_ap_data {
//     constexpr int64_t min = hvx::CheckOverflowAdd<MIN_A, MIN_B>();
//     constexpr int64_t max = hvx::CheckOverflowAdd<MAX_A, MAX_B>();
//     constexpr int64_t bit = hvx::GetNeededBitsMax<is_signed, min, max>();
//     return hvx_ap_data{min, max, bit};
// }
//
///*!
// * @brief
// */
// template<typename A_TYPE, typename B_TYPE>
// struct add1_t {
// #define a_limit hvx::GetNumericLimits<A_TYPE>()
// #define b_limit hvx::GetNumericLimits<B_TYPE>()
// #define ap_data GetNeededBitsAdd<IsSigned<A_TYPE, B_TYPE>(), a_limit.min_, a_limit.max_, b_limit.min_, b_limit.max_>()
//    using comp_type      = HIFLIP_NEEDED_DATA_TYPE(A_TYPE, B_TYPE, ap_data.bit_);
//    const comp_type min_ = static_cast<comp_type>(ap_data.min_); // minimum possible value
//    const comp_type max_ = static_cast<comp_type>(ap_data.max_); // maximum possible value
//    const comp_type bit_ = static_cast<comp_type>(ap_data.bit_); // minimum number of bits required
// #undef a_limit
// #undef b_limit
// #undef ap_is_signed
// #undef ap_data
//};
//
///*!
// * @brief
// */
// template<>
// struct add1_t<float, float> {
//    const float min_ = std::numeric_limits<float>::lowest(); // minimum possible value
//    const float max_ = std::numeric_limits<float>::max();    // maximum possible value
//    const float bit_ = 24;                                        // minimum number of bits required
//};
//
///*!
// * @brief
// */
// template<typename A_TYPE, int64_t MIN_A, int64_t MAX_A, typename B_TYPE, int64_t MIN_B, int64_t MAX_B>
// struct add2_t {
// #define ap_data GetNeededBitsAdd<IsSigned<A_TYPE, B_TYPE>(), MIN_A, MAX_A, MIN_B, MAX_B>()
//    using comp_type      = HIFLIP_NEEDED_DATA_TYPE(A_TYPE, B_TYPE, ap_data.bit_);
//    const comp_type min_ = static_cast<comp_type>(ap_data.min_); // minimum possible value
//    const comp_type max_ = static_cast<comp_type>(ap_data.max_); // maximum possible value
//    const comp_type bit_ = static_cast<comp_type>(ap_data.bit_); // minimum number of bits required
// #undef ap_is_signed
// #undef ap_data
//};
//
///*!
// * @brief
// */
// template<int64_t MIN_A, int64_t MAX_A, int64_t MIN_B, int64_t MAX_B>
// struct add2_t<float, MIN_A, MAX_A, float, MIN_B, MAX_B> {
// #define ap_data hvx::GetNeededBitsAdd<true, MIN_A, MAX_A, MIN_B, MAX_B>()
//    const float min_ = ap_data.min_; // minimum possible value
//    const float max_ = ap_data.max_; // maximum possible value
//    const float bit_ = ap_data.bit_; // minimum number of bits required
// #undef ap_data
//};
//
////
// template<typename SRC_TYPE, typename DST_TYPE>
// HVX_FORCE_INLINE constexpr auto
// OprAdd(SRC_TYPE in1, SRC_TYPE in2) -> DST_TYPE {
//     HVX_INLINE_TOP();
//     return static_cast<DST_TYPE>(in1) + static_cast<DST_TYPE>(in2);
// }
//
// template<typename SrcType>
// HVX_FORCE_INLINE constexpr auto
// OprAdd(SrcType in1, SrcType in2) {
//     HVX_INLINE_TOP();
//     constexpr hvx::add1_t<SrcType, SrcType> res_t{};
//     (void)res_t; // [[maybe_unused]]
//     return static_cast<decltype(res_t.max_)>(in1) + static_cast<decltype(res_t.max_)>(in2);
// }
//
////
// template<int64_t A, int64_t B>
// HVX_FORCE_INLINE constexpr auto
// CheckOverflowSub() -> int64_t {
//     constexpr bool oflow = (B < 0) && (A > static_cast<int64_t>(static_cast<uint64_t>(INT64_MAX) + static_cast<uint64_t>(B)));
//     constexpr bool uflow = (B > 0) && (A < static_cast<int64_t>(static_cast<uint64_t>(INT64_MIN) + static_cast<uint64_t>(B)));
//     static_assert(!oflow || !uflow, "Possible overflow/underflow!");
//     return A - B;
// }
//
////
// template<int64_t MIN_A, int64_t MAX_A, int64_t MIN_B, int64_t MAX_B>
// HVX_FORCE_INLINE constexpr auto
// GetNeededBitsSub() noexcept -> hvx_ap_data {
//     constexpr int64_t min = hvx::CheckOverflowSub<MIN_A, MAX_B>();
//     constexpr int64_t max = hvx::CheckOverflowSub<MAX_A, MIN_B>();
//     constexpr int64_t bit = hvx::GetNeededBitsMax<true, min, max>();
//     return hvx_ap_data{min, max, bit};
// }
//
///*!
// * @brief
// */
// template<typename A_TYPE, typename B_TYPE>
// struct sub1_t {
// #define a_limit hvx::GetNumericLimits<A_TYPE>()
// #define b_limit hvx::GetNumericLimits<B_TYPE>()
// #define ap_data hvx::GetNeededBitsSub<a_limit.min_, a_limit.max_, b_limit.min_, b_limit.max_>()
//    using comp_type      = HIFLIP_NEEDED_SIGNED_TYPE(ap_data.bit_);
//    const comp_type min_ = static_cast<comp_type>(ap_data.min_); // minimum possible value
//    const comp_type max_ = static_cast<comp_type>(ap_data.max_); // maximum possible value
//    const comp_type bit_ = static_cast<comp_type>(ap_data.bit_); // minimum number of bits required
// #undef a_limit
// #undef b_limit
// #undef ap_data
//};
//
///*!
// * @brief
// */
// template<>
// struct sub1_t<float, float> {
//    const float min_ = std::numeric_limits<float>::lowest(); // minimum possible value
//    const float max_ = std::numeric_limits<float>::max();    // maximum possible value
//    const float bit_ = 24;                                        // minimum number of bits required
//};
//
///*!
// * @brief
// */
// template<typename A_TYPE, int64_t MIN_A, int64_t MAX_A, typename B_TYPE, int64_t MIN_B, int64_t MAX_B>
// struct sub2_t {
// #define ap_data hvx::GetNeededBitsSub<MIN_A, MAX_A, MIN_B, MAX_B>()
//    using comp_type      = HIFLIP_NEEDED_SIGNED_TYPE(ap_data.bit_);
//    const comp_type min_ = static_cast<comp_type>(ap_data.min_); // minimum possible value
//    const comp_type max_ = static_cast<comp_type>(ap_data.max_); // maximum possible value
//    const comp_type bit_ = static_cast<comp_type>(ap_data.bit_); // minimum number of bits required
// #undef ap_data
//};
//
///*!
// * @brief
// */
// template<int64_t MIN_A, int64_t MAX_A, int64_t MIN_B, int64_t MAX_B>
// struct sub2_t<float, MIN_A, MAX_A, float, MIN_B, MAX_B> {
// #define ap_data hvx::GetNeededBitsSub<MIN_A, MAX_A, MIN_B, MAX_B>()
//    using comp_type      = HIFLIP_NEEDED_SIGNED_TYPE(ap_data.bit_);
//    const comp_type min_ = static_cast<comp_type>(ap_data.min_); // minimum possible value
//    const comp_type max_ = static_cast<comp_type>(ap_data.max_); // maximum possible value
//    const comp_type bit_ = static_cast<comp_type>(ap_data.bit_); // minimum number of bits required
// #undef ap_data
//};
//
////
// template<typename SRC_TYPE, typename DST_TYPE>
// HVX_FORCE_INLINE constexpr auto
// OprSub(SRC_TYPE in1, SRC_TYPE in2) -> DST_TYPE {
//     HVX_INLINE_TOP();
//     return static_cast<DST_TYPE>(in1) - static_cast<DST_TYPE>(in2);
// }
//
// template<typename SrcType>
// HVX_FORCE_INLINE constexpr auto
// OprSub(SrcType in1, SrcType in2) {
//     HVX_INLINE_TOP();
//     constexpr hvx::sub1_t<SrcType, SrcType> res_t{};
//     (void)res_t; // [[maybe_unused]]
//     return static_cast<decltype(res_t.max_)>(in1) - static_cast<decltype(res_t.max_)>(in2);
// }
//
///*!
// * @brief
// */
// template<typename SrcType>
// struct AbsType1 {
// #define limit    hvx::GetNumericLimits<SrcType>()
// #define max_val  hvx::Max(hvx::Abs(limit.min_), hvx::Abs(limit.max_))
// #define opr_bits hvx::GetNeededBits<IsSigned<SrcType>(), max_val>()
// #define dst_bits hvx::GetNeededBits<false, max_val>()
//    using OprType           = HIFLIP_NEEDED_DATA_TYPE(SrcType, SrcType, opr_bits);
//    using DstType           = HIFLIP_NEEDED_UNSIGNED_TYPE(dst_bits);
//    const int64_t dst_min_ = 0;        // minimum possible value
//    const int64_t dst_max_ = max_val;  // maximum possible value
//    const uint8_t opr_bit_ = opr_bits; // minimum number of bits required
//    const uint8_t dst_bit_ = dst_bits; // minimum number of bits required
//    const OprType opr_type_ = 0;
//    const DstType dst_type_ = 0;
// #undef limit
// #undef max_val
// #undef opr_bits
// #undef dst_bits
//};
//
///*!
// * @brief
// */
// template<>
// struct AbsType1<float> {
//    const float dst_min_  = 0;                                      // minimum possible value
//    const float dst_max_  = std::numeric_limits<float>::max(); // maximum possible value
//    const uint8_t opr_bit_    = 255;                                    // minimum number of bits required
//    const uint8_t dst_bit_    = 255;                                    // minimum number of bits required
//    const float opr_type_ = 0;
//    const float dst_type_ = 0;
//};
//
///*!
// * @brief
// */
// template<typename SrcType, int64_t src_min, int64_t src_max>
// struct AbsType2 {
// #define max_val  hvx::Max(hvx::Abs(src_min), hvx::Abs(src_max))
// #define opr_bits hvx::GetNeededBits<true, max_val>()
// #define dst_bits hvx::GetNeededBits<false, max_val>()
//    using OprType           = HIFLIP_NEEDED_SIGNED_TYPE(opr_bits);
//    using DstType           = HIFLIP_NEEDED_UNSIGNED_TYPE(dst_bits);
//    const int64_t dst_min_ = 0;                               // minimum possible value
//    const int64_t dst_max_ = max_val;                         // maximum possible value
//    const uint8_t opr_bit_ = static_cast<uint8_t>(opr_bits); // minimum number of bits required
//    const uint8_t dst_bit_ = static_cast<uint8_t>(dst_bits); // minimum number of bits required
//    const OprType opr_type_ = 0;
//    const DstType dst_type_ = 0;
// #undef max_val
// #undef opr_bits
// #undef dst_bits
//};
//
///*!
// * @brief
// */
// template<int64_t src_min, int64_t src_max>
// struct AbsType2<float, src_min, src_max> {
//    const float dst_min_  = 0;                                      // minimum possible value
//    const float dst_max_  = std::numeric_limits<float>::max(); // maximum possible value
//    const uint8_t opr_bit_    = 255;                                    // minimum number of bits required
//    const uint8_t dst_bit_    = 255;                                    // minimum number of bits required
//    const float opr_type_ = 0;
//    const float dst_type_ = 0;
//};
//
////
// template<typename SrcType, typename OprType, typename DstType>
// HVX_FORCE_INLINE constexpr auto
// AbsOpr(SrcType in1) noexcept -> DstType {
//     HVX_INLINE_TOP();
//     return static_cast<DstType>(hvx::Abs(static_cast<OprType>(in1)));
// }
//
// #undef HIFLIP_NEEDED_DATA_TYPE
// #undef HIFLIP_NEEDED_SIGNED_TYPE
// #undef HIFLIP_GET_TYPE
