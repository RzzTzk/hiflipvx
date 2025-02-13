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
 * @file    hvx_util_macro.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_MACRO_H_
#define HVX_UTIL_MACRO_H_

#include "../hvx_config.h"
/******************************************************************************************************************************************/
// NOLINTBEGIN (TODO: consider using template functions inseatd of macros if possible)

// an alternative to the XILINX inline directive
#if defined(__GNUC__) || defined(__clang__)
#define HVX_FORCE_INLINE __attribute((always_inline)) inline
#elif defined(_MSC_VER)
#define HVX_FORCE_INLINE __forceinline
#else
#define HVX_FORCE_INLINE inline
#endif

//
#if defined(HVX_SYNTHESIS_ACTIVE)
#define HVX_PRAGMA(STR) _Pragma(#STR)
#else
#define HVX_PRAGMA(STR)
#endif

#define HVX_GET_MACRO17(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, NAME, ...) NAME
#define HVX_GET_MACRO8(_1, _2, _3, _4, _5, _6, _7, _8, NAME, ...)                                              NAME

//
#if defined(__VITIS_HLS__)
#if defined(__URAM__)
#define HVX_RESOURCE_MEM(VAR, ELEMENT_BYTE_SIZE, ELEMENT_NUM)                    \
    if ((ELEMENT_BYTE_SIZE * ELEMENT_NUM) >= 8192 && ELEMENT_NUM >= 2048) {      \
        HVX_PRAGMA(HLS bind_storage variable = VAR type = RAM_S2P impl = URAM)   \
    } else if ((ELEMENT_BYTE_SIZE * ELEMENT_NUM) >= 128 && ELEMENT_NUM >= 32) {  \
        HVX_PRAGMA(HLS bind_storage variable = VAR type = RAM_S2P impl = BRAM)   \
    } else if (ELEMENT_NUM >= 4) {                                               \
        HVX_PRAGMA(HLS bind_storage variable = VAR type = RAM_S2P impl = LUTRAM) \
    } else {                                                                     \
        HVX_PRAGMA(HLS array_partition variable = VAR type = complete dim = 0)   \
    }
#else
#define HVX_RESOURCE_MEM(VAR, ELEMENT_BYTE_SIZE, ELEMENT_NUM)                    \
    if ((ELEMENT_BYTE_SIZE * ELEMENT_NUM) >= 128 && ELEMENT_NUM >= 32) {         \
        HVX_PRAGMA(HLS bind_storage variable = VAR type = RAM_S2P impl = BRAM)   \
    } else if (ELEMENT_NUM >= 4) {                                               \
        HVX_PRAGMA(HLS bind_storage variable = VAR type = RAM_S2P impl = LUTRAM) \
    } else {                                                                     \
        HVX_PRAGMA(HLS array_partition variable = VAR type = complete dim = 0)   \
    }
#endif
#else
#if defined(__URAM__)
#define HVX_RESOURCE_MEM(VAR, ELEMENT_BYTE_SIZE, ELEMENT_NUM)                   \
    if ((ELEMENT_BYTE_SIZE * ELEMENT_NUM) >= 8192 && ELEMENT_NUM >= 2048) {     \
        HVX_PRAGMA(HLS RESOURCE variable = VAR core = XPM_MEMORY uram)          \
    } else if ((ELEMENT_BYTE_SIZE * ELEMENT_NUM) >= 128 && ELEMENT_NUM >= 32) { \
        HVX_PRAGMA(HLS RESOURCE variable = VAR core = RAM_2P_BRAM)              \
    } else if (ELEMENT_NUM >= 4) {                                              \
        HVX_PRAGMA(HLS RESOURCE variable = VAR core = RAM_2P_LUTRAM)            \
    } else {                                                                    \
        HVX_PRAGMA(HLS array_partition variable = VAR type = complete dim = 0)  \
    }
#else
#define HVX_RESOURCE_MEM(VAR, ELEMENT_BYTE_SIZE, ELEMENT_NUM)                  \
    if ((ELEMENT_BYTE_SIZE * ELEMENT_NUM) >= 128 && ELEMENT_NUM >= 32) {       \
        HVX_PRAGMA(HLS RESOURCE variable = VAR core = RAM_2P_BRAM)             \
    } else if (ELEMENT_NUM >= 4) {                                             \
        HVX_PRAGMA(HLS RESOURCE variable = VAR core = RAM_2P_LUTRAM)           \
    } else {                                                                   \
        HVX_PRAGMA(HLS array_partition variable = VAR type = complete dim = 0) \
    }
#endif
#endif

// TODO: bit mode should not be used on top function
#if defined(__VITIS_HLS__)
#define HVX_DATAPACK1(var1) HVX_PRAGMA(HLS AGGREGATE variable = var1 compact = bit)
#define HVX_DATAPACK2(var1, var2) \
    HVX_DATAPACK1(var1)           \
    HVX_PRAGMA(HLS AGGREGATE variable = var2 compact = bit)
#define HVX_DATAPACK3(var1, var2, var3) \
    HVX_DATAPACK2(var1, var2)           \
    HVX_PRAGMA(HLS AGGREGATE variable = var3 compact = bit)
#define HVX_DATAPACK4(var1, var2, var3, var4) \
    HVX_DATAPACK3(var1, var2, var3)           \
    HVX_PRAGMA(HLS AGGREGATE variable = var4 compact = bit)
#define HVX_DATAPACK5(var1, var2, var3, var4, var5) \
    HVX_DATAPACK4(var1, var2, var3, var4)           \
    HVX_PRAGMA(HLS AGGREGATE variable = var5 compact = bit)
#define HVX_DATAPACK6(var1, var2, var3, var4, var5, var6) \
    HVX_DATAPACK5(var1, var2, var3, var4, var5)           \
    HVX_PRAGMA(HLS AGGREGATE variable = var6 compact = bit)
#define HVX_DATAPACK7(var1, var2, var3, var4, var5, var6, var7) \
    HVX_DATAPACK6(var1, var2, var3, var4, var5, var6)           \
    HVX_PRAGMA(HLS AGGREGATE variable = var7 compact = bit)
#define HVX_DATAPACK8(var1, var2, var3, var4, var5, var6, var7, var8) \
    HVX_DATAPACK7(var1, var2, var3, var4, var5, var6, var7)           \
    HVX_PRAGMA(HLS AGGREGATE variable = var8 compact = bit)
#define HVX_DATAPACK9(var1, var2, var3, var4, var5, var6, var7, var8, var9) \
    HVX_DATAPACK8(var1, var2, var3, var4, var5, var6, var7, var8)           \
    HVX_PRAGMA(HLS AGGREGATE variable = var9 compact = bit)
#define HVX_DATAPACK10(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10) \
    HVX_DATAPACK9(var1, var2, var3, var4, var5, var6, var7, var8, var9)             \
    HVX_PRAGMA(HLS AGGREGATE variable = var10 compact = bit)
#define HVX_DATAPACK11(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11) \
    HVX_DATAPACK10(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10)            \
    HVX_PRAGMA(HLS AGGREGATE variable = var11 compact = bit)
#define HVX_DATAPACK12(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12) \
    HVX_DATAPACK11(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11)            \
    HVX_PRAGMA(HLS AGGREGATE variable = var12 compact = bit)
#define HVX_DATAPACK13(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13) \
    HVX_DATAPACK12(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12)            \
    HVX_PRAGMA(HLS AGGREGATE variable = var13 compact = bit)
#define HVX_DATAPACK14(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14) \
    HVX_DATAPACK13(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13)            \
    HVX_PRAGMA(HLS AGGREGATE variable = var14 compact = bit)
#define HVX_DATAPACK15(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15) \
    HVX_DATAPACK14(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14)            \
    HVX_PRAGMA(HLS AGGREGATE variable = var15 compact = bit)
#define HVX_DATAPACK16(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15, var16) \
    HVX_DATAPACK15(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15)            \
    HVX_PRAGMA(HLS AGGREGATE variable = var16 compact = bit)
#define HVX_DATAPACK17(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17) \
    HVX_DATAPACK16(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15, var16)            \
    HVX_PRAGMA(HLS AGGREGATE variable = var17 compact = bit)
#else
#define HVX_DATAPACK1(var1) HVX_PRAGMA(HLS DATA_PACK variable = var1)
#define HVX_DATAPACK2(var1, var2) \
    HVX_DATAPACK1(var1)           \
    HVX_PRAGMA(HLS DATA_PACK variable = var2)
#define HVX_DATAPACK3(var1, var2, var3) \
    HVX_DATAPACK2(var1, var2)           \
    HVX_PRAGMA(HLS DATA_PACK variable = var3)
#define HVX_DATAPACK4(var1, var2, var3, var4) \
    HVX_DATAPACK3(var1, var2, var3)           \
    HVX_PRAGMA(HLS DATA_PACK variable = var4)
#define HVX_DATAPACK5(var1, var2, var3, var4, var5) \
    HVX_DATAPACK4(var1, var2, var3, var4)           \
    HVX_PRAGMA(HLS DATA_PACK variable = var5)
#define HVX_DATAPACK6(var1, var2, var3, var4, var5, var6) \
    HVX_DATAPACK5(var1, var2, var3, var4, var5)           \
    HVX_PRAGMA(HLS DATA_PACK variable = var6)
#define HVX_DATAPACK7(var1, var2, var3, var4, var5, var6, var7) \
    HVX_DATAPACK6(var1, var2, var3, var4, var5, var6)           \
    HVX_PRAGMA(HLS DATA_PACK variable = var7)
#define HVX_DATAPACK8(var1, var2, var3, var4, var5, var6, var7, var8) \
    HVX_DATAPACK7(var1, var2, var3, var4, var5, var6, var7)           \
    HVX_PRAGMA(HLS DATA_PACK variable = var8)
#define HVX_DATAPACK9(var1, var2, var3, var4, var5, var6, var7, var8, var9) \
    HVX_DATAPACK8(var1, var2, var3, var4, var5, var6, var7, var8)           \
    HVX_PRAGMA(HLS DATA_PACK variable = var9)
#define HVX_DATAPACK10(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10) \
    HVX_DATAPACK9(var1, var2, var3, var4, var5, var6, var7, var8, var9)             \
    HVX_PRAGMA(HLS DATA_PACK variable = var10)
#define HVX_DATAPACK11(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11) \
    HVX_DATAPACK10(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10)            \
    HVX_PRAGMA(HLS DATA_PACK variable = var11)
#define HVX_DATAPACK12(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12) \
    HVX_DATAPACK11(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11)            \
    HVX_PRAGMA(HLS DATA_PACK variable = var12)
#define HVX_DATAPACK13(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13) \
    HVX_DATAPACK12(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12)            \
    HVX_PRAGMA(HLS DATA_PACK variable = var13)
#define HVX_DATAPACK14(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14) \
    HVX_DATAPACK13(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13)            \
    HVX_PRAGMA(HLS DATA_PACK variable = var14)
#define HVX_DATAPACK15(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15) \
    HVX_DATAPACK14(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14)            \
    HVX_PRAGMA(HLS DATA_PACK variable = var15)
#define HVX_DATAPACK16(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15, var16) \
    HVX_DATAPACK15(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15)            \
    HVX_PRAGMA(HLS DATA_PACK variable = var16)
#define HVX_DATAPACK17(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17) \
    HVX_DATAPACK16(var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15, var16)            \
    HVX_PRAGMA(HLS DATA_PACK variable = var17)
#endif

//
#define HVX_DATAPACK(...)                                                                                                        \
    HVX_GET_MACRO17(__VA_ARGS__, HVX_DATAPACK17, HVX_DATAPACK16, HVX_DATAPACK15, HVX_DATAPACK14, HVX_DATAPACK13, HVX_DATAPACK12, \
                    HVX_DATAPACK11, HVX_DATAPACK10, HVX_DATAPACK9, HVX_DATAPACK8, HVX_DATAPACK7, HVX_DATAPACK6, HVX_DATAPACK5,   \
                    HVX_DATAPACK4, HVX_DATAPACK3, HVX_DATAPACK2, HVX_DATAPACK1)                                                  \
    (__VA_ARGS__)
#define HVX_DATAPACK_TOP(...)                                                                                                    \
    HVX_PRAGMA(HLS INLINE)                                                                                                       \
    HVX_GET_MACRO17(__VA_ARGS__, HVX_DATAPACK17, HVX_DATAPACK16, HVX_DATAPACK15, HVX_DATAPACK14, HVX_DATAPACK13, HVX_DATAPACK12, \
                    HVX_DATAPACK11, HVX_DATAPACK10, HVX_DATAPACK9, HVX_DATAPACK8, HVX_DATAPACK7, HVX_DATAPACK6, HVX_DATAPACK5,   \
                    HVX_DATAPACK4, HVX_DATAPACK3, HVX_DATAPACK2, HVX_DATAPACK1)                                                  \
    (__VA_ARGS__)

//
#define HVX_INTERFACE_STREAM1(var1) HVX_PRAGMA(HLS INTERFACE mode = axis port = var1)
#define HVX_INTERFACE_STREAM2(var1, var2) \
    HVX_INTERFACE_STREAM1(var1)           \
    HVX_PRAGMA(HLS INTERFACE mode = axis port = var2)
#define HVX_INTERFACE_STREAM3(var1, var2, var3) \
    HVX_INTERFACE_STREAM2(var1, var2)           \
    HVX_PRAGMA(HLS INTERFACE mode = axis port = var3)
#define HVX_INTERFACE_STREAM4(var1, var2, var3, var4) \
    HVX_INTERFACE_STREAM3(var1, var2, var3)           \
    HVX_PRAGMA(HLS INTERFACE mode = axis port = var4)
#define HVX_INTERFACE_STREAM5(var1, var2, var3, var4, var5) \
    HVX_INTERFACE_STREAM4(var1, var2, var3, var4)           \
    HVX_PRAGMA(HLS INTERFACE mode = axis port = var5)
#define HVX_INTERFACE_STREAM6(var1, var2, var3, var4, var5, var6) \
    HVX_INTERFACE_STREAM5(var1, var2, var3, var4, var5)           \
    HVX_PRAGMA(HLS INTERFACE mode = axis port = var6)
#define HVX_INTERFACE_STREAM7(var1, var2, var3, var4, var5, var6, var7) \
    HVX_INTERFACE_STREAM6(var1, var2, var3, var4, var5, var6)           \
    HVX_PRAGMA(HLS INTERFACE mode = axis port = var6)
#define HVX_INTERFACE_STREAM8(var1, var2, var3, var4, var5, var6, var7, var8) \
    HVX_INTERFACE_STREAM7(var1, var2, var3, var4, var5, var6, var7)           \
    HVX_PRAGMA(HLS INTERFACE mode = axis port = var6)

//
#define HVX_INTERFACE_STREAM(...)                                                                                           \
    HVX_GET_MACRO8(__VA_ARGS__, HVX_INTERFACE_STREAM8, HVX_INTERFACE_STREAM7, HVX_INTERFACE_STREAM6, HVX_INTERFACE_STREAM5, \
                   HVX_INTERFACE_STREAM4, HVX_INTERFACE_STREAM3, HVX_INTERFACE_STREAM2, HVX_INTERFACE_STREAM1)              \
    (__VA_ARGS__)
#define HVX_INTERFACE_STREAM_TLP(...)                                                                                       \
    HVX_PRAGMA(HLS DATAFLOW)                                                                                                \
    HVX_GET_MACRO8(__VA_ARGS__, HVX_INTERFACE_STREAM8, HVX_INTERFACE_STREAM7, HVX_INTERFACE_STREAM6, HVX_INTERFACE_STREAM5, \
                   HVX_INTERFACE_STREAM4, HVX_INTERFACE_STREAM3, HVX_INTERFACE_STREAM2, HVX_INTERFACE_STREAM1)              \
    (__VA_ARGS__)

#define HVX_INTERFACE_STREAM_NO_CTRL(...)                                                                                   \
    HVX_PRAGMA(HLS INTERFACE ap_ctrl_none port = return)                                                                    \
    HVX_GET_MACRO8(__VA_ARGS__, HVX_INTERFACE_STREAM8, HVX_INTERFACE_STREAM7, HVX_INTERFACE_STREAM6, HVX_INTERFACE_STREAM5, \
                   HVX_INTERFACE_STREAM4, HVX_INTERFACE_STREAM3, HVX_INTERFACE_STREAM2, HVX_INTERFACE_STREAM1)              \
    (__VA_ARGS__)
#define HVX_INTERFACE_STREAM_NO_CTRL_TLP(...)                                                                               \
    HVX_PRAGMA(HLS INTERFACE ap_ctrl_none port = return)                                                                    \
    HVX_PRAGMA(HLS DATAFLOW)                                                                                                \
    HVX_GET_MACRO8(__VA_ARGS__, HVX_INTERFACE_STREAM8, HVX_INTERFACE_STREAM7, HVX_INTERFACE_STREAM6, HVX_INTERFACE_STREAM5, \
                   HVX_INTERFACE_STREAM4, HVX_INTERFACE_STREAM3, HVX_INTERFACE_STREAM2, HVX_INTERFACE_STREAM1)              \
    (__VA_ARGS__)

// INLINE
#define HVX_INLINE_TOP()    HVX_PRAGMA(HLS INLINE)
#define HVX_INLINE_BOTTOM() HVX_PRAGMA(HLS INLINE recursive)

// PIPELINE
#define HVX_PIPELINE_ON(INTERVAL, STYLE)  HVX_PRAGMA(HLS pipeline II = INTERVAL style = STYLE)
#define HVX_PIPELINE_OFF(INTERVAL, STYLE) HVX_PRAGMA(HLS pipeline off)

// UNROLL
#define HVX_UNROLL() HVX_PRAGMA(HLS UNROLL)

// PARTITION ARRAY
#define HVX_ARRAY_PARTITION_COMPLETE(NAME, DIM)      HVX_PRAGMA(HLS ARRAY_PARTITION variable = NAME type = complete dim = DIM)
#define HVX_ARRAY_PARTITION_BLOCK(NAME, DIM, FACTOR) HVX_PRAGMA(HLS ARRAY_PARTITION variable = NAME type = block factor = FACTOR dim = DIM)

// FALSE DEPENDENCE
#define HVX_FALSE_DEPENDENCE(VAR) HVX_PRAGMA(HLS DEPENDENCE variable = VAR inter false)

// Thread/Task Level Parallelism
#define HVX_TLP() HVX_PRAGMA(HLS DATAFLOW)

//  STREAM
#define HVX_BUFFER(NAME, TYPE, DEPTH) HVX_PRAGMA(HLS stream variable = NAME type = TYPE depth = DEPTH)

// To create a pipelined execute unit
#define HVX_EXEC_UNIT_2SRC()                                  \
    HVX_PRAGMA(HLS INTERFACE ap_ctrl_none port = return) \
    HVX_PRAGMA(HLS INTERFACE mode = ap_none port = src1) \
    HVX_PRAGMA(HLS INTERFACE mode = ap_none port = src2) \
    HVX_PRAGMA(HLS INTERFACE mode = ap_none port = dst)  \
    HVX_PRAGMA(HLS pipeline II = 1)

#define HVX_EXEC_UNIT_1SRC()                                  \
    HVX_PRAGMA(HLS INTERFACE ap_ctrl_none port = return) \
    HVX_PRAGMA(HLS INTERFACE mode = ap_none port = src1) \
    HVX_PRAGMA(HLS INTERFACE mode = ap_none port = dst)  \
    HVX_PRAGMA(HLS pipeline II = 1)

// NOLINTEND
/******************************************************************************************************************************************/
#endif // HVX_UTIL_MACRO_H_
