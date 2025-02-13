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
 * @file    hvx_config.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_CONFIG_H_
#define HVX_CONFIG_H_

// STL
#include "../dynfloat/dynfloat.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// activate XILINX HLS
#if defined(__XILINX__) || defined(__VITIS_HLS__)
#define HVX_SYNTHESIS_ACTIVE
// #define DYNFLOAT_XILINX_SYNTHESIS
#endif
#define HIFLIPVX_DYNFLOAT_ACTIVE

// XILINX HLS libraries
#ifdef HVX_SYNTHESIS_ACTIVE
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
// Enablement for axis signals
#ifndef AXIS_ENABLE_DATA
#define AXIS_ENABLE_DATA 0b00000001
#endif
#ifndef AXIS_ENABLE_DEST
#define AXIS_ENABLE_DEST 0b00000010
#endif
#ifndef AXIS_ENABLE_ID
#define AXIS_ENABLE_ID 0b00000100
#endif
#ifndef AXIS_ENABLE_KEEP
#define AXIS_ENABLE_KEEP 0b00001000
#endif
#ifndef AXIS_ENABLE_LAST
#define AXIS_ENABLE_LAST 0b00010000
#endif
#ifndef AXIS_ENABLE_STRB
#define AXIS_ENABLE_STRB 0b00100000
#endif
#ifndef AXIS_ENABLE_USER
#define AXIS_ENABLE_USER 0b01000000
#endif
#endif

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127) // warning C4127: conditional expression is constant
#endif

#endif // HVX_CONFIG_H_
