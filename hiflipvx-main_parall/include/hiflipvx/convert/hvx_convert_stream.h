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
 * @file    hvx_convert_stream.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_CONVERT_STREAM_H
#define HVX_CONVERT_STREAM_H

#include "../util/hvx_util_helper.h"

namespace hvx {
namespace convert {
/******************************************************************************************************************************************/
#if defined(HVX_SYNTHESIS_ACTIVE)

/*!
 * @brief Contains the data and side channels of an AXI4-stream signal
 */
template<typename type_, int64_t flag_>
using hls_stream_data = hls::axis<type_, ((flag_ & AXIS_ENABLE_USER) != 0), 0, 0, flag_ | AXIS_ENABLE_DATA, true>;

/*!
 * @brief Wraps the AXI4-stream signal into HLS stream class for extra operations (read/write)
 */
template<typename type_, int64_t flag_>
using hls_stream_port = hls::stream<hvx::convert::hls_stream_data<type_, flag_>>;

#endif
/******************************************************************************************************************************************/

/*!
 * @brief This class is needed for the HVX <-> Stream conversion functions (for side channels)
 */
template<typename type_, typename dim_, int64_t flags_>
struct StreamParam {
    using dim  = dim_;
    using type = type_;
    using vec  = hvx::util::vector<type_, dim_::vec_size>;
#if defined(HVX_SYNTHESIS_ACTIVE)
    using port = hvx::convert::hls_stream_port<vec, flags_>;
#else
    using port = vec;
#endif
    static constexpr auto flags = flags_;
};

/******************************************************************************************************************************************/
#if defined(HVX_SYNTHESIS_ACTIVE)

/*!
 * @brief Sets End-of-Frame flag (for side channels)
 */
template<typename type_, typename dim_, int64_t flag_, std::enable_if_t<(flag_ & AXIS_ENABLE_LAST) != 0, bool> = true>
HVX_FORCE_INLINE constexpr auto
StreamSetEof(hvx::convert::hls_stream_data<type_, flag_>& dst_data, int64_t ptr) noexcept -> void {
    HVX_INLINE_TOP();
    dst_data.last = (ptr == (dim_::vec_elms - 1));
}

/*!
 * @brief
 */
template<typename type_, typename dim_, int64_t flag_, std::enable_if_t<(flag_ & AXIS_ENABLE_LAST) == 0, bool> = true>
HVX_FORCE_INLINE constexpr auto
StreamSetEof(hvx::convert::hls_stream_data<type_, flag_>& dst_data, int64_t ptr) noexcept -> void {
    HVX_INLINE_TOP();
    (void)dst_data;
    (void)ptr;
}

/*!
 * @brief Sets Start-of-Frame flag (for side channels)
 */
template<typename type_, int64_t flag_, std::enable_if_t<(flag_ & AXIS_ENABLE_USER) != 0, bool> = true>
HVX_FORCE_INLINE constexpr auto
StreamSetSof(hvx::convert::hls_stream_data<type_, flag_>& dst_data, int64_t ptr) noexcept -> void {
    HVX_INLINE_TOP();
    dst_data.user = (ptr == 0);
}

/*!
 * @brief
 */
template<typename type_, int64_t flag_, std::enable_if_t<(flag_ & AXIS_ENABLE_USER) == 0, bool> = true>
HVX_FORCE_INLINE constexpr auto
StreamSetSof(hvx::convert::hls_stream_data<type_, flag_>& dst_data, int64_t ptr) noexcept -> void {
    HVX_INLINE_TOP();
    (void)dst_data;
    (void)ptr;
}

#endif
/******************************************************************************************************************************************/
#if defined(HVX_SYNTHESIS_ACTIVE)
/*!
 * @brief Converts an HLS stream to an HVX stream to ensure correct AXI4-stream behavior (ignores side channel input)
 */
template<typename type_, typename dim_, int64_t flag_>
HVX_FORCE_INLINE auto
StreamToHvx(hvx::convert::hls_stream_port<type_, flag_>& src, type_* dst) noexcept -> void {
    HVX_INLINE_TOP();

    // iterates through the tensor vector by vector
    for (int64_t i = 0; i < dim_::vec_elms; ++i) {
        HVX_PIPELINE_ON(1, frp);

        // read data
        auto src_data = src.read();

        // transform data
        type_ dst_data{};
        for (int64_t j = 0; j < dim_::vec_size; ++j)
            dst_data.Set(src_data.data.Get(j), j);

        // write data
        dst[i] = dst_data;
    }
}
#else
/*!
 * @brief Converts an HLS stream to an HVX stream to ensure correct AXI4-stream behavior (ignores side channel input)
 */
template<typename type_, typename dim_, int64_t flag_>
HVX_FORCE_INLINE auto
StreamToHvx(type_& src, type_* dst) noexcept -> void {
    HVX_INLINE_TOP();
    for (int64_t i = 0; i < dim_::vec_elms; ++i) {
        HVX_PIPELINE_ON(1, frp);
        dst[i] = (&src)[i]; // NOLINT
    }
}
#endif

#if defined(HVX_SYNTHESIS_ACTIVE)
/*!
 * @brief Converts an HVX stream to an HLS stream to add side channels and ensure correct AXI4-stream behavior
 */
template<typename type_, typename dim_, int64_t flag_>
HVX_FORCE_INLINE auto
HvxToStream(type_* src, hvx::convert::hls_stream_port<type_, flag_>& dst) noexcept -> void {
    HVX_INLINE_TOP();

    // iterates through the tensor vector by vector
    for (int64_t i = 0; i < dim_::vec_elms; ++i) {
        HVX_PIPELINE_ON(1, frp);

        // read data
        auto src_data = src[i];

        // transform data
        hvx::convert::hls_stream_data<type_, flag_> dst_data{};
        for (int64_t j = 0; j < dim_::vec_size; ++j)
            dst_data.data.Set(src_data.Get(j), j);

        // create side channels
        hvx::convert::StreamSetSof<type_, flag_>(dst_data, i);
        hvx::convert::StreamSetEof<type_, dim_, flag_>(dst_data, i);

        // write data
        dst.write(dst_data);
    }
}
#else
/*!
 * @brief Converts an HVX stream to an HLS stream to add side channels and ensure correct AXI4-stream behavior
 */
template<typename type_, typename dim_, int64_t flag_>
HVX_FORCE_INLINE auto
HvxToStream(type_* src, type_& dst) noexcept -> void {
    HVX_INLINE_TOP();
    for (int64_t i = 0; i < dim_::vec_elms; ++i) {
        HVX_PIPELINE_ON(1, frp);
        (&dst)[i] = src[i]; // NOLINT
    }
}
#endif

/******************************************************************************************************************************************/

/*!
 * @brief creates a fifo with "fifo_size" instead of an array with "original_buf_size_" to stream data
 */
template<typename type_, int64_t original_buf_size_, int64_t fifo_size>
HVX_FORCE_INLINE auto
StreamFifo() noexcept -> hvx::util::array1d<type_, original_buf_size_> {
    HVX_INLINE_TOP();
    hvx::util::array1d<type_, original_buf_size_> dst_fifo;
    HVX_BUFFER(dst_fifo.data, fifo, fifo_size);
    return dst_fifo;
}

/******************************************************************************************************************************************/

/*!
 * @brief resends a data beat of size "vec_elms_" after latency + delay
 */
template<typename type_, int64_t latency_, int64_t delay_, int64_t vec_elms_>
HVX_FORCE_INLINE auto
SrcGenerator(type_* src, type_* dst) noexcept -> void {
    HVX_INLINE_TOP();

    // the pipeline loops with (latency + delay) clock cycles
    for (int64_t v = 0; v < latency_ + delay_; ++v) {
        HVX_PIPELINE_ON(1, frp);

        // check if all elements have been sent
        if (v < vec_elms_)
            dst[v] = src[v]; // NOLINT
    }
}

/*!
 * @brief resends a data beat of size "vec_elms_" after latency + delay
 */
template<typename dst_vec_, int64_t latency_, int64_t delay_, int64_t vec_elms_>
HVX_FORCE_INLINE auto
SrcGenerator(dst_vec_* dst) noexcept -> void {
    HVX_INLINE_TOP();

    // the pipeline loops with (latency + delay) clock cycles
    for (int64_t ptr_dst = 0; ptr_dst < latency_ + delay_;) {
        HVX_PIPELINE_ON(1, frp);

        // check if all elements have been sent
        if (ptr_dst < vec_elms_) {
            dst_vec_ data{};

            // create random data (normalize data if floating point)
            if (dst_vec_::is_flt)
                data.data = static_cast<float>(ptr_dst) / vec_elms_;
            else
                data.data = static_cast<typename dst_vec_::data_type>(vec_elms_);

            // write to output
            hvx::util::StreamWriteData<dst_vec_>(dst, data, ptr_dst, true);
        }
    }
}

/******************************************************************************************************************************************/
} // namespace convert
} // namespace hvx

#endif // HVX_CONVERT_STREAM_H
