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
 * @file    hvx_convert_concat.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_CONVERT_CONCAT_H
#define HVX_CONVERT_CONCAT_H

#include "../util/hvx_util_helper.h"

namespace hvx {
namespace convert {
/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the concat function
 */
template<typename type_    = hvx::util::dfixed<int16_t, 15>,
         typename dst_dim_ = hvx::util::TensorParam<1, hvx::util::VectorParam<2, 1>>,
         typename params_  = hvx::util::ConcatSplitParam<0, 1, 1>>
struct ConcatParam {
    // dimensions
    using dim    = dst_dim_;
    using split0 = hvx::util::ConcatSplitTensorParam<dim, params_::dim_id, params_::port0_elms>;
    using split1 = hvx::util::ConcatSplitTensorParam<dim, params_::dim_id, params_::port1_elms>;
    using split2 = hvx::util::ConcatSplitTensorParam<dim, params_::dim_id, params_::port2_elms>;
    using split3 = hvx::util::ConcatSplitTensorParam<dim, params_::dim_id, params_::port3_elms>;

    // types
    using type        = type_;
    using split0_vec  = hvx::util::vector<type, split0::dim::vec_size>;
    using split1_vec  = hvx::util::vector<type, split1::dim::vec_size>;
    using split2_vec  = hvx::util::vector<type, split2::dim::vec_size>;
    using split3_vec  = hvx::util::vector<type, split3::dim::vec_size>;
    using vec         = hvx::util::vector<type, dim::vec_size>;
    using split0_port = split0_vec;
    using split1_port = split1_vec;
    using split2_port = split2_vec;
    using split3_port = split3_vec;
    using port        = vec;

    // assertions
    constexpr ConcatParam() {
        hvx::util::TensorVerifyIfVecSizeIs1<dim, false, true, true, true, true, true>();
        static_assert(params_::dim_id < dim::dim_num, "This dimension cannot be concatenated!");
        static_assert((params_::port0_elms + params_::port1_elms + params_::port2_elms + params_::port3_elms + params_::port4_elms +
                       params_::port5_elms) == dim::dims[params_::dim_id],
                      "Number of elements of concatenated dimension does not fit!");
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Gets the dimension that is concatenated
 */
template<typename src_dim_, typename dst_dim_>
HVX_FORCE_INLINE constexpr auto
HwConcatGetId(int64_t id) -> int64_t {
    HVX_INLINE_TOP();
    return (id == (dst_dim_::dim_num - 1))
             ? (dst_dim_::dim_num - 1)
             : ((dst_dim_::dims[id] != src_dim_::dims[id]) ? (id) : (hvx::convert::HwConcatGetId<src_dim_, dst_dim_>(id + 1))); // NOLINT
}

/*!
 * @brief Reads from correct input
 */
template<int64_t src_num_, int64_t id_, int64_t src_dim_beg_, typename param_>
HVX_FORCE_INLINE constexpr auto
HwConcatSrc(int64_t& ptr_dim,
            hvx::util::vector<int64_t, src_num_> ptr_src,
            hvx::util::vector<typename param_::type, param_::dim::vec_size>& src_data) noexcept -> void {
    HVX_INLINE_TOP();
    (void)ptr_dim;
    (void)ptr_src;
    (void)src_data;
}

/*!
 * @brief Reads from correct input
 */
template<int64_t src_num_, int64_t id_, int64_t src_dim_beg_, typename param_, typename src_dim_, typename... src_dim_rest_>
HVX_FORCE_INLINE constexpr auto
HwConcatSrc(int64_t& ptr_dim,
            hvx::util::vector<int64_t, src_num_>& ptr_src,
            hvx::util::vector<typename param_::type, src_dim_::vec_size>& src_data,
            hvx::util::vector<typename param_::type, src_dim_::vec_size>* src,
            hvx::util::vector<typename param_::type, src_dim_rest_::vec_size>*... src_rest) noexcept -> void {
    HVX_INLINE_TOP();

    // gets a pointer that points to the end of the region of a given input
    constexpr int64_t src_dim_end = src_dim_beg_ + ((id_ == 0) ? (src_dim_::dims[id_] / src_dim_::vec_size) : (src_dim_::dims[id_]));

    // select the correct input to read from it
    if (ptr_dim < src_dim_end)
        hvx::util::StreamReadData<hvx::util::vector<typename param_::type, src_dim_::vec_size>>(
            src, src_data, ptr_src.Get(sizeof...(src_dim_rest_)), true);
    else
        hvx::convert::HwConcatSrc<src_num_, id_, src_dim_end, param_, src_dim_rest_...>(ptr_dim, ptr_src, src_data, src_rest...);
}

/*!
 * @brief Gets dimension ID, reads from correct input and writes to output
 */
template<int64_t src_num_, typename param_, typename src_dim_, typename... src_dim_rest_>
HVX_FORCE_INLINE constexpr auto
HwConcatBase(hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax>& ptr_dim,
             hvx::util::vector<int64_t, src_num_>& ptr_src,
             int64_t& ptr_dst,
             typename param_::vec* dst,
             hvx::util::vector<typename param_::type, src_dim_::vec_size>* src,
             hvx::util::vector<typename param_::type, src_dim_rest_::vec_size>*... src_rest) noexcept -> void {
    HVX_INLINE_TOP();

    // gets the dimension that is concatenated
    constexpr int64_t id = hvx::convert::HwConcatGetId<src_dim_, typename param_::dim>(0);

    // stores the input data
    hvx::util::vector<typename param_::type, src_dim_::vec_size> dst_data{};

    // read from correct input
    hvx::convert::HwConcatSrc<src_num_, id, 0, param_, src_dim_, src_dim_rest_...>(ptr_dim.Get(id), ptr_src, dst_data, src, src_rest...);

    // write to output
    hvx::util::StreamWriteData<typename param_::vec>(dst, dst_data, ptr_dst, true);
}

/*!
 * @brief Top HW concat function
 */
template<typename param_, typename src_dim_, typename... src_dim_rest_>
HVX_FORCE_INLINE constexpr auto
HwConcatTop(typename param_::vec* dst,
            hvx::util::vector<typename param_::type, src_dim_::vec_size>* src,
            hvx::util::vector<typename param_::type, src_dim_rest_::vec_size>*... src_rest) noexcept -> void {
    HVX_INLINE_TOP();

    // number of inputs and dimensions
    constexpr int64_t src_num = sizeof...(src_dim_rest_) + 1;

    // pointer to inputs/outputs
    hvx::util::vector<int64_t, src_num> ptr_src{};
    hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax> ptr_dim{};
    HVX_ARRAY_PARTITION_COMPLETE(ptr_src.data, 0);
    HVX_ARRAY_PARTITION_COMPLETE(ptr_dim.data, 0);

    // iterates over all dimensions of the tensor
    for (int64_t i = 0, ptr_dst = 0; i < param_::dim::vec_elms; ++i) {
        HVX_PIPELINE_ON(1, frp);

        // flattening loop to improve latency
        hvx::util::TensorUpdateDimPtr<typename param_::dim>(ptr_dim, i);

        // gets dimension ID, reads from correct input and writes to output
        hvx::convert::HwConcatBase<src_num, param_, src_dim_, src_dim_rest_...>(ptr_dim, ptr_src, ptr_dst, dst, src, src_rest...);
    }
}

/******************************************************************************************************************************************/
} // namespace convert
} // namespace hvx

#endif // HVX_CONVERT_CONCAT_H
