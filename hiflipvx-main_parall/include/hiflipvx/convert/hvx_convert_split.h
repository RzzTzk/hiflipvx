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
 * @file    hvx_convert_split.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_CONVERT_SPLIT_H
#define HVX_CONVERT_SPLIT_H

#include "../util/hvx_util_helper.h"

namespace hvx {
namespace convert {
/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the split function
 */
template<typename type_    = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_ = hvx::util::TensorParam<1, hvx::util::VectorParam<2, 1>>,
         typename params_  = hvx::util::ConcatSplitParam<0, 1, 1>>
struct SplitParam {
    // dimensions
    using dim    = src_dim_;
    using split0 = hvx::util::ConcatSplitTensorParam<dim, params_::dim_id, params_::port0_elms>;
    using split1 = hvx::util::ConcatSplitTensorParam<dim, params_::dim_id, params_::port1_elms>;
    using split2 = hvx::util::ConcatSplitTensorParam<dim, params_::dim_id, params_::port2_elms>;
    using split3 = hvx::util::ConcatSplitTensorParam<dim, params_::dim_id, params_::port3_elms>;

    // types
    using type        = type_;
    using vec         = hvx::util::vector<type, dim::vec_size>;
    using split0_vec  = hvx::util::vector<type, split0::dim::vec_size>;
    using split1_vec  = hvx::util::vector<type, split1::dim::vec_size>;
    using split2_vec  = hvx::util::vector<type, split2::dim::vec_size>;
    using split3_vec  = hvx::util::vector<type, split3::dim::vec_size>;
    using port        = vec;
    using split0_port = split0_vec;
    using split1_port = split1_vec;
    using split2_port = split2_vec;
    using split3_port = split3_vec;

    // assertions
    constexpr SplitParam() {
        hvx::util::TensorVerifyIfVecSizeIs1<dim, false, true, true, true, true, true>();
        static_assert(params_::dim_id < dim::dim_num, "This dimension cannot be concatenated!");
        static_assert((params_::port0_elms + params_::port1_elms + params_::port2_elms + params_::port3_elms + params_::port4_elms +
                       params_::port5_elms) == dim::dims[params_::dim_id],
                      "Number of elements of concatenated dimension does not fit!");
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Gets the dimension that is split
 */
template<typename src_dim_, typename dst_dim_>
HVX_FORCE_INLINE constexpr auto
HwSplitGetId(int64_t id) -> int64_t {
    HVX_INLINE_TOP();
    return (id == (src_dim_::dim_num - 1))
             ? (src_dim_::dim_num - 1)
             : ((src_dim_::dims[id] != dst_dim_::dims[id]) ? (id) : (hvx::convert::HwSplitGetId<dst_dim_, src_dim_>(id + 1))); // NOLINT
}

/*!
 * @brief Writes to correct output
 */
template<int64_t dst_num, int64_t id_, int64_t dst_dim_beg_, typename param_>
HVX_FORCE_INLINE constexpr auto
HwSplitOutput(int64_t& ptr_dim, hvx::util::vector<int64_t, dst_num>& ptr_dst, typename param_::vec& dst_data) noexcept -> void {
    HVX_INLINE_TOP();
    (void)ptr_dim;
    (void)ptr_dst;
    (void)dst_data;
}

/*!
 * @brief Writes to correct output
 */
template<int64_t dst_num, int64_t id_, int64_t dst_dim_beg_, typename param_, typename dst_dim_, typename... dst_dim_rest_>
HVX_FORCE_INLINE constexpr auto
HwSplitOutput(int64_t& ptr_dim,
              hvx::util::vector<int64_t, dst_num>& ptr_dst,
              typename param_::vec& dst_data,
              hvx::util::vector<typename param_::type, dst_dim_::vec_size>* dst,
              hvx::util::vector<typename param_::type, dst_dim_rest_::vec_size>*... dst_rest) noexcept -> void {
    HVX_INLINE_TOP();

    // gets a pointer that points to the end of the region of a given input
    constexpr int64_t dst_dim_end = dst_dim_beg_ + ((id_ == 0) ? (dst_dim_::dims[id_] / dst_dim_::vec_size) : (dst_dim_::dims[id_]));

    // select the correct output to write to
    if (ptr_dim < dst_dim_end)
        hvx::util::StreamWriteData<hvx::util::vector<typename param_::type, dst_dim_::vec_size>>(
            dst, dst_data, ptr_dst.Get(sizeof...(dst_dim_rest_)), true);
    else
        hvx::convert::HwSplitOutput<dst_num, id_, dst_dim_end, param_, dst_dim_rest_...>(ptr_dim, ptr_dst, dst_data, dst_rest...);
}

/*!
 * @brief Gets dimension ID, reads input data and writes to correct output
 */
template<int64_t dst_num, typename param_, typename dst_dim_, typename... dst_dim_rest_>
HVX_FORCE_INLINE constexpr auto
HwSplitBase(hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax>& ptr_dim,
            hvx::util::vector<int64_t, dst_num>& ptr_dst,
            int64_t& ptr_src,
            typename param_::vec* src,
            hvx::util::vector<typename param_::type, dst_dim_::vec_size>* dst,
            hvx::util::vector<typename param_::type, dst_dim_rest_::vec_size>*... dst_rest) noexcept -> void {
    HVX_INLINE_TOP();

    // gets the dimension that is split
    constexpr int64_t id = hvx::convert::HwSplitGetId<typename param_::dim, dst_dim_>(0);

    // stores the input data
    typename param_::vec src_data{};

    // read input data
    hvx::util::StreamReadData<typename param_::vec>(src, src_data, ptr_src, true);

    // writes to correct output
    hvx::convert::HwSplitOutput<dst_num, id, 0, param_, dst_dim_, dst_dim_rest_...>(ptr_dim.Get(id), ptr_dst, src_data, dst, dst_rest...);
}

/*!
 * @brief Top HW split function
 */
template<typename param_, typename dst_dim_, typename... dst_dim_rest_>
HVX_FORCE_INLINE constexpr auto
HwSplitTop(typename param_::vec* src,
           hvx::util::vector<typename param_::type, dst_dim_::vec_size>* dst,
           hvx::util::vector<typename param_::type, dst_dim_rest_::vec_size>*... dst_rest) noexcept -> void {
    HVX_INLINE_TOP();

    // number of output and dimensions
    constexpr int64_t dst_num = sizeof...(dst_dim_rest_) + 1;

    // pointer to inputs/outputs
    hvx::util::vector<int64_t, dst_num> ptr_dst{};
    hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax> ptr_dim{};
    HVX_ARRAY_PARTITION_COMPLETE(ptr_dst.data, 0);
    HVX_ARRAY_PARTITION_COMPLETE(ptr_dim.data, 0);

    // iterates over all dimensions of the tensor
    for (int64_t i = 0, ptr_src = 0; i < param_::dim::vec_elms; ++i) {
        HVX_PIPELINE_ON(1, frp);

        // flattening loop to improve latency
        hvx::util::TensorUpdateDimPtr<typename param_::dim>(ptr_dim, i);

        // Gets dimension ID, reads input data and writes to correct output
        hvx::convert::HwSplitBase<dst_num, param_, dst_dim_, dst_dim_rest_...>(ptr_dim, ptr_dst, ptr_src, src, dst, dst_rest...);
    }
}

/******************************************************************************************************************************************/
} // namespace convert
} // namespace hvx

#endif // HVX_CONVERT_SPLIT_H
