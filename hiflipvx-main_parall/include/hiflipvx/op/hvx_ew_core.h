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
 * @file    hvx_ew_core.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_EW_CORE_H_
#define HVX_EW_CORE_H_

#include "impl/hvx_ew_dfixed.h"
#include "impl/hvx_ew_dfloat.h"

namespace hvx {
namespace ew {
/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the elementwise operations
 */
template<typename src1_type_                    = hvx::util::dfixed<int16_t, 15>,
         typename src2_type_                    = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename arg_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                      = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact,
         hvx::util::elmwise_e op_type_          = hvx::util::elmwise_e::None>
struct Elmwise {
    // dimensions
    using src1_dim = src_dim_;
    using src2_dim = src_dim_;
    using dst_dim  = src_dim_;

    // types
    using src1_type = src1_type_;
    using src2_type = src2_type_;
    using dst_type  = dst_type_;
    using arg_type  = arg_type_;
    using src1_vec  = hvx::util::vector<src1_type, src1_dim::vec_size>;
    using src2_vec  = hvx::util::vector<src2_type, src2_dim::vec_size>;
    using dst_vec   = hvx::util::vector<dst_type, dst_dim::vec_size>;
    using src1_port = src1_vec;
    using src2_port = src2_vec;
    using dst_port  = dst_vec;

    // parameters
    static constexpr auto vec_elms       = dst_dim::vec_elms;
    static constexpr auto vec_size       = dst_dim::vec_size;
    static constexpr auto overflow_type  = overflow_type_;
    static constexpr auto underflow_type = underflow_type_;
    static constexpr auto exec_type      = exec_type_;
    static constexpr auto op_type        = op_type_;

    // condition if data needs to be readed for operand A and B
    static constexpr bool src1_cond = true;
    static constexpr bool src2_cond = (op_type_ == hvx::util::elmwise_e::Add) || (op_type_ == hvx::util::elmwise_e::Max) ||
                                      (op_type_ == hvx::util::elmwise_e::Min) || (op_type_ == hvx::util::elmwise_e::Mul) ||
                                      (op_type_ == hvx::util::elmwise_e::Sub);

    // assertions
    constexpr Elmwise() {
        hvx::ew::impl::VerifyDataType<src1_type, src2_type, dst_type, arg_type>();
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Top function of all HW elementwise operations
 */
template<typename param_>
HVX_FORCE_INLINE constexpr auto
ElementwiseTop(typename param_::src1_port* src1,
               typename param_::src2_port* src2,
               typename param_::dst_port* dst,
               const typename param_::arg_type arg1,
               const typename param_::arg_type arg2) noexcept -> void {
    // iterates through the tensor vector by vector
    int64_t src1_ptr = 0, src2_ptr = 0, ptr_dst = 0;
    for (int64_t i = 0; i < param_::vec_elms; ++i) {
        HVX_PIPELINE_ON(1, frp);

        // buffer the src and dst vectors
        typename param_::src1_vec src1_data{};
        typename param_::src2_vec src2_data{};
        typename param_::dst_vec dst_data{};

        // read next src vector
        hvx::util::StreamReadData<>(src1, src1_data, src1_ptr, param_::src1_cond);
        hvx::util::StreamReadData<>(src2, src2_data, src2_ptr, param_::src2_cond);

        // applies the operator on a complete vector
        for (int64_t j = 0; j < param_::vec_size; j++) {
            HVX_UNROLL();
            hvx::ew::impl::ElementwiseComp<param_>(src1_data.Get(j), src2_data.Get(j), arg1, arg2, dst_data.Get(j));
        }

        // write next dst vector
        hvx::util::StreamWriteData<>(dst, dst_data, ptr_dst, true);
    }
    hvx::util::StreamSignalVerify<typename param_::src1_dim, typename param_::dst_dim>(src1_ptr, ptr_dst);
}

/******************************************************************************************************************************************/
} // namespace ew
} // namespace hvx

#endif // HVX_EW_CORE_H_
