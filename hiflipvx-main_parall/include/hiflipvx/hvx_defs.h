/**
 *  Copyright <2024> <Lester Kalms>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
 * ï¿½Softwareï¿½), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED ï¿½AS ISï¿½, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
 * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Additional restriction: The Software and its derivatives may not be used for, or in support of, any military purposes.
 *
 * @file    hvx_defs.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_DEFS_H_
#define HVX_DEFS_H_

#include "convert/hvx_convert_concat.h"
#include "convert/hvx_convert_multicast.h"
#include "convert/hvx_convert_reshape.h"
#include "convert/hvx_convert_split.h"
#include "convert/hvx_convert_stream.h"
#include "convert/hvx_convert_transpose.h"
#include "nn/hvx_nn_conv.h"
#include "nn/hvx_nn_dense.h"
#include "nn/hvx_nn_depthwise.h"
#include "nn/hvx_nn_layernorm.h"
#include "nn/hvx_nn_pool.h"
#include "nn/hvx_nn_softmax.h"
#include "op/hvx_ew_core.h"
#include "op/hvx_reduce_core.h"

namespace hvx {
/******************************************************************************************************************************************/

// global enums
using overflow_e  = hvx::util::overflow_e;
using underflow_e = hvx::util::underflow_e;
using execution_e = hvx::util::execution_e;
using axis_e      = hvx::util::axis_e;

/*!
 * @brief Definition of a fixed-point data type
 */
template<int64_t exp_bits_, int64_t man_bits_>
using dfloat = dynfloat::dfloat<exp_bits_, man_bits_>;

/*!
 * @brief Definition of a fixed-point data type
 */
template<typename type_, int64_t frac_bits_ = 0>
using dfixed = hvx::util::dfixed<type_, frac_bits_>;

/*!
 * @brief A simple array data type
 */
template<typename type_, int64_t cols_>
using array1d = hvx::util::array1d<type_, cols_>;

/*!
 * @brief Dimensions of a 2-dimensional array
 */
template<int64_t rows_, int64_t cols_>
using array2d_param = hvx::util::Array2dParam<rows_, cols_>;

/*!
 * @brief Parameters to clip a value to a maximum and minimum value
 */
template<int64_t max_, int64_t min_>
using clip_param = hvx::util::ClipParam<max_, min_>;

/*!
 * @brief Compile time parameters and checks of a vector
 */
template<int64_t elms_, int64_t vec_size_>
using vector_param = hvx::util::VectorParam<elms_, vec_size_>;

/*!
 * @brief Compile time parameters and checks of a 1-6 dimensional tensor
 */
template<int64_t dim_num_ = 1,
         typename dim0_   = hvx::vector_param<1, 1>,
         typename dim1_   = hvx::vector_param<1, 1>,
         typename dim2_   = hvx::vector_param<1, 1>,
         typename dim3_   = hvx::vector_param<1, 1>,
         typename dim4_   = hvx::vector_param<1, 1>,
         typename dim5_   = hvx::vector_param<1, 1>>
using tensor_param = hvx::util::TensorParam<dim_num_, dim0_, dim1_, dim2_, dim3_, dim4_, dim5_>;

/******************************************************************************************************************************************/

/*!
 * @brief This class is needed for the HVX <-> Stream conversion functions (for side channels)
 */
template<typename type_, typename dim_, int64_t flags_>
using stream_param = hvx::convert::StreamParam<type_, dim_, flags_>;

/*!
 * @brief All compile time parameters and checks for the concat function
 */
template<typename type_, typename dst_dim_, typename params_>
using concat_param = hvx::convert::ConcatParam<type_, dst_dim_, params_>;

/*!
 * @brief All compile time parameters and checks for the multicast function
 */
template<typename type_, typename dim_>
using multicast_param = hvx::convert::MulticastParam<type_, dim_>;

/*!
 * @brief All compile time parameters and checks for the reshape function
 */
template<typename type_, typename src_dim_, typename dst_dim_>
using reshape_param = hvx::convert::ReshapeParam<type_, src_dim_, dst_dim_>;

/*!
 * @brief All compile time parameters and checks for the split function
 */
template<typename type_, typename src_dim_, typename params_>
using split_param = hvx::convert::SplitParam<type_, src_dim_, params_>;

/*!
 * @brief All compile time parameters and checks for the transpose function
 */
template<typename type_, typename src_dim_, typename perm_, int64_t dst_dim0_vec_size_>
using transpose_param = hvx::convert::TransposeParam<type_, src_dim_, perm_, dst_dim0_vec_size_>;

/*!
 * @brief All compile time parameters and checks for the transpose function
 */
template<typename type_, typename src_dim_, typename perm_, int64_t dst_dim0_vec_size_>
using transpose_param = hvx::convert::TransposeParam<type_, src_dim_, perm_, dst_dim0_vec_size_>;

/*!
 * @brief The permutation for the transpose function
 */
template<int64_t dim0_, int64_t dim1_, int64_t dim2_ = 2, int64_t dim3_ = 3, int64_t dim4_ = 4, int64_t dim5_ = 5>
using transpose_perm = hvx::util::TransposePerm<dim0_, dim1_, dim2_, dim3_, dim4_, dim5_>;

/*!
 * @brief All compile time parameters and checks for the reorder function
 */
template<typename type_, typename src_dim_, typename perm_, int64_t dst_dim0_vec_size_>
using transpose_param = hvx::convert::TransposeParam<type_, src_dim_, perm_, dst_dim0_vec_size_>;

/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the elementwise absolute function
 */
template<typename src_type_                = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using abs_param =
    hvx::ew::Elmwise<src_type_, src_type_, dst_type_, src_type_, src_dim_, overflow_, underflow_, exec_, hvx::util::elmwise_e::Abs>;

/*!
 * @brief All compile time parameters and checks for the elementwise addition
 */
template<typename src1_type_               = hvx::util::dfixed<int16_t, 15>,
         typename src2_type_               = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using add_param =
    hvx::ew::Elmwise<src1_type_, src2_type_, dst_type_, dst_type_, src_dim_, overflow_, underflow_, exec_, hvx::util::elmwise_e::Add>;

/*!
 * @brief  All compile time parameters and checks for the elementwise addition with a constant
 */
template<typename src_type_                = hvx::util::dfixed<int16_t, 15>,
         typename arg_type_                = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using addconst_param =
    hvx::ew::Elmwise<src_type_, src_type_, dst_type_, arg_type_, src_dim_, overflow_, underflow_, exec_, hvx::util::elmwise_e::AddConst>;

/*!
 * @brief  All compile time parameters and checks for the elementwise clip
 */
template<typename src_type_                = hvx::util::dfixed<int16_t, 15>,
         typename arg_type_                = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using clip2_param =
    hvx::ew::Elmwise<src_type_, src_type_, dst_type_, arg_type_, src_dim_, overflow_, underflow_, exec_, hvx::util::elmwise_e::Clip>;

/*!
 * @brief All compile time parameters and checks for the elementwise max function
 */
template<typename src1_type_               = hvx::util::dfixed<int16_t, 15>,
         typename src2_type_               = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using max_param =
    hvx::ew::Elmwise<src1_type_, src2_type_, dst_type_, dst_type_, src_dim_, overflow_, underflow_, exec_, hvx::util::elmwise_e::Max>;

/*!
 * @brief  All compile time parameters and checks for the elementwise max function with a constant (RELU)
 */
template<typename src_type_                = hvx::util::dfixed<int16_t, 15>,
         typename arg_type_                = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using maxconst_param =
    hvx::ew::Elmwise<src_type_, src_type_, dst_type_, arg_type_, src_dim_, overflow_, underflow_, exec_, hvx::util::elmwise_e::MaxConst>;

/*!
 * @brief  All compile time parameters and checks for the elementwise min function
 */
template<typename src1_type_               = hvx::util::dfixed<int16_t, 15>,
         typename src2_type_               = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using min_param =
    hvx::ew::Elmwise<src1_type_, src2_type_, dst_type_, dst_type_, src_dim_, overflow_, underflow_, exec_, hvx::util::elmwise_e::Min>;

/*!
 * @brief  All compile time parameters and checks for the elementwise min function with a constant
 */
template<typename src_type_                = hvx::util::dfixed<int16_t, 15>,
         typename arg_type_                = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using minconst_param =
    hvx::ew::Elmwise<src_type_, src_type_, dst_type_, arg_type_, src_dim_, overflow_, underflow_, exec_, hvx::util::elmwise_e::MinConst>;

/*!
 * @brief All compile time parameters and checks for the elementwise multiplication
 */
template<typename src1_type_               = hvx::util::dfixed<int16_t, 15>,
         typename src2_type_               = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using mul_param =
    hvx::ew::Elmwise<src1_type_, src2_type_, dst_type_, dst_type_, src_dim_, overflow_, underflow_, exec_, hvx::util::elmwise_e::Mul>;

/*!
 * @brief  All compile time parameters and checks for the elementwise multiplication with a constant
 */
template<typename src_type_                = hvx::util::dfixed<int16_t, 15>,
         typename arg_type_                = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using mulconst_param =
    hvx::ew::Elmwise<src_type_, src_type_, dst_type_, arg_type_, src_dim_, overflow_, underflow_, exec_, hvx::util::elmwise_e::MulConst>;

/*!
 * @brief  All compile time parameters and checks for the elementwise Sigmoid function
 */
template<typename src_type_                = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using sigmoid_param =
    hvx::ew::Elmwise<src_type_, src_type_, dst_type_, dst_type_, src_dim_, overflow_, underflow_, exec_, hvx::util::elmwise_e::Sigmoid>;

/*!
 * @brief  All compile time parameters and checks for the elementwise subtraction
 */
template<typename src1_type_               = hvx::util::dfixed<int16_t, 15>,
         typename src2_type_               = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using sub_param =
    hvx::ew::Elmwise<src1_type_, src2_type_, dst_type_, dst_type_, src_dim_, overflow_, underflow_, exec_, hvx::util::elmwise_e::Sub>;

/*!
 * @brief  All compile time parameters and checks for the elementwise Tanh function
 */
template<typename src_type_                = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using tanh_param =
    hvx::ew::Elmwise<src_type_, src_type_, dst_type_, dst_type_, src_dim_, overflow_, underflow_, exec_, hvx::util::elmwise_e::Tanh>;

/******************************************************************************************************************************************/

template<typename src_type_                = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         typename reduce_                  = hvx::util::ReduceParam<false, false, false, false, false, false>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using reduce_max_param = hvx::red::Reduce<src_type_, dst_type_, src_dim_, reduce_, overflow_, underflow_, exec_, hvx::util::reduce_e::Max>;

template<typename src_type_                = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         typename reduce_                  = hvx::util::ReduceParam<false, false, false, false, false, false>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using reduce_mean_param =
    hvx::red::Reduce<src_type_, dst_type_, src_dim_, reduce_, overflow_, underflow_, exec_, hvx::util::reduce_e::Mean>;

template<typename src_type_                = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         typename reduce_                  = hvx::util::ReduceParam<false, false, false, false, false, false>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using reduce_min_param = hvx::red::Reduce<src_type_, dst_type_, src_dim_, reduce_, overflow_, underflow_, exec_, hvx::util::reduce_e::Min>;

template<typename src_type_                = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                 = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         typename reduce_                  = hvx::util::ReduceParam<false, false, false, false, false, false>,
         hvx::util::overflow_e overflow_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_      = hvx::util::execution_e::kExact>
using reduce_sum_param = hvx::red::Reduce<src_type_, dst_type_, src_dim_, reduce_, overflow_, underflow_, exec_, hvx::util::reduce_e::Sum>;

/******************************************************************************************************************************************/

/*!
 * @brief Compile time parameters and checks for layer normalization function
 */
template<typename src_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename wgts_type_                    = hvx::util::dfixed<int16_t, 15>,
         typename bias_type_                    = hvx::util::dfixed<int16_t, 15>,
         typename batch_v                       = hvx::util::VectorParam<1, 1>,
         typename src_rows_v                    = hvx::util::VectorParam<1, 1>,
         typename src_cols_v                    = hvx::util::VectorParam<1, 1>,
         typename chnls_v                       = hvx::util::VectorParam<1, 1>,
         int64_t buf_wgts_                      = false,
         int64_t buf_bias_                      = false,
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact,
         typename clip_                         = hvx::util::ClipParam<1, 0>>
using layernorm_param = hvx::nn::LayernormParam<src_type_,
                                                dst_type_,
                                                wgts_type_,
                                                bias_type_,
                                                batch_v,
                                                src_rows_v,
                                                src_cols_v,
                                                chnls_v,
                                                buf_wgts_,
                                                buf_bias_,
                                                overflow_type_,
                                                underflow_type_,
                                                exec_type_,
                                                clip_>;

/*!
 * @brief Compile time parameters and checks for softmax function
 */
template<typename src_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename batch_v                       = hvx::util::VectorParam<1, 1>,
         typename src_rows_v                    = hvx::util::VectorParam<1, 1>,
         typename src_cols_v                    = hvx::util::VectorParam<1, 1>,
         typename chnls_v                       = hvx::util::VectorParam<1, 1>,
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact>
using softmax_param =
    hvx::nn::SoftmaxParam<src_type_, dst_type_, batch_v, src_rows_v, src_cols_v, chnls_v, overflow_type_, underflow_type_, exec_type_>;

/*!
 * @brief Compile time parameters and checks for dense (fully connected) function
 */
template<typename src_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename wgts_type_                    = hvx::util::dfixed<int16_t, 15>,
         typename bias_type_                    = hvx::util::dfixed<int16_t, 15>,
         typename batch_v                       = hvx::util::VectorParam<1, 1>,
         typename chnls_v                       = hvx::util::VectorParam<1, 1>,
         typename fms_v                         = hvx::util::VectorParam<1, 1>,
         int64_t buf_wgts_                      = false,
         int64_t buf_bias_                      = false,
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact>
using dense_param = hvx::nn::DenseParam<src_type_,
                                        dst_type_,
                                        wgts_type_,
                                        bias_type_,
                                        batch_v,
                                        chnls_v,
                                        fms_v,
                                        buf_wgts_,
                                        buf_bias_,
                                        overflow_type_,
                                        underflow_type_,
                                        exec_type_>;

/*!
 * @brief Compile time parameters and checks for average pooling function
 */
template<typename src_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename batch_size_                   = hvx::util::VectorParam<1, 1>,
         typename src_rows_                     = hvx::util::VectorParam<1, 1>,
         typename src_cols_                     = hvx::util::VectorParam<1, 1>,
         typename chnls_                        = hvx::util::VectorParam<1, 1>,
         typename knl_rows_                     = hvx::util::VectorParam<1, 1>,
         typename knl_cols_                     = hvx::util::VectorParam<1, 1>,
         typename pad_                          = hvx::util::Array2dParam<0, 0>,
         typename dil_                          = hvx::util::Array2dParam<0, 0>,
         typename str_                          = hvx::util::Array2dParam<1, 1>,
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact>
using pool_avg_param = hvx::nn::PoolParam<src_type_,
                                          dst_type_,
                                          batch_size_,
                                          src_rows_,
                                          src_cols_,
                                          chnls_,
                                          knl_rows_,
                                          knl_cols_,
                                          pad_,
                                          dil_,
                                          str_,
                                          overflow_type_,
                                          underflow_type_,
                                          exec_type_,
                                          hvx::util::pooling_e::kAvg>;

/*!
 * @brief Compile time parameters and checks for max pooling function
 */
template<typename src_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename batch_size_                   = hvx::util::VectorParam<1, 1>,
         typename src_rows_                     = hvx::util::VectorParam<1, 1>,
         typename src_cols_                     = hvx::util::VectorParam<1, 1>,
         typename chnls_                        = hvx::util::VectorParam<1, 1>,
         typename knl_rows_                     = hvx::util::VectorParam<1, 1>,
         typename knl_cols_                     = hvx::util::VectorParam<1, 1>,
         typename pad_                          = hvx::util::Array2dParam<0, 0>,
         typename dil_                          = hvx::util::Array2dParam<0, 0>,
         typename str_                          = hvx::util::Array2dParam<1, 1>,
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact>
using pool_max_param = hvx::nn::PoolParam<src_type_,
                                          dst_type_,
                                          batch_size_,
                                          src_rows_,
                                          src_cols_,
                                          chnls_,
                                          knl_rows_,
                                          knl_cols_,
                                          pad_,
                                          dil_,
                                          str_,
                                          overflow_type_,
                                          underflow_type_,
                                          exec_type_,
                                          hvx::util::pooling_e::kMax>;

/*!
 * @brief Compile time parameters and checks for depthwise convolution function
 */
template<typename src_type_               = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_               = hvx::util::dfixed<int16_t, 15>,
         typename wgts_type_              = hvx::util::dfixed<int16_t, 15>,
         typename bias_type_              = dst_type_,
         typename batch_size_             = hvx::vector_param<1, 1>,
         typename src_rows_               = hvx::vector_param<1, 1>,
         typename src_cols_               = hvx::vector_param<1, 1>,
         typename chnls_                  = hvx::vector_param<1, 1>,
         typename knl_rows_               = hvx::vector_param<1, 1>,
         typename knl_cols_               = hvx::vector_param<1, 1>,
         typename pad_                    = hvx::array2d_param<0, 0>,
         typename dil_                    = hvx::array2d_param<0, 0>,
         typename str_                    = hvx::array2d_param<1, 1>,
         int64_t buf_wgts_                = false,
         int64_t buf_bias_                = false,
         hvx::overflow_e overflow_type_   = hvx::overflow_e::kSaturate,
         hvx::underflow_e underflow_type_ = hvx::underflow_e::kTrunc,
         hvx::execution_e exec_type_      = hvx::execution_e::kExact>
using depthwise_param = hvx::nn::DepthwiseParam<src_type_,
                                                dst_type_,
                                                wgts_type_,
                                                bias_type_,
                                                batch_size_,
                                                src_rows_,
                                                src_cols_,
                                                chnls_,
                                                knl_rows_,
                                                knl_cols_,
                                                pad_,
                                                dil_,
                                                str_,
                                                buf_wgts_,
                                                buf_bias_,
                                                overflow_type_,
                                                underflow_type_,
                                                exec_type_>;

/*!
 * @brief Compile time parameters and checks for convolution function
 */
template<typename src_type_               = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_               = hvx::util::dfixed<int16_t, 15>,
         typename wgts_type_              = hvx::util::dfixed<int16_t, 15>,
         typename bias_type_              = hvx::util::dfixed<int16_t, 15>,
         typename batch_size_             = hvx::vector_param<1, 1>,
         typename src_rows_               = hvx::vector_param<1, 1>,
         typename src_cols_               = hvx::vector_param<1, 1>,
         typename chnls_                  = hvx::vector_param<1, 1>,
         typename fms_                    = hvx::vector_param<1, 1>,
         typename knl_rows_               = hvx::vector_param<1, 1>,
         typename knl_cols_               = hvx::vector_param<1, 1>,
         typename pad_                    = hvx::array2d_param<0, 0>,
         typename dil_                    = hvx::array2d_param<0, 0>,
         typename str_                    = hvx::array2d_param<1, 1>,
         int64_t buf_wgts_                = false,
         int64_t buf_bias_                = false,
         hvx::overflow_e overflow_type_   = hvx::overflow_e::kSaturate,
         hvx::underflow_e underflow_type_ = hvx::underflow_e::kTrunc,
         hvx::execution_e exec_type_      = hvx::execution_e::kExact>
using conv_param = hvx::nn::ConvParam<src_type_,
                                      dst_type_,
                                      wgts_type_,
                                      bias_type_,
                                      batch_size_,
                                      src_rows_,
                                      src_cols_,
                                      chnls_,
                                      fms_,
                                      knl_rows_,
                                      knl_cols_,
                                      pad_,
                                      dil_,
                                      str_,
                                      buf_wgts_,
                                      buf_bias_,
                                      overflow_type_,
                                      underflow_type_,
                                      exec_type_>;



/******************************************************************************************************************************************/
} // namespace hvx

#endif // HVX_DEFS_H_
