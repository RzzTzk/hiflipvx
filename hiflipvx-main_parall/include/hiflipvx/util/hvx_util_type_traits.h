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
 * @file    hvx_util_type_traits.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_TYPE_TRAITS_H_
#define HVX_UTIL_TYPE_TRAITS_H_

#include "hvx_util_enum.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

/*!
 * @brief Splits/Concatenates multiple tensors on a certain dimension
 */
template<int64_t dim_id_,     // ID of the dimension that should be split/concat
         int64_t port0_elms_, // number of elements of the first port
         int64_t port1_elms_, // number of elements of the second port
         int64_t port2_elms_ = 0,
         int64_t port3_elms_ = 0,
         int64_t port4_elms_ = 0,
         int64_t port5_elms_ = 0>
struct ConcatSplitParam {
    static constexpr auto dim_id     = dim_id_;
    static constexpr auto port0_elms = port0_elms_;
    static constexpr auto port1_elms = port1_elms_;
    static constexpr auto port2_elms = port2_elms_;
    static constexpr auto port3_elms = port3_elms_;
    static constexpr auto port4_elms = port4_elms_;
    static constexpr auto port5_elms = port5_elms_;
};

/*!
 * @brief Determines which params should be reduced
 */
template<bool dim0_ = false, bool dim1_ = false, bool dim2_ = false, bool dim3_ = false, bool dim4_ = false, bool dim5_ = false>
struct ReduceParam {
    static constexpr bool dims[hvx::util::limits_e::kTensorDimMax] = {dim0_, dim1_, dim2_, dim3_, dim4_, dim5_};//NOLINT
};

/*!
 * @brief The permutation for the transpose function
 */
template<int64_t dim0_, int64_t dim1_, int64_t dim2_ = 2, int64_t dim3_ = 3, int64_t dim4_ = 4, int64_t dim5_ = 5>
struct TransposePerm {
    static constexpr auto dim0 = dim0_;
    static constexpr auto dim1 = dim1_;
    static constexpr auto dim2 = dim2_;
    static constexpr auto dim3 = dim3_;
    static constexpr auto dim4 = dim4_;
    static constexpr auto dim5 = dim5_;
};

/*!
 * @brief Dimensions of a 2-dimensional array
 */
template<int64_t rows_, int64_t cols_>
struct Array2dParam {
    static constexpr auto rows = rows_;
    static constexpr auto cols = cols_;
};

/*!
 * @brief Parameters to clip a value to a maximum and minimum value
 */
template<int64_t max_, int64_t min_>
struct ClipParam {
    static constexpr auto max = max_;
    static constexpr auto min = min_;
};

/*!
 * @brief Compile time parameters and checks of a vector
 */
template<int64_t elms_, int64_t vec_size_>
struct VectorParam {
    // TODO: add dimension name (dimension_e)
    static constexpr auto elms     = elms_;
    static constexpr auto vec_size = vec_size_;
    static constexpr auto vec_elms = elms_ / vec_size_;
    static_assert((elms >= vec_size) && ((elms % vec_size) == 0), "Elements should be multiple of vector size!");
    static_assert((elms >= 1) && (elms <= 8192), "Number of elements invalid!");
    static_assert((vec_size >= 1) && (vec_size <= 8192), "Vector size invalid!");
};

/*!
 * @brief Compile time parameters and checks of a 1-6 dimensional tensor
 */
template<int64_t dim_num_ = 1,
         typename dim0_   = hvx::util::VectorParam<1, 1>,
         typename dim1_   = hvx::util::VectorParam<1, 1>,
         typename dim2_   = hvx::util::VectorParam<1, 1>,
         typename dim3_   = hvx::util::VectorParam<1, 1>,
         typename dim4_   = hvx::util::VectorParam<1, 1>,
         typename dim5_   = hvx::util::VectorParam<1, 1>>
struct TensorParam {
    using type_t = std::array<int64_t, hvx::util::limits_e::kTensorDimMax>;

    // storing the vector parameters
    using dim0 = dim0_;
    using dim1 = dim1_;
    using dim2 = dim2_;
    using dim3 = dim3_;
    using dim4 = dim4_;
    using dim5 = dim5_;

    // constant parameters
    static constexpr type_t dims = {dim0_::elms, dim1_::elms, dim2_::elms, dim3_::elms, dim4_::elms, dim5_::elms};
    static constexpr type_t vecs = {dim0_::vec_size, dim1_::vec_size, dim2_::vec_size, dim3_::vec_size, dim4_::vec_size, dim5_::vec_size};
    static constexpr type_t dim_vec_elms = {dim0_::vec_elms, dim1_::vec_elms, dim2_::vec_elms,
                                            dim3_::vec_elms, dim4_::vec_elms, dim5_::vec_elms};

    static constexpr int64_t elms = dim0_::elms * dim1_::elms * dim2_::elms * dim3_::elms * dim4_::elms * dim5_::elms;
    static constexpr int64_t vec_size =
        dim0_::vec_size * dim1_::vec_size * dim2_::vec_size * dim3_::vec_size * dim4_::vec_size * dim5_::vec_size;
    static constexpr int64_t vec_elms = elms / vec_size;
    static constexpr int64_t dim_num  = dim_num_;

    // compile time assertions
    static_assert(dim_num >= 1 && dim_num <= hvx::util::limits_e::kTensorDimMax, "Tensor has to many or to few dimensions!");
    static_assert((0 < dim_num) ? (true) : (dims[0] == 1), "Dimension 0 invalid!");
    static_assert((1 < dim_num) ? (true) : (dims[1] == 1), "Dimension 1 invalid!");
    static_assert((2 < dim_num) ? (true) : (dims[2] == 1), "Dimension 2 invalid!");
    static_assert((3 < dim_num) ? (true) : (dims[3] == 1), "Dimension 3 invalid!");
    static_assert((4 < dim_num) ? (true) : (dims[4] == 1), "Dimension 4 invalid!");
    static_assert((5 < dim_num) ? (true) : (dims[5] == 1), "Dimension 5 invalid!");
};

/*!
 * @brief Creates the tensor dimensions for the split/concat functions based on their parameter setting
 */
template<typename dst_dim_, int64_t dim_id_, int64_t elms_>
struct ConcatSplitTensorParam {
    using dim = hvx::util::TensorParam<
        dst_dim_::dim_num,
        std::conditional_t<dim_id_ == 0, hvx::util::VectorParam<elms_, dst_dim_::vecs[dim_id_]>, typename dst_dim_::dim0>,
        std::conditional_t<dim_id_ == 1, hvx::util::VectorParam<elms_, dst_dim_::vecs[dim_id_]>, typename dst_dim_::dim1>,
        std::conditional_t<dim_id_ == 2, hvx::util::VectorParam<elms_, dst_dim_::vecs[dim_id_]>, typename dst_dim_::dim2>,
        std::conditional_t<dim_id_ == 3, hvx::util::VectorParam<elms_, dst_dim_::vecs[dim_id_]>, typename dst_dim_::dim3>,
        std::conditional_t<dim_id_ == 4, hvx::util::VectorParam<elms_, dst_dim_::vecs[dim_id_]>, typename dst_dim_::dim4>,
        std::conditional_t<dim_id_ == 5, hvx::util::VectorParam<elms_, dst_dim_::vecs[dim_id_]>, typename dst_dim_::dim5>>;
};

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_UTIL_TYPE_TRAITS_H_
