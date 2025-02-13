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
 * @file    hvx_util_tensor.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_UTIL_TENSOR_H_
#define HVX_UTIL_TENSOR_H_

#include "hvx_util_array.h"

namespace hvx {
namespace util {
/******************************************************************************************************************************************/

/*!
 * @brief verifies if the selected dimensions of two tensor are the same
 */
template<typename src_dim_, typename dst_dim_, int64_t dim_num_, bool dim0_, bool dim1_, bool dim2_, bool dim3_, bool dim4_, bool dim5_>
HVX_FORCE_INLINE constexpr auto
TensorVerifySameDims() noexcept -> void {
    HVX_INLINE_TOP();
    static_assert((src_dim_::dim_num == dim_num_) && (dst_dim_::dim_num == dim_num_), "Invalid number of dimension!");
    static_assert(((src_dim_::dims[0] == dst_dim_::dims[0]) && (src_dim_::vecs[0] == dst_dim_::vecs[0])) || !dim0_, "Dim 0 not equal!");
    static_assert(((src_dim_::dims[1] == dst_dim_::dims[1]) && (src_dim_::vecs[1] == dst_dim_::vecs[1])) || !dim1_, "Dim 1 not equal!");
    static_assert(((src_dim_::dims[2] == dst_dim_::dims[2]) && (src_dim_::vecs[2] == dst_dim_::vecs[2])) || !dim2_, "Dim 2 not equal!");
    static_assert(((src_dim_::dims[3] == dst_dim_::dims[3]) && (src_dim_::vecs[3] == dst_dim_::vecs[3])) || !dim3_, "Dim 3 not equal!");
    static_assert(((src_dim_::dims[4] == dst_dim_::dims[4]) && (src_dim_::vecs[4] == dst_dim_::vecs[4])) || !dim4_, "Dim 4 not equal!");
    static_assert(((src_dim_::dims[5] == dst_dim_::dims[5]) && (src_dim_::vecs[5] == dst_dim_::vecs[5])) || !dim5_, "Dim 5 not equal!");
}

/*!
 * @brief verifies if the vector size of the selected dimensions is 1 (e.g. if no vectorization has been implemented)
 */
template<typename dim_, bool dim0_, bool dim1_, bool dim2_, bool dim3_, bool dim4_, bool dim5_>
HVX_FORCE_INLINE constexpr auto
TensorVerifyIfVecSizeIs1() noexcept -> void {
    HVX_INLINE_TOP();
    static_assert((dim_::vecs[0] == 1) || !dim0_, "Vector size of dim6 is not equal 1!");
    static_assert((dim_::vecs[1] == 1) || !dim1_, "Vector size of dim5 is not equal 1!");
    static_assert((dim_::vecs[2] == 1) || !dim2_, "Vector size of dim4 is not equal 1!");
    static_assert((dim_::vecs[3] == 1) || !dim3_, "Vector size of dim3 is not equal 1!");
    static_assert((dim_::vecs[4] == 1) || !dim4_, "Vector size of dim2 is not equal 1!");
    static_assert((dim_::vecs[5] == 1) || !dim5_, "Vector size of dim1 is not equal 1!");
}

/******************************************************************************************************************************************/
// TODO: try to delete these functions

/*!
 * @brief gets the number of elements of a specific dimension
 */
template<typename dim_, int64_t dim_id_>
HVX_FORCE_INLINE constexpr auto
TensorGetDimElms() noexcept -> int64_t {
    HVX_INLINE_TOP();
    static_assert((dim_id_ >= 0) && (dim_id_ < hvx::util::limits_e::kTensorDimMax), "Out of range!");
    return ((dim_id_ < 0) || (dim_id_ >= dim_::dim_num)) ? (1) : (dim_::dims[dim_id_]);
}

/*!
 * @brief gets the vector size of a specific dimension
 */
template<typename dim_, int64_t dim_id_>
HVX_FORCE_INLINE constexpr auto
TensorGetDimVecSize() noexcept -> int64_t {
    HVX_INLINE_TOP();
    static_assert((dim_id_ >= 0) && (dim_id_ < hvx::util::limits_e::kTensorDimMax), "Out of range!");
    return ((dim_id_ < 0) || (dim_id_ >= dim_::dim_num)) ? (1) : (dim_::vecs[dim_id_]);
}

/*!
 * @brief gets the total number of elements from dim[0] to dim[dim_num-1]
 */
template<typename dim_, int64_t dim_num_>
HVX_FORCE_INLINE constexpr auto
TensorGetRangeElms() noexcept -> int64_t {
    HVX_INLINE_TOP();
    static_assert((dim_num_ >= 0) && (dim_num_ <= hvx::util::limits_e::kTensorDimMax), "Out of range!");
    constexpr int64_t dim_id = hvx::util::Max(dim_num_ - 1, static_cast<int64_t>(0));
    return (dim_num_ > 0) ? hvx::util::TensorGetDimElms<dim_, dim_id>() * hvx::util::TensorGetRangeElms<dim_, dim_id>()
                          : static_cast<int64_t>(1);
}

/*!
 * @brief get the total number of elements
 */
template<typename dim_>
HVX_FORCE_INLINE constexpr auto
TensorGetElms() noexcept -> int64_t {
    HVX_INLINE_TOP();
    return hvx::util::TensorGetRangeElms<dim_, dim_::dim_num>();
}

/*!
 * @brief get the pointer to a flattened n dimensional array starting with the highest dimension
 */
template<typename dim_>
constexpr auto
TensorGetPtr(int64_t dim) noexcept -> int64_t {
    HVX_INLINE_TOP();
    return dim;
}

/*!
 * @brief get the pointer to a flattened n dimensional array starting with the highest dimension
 */
template<typename dim_, typename... rest_dims_>
constexpr auto
TensorGetPtr(int64_t dim, rest_dims_... rest_dims) noexcept -> int64_t {
    HVX_INLINE_TOP();
    constexpr int64_t ptr = hvx::util::TensorGetRangeElms<dim_, sizeof...(rest_dims)>();
    return dim * ptr + hvx::util::TensorGetPtr<dim_>(rest_dims...);
}

/******************************************************************************************************************************************/

/*!
 * @brief Updates the pointers of all dimensions of a tensor (to flatten loops)
 */
template<typename dim_>
HVX_FORCE_INLINE constexpr auto
TensorUpdateDimPtr(hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax>& ptr_dim, int64_t iter) {
    HVX_INLINE_TOP();
    constexpr auto dim_vec_elms = dim_::dim_vec_elms;
    ptr_dim.data[5]             = iter / (dim_vec_elms[0] * dim_vec_elms[1] * dim_vec_elms[2] * dim_vec_elms[3] * dim_vec_elms[4]);
    ptr_dim.data[4]             = (iter / (dim_vec_elms[0] * dim_vec_elms[1] * dim_vec_elms[2] * dim_vec_elms[3])) % dim_vec_elms[4];
    ptr_dim.data[3]             = (iter / (dim_vec_elms[0] * dim_vec_elms[1] * dim_vec_elms[2])) % dim_vec_elms[3];
    ptr_dim.data[2]             = (iter / (dim_vec_elms[0] * dim_vec_elms[1])) % dim_vec_elms[2];
    ptr_dim.data[1]             = (iter / dim_vec_elms[0]) % dim_vec_elms[1];
    ptr_dim.data[0]             = iter % dim_vec_elms[0];
}

/******************************************************************************************************************************************/

/*!
 * @brief Updates the pointers of all dimensions of a tensor (to flatten loops)
 */
template<typename dim_>
HVX_FORCE_INLINE constexpr auto
TensorDimElmsIter(hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax>& ptr_dim, int64_t iter) {
    HVX_INLINE_TOP();
    constexpr std::array<int64_t, hvx::util::limits_e::kTensorDimMax> dims = {dim_::dim0::elms, dim_::dim1::elms, dim_::dim2::elms,
                                                                              dim_::dim3::elms, dim_::dim4::elms, dim_::dim5::elms};

    ptr_dim.data[5] = iter / (dims.at(0) * dims.at(1) * dims.at(2) * dims.at(3) * dims.at(4));
    ptr_dim.data[4] = (iter / (dims.at(0) * dims.at(1) * dims.at(2) * dims.at(3))) % dims.at(4);
    ptr_dim.data[3] = (iter / (dims.at(0) * dims.at(1) * dims.at(2))) % dims.at(3);
    ptr_dim.data[2] = (iter / (dims.at(0) * dims.at(1))) % dims.at(2);
    ptr_dim.data[1] = (iter / dims.at(0)) % dims.at(1);
    ptr_dim.data[0] = iter % dims.at(0);
}

/*!
 * @brief Computes the vector and vector element pointers out of the global pointer of a tensor
 */
template<typename dim_>
HVX_FORCE_INLINE constexpr auto
TensorDimVecElmsIter(hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax>& ptr_elms,
                     hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax>& ptr_elms_v,
                     hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax>& ptr_elms_p,
                     int64_t iter) {
    HVX_INLINE_TOP();
    constexpr std::array<int64_t, hvx::util::limits_e::kTensorDimMax> vecs = {
        dim_::dim0::vec_size, dim_::dim1::vec_size, dim_::dim2::vec_size, dim_::dim3::vec_size, dim_::dim4::vec_size, dim_::dim5::vec_size};

    hvx::util::TensorDimElmsIter<dim_>(ptr_elms, iter);
    for (int64_t i = 0; i < hvx::util::limits_e::kTensorDimMax; ++i) {
        auto v = ptr_elms.Get(i) / vecs.at(i);
        auto p = ptr_elms.Get(i) % vecs.at(i);
        ptr_elms_v.Set(v, i);
        ptr_elms_p.Set(p, i);
    }
}

/*!
 * @brief Compute pointer of a tensor
 */
template<typename dim_>
HVX_FORCE_INLINE auto
TensorPtrElms(hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax>& ptr_elms) noexcept -> int64_t {
    HVX_INLINE_TOP();
    constexpr std::array<int64_t, hvx::util::limits_e::kTensorDimMax> ptr_elms_ = {
        1,
        dim_::dim0::elms,
        dim_::dim0::elms * dim_::dim1::elms,
        dim_::dim0::elms * dim_::dim1::elms * dim_::dim2::elms,
        dim_::dim0::elms * dim_::dim1::elms * dim_::dim2::elms * dim_::dim3::elms,
        dim_::dim0::elms * dim_::dim1::elms * dim_::dim2::elms * dim_::dim3::elms * dim_::dim4::elms};

    int64_t ptr = 0;
    for (int64_t i = 0; i < hvx::util::limits_e::kTensorDimMax; ++i)
        ptr += ptr_elms.Get(i) * ptr_elms_.at(i);
    return ptr;
}

/*!
 * @brief Compute pointer for a vector of a vectorized tensor
 */
template<typename dim_>
HVX_FORCE_INLINE auto
TensorPtrElmsV(hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax>& ptr_elms_v) noexcept -> int64_t {
    HVX_INLINE_TOP();
    constexpr std::array<int64_t, hvx::util::limits_e::kTensorDimMax> ptr_elms_v_ = {
        1,
        dim_::dim0::vec_elms,
        dim_::dim0::vec_elms * dim_::dim1::vec_elms,
        dim_::dim0::vec_elms * dim_::dim1::vec_elms * dim_::dim2::vec_elms,
        dim_::dim0::vec_elms * dim_::dim1::vec_elms * dim_::dim2::vec_elms * dim_::dim3::vec_elms,
        dim_::dim0::vec_elms * dim_::dim1::vec_elms * dim_::dim2::vec_elms * dim_::dim3::vec_elms * dim_::dim4::vec_elms};

    int64_t ptr = 0;
    for (int64_t i = 0; i < hvx::util::limits_e::kTensorDimMax; ++i)
        ptr += ptr_elms_v.Get(i) * ptr_elms_v_.at(i);
    return ptr;
}

/*!
 * @brief Compute pointer for a vector element of a vectorized tensor
 */
template<typename dim_>
HVX_FORCE_INLINE auto
TensorPtrElmsP(hvx::util::vector<int64_t, hvx::util::limits_e::kTensorDimMax>& ptr_elms_p) noexcept -> int64_t {
    HVX_INLINE_TOP();
    constexpr std::array<int64_t, hvx::util::limits_e::kTensorDimMax> ptr_elms_p_ = {
        1,
        dim_::dim0::vec_size,
        dim_::dim0::vec_size * dim_::dim1::vec_size,
        dim_::dim0::vec_size * dim_::dim1::vec_size * dim_::dim2::vec_size,
        dim_::dim0::vec_size * dim_::dim1::vec_size * dim_::dim2::vec_size * dim_::dim3::vec_size,
        dim_::dim0::vec_size * dim_::dim1::vec_size * dim_::dim2::vec_size * dim_::dim3::vec_size * dim_::dim4::vec_size};

    int64_t ptr = 0;
    for (int64_t i = 0; i < hvx::util::limits_e::kTensorDimMax; ++i)
        ptr += ptr_elms_p.Get(i) * ptr_elms_p_.at(i);
    return ptr;
}

/******************************************************************************************************************************************/
} // namespace util
} // namespace hvx

#endif // HVX_UTIL_TENSOR_H_
