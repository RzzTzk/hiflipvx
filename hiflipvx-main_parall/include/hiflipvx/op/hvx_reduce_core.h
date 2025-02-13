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
 * @file    hvx_reduce_core.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_REDUCE_CORE_H_
#define HVX_REDUCE_CORE_H_

#include "impl/hvx_reduce_dfixed.h"
#include "impl/hvx_reduce_dfloat.h"

namespace hvx {
namespace red {
/******************************************************************************************************************************************/

/*!
 * @brief Calculates for a buffer of a certain dimension if it is needed
 */
template<typename reduce_, int64_t dim_num_, int64_t id_>
struct UseBuffer {
    static constexpr bool value =
        (dim_num_ >= id_ + 1) && reduce_::dims[id_] && (!reduce_::dims[id_ - 1] || UseBuffer<reduce_, dim_num_, id_ - 1>::value);
};

/*!
 * @brief Calculates for a buffer of a certain dimension if it is needed (Termination condition)
 */
template<typename reduce_, int64_t dim_num_>
struct UseBuffer<reduce_, dim_num_, 0> {
    static constexpr bool value = false;
};

/*!
 * @brief Calculates for each buffer inside an array if it is needed
 */
template<typename reduce_, int64_t dim_num_, std::size_t... id_>
constexpr std::array<bool, sizeof...(id_)>
UseBufferCalc(std::index_sequence<id_...>) { // NOLINT
    return {UseBuffer<reduce_, dim_num_, id_>::value...};
}

/*!
 * @brief Calculates the buffer size of a single buffer (it is size 1 if not needed)
 */
template<typename dst_dim, int64_t dim_num_, std::size_t idx>
constexpr auto
BufSizeCalc(const std::array<bool, dim_num_>& use_buffer) noexcept -> int64_t {
    return !use_buffer[idx] ? 1 : hvx::util::TensorGetRangeElms<dst_dim, static_cast<int64_t>(idx)>();
}

/*!
 * @brief Calculates the buffer size of each buffer inside of an array (it is size 1 if not needed)
 */
template<typename dst_dim, int64_t dim_num_, std::size_t... idx_>
constexpr auto
BufSizeArray(const std::array<bool, dim_num_>& use_buffer, std::index_sequence<idx_...>) {
    return std::array<int64_t, sizeof...(idx_)>{{hvx::red::BufSizeCalc<dst_dim, dim_num_, idx_>(use_buffer)...}};
}

/******************************************************************************************************************************************/

/*!
 * @brief All compile time parameters and checks for the elementwise operations
 */
template<typename src_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename src_dim_                      = hvx::util::TensorParam<1, hvx::util::VectorParam<1, 1>>,
         typename reduce_                       = hvx::util::ReduceParam<false, false, false, false, false, false>,
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kTrunc,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact,
         hvx::util::reduce_e op_type_           = hvx::util::reduce_e::None>
struct Reduce {
    // tells which dimensions need to be reduced
    using reduce = reduce_;

    // dimensions (dst dimension changes to size one if it should be reduced)
    using src_dim = src_dim_;
    using dst_dim =
        hvx::util::TensorParam<src_dim::dim_num,
                               std::conditional_t<(reduce::dims[0] == true), hvx::util::VectorParam<1, 1>, typename src_dim::dim0>,
                               std::conditional_t<(reduce::dims[1] == true), hvx::util::VectorParam<1, 1>, typename src_dim::dim1>,
                               std::conditional_t<(reduce::dims[2] == true), hvx::util::VectorParam<1, 1>, typename src_dim::dim2>,
                               std::conditional_t<(reduce::dims[3] == true), hvx::util::VectorParam<1, 1>, typename src_dim::dim3>,
                               std::conditional_t<(reduce::dims[4] == true), hvx::util::VectorParam<1, 1>, typename src_dim::dim4>,
                               std::conditional_t<(reduce::dims[5] == true), hvx::util::VectorParam<1, 1>, typename src_dim::dim5>>;

    // parameters (TODO: vectorization and double buffering)
    static constexpr auto dim_num        = src_dim::dim_num;
    static constexpr auto overflow_type  = overflow_type_;
    static constexpr auto underflow_type = underflow_type_;
    static constexpr auto exec_type      = exec_type_;
    static constexpr auto op_type        = op_type_;
    static constexpr auto extra_bits     = hvx::red::impl::ExtraBitsCalculate<src_dim, reduce, op_type_, dim_num>();
    static constexpr auto use_buffer     = hvx::red::UseBufferCalc<reduce, dim_num>(std::make_index_sequence<dim_num>{});
    static constexpr auto buf_size       = hvx::red::BufSizeArray<dst_dim, dim_num>(use_buffer, std::make_index_sequence<dim_num>{});

    // data types
    using src_type = src_type_;
    using dst_type = dst_type_;
    using buf_type = typename hvx::util::def_buf_type<src_type, extra_bits>::type;
    using src_vec  = hvx::util::vector<src_type, src_dim::vec_size>;
    using dst_vec  = hvx::util::vector<dst_type, dst_dim::vec_size>;
    using src_port = src_vec;
    using dst_port = dst_vec;

    // data types for globale registers arrays
    using data_arr_type = std::array<buf_type, dim_num + 1>;
    using cond_arr_type = std::array<bool, dim_num + 1>;
    using bufs_ptr_type = std::array<int64_t, dim_num>;
    using data_ptr_type = std::array<int64_t, dim_num>;

    //// constructor (not used)
    // constexpr Reduce() {
    // }
};

/******************************************************************************************************************************************/

/*!
 * @brief Creates a buffer of a certain size
 */
template<typename param_, std::size_t... ids_>
HVX_FORCE_INLINE constexpr auto
BuffersArrayCreate(std::index_sequence<ids_...>) {
    return std::make_tuple(std::array<typename param_::buf_type, param_::buf_size[ids_]>{}...);
}

/*!
 * @brief Contains an array of buffers with different sizes
 */
template<typename param_, std::size_t size_>
struct BuffersArray {
    static HVX_FORCE_INLINE auto& Compute() { // NOLINT
        static auto buffers = hvx::red::BuffersArrayCreate<param_>(std::make_index_sequence<size_>{});
        return buffers;
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Computes the reduction function for a certain dimension (dim_id_)
 */
template<typename param_, int64_t dim_id_, typename bufs_type_>
HVX_FORCE_INLINE constexpr auto
ReduceCompSingle(typename param_::data_arr_type& data,
                 typename param_::data_ptr_type& data_ptr,
                 typename param_::cond_arr_type& cond,
                 bufs_type_& bufs,
                 typename param_::bufs_ptr_type& bufs_ptr) noexcept -> void {
    HVX_INLINE_TOP();

    // get the buffer of the current dimension (buffer will be size one if not needed)
    auto buf_data = std::get<dim_id_>(bufs).data();

    // check if this dimension needs to be reduced
    if (param_::reduce::dims[dim_id_] == true) {
        // check if condition for this dimension to operate is met
        if (cond.at(dim_id_)) {
            // needed elements for the reduction operator
            auto& data_curr = data.at(dim_id_);
            auto& data_next = data.at(dim_id_ + 1);
            auto& data_ptr_ = data_ptr.at(dim_id_);
            auto& buf_data_ = (param_::use_buffer.at(dim_id_) == false) ? data.at(dim_id_ + 1) : buf_data[bufs_ptr.at(dim_id_)]; // NOLINT

            // compute the reduction operator
            auto result = hvx::red::impl::ReduceUpdate<param_, param_::src_dim::dims[dim_id_]>(data_curr, data_next, data_ptr_, buf_data_);

            // update data for next dimension
            data.at(dim_id_ + 1) = result;

            // buffer data if needed (TODO: double buffering)
            if (param_::use_buffer.at(dim_id_) == true) {
                buf_data[bufs_ptr.at(dim_id_)] = result; // NOLINT
                bufs_ptr.at(dim_id_) = (bufs_ptr.at(dim_id_) >= param_::buf_size.at(dim_id_) - 1) ? (0) : (bufs_ptr.at(dim_id_) + 1);
            }
        }

        // update condition for next dimension
        cond.at(dim_id_ + 1) = cond.at(dim_id_) && (data_ptr.at(dim_id_) == param_::src_dim::dims[dim_id_] - 1);
    }

    // otherwise just forward data and condition to next dimension
    else {
        data.at(dim_id_ + 1) = data.at(dim_id_);
        cond.at(dim_id_ + 1) = cond.at(dim_id_);
    }
}

/*!
 * @brief Computes the reduction function for multiple dimensions (template recursion)
 */
template<typename param_, typename bufs_type_, int64_t id_, int64_t max_id_>
struct ReduceCompMultiple {
    static HVX_FORCE_INLINE constexpr void Compute(typename param_::data_arr_type& data,
                                                   typename param_::data_ptr_type& data_ptr,
                                                   typename param_::cond_arr_type& cond,
                                                   bufs_type_& bufs,
                                                   typename param_::bufs_ptr_type& bufs_ptr) {
        hvx::red::ReduceCompSingle<param_, id_>(data, data_ptr, cond, bufs, bufs_ptr);
        ReduceCompMultiple<param_, bufs_type_, id_ + 1, max_id_>::Compute(data, data_ptr, cond, bufs, bufs_ptr);
    }
};

/*!
 * @brief Computes the reduction function for multiple dimensions (termination condition)
 */
template<typename param_, typename bufs_type_, int64_t max_id_>
struct ReduceCompMultiple<param_, bufs_type_, max_id_, max_id_> {
    static HVX_FORCE_INLINE constexpr void Compute(typename param_::data_arr_type& data,
                                                   typename param_::data_ptr_type& data_ptr,
                                                   typename param_::cond_arr_type& cond,
                                                   bufs_type_& bufs,
                                                   typename param_::bufs_ptr_type& bufs_ptr) {
        hvx::red::ReduceCompSingle<param_, max_id_>(data, data_ptr, cond, bufs, bufs_ptr);
    }
};

/******************************************************************************************************************************************/

/*!
 * @brief Different pipeline stages of the loop body
 */
template<typename param_, typename bufs_type_>
HVX_FORCE_INLINE constexpr auto
ReduceLoopBody(typename param_::src_port* src,
               typename param_::dst_port* dst,
               int64_t& src_ptr,
               int64_t& dst_ptr,
               typename param_::data_arr_type& data,
               typename param_::data_ptr_type& data_ptr,
               typename param_::cond_arr_type& cond,
               bufs_type_& bufs,
               typename param_::bufs_ptr_type& bufs_ptr) noexcept -> void {
    // read next src vector
    typename param_::src_vec src_data{};
    hvx::util::StreamReadData<>(src, src_data, src_ptr, true);
    hvx::red::impl::ReduceRead<param_>(src_data, data[0]);

    // Computes the reduction function for multiple dimensions
    ReduceCompMultiple<param_, bufs_type_, 0, param_::dim_num - 1>::Compute(data, data_ptr, cond, bufs, bufs_ptr);

    // write next dst vector
    typename param_::dst_vec dst_data{};
    hvx::red::impl::ReduceWrite<param_>(data[param_::dim_num], dst_data);
    hvx::util::StreamWriteData<>(dst, dst_data, dst_ptr, cond[param_::dim_num]);
}

/*!
 * @brief Creates the nested loops needed to iterate over all tensor dimensions (template recursion)
 */
template<typename param_, typename bufs_type, int dim>
struct ReduceLoop {
    static HVX_FORCE_INLINE constexpr void Compute(typename param_::src_port* src,
                                                   typename param_::dst_port* dst,
                                                   int64_t& src_ptr,
                                                   int64_t& dst_ptr,
                                                   typename param_::data_arr_type& data,
                                                   typename param_::data_ptr_type& data_ptr,
                                                   typename param_::cond_arr_type& cond,
                                                   bufs_type& bufs,
                                                   typename param_::bufs_ptr_type& bufs_ptr) noexcept {
        for (data_ptr[dim - 1] = 0; data_ptr[dim - 1] < param_::src_dim::dims[dim - 1]; ++data_ptr[dim - 1]) {
            ReduceLoop<param_, bufs_type, dim - 1>::Compute(src, dst, src_ptr, dst_ptr, data, data_ptr, cond, bufs, bufs_ptr);
        }
    }
};

/*!
 * @brief Creates the nested loops needed to iterate over all tensor dimensions (termination condition)
 */
template<typename param_, typename bufs_type>
struct ReduceLoop<param_, bufs_type, 1> {
    static HVX_FORCE_INLINE constexpr void Compute(typename param_::src_port* src,
                                                   typename param_::dst_port* dst,
                                                   int64_t& src_ptr,
                                                   int64_t& dst_ptr,
                                                   typename param_::data_arr_type& data,
                                                   typename param_::data_ptr_type& data_ptr,
                                                   typename param_::cond_arr_type& cond,
                                                   bufs_type& bufs,
                                                   typename param_::bufs_ptr_type& bufs_ptr) noexcept {
        for (data_ptr[0] = 0; data_ptr[0] < param_::src_dim::dims[0]; ++data_ptr[0]) {
            HVX_PIPELINE_ON(1, frp);
            hvx::red::ReduceLoopBody<param_>(src, dst, src_ptr, dst_ptr, data, data_ptr, cond, bufs, bufs_ptr);
        }
    }
};

/*!
 * @brief Top function of all HW reduce operations
 */
template<typename param_>
HVX_FORCE_INLINE auto
ReduceTop(typename param_::src_port* src, typename param_::dst_port* dst) noexcept -> void {
    // compile time checks
    hvx::util::TensorVerifyIfVecSizeIs1<typename param_::src_dim, true, true, true, true, true, true>();
    hvx::red::impl::VerifyDataType<typename param_::src_type, typename param_::dst_type>();

    // buffers and registers
    auto& bufs = hvx::red::BuffersArray<param_, param_::dim_num>::Compute();
    typename param_::data_arr_type data{};
    typename param_::cond_arr_type cond{true}; // first condition is always true
    typename param_::bufs_ptr_type bufs_ptr{};
    typename param_::data_ptr_type data_ptr{};

    // iterates through the tensor (with dim0 as inner most loop)
    int64_t src_ptr = 0, dst_ptr = 0;
    ReduceLoop<param_, decltype(bufs), param_::dim_num>::Compute(src, dst, src_ptr, dst_ptr, data, data_ptr, cond, bufs, bufs_ptr);

    // verifies if src ptr and dst ptr have accessed all elements
    hvx::util::StreamSignalVerify<typename param_::src_dim, typename param_::dst_dim>(src_ptr, dst_ptr);
}

/******************************************************************************************************************************************/
} // namespace red
} // namespace hvx

#endif // HVX_REDUCE_CORE_H_
