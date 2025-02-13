#include "../../include/sw_test/hvx_sw_test_core.h"

namespace hvx {

/******************************************************************************************************************************************/

// TODO
template<typename src_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename dst_type_                     = hvx::util::dfixed<int16_t, 15>,
         typename wgts_type_                    = hvx::util::dfixed<int16_t, 15>,
         typename batch_v                       = hvx::util::VectorParam<1, 1>, // conv batch size
         typename rows_v                        = hvx::util::VectorParam<1, 1>, // conv src_cols
         typename src_cols_v                    = hvx::util::VectorParam<1, 1>, // conv chnls
         typename dst_cols_v                    = hvx::util::VectorParam<1, 1>, // conv fms
         int64_t buf_wgts_                      = false,
         int64_t buf_bias_                      = false,
         hvx::util::overflow_e overflow_type_   = hvx::util::overflow_e::kSaturate,
         hvx::util::underflow_e underflow_type_ = hvx::util::underflow_e::kFloor,
         hvx::util::execution_e exec_type_      = hvx::util::execution_e::kExact>
struct MatMulParam {};

// TODO
template<typename param_>
HVX_FORCE_INLINE constexpr auto
HwMatMul(typename param_::src_vec* src, typename param_::wgts_vec* wgts, typename param_::dst_vec* dst) noexcept -> void {
    HVX_INLINE_TOP();
    

    // using mat_mul = hvx::nn::ConvParam<....>;
    // hvx::nn::ConvTop<mat_mul, false>(src, wgts, nullptr, dst);
}

/******************************************************************************************************************************************/

template<typename type_,                                     //
         typename sequence_v = hvx::util::VectorParam<1, 1>, //
         typename batch_v    = hvx::util::VectorParam<1, 1>, //
         typename hidden_v   = hvx::util::VectorParam<1, 1>>   //
struct RnnWriterParam {
    using type     = type_;
    using dim      = hvx::util::TensorParam<3, hidden_v, batch_v, sequence_v>;
    using vec      = hvx::util::vector<type_, dim::vec_size>;
    using src_port = vec;
    using hid_port = vec;
    using dst_port = vec;

    // constructor (verifies the dimensions and types)
    constexpr RnnWriterParam() {
        hvx::util::TensorVerifyIfVecSizeIs1<dim, false, true, true, true, true, true>();
    }
};

template<typename param_>
HVX_FORCE_INLINE auto
HwRnnWriter(typename param_::src_port* src, typename param_::hid_port* hid, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, hid, dst);

    //
    int64_t src_ptr = 0, hid_ptr = 0, dst_ptr = 0;

    for (int64_t s = 0; s < param_::dim::dim_vec_elms[2]; ++s) {
        for (int64_t b = 0; b < param_::dim::dim_vec_elms[1]; ++b) {
            for (int64_t h = 0; h < param_::dim::dim_vec_elms[0]; ++h) {
                HVX_PIPELINE_ON(1, frp);

                //
                typename param_::vec data{};

                //
                hvx::util::StreamReadData<>(src, data, src_ptr, true);

                //
                if (s < param_::dim::dim_vec_elms[2] - 1)
                    hvx::util::StreamWriteData<>(hid, data, hid_ptr, true);
                hvx::util::StreamWriteData<>(dst, data, dst_ptr, true);
            }
        }
    }
}

/******************************************************************************************************************************************/

template<typename type_,                                     //
         typename sequence_v = hvx::util::VectorParam<1, 1>, //
         typename batch_v    = hvx::util::VectorParam<1, 1>, //
         typename hidden_v   = hvx::util::VectorParam<1, 1>>   //
struct RnnReaderParam {
    using type     = type_;
    using dim      = hvx::util::TensorParam<3, hidden_v, batch_v, sequence_v>;
    using vec      = hvx::util::vector<type_, dim::vec_size>;
    using src_port = vec;
    using h0_port  = vec;
    using dst_port = vec;

    // constructor (verifies the dimensions and types)
    constexpr RnnReaderParam() {
        hvx::util::TensorVerifyIfVecSizeIs1<dim, false, true, true, true, true, true>();
    }
};

template<typename param_>
HVX_FORCE_INLINE auto
HwRnnReader(typename param_::src_port* src, typename param_::h0_port* h0, typename param_::dst_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, h0, dst);

    //
    int64_t src_ptr = 0, h0_ptr = 0, dst_ptr = 0;

    for (int64_t s = 0; s < param_::dim::dim_vec_elms[2]; ++s) {
        for (int64_t b = 0; b < param_::dim::dim_vec_elms[1]; ++b) {
            for (int64_t h = 0; h < param_::dim::dim_vec_elms[0]; ++h) {
                HVX_PIPELINE_ON(1, frp);

                //
                typename param_::vec data{};

                //
                if (s == 0)
                    hvx::util::StreamReadData<>(h0, data, h0_ptr, true);
                else
                    hvx::util::StreamReadData<>(src, data, src_ptr, true);

                //
                hvx::util::StreamWriteData<>(dst, data, dst_ptr, true);
            }
        }
    }
}

/******************************************************************************************************************************************/

/*
template<typename wgts_type_,                             //
         typename hout_v  = hvx::util::VectorParam<1, 1>, //
         typename batch_v = hvx::util::VectorParam<1, 1>> //

struct hidden_param {
    using wgts_type = wgts_type_;
    using wgts_dim  = hvx::util::TensorParam<2, hout_v, batch_v>;
    using wgts_vec  = hvx::util::vector<wgts_type_, wgts_dim::vec_size>;
    using wgts_port = wgts_vec;
};

// HW accelerator
template<typename param_>
HVX_FORCE_INLINE auto
HwHidden(typename param_::wgts_port* src, typename param_::wgts_port* h0_init, typename param_::wgts_port* dst) noexcept -> void {
    HVX_DATAPACK_TOP(src, h0_init, dst);

    //
    static typename param_::wgts_vec h0[param_::wgts_dim::vec_elms];
    static typename param_::wgts_vec h1[param_::wgts_dim::vec_elms];

    //
    static bool rd_h0      = true;
    static bool wr_h0      = false;
    static bool rd_h0_init = false;

    //
    int64_t src_ptr  = 0;
    int64_t wgts_ptr = 0;
    int64_t dst_ptr  = 0;

    //
    for (int64_t i = 0; i < param_::wgts_dim::vec_elms; ++i) {
        HVX_PIPELINE_ON(1, frp);

        //
        typename param_::wgts_vec wgts_data{};
        hvx::util::StreamReadData<>(src, wgts_data, src_ptr, true);

        //
        if (wr_h0 == false)
            h1[i] = wgts_data;
        else
            h0[i] = wgts_data;

        //
        if (src_ptr == param_::wgts_dim::vec_elms - 1)
            wr_h0 = !wr_h0;
    }

    for (int64_t i = 0; i < param_::wgts_dim::vec_elms; ++i) {
        HVX_PIPELINE_ON(1, frp);

        //
        typename param_::wgts_vec wgts_data{};

        //
        if (rd_h0 == true) {
            if (rd_h0_init == false) {
                hvx::util::StreamReadData<>(h0_init, wgts_data, wgts_ptr, true);
                if (wgts_ptr == param_::wgts_dim::vec_elms - 1)
                    rd_h0_init = true;
            } else {
                wgts_data = h0[i];
            }
        } else {
            wgts_data = h1[i];
        }

        //
        hvx::util::StreamWriteData<>(dst, wgts_data, dst_ptr, true);

        //
        if (dst_ptr == param_::wgts_dim::vec_elms - 1)
            rd_h0 = !rd_h0;
    }
}
*/

/******************************************************************************************************************************************/
} // namespace hvx