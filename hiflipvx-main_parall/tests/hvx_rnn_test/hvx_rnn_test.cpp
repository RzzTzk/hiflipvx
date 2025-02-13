#include "hvx_rnn_config.h"

// Reader
using rnn_reader_param = hvx::RnnReaderParam<rnn_hid_type, rnn_seq_v, rnn_batch_v, rnn_hid_dst_v>;

// TODO: rather use the matmul function here
using rnn_mm_wx_param  = hvx::conv_param<rnn_hid_type,
                                        rnn_hid_type,
                                        rnn_wgts_type,
                                        rnn_bias_type,
                                        rnn_seq_v,                    // batch_v
                                        hvx::util::VectorParam<1, 1>, // src_row
                                        rnn_batch_v,                  // src_col: here x_h: bs
                                        rnn_hid_src_v,                // chnls:   here x_w: i_size
                                        rnn_hid_dst_v,                // fms_v:   here weight_w: h_size
                                        hvx::util::VectorParam<1, 1>,
                                        hvx::util::VectorParam<1, 1>,
                                        hvx::util::Array2dParam<0, 0>,
                                        hvx::util::Array2dParam<0, 0>,
                                        hvx::util::Array2dParam<1, 1>,
                                        rnn_buf,
                                        rnn_buf,
                                        rnn_overflow,
                                        rnn_underflow,
                                        rnn_exec>;
using rnn_mm_wx_stream = hvx::stream_param<typename rnn_mm_wx_param::dst_type, typename rnn_mm_wx_param::dst_dim, rnn_axis>;

// TODO: rather use the matmul function here
using rnn_mm_wh_param  = hvx::conv_param<rnn_hid_type,
                                        rnn_hid_type,
                                        rnn_wgts_type,
                                        rnn_bias_type,
                                        rnn_seq_v,                    // batch_v
                                        hvx::util::VectorParam<1, 1>, // src_row
                                        rnn_batch_v,                  // src_col ,here h_h: bs
                                        rnn_hid_dst_v,                // chnls_v, here h_w: h_size
                                        rnn_hid_dst_v,                // fms_v, here weight_w: h_size
                                        hvx::util::VectorParam<1, 1>,
                                        hvx::util::VectorParam<1, 1>,
                                        hvx::util::Array2dParam<0, 0>,
                                        hvx::util::Array2dParam<0, 0>,
                                        hvx::util::Array2dParam<1, 1>,
                                        rnn_buf,
                                        rnn_buf,
                                        rnn_overflow,
                                        rnn_underflow,
                                        rnn_exec>;
using rnn_mm_wh_stream = hvx::stream_param<typename rnn_mm_wh_param::dst_type, typename rnn_mm_wh_param::dst_dim, rnn_axis>;

// Add
using rnn_add_param = hvx::add_param<rnn_hid_type, rnn_hid_type, rnn_hid_type, rnn_dst_dim, rnn_overflow, rnn_underflow, rnn_exec>;

// Tanh
using rnn_tanh_param = hvx::tanh_param<rnn_hid_type, rnn_hid_type, rnn_dst_dim, rnn_overflow, rnn_underflow, rnn_exec>;

// Writer (with stream converter for final output)
using rnn_writer_param  = hvx::RnnWriterParam<rnn_hid_type, rnn_seq_v, rnn_batch_v, rnn_hid_dst_v>;
using rnn_writer_stream = hvx::stream_param<typename rnn_writer_param::type, typename rnn_writer_param::dim, rnn_axis>;

/******************************************************************************************************************************************/

void
RnnReader(rnn_reader_param::src_port* src, rnn_reader_param::h0_port* h0, rnn_reader_param::dst_port* dst) {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, h0, dst);
    hvx::HwRnnReader<rnn_reader_param>(src, h0, dst);
}

// TODO: We don't actually need a stream conversion here.
// However, in the past we have had a few problems with the pipeline with the conv function.
// Hopefully this problem is solved when addin hls::stream everywhere and using the vitis compiler.
auto
RnnMatMulWx(rnn_mm_wx_param::src_port* src,
            rnn_mm_wx_param::wgts_port* wgts,
            rnn_mm_wx_param::bias_port* bias,
            rnn_mm_wx_stream::port* dst) noexcept -> void {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, wgts, bias, dst);
    auto dst_fifo = hvx::HwFifo<rnn_mm_wx_param::dst_vec, rnn_mm_wx_param::dst_dim::vec_elms, 2>();
    hvx::HwConv<rnn_mm_wx_param>(src, wgts, bias, dst_fifo.data);
    hvx::HwHvxToStream<rnn_mm_wx_stream>(dst_fifo.data, *dst);
}

auto
RnnMatMulWh(rnn_mm_wh_param::src_port* src,
            rnn_mm_wh_param::wgts_port* wgts,
            rnn_mm_wh_param::bias_port* bias,
            rnn_mm_wh_stream::port* dst) noexcept -> void {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, wgts, bias, dst);
    auto dst_fifo = hvx::HwFifo<rnn_mm_wh_param::dst_vec, rnn_mm_wh_param::dst_dim::vec_elms, 2>();
    hvx::HwConv<rnn_mm_wh_param>(src, wgts, bias, dst_fifo.data);
    hvx::HwHvxToStream<rnn_mm_wh_stream>(dst_fifo.data, *dst);
}

void
RnnAddHw(rnn_add_param::src1_port* src1, rnn_add_param::src2_port* src2, rnn_add_param::dst_port* dst) {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src1, src2, dst);
    hvx::HwAdd<rnn_add_param>(src1, src2, dst);
}

void
RnnTanhHw(rnn_tanh_param::src1_port* src, rnn_tanh_param::dst_port* dst) {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, dst);
    hvx::HwTanh<rnn_tanh_param>(src, dst);
}

void
RnnWriter(rnn_writer_param::src_port* src, rnn_writer_param::hid_port* hid, rnn_writer_stream::port* dst) {
    HVX_INTERFACE_STREAM_NO_CTRL_TLP(src, hid, dst);
    auto dst_fifo = hvx::HwFifo<rnn_writer_param::vec, rnn_writer_param::dim::vec_elms, 2>();
    hvx::HwRnnWriter<rnn_writer_param>(src, hid, dst_fifo.data);
    hvx::HwHvxToStream<rnn_writer_stream>(dst_fifo.data, *dst);
}

/******************************************************************************************************************************************/

// testbench
auto
main() -> int {
    return 0;
}