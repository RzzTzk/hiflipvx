#include "hvx_rnn_library.h"
// TODO: You can add the auto-generated values here
// TODO: do not forget that the weight matrix is transposed when you use the conv function
// TODO: [next phase] I guess that it makes sense to generate 1 RNN class (struct). need to see how to make it as user friendly as possible
// TODO: [next phase] also use other data type
// TODO: [next phase] also work with vectorization: In this case we "might" need 1 reshape function before and after each matmul layer

//
using rnn_hid_type  = hvx::dfixed<float, 15>;
using rnn_wgts_type = hvx::dfixed<float, 15>;
using rnn_bias_type = hvx::dfixed<float, 15>;

//
using rnn_hid_src_v = hvx::vector_param<32, 1>;
using rnn_hid_dst_v = hvx::vector_param<32, 1>;
using rnn_batch_v   = hvx::vector_param<32, 1>;
using rnn_seq_v     = hvx::vector_param<32, 1>;

//
using rnn_src_dim = hvx::tensor_param<3, rnn_hid_src_v, rnn_batch_v, rnn_seq_v>; // h_size,bs, L
using rnn_dst_dim = hvx::tensor_param<3, rnn_hid_dst_v, rnn_batch_v, rnn_seq_v>; // h_size,bs, L

//
constexpr bool rnn_buf       = false;
constexpr auto rnn_overflow  = hvx::overflow_e::kSaturate;
constexpr auto rnn_underflow = hvx::underflow_e::kFloor;
constexpr auto rnn_exec      = hvx::execution_e::kExact;
constexpr auto rnn_axis      = hvx::axis_e::kEof;
