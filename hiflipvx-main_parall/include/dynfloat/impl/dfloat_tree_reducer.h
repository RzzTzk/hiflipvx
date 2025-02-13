#pragma once
#include "dfloat.h"

namespace dynfloat {
template<std::size_t width_, typename data_type_, typename operation_>
struct tree_reducer {
    using operation             = operation_;
    using dfloat_type           = data_type_;
    static constexpr auto width = width_;
    static constexpr auto depth = utils::lg2(width) + 1;
    operation op                = {};
    utils::array<utils::array<data_type_, width>, depth> regs{};

    explicit constexpr tree_reducer(operation op = {}): op{op} {
    }

    constexpr auto clear() noexcept {
        regs = {};
    }

    constexpr auto step(const data_type_ (&vals)[width]) noexcept -> void // NOLINT
    {
        utils::array<data_type_, width> res{};
        for (std::int64_t col = 0; col < width; ++col) {
            res[col] = vals[col];
        }
        return this->step(res);
    }

    constexpr auto step(const utils::array<data_type_, width> vals) noexcept -> void {
        for (std::int64_t row = 0, col_size = 1; row < depth - 1; ++row, col_size <<= 1) {
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
#pragma HLS unroll
#endif
            for (std::int64_t col = 0; col < col_size; ++col) {
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
#pragma HLS unroll
#endif
                regs[row][col] = op(regs[row + 1][col * 2], regs[row + 1][col * 2 + 1]);
            }
        }
        regs[regs.size() - 1] = vals;
    }

    constexpr auto step() noexcept -> void {
        return this->step(utils::array<data_type_, width>{});
    }

    constexpr auto result() const noexcept -> dfloat_type {
        return regs[0][0];
    }
};

} // namespace dynfloat
