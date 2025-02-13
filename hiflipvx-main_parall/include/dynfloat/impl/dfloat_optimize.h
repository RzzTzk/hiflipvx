#pragma once
#include "dfloat.h"

namespace dynfloat {
template<std::int64_t exp_bits_, std::int64_t man_bits_, std::int64_t num_, std::int64_t dnm_, memory_type mem_type>
struct optimize_for_memory<dfloat<exp_bits_, man_bits_>, std::ratio<num_, dnm_>, mem_type> {
    struct optimize_res {
        std::int64_t exp, man;
    };

    static constexpr auto xilinx_optimize() noexcept -> optimize_res {
        constexpr auto lowest_bits_possible = exp_bits_ + man_bits_ * num_ / dnm_;

        // find first memory size below needed bits
        for (const auto size: xilinx::bram_sizes) {
            // found a memory size that fits
            if (lowest_bits_possible <= size) {
                return {exp_bits_, utils::min(size - exp_bits_, man_bits_)};
            }
        }
        return {exp_bits_, man_bits_};
    }

    static constexpr auto optimize() noexcept -> optimize_res {
        switch (mem_type) {
            case memory_type::xilinx_bram: {
                return xilinx_optimize();
            }
            case memory_type::tool_def: {
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
                return xilinx_optimize();
#else
                return {exp_bits_, man_bits_};
#endif
            }
        }
        return {exp_bits_, man_bits_};
    }

    using type = dfloat<optimize().exp, optimize().man>;
};

template<std::int64_t exp_bits_, std::int64_t man_bits_, std::int64_t num_, std::int64_t dnm_, dsp_type dsp_type>
struct optimize_for_dsp<dfloat<exp_bits_, man_bits_>, std::ratio<num_, dnm_>, dsp_type> {
    struct optimize_res {
        std::int64_t exp, man;
    };

    static constexpr auto xilinx_optimize() noexcept -> optimize_res {
        constexpr auto lowest_dman_bits_possible = (1 + man_bits_) * num_ / dnm_;

        // find first size below needed bits
        for (const auto size: xilinx::dsp_input_sizes) {
            // found a size that fits
            if (lowest_dman_bits_possible <= size) {
                return {exp_bits_, utils::min(size - 1, man_bits_)};
            }
        }
        return {exp_bits_, man_bits_};
    }

    static constexpr auto optimize() noexcept -> optimize_res {
        switch (dsp_type) {
            case dsp_type::xilinx_dsp18: {
                return xilinx_optimize();
            }
            case dsp_type::tool_def: {
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
                return xilinx_optimize();
#else
                return {exp_bits_, man_bits_};
#endif
            }
        }
        return {exp_bits_, man_bits_};
    }

    using type = dfloat<optimize().exp, optimize().man>;
};

} // namespace dynfloat
