#pragma once
#include "dfloat.h"

namespace dynfloat {
namespace details_fixed_ln {

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)

template<std::int64_t scale>
constexpr auto
cordic_hyper_tanh_as_fixed() -> utils::array<std::int64_t, 64> {
    constexpr auto scale_value         = static_cast<double>(1LL << scale);
    utils::array<std::int64_t, 64> res = {};
    for (auto i = 0; i < 64; ++i) {
        res[i] = static_cast<std::int64_t>(constants::cordic_hyper_tanh[i] * scale_value);
    }
    return res;
}

template<std::int64_t scale, std::int64_t iterations, typename int_type_>
constexpr auto
cordic_vectoring_hyperbolic_fixed(int_type_& x_n,
                                  int_type_& y_n,
                                  int_type_& z_n,
                                  const int_type_ x0,
                                  const int_type_ y0,
                                  const int_type_ z0) -> void {
    constexpr auto hyper_tanh = cordic_hyper_tanh_as_fixed<scale>();

    int_type_ x = x0, y = y0, z = z0;
    for (auto i_index = 1; i_index < iterations; ++i_index) {
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
#pragma HLS UNROLL
#endif
        const auto i           = constants::cordic_hyper_indexes[i_index]; // NOLINT
        const auto angle       = static_cast<int_type_>(hyper_tanh[i]);    // NOLINT
        const auto neg_y       = y <= 0;
        const auto neg_y_new_x = x + (y >> i);
        const auto neg_y_new_y = y + (x >> i);
        const auto neg_y_new_z = z - angle;
        const auto pos_y_new_x = x - (y >> i);
        const auto pos_y_new_y = y - (x >> i);
        const auto pos_y_new_z = z + angle;
        x                      = neg_y ? neg_y_new_x : pos_y_new_x;
        y                      = neg_y ? neg_y_new_y : pos_y_new_y;
        z                      = neg_y ? neg_y_new_z : pos_y_new_z;
    }

    x_n = x;
    y_n = y;
    z_n = z;
}
} // namespace details_fixed_ln

template<std::int64_t scale, std::int64_t iterations, typename int_type_>
DYNFLOAT_FORCE_INLINE static constexpr auto
fixed_ln_limited_range(int_type_ x) noexcept -> std::int64_t {
    const int_type_ x0     = x + (int_type_{1} << scale);
    const int_type_ y0     = x - (int_type_{1} << scale);
    constexpr int_type_ z0 = 0;
    int_type_ x_n{}, y_n{}, z_n{};
    details_fixed_ln::cordic_vectoring_hyperbolic_fixed<scale, iterations>(x_n, y_n, z_n, x0, y0, z0);

    const auto result = z_n << 1;
    return result;
}

template<std::int64_t scale, std::int64_t iterations, typename int_type_>
DYNFLOAT_FORCE_INLINE static constexpr auto
fixed_sqrt_limited_range(int_type_ x) noexcept -> std::int64_t {
    constexpr auto after_scale       = 1. / (2. * constants::cordic_hyper_F);
    constexpr auto after_scale_fixed = static_cast<int_type_>(after_scale * (1LL << scale));
    const auto x0                    = static_cast<int_type_>(x + (1LL << scale));
    const auto y0                    = static_cast<int_type_>(x - (1LL << scale));
    constexpr int_type_ z0           = 0;
    int_type_ x_n{}, y_n{}, z_n{};
    details_fixed_ln::cordic_vectoring_hyperbolic_fixed<scale, iterations>(x_n, y_n, z_n, x0, y0, z0);
    constexpr auto res_mask = utils::make_mask<scale + 2>(true);
    const auto result       = (static_cast<std::uint64_t>(x_n & res_mask) * after_scale_fixed) >> scale;
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
#pragma HLS bind_op variable = result op = mul impl = fabric
#endif
    return result;
}

// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
} // namespace dynfloat
