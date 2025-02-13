#pragma once
#include "dfloat_abs.h"
#include "dfloat_operator_lshift.h"
#include "make_dfloat.h"

namespace dynfloat {

template<std::int64_t exp_bits_, std::int64_t man_bits_>
template<typename sign_t_, typename exp_t_, typename man_t_>
DYNFLOAT_FORCE_INLINE constexpr dfloat<exp_bits_, man_bits_>::dfloat(const sign_t_ s, const exp_t_ exp, const man_t_ man) noexcept:
    sign_(static_cast<storage_type>(s)), exp_(static_cast<storage_type>(exp)), man_(static_cast<storage_type>(man)) {
}

template<std::int64_t exp_bits_, std::int64_t man_bits_>
template<typename integer_, std::enable_if_t<utils::is_signed_integer<integer_>, bool>>
DYNFLOAT_FORCE_INLINE constexpr dfloat<exp_bits_, man_bits_>::dfloat(const integer_ i) noexcept: sign_(0), exp_(0), man_(0) {
    *this = dynfloat::make_dfloat<dfloat<exp_bits_, man_bits_>>(i);
}

template<std::int64_t exp_bits_, std::int64_t man_bits_>
template<typename integer_, std::enable_if_t<utils::is_unsigned_integer<integer_>, bool>>
DYNFLOAT_FORCE_INLINE constexpr dfloat<exp_bits_, man_bits_>::dfloat(const integer_ i) noexcept: sign_(0), exp_(0), man_(0) {
    *this = dynfloat::make_dfloat<dfloat<exp_bits_, man_bits_>>(i);
}

namespace details_dfloat_from_float {
template<std::int64_t index_,
         std::int64_t segments_count_,
         std::int64_t target_window_size_,
         std::int64_t current_scale_,
         typename float_,
         typename dfloat_>
struct converter {
    DYNFLOAT_FORCE_INLINE constexpr auto operator()(const float_ abs_input) const noexcept -> dfloat_ {
        if (abs_input > static_cast<float_>(std::numeric_limits<std::uint64_t>::max() >> 1)) {
            if (index_ + 1 != segments_count_)
                return converter<index_ + 1, segments_count_, target_window_size_, current_scale_ + target_window_size_, float_, dfloat_>{}(
                    utils::float_lshift<-target_window_size_>(abs_input));
        }

        if (static_cast<float_>(std::numeric_limits<std::uint64_t>::max() >> target_window_size_) > abs_input) {
            if (index_ + 1 != segments_count_)
                return converter<index_ + 1, segments_count_, target_window_size_, current_scale_ - target_window_size_, float_, dfloat_>{}(
                    utils::float_lshift<target_window_size_>(abs_input));
        }

        const auto c = dynfloat::make_dfloat<std_f64>(static_cast<std::uint64_t>(abs_input));
        return static_cast<dfloat_>(
            dynfloat::safe_lshift<execution<strategy::exact, rounding_behavior::to_zero, special_values::zero_and_saturation>>(
                c, current_scale_));
    }
};

template<std::int64_t index_, std::int64_t current_scale_, std::int64_t target_window_size_, typename float_, typename dfloat_>
struct converter<index_, index_, target_window_size_, current_scale_, float_, dfloat_> {
    DYNFLOAT_FORCE_INLINE constexpr auto operator()(const float_) const noexcept -> dfloat_ {
        return dfloat_{};
    }
};

template<typename float_, typename dfloat_>
DYNFLOAT_FORCE_INLINE static constexpr auto
from_float(const float_ i) noexcept -> dfloat_ {
    constexpr auto man_bits           = std::numeric_limits<float_>::digits;
    constexpr auto target_window_size = 63 - man_bits;
    constexpr auto max_iter_count =
        (std::numeric_limits<float_>::max_exponent - std::numeric_limits<float_>::min_exponent) / (2 * target_window_size) + 1;
    const auto res = details_dfloat_from_float::converter<0, max_iter_count, target_window_size, 0, float_, dfloat_>{}(utils::abs(i));
    return i < 0 ? -res : res;
}

} // namespace details_dfloat_from_float

template<std::int64_t exp_bits_, std::int64_t man_bits_>
template<typename float_, std::enable_if_t<std::is_floating_point<float_>::value, bool>>
DYNFLOAT_FORCE_INLINE constexpr dfloat<exp_bits_, man_bits_>::dfloat(const float_ i) noexcept: sign_{0}, exp_{0}, man_{0} {
    if (static_cast<bool>(i))
        *this = details_dfloat_from_float::from_float<float_, dfloat<exp_bits_, man_bits_>>(i);
}
} // namespace dynfloat
