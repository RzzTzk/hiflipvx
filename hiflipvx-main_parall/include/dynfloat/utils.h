#pragma once
#include "deps.h"

namespace dynfloat {
namespace utils {
template<std::int64_t bits_>
struct fit_int {
    static constexpr bool fits_int8  = bits_ <= 8;
    static constexpr bool fits_int16 = bits_ <= 16;
    static constexpr bool fits_int32 = bits_ <= 32;
    static constexpr bool fits_int64 = bits_ <= 64;
    static constexpr bool fits       = fits_int64;
    using type                       = std::conditional_t<fits_int32, int32_t, int64_t>;
};

template<std::int64_t bits_>
using fit_int_t = typename fit_int<bits_>::type;

template<std::int64_t bits_>
static constexpr bool can_fit_int = fit_int<bits_>::fits;

template<std::int64_t bits_>
struct min_int {
    static constexpr bool fits_int8  = bits_ <= 8;
    static constexpr bool fits_int16 = bits_ <= 16;
    static constexpr bool fits_int32 = bits_ <= 32;
    static constexpr bool fits_int64 = bits_ <= 64;
    static constexpr bool fits       = fits_int64;
    using type =
        std::conditional_t<fits_int32, std::conditional_t<fits_int16, std::conditional_t<fits_int8, int8_t, int16_t>, int32_t>, int64_t>;
};

template<std::int64_t bits_>
using min_int_t = typename min_int<bits_>::type;

template<typename integer_>
static constexpr bool is_unsigned_integer = std::is_integral<integer_>::value && std::is_unsigned<integer_>::value;

template<typename integer_>
static constexpr bool is_signed_integer = std::is_integral<integer_>::value && std::is_signed<integer_>::value;

template<std::int64_t bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
make_mask(bool value) noexcept -> std::make_unsigned_t<fit_int_t<bits_>> {
    using res_t = std::make_unsigned_t<fit_int_t<bits_>>;
    static_assert(std ::numeric_limits<res_t>::digits >= bits_, "Invalid bit count!");
    return static_cast<res_t>(static_cast<res_t>(res_t{} - res_t{value}) >>
                              static_cast<std::uint8_t>(std ::numeric_limits<res_t>::digits - bits_));
}

template<typename T>
constexpr auto
sign(T x) noexcept -> std::int32_t {
    return x >= 0 ? 1 : -1;
}

template<std::int64_t idx_, typename number_type_>
DYNFLOAT_FORCE_INLINE static constexpr auto
bit_select(const number_type_ n) -> bool {
    return ((n >> idx_) & 0x1) != 0;
}

template<typename number_type_>
DYNFLOAT_FORCE_INLINE static constexpr auto
bit_select(const number_type_ n, std::int64_t idx_) -> bool {
    return ((n >> idx_) & 0x1) != 0;
}

template<typename number1_type_, typename number2_type_>
DYNFLOAT_FORCE_INLINE static constexpr auto
non_overflowing_sub(const number1_type_ x, const number2_type_ y) -> auto {
    return x < y ? decltype(x - y){} : x - y;
}

#if defined(_MSC_VER)
template<typename integer_>
DYNFLOAT_FORCE_INLINE static constexpr auto
bit_width(const integer_ n) noexcept -> std::uint8_t {
    using unsigned_int = std::make_unsigned_t<integer_>;
    auto unsigned_n    = static_cast<unsigned_int>(n);
#if _HAS_CXX20
    return static_cast<std::uint8_t>(std::bit_width(unsigned_n));
#else
    std::uint8_t count = 0;
    while (unsigned_n != 0) {
        count++;
        unsigned_n >>= 1;
    }
    return count;
#endif
}
#elif defined(__GNUC__) || defined(__clang__)
template<typename integer_>
DYNFLOAT_FORCE_INLINE static constexpr auto
bit_width(const integer_ n) noexcept -> std::uint8_t {
    using unsigned_int    = std::make_unsigned_t<integer_>;
    const auto unsigned_n = static_cast<unsigned_int>(n);

#if __cplusplus > 202000
    return static_cast<std::uint8_t>(std::bit_width(unsigned_n));
#else
    if (!n)
        return 0;

    if (std::is_same<unsigned_int, std::uint8_t>::value) {
        const auto bit_repr      = static_cast<std::uint32_t>(unsigned_n);
        const auto leading_zeros = static_cast<std::uint8_t>(__builtin_clz(bit_repr) - 24);
        return static_cast<std::uint8_t>(std::numeric_limits<unsigned_int>::digits) - leading_zeros;
    } else if (std::is_same<unsigned_int, std::uint16_t>::value) {
        const auto bit_repr      = static_cast<std::uint32_t>(unsigned_n);
        const auto leading_zeros = static_cast<std::uint8_t>(__builtin_clz(bit_repr) - 16);
        return static_cast<std::uint8_t>(std::numeric_limits<unsigned_int>::digits) - leading_zeros;
    } else if (std::is_same<unsigned_int, std::uint32_t>::value) {
        const auto leading_zeros = static_cast<std::uint8_t>(__builtin_clz(static_cast<std::uint32_t>(n)));
        return static_cast<std::uint8_t>(std::numeric_limits<unsigned_int>::digits) - leading_zeros;
    } else if (std::is_same<unsigned_int, std::uint64_t>::value) {
        const auto leading_zeros = static_cast<std::uint8_t>(__builtin_clzll(static_cast<std::uint64_t>(n)));
        return static_cast<std::uint8_t>(std::numeric_limits<unsigned_int>::digits) - leading_zeros;
    } else {
        return 0;
    }
#endif
}
#else
static_assert(false, "Provide bit_width() implementation!");
#endif

DYNFLOAT_FORCE_INLINE static constexpr auto
int_repr_sqrt_approx_factor(const std::int64_t exp_bits, const std::int64_t man_bits) noexcept -> double {
    const auto zero_exp  = (1u << (exp_bits - 1u)) - 1u;
    const auto L         = static_cast<double>(uint64_t{1} << man_bits);
    const auto b         = static_cast<double>(zero_exp);
    constexpr auto sigma = 0.0688; // balanced positive and negative error
    const auto res       = L * b - L * sigma;
    return res;
}

DYNFLOAT_FORCE_INLINE static constexpr auto
int_repr_approx_factor(const std::int64_t exp_bits, const std::int64_t man_bits) noexcept -> double {
    const auto zero_exp = (1u << (exp_bits - 1u)) - 1u;
    const auto L        = static_cast<double>(uint64_t{1} << man_bits);
    const auto b        = static_cast<double>(zero_exp);
    const auto sigma    = 0.05; // balanced positive and negative error
    const auto res      = L * b - L * sigma;
    return res;
}

DYNFLOAT_FORCE_INLINE static constexpr auto
int_repr_mul_approx_factor(const std::int64_t exp_bits, const std::int64_t man_bits) noexcept -> double {
    const auto zero_exp  = (1u << (exp_bits - 1u)) - 1u;
    const auto L         = static_cast<double>(uint64_t{1} << man_bits);
    const auto b         = static_cast<double>(zero_exp);
    constexpr auto sigma = 0.0688; // balanced positive and negative error
    const auto res       = L * b - L * sigma;
    return res;
}

DYNFLOAT_FORCE_INLINE static constexpr auto
int_repr_exponent_approx_factor(const std::int64_t exp_bits, const std::int64_t man_bits) noexcept -> double {
    const auto zero_exp  = (1u << (exp_bits - 1u)) - 1u;
    const auto L         = static_cast<double>(uint64_t{1} << man_bits);
    const auto b         = static_cast<double>(zero_exp);
    constexpr auto sigma = 0.04368; // balanced positive and negative error
    const auto res       = L * b - L * sigma;
    return res;
}

DYNFLOAT_FORCE_INLINE static constexpr auto
int_repr_lg2_approx_factor(const std::int64_t exp_bits) noexcept -> double {
    const auto zero_exp  = (1u << (exp_bits - 1u)) - 1u;
    const auto b         = static_cast<double>(zero_exp);
    constexpr auto sigma = 0.04368; // balanced positive and negative error
    const auto res       = b - sigma;
    return res;
}

template<typename shifted_type_, typename shift_type_>
DYNFLOAT_FORCE_INLINE static constexpr auto
safe_rshift(const shifted_type_ val, const shift_type_ shift_amount) noexcept -> auto {
    const auto signed_shift = static_cast<std::make_signed_t<shift_type_>>(shift_amount);
    return signed_shift < 0 ? val << (-signed_shift) : val >> signed_shift; // NOLINT
}

template<typename shifted_type_, typename shift_type_>
DYNFLOAT_FORCE_INLINE static constexpr auto
safe_lshift(const shifted_type_ val, const shift_type_ shift_amount) noexcept -> auto {
    const auto signed_shift = static_cast<std::make_signed_t<shift_type_>>(shift_amount);
    return signed_shift < 0 ? val >> (-signed_shift) : val << signed_shift;
}

template<typename input_>
DYNFLOAT_FORCE_INLINE static constexpr auto
pow2(const input_ val, std::int64_t power) noexcept -> input_ {
    if (power >= 0) {
        auto res = static_cast<input_>(1);
        while (power--)
            res *= val;
        return res;
    } else {
        auto res = static_cast<input_>(1);
        while (power++)
            res /= val;
        return res;
    }
}

template<typename input_>
DYNFLOAT_FORCE_INLINE static constexpr auto
abs(input_ i) noexcept -> input_ {
    return i > 0 ? i : -i;
}

template<typename lhs_, typename rhs_>
DYNFLOAT_FORCE_INLINE static constexpr auto
max(lhs_ lhs, rhs_ rhs) noexcept -> auto {
    using c_ = std::common_type_t<lhs_, rhs_>;
    return std::max(static_cast<c_>(lhs), static_cast<c_>(rhs));
}

template<typename lhs_, typename rhs_>
DYNFLOAT_FORCE_INLINE static constexpr auto
min(lhs_ lhs, rhs_ rhs) noexcept -> auto {
    using c_ = std::common_type_t<lhs_, rhs_>;
    return std::min(static_cast<c_>(lhs), static_cast<c_>(rhs));
}

template<typename enum_, std::enable_if_t<std::is_enum<enum_>::value, bool> = true>
DYNFLOAT_FORCE_INLINE static constexpr auto
max(enum_ lhs, enum_ rhs) noexcept -> auto {
    using u_ = std::underlying_type_t<enum_>;
    return static_cast<enum_>(std::max(static_cast<u_>(lhs), static_cast<u_>(rhs)));
}

template<typename enum_, std::enable_if_t<std::is_enum<enum_>::value, bool> = true>
DYNFLOAT_FORCE_INLINE static constexpr auto
min(enum_ lhs, enum_ rhs) noexcept -> auto {
    using u_ = std::underlying_type_t<enum_>;
    return static_cast<enum_>(std::min(static_cast<u_>(lhs), static_cast<u_>(rhs)));
}

template<std::int64_t shift_amount_, typename input_>
DYNFLOAT_FORCE_INLINE static constexpr auto
float_lshift(input_ val) noexcept -> input_ {
    constexpr auto abs_shift_amount = abs(shift_amount_);
    constexpr auto iteration_count  = abs_shift_amount / 63;
    if (shift_amount_ > 0) {
        for (std::int64_t shifted = 0; shifted < iteration_count * 63; shifted += 63) {
            constexpr auto factor = static_cast<input_>(1ULL << 63);
            val *= factor;
        }
        constexpr auto factor = static_cast<input_>(1ULL << (abs_shift_amount % 63));
        val *= factor;
        return val;
    }
    for (std::int64_t shifted = 0; shifted < iteration_count * 63; shifted += 63) {
        constexpr auto factor = static_cast<input_>(1ULL << 63);
        val /= factor;
    }
    constexpr auto factor = static_cast<input_>(1ULL << (abs_shift_amount % 63));
    val /= factor;
    return val;
}

template<typename type_, std::size_t size_>
struct array {
    type_ data_[size_]; // NOLINT

    constexpr auto operator[](std::ptrdiff_t idx) noexcept -> auto& {
        return data_[idx]; // NOLINT
    }

    constexpr auto operator[](std::ptrdiff_t idx) const noexcept -> const auto& {
        return data_[idx]; // NOLINT
    }

    static constexpr auto size() noexcept {
        return size_;
    }

    constexpr auto begin() noexcept {
        return std::begin(data_);
    }

    constexpr auto end() noexcept {
        return std::end(data_);
    }

    constexpr auto begin() const noexcept {
        return std::begin(data_);
    }

    constexpr auto end() const noexcept {
        return std::end(data_);
    }
};

template<typename enum_type_>
constexpr auto
enum_to_num(enum_type_ e) {
    return static_cast<std::underlying_type_t<enum_type_>>(e);
}

namespace details_fallback_sqrt {
template<std::size_t used_bits_, bool round_, typename input_type_>
constexpr auto
fallback_sqrt(const input_type_ n) -> input_type_ {
    // ROLFE, Timothy J. On a fast integer square root algorithm.
    // ACM SIGNUM Newsletter, 1987, 22. Jg., Nr. 4, S. 6-11.
    auto mask      = static_cast<input_type_>(1ULL << (used_bits_ - 2));
    auto root      = input_type_{};
    auto remainder = n;
    for (; mask; mask >>= 2) {
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
#pragma HLS UNROLL
#endif
        const auto sub = (root | mask) <= remainder;
        remainder      = sub ? remainder - (root | mask) : remainder;
        root           = sub ? (root | (mask << 1)) >> 1 : root >> 1;
    }

    if (round_)
        root += remainder > root;
    return root;
}

#if defined(DYNFLOAT_XILINX_SYNTHESIS)
template<std::size_t used_bits_, bool round_, typename input_type_>
constexpr auto
xilinx_sqrt(const input_type_ n) -> input_type_ {
    if (used_bits_ <= 32) {
        constexpr auto mask = utils::make_mask<used_bits_>(true);
        return hls::sqrt(static_cast<std::uint32_t>(n & mask));
    } else {
        return fallback_sqrt<used_bits_, round_>(n);
    }
}
#endif
} // namespace details_fallback_sqrt

template<std::size_t used_bits_,
         bool round_ = false,
         typename input_type_,
         std::enable_if_t<std::is_integral<input_type_>::value, bool> = true>
constexpr auto
sqrt(const input_type_ n) -> input_type_ {
#if defined(DYNFLOAT_XILINX_SYNTHESIS)
    return details_fallback_sqrt::xilinx_sqrt<used_bits_, round_>(n);
#else
    return details_fallback_sqrt::fallback_sqrt<used_bits_, round_>(n);
#endif
}

template<typename result_, std::size_t size, typename bound_>
constexpr auto
make_linear_space(array<result_, size>& result, const bound_ lower_bound, const bound_ upper_bound) noexcept -> array<result_, size>& {
    const auto start = static_cast<result_>(lower_bound);
    const auto end   = static_cast<result_>(upper_bound);

    if (size == 0)
        return result;
    if (size == 1) {
        result[0] = start;
        return result;
    }

    const auto delta = (end - start) / static_cast<bound_>(size - 1);
    for (std::size_t i = 0; i < size - 1; ++i) {
        result[i] = start + delta * static_cast<bound_>(i);
    }
    result[size - 1] = end;
    return result;
}

template<typename result_, std::ptrdiff_t size, typename bound_>
constexpr auto
make_linear_space(const bound_ lower_bound, const bound_ upper_bound) noexcept -> array<result_, static_cast<std::size_t>(size)> {
    array<result_, size> result{};
    make_linear_space(result, lower_bound, upper_bound);
    return result;
}

constexpr auto
lg2(std::int64_t value) noexcept -> std::int64_t {
    std::int64_t res = 0;
    while (value > 1) {
        value /= 2;
        ++res;
    }
    return res;
}

template<bool saturate_, typename t1_, typename t2_>
constexpr auto
add_saturate(const t1_ a, const t2_ b, const std::common_type_t<t1_, t2_> lower, const std::common_type_t<t1_, t2_> upper)
    -> std::common_type_t<t1_, t2_> {
    using res_t      = std::common_type_t<t1_, t2_>;
    const auto a_ext = static_cast<res_t>(a);
    const auto b_ext = static_cast<res_t>(b);

    const auto sum = a_ext + b_ext;

    if (saturate_) {
        const auto pos_overflow = b_ext > 0 && a_ext > upper - b_ext;
        const auto neg_overflow = b_ext < 0 && a_ext < lower - b_ext;
        if (pos_overflow)
            return upper;
        if (neg_overflow)
            return lower;
    }

    return sum;
}

template<bool saturate_, typename t1_, typename t2_>
constexpr auto
sub_saturate(const t1_ a, const t2_ b, const std::common_type_t<t1_, t2_> lower, const std::common_type_t<t1_, t2_> upper)
    -> std::common_type_t<t1_, t2_> {
    using res_t      = std::common_type_t<t1_, t2_>;
    const auto a_ext = static_cast<res_t>(a);
    const auto b_ext = static_cast<res_t>(b);

    const auto sub = a_ext - b_ext;

    if (saturate_) {
        const auto pos_overflow = b_ext < 0 && a_ext > upper + b_ext;
        const auto neg_overflow = b_ext > 0 && a_ext < lower + b_ext;
        if (pos_overflow)
            return upper;
        if (neg_overflow)
            return lower;
    }

    return sub;
}

} // namespace utils
} // namespace dynfloat
