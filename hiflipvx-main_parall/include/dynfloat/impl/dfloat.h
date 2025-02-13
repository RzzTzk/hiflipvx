#pragma once
#include "error_info.h"

// #include "dfloat_abs.h"

namespace dynfloat {
template<std::int64_t exp_bits_, std::int64_t man_bits_>
struct dfloat {
    static constexpr bool is_signed = true;

    static constexpr std::int64_t exp_bits  = exp_bits_;
    static constexpr std::int64_t man_bits  = man_bits_;
    static constexpr std::int64_t dman_bits = man_bits + 1;
    static constexpr std::int64_t sign_mask = 0x01u;
    static constexpr auto exp_mask          = utils::make_mask<exp_bits>(true);
    static constexpr auto man_mask          = utils::make_mask<man_bits>(true);
    static constexpr auto dman_mask         = utils::make_mask<dman_bits>(true);
    static constexpr bool exp_valid         = utils::can_fit_int<exp_bits + 1>;
    static constexpr bool man_valid         = utils::can_fit_int<man_bits + 1>;
    static constexpr bool int_repr_valid    = utils::can_fit_int<exp_bits + man_bits>;
    static constexpr std::int64_t zero_exp  = (1LL << (exp_bits - 1)) - 1;
    static constexpr std::int64_t max_exp   = (1LL << exp_bits) - 1;
    static constexpr std::int64_t max_man   = (1LL << man_bits) - 1;
    static_assert(exp_valid && man_valid && int_repr_valid, "requested exponent and mantissa types are too wide");

    using exp_type         = std::make_unsigned_t<utils::fit_int_t<exp_bits + 1>>;
    using man_type         = std::make_unsigned_t<utils::fit_int_t<man_bits + 1>>;
    using dman_type        = std::make_unsigned_t<utils::fit_int_t<dman_bits + 1>>;
    using double_dman_type = std::make_unsigned_t<utils::fit_int_t<dman_bits * 2>>;
    using int_repr_type    = utils::fit_int_t<exp_bits + man_bits + 1>;
    using uint_repr_type   = std::make_unsigned_t<int_repr_type>;
    using int_type         = utils::fit_int_t<utils::max(exp_bits, man_bits) + 1>;
    using uint_type        = std::make_unsigned_t<int_type>;
    using storage_type     = std::make_unsigned_t<utils::min_int_t<utils::max(exp_bits, man_bits)>>;

    DYNFLOAT_FORCE_INLINE constexpr dfloat() noexcept = default;
    template<typename sign_t_, typename exp_t_, typename man_t_>
    DYNFLOAT_FORCE_INLINE constexpr dfloat(sign_t_ s, exp_t_ exp, man_t_ man) noexcept;
    template<typename integer_, std::enable_if_t<utils::is_signed_integer<integer_>, bool> = true>
    DYNFLOAT_FORCE_INLINE explicit constexpr dfloat(integer_ i) noexcept;
    template<typename integer_, std::enable_if_t<utils::is_unsigned_integer<integer_>, bool> = true>
    DYNFLOAT_FORCE_INLINE explicit constexpr dfloat(integer_ i) noexcept;
    template<typename float_, std::enable_if_t<std::is_floating_point<float_>::value, bool> = true>
    DYNFLOAT_FORCE_INLINE explicit constexpr dfloat(float_ i) noexcept;
    template<typename integer_, std::enable_if_t<utils::is_signed_integer<integer_>, bool> = true>
    DYNFLOAT_FORCE_INLINE explicit constexpr operator integer_() const noexcept;
    template<typename integer_, std::enable_if_t<utils::is_unsigned_integer<integer_>, bool> = true>
    DYNFLOAT_FORCE_INLINE explicit constexpr operator integer_() const noexcept;
    template<std::int64_t other_exp_bits_, std::int64_t other_man_bits_>
    DYNFLOAT_FORCE_INLINE explicit constexpr operator dfloat<other_exp_bits_, other_man_bits_>() const noexcept;
    DYNFLOAT_FORCE_INLINE explicit operator float() const noexcept;  // No constexpr support as of C++14
    DYNFLOAT_FORCE_INLINE explicit operator double() const noexcept; // No constexpr support as of C++14

    DYNFLOAT_FORCE_INLINE constexpr auto is_saturated() const noexcept -> bool;
    DYNFLOAT_FORCE_INLINE constexpr auto is_zero() const noexcept -> bool;

    DYNFLOAT_FORCE_INLINE constexpr auto is_zero(const zero_behavior z) const noexcept -> bool {
        if (z == zero_behavior::non_zero)
            return false;
        return is_zero();
    }

    DYNFLOAT_FORCE_INLINE constexpr auto dman() const noexcept -> dman_type;
    DYNFLOAT_FORCE_INLINE constexpr auto int_repr() const noexcept -> int_repr_type;
    DYNFLOAT_FORCE_INLINE static constexpr auto from_int_repr(bool sign, int_repr_type in) noexcept -> dfloat;

    template<typename sign_, typename exp_, typename man_>
    DYNFLOAT_FORCE_INLINE static constexpr auto from_components(sign_ s, exp_ e, man_ m) noexcept -> dfloat;
    template<typename sign_, typename exp_, typename man_>
    DYNFLOAT_FORCE_INLINE static constexpr auto from_components_saturate(sign_ s, exp_ e, man_ m) noexcept -> dfloat;
    DYNFLOAT_FORCE_INLINE static constexpr auto saturated_copy(bool sign) noexcept -> dfloat;

    DYNFLOAT_FORCE_INLINE constexpr auto signed_exp() const noexcept -> int_type {
        return static_cast<int_type>(exp_);
    }

    DYNFLOAT_FORCE_INLINE constexpr auto real_exp() const noexcept -> int_type {
        return static_cast<int_type>(exp_) - static_cast<int_type>(zero_exp);
    }

    DYNFLOAT_FORCE_INLINE constexpr auto unsigned_exp() const noexcept -> uint_type {
        return static_cast<uint_type>(exp_);
    }

    DYNFLOAT_FORCE_INLINE constexpr auto unsigned_man() const noexcept -> int_type {
        return static_cast<uint_type>(man_);
    }

    DYNFLOAT_FORCE_INLINE constexpr auto signed_man() const noexcept -> int_type {
        return static_cast<int_type>(man_);
    }

    DYNFLOAT_FORCE_INLINE constexpr auto unsigned_sign() const noexcept -> uint_type {
        return static_cast<uint_type>(sign_);
    }

    DYNFLOAT_FORCE_INLINE constexpr auto bool_sign() const noexcept -> bool {
        return sign_;
    }

    DYNFLOAT_FORCE_INLINE constexpr auto signed_sign() const noexcept -> int_type {
        return static_cast<int_type>(sign_);
    }

    DYNFLOAT_FORCE_INLINE constexpr auto assumed_sign(const sign_behavior s) const noexcept -> int_type {
        switch (s) {
            case sign_behavior::positive:
                return 0;
            case sign_behavior::negative:
                return 1;
            case sign_behavior::unknown:
                return sign_;
        }
        return sign_;
    }

    static constexpr auto digits = 1 + exp_bits + man_bits;

    constexpr auto to_bits() {
        return int_repr() | sign_ << (exp_bits + man_bits);
    }

    // Members
    storage_type sign_ : 1;
    storage_type exp_: exp_bits;
    storage_type man_: man_bits;
};

// Common Formats
using std_bf16 = dfloat<8, 7>;
using std_f16  = dfloat<5, 10>;
using std_f32  = dfloat<8, 23>;
using std_f64  = dfloat<11, 52>;

// Xilinx
using xilinx_mem_opt  = dfloat<8, 9>;
using xilinx_dsp_opt  = dfloat<8, 17>;
using xilinx_dsp_opt2 = dfloat<8, 24>;

// Platform Aliases
using pltfrm_mem_opt  = xilinx_mem_opt;
using pltfrm_dsp_opt  = xilinx_dsp_opt;
using pltfrm_dsp_opt2 = xilinx_dsp_opt2;

template<typename df1, typename df2>
using max_dfloat = dfloat<std::max(df1::exp_bits, df2::exp_bits), std::max(df1::man_bits, df2::man_bits)>;

template<typename df1, typename df2>
using min_dfloat = dfloat<std::max(df1::exp_bits, df2::exp_bits), std::max(df1::man_bits, df2::man_bits)>;

template<typename dfloat_>
using expand_to_dsp_opt =
    dfloat<utils::max(dfloat_::exp_bits, pltfrm_dsp_opt::exp_bits), utils::max(dfloat_::man_bits, pltfrm_dsp_opt::man_bits)>;

template<typename dfloat_>
using expand_to_dsp_opt2 =
    dfloat<utils::max(dfloat_::exp_bits, pltfrm_dsp_opt2::exp_bits), utils::max(dfloat_::man_bits, pltfrm_dsp_opt2::man_bits)>;

template<typename dfloat_>
using expand_to_dbl_dsp_opt = dfloat<
    (dfloat_::exp_bits <= pltfrm_dsp_opt::exp_bits) ? pltfrm_dsp_opt::exp_bits : utils::max(dfloat_::exp_bits, xilinx_dsp_opt2::exp_bits),
    (dfloat_::man_bits <= pltfrm_dsp_opt::man_bits) ? pltfrm_dsp_opt::man_bits : utils::max(dfloat_::man_bits, xilinx_dsp_opt2::man_bits)>;

template<typename dfloat_>
using expand_to_dbl_dsp_opt2 =
    dfloat<utils::max(dfloat_::exp_bits, xilinx_dsp_opt2::exp_bits), utils::max(dfloat_::man_bits, xilinx_dsp_opt2::man_bits)>;

template<typename dfloat_, typename exec_>
using expand_for_rounding =
    dfloat<dfloat_::exp_bits,
           exec_{} == expand_behavior::no_expand ? dfloat_::man_bits
                                                 : dfloat_::man_bits + std::min(2LL, 64LL - dfloat_::man_bits - dfloat_::exp_bits - 1)>;

template<typename dfloat_>
using shrink_to_dsp_opt = dfloat<dfloat_::exp_bits, utils::min(dfloat_::man_bits, pltfrm_dsp_opt::man_bits)>;

template<typename dfloat_>
using shrink_to_dsp_opt2 = dfloat<dfloat_::exp_bits, utils::min(dfloat_::man_bits, pltfrm_dsp_opt2::man_bits)>;

static_assert(std::is_standard_layout<std_f32>::value, "dfloat should be a standard layout class");
static_assert(std::is_trivial<std_f32>::value, "dfloat should be a trivial class");
static_assert(sizeof(std_f16) == sizeof(std::int16_t), "Size mismatch!");
static_assert(sizeof(std_f32) == sizeof(float), "Size mismatch!");
static_assert(sizeof(std_f64) == sizeof(double), "Size mismatch!");

namespace details_is_dfloat {
template<typename>
struct is_dfloat: std::false_type {};

template<std::int64_t exp_bits_, std::int64_t man_bits_>
struct is_dfloat<dfloat<exp_bits_, man_bits_>>: std::true_type {};
} // namespace details_is_dfloat

template<typename type_>
struct is_dfloat: details_is_dfloat::is_dfloat<std::remove_volatile_t<std::remove_const_t<type_>>> {};

template<typename type_>
static constexpr auto is_dfloat_v = is_dfloat<type_>::value;

enum class memory_type : std::int8_t {
    xilinx_bram = 0,
    tool_def    = 1,
};

enum class dsp_type : std::int8_t {
    xilinx_dsp18 = 0,
    tool_def     = 1,
};

template<typename dfloat_type, typename least_allowed_precision_ratio, memory_type mem_type>
struct optimize_for_memory;

template<typename dfloat_type, typename least_allowed_precision_ratio, memory_type mem_type>
using optimize_for_memory_t = typename optimize_for_memory<dfloat_type, least_allowed_precision_ratio, mem_type>::type;

template<typename dfloat_type, typename least_allowed_precision_ratio, dsp_type dsp_type>
struct optimize_for_dsp;

template<typename dfloat_type, typename least_allowed_precision_ratio, dsp_type dsp_type>
using optimize_for_dsp_t = typename optimize_for_dsp<dfloat_type, least_allowed_precision_ratio, dsp_type>::type;

template<typename other_dfloat_, typename asm1 = default_assumptions, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
dfloat_cast(dfloat<exp_bits_, man_bits_> x) noexcept -> other_dfloat_;

template<typename smaller_dfloat,
         rounding_behavior rnd,
         typename asm1 = default_assumptions,
         std::int64_t exp_bits_,
         std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
round(dfloat<exp_bits_, man_bits_> x) noexcept -> smaller_dfloat;

//template<typename dfloat_, typename integer_, std::enable_if_t<is_dfloat_v<dfloat_>, bool> = true>
//DYNFLOAT_FORCE_INLINE static constexpr auto
//make_dfloat(integer_ i) noexcept -> dfloat_;
//
//template<std::int64_t exp_bits_,
//         std::int64_t man_bits_,
//         typename integer_,
//         std::enable_if_t<utils::is_unsigned_integer<integer_>, bool> = true>
//DYNFLOAT_FORCE_INLINE static constexpr auto
//make_dfloat(integer_ i) noexcept -> dfloat<exp_bits_, man_bits_>;
//
//template<std::int64_t exp_bits_,
//         std::int64_t man_bits_,
//         typename integer_,
//         std::enable_if_t<utils::is_signed_integer<integer_>, bool> = true>
//DYNFLOAT_FORCE_INLINE static constexpr auto
//make_dfloat(integer_ i) noexcept -> dfloat<exp_bits_, man_bits_>;

template<std::int64_t scale, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
to_ufixed(dfloat<exp_bits_, man_bits_> lhs) noexcept -> std::uint64_t;

template<std::int64_t scale, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
to_fixed(const dfloat<exp_bits_, man_bits_> lhs) noexcept -> std::int64_t;

template<typename execution_,
         typename asm1 = default_assumptions,
         typename asm2 = default_assumptions,
         std::int64_t exponent_bits_,
         std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
add(dfloat<exponent_bits_, mantissa_bits_> lhs, dfloat<exponent_bits_, mantissa_bits_> rhs) noexcept
    -> dfloat<exponent_bits_, mantissa_bits_>;

template<typename execution_,
         typename asm1 = default_assumptions,
         typename asm2 = default_assumptions,
         std::int64_t exponent_bits_,
         std::int64_t mantissa_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
sub(dfloat<exponent_bits_, mantissa_bits_> lhs, dfloat<exponent_bits_, mantissa_bits_> rhs) noexcept
    -> dfloat<exponent_bits_, mantissa_bits_>;

template<typename execution_,
         typename asm1 = default_assumptions,
         typename asm2 = default_assumptions,
         std::int64_t exp_bits_,
         std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
div(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> dfloat<exp_bits_, man_bits_>;

template<typename execution_,
         typename asm1 = default_assumptions,
         typename asm2 = default_assumptions,
         std::int64_t exp_bits_,
         std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
mul(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> dfloat<exp_bits_, man_bits_>;

template<typename result_type,
         typename execution_,
         typename asm1 = default_assumptions,
         typename asm2 = default_assumptions,
         std::int64_t exp_bits1_,
         std::int64_t man_bits1_,
         std::int64_t exp_bits2_,
         std::int64_t man_bits2_>
DYNFLOAT_FORCE_INLINE static constexpr auto
mixed_mul(dfloat<exp_bits1_, man_bits1_> lhs, dfloat<exp_bits2_, man_bits2_> rhs) noexcept -> result_type;

template<typename execution_, typename asm1 = default_assumptions, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE constexpr auto
reciprocal(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_>;

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
exp(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_>;

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
pow2(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_>;

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
sqrt(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_>;

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
inv_sqrt(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_>;

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
lg2(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_>;

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
ln(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_>;

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
tanh(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_>;

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
logistic(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_>;

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
floor_int(dfloat<exp_bits_, man_bits_> x) noexcept -> std::int64_t;

template<typename execution_, std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
floor(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_>;

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
floor_to_zero(dfloat<exp_bits_, man_bits_> x) noexcept -> dfloat<exp_bits_, man_bits_>;

template<std::int64_t scale, std::int64_t iterations, typename int_type_>
DYNFLOAT_FORCE_INLINE static constexpr auto
fixed_ln_limited_range(int_type_ x) noexcept -> std::int64_t;

template<std::int64_t scale, std::int64_t iterations, typename int_type_>
DYNFLOAT_FORCE_INLINE static constexpr auto
fixed_sqrt_limited_range(int_type_ x) noexcept -> std::int64_t;

namespace tree {
template<typename execution_, typename asm1 = default_assumptions, std::int64_t exp_bits_, std::int64_t man_bits_, std::size_t size_>
DYNFLOAT_FORCE_INLINE static constexpr auto
add(utils::array<dfloat<exp_bits_, man_bits_>, size_> eles) noexcept -> dfloat<exp_bits_, man_bits_>;

template<typename execution_, typename asm1 = default_assumptions, std::int64_t exp_bits_, std::int64_t man_bits_, std::size_t size_>
DYNFLOAT_FORCE_INLINE static constexpr auto
add(const dfloat<exp_bits_, man_bits_> (&eles)[size_]) noexcept // NOLINT
    -> dfloat<exp_bits_, man_bits_>;

template<typename execution_, typename asm1 = default_assumptions, std::int64_t exp_bits_, std::int64_t man_bits_, typename... Args>
DYNFLOAT_FORCE_INLINE static constexpr auto
add(dfloat<exp_bits_, man_bits_> ele, dfloat<exp_bits_, man_bits_> ele2, dfloat<exp_bits_, man_bits_> ele3, Args... args) noexcept
    -> dfloat<exp_bits_, man_bits_>;

template<typename execution_, typename asm1 = default_assumptions, std::int64_t exp_bits_, std::int64_t man_bits_, typename... Args>
DYNFLOAT_FORCE_INLINE static constexpr auto
mul(dfloat<exp_bits_, man_bits_> ele1, dfloat<exp_bits_, man_bits_> ele2, dfloat<exp_bits_, man_bits_> ele3, Args... eles) noexcept
    -> dfloat<exp_bits_, man_bits_>;

template<typename execution_, typename asm1 = default_assumptions, std::int64_t exp_bits_, std::int64_t man_bits_, std::size_t size_>
DYNFLOAT_FORCE_INLINE static constexpr auto
mul(utils::array<dfloat<exp_bits_, man_bits_>, size_> eles) noexcept -> dfloat<exp_bits_, man_bits_>;
} // namespace tree

//template<typename execution_,
//         typename asm1 = default_assumptions,
//         std::int64_t exp_bits_,
//         std::int64_t man_bits_,
//         typename integer_,
//         std::enable_if_t<std::is_integral<integer_>::value, bool> = true>
//DYNFLOAT_FORCE_INLINE static constexpr auto
//lshift(dfloat<exp_bits_, man_bits_> lhs, integer_ rhs) noexcept -> dfloat<exp_bits_, man_bits_>;

//template<std::int64_t exp_bits_,
//         std::int64_t man_bits_,
//         typename integer_/*,
//         std::enable_if_t<std::is_integral<integer_>::value, bool> = true*/>
//DYNFLOAT_FORCE_INLINE static constexpr auto
//operator>>(dfloat<exp_bits_, man_bits_> lhs, integer_ rhs) noexcept -> dfloat<exp_bits_, man_bits_>;

//template<typename execution_, typename asm1 = default_assumptions, typename shifted_type_, typename shift_type_>
//DYNFLOAT_FORCE_INLINE static constexpr auto
//safe_lshift(const shifted_type_ val, const shift_type_ shift_amount) noexcept -> auto {
//    const auto signed_shift = static_cast<std::make_signed_t<shift_type_>>(shift_amount);
//    return signed_shift < 0 ? val >> (-signed_shift) : lshift<execution_, asm1>(val, signed_shift);
//}

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator<(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> bool;

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator<=(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> bool;

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator==(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> bool;

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator>=(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> bool;

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator>(dfloat<exp_bits_, man_bits_> lhs, dfloat<exp_bits_, man_bits_> rhs) noexcept -> bool;

//template<typename dfloat_, typename execution_, typename num_, typename denom_, std::enable_if_t<is_dfloat_v<dfloat_>, bool> = true>
//DYNFLOAT_FORCE_INLINE static constexpr auto
//make_dfloat(num_ num, denom_ denom) noexcept -> dfloat_;
//
//template<std::int64_t exp_bits_, std::int64_t man_bits_, typename execution_, typename num_, typename denom_>
//DYNFLOAT_FORCE_INLINE static constexpr auto
//make_dfloat(num_ num, denom_ denom) noexcept -> dfloat<exp_bits_, man_bits_>;

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator+(dfloat<exp_bits_, man_bits_> rhs) noexcept -> dfloat<exp_bits_, man_bits_> {
    return rhs;
}

template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
operator-(dfloat<exp_bits_, man_bits_> rhs) noexcept -> dfloat<exp_bits_, man_bits_> {
    return dfloat<exp_bits_, man_bits_>::from_components(~rhs.unsigned_sign(), rhs.unsigned_exp(), rhs.unsigned_man());
}

} // namespace dynfloat

//// NOLINTBEGIN(cert-dcl58-cpp)
//namespace std {
////template<std::int64_t exp_bits_, std::int64_t man_bits_>
////DYNFLOAT_FORCE_INLINE static constexpr auto
////swap(dynfloat::dfloat<exp_bits_, man_bits_>& f1, dynfloat::dfloat<exp_bits_, man_bits_>& f2) noexcept;
//
////template<std::int64_t exp_bits_, std::int64_t man_bits_>
////class numeric_limits<dynfloat::dfloat<exp_bits_, man_bits_>>;
//
////template<std::int64_t exp_bits_, std::int64_t man_bits_>
////struct hash<dynfloat::dfloat<exp_bits_, man_bits_>>;
//
//// template<std::int64_t exp_bits_, std::int64_t man_bits_>
//// DYNFLOAT_FORCE_INLINE static constexpr auto
//// abs(dynfloat::dfloat<exp_bits_, man_bits_> f1) noexcept -> dynfloat::dfloat<exp_bits_, man_bits_>;
//
//} // namespace std
//
//// NOLINTEND(cert-dcl58-cpp)
