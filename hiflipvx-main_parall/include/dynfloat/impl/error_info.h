#pragma once
#include "../utils.h"
#include "constants.h"
#include "execution.h"
#include "operation_id.h"

namespace dynfloat {

template<operation_id, strategy>
struct error_info {};

// NOLINTNEXTLINE
#define DFLOAT_DEFINE_ERR_OF(OPERATION, PRECISION, L, U)         \
    template<>                                                   \
    struct error_info<OPERATION, PRECISION> {                    \
        static constexpr double lower     = L;                   \
        static constexpr double upper     = U;                   \
        static constexpr double mid_point = (upper + lower) / 2; \
        static constexpr double delta     = upper - lower;       \
    }

DFLOAT_DEFINE_ERR_OF(operation_id::mul, strategy::fast, -7e-2, 7e-2);
DFLOAT_DEFINE_ERR_OF(operation_id::mul, strategy::refined_fast, -2.3e-05, 0);
DFLOAT_DEFINE_ERR_OF(operation_id::mul, strategy::refined, -2.3e-05, 0);
DFLOAT_DEFINE_ERR_OF(operation_id::mul, strategy::refined_exact, 0, 0);
DFLOAT_DEFINE_ERR_OF(operation_id::mul, strategy::exact, 0, 0);

DFLOAT_DEFINE_ERR_OF(operation_id::div, strategy::fast, -5.0e-02, 5.2e-02);
DFLOAT_DEFINE_ERR_OF(operation_id::div, strategy::refined_fast, -2.0e-03, 7.4e-04);
DFLOAT_DEFINE_ERR_OF(operation_id::div, strategy::refined, -3.6e-05, 5.6e-05);
DFLOAT_DEFINE_ERR_OF(operation_id::div, strategy::refined_exact, -3.7e-06, 2.3e-07);
DFLOAT_DEFINE_ERR_OF(operation_id::div, strategy::exact, 0, 2.3e-07);

DFLOAT_DEFINE_ERR_OF(operation_id::exp, strategy::fast, -3.3e-02, 3.7e-02);
DFLOAT_DEFINE_ERR_OF(operation_id::exp, strategy::refined_fast, -6.6e-04, 3.2e-04);
DFLOAT_DEFINE_ERR_OF(operation_id::exp, strategy::refined, -1.7e-04, 7.9e-05);
DFLOAT_DEFINE_ERR_OF(operation_id::exp, strategy::refined_exact, -1.2e-05, 5.6e-06);
DFLOAT_DEFINE_ERR_OF(operation_id::exp, strategy::exact, -6.6e-06, 6e-08);

DFLOAT_DEFINE_ERR_OF(operation_id::sqrt, strategy::fast, -1.7e-02, 4.9e-02);
DFLOAT_DEFINE_ERR_OF(operation_id::sqrt, strategy::refined_fast, -2.2e-05, 2.1e-05);
DFLOAT_DEFINE_ERR_OF(operation_id::sqrt, strategy::refined, -2.2e-05, 2.1e-05);
DFLOAT_DEFINE_ERR_OF(operation_id::sqrt, strategy::refined_exact, -2.2e-05, 2.1e-05);
DFLOAT_DEFINE_ERR_OF(operation_id::sqrt, strategy::exact, 0, 0);

DFLOAT_DEFINE_ERR_OF(operation_id::inv_sqrt, strategy::fast, -5.2e-02, 7.9e-03);
DFLOAT_DEFINE_ERR_OF(operation_id::inv_sqrt, strategy::refined_fast, -4.0e-03, 1.5e-08);
DFLOAT_DEFINE_ERR_OF(operation_id::inv_sqrt, strategy::refined, -4.0e-03, 1.5e-08);
DFLOAT_DEFINE_ERR_OF(operation_id::inv_sqrt, strategy::refined_exact, -2.4e-05, 1.8e-07);
DFLOAT_DEFINE_ERR_OF(operation_id::inv_sqrt, strategy::exact, -2.4e-05, 1.8e-07);

DFLOAT_DEFINE_ERR_OF(operation_id::lg2, strategy::fast, -1e-2, 2.1e-2);
DFLOAT_DEFINE_ERR_OF(operation_id::lg2, strategy::refined_fast, -3e-3, 6.1e-3);
DFLOAT_DEFINE_ERR_OF(operation_id::lg2, strategy::refined, -8.5e-4, 1.6e-3);
DFLOAT_DEFINE_ERR_OF(operation_id::lg2, strategy::refined_exact, -2.7e-5, 3.5e-5);
DFLOAT_DEFINE_ERR_OF(operation_id::lg2, strategy::exact, -4.7e-7, 7.4e-7);

DFLOAT_DEFINE_ERR_OF(operation_id::tanh, strategy::fast, -7.1e-02, 7.1e-02);
DFLOAT_DEFINE_ERR_OF(operation_id::tanh, strategy::refined_fast, -2.3e-03, 4.6e-03);
DFLOAT_DEFINE_ERR_OF(operation_id::tanh, strategy::refined, -1.1e-04, 1.6e-04);
DFLOAT_DEFINE_ERR_OF(operation_id::tanh, strategy::refined_exact, -2.7e-06, 1.1e-05);
DFLOAT_DEFINE_ERR_OF(operation_id::tanh, strategy::exact, -5.4e-07, 1.2e-07);

using relative_error_type = double;

template<typename input_, typename expected_>
constexpr auto
clear_ulp(input_& input, const expected_ expected, std::int64_t ignore_ulps) -> void {
    const auto expected_d = static_cast<input_>(expected);
    const auto exact_man  = static_cast<std::int64_t>(expected_d.unsigned_man());
    const auto input_man  = static_cast<std::int64_t>(input.unsigned_man());
    const auto all_eq     = expected_d.unsigned_exp() == input.unsigned_exp() && expected_d.unsigned_sign() == input.unsigned_sign();
    if (std::abs(exact_man - input_man) <= ignore_ulps && all_eq)
        input = expected_d;
}

template<typename input_, typename expected_>
constexpr auto
relative_difference(input_ input, const expected_ expected, std::int64_t ignore_ulps = 1) -> relative_error_type {
    clear_ulp(input, expected, ignore_ulps);
    const auto input2    = static_cast<relative_error_type>(input);
    const auto expected2 = static_cast<relative_error_type>(static_cast<input_>(expected));
    const auto num       = input2 - expected2;
    return -1 < expected && expected < 1 ? num : num / expected2;
}

template<typename input_, typename expected_>
constexpr auto
ulp_difference(input_ input, const expected_ expected) -> std::int64_t {
    const auto expected_d = static_cast<input_>(expected);
    const auto exact_man  = static_cast<std::int64_t>(expected_d.unsigned_man());
    const auto input_man  = static_cast<std::int64_t>(input.unsigned_man());
    const auto man_diff   = input_man - exact_man;
    if (input.unsigned_exp() == expected_d.unsigned_exp())
        return man_diff;
    const auto exp_diff = static_cast<std::int64_t>(input.unsigned_exp()) - static_cast<std::int64_t>(expected_d.unsigned_exp());
    return man_diff + exp_diff * (input_::max_man + 1);
}

} // namespace dynfloat
