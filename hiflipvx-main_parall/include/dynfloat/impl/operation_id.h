#pragma once
#include "../deps.h"

namespace dynfloat {
enum class operation_id : std::int8_t {
    add,
    sub,
    mul,
    div,
    exp,
    sqrt,
    inv_sqrt,
    lg2,
    tanh,
    logistic
};

} // namespace dynfloat
