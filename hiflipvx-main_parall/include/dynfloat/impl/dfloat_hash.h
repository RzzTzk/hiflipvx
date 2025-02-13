#pragma once
#include "dfloat.h"

// NOLINTBEGIN(cert-dcl58-cpp)
namespace std {
template<std::int64_t exp_bits_, std::int64_t man_bits_>
struct hash<dynfloat::dfloat<exp_bits_, man_bits_>> // NOLINT
{
    using float_type = dynfloat::dfloat<exp_bits_, man_bits_>;

    constexpr auto operator()(const float_type number) const {
        return std::hash<typename float_type::int_repr_type>(number.int_repr());
    }
};
} // namespace std

// NOLINTEND(cert-dcl58-cpp)
