#pragma once
#include "dfloat.h"

// NOLINTBEGIN(cert-dcl58-cpp)
namespace std {
template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
swap(dynfloat::dfloat<exp_bits_, man_bits_>& f1, dynfloat::dfloat<exp_bits_, man_bits_>& f2) noexcept // NOLINT
{
    auto tmp = f1;
    f1       = f2;
    f2       = tmp;
}
} // namespace std

// NOLINTEND(cert-dcl58-cpp)
