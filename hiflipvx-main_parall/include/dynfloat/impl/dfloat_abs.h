#pragma once
#include "dfloat.h"

// NOLINTBEGIN(cert-dcl58-cpp)
namespace std {
template<std::int64_t exp_bits_, std::int64_t man_bits_>
DYNFLOAT_FORCE_INLINE static constexpr auto
abs(dynfloat::dfloat<exp_bits_, man_bits_> f1) noexcept -> dynfloat::dfloat<exp_bits_, man_bits_> // NOLINT
{
    f1.sign_ = 0;
    return f1;
}
} // namespace std

// NOLINTEND(cert-dcl58-cpp)
