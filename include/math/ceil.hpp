#pragma once

#include "../base_types.hpp"

namespace math
{
    template<typename T>
    constexpr T ceil(tfloat x)
    {
        T xi = static_cast<T>(x);
        return x > xi ? xi + 1 : xi;
    }
} // namespace math
