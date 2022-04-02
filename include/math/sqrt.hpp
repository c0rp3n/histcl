#pragma once

#include <limits>

#include "utils/template_helpers.hpp"

namespace math
{
    template<class T = float, utils::enable_if_floating_t<T> = 0>
    constexpr T _sqrt_newton_raphson(T x, T curr, T prev)
    {
        return curr == prev
                   ? curr
                   : _sqrt_newton_raphson(x, 0.5 * (curr + x / curr), curr);
    }

    // Constant expresion square root implementation
    // https://stackoverflow.com/questions/8622256/in-c11-is-sqrt-defined-as-constexpr
    template<class T = float, utils::enable_if_floating_t<T> = 0>
    constexpr T sqrt(T x)
    {
        return (x >= 0 && x < std::numeric_limits<T>::infinity())
                   ? _sqrt_newton_raphson(x, x, 0)
                   : std::numeric_limits<T>::quiet_NaN();
    }
} // namespace math