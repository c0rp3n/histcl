#pragma once

#include <array>
#include <limits>
#include <vector>

template<class T, class Array = std::vector<T>>
void hist_eq(Array& data)
{
    size_t count = data.size();
    constexpr auto max = std::numeric_limits<T>::max();
    std::array<size_t, max + 1> buffer;

    // histogram
    for (const auto v : data)
    {
        ++buffer[v];
    }

    // cumulative histogram
    for (size_t i = 1; i < count; ++i)
    {
        buffer[i] += buffer[i - 1];
    }

    // normalised histogram (lut)
    for (auto& v : buffer)
    {
        v = (v * max) / count;
    }

    // equalise
    for (auto& v : data)
    {
        v = static_cast<T>(buffer[v]);
    }
}
