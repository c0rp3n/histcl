#pragma once

#include <array>
#include <limits>
#include <vector>

#include "compute/engine_cl.hpp"

template<class T, class Array = std::vector<T>>
void hist_eq(compute::engine_cl& engine, Array& data)
{
    size_t count = data.size();
    constexpr auto max = std::numeric_limits<T>::max();
    constexpr auto stride = max + 1;
    std::array<size_t, stride> histogram;

    static cl::Program program(engine.get_context(), engine.get_sources());
    auto& queue = engine.get_queue();
}
