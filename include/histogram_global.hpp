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

    auto& context = engine.get_context();
    auto& queue = engine.get_queue();
    static cl::Program program(engine.create_program(
    {
        "kernels/histogram_global.cl",
        "kernels/histogram_eq.cl"
    }));

    cl::Buffer buffer_histogram(context, CL_MEM_READ_WRITE, stride);
}
