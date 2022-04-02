#pragma once

#include <array>
#include <limits>
#include <vector>

#include "compute/engine_cl.hpp"
#include "compute/kernels.hpp"

void equalise(cl::Buffer& buffer_data, size_t data_count, cl::Buffer& buffer_lut)
{
    auto& engine = compute::engine_cl::instance();
    auto& kernels = compute::kernels::instance();

    try
    {
        auto& context = engine.get_context();
        auto& queue = engine.get_queue();

        auto& kernel_equalise = kernels.get<compute::kernel::equalise>();
        kernel_equalise.setArg(0, buffer_data);
        kernel_equalise.setArg(1, buffer_lut);

        queue.enqueueNDRangeKernel(kernel_equalise,
                                   cl::NullRange,
                                   cl::NDRange(data_count / 4),
                                   cl::NullRange);
    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ", " << compute::engine_cl::get_error_string(err.err()) << std::endl;
    }
}