#pragma once

#include <array>
#include <limits>
#include <vector>

#include "compute/engine_cl.hpp"
#include "compute/kernels.hpp"

void lut(cl::Buffer& buffer_data, size_t data_count)
{
    auto& engine = compute::engine_cl::instance();
    auto& kernels = compute::kernels::instance();

    try
    {
        auto& context = engine.get_context();
        auto& queue = engine.get_queue();

        auto& kernel_lut = kernels.get<compute::kernel::lut>();
        kernel_lut.setArg(0, buffer_data);
        kernel_lut.setArg(1, static_cast<cl_float>(255));
        kernel_lut.setArg(2, static_cast<cl_float>(data_count));

        queue.enqueueNDRangeKernel(kernel_lut,
                                   cl::NullRange,
                                   cl::NDRange(256),
                                   cl::NullRange);
    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ", " << compute::engine_cl::get_error_string(err.err()) << std::endl;
    }
}
