#pragma once

#include <array>
#include <limits>
#include <vector>

#include "compute/engine_cl.hpp"
#include "compute/kernels.hpp"

void scan_lut(cl::Buffer& buffer_data, size_t hist_bins, size_t hist_size, size_t data_count)
{
    auto& engine = compute::engine_cl::instance();
    auto& kernels = compute::kernels::instance();

    try
    {
        auto& context = engine.get_context();
        auto& queue = engine.get_queue();

        // create buffers
        cl::Buffer buffer_temp(context, CL_MEM_READ_WRITE, hist_size);

        auto& kernel_scan_lut = kernels.get<compute::kernel::scan_lut>();
        kernel_scan_lut.setArg(0, buffer_data);
        kernel_scan_lut.setArg(1, buffer_temp);
        kernel_scan_lut.setArg(2, static_cast<cl_float>(255));
        kernel_scan_lut.setArg(3, static_cast<cl_float>(data_count));

        queue.enqueueNDRangeKernel(kernel_scan_lut,
                                   cl::NullRange,
                                   cl::NDRange(hist_bins),
                                   cl::NullRange);
    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ", " << compute::engine_cl::get_error_string(err.err()) << std::endl;
    }
}
