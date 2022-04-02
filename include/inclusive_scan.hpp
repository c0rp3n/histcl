#pragma once

#include <array>
#include <limits>
#include <vector>

#include "compute/engine_cl.hpp"
#include "compute/kernels.hpp"

void inclusive_scan(cl::Buffer& buffer_data, size_t data_count, size_t data_size)
{
    auto& engine = compute::engine_cl::instance();
    auto& kernels = compute::kernels::instance();

    try
    {
        auto& context = engine.get_context();
        auto& queue = engine.get_queue();

        // create buffers
        cl::Buffer buffer_temp(context, CL_MEM_READ_WRITE, data_size);

        auto& kernel_scan = kernels.get<compute::kernel::scan_hs>();
        kernel_scan.setArg(0, buffer_data);
        kernel_scan.setArg(1, buffer_temp);

        queue.enqueueNDRangeKernel(kernel_scan,
                                   cl::NullRange,
                                   cl::NDRange(data_count),
                                   cl::NullRange);
    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ", " << compute::engine_cl::get_error_string(err.err()) << std::endl;
    }
}
