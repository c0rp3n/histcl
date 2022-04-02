#pragma once

#include <array>
#include <limits>
#include <vector>

#include "compute/engine_cl.hpp"
#include "compute/kernels.hpp"
#include "histogram.hpp"
#include "inclusive_scan.hpp"
#include "lut.hpp"
//#include "scan_lut.hpp"
#include "equalise.hpp"

template<class T, class Array = std::vector<T>>
void histogram_eq(Array& data)
{
    auto& engine = compute::engine_cl::instance();
    auto& kernels = compute::kernels::instance();

    try
    {
        auto& context = engine.get_context();
        auto& queue = engine.get_queue();

        // create buffers
        auto hist_size = 256 * sizeof(cl_uint);
        cl::Buffer buffer_hist(context, CL_MEM_READ_WRITE, hist_size);

        // get histogram
        histogram_l<T>(data, buffer_hist, hist_size);

        // cumlative sum
        inclusive_scan(buffer_hist, 256, hist_size);

        // normalised histogram (lut)
        size_t data_count = data.size();
        lut(buffer_hist, data_count);

        // combine sum and lut?
        //size_t data_count = data.size();
        //scan_lut(buffer_hist, 256, hist_size, data_count);

        // equalise
        size_t data_size = data_count * sizeof(T);
        cl::Buffer buffer_data(context, CL_MEM_READ_ONLY, data.size());
        queue.enqueueWriteBuffer(buffer_data,
                                 CL_TRUE,
                                 0,
                                 data_size,
                                 &data[0]);

        equalise(buffer_data, data_count, buffer_hist);

        queue.enqueueReadBuffer(buffer_data,
                                CL_FALSE,
                                0,
                                data_size,
                                &data[0]);
    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ", " << compute::engine_cl::get_error_string(err.err()) << std::endl;
    }
}
