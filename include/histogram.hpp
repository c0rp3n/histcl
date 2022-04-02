#pragma once

#include <array>
#include <limits>
#include <tuple>
#include <vector>

#include "compute/engine_cl.hpp"

inline std::tuple<size_t, size_t> get_data_count(size_t count, size_t required)
{
    size_t rem = count % required;
    size_t ext = 0;
    if (rem > 0)
    {
        ext = required - rem;
        count += ext;
    }

    // divide by 4 as using uchar4
    count = count >> 2;

    return std::make_tuple(count, ext);
}

template<
    class T,
    class InArray = std::vector<T>,
    class OutArray = std::vector<uint32_t>
    >
void histogram_g(InArray& data, OutArray& histogram)
{
    auto& engine = compute::engine_cl::instance();
    auto& functions = compute::functions::instance();

    try
    {
        // round up data count so it a multiple of 4 then dived by 4 as we're
        // using uchar4.
        auto [data_count, data_extension] = get_data_count(data_count, 4);
        auto data_size = data.size() * sizeof(InArray::value_type);
        auto hist_size = histogram.size() * sizeof(OutArray::value_type);

        auto& context = engine.get_context();
        auto& queue = engine.get_queue();

        // create buffers
        cl::Buffer buffer_data(context, CL_MEM_READ_ONLY, data_size);
        cl::Buffer buffer_hist(context, CL_MEM_READ_WRITE, hist_size);

        queue.enqueueWriteBuffer(buffer_data,
                                 CL_TRUE,
                                 0,
                                 data_size,
                                 &data[0]);

        // write 0 bytes for the extra data
        if (data_extension > 0)
        {
            queue.enqueueFillBuffer<char>(buffer_data, 0, data_size, data_extension);
        }

        // run histogram kernel
        auto& kernel_hist = functions.get<compute::kernel::histogram_atomic>();
        kernel_hist.setArg(0, buffer_data);
        kernel_hist.setArg(1, buffer_hist);

        queue.enqueueNDRangeKernel(kernel_hist,
                                   cl::NullRange,
                                   cl::NDRange(data_count),
                                   cl::NullRange);

        // read histogram
        queue.enqueueReadBuffer(buffer_hist,
                                CL_TRUE,
                                0,
                                hist_size,
                                &histogram[0]);

        // correct the histogram
        if (data_extension)
        {
            histogram[0] - data_extension;
        }
    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ", " << compute::engine_cl::get_error_string(err.err()) << std::endl;
    }
}

template<
    class T,
    class InArray = std::vector<T>,
    class OutArray = std::vector<size_t>
    >
void histogram_l(InArray& data, OutArray& histogram)
{
    auto& engine = compute::engine_cl::instance();
    auto& kernels = compute::kernels::instance();

    try
    {
        auto& context = engine.get_context();
        auto& queue = engine.get_queue();

        // create buffers
        auto hist_size = histogram.size() * sizeof(cl_uint);
        cl::Buffer buffer_hist(context, CL_MEM_READ_WRITE, hist_size);

        histogram_l<T>(data, buffer_hist, hist_size);

        // read the histogram
        queue.enqueueReadBuffer(buffer_hist,
                                CL_TRUE,
                                0,
                                hist_size,
                                &histogram[0]);
    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ", " << compute::engine_cl::get_error_string(err.err()) << std::endl;
    }
}

template<
    class T,
    class InArray = std::vector<T>
    >
void histogram_l(InArray& data, cl::Buffer& buffer_hist, size_t hist_size)
{
    auto& engine = compute::engine_cl::instance();
    auto& kernels = compute::kernels::instance();

    try
    {
        // round up data count so it a multiple of 128 then dived by 4 as we're
        // using uchar4
        auto [data_count, data_extension] = get_data_count(data.size(), 128ull);
        auto data_size = data.size() * sizeof(InArray::value_type);

        size_t work_group_count = data_count / 32ull;
        const size_t work_group_size = 32ull;
        std::cout << "work_group_count: " << work_group_count << std::endl;

        auto& context = engine.get_context();
        auto& queue = engine.get_queue();

        // create buffers
        cl::Buffer buffer_data(context, CL_MEM_READ_ONLY, data_size);
        cl::Buffer buffer_phist(context, CL_MEM_READ_WRITE, work_group_count * 256);

        queue.enqueueWriteBuffer(buffer_data,
                                 CL_TRUE,
                                 0,
                                 data_size,
                                 &data[0]);

        // write null bytes for the extra required data
        if (data_extension > 0)
        {
            queue.enqueueFillBuffer<cl_char>(buffer_data, 0, data_size, data_extension);
        }

        // run the local histogram kernel
        auto& kernel_hist = kernels.get<compute::kernel::histogram_local>();
        kernel_hist.setArg(0, buffer_data);
        kernel_hist.setArg(1, buffer_phist);

        queue.enqueueNDRangeKernel(kernel_hist,
                                   cl::NullRange,
                                   cl::NDRange(data_count),
                                   cl::NDRange(32));

        // run the reduce kernel
        auto& kernel_reduce = kernels.get<compute::kernel::histogram_reduce>();
        kernel_reduce.setArg(0, buffer_phist);
        kernel_reduce.setArg(1, buffer_hist);
        kernel_reduce.setArg(2, static_cast<cl_uint>(work_group_count));
        kernel_reduce.setArg(3, static_cast<cl_uint>(data_extension));

        queue.enqueueNDRangeKernel(kernel_reduce,
                                   cl::NullRange,
                                   cl::NDRange(256),
                                   cl::NullRange);
    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ", " << compute::engine_cl::get_error_string(err.err()) << std::endl;
    }
}
