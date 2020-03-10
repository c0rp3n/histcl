#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include "CL/cl2.hpp"

namespace compute
{
    class engine_cl
    {
        cl::Device device;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Program::Sources sources;
        cl::Program program;

    public:
        engine_cl();
        engine_cl(size_t device_idx);

        static std::vector<std::pair<cl::Platform, cl::Device>> get_device_list();

        void init();

        cl::Program create_program(const std::initializer_list<std::string>& sources);
        void add_source(const std::string& file_name);

        constexpr cl::Device& get_device()
        {
            return this->device;
        }
        constexpr cl::Context& get_context()
        {
            return this->context;
        }
        constexpr cl::CommandQueue& get_queue()
        {
            return this->queue;
        }
        constexpr cl::Program::Sources& get_sources()
        {
            return this->sources;
        }
        constexpr cl::Program& get_program()
        {
            return this->program;
        }
    };
}