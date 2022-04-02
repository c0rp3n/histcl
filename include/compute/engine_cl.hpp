#pragma once

#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include "CL/cl2.hpp"

#include "utils/singleton.hpp"

namespace compute
{
    class engine_cl : public utils::singleton<engine_cl>
    {
    public:
        engine_cl();

        void init(size_t device_idx = 0);

        cl::Program create_program(const std::initializer_list<std::string_view>& sources);

        constexpr cl::Device& get_device()
        {
            return this->m_device;
        }
        constexpr cl::Context& get_context()
        {
            return this->m_context;
        }
        constexpr cl::CommandQueue& get_queue()
        {
            return this->m_queue;
        }

        static std::vector<std::pair<cl::Platform, cl::Device>> get_device_list();
        static const char* get_error_string(cl_int error);

    private:
        cl::Device m_device;
        cl::Context m_context;
        cl::CommandQueue m_queue;
    };
}