#include "compute/engine_cl.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace compute;

inline cl::Device cl_get_device(size_t device_idx = 0)
{
    auto device_list = engine_cl::get_device_list();
    if (device_list.size() > device_idx)
    {
        return device_list[device_idx].second;
    }

    return cl::Device();
}

engine_cl::engine_cl() : m_device(), m_context(), m_queue()
{
}

void engine_cl::init(size_t device_idx)
{
    this->m_device = cl_get_device(device_idx);
    this->m_context = cl::Context(this->m_device);
    this->m_queue = cl::CommandQueue(this->m_context);
}

cl::Program engine_cl::create_program(const std::initializer_list<std::string_view>& sources)
{
    cl::Program::Sources src;
    for (const auto& file_name : sources)
    {
        std::ifstream file(file_name.data());
        if (file.good())
        {
            std::string* source_code = new std::string(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
            src.push_back((*source_code).c_str());
            // TODO should we not delete this
            // delete source_code;
        }
    }
    cl::Program program(this->m_context, src);

    try
    {
        program.build();
    }
    catch (const cl::Error& err)
    {
        std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(this->m_context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
        std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(this->m_context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
        std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->m_context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;

        throw err;
    }

    return program;
}

std::vector<std::pair<cl::Platform, cl::Device>> engine_cl::get_device_list()
{
    // Find all devices
    std::vector<std::pair<cl::Platform, cl::Device>> all_device_list;
    std::vector<cl::Platform> platform_list;
    cl::Platform::get(&platform_list);
    for (const auto& platform : platform_list)
    {
        std::vector<cl::Device> device_list;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &device_list);
        for (const auto& device : device_list)
        {
            all_device_list.emplace_back(std::make_pair(platform, device));
        }
    }

    // Sort devices by type
    std::vector<std::pair<cl::Platform, cl::Device>> ret;
    constexpr cl_device_type type_list[] = {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL};
    for (auto type : type_list)
    {
        // Find all devices of type 'type' and move them into `ret`
        for (auto it = all_device_list.begin(); it != all_device_list.end();)
        {
            if (it->second.getInfo<CL_DEVICE_TYPE>() & type)
            {
                ret.emplace_back(*it);
                it = all_device_list.erase(it); // `.erase()` returns the next valid iterator
            }
            else
            {
                ++it;
            }
        }
    }
    return ret;
}

const char* engine_cl::get_error_string(cl_int error)
{
    switch (error)
    {
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}
