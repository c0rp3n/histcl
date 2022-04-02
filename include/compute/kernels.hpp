#pragma once

#include <array>

#include "engine_cl.hpp"
#include "utils/singleton.hpp"

namespace compute
{
    enum struct kernel : size_t
    {
        histogram_atomic = 0,
        histogram_local,
        histogram_reduce,
        scan_hs,
        scan_bl,
        lut,
        scan_lut,
        equalise,
        count,
    };

    class kernels : public utils::singleton<kernels>
    {
    public:
        kernels();

        void init();

        template<kernel k>
        constexpr cl::Kernel& get()
        {
            return this->m_map[static_cast<size_t>(k)];
        }

    private:
        typedef std::array<cl::Kernel, static_cast<size_t>(kernel::count)> kernel_map;

        cl::Program m_program;
        kernel_map m_map;
    };
}