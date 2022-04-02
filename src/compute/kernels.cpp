#include "compute/kernels.hpp"

#include <string>
#include <utility>

using namespace compute;

kernels::kernels() : m_program(), m_map()
{
}

typedef std::pair<kernel, const char*> kernel_pair;

const std::array<kernel_pair, static_cast<size_t>(kernel::count)> kernel_pairs =
{
    std::make_pair(kernel::histogram_atomic,    "hist_atomic"),
    std::make_pair(kernel::histogram_local,     "hist_local"),
    std::make_pair(kernel::histogram_reduce,    "hist_reduce"),
    std::make_pair(kernel::scan_hs,             "scan_hs"),
    std::make_pair(kernel::scan_bl,             "scan_bl"),
    std::make_pair(kernel::lut,                 "lut"),
    std::make_pair(kernel::scan_lut,            "scan_lut"),
    std::make_pair(kernel::equalise,            "equalise")
};

void kernels::init()
{
    this->m_program = engine_cl::instance().create_program(
    {
        "kernels/histogram.cl",
        "kernels/scan.cl",
        "kernels/lut.cl",
        "kernels/scan_lut.cl",
        "kernels/equalise.cl"
    });

    for (auto& kp : kernel_pairs)
    {
        this->m_map[static_cast<size_t>(kp.first)] = cl::Kernel(this->m_program, kp.second);
    }
}
