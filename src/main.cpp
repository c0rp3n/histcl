#include <array>
#include <iostream>
#include <string>
#include <vector>

#include "argh.h"

#include "compute/engine_cl.hpp"
#include "compute/kernels.hpp"
#include "io/image.hpp"
#include "histogram_eq.hpp"
#include "fill_random.hpp"

/*
 * This project was mainly developed without the workshop implentations, and is
 * based upon my understanding of the methodologies proposed in the lectures.
 * 
 * HistCL implmentes a local memory histograms which uses a work-group size of
 * 32 and a work-item size of 1 as it uses uchar4 as a datatype to reduce
 * global memeory access, the local histograms are then moved to a partial
 * histogram array and then reduced.
 * A harris inclusive scan is used to get the cumulated histogram, this is then
 * turned into the lookup table to save memory.
 * The look up table is then used to equalise the input image, the equalisation
 * step also uses uchar4 to reduce global memory access.
 * 
 * CMake is used as a build-system so the project remained cross-platform and
 * that is the only dependency that shall not be provided.
 * 
 * Argument parsing is done by Argh! (https://github.com/adishavit/argh), Image
 * IO is provided by STB (https://github.com/nothings/stb), I use a custom
 * OpenCL wrapper to abstract away kernels and device selection.
 */

int main(int argc, char** argv)
{
    auto cmdl = argh::parser(argv, argc);

    std::string filepath;
    std::string outpath;

    if (cmdl[{"-h", "--help"}])
    {
        std::cout << "HistCL" << std::endl;
        std::cout << "usage: histcl <image path> [args ...]" << std::endl;
        std::cout << "-o, --output: output path" << std::endl;
        std::cout << "-h, --help: this help text" << std::endl;

        return 0;
    }

    if (cmdl[1].empty())
    {
        std::cout << "usage: histcl <image path> [args ...]" << std::endl;

        return 0;
    }
    cmdl(1) >> filepath;

    cmdl({"-o", "--output"}, "output.png") >> outpath;

    auto [width, height, raster] = io::load_image(filepath);

    auto& engine = compute::engine_cl::instance();
    auto& kernels = compute::kernels::instance();

    try
    {
        engine.init();
        kernels.init();
    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ", " << compute::engine_cl::get_error_string(err.err()) << std::endl;
    }

    histogram_eq<uint8_t>(raster);

    io::write_image(outpath, width, height, raster);

    return 0;
}
