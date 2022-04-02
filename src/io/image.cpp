#include "io/image.hpp"

#include <iostream>
#include <fstream>

#include "../stb/stb_image.h"
#include "../stb/stb_image_write.h"

io::image_t io::load_image(const std::string_view& filepath)
{
    int x = 0, y = 0, channels = 0;
    auto* image_buffer = stbi_load(filepath.data(), &x, &y, &channels, 1);
    if(!image_buffer)
    {
        throw std::runtime_error("failed to read image");
    }

    // Get the width and height
    size_t width = static_cast<size_t>(x);
    size_t height = static_cast<size_t>(y);
    size_t spp = static_cast<size_t>(channels);
    size_t num_pixels = width * height;
    size_t size = num_pixels * channels;
    std::vector<uint8_t> raster(size);
    {
        auto* image_data = reinterpret_cast<uint8_t*>(image_buffer);
        std::copy(image_data, image_data + size, raster.data());
    }

    // Free the buffer
    stbi_image_free(image_buffer);

    return std::make_tuple(width, height, raster);
}

void io::write_image(const std::string_view& filename,
                     size_t width,
                     size_t height,
                     const std::vector<uint8_t>& raster)
{
    stbi_write_png(filename.data(),
                   static_cast<int>(width),
                   static_cast<int>(height),
                   static_cast<int>(1),
                   raster.data(),
                   static_cast<int>(width));
}
