#pragma once

#include <cstdint>
#include <string_view>
#include <tuple>
#include <vector>

namespace io
{
    typedef std::tuple<size_t, size_t, std::vector<uint8_t>> image_t;

    image_t load_image(const std::string_view& filepath);
    void write_image(const std::string_view& filepath, size_t width, size_t height, const std::vector<uint8_t>& raster);
}