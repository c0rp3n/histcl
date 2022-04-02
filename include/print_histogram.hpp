#pragma once

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

void print_histogram(std::array<uint32_t, 256> hist)
{
    std::cout << "histogram:" << std::endl;
    size_t c = 0;
    for (size_t i = 0; i < 256; ++i)
    {
        std::cout << std::setw(3) << std::to_string(i) << ": " << std::setw(4) << std::to_string(hist[i]);

        // formatting
        ++c;
        if (c >= 8)
        {
            std::cout << std::endl;
            c = 0;
        }
        else
        {
            std::cout << ", ";
        }
    }

    size_t sum = hist[0];
    for (size_t i = 1; i < 256; ++i)
    {
        sum += hist[i];
    }
    std::cout << "hist total: " << sum << std::endl;
}