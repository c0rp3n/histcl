#pragma once

#include <algorithm>
#include <iterator>
#include <functional>
#include <limits>
#include <random>

template<
    class It,
    class T,
    size_t min = std::numeric_limits<T>::min(),
    size_t max = std::numeric_limits<T>::max()
    >
void fill_random(It begin, It end)
{
    // First create an instance of an engine.
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine(rnd_device());  // Generates random integers
    std::uniform_int_distribution<size_t> dist(min, max);

    auto gen = [&dist, &mersenne_engine]()
    {
        return static_cast<T>(dist(mersenne_engine));
    };

    std::generate(begin, end, gen);
}