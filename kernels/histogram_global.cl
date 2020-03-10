// #define T unsigned char
// #define T_MAX 255

kernel void histogram_global(global const T* in_data,
                            global size_t* histogram)
{
    uint id = get_global_id(0);
    uint n = get_global_size(0);

    for (uint i = id * n; i < (id + 1) * n; ++i)
    {
        atomic_inc(histogram[in_data[i]]);
    }
}