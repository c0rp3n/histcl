// #define T unsigned char
// #define T_MAX 255
#define STRIDE T_MAX + 1

kernel void histogram(global T* in_data,
                      local size_t* local_histogram,
                      global size_t* partial_histograms)
{
    uint gid = get_group_id(0);
    uint lid = get_local_id(0);
    uint n = get_local_size(0);

    local uint8 local_histogram[STRIDE];

    //clear the scratch bins
    if (lid < STRIDE)
        H[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_inc(&local_histogram[bin_index]);

    if (lid < STRIDE)
    {
        uint offset = (gid * STRIDE) + lid;
        partial_histograms[offset] = local_histogram[lid];
    }
}

kernel void histogram_accum_global(global size_t* partial_histograms,
                                   global size_t* histogram
                                   global size_t hist_count)
{
    int id = get_global_id();

    int count = histograms[id];
    for (int i = 1; i < hist_count; ++i)
    {
        count += histograms[(i * STRIDE) + id];
    }

    histogram[id] = count;
}
