kernel void lut(global uint* cuml_hist, float data_max, float data_count)
{
    uint id = get_global_id(0);

    uint count = cuml_hist[id];
    cuml_hist[id] = (uint)floor((float)((float)count / data_count) * data_max);
}