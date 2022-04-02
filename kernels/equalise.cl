kernel void equalise(global uchar4* data, global const uint* lut)
{
    uint id = get_global_id(0);

    uchar4 v = data[id];
    v.x = (uchar)lut[v.x];
    v.y = (uchar)lut[v.y];
    v.z = (uchar)lut[v.z];
    v.w = (uchar)lut[v.w];

    data[id] = v;
}