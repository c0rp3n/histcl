kernel void scan_lut(global uint* A, global uint* B, float data_count, float data_max)
{
    int id = get_global_id(0);
    int N = get_global_size(0);
    global int* C;

    for (int stride=1; stride<N; stride*=2)
    {
        B[id] = A[id];
        if (id >= stride)
        {
            B[id] += A[id - stride];
        }

        //sync the step
        barrier(CLK_GLOBAL_MEM_FENCE);

        //swap A & B between steps
        C = A;
        A = B;
        B = C;
    }

    // wait for scan completion
    barrier(CLK_GLOBAL_MEM_FENCE);

    uint count = A[id];
    A[id] = (uint)floor((float)((float)count / data_count) * data_max);
}