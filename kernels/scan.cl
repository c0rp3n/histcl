kernel void scan_hs(global uint* A, global uint* B)
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
}

kernel void scan_bl(global uint* A)
{
    uint id = get_global_id(0);
    uint N = get_global_size(0);
    uint t;

    // up-sweep
    for (uint stride = 1; stride < N; stride *= 2)
    {
        if (((id + 1) % (stride*2)) == 0)
        {
            A[id] += A[id - stride];
        }

        // sync the step
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // down-sweep
    if (id == 0)
    {
        A[N-1] = 0;
    }

    // exclusive scan
    // sync the step
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (uint stride = N/2; stride > 0; stride /= 2)
    {
        if (((id + 1) % (stride*2)) == 0)
        {
            t = A[id];
            // reduce
            A[id] += A[id - stride];
            // move
            A[id - stride] = t;
        }
        barrier(CLK_GLOBAL_MEM_FENCE); // sync the step
    }
}
