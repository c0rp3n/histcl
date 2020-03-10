// #define T unsigned char
// #define T_MAX 255
// #define STRIDE T_MAX + 1

kernel void histogram_eq_global(global const T* in_data,
                                global size_t* histogram
                                global T* out_data)
{
    uint id = get_global_id(0);

    if (id < STRIDE)
    {
        
    }
}

inline void scan_hs(global int* A, global int* B)
{
    int id = get_global_id(0);
    int N = get_global_size(0);
    global int* C;

    for (int stride=1; stride<N; stride*=2)
    {
        B[id] = A[id];
        if (id >= stride)
        B[id] += A[id - stride];
        barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
        C = A; A = B; B = C; //swap A & B between steps
    }
}

inline void scan_bl(global size_t* A)
{
    int id = get_global_id(0);
    int N = get_global_size(0);
    int t;

    //up-sweep
    for (int stride = 1; stride < N; stride *= 2)
    {
        if (((id + 1) % (stride*2)) == 0)
            A[id] += A[id - stride];
        barrier(CLK_GLOBAL_MEM_FENCE); // sync the step
    }

    //down-sweep
    if (id == 0)
        A[N-1] = 0;

    //exclusive scan
    barrier(CLK_GLOBAL_MEM_FENCE); // sync the step
    for (int stride = N/2; stride > 0; stride /= 2)
    {
        if (((id + 1) % (stride*2)) == 0)
        {
            t = A[id];
            A[id] += A[id - stride];
            //reduce
            A[id - stride] = t;
            //move
        }
        barrier(CLK_GLOBAL_MEM_FENCE); // sync the step
    }
}
