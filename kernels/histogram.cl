#define WORK_GROUP_SIZE 32
#define WORK_ITEM_SIZE 1
#define BIN_COUNT 256

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

inline void hist_inc(local uchar* H, const uchar4 idxs)
{
    ++H[idxs.x];
    ++H[idxs.y];
    ++H[idxs.z];
    ++H[idxs.w];
}

inline void hist_atomic_inc(global size_t* H, const uchar4 idxs)
{
    atomic_inc(&H[idxs.x]);
    atomic_inc(&H[idxs.y]);
    atomic_inc(&H[idxs.z]);
    atomic_inc(&H[idxs.w]);
}

kernel void hist_atomic(global const uchar4* A, global size_t* H)
{
    size_t gid = get_global_id(0);

    hist_atomic_inc(H, A[gid]);
}

__attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
kernel void hist_local(global const uchar4* A, global uchar* PH)
{
    uint group_id = get_group_id(0);
    uint local_id = get_local_id(0);

    //printf("gid: %d, lid: %d\n", group_id, local_id);

    local uchar LH[WORK_GROUP_SIZE * BIN_COUNT];

    // clear the local histogram for this work-group
    {
        // each work-item needs to clear its local histogram
        local uchar* h = &LH[local_id * BIN_COUNT];
        for (uint i = 0; i < BIN_COUNT; ++i)
        {
            h[i] = 0;
        }
    }

    //printf("[%d, %d] - Local histogram cleared.\n", group_id, local_id);

    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate work-item histogram
    {
        local uchar* h = &LH[local_id * BIN_COUNT];
        const uint i = (group_id * WORK_GROUP_SIZE) + local_id;
        hist_inc(h, A[i]);
    }

    //printf("[%d, %d] - Local histogram incremented.\n", group_id, local_id);

    barrier(CLK_LOCAL_MEM_FENCE);

    // reduce work-item partial histograms to partial work-group histograms
    {
        global uchar* ph = &PH[group_id * BIN_COUNT];

        // there are 32 work-items in a thread so each work-item needs to reduce
        // 4 values of the local partial histogram into one histogram
        uint offset = local_id * 8;
        for (uint i = 0; i < 8; ++i)
        {
            uint index = offset + i; // index into the partial histograms
            ph[index] = LH[index]; // copy the first partial

            for (uint j = 1; j < WORK_GROUP_SIZE; ++j)
            {
                // add the remaining partial histograms
                ph[index] += LH[(j * BIN_COUNT) + index];
            }

            //printf("[%d, %d] - Reduced %d.\n", group_id, local_id, index);
        }
    }

    //printf("[%d, %d] - Local histogram reduced.\n", group_id, local_id);
}

kernel void hist_reduce(global const uchar* PH, global uint* H, uint pc, uint ext)
{
    uint gid = get_global_id(0);

    H[gid] = PH[gid]; // copy the first value.
    for (uint i = 1; i < pc; ++i)
    {
        // copy the remaining local bins into the partial histogram.
        H[gid] += PH[(i * BIN_COUNT) + gid];
    }

    // remove any added extra data from bin 0.
    // this is only if we had to round up the data count.
    if (gid == 0 && ext > 0)
    {
        H[gid] -= ext;
    }
}
