package ptx

// Conv1DK3S1PTX is a Conv1D kernel for kernel_size=3, stride=1 with zero-padding.
// Grid: (ceil(out_length/256), out_channels, 1), Block: (256, 1, 1)
// Each thread computes one output element: out[oc][tid + blockIdx.x*256]
const Conv1DK3S1PTX = `
.version 7.0
.target sm_70
.address_size 64

// Conv1D k=3, stride=1, padding=1
// Shared memory: input tile [in_channels * (256+2)] for halo
.visible .entry conv1d_k3_s1(
    .param .u64 out_ptr,
    .param .u64 in_ptr,
    .param .u64 wt_ptr,
    .param .u64 bias_ptr,
    .param .u32 in_channels,
    .param .u32 in_length,
    .param .u32 out_channels,
    .param .u32 out_length
) {
    .reg .u32 %oc, %j, %ic, %tid, %bid;
    .reg .u64 %in_base, %wt_base, %out_base;
    .reg .f32 %sum, %w0, %w1, %w2, %x0, %x1, %x2, %bias_val;
    .shared .f32 sh_tile[258]; // 256 + 2 halo for one input channel

    // oc = blockIdx.y
    mov.u32 %oc, %ctaid.y;
    // j = blockIdx.x * 256 + threadIdx.x
    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %j, %bid, 256, %tid;

    // Load bias
    // sum = bias[oc]

    // For each input channel:
    //   Load tile with halo into shared memory
    //   bar.sync
    //   Accumulate: sum += w[0]*sh[tid] + w[1]*sh[tid+1] + w[2]*sh[tid+2]
    //   bar.sync

    // Store: out[oc * out_length + j] = sum

    ret;
}
`

// Conv1DK3S2PTX is a Conv1D kernel for kernel_size=3, stride=2 with zero-padding.
const Conv1DK3S2PTX = `
.version 7.0
.target sm_70
.address_size 64

.visible .entry conv1d_k3_s2(
    .param .u64 out_ptr,
    .param .u64 in_ptr,
    .param .u64 wt_ptr,
    .param .u64 bias_ptr,
    .param .u32 in_channels,
    .param .u32 in_length,
    .param .u32 out_channels,
    .param .u32 out_length
) {
    // TODO: implement strided Conv1D
    ret;
}
`
