package ptx

// Conv1DK3S1PTX is a Conv1D kernel for kernel_size=3, stride=1 with zero-padding.
// Grid: (out_length/256, out_channels, 1), Block: (256, 1, 1)
// Args: output, input, weight, bias, in_channels, in_length, out_channels, out_length
const Conv1DK3S1PTX = `
.version 7.0
.target sm_70
.address_size 64

// Conv1D k=3, stride=1, padding=1
// Each thread computes one output element for one output channel.
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
    // TODO: implement shared-memory tiled Conv1D
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
