package ptx

// Conv1DK3S1PTX is a Conv1D kernel for kernel_size=3, stride=1 with zero-padding.
// Grid: (ceil(out_length/256), out_channels, 1), Block: (256, 1, 1)
// Each thread computes one output element: out[oc][tid + blockIdx.x*256].
// Layouts are channel-major: in[in_channels, in_length], out[out_channels, out_length],
// weights[out_channels, in_channels, 3]. Padding is one sample on both sides.
const Conv1DK3S1PTX = `
.version 7.0
.target sm_70
.address_size 64

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
    .reg .pred %p_oob, %p_bias_nil, %p_loop_done;
    .reg .pred %p0_ge0, %p0_lt, %p0_ok, %p1_ge0, %p1_lt, %p1_ok, %p2_ge0, %p2_lt, %p2_ok;
    .reg .u32 %oc, %j, %ic, %tidx, %bid;
    .reg .u32 %in_ch, %in_len, %out_len;
    .reg .u32 %elem, %chan_base, %w_base, %idx_u;
    .reg .s32 %j_s, %in_len_s, %i0, %i1, %i2;
    .reg .u64 %outp, %inp, %wp, %biasp, %addr, %off;
    .reg .f32 %sum, %bias_val, %x0, %x1, %x2, %w0, %w1, %w2;

    ld.param.u64 %outp, [out_ptr];
    ld.param.u64 %inp, [in_ptr];
    ld.param.u64 %wp, [wt_ptr];
    ld.param.u64 %biasp, [bias_ptr];
    ld.param.u32 %in_ch, [in_channels];
    ld.param.u32 %in_len, [in_length];
    ld.param.u32 %out_len, [out_length];

    mov.u32 %oc, %ctaid.y;
    mov.u32 %tidx, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %j, %bid, 256, %tidx;
    setp.ge.u32 %p_oob, %j, %out_len;
    @%p_oob bra DONE;

    mov.f32 %sum, 0f00000000;
    setp.eq.u64 %p_bias_nil, %biasp, 0;
    @%p_bias_nil bra BIAS_DONE;
    mul.wide.u32 %off, %oc, 4;
    add.u64 %addr, %biasp, %off;
    ld.global.f32 %sum, [%addr];
BIAS_DONE:

    cvt.s32.u32 %j_s, %j;
    cvt.s32.u32 %in_len_s, %in_len;
    add.s32 %i0, %j_s, -1;
    mov.s32 %i1, %j_s;
    add.s32 %i2, %j_s, 1;

    mov.u32 %ic, 0;
LOOP_IC:
    setp.ge.u32 %p_loop_done, %ic, %in_ch;
    @%p_loop_done bra STORE;

    // Weight base: ((oc * in_channels + ic) * 3)
    mul.lo.u32 %w_base, %oc, %in_ch;
    add.u32 %w_base, %w_base, %ic;
    mul.lo.u32 %w_base, %w_base, 3;

    mul.wide.u32 %off, %w_base, 4;
    add.u64 %addr, %wp, %off;
    ld.global.f32 %w0, [%addr];
    add.u32 %elem, %w_base, 1;
    mul.wide.u32 %off, %elem, 4;
    add.u64 %addr, %wp, %off;
    ld.global.f32 %w1, [%addr];
    add.u32 %elem, %w_base, 2;
    mul.wide.u32 %off, %elem, 4;
    add.u64 %addr, %wp, %off;
    ld.global.f32 %w2, [%addr];

    mul.lo.u32 %chan_base, %ic, %in_len;

    mov.f32 %x0, 0f00000000;
    setp.ge.s32 %p0_ge0, %i0, 0;
    setp.lt.s32 %p0_lt, %i0, %in_len_s;
    and.pred %p0_ok, %p0_ge0, %p0_lt;
    @!%p0_ok bra LOAD_X1;
    cvt.u32.s32 %idx_u, %i0;
    add.u32 %elem, %chan_base, %idx_u;
    mul.wide.u32 %off, %elem, 4;
    add.u64 %addr, %inp, %off;
    ld.global.f32 %x0, [%addr];
LOAD_X1:

    mov.f32 %x1, 0f00000000;
    setp.ge.s32 %p1_ge0, %i1, 0;
    setp.lt.s32 %p1_lt, %i1, %in_len_s;
    and.pred %p1_ok, %p1_ge0, %p1_lt;
    @!%p1_ok bra LOAD_X2;
    cvt.u32.s32 %idx_u, %i1;
    add.u32 %elem, %chan_base, %idx_u;
    mul.wide.u32 %off, %elem, 4;
    add.u64 %addr, %inp, %off;
    ld.global.f32 %x1, [%addr];
LOAD_X2:

    mov.f32 %x2, 0f00000000;
    setp.ge.s32 %p2_ge0, %i2, 0;
    setp.lt.s32 %p2_lt, %i2, %in_len_s;
    and.pred %p2_ok, %p2_ge0, %p2_lt;
    @!%p2_ok bra ACCUM;
    cvt.u32.s32 %idx_u, %i2;
    add.u32 %elem, %chan_base, %idx_u;
    mul.wide.u32 %off, %elem, 4;
    add.u64 %addr, %inp, %off;
    ld.global.f32 %x2, [%addr];
ACCUM:

    fma.rn.f32 %sum, %x0, %w0, %sum;
    fma.rn.f32 %sum, %x1, %w1, %sum;
    fma.rn.f32 %sum, %x2, %w2, %sum;

    add.u32 %ic, %ic, 1;
    bra LOOP_IC;

STORE:
    mad.lo.u32 %elem, %oc, %out_len, %j;
    mul.wide.u32 %off, %elem, 4;
    add.u64 %addr, %outp, %off;
    st.global.f32 [%addr], %sum;
DONE:
    ret;
}
`

// Conv1DK3S2PTX is a Conv1D kernel for kernel_size=3, stride=2 with zero-padding.
// Grid: (ceil(out_length/256), out_channels, 1), Block: (256, 1, 1)
// Input index for output j and kernel tap k is j*2 + k - 1.
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
    .reg .pred %p_oob, %p_bias_nil, %p_loop_done;
    .reg .pred %p0_ge0, %p0_lt, %p0_ok, %p1_ge0, %p1_lt, %p1_ok, %p2_ge0, %p2_lt, %p2_ok;
    .reg .u32 %oc, %j, %ic, %tidx, %bid;
    .reg .u32 %in_ch, %in_len, %out_len;
    .reg .u32 %elem, %chan_base, %w_base, %idx_u;
    .reg .s32 %j_s, %j2_s, %in_len_s, %i0, %i1, %i2;
    .reg .u64 %outp, %inp, %wp, %biasp, %addr, %off;
    .reg .f32 %sum, %x0, %x1, %x2, %w0, %w1, %w2;

    ld.param.u64 %outp, [out_ptr];
    ld.param.u64 %inp, [in_ptr];
    ld.param.u64 %wp, [wt_ptr];
    ld.param.u64 %biasp, [bias_ptr];
    ld.param.u32 %in_ch, [in_channels];
    ld.param.u32 %in_len, [in_length];
    ld.param.u32 %out_len, [out_length];

    mov.u32 %oc, %ctaid.y;
    mov.u32 %tidx, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %j, %bid, 256, %tidx;
    setp.ge.u32 %p_oob, %j, %out_len;
    @%p_oob bra DONE;

    mov.f32 %sum, 0f00000000;
    setp.eq.u64 %p_bias_nil, %biasp, 0;
    @%p_bias_nil bra BIAS_DONE;
    mul.wide.u32 %off, %oc, 4;
    add.u64 %addr, %biasp, %off;
    ld.global.f32 %sum, [%addr];
BIAS_DONE:

    cvt.s32.u32 %j_s, %j;
    cvt.s32.u32 %in_len_s, %in_len;
    shl.b32 %j2_s, %j_s, 1;
    add.s32 %i0, %j2_s, -1;
    mov.s32 %i1, %j2_s;
    add.s32 %i2, %j2_s, 1;

    mov.u32 %ic, 0;
LOOP_IC:
    setp.ge.u32 %p_loop_done, %ic, %in_ch;
    @%p_loop_done bra STORE;

    mul.lo.u32 %w_base, %oc, %in_ch;
    add.u32 %w_base, %w_base, %ic;
    mul.lo.u32 %w_base, %w_base, 3;

    mul.wide.u32 %off, %w_base, 4;
    add.u64 %addr, %wp, %off;
    ld.global.f32 %w0, [%addr];
    add.u32 %elem, %w_base, 1;
    mul.wide.u32 %off, %elem, 4;
    add.u64 %addr, %wp, %off;
    ld.global.f32 %w1, [%addr];
    add.u32 %elem, %w_base, 2;
    mul.wide.u32 %off, %elem, 4;
    add.u64 %addr, %wp, %off;
    ld.global.f32 %w2, [%addr];

    mul.lo.u32 %chan_base, %ic, %in_len;

    mov.f32 %x0, 0f00000000;
    setp.ge.s32 %p0_ge0, %i0, 0;
    setp.lt.s32 %p0_lt, %i0, %in_len_s;
    and.pred %p0_ok, %p0_ge0, %p0_lt;
    @!%p0_ok bra LOAD_X1;
    cvt.u32.s32 %idx_u, %i0;
    add.u32 %elem, %chan_base, %idx_u;
    mul.wide.u32 %off, %elem, 4;
    add.u64 %addr, %inp, %off;
    ld.global.f32 %x0, [%addr];
LOAD_X1:

    mov.f32 %x1, 0f00000000;
    setp.ge.s32 %p1_ge0, %i1, 0;
    setp.lt.s32 %p1_lt, %i1, %in_len_s;
    and.pred %p1_ok, %p1_ge0, %p1_lt;
    @!%p1_ok bra LOAD_X2;
    cvt.u32.s32 %idx_u, %i1;
    add.u32 %elem, %chan_base, %idx_u;
    mul.wide.u32 %off, %elem, 4;
    add.u64 %addr, %inp, %off;
    ld.global.f32 %x1, [%addr];
LOAD_X2:

    mov.f32 %x2, 0f00000000;
    setp.ge.s32 %p2_ge0, %i2, 0;
    setp.lt.s32 %p2_lt, %i2, %in_len_s;
    and.pred %p2_ok, %p2_ge0, %p2_lt;
    @!%p2_ok bra ACCUM;
    cvt.u32.s32 %idx_u, %i2;
    add.u32 %elem, %chan_base, %idx_u;
    mul.wide.u32 %off, %elem, 4;
    add.u64 %addr, %inp, %off;
    ld.global.f32 %x2, [%addr];
ACCUM:

    fma.rn.f32 %sum, %x0, %w0, %sum;
    fma.rn.f32 %sum, %x1, %w1, %sum;
    fma.rn.f32 %sum, %x2, %w2, %sum;

    add.u32 %ic, %ic, 1;
    bra LOOP_IC;

STORE:
    mad.lo.u32 %elem, %oc, %out_len, %j;
    mul.wide.u32 %off, %elem, 4;
    add.u64 %addr, %outp, %off;
    st.global.f32 [%addr], %sum;
DONE:
    ret;
}
`
