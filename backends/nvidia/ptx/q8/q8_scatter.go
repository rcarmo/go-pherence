package q8

const GemvQ8_0ScatterPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gemv_q8_0_scatter(
    .param .u64 param_x,
    .param .u64 param_q,
    .param .u64 param_scales,
    .param .u64 param_dst,
    .param .u64 param_pos,
    .param .u64 param_weights,
    .param .u32 param_inDim,
    .param .u32 param_outDim,
    .param .u32 param_batch
) {
    .reg .u32 %r<44>;
    .reg .u64 %rd<26>;
    .reg .f32 %f<16>;
    .reg .pred %p<5>;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;   // output row
    mov.u32 %r20, %ctaid.y;  // batch row
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [param_inDim];
    ld.param.u32 %r3, [param_outDim];
    ld.param.u32 %r21, [param_batch];
    setp.ge.u32 %p0, %r0, %r3;
    setp.ge.u32 %p4, %r20, %r21;
    or.pred %p0, %p0, %p4;
    @%p0 bra done;

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_q];
    ld.param.u64 %rd2, [param_scales];
    ld.param.u64 %rd3, [param_dst];
    ld.param.u64 %rd20, [param_pos];
    ld.param.u64 %rd21, [param_weights];

    mul.lo.u32 %r22, %r20, %r2; // x batch base

    mov.f32 %f0, 0f00000000;
    mov.u32 %r4, %r1;
loop_k:
    setp.ge.u32 %p1, %r4, %r2;
    @%p1 bra done_k;
    shr.u32 %r5, %r4, 5;
    shr.u32 %r6, %r2, 5;
    mad.lo.u32 %r7, %r0, %r6, %r5;
    mul.wide.u32 %rd4, %r7, 4;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.f32 %f1, [%rd5];

    mad.lo.u32 %r8, %r0, %r2, %r4;
    mul.wide.u32 %rd6, %r8, 1;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.u8 %r9, [%rd7];
    shl.b32 %r10, %r9, 24;
    shr.s32 %r10, %r10, 24;
    cvt.rn.f32.s32 %f2, %r10;
    mul.f32 %f2, %f2, %f1;

    add.u32 %r24, %r22, %r4;
    mul.wide.u32 %rd8, %r24, 4;
    add.u64 %rd9, %rd0, %rd8;
    ld.global.f32 %f3, [%rd9];
    fma.rn.f32 %f0, %f2, %f3, %f0;

    add.u32 %r4, %r4, 256;
    bra loop_k;
done_k:
    mov.u64 %rd10, sdata;
    mul.wide.u32 %rd11, %r1, 4;
    add.u64 %rd12, %rd10, %rd11;
    st.shared.f32 [%rd12], %f0;
    bar.sync 0;

    mov.u32 %r11, 128;
reduce_loop:
    setp.ge.u32 %p2, %r1, %r11;
    @%p2 bra reduce_skip;
    add.u32 %r12, %r1, %r11;
    mul.wide.u32 %rd13, %r12, 4;
    add.u64 %rd14, %rd10, %rd13;
    ld.shared.f32 %f4, [%rd14];
    ld.shared.f32 %f5, [%rd12];
    add.f32 %f5, %f5, %f4;
    st.shared.f32 [%rd12], %f5;
reduce_skip:
    bar.sync 0;
    shr.u32 %r11, %r11, 1;
    setp.gt.u32 %p3, %r11, 0;
    @%p3 bra reduce_loop;

    setp.ne.u32 %p1, %r1, 0;
    @%p1 bra done;
    ld.shared.f32 %f6, [sdata];

    // weighted scatter to dst[pos[batch], row]
    mul.wide.u32 %rd15, %r20, 4;
    add.u64 %rd16, %rd20, %rd15;
    ld.global.u32 %r30, [%rd16];
    add.u64 %rd17, %rd21, %rd15;
    ld.global.f32 %f7, [%rd17];
    mad.lo.u32 %r31, %r30, %r3, %r0;
    mul.wide.u32 %rd18, %r31, 4;
    add.u64 %rd19, %rd3, %rd18;
    ld.global.f32 %f8, [%rd19];
    fma.rn.f32 %f9, %f6, %f7, %f8;
    st.global.f32 [%rd19], %f9;

done:
    ret;
}
`
