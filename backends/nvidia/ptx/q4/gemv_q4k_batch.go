package q4

const GemvQ4KBatchPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gemv_q4_k_batch(
    .param .u64 param_x,
    .param .u64 param_q,
    .param .u64 param_scales,
    .param .u64 param_mins,
    .param .u64 param_out,
    .param .u32 param_inDim,
    .param .u32 param_outDim,
    .param .u32 param_batch
) {
    .reg .u32 %r<54>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<16>;
    .reg .pred %p<7>;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;   // output row
    mov.u32 %r30, %ctaid.y;  // batch row
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [param_inDim];
    ld.param.u32 %r3, [param_outDim];
    ld.param.u32 %r31, [param_batch];
    setp.ge.u32 %p0, %r0, %r3;
    setp.ge.u32 %p6, %r30, %r31;
    or.pred %p0, %p0, %p6;
    @%p0 bra done;

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_q];
    ld.param.u64 %rd2, [param_scales];
    ld.param.u64 %rd3, [param_mins];
    ld.param.u64 %rd4, [param_out];

    shr.u32 %r4, %r2, 8;     // nBlocks = inDim / 256
    mul.lo.u32 %r32, %r30, %r2; // x batch base
    mul.lo.u32 %r33, %r30, %r3; // out batch base
    mov.f32 %f0, 0f00000000;
    mov.u32 %r5, %r1;
loop_k:
    setp.ge.u32 %p1, %r5, %r2;
    @%p1 bra done_k;

    shr.u32 %r6, %r5, 8;
    and.b32 %r7, %r5, 255;
    shr.u32 %r8, %r7, 5;
    and.b32 %r9, %r7, 31;
    shr.u32 %r10, %r8, 1;
    mad.lo.u32 %r11, %r10, 32, %r9;

    mad.lo.u32 %r12, %r0, %r4, %r6;
    shl.b32 %r13, %r12, 7;
    add.u32 %r13, %r13, %r11;
    mul.wide.u32 %rd5, %r13, 1;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u8 %r14, [%rd6];

    and.b32 %r15, %r8, 1;
    setp.eq.u32 %p2, %r15, 0;
    @%p2 bra low_nib;
    shr.u32 %r16, %r14, 4;
    bra got_nib;
low_nib:
    and.b32 %r16, %r14, 15;
got_nib:
    cvt.rn.f32.u32 %f1, %r16;

    shl.b32 %r17, %r12, 3;
    add.u32 %r17, %r17, %r8;
    mul.wide.u32 %rd7, %r17, 4;
    add.u64 %rd8, %rd2, %rd7;
    add.u64 %rd9, %rd3, %rd7;
    ld.global.f32 %f2, [%rd8];
    ld.global.f32 %f3, [%rd9];
    neg.f32 %f3, %f3;
    fma.rn.f32 %f4, %f1, %f2, %f3;

    add.u32 %r34, %r32, %r5;
    mul.wide.u32 %rd10, %r34, 4;
    add.u64 %rd11, %rd0, %rd10;
    ld.global.f32 %f5, [%rd11];
    fma.rn.f32 %f0, %f4, %f5, %f0;

    add.u32 %r5, %r5, 256;
    bra loop_k;
done_k:
    mov.u64 %rd12, sdata;
    mul.wide.u32 %rd13, %r1, 4;
    add.u64 %rd14, %rd12, %rd13;
    st.shared.f32 [%rd14], %f0;
    bar.sync 0;

    mov.u32 %r18, 128;
reduce_loop:
    setp.ge.u32 %p3, %r1, %r18;
    @%p3 bra reduce_skip;
    add.u32 %r19, %r1, %r18;
    mul.wide.u32 %rd15, %r19, 4;
    add.u64 %rd16, %rd12, %rd15;
    ld.shared.f32 %f6, [%rd16];
    ld.shared.f32 %f7, [%rd14];
    add.f32 %f7, %f7, %f6;
    st.shared.f32 [%rd14], %f7;
reduce_skip:
    bar.sync 0;
    shr.u32 %r18, %r18, 1;
    setp.gt.u32 %p4, %r18, 0;
    @%p4 bra reduce_loop;

    setp.ne.u32 %p5, %r1, 0;
    @%p5 bra done;
    ld.shared.f32 %f8, [sdata];
    add.u32 %r35, %r33, %r0;
    mul.wide.u32 %rd17, %r35, 4;
    add.u64 %rd18, %rd4, %rd17;
    st.global.f32 [%rd18], %f8;

done:
    ret;
}
`
