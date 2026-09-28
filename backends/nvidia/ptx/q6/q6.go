package q6

// GemvQ6KBatchPTX computes row-major Q6_K matrix products for a batch of F32
// activation rows. The packed layout matches GGUF block_q6_K exactly.
const GemvQ6KBatchPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gemv_q6_k_batch(
    .param .u64 param_x,
    .param .u64 param_raw,
    .param .u64 param_out,
    .param .u32 param_inDim,
    .param .u32 param_outDim,
    .param .u32 param_batch
) {
    .reg .u32 %r<56>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<12>;
    .reg .b16 %h<2>;
    .reg .pred %p<7>;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ctaid.y;
    mov.u32 %r2, %tid.x;
    ld.param.u32 %r3, [param_inDim];
    ld.param.u32 %r4, [param_outDim];
    ld.param.u32 %r5, [param_batch];
    setp.ge.u32 %p0, %r0, %r4;
    setp.ge.u32 %p1, %r1, %r5;
    or.pred %p0, %p0, %p1;
    @%p0 bra done;
    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_raw];
    ld.param.u64 %rd2, [param_out];
    shr.u32 %r6, %r3, 8;
    mul.lo.u32 %r7, %r0, %r6;
    mul.lo.u32 %r8, %r1, %r3;
    mov.f32 %f0, 0f00000000;
    mov.u32 %r9, %r2;
loop_k:
    setp.ge.u32 %p2, %r9, %r3;
    @%p2 bra done_k;
    shr.u32 %r10, %r9, 8;
    and.b32 %r11, %r9, 255;
    shr.u32 %r12, %r11, 7;
    shr.u32 %r13, %r11, 5;
    and.b32 %r13, %r13, 3;
    and.b32 %r14, %r11, 31;
    add.u32 %r15, %r7, %r10;
    mul.lo.u32 %r16, %r15, 210;

    shl.b32 %r17, %r12, 6;
    and.b32 %r18, %r13, 1;
    shl.b32 %r18, %r18, 5;
    add.u32 %r17, %r17, %r18;
    add.u32 %r17, %r17, %r14;
    add.u32 %r17, %r17, %r16;
    mul.wide.u32 %rd3, %r17, 1;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u8 %r19, [%rd4];
    setp.lt.u32 %p3, %r13, 2;
    @%p3 bra ql_low;
    shr.u32 %r20, %r19, 4;
    bra ql_done;
ql_low:
    and.b32 %r20, %r19, 15;
ql_done:
    shl.b32 %r21, %r12, 5;
    add.u32 %r21, %r21, %r14;
    add.u32 %r21, %r21, %r16;
    add.u32 %r21, %r21, 128;
    mul.wide.u32 %rd5, %r21, 1;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u8 %r22, [%rd6];
    shl.b32 %r23, %r13, 1;
    shr.b32 %r24, %r22, %r23;
    and.b32 %r24, %r24, 3;
    shl.b32 %r24, %r24, 4;
    or.b32 %r25, %r20, %r24;
    add.s32 %r25, %r25, -32;

    shl.b32 %r26, %r12, 3;
    shr.u32 %r27, %r14, 4;
    shl.b32 %r28, %r13, 1;
    add.u32 %r26, %r26, %r27;
    add.u32 %r26, %r26, %r28;
    add.u32 %r26, %r26, %r16;
    add.u32 %r26, %r26, 192;
    mul.wide.u32 %rd7, %r26, 1;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.s8 %r29, [%rd8];
    add.u32 %r30, %r16, 208;
    mul.wide.u32 %rd9, %r30, 1;
    add.u64 %rd10, %rd1, %rd9;
    ld.global.b16 %h0, [%rd10];
    cvt.f32.f16 %f1, %h0;
    cvt.rn.f32.s32 %f2, %r29;
    cvt.rn.f32.s32 %f3, %r25;
    mul.f32 %f4, %f1, %f2;
    mul.f32 %f4, %f4, %f3;

    add.u32 %r32, %r8, %r9;
    mul.wide.u32 %rd11, %r32, 4;
    add.u64 %rd12, %rd0, %rd11;
    ld.global.f32 %f5, [%rd12];
    fma.rn.f32 %f0, %f4, %f5, %f0;
    add.u32 %r9, %r9, 256;
    bra loop_k;
done_k:
    mov.u64 %rd13, sdata;
    mul.wide.u32 %rd14, %r2, 4;
    add.u64 %rd15, %rd13, %rd14;
    st.shared.f32 [%rd15], %f0;
    bar.sync 0;
    mov.u32 %r33, 128;
reduce:
    setp.ge.u32 %p4, %r2, %r33;
    @%p4 bra reduce_skip;
    add.u32 %r34, %r2, %r33;
    mul.wide.u32 %rd16, %r34, 4;
    add.u64 %rd17, %rd13, %rd16;
    ld.shared.f32 %f6, [%rd17];
    ld.shared.f32 %f7, [%rd15];
    add.f32 %f7, %f7, %f6;
    st.shared.f32 [%rd15], %f7;
reduce_skip:
    bar.sync 0;
    shr.u32 %r33, %r33, 1;
    setp.gt.u32 %p5, %r33, 0;
    @%p5 bra reduce;
    setp.ne.u32 %p6, %r2, 0;
    @%p6 bra done;
    ld.shared.f32 %f8, [sdata];
    mad.lo.u32 %r35, %r1, %r4, %r0;
    mul.wide.u32 %rd18, %r35, 4;
    add.u64 %rd19, %rd2, %rd18;
    st.global.f32 [%rd19], %f8;
done:
    ret;
}
`
