package q5

// GemvQ5KBatchPTX computes row-major Q5_K matrix products for a batch of F32
// activation rows. The packed layout matches GGUF block_q5_K exactly.
const GemvQ5KBatchPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gemv_q5_k_batch(
    .param .u64 param_x,
    .param .u64 param_raw,
    .param .u64 param_out,
    .param .u32 param_inDim,
    .param .u32 param_outDim,
    .param .u32 param_batch
) {
    .reg .u32 %r<64>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<16>;
    .reg .b16 %h<4>;
    .reg .pred %p<8>;
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
    shr.u32 %r12, %r11, 6;
    shr.u32 %r13, %r11, 5;
    and.b32 %r14, %r11, 31;
    and.b32 %r15, %r13, 1;

    add.u32 %r16, %r7, %r10;
    mul.lo.u32 %r17, %r16, 176;
    mul.wide.u32 %rd3, %r17, 1;
    add.u64 %rd4, %rd1, %rd3;

    ld.global.b16 %h0, [%rd4];
    cvt.f32.f16 %f1, %h0;
    add.u64 %rd5, %rd4, 2;
    ld.global.b16 %h1, [%rd5];
    cvt.f32.f16 %f2, %h1;

    add.u64 %rd6, %rd4, 4;
    cvt.u64.u32 %rd7, %r13;
    add.u64 %rd7, %rd6, %rd7;
    ld.global.u8 %r20, [%rd7];
    setp.lt.u32 %p3, %r13, 4;
    @%p3 bra scale_low;
    add.u32 %r21, %r13, -4;
    add.u32 %r22, %r13, 4;
    cvt.u64.u32 %rd8, %r22;
    add.u64 %rd8, %rd6, %rd8;
    ld.global.u8 %r23, [%rd8];
    and.b32 %r24, %r23, 15;
    cvt.u64.u32 %rd9, %r21;
    add.u64 %rd9, %rd6, %rd9;
    ld.global.u8 %r25, [%rd9];
    shr.u32 %r25, %r25, 6;
    shl.b32 %r25, %r25, 4;
    or.b32 %r26, %r24, %r25;
    add.u32 %r27, %r21, 4;
    cvt.u64.u32 %rd10, %r27;
    add.u64 %rd10, %rd6, %rd10;
    ld.global.u8 %r28, [%rd10];
    shr.u32 %r29, %r23, 4;
    shr.u32 %r30, %r28, 6;
    shl.b32 %r30, %r30, 4;
    or.b32 %r31, %r29, %r30;
    bra scale_done;
scale_low:
    and.b32 %r26, %r20, 63;
    add.u32 %r27, %r13, 4;
    cvt.u64.u32 %rd10, %r27;
    add.u64 %rd10, %rd6, %rd10;
    ld.global.u8 %r28, [%rd10];
    and.b32 %r31, %r28, 63;
scale_done:
    cvt.rn.f32.u32 %f3, %r26;
    cvt.rn.f32.u32 %f4, %r31;
    mul.f32 %f3, %f3, %f1;
    mul.f32 %f4, %f4, %f2;

    add.u32 %r32, %r17, 16;
    add.u32 %r32, %r32, %r14;
    mul.wide.u32 %rd11, %r32, 1;
    add.u64 %rd12, %rd1, %rd11;
    ld.global.u8 %r33, [%rd12];
    mov.u32 %r34, 1;
    shl.b32 %r35, %r12, 1;
    add.u32 %r35, %r35, %r15;
    shl.b32 %r34, %r34, %r35;
    and.b32 %r36, %r33, %r34;

    shl.b32 %r37, %r12, 5;
    add.u32 %r37, %r37, %r14;
    add.u32 %r37, %r37, %r17;
    add.u32 %r37, %r37, 48;
    mul.wide.u32 %rd13, %r37, 1;
    add.u64 %rd14, %rd1, %rd13;
    ld.global.u8 %r38, [%rd14];
    setp.eq.u32 %p4, %r15, 0;
    @%p4 bra q_low;
    shr.u32 %r39, %r38, 4;
    bra q_nib_done;
q_low:
    and.b32 %r39, %r38, 15;
q_nib_done:
    setp.eq.u32 %p5, %r36, 0;
    @%p5 bra q_high_done;
    add.u32 %r39, %r39, 16;
q_high_done:
    cvt.rn.f32.u32 %f5, %r39;
    neg.f32 %f4, %f4;
    fma.rn.f32 %f6, %f5, %f3, %f4;

    add.u32 %r40, %r8, %r9;
    mul.wide.u32 %rd15, %r40, 4;
    add.u64 %rd16, %rd0, %rd15;
    ld.global.f32 %f7, [%rd16];
    fma.rn.f32 %f0, %f6, %f7, %f0;
    add.u32 %r9, %r9, 256;
    bra loop_k;

done_k:
    mov.u64 %rd17, sdata;
    mul.wide.u32 %rd18, %r2, 4;
    add.u64 %rd19, %rd17, %rd18;
    st.shared.f32 [%rd19], %f0;
    bar.sync 0;
    mov.u32 %r41, 128;
reduce:
    setp.ge.u32 %p6, %r2, %r41;
    @%p6 bra reduce_skip;
    add.u32 %r42, %r2, %r41;
    mul.wide.u32 %rd20, %r42, 4;
    add.u64 %rd21, %rd17, %rd20;
    ld.shared.f32 %f8, [%rd21];
    ld.shared.f32 %f9, [%rd19];
    add.f32 %f9, %f9, %f8;
    st.shared.f32 [%rd19], %f9;
reduce_skip:
    bar.sync 0;
    shr.u32 %r41, %r41, 1;
    setp.gt.u32 %p7, %r41, 0;
    @%p7 bra reduce;
    setp.ne.u32 %p7, %r2, 0;
    @%p7 bra done;
    ld.shared.f32 %f10, [sdata];
    mad.lo.u32 %r43, %r1, %r4, %r0;
    mul.wide.u32 %rd22, %r43, 4;
    add.u64 %rd23, %rd2, %rd22;
    st.global.f32 [%rd23], %f10;
done:
    ret;
}
`
