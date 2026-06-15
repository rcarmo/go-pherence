package q5

const GemvQ5_0BatchPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gemv_q5_0_batch(
    .param .u64 param_x,
    .param .u64 param_q,
    .param .u64 param_high,
    .param .u64 param_scale,
    .param .u64 param_out,
    .param .u32 param_inDim,
    .param .u32 param_outDim,
    .param .u32 param_batch
) {
    .reg .u32 %r<44>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<12>;
    .reg .pred %p<6>;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;   // output row
    mov.u32 %r1, %ctaid.y;   // batch row
    mov.u32 %r2, %tid.x;
    ld.param.u32 %r3, [param_inDim];
    ld.param.u32 %r4, [param_outDim];
    ld.param.u32 %r5, [param_batch];
    setp.ge.u32 %p0, %r0, %r4;
    setp.ge.u32 %p1, %r1, %r5;
    or.pred %p0, %p0, %p1;
    @%p0 bra done;

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_q];
    ld.param.u64 %rd2, [param_high];
    ld.param.u64 %rd3, [param_scale];
    ld.param.u64 %rd4, [param_out];

    shr.u32 %r6, %r3, 5;       // blocks = inDim/32
    mul.lo.u32 %r7, %r1, %r3;  // x base
    mov.f32 %f0, 0f00000000;
    mov.u32 %r8, %r2;
loop_k:
    setp.ge.u32 %p2, %r8, %r3;
    @%p2 bra done_k;
    shr.u32 %r9, %r8, 5;       // block
    and.b32 %r10, %r8, 31;     // elem in block

    mad.lo.u32 %r11, %r0, %r6, %r9;
    mul.wide.u32 %rd5, %r11, 4;
    add.u64 %rd6, %rd3, %rd5;
    ld.global.f32 %f1, [%rd6]; // scale
    add.u64 %rd7, %rd2, %rd5;
    ld.global.u32 %r12, [%rd7]; // high bits

    shr.u32 %r13, %r10, 4;     // 0 for low half, 1 for high half
    and.b32 %r14, %r10, 15;
    mad.lo.u32 %r15, %r11, 16, %r14;
    mul.wide.u32 %rd8, %r15, 1;
    add.u64 %rd9, %rd1, %rd8;
    ld.global.u8 %r16, [%rd9];
    setp.eq.u32 %p3, %r13, 0;
    @%p3 bra q_low;
    shr.u32 %r17, %r16, 4;
    bra q_nib;
q_low:
    and.b32 %r17, %r16, 15;
q_nib:
    mov.u32 %r18, 1;
    shl.b32 %r18, %r18, %r10;
    and.b32 %r19, %r12, %r18;
    setp.eq.u32 %p4, %r19, 0;
    @%p4 bra no_high;
    or.b32 %r17, %r17, 16;
no_high:
    add.s32 %r20, %r17, -16;
    cvt.rn.f32.s32 %f2, %r20;
    mul.f32 %f2, %f2, %f1;

    add.u32 %r21, %r7, %r8;
    mul.wide.u32 %rd10, %r21, 4;
    add.u64 %rd11, %rd0, %rd10;
    ld.global.f32 %f3, [%rd11];
    fma.rn.f32 %f0, %f2, %f3, %f0;

    add.u32 %r8, %r8, 256;
    bra loop_k;

done_k:
    mov.u64 %rd12, sdata;
    mul.wide.u32 %rd13, %r2, 4;
    add.u64 %rd14, %rd12, %rd13;
    st.shared.f32 [%rd14], %f0;
    bar.sync 0;

    mov.u32 %r22, 128;
reduce:
    setp.ge.u32 %p5, %r2, %r22;
    @%p5 bra red_skip;
    add.u32 %r23, %r2, %r22;
    mul.wide.u32 %rd15, %r23, 4;
    add.u64 %rd16, %rd12, %rd15;
    ld.shared.f32 %f4, [%rd16];
    ld.shared.f32 %f5, [%rd14];
    add.f32 %f5, %f5, %f4;
    st.shared.f32 [%rd14], %f5;
red_skip:
    bar.sync 0;
    shr.u32 %r22, %r22, 1;
    setp.gt.u32 %p5, %r22, 0;
    @%p5 bra reduce;

    setp.ne.u32 %p5, %r2, 0;
    @%p5 bra done;
    ld.shared.f32 %f6, [sdata];
    mad.lo.u32 %r24, %r1, %r4, %r0;
    mul.wide.u32 %rd17, %r24, 4;
    add.u64 %rd18, %rd4, %rd17;
    st.global.f32 [%rd18], %f6;

done:
    ret;
}
`
