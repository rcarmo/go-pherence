package q4

const GateUpGELUQ4KPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gate_up_gelu_q4_k(
    .param .u64 param_x,
    .param .u64 param_q,
    .param .u64 param_scales,
    .param .u64 param_mins,
    .param .u64 param_out,
    .param .u32 param_inDim,
    .param .u32 param_intermediate,
    .param .u32 param_batch
) {
    .reg .u32 %r<64>;
    .reg .u64 %rd<32>;
    .reg .f32 %f<32>;
    .reg .pred %p<8>;
    .shared .align 4 .f32 s_gate[256];
    .shared .align 4 .f32 s_up[256];

    mov.u32 %r0, %ctaid.x;   // intermediate row
    mov.u32 %r1, %ctaid.y;   // batch row
    mov.u32 %r2, %tid.x;
    ld.param.u32 %r3, [param_inDim];
    ld.param.u32 %r4, [param_intermediate];
    ld.param.u32 %r5, [param_batch];
    setp.ge.u32 %p0, %r0, %r4;
    setp.ge.u32 %p1, %r1, %r5;
    or.pred %p0, %p0, %p1;
    @%p0 bra done;

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_q];
    ld.param.u64 %rd2, [param_scales];
    ld.param.u64 %rd3, [param_mins];
    ld.param.u64 %rd4, [param_out];

    shr.u32 %r6, %r3, 8;       // nBlocks = inDim/256
    mul.lo.u32 %r7, %r1, %r3;  // x batch base
    add.u32 %r8, %r0, %r4;     // up row = gate row + intermediate

    mov.f32 %f0, 0f00000000;   // gate acc
    mov.f32 %f1, 0f00000000;   // up acc
    mov.u32 %r9, %r2;          // k
loop_k:
    setp.ge.u32 %p2, %r9, %r3;
    @%p2 bra done_k;

    shr.u32 %r10, %r9, 8;      // block
    and.b32 %r11, %r9, 255;    // within block
    shr.u32 %r12, %r11, 5;     // group
    and.b32 %r13, %r11, 31;    // group offset
    shr.u32 %r14, %r12, 1;
    mad.lo.u32 %r15, %r14, 32, %r13; // q byte index in block

    // load x
    add.u32 %r16, %r7, %r9;
    mul.wide.u32 %rd5, %r16, 4;
    add.u64 %rd6, %rd0, %rd5;
    ld.global.f32 %f2, [%rd6];

    // compute gate row value
    mad.lo.u32 %r17, %r0, %r6, %r10;
    shl.b32 %r18, %r17, 7;
    add.u32 %r18, %r18, %r15;
    mul.wide.u32 %rd7, %r18, 1;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u8 %r19, [%rd8];
    and.b32 %r20, %r12, 1;
    setp.eq.u32 %p3, %r20, 0;
    @%p3 bra gate_low;
    shr.u32 %r21, %r19, 4;
    bra gate_nib;
gate_low:
    and.b32 %r21, %r19, 15;
gate_nib:
    cvt.rn.f32.u32 %f3, %r21;
    shl.b32 %r22, %r17, 3;
    add.u32 %r22, %r22, %r12;
    mul.wide.u32 %rd9, %r22, 4;
    add.u64 %rd10, %rd2, %rd9;
    add.u64 %rd11, %rd3, %rd9;
    ld.global.f32 %f4, [%rd10];
    ld.global.f32 %f5, [%rd11];
    neg.f32 %f5, %f5;
    fma.rn.f32 %f6, %f3, %f4, %f5;
    fma.rn.f32 %f0, %f6, %f2, %f0;

    // compute up row value
    mad.lo.u32 %r23, %r8, %r6, %r10;
    shl.b32 %r24, %r23, 7;
    add.u32 %r24, %r24, %r15;
    mul.wide.u32 %rd12, %r24, 1;
    add.u64 %rd13, %rd1, %rd12;
    ld.global.u8 %r25, [%rd13];
    @%p3 bra up_low;
    shr.u32 %r26, %r25, 4;
    bra up_nib;
up_low:
    and.b32 %r26, %r25, 15;
up_nib:
    cvt.rn.f32.u32 %f7, %r26;
    shl.b32 %r27, %r23, 3;
    add.u32 %r27, %r27, %r12;
    mul.wide.u32 %rd14, %r27, 4;
    add.u64 %rd15, %rd2, %rd14;
    add.u64 %rd16, %rd3, %rd14;
    ld.global.f32 %f8, [%rd15];
    ld.global.f32 %f9, [%rd16];
    neg.f32 %f9, %f9;
    fma.rn.f32 %f10, %f7, %f8, %f9;
    fma.rn.f32 %f1, %f10, %f2, %f1;

    add.u32 %r9, %r9, 256;
    bra loop_k;

done_k:
    mov.u64 %rd17, s_gate;
    mov.u64 %rd18, s_up;
    mul.wide.u32 %rd19, %r2, 4;
    add.u64 %rd20, %rd17, %rd19;
    add.u64 %rd21, %rd18, %rd19;
    st.shared.f32 [%rd20], %f0;
    st.shared.f32 [%rd21], %f1;
    bar.sync 0;

    mov.u32 %r28, 128;
reduce_loop:
    setp.ge.u32 %p4, %r2, %r28;
    @%p4 bra reduce_skip;
    add.u32 %r29, %r2, %r28;
    mul.wide.u32 %rd22, %r29, 4;
    add.u64 %rd23, %rd17, %rd22;
    add.u64 %rd24, %rd18, %rd22;
    ld.shared.f32 %f11, [%rd23];
    ld.shared.f32 %f12, [%rd24];
    ld.shared.f32 %f13, [%rd20];
    ld.shared.f32 %f14, [%rd21];
    add.f32 %f13, %f13, %f11;
    add.f32 %f14, %f14, %f12;
    st.shared.f32 [%rd20], %f13;
    st.shared.f32 [%rd21], %f14;
reduce_skip:
    bar.sync 0;
    shr.u32 %r28, %r28, 1;
    setp.gt.u32 %p5, %r28, 0;
    @%p5 bra reduce_loop;

    setp.ne.u32 %p6, %r2, 0;
    @%p6 bra done;
    ld.shared.f32 %f15, [s_gate];
    ld.shared.f32 %f16, [s_up];

    // GELU tanh approximation
    mul.f32 %f17, %f15, %f15;
    mul.f32 %f18, %f17, %f15;
    mov.f32 %f19, 0f3d372713;
    fma.rn.f32 %f20, %f19, %f18, %f15;
    mov.f32 %f21, 0f3f4c422a;
    mul.f32 %f22, %f21, %f20;
    mov.f32 %f23, 0fc0000000;
    mul.f32 %f24, %f23, %f22;
    mov.f32 %f25, 0f3fb8aa3b;
    mul.f32 %f26, %f24, %f25;
    ex2.approx.ftz.f32 %f27, %f26;
    mov.f32 %f28, 0f3f800000;
    add.f32 %f29, %f28, %f27;
    rcp.approx.ftz.f32 %f29, %f29;
    mov.f32 %f30, 0f40000000;
    mul.f32 %f29, %f30, %f29;
    sub.f32 %f29, %f29, %f28;
    add.f32 %f29, %f28, %f29;
    mov.f32 %f30, 0f3f000000;
    mul.f32 %f31, %f30, %f15;
    mul.f32 %f31, %f31, %f29;
    mul.f32 %f31, %f31, %f16;

    mad.lo.u32 %r30, %r1, %r4, %r0;
    mul.wide.u32 %rd25, %r30, 4;
    add.u64 %rd26, %rd4, %rd25;
    st.global.f32 [%rd26], %f31;

done:
    ret;
}
`
