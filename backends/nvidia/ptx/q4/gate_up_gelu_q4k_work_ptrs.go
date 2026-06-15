package q4

const GateUpGELUQ4KByWorkPtrsPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gate_up_gelu_q4_k_by_work_ptrs(
    .param .u64 param_x,
    .param .u64 param_work_experts,
    .param .u64 param_q_ptrs,
    .param .u64 param_scale_ptrs,
    .param .u64 param_min_ptrs,
    .param .u64 param_out,
    .param .u32 param_inDim,
    .param .u32 param_intermediate,
    .param .u32 param_workLen,
    .param .u32 param_activeExperts
) {
    .reg .u32 %r<72>;
    .reg .u64 %rd<40>;
    .reg .f32 %f<32>;
    .reg .pred %p<9>;
    .shared .align 4 .f32 s_gate[256];
    .shared .align 4 .f32 s_up[256];

    mov.u32 %r0, %ctaid.x;   // intermediate row
    mov.u32 %r1, %ctaid.y;   // work row
    mov.u32 %r2, %tid.x;
    ld.param.u32 %r3, [param_inDim];
    ld.param.u32 %r4, [param_intermediate];
    ld.param.u32 %r5, [param_workLen];
    ld.param.u32 %r6, [param_activeExperts];
    setp.ge.u32 %p0, %r0, %r4;
    setp.ge.u32 %p1, %r1, %r5;
    or.pred %p0, %p0, %p1;
    @%p0 bra done;

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_work_experts];
    ld.param.u64 %rd2, [param_q_ptrs];
    ld.param.u64 %rd3, [param_scale_ptrs];
    ld.param.u64 %rd4, [param_min_ptrs];
    ld.param.u64 %rd5, [param_out];

    mul.wide.u32 %rd6, %r1, 4;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.u32 %r7, [%rd7]; // active expert index
    setp.ge.u32 %p2, %r7, %r6;
    @%p2 bra done;

    mul.wide.u32 %rd30, %r7, 8;
    add.u64 %rd31, %rd2, %rd30;
    ld.global.u64 %rd32, [%rd31]; // q base
    add.u64 %rd33, %rd3, %rd30;
    ld.global.u64 %rd34, [%rd33]; // scales base
    add.u64 %rd35, %rd4, %rd30;
    ld.global.u64 %rd36, [%rd35]; // mins base

    shr.u32 %r9, %r3, 8;       // nBlocks = inDim/256
    mul.lo.u32 %r10, %r1, %r3; // x batch base
    add.u32 %r12, %r0, 0;      // gate row inside one expert
    add.u32 %r13, %r12, %r4;   // up row inside one expert

    mov.f32 %f0, 0f00000000;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r14, %r2;
loop_k:
    setp.ge.u32 %p3, %r14, %r3;
    @%p3 bra done_k;
    shr.u32 %r15, %r14, 8;
    and.b32 %r16, %r14, 255;
    shr.u32 %r17, %r16, 5;
    and.b32 %r18, %r16, 31;
    shr.u32 %r19, %r17, 1;
    mad.lo.u32 %r20, %r19, 32, %r18;

    add.u32 %r21, %r10, %r14;
    mul.wide.u32 %rd8, %r21, 4;
    add.u64 %rd9, %rd0, %rd8;
    ld.global.f32 %f2, [%rd9];

    mad.lo.u32 %r22, %r12, %r9, %r15;
    shl.b32 %r23, %r22, 7;
    add.u32 %r23, %r23, %r20;
    mul.wide.u32 %rd10, %r23, 1;
    add.u64 %rd11, %rd32, %rd10;
    ld.global.u8 %r24, [%rd11];
    and.b32 %r25, %r17, 1;
    setp.eq.u32 %p4, %r25, 0;
    @%p4 bra gate_low;
    shr.u32 %r26, %r24, 4;
    bra gate_nib;
gate_low:
    and.b32 %r26, %r24, 15;
gate_nib:
    cvt.rn.f32.u32 %f3, %r26;
    shl.b32 %r27, %r22, 3;
    add.u32 %r27, %r27, %r17;
    mul.wide.u32 %rd12, %r27, 4;
    add.u64 %rd13, %rd34, %rd12;
    add.u64 %rd14, %rd36, %rd12;
    ld.global.f32 %f4, [%rd13];
    ld.global.f32 %f5, [%rd14];
    neg.f32 %f5, %f5;
    fma.rn.f32 %f6, %f3, %f4, %f5;
    fma.rn.f32 %f0, %f6, %f2, %f0;

    mad.lo.u32 %r28, %r13, %r9, %r15;
    shl.b32 %r29, %r28, 7;
    add.u32 %r29, %r29, %r20;
    mul.wide.u32 %rd15, %r29, 1;
    add.u64 %rd16, %rd32, %rd15;
    ld.global.u8 %r30, [%rd16];
    @%p4 bra up_low;
    shr.u32 %r31, %r30, 4;
    bra up_nib;
up_low:
    and.b32 %r31, %r30, 15;
up_nib:
    cvt.rn.f32.u32 %f7, %r31;
    shl.b32 %r32, %r28, 3;
    add.u32 %r32, %r32, %r17;
    mul.wide.u32 %rd17, %r32, 4;
    add.u64 %rd18, %rd34, %rd17;
    add.u64 %rd19, %rd36, %rd17;
    ld.global.f32 %f8, [%rd18];
    ld.global.f32 %f9, [%rd19];
    neg.f32 %f9, %f9;
    fma.rn.f32 %f10, %f7, %f8, %f9;
    fma.rn.f32 %f1, %f10, %f2, %f1;

    add.u32 %r14, %r14, 256;
    bra loop_k;

done_k:
    mov.u64 %rd20, s_gate;
    mov.u64 %rd21, s_up;
    mul.wide.u32 %rd22, %r2, 4;
    add.u64 %rd23, %rd20, %rd22;
    add.u64 %rd24, %rd21, %rd22;
    st.shared.f32 [%rd23], %f0;
    st.shared.f32 [%rd24], %f1;
    bar.sync 0;

    mov.u32 %r33, 128;
reduce_loop:
    setp.ge.u32 %p5, %r2, %r33;
    @%p5 bra reduce_skip;
    add.u32 %r34, %r2, %r33;
    mul.wide.u32 %rd25, %r34, 4;
    add.u64 %rd26, %rd20, %rd25;
    add.u64 %rd27, %rd21, %rd25;
    ld.shared.f32 %f11, [%rd26];
    ld.shared.f32 %f12, [%rd27];
    ld.shared.f32 %f13, [%rd23];
    ld.shared.f32 %f14, [%rd24];
    add.f32 %f13, %f13, %f11;
    add.f32 %f14, %f14, %f12;
    st.shared.f32 [%rd23], %f13;
    st.shared.f32 [%rd24], %f14;
reduce_skip:
    bar.sync 0;
    shr.u32 %r33, %r33, 1;
    setp.gt.u32 %p6, %r33, 0;
    @%p6 bra reduce_loop;

    setp.ne.u32 %p7, %r2, 0;
    @%p7 bra done;
    ld.shared.f32 %f15, [s_gate];
    ld.shared.f32 %f16, [s_up];

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

    mad.lo.u32 %r35, %r1, %r4, %r0;
    mul.wide.u32 %rd28, %r35, 4;
    add.u64 %rd29, %rd5, %rd28;
    st.global.f32 [%rd29], %f31;

done:
    ret;
}
`
