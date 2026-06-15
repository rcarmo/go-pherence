package q5

const GemvQ5_0ScatterByWorkPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gemv_q5_0_scatter_by_work(
    .param .u64 param_x,
    .param .u64 param_work_active,
    .param .u64 param_work_pos,
    .param .u64 param_work_weight,
    .param .u64 param_q,
    .param .u64 param_high,
    .param .u64 param_scale,
    .param .u64 param_dst,
    .param .u32 param_inDim,
    .param .u32 param_matrixRows,
    .param .u32 param_expertOutDim,
    .param .u32 param_workLen,
    .param .u32 param_activeExperts
) {
    .reg .u32 %r<56>;
    .reg .u64 %rd<36>;
    .reg .f32 %f<16>;
    .reg .pred %p<7>;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;   // output row inside expert
    mov.u32 %r1, %ctaid.y;   // work row
    mov.u32 %r2, %tid.x;
    ld.param.u32 %r3, [param_inDim];
    ld.param.u32 %r4, [param_matrixRows];
    ld.param.u32 %r5, [param_expertOutDim];
    ld.param.u32 %r6, [param_workLen];
    ld.param.u32 %r7, [param_activeExperts];
    setp.ge.u32 %p0, %r0, %r5;
    setp.ge.u32 %p1, %r1, %r6;
    or.pred %p0, %p0, %p1;
    @%p0 bra done;

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_work_active];
    ld.param.u64 %rd2, [param_work_pos];
    ld.param.u64 %rd3, [param_work_weight];
    ld.param.u64 %rd4, [param_q];
    ld.param.u64 %rd5, [param_high];
    ld.param.u64 %rd6, [param_scale];
    ld.param.u64 %rd7, [param_dst];

    mul.wide.u32 %rd8, %r1, 4;
    add.u64 %rd9, %rd1, %rd8;
    ld.global.u32 %r8, [%rd9]; // active expert index
    setp.ge.u32 %p2, %r8, %r7;
    @%p2 bra done;

    mad.lo.u32 %r9, %r8, %r5, %r0; // matrix row = active*expertOut + row
    setp.ge.u32 %p3, %r9, %r4;
    @%p3 bra done;
    shr.u32 %r10, %r3, 5;       // blocks = inDim/32
    mul.lo.u32 %r11, %r1, %r3;  // x base
    mov.f32 %f0, 0f00000000;
    mov.u32 %r12, %r2;
loop_k:
    setp.ge.u32 %p4, %r12, %r3;
    @%p4 bra done_k;
    shr.u32 %r13, %r12, 5;      // block
    and.b32 %r14, %r12, 31;     // elem
    mad.lo.u32 %r15, %r9, %r10, %r13;

    mul.wide.u32 %rd10, %r15, 4;
    add.u64 %rd11, %rd6, %rd10;
    ld.global.f32 %f1, [%rd11];
    add.u64 %rd12, %rd5, %rd10;
    ld.global.u32 %r16, [%rd12];

    shr.u32 %r17, %r14, 4;
    and.b32 %r18, %r14, 15;
    mad.lo.u32 %r19, %r15, 16, %r18;
    mul.wide.u32 %rd13, %r19, 1;
    add.u64 %rd14, %rd4, %rd13;
    ld.global.u8 %r20, [%rd14];
    setp.eq.u32 %p5, %r17, 0;
    @%p5 bra q_low;
    shr.u32 %r21, %r20, 4;
    bra q_nib;
q_low:
    and.b32 %r21, %r20, 15;
q_nib:
    mov.u32 %r22, 1;
    shl.b32 %r22, %r22, %r14;
    and.b32 %r23, %r16, %r22;
    setp.eq.u32 %p6, %r23, 0;
    @%p6 bra no_high;
    or.b32 %r21, %r21, 16;
no_high:
    add.s32 %r24, %r21, -16;
    cvt.rn.f32.s32 %f2, %r24;
    mul.f32 %f2, %f2, %f1;

    add.u32 %r25, %r11, %r12;
    mul.wide.u32 %rd15, %r25, 4;
    add.u64 %rd16, %rd0, %rd15;
    ld.global.f32 %f3, [%rd16];
    fma.rn.f32 %f0, %f2, %f3, %f0;

    add.u32 %r12, %r12, 256;
    bra loop_k;

done_k:
    mov.u64 %rd17, sdata;
    mul.wide.u32 %rd18, %r2, 4;
    add.u64 %rd19, %rd17, %rd18;
    st.shared.f32 [%rd19], %f0;
    bar.sync 0;
    mov.u32 %r26, 128;
reduce:
    setp.ge.u32 %p6, %r2, %r26;
    @%p6 bra red_skip;
    add.u32 %r27, %r2, %r26;
    mul.wide.u32 %rd20, %r27, 4;
    add.u64 %rd21, %rd17, %rd20;
    ld.shared.f32 %f4, [%rd21];
    ld.shared.f32 %f5, [%rd19];
    add.f32 %f5, %f5, %f4;
    st.shared.f32 [%rd19], %f5;
red_skip:
    bar.sync 0;
    shr.u32 %r26, %r26, 1;
    setp.gt.u32 %p6, %r26, 0;
    @%p6 bra reduce;

    setp.ne.u32 %p6, %r2, 0;
    @%p6 bra done;
    ld.shared.f32 %f6, [sdata];
    add.u64 %rd22, %rd2, %rd8;
    ld.global.u32 %r28, [%rd22];
    add.u64 %rd23, %rd3, %rd8;
    ld.global.f32 %f7, [%rd23];
    mad.lo.u32 %r29, %r28, %r5, %r0;
    mul.wide.u32 %rd24, %r29, 4;
    add.u64 %rd25, %rd7, %rd24;
    mul.f32 %f8, %f6, %f7;
    atom.global.add.f32 %f9, [%rd25], %f8;

done:
    ret;
}
`
