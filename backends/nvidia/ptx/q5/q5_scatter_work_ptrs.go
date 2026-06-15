package q5

const GemvQ5_0ScatterByWorkPtrsPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gemv_q5_0_scatter_by_work_ptrs(
    .param .u64 param_x,
    .param .u64 param_work_active,
    .param .u64 param_work_pos,
    .param .u64 param_work_weight,
    .param .u64 param_q_ptrs,
    .param .u64 param_high_ptrs,
    .param .u64 param_scale_ptrs,
    .param .u64 param_dst,
    .param .u32 param_inDim,
    .param .u32 param_expertOutDim,
    .param .u32 param_workLen,
    .param .u32 param_activeExperts
) {
    .reg .u32 %r<56>;
    .reg .u64 %rd<42>;
    .reg .f32 %f<16>;
    .reg .pred %p<7>;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ctaid.y;
    mov.u32 %r2, %tid.x;
    ld.param.u32 %r3, [param_inDim];
    ld.param.u32 %r4, [param_expertOutDim];
    ld.param.u32 %r5, [param_workLen];
    ld.param.u32 %r6, [param_activeExperts];
    setp.ge.u32 %p0, %r0, %r4;
    setp.ge.u32 %p1, %r1, %r5;
    or.pred %p0, %p0, %p1;
    @%p0 bra done;

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_work_active];
    ld.param.u64 %rd2, [param_work_pos];
    ld.param.u64 %rd3, [param_work_weight];
    ld.param.u64 %rd4, [param_q_ptrs];
    ld.param.u64 %rd5, [param_high_ptrs];
    ld.param.u64 %rd6, [param_scale_ptrs];
    ld.param.u64 %rd7, [param_dst];

    mul.wide.u32 %rd8, %r1, 4;
    add.u64 %rd9, %rd1, %rd8;
    ld.global.u32 %r7, [%rd9];
    setp.ge.u32 %p2, %r7, %r6;
    @%p2 bra done;

    mul.wide.u32 %rd26, %r7, 8;
    add.u64 %rd27, %rd4, %rd26;
    ld.global.u64 %rd28, [%rd27]; // q base
    add.u64 %rd29, %rd5, %rd26;
    ld.global.u64 %rd30, [%rd29]; // high base
    add.u64 %rd31, %rd6, %rd26;
    ld.global.u64 %rd32, [%rd31]; // scale base

    shr.u32 %r8, %r3, 5;
    mul.lo.u32 %r9, %r1, %r3;
    mov.f32 %f0, 0f00000000;
    mov.u32 %r10, %r2;
loop_k:
    setp.ge.u32 %p3, %r10, %r3;
    @%p3 bra done_k;
    shr.u32 %r11, %r10, 5;
    and.b32 %r12, %r10, 31;
    mad.lo.u32 %r13, %r0, %r8, %r11;

    mul.wide.u32 %rd10, %r13, 4;
    add.u64 %rd11, %rd32, %rd10;
    ld.global.f32 %f1, [%rd11];
    add.u64 %rd12, %rd30, %rd10;
    ld.global.u32 %r14, [%rd12];

    shr.u32 %r15, %r12, 4;
    and.b32 %r16, %r12, 15;
    mad.lo.u32 %r17, %r13, 16, %r16;
    mul.wide.u32 %rd13, %r17, 1;
    add.u64 %rd14, %rd28, %rd13;
    ld.global.u8 %r18, [%rd14];
    setp.eq.u32 %p4, %r15, 0;
    @%p4 bra q_low;
    shr.u32 %r19, %r18, 4;
    bra q_nib;
q_low:
    and.b32 %r19, %r18, 15;
q_nib:
    mov.u32 %r20, 1;
    shl.b32 %r20, %r20, %r12;
    and.b32 %r21, %r14, %r20;
    setp.eq.u32 %p5, %r21, 0;
    @%p5 bra no_high;
    or.b32 %r19, %r19, 16;
no_high:
    add.s32 %r22, %r19, -16;
    cvt.rn.f32.s32 %f2, %r22;
    mul.f32 %f2, %f2, %f1;

    add.u32 %r23, %r9, %r10;
    mul.wide.u32 %rd15, %r23, 4;
    add.u64 %rd16, %rd0, %rd15;
    ld.global.f32 %f3, [%rd16];
    fma.rn.f32 %f0, %f2, %f3, %f0;

    add.u32 %r10, %r10, 256;
    bra loop_k;

done_k:
    mov.u64 %rd17, sdata;
    mul.wide.u32 %rd18, %r2, 4;
    add.u64 %rd19, %rd17, %rd18;
    st.shared.f32 [%rd19], %f0;
    bar.sync 0;
    mov.u32 %r24, 128;
reduce:
    setp.ge.u32 %p6, %r2, %r24;
    @%p6 bra red_skip;
    add.u32 %r25, %r2, %r24;
    mul.wide.u32 %rd20, %r25, 4;
    add.u64 %rd21, %rd17, %rd20;
    ld.shared.f32 %f4, [%rd21];
    ld.shared.f32 %f5, [%rd19];
    add.f32 %f5, %f5, %f4;
    st.shared.f32 [%rd19], %f5;
red_skip:
    bar.sync 0;
    shr.u32 %r24, %r24, 1;
    setp.gt.u32 %p6, %r24, 0;
    @%p6 bra reduce;

    setp.ne.u32 %p6, %r2, 0;
    @%p6 bra done;
    ld.shared.f32 %f6, [sdata];
    add.u64 %rd22, %rd2, %rd8;
    ld.global.u32 %r26, [%rd22];
    add.u64 %rd23, %rd3, %rd8;
    ld.global.f32 %f7, [%rd23];
    mad.lo.u32 %r27, %r26, %r4, %r0;
    mul.wide.u32 %rd24, %r27, 4;
    add.u64 %rd25, %rd7, %rd24;
    mul.f32 %f8, %f6, %f7;
    atom.global.add.f32 %f9, [%rd25], %f8;

done:
    ret;
}
`
