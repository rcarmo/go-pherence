package q8

const GemvQ8_0ScatterByWorkPtrsPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gemv_q8_0_scatter_by_work_ptrs(
    .param .u64 param_x,
    .param .u64 param_work_active,
    .param .u64 param_work_pos,
    .param .u64 param_work_weight,
    .param .u64 param_q_ptrs,
    .param .u64 param_scale_ptrs,
    .param .u64 param_dst,
    .param .u32 param_inDim,
    .param .u32 param_expertOutDim,
    .param .u32 param_workLen,
    .param .u32 param_activeExperts
) {
    .reg .u32 %r<50>;
    .reg .u64 %rd<34>;
    .reg .f32 %f<16>;
    .reg .pred %p<6>;
    .shared .align 4 .f32 sdata[256];

    mov.u32 %r0, %ctaid.x;   // hidden/out row inside one expert
    mov.u32 %r1, %ctaid.y;   // work row
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
    ld.param.u64 %rd5, [param_scale_ptrs];
    ld.param.u64 %rd6, [param_dst];

    mul.wide.u32 %rd7, %r1, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u32 %r7, [%rd8]; // active expert index
    setp.ge.u32 %p2, %r7, %r6;
    @%p2 bra done;

    mul.wide.u32 %rd24, %r7, 8;
    add.u64 %rd25, %rd4, %rd24;
    ld.global.u64 %rd26, [%rd25]; // q base for active expert
    add.u64 %rd27, %rd5, %rd24;
    ld.global.u64 %rd28, [%rd27]; // scale base for active expert

    mul.lo.u32 %r9, %r1, %r3; // x base = work * inDim
    mov.f32 %f0, 0f00000000;
    mov.u32 %r10, %r2;
loop_k:
    setp.ge.u32 %p3, %r10, %r3;
    @%p3 bra done_k;
    shr.u32 %r11, %r10, 5;
    shr.u32 %r12, %r3, 5;
    mad.lo.u32 %r13, %r0, %r12, %r11;
    mul.wide.u32 %rd9, %r13, 4;
    add.u64 %rd10, %rd28, %rd9;
    ld.global.f32 %f1, [%rd10];

    mad.lo.u32 %r14, %r0, %r3, %r10;
    mul.wide.u32 %rd11, %r14, 1;
    add.u64 %rd12, %rd26, %rd11;
    ld.global.u8 %r15, [%rd12];
    shl.b32 %r16, %r15, 24;
    shr.s32 %r16, %r16, 24;
    cvt.rn.f32.s32 %f2, %r16;
    mul.f32 %f2, %f2, %f1;

    add.u32 %r17, %r9, %r10;
    mul.wide.u32 %rd13, %r17, 4;
    add.u64 %rd14, %rd0, %rd13;
    ld.global.f32 %f3, [%rd14];
    fma.rn.f32 %f0, %f2, %f3, %f0;

    add.u32 %r10, %r10, 256;
    bra loop_k;
done_k:
    mov.u64 %rd15, sdata;
    mul.wide.u32 %rd16, %r2, 4;
    add.u64 %rd17, %rd15, %rd16;
    st.shared.f32 [%rd17], %f0;
    bar.sync 0;

    mov.u32 %r18, 128;
reduce:
    setp.ge.u32 %p4, %r2, %r18;
    @%p4 bra red_skip;
    add.u32 %r19, %r2, %r18;
    mul.wide.u32 %rd18, %r19, 4;
    add.u64 %rd19, %rd15, %rd18;
    ld.shared.f32 %f4, [%rd19];
    ld.shared.f32 %f5, [%rd17];
    add.f32 %f5, %f5, %f4;
    st.shared.f32 [%rd17], %f5;
red_skip:
    bar.sync 0;
    shr.u32 %r18, %r18, 1;
    setp.gt.u32 %p5, %r18, 0;
    @%p5 bra reduce;

    setp.ne.u32 %p4, %r2, 0;
    @%p4 bra done;
    ld.shared.f32 %f6, [sdata];

    add.u64 %rd20, %rd2, %rd7;
    ld.global.u32 %r20, [%rd20];
    add.u64 %rd21, %rd3, %rd7;
    ld.global.f32 %f7, [%rd21];
    mad.lo.u32 %r21, %r20, %r4, %r0;
    mul.wide.u32 %rd22, %r21, 4;
    add.u64 %rd23, %rd6, %rd22;
    mul.f32 %f9, %f6, %f7;
    atom.global.add.f32 %f10, [%rd23], %f9;

done:
    ret;
}
`
