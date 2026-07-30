package mlx

// MLXSelectedExpertPersistentPTX is an opt-in candidate kernel for selected
// expert projections that share one input vector and emit one contiguous
// [experts,outDim] output tile in a single launch.
var MLXSelectedExpertPersistentPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry mlx_selected_expert_gemv_persistent(
    .param .u64 x,
    .param .u64 q_ptrs,
    .param .u64 scale_ptrs,
    .param .u64 bias_ptrs,
    .param .u64 work_experts,
    .param .u64 claim,
    .param .u64 output,
    .param .u32 inDim,
    .param .u32 outDim,
    .param .u32 numGroups,
    .param .u32 groupSize,
    .param .u32 workLen,
    .param .u32 activeExperts
) {
    .reg .u32 %r<64>;
    .reg .u64 %rd<48>;
    .reg .f32 %f<16>;
    .reg .pred %p<8>;
    .shared .align 4 .f32 sdata[256];
    .shared .align 4 .u32 s_task;

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [inDim];
    ld.param.u32 %r2, [outDim];
    ld.param.u32 %r3, [numGroups];
    ld.param.u32 %r4, [groupSize];
    ld.param.u32 %r5, [workLen];
    ld.param.u32 %r6, [activeExperts];

    mul.lo.u32 %r7, %r5, %r2;
    shr.u32 %r8, %r1, 3;

    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd1, [q_ptrs];
    ld.param.u64 %rd2, [scale_ptrs];
    ld.param.u64 %rd3, [bias_ptrs];
    ld.param.u64 %rd4, [work_experts];
    ld.param.u64 %rd5, [claim];
    ld.param.u64 %rd6, [output];
    mov.u64 %rd40, sdata;
    mov.u64 %rd41, s_task;

claim_row:
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra claim_wait;
    mov.u32 %r9, 1;
    atom.global.add.u32 %r10, [%rd5], %r9;
    st.shared.u32 [%rd41], %r10;
claim_wait:
    bar.sync 0;

    ld.shared.u32 %r11, [%rd41];
    setp.ge.u32 %p1, %r11, %r7;
    @%p1 bra done;

    div.u32 %r12, %r11, %r2;
    rem.u32 %r13, %r11, %r2;

    mul.wide.u32 %rd7, %r12, 4;
    add.u64 %rd8, %rd4, %rd7;
    ld.global.u32 %r14, [%rd8];
    setp.ge.u32 %p2, %r14, %r6;
    @%p2 bra zero_reduce;

    mul.wide.u32 %rd9, %r14, 8;
    add.u64 %rd10, %rd1, %rd9;
    ld.global.u64 %rd11, [%rd10];
    add.u64 %rd12, %rd2, %rd9;
    ld.global.u64 %rd13, [%rd12];
    add.u64 %rd14, %rd3, %rd9;
    ld.global.u64 %rd15, [%rd14];

    mul.lo.u32 %r15, %r13, %r8;
    mul.wide.u32 %rd16, %r15, 4;
    add.u64 %rd11, %rd11, %rd16;

    mul.lo.u32 %r16, %r13, %r3;
    mul.wide.u32 %rd17, %r16, 4;
    add.u64 %rd13, %rd13, %rd17;
    add.u64 %rd15, %rd15, %rd17;

    mov.f32 %f0, 0f00000000;
    mov.u32 %r17, %r0;

pack_loop:
    setp.ge.u32 %p3, %r17, %r8;
    @%p3 bra reduce;

    mul.wide.u32 %rd18, %r17, 4;
    add.u64 %rd19, %rd11, %rd18;
    ld.global.u32 %r18, [%rd19];

    shl.b32 %r19, %r17, 3;
    div.u32 %r20, %r19, %r4;
    mul.wide.u32 %rd20, %r20, 4;
    add.u64 %rd21, %rd13, %rd20;
    ld.global.f32 %f1, [%rd21];
    add.u64 %rd22, %rd15, %rd20;
    ld.global.f32 %f2, [%rd22];

    mul.wide.u32 %rd23, %r19, 4;
    add.u64 %rd24, %rd0, %rd23;

    and.b32 %r21, %r18, 15;
    cvt.rn.f32.u32 %f3, %r21;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd24];
    fma.rn.f32 %f0, %f3, %f4, %f0;

    shr.u32 %r21, %r18, 4;
    and.b32 %r21, %r21, 15;
    cvt.rn.f32.u32 %f3, %r21;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd24+4];
    fma.rn.f32 %f0, %f3, %f4, %f0;

    shr.u32 %r21, %r18, 8;
    and.b32 %r21, %r21, 15;
    cvt.rn.f32.u32 %f3, %r21;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd24+8];
    fma.rn.f32 %f0, %f3, %f4, %f0;

    shr.u32 %r21, %r18, 12;
    and.b32 %r21, %r21, 15;
    cvt.rn.f32.u32 %f3, %r21;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd24+12];
    fma.rn.f32 %f0, %f3, %f4, %f0;

    shr.u32 %r21, %r18, 16;
    and.b32 %r21, %r21, 15;
    cvt.rn.f32.u32 %f3, %r21;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd24+16];
    fma.rn.f32 %f0, %f3, %f4, %f0;

    shr.u32 %r21, %r18, 20;
    and.b32 %r21, %r21, 15;
    cvt.rn.f32.u32 %f3, %r21;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd24+20];
    fma.rn.f32 %f0, %f3, %f4, %f0;

    shr.u32 %r21, %r18, 24;
    and.b32 %r21, %r21, 15;
    cvt.rn.f32.u32 %f3, %r21;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd24+24];
    fma.rn.f32 %f0, %f3, %f4, %f0;

    shr.u32 %r21, %r18, 28;
    cvt.rn.f32.u32 %f3, %r21;
    fma.rn.f32 %f3, %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd24+28];
    fma.rn.f32 %f0, %f3, %f4, %f0;

    add.u32 %r17, %r17, 256;
    bra pack_loop;

zero_reduce:
    mov.f32 %f0, 0f00000000;

reduce:
    mul.wide.u32 %rd25, %r0, 4;
    add.u64 %rd26, %rd40, %rd25;
    st.shared.f32 [%rd26], %f0;
    bar.sync 0;

    mov.u32 %r22, 128;
reduce_loop:
    setp.ge.u32 %p4, %r0, %r22;
    @%p4 bra reduce_skip;
    add.u32 %r23, %r0, %r22;
    mul.wide.u32 %rd27, %r23, 4;
    add.u64 %rd28, %rd40, %rd27;
    ld.shared.f32 %f5, [%rd28];
    ld.shared.f32 %f6, [%rd26];
    add.f32 %f6, %f6, %f5;
    st.shared.f32 [%rd26], %f6;
reduce_skip:
    bar.sync 0;
    shr.u32 %r22, %r22, 1;
    setp.gt.u32 %p5, %r22, 0;
    @%p5 bra reduce_loop;

    setp.ne.u32 %p6, %r0, 0;
    @%p6 bra next_row;
    ld.shared.f32 %f7, [%rd40];
    mul.wide.u32 %rd29, %r11, 4;
    add.u64 %rd30, %rd6, %rd29;
    st.global.f32 [%rd30], %f7;

next_row:
    bar.sync 0;
    bra claim_row;

done:
    ret;
}
`
