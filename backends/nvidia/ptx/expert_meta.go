package ptx

const ExpertMetaReducePTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry expert_meta_reduce(
    .param .u64 param_offsets,
    .param .u64 param_weights,
    .param .u64 param_counts,
    .param .u64 param_sums,
    .param .u32 param_groups
) {
    .reg .u32 %r<16>;
    .reg .u64 %rd<14>;
    .reg .f32 %f<6>;
    .reg .pred %p;
    .shared .align 4 .f32 ssum[256];

    mov.u32 %r0, %ctaid.x; // group
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [param_groups];
    setp.ge.u32 %p, %r0, %r2;
    @%p bra done;

    ld.param.u64 %rd0, [param_offsets];
    ld.param.u64 %rd1, [param_weights];
    ld.param.u64 %rd2, [param_counts];
    ld.param.u64 %rd3, [param_sums];

    mul.wide.u32 %rd4, %r0, 4;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.u32 %r3, [%rd5];
    ld.global.u32 %r4, [%rd5+4];
    sub.u32 %r5, %r4, %r3; // count

    mov.f32 %f0, 0f00000000;
    mov.u32 %r6, %r1;
loop:
    setp.ge.u32 %p, %r6, %r5;
    @%p bra loop_done;
    add.u32 %r7, %r3, %r6;
    mul.wide.u32 %rd6, %r7, 4;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.f32 %f1, [%rd7];
    add.f32 %f0, %f0, %f1;
    add.u32 %r6, %r6, 256;
    bra loop;
loop_done:
    mov.u64 %rd8, ssum;
    mul.wide.u32 %rd9, %r1, 4;
    add.u64 %rd10, %rd8, %rd9;
    st.shared.f32 [%rd10], %f0;
    bar.sync 0;

    mov.u32 %r8, 128;
reduce:
    setp.ge.u32 %p, %r1, %r8;
    @%p bra red_skip;
    add.u32 %r9, %r1, %r8;
    mul.wide.u32 %rd11, %r9, 4;
    add.u64 %rd12, %rd8, %rd11;
    ld.shared.f32 %f2, [%rd12];
    ld.shared.f32 %f3, [%rd10];
    add.f32 %f3, %f3, %f2;
    st.shared.f32 [%rd10], %f3;
red_skip:
    bar.sync 0;
    shr.u32 %r8, %r8, 1;
    setp.gt.u32 %p, %r8, 0;
    @%p bra reduce;

    setp.ne.u32 %p, %r1, 0;
    @%p bra done;
    add.u64 %rd13, %rd2, %rd4;
    st.global.u32 [%rd13], %r5;
    add.u64 %rd13, %rd3, %rd4;
    ld.shared.f32 %f4, [ssum];
    st.global.f32 [%rd13], %f4;

done:
    ret;
}
`
