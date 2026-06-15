package ptx

const SplitGateUpPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry split_gate_up(
    .param .u64 param_src,
    .param .u64 param_gate,
    .param .u64 param_up,
    .param .u32 param_batch,
    .param .u32 param_intermediate
) {
    .reg .u32 %r<12>;
    .reg .u64 %rd<12>;
    .reg .f32 %f<2>;
    .reg .pred %p;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [param_batch];
    ld.param.u32 %r5, [param_intermediate];
    mul.lo.u32 %r6, %r4, %r5;
    setp.ge.u32 %p, %r3, %r6;
    @%p bra done;

    div.u32 %r7, %r3, %r5; // batch
    rem.u32 %r8, %r3, %r5; // col
    mul.lo.u32 %r9, %r7, %r5;
    add.u32 %r10, %r9, %r8;

    ld.param.u64 %rd0, [param_src];
    ld.param.u64 %rd1, [param_gate];
    ld.param.u64 %rd2, [param_up];

    // src row is [gate(intermediate), up(intermediate)]
    shl.b32 %r11, %r5, 1;
    mad.lo.u32 %r0, %r7, %r11, %r8;
    mul.wide.u32 %rd3, %r0, 4;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u32 %r0, %r0, %r5;
    mul.wide.u32 %rd5, %r0, 4;
    add.u64 %rd6, %rd0, %rd5;
    ld.global.f32 %f1, [%rd6];

    mul.wide.u32 %rd7, %r10, 4;
    add.u64 %rd8, %rd1, %rd7;
    add.u64 %rd9, %rd2, %rd7;
    st.global.f32 [%rd8], %f0;
    st.global.f32 [%rd9], %f1;

done:
    ret;
}
`
