package ptx

const GatherRowsPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gather_rows(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u64 param_pos,
    .param .u32 param_rows,
    .param .u32 param_hidden
) {
    .reg .u32 %r<14>;
    .reg .u64 %rd<12>;
    .reg .f32 %f<2>;
    .reg .pred %p;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [param_rows];
    ld.param.u32 %r5, [param_hidden];
    mul.lo.u32 %r6, %r4, %r5;
    setp.ge.u32 %p, %r3, %r6;
    @%p bra done;

    div.u32 %r7, %r3, %r5; // work row
    rem.u32 %r8, %r3, %r5; // hidden col

    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u64 %rd2, [param_pos];

    mul.wide.u32 %rd3, %r7, 4;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.u32 %r9, [%rd4];

    mad.lo.u32 %r10, %r9, %r5, %r8;
    mul.wide.u32 %rd5, %r10, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.f32 %f0, [%rd6];

    mul.wide.u32 %rd7, %r3, 4;
    add.u64 %rd8, %rd0, %rd7;
    st.global.f32 [%rd8], %f0;

done:
    ret;
}
`
