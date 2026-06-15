package ptx

const MulWeightsPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry mul_weights(
    .param .u64 param_out,
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u32 param_n
) {
    .reg .u32 %r<8>;
    .reg .u64 %rd<10>;
    .reg .f32 %f<4>;
    .reg .pred %p;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [param_n];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra done;

    ld.param.u64 %rd0, [param_out];
    ld.param.u64 %rd1, [param_a];
    ld.param.u64 %rd2, [param_b];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd5, %rd2, %rd3;
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];
    mul.f32 %f2, %f0, %f1;
    add.u64 %rd6, %rd0, %rd3;
    st.global.f32 [%rd6], %f2;

done:
    ret;
}
`
