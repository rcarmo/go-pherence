package conversion

const ToBF16F32PTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry to_bf16_f32(.param .u64 A, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<4>; .reg .f32 %f<2>; .reg .pred %p;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [A];
    mul.wide.u32 %rd1, %r3, 4; add.u64 %rd2, %rd0, %rd1;
    ld.global.f32 %f0, [%rd2];
    mov.b32 %r5, %f0;
    shr.u32 %r5, %r5, 16;
    shl.b32 %r5, %r5, 16;
    mov.b32 %f1, %r5;
    st.global.f32 [%rd2], %f1;
done: ret;
}
`
