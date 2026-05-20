package vector

const VecAddPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry vec_add(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<4>; .reg .pred %p;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B]; ld.param.u64 %rd2, [C];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3; add.u64 %rd6, %rd2, %rd3;
    ld.global.f32 %f0, [%rd4]; ld.global.f32 %f1, [%rd5];
    add.f32 %f2, %f0, %f1;
    st.global.f32 [%rd6], %f2;
done: ret;
}
`

const VecMulPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry vec_mul(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<4>; .reg .pred %p;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B]; ld.param.u64 %rd2, [C];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3; add.u64 %rd6, %rd2, %rd3;
    ld.global.f32 %f0, [%rd4]; ld.global.f32 %f1, [%rd5];
    mul.f32 %f2, %f0, %f1;
    st.global.f32 [%rd6], %f2;
done: ret;
}
`

const VecScalePTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry vec_scale(.param .u64 A, .param .u64 B, .param .f32 S, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<6>; .reg .f32 %f<4>; .reg .pred %p;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B]; ld.param.f32 %f2, [S];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];
    mul.f32 %f1, %f0, %f2;
    st.global.f32 [%rd5], %f1;
done: ret;
}
`

const VecAddScaledPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry vec_add_scaled(.param .u64 A, .param .u64 B, .param .u64 C, .param .f32 S, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<5>; .reg .pred %p;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B]; ld.param.u64 %rd2, [C]; ld.param.f32 %f2, [S];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3; add.u64 %rd6, %rd2, %rd3;
    ld.global.f32 %f0, [%rd4]; ld.global.f32 %f1, [%rd5];
    fma.rn.f32 %f3, %f1, %f2, %f0;
    st.global.f32 [%rd6], %f3;
done: ret;
}
`

// Truncate F32 values in-place to BF16 precision by clearing the low 16 mantissa bits.
