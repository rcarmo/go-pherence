package ptx

const VecSiLUPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry vec_silu(.param .u64 A, .param .u64 B, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<6>; .reg .f32 %f<8>; .reg .pred %p;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];
    neg.f32 %f1, %f0;
    mul.f32 %f1, %f1, 0f3FB8AA3B;
    ex2.approx.f32 %f2, %f1;
    add.f32 %f3, %f2, 0f3F800000;
    div.rn.f32 %f4, %f0, %f3;
    st.global.f32 [%rd5], %f4;
done: ret;
}
`

const FusedSiLUMulPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry fused_silu_mul(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<8>; .reg .pred %p;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B]; ld.param.u64 %rd2, [C];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3; add.u64 %rd6, %rd2, %rd3;
    ld.global.f32 %f0, [%rd4]; ld.global.f32 %f1, [%rd5];
    // silu(a) = a / (1 + exp(-a))
    neg.f32 %f2, %f0;
    mul.f32 %f2, %f2, 0f3FB8AA3B;
    ex2.approx.f32 %f3, %f2;
    add.f32 %f4, %f3, 0f3F800000;
    div.rn.f32 %f5, %f0, %f4;
    // out = silu(a) * b
    mul.f32 %f6, %f5, %f1;
    st.global.f32 [%rd6], %f6;
done: ret;
}
`

const GELUTanhMulPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry gelu_tanh_mul(
    .param .u64 param_gate,
    .param .u64 param_up,
    .param .u32 param_n
) {
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<16>;
    .reg .pred %p;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.u32 %r0, %r0, %r2, %r1;  // global idx

    ld.param.u32 %r3, [param_n];
    setp.ge.u32 %p, %r0, %r3;
    @%p bra done;

    ld.param.u64 %rd0, [param_gate];
    ld.param.u64 %rd1, [param_up];

    // Load gate[i] and up[i]
    mul.wide.u32 %rd2, %r0, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];      // x = gate[i]
    add.u64 %rd2, %rd1, %rd2;
    ld.global.f32 %f1, [%rd2];      // up[i]

    // gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Let z = sqrt(2/pi) * (x + 0.044715 * x^3)
    // sqrt(2/pi) = 0.7978845608
    mul.f32 %f2, %f0, %f0;          // x^2
    mul.f32 %f3, %f2, %f0;          // x^3
    mul.f32 %f3, %f3, 0f3D372713;   // 0.044715 * x^3
    add.f32 %f3, %f0, %f3;          // x + 0.044715*x^3
    mul.f32 %f3, %f3, 0f3F4C422A;   // z = sqrt(2/pi) * (...)

    // tanh(z) = 1 - 2/(1 + exp(2z))
    // exp(2z) via ex2: exp(2z) = 2^(2z * log2(e))
    mul.f32 %f4, %f3, 0f4038AA3B;   // 2z * log2(e) = 2 * 1.4426950 * z
    ex2.approx.f32 %f4, %f4;         // exp(2z)
    add.f32 %f5, %f4, 0f3F800000;   // 1 + exp(2z)
    mov.f32 %f6, 0f40000000;         // 2.0
    div.approx.f32 %f6, %f6, %f5;   // 2/(1+exp(2z))
    mov.f32 %f7, 0f3F800000;         // 1.0
    sub.f32 %f7, %f7, %f6;          // tanh(z)

    // gelu = 0.5 * x * (1 + tanh)
    add.f32 %f7, %f7, 0f3F800000;   // 1 + tanh(z)
    mul.f32 %f7, %f7, 0f3F000000;   // 0.5 * (1 + tanh)
    mul.f32 %f7, %f0, %f7;          // x * 0.5 * (1 + tanh) = gelu(x)

    // gate[i] = gelu(gate) * up
    mul.f32 %f7, %f7, %f1;
    st.global.f32 [%rd3], %f7;

done:
    ret;
}
`

// GELUErfPTX applies PyTorch's exact-form GELU using a maximum-error 1.5e-7
// Abramowitz-Stegun erf approximation. The input is updated in place.
const GELUErfPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry gelu_erf(.param .u64 A, .param .u32 N) {
    .reg .u32 %r<8>; .reg .u64 %rd<5>; .reg .f32 %f<20>; .reg .pred %p<2>;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p0, %r3, %r4; @%p0 bra done;
    ld.param.u64 %rd0, [A]; mul.wide.u32 %rd1, %r3, 4; add.u64 %rd2, %rd0, %rd1;
    ld.global.f32 %f0, [%rd2];
    mul.rn.f32 %f1, %f0, 0f3F3504F3; // z=x/sqrt(2)
    abs.f32 %f2, %f1;
    fma.rn.f32 %f3, %f2, 0f3EA7BA05, 0f3F800000;
    rcp.rn.f32 %f3, %f3;             // t=1/(1+p|z|)
    fma.rn.f32 %f4, %f3, 0f3F87DC22, 0fBFBA00E3;
    fma.rn.f32 %f4, %f4, %f3, 0f3FB5F0E3;
    fma.rn.f32 %f4, %f4, %f3, 0fBE91A98E;
    fma.rn.f32 %f4, %f4, %f3, 0f3E827906;
    mul.rn.f32 %f4, %f4, %f3;
    mul.rn.f32 %f5, %f2, %f2;
    neg.f32 %f5, %f5;
    mul.rn.f32 %f5, %f5, 0f3FB8AA3B;
    ex2.approx.f32 %f5, %f5;
    fma.rn.f32 %f6, %f4, %f5, 0fBF800000; // poly*exp-1 = -erf(|z|)
    neg.f32 %f6, %f6;
    setp.lt.f32 %p1, %f1, 0f00000000;
    @%p1 neg.f32 %f6, %f6;
    add.rn.f32 %f6, %f6, 0f3F800000;
    mul.rn.f32 %f6, %f6, %f0;
    mul.rn.f32 %f6, %f6, 0f3F000000;
    st.global.f32 [%rd2], %f6;
done: ret;
}
`
