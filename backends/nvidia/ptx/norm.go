package ptx

const RmsNormPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry rms_norm(.param .u64 A, .param .u64 W, .param .u64 B, .param .u32 N, .param .f32 eps) {
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<8>; .reg .pred %p;
    .shared .align 4 .f32 sdata[256];
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [N];
    ld.param.u64 %rd0, [A];
    mov.f32 %f0, 0f00000000;
    mov.u32 %r2, %r0;
L1: setp.ge.u32 %p, %r2, %r1; @%p bra L2;
    mul.wide.u32 %rd3, %r2, 4; add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4]; fma.rn.f32 %f0, %f1, %f1, %f0;
    add.u32 %r2, %r2, 256; bra L1;
L2: mul.wide.u32 %rd5, %r0, 4; mov.u64 %rd6, sdata; add.u64 %rd5, %rd6, %rd5;
    st.shared.f32 [%rd5], %f0; bar.sync 0;
    mov.u32 %r3, 128;
L3: setp.lt.u32 %p, %r3, 1; @%p bra L4;
    setp.ge.u32 %p, %r0, %r3; @%p bra L3b;
    mul.wide.u32 %rd5, %r0, 4; add.u64 %rd5, %rd6, %rd5; ld.shared.f32 %f1, [%rd5];
    add.u32 %r4, %r0, %r3; mul.wide.u32 %rd7, %r4, 4; add.u64 %rd7, %rd6, %rd7;
    ld.shared.f32 %f2, [%rd7]; add.f32 %f1, %f1, %f2; st.shared.f32 [%rd5], %f1;
L3b: bar.sync 0; shr.u32 %r3, %r3, 1; bra L3;
L4: setp.ne.u32 %p, %r0, 0; @%p bra L5;
    ld.shared.f32 %f0, [sdata]; cvt.rn.f32.u32 %f3, %r1;
    div.rn.f32 %f0, %f0, %f3; ld.param.f32 %f4, [eps];
    add.f32 %f0, %f0, %f4; rsqrt.approx.f32 %f1, %f0; mul.f32 %f2, %f0, %f1; mul.f32 %f2, %f2, %f1; mul.f32 %f2, %f2, 0fBF000000; add.f32 %f2, %f2, 0f3FC00000; mul.f32 %f0, %f1, %f2; st.shared.f32 [sdata], %f0;
L5: bar.sync 0; ld.shared.f32 %f5, [sdata];
    ld.param.u64 %rd1, [W]; ld.param.u64 %rd2, [B];
    mov.u32 %r2, %r0;
L6: setp.ge.u32 %p, %r2, %r1; @%p bra L7;
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3; add.u64 %rd7, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4]; ld.global.f32 %f2, [%rd5];
    mul.f32 %f3, %f1, %f5; mul.f32 %f3, %f3, %f2; st.global.f32 [%rd7], %f3;
    add.u32 %r2, %r2, 256; bra L6;
L7: ret;
}
`

const RmsNormNoScalePTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry rms_norm_no_scale(.param .u64 A, .param .u64 B, .param .u32 N, .param .f32 eps) {
    .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<8>; .reg .pred %p;
    .shared .align 4 .f32 sdata[256];
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [N];
    ld.param.u64 %rd0, [A];
    mov.f32 %f0, 0f00000000;
    mov.u32 %r2, %r0;
L1ns: setp.ge.u32 %p, %r2, %r1; @%p bra L2ns;
    mul.wide.u32 %rd3, %r2, 4; add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4]; fma.rn.f32 %f0, %f1, %f1, %f0;
    add.u32 %r2, %r2, 256; bra L1ns;
L2ns: mul.wide.u32 %rd5, %r0, 4; mov.u64 %rd6, sdata; add.u64 %rd5, %rd6, %rd5;
    st.shared.f32 [%rd5], %f0; bar.sync 0;
    mov.u32 %r3, 128;
L3ns: setp.lt.u32 %p, %r3, 1; @%p bra L4ns;
    setp.ge.u32 %p, %r0, %r3; @%p bra L3bns;
    mul.wide.u32 %rd5, %r0, 4; add.u64 %rd5, %rd6, %rd5; ld.shared.f32 %f1, [%rd5];
    add.u32 %r4, %r0, %r3; mul.wide.u32 %rd7, %r4, 4; add.u64 %rd7, %rd6, %rd7;
    ld.shared.f32 %f2, [%rd7]; add.f32 %f1, %f1, %f2; st.shared.f32 [%rd5], %f1;
L3bns: bar.sync 0; shr.u32 %r3, %r3, 1; bra L3ns;
L4ns: setp.ne.u32 %p, %r0, 0; @%p bra L5ns;
    ld.shared.f32 %f0, [sdata]; cvt.rn.f32.u32 %f3, %r1;
    div.rn.f32 %f0, %f0, %f3; ld.param.f32 %f4, [eps];
    add.f32 %f0, %f0, %f4; rsqrt.approx.f32 %f1, %f0; mul.f32 %f2, %f0, %f1; mul.f32 %f2, %f2, %f1; mul.f32 %f2, %f2, 0fBF000000; add.f32 %f2, %f2, 0f3FC00000; mul.f32 %f0, %f1, %f2; st.shared.f32 [sdata], %f0;
L5ns: bar.sync 0; ld.shared.f32 %f5, [sdata];
    ld.param.u64 %rd2, [B];
    mov.u32 %r2, %r0;
L6ns: setp.ge.u32 %p, %r2, %r1; @%p bra L7ns;
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd0, %rd3; add.u64 %rd7, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    mul.f32 %f3, %f1, %f5; st.global.f32 [%rd7], %f3;
    add.u32 %r2, %r2, 256; bra L6ns;
L7ns: ret;
}
`

// GELUTanhMulPTX: fused gelu_tanh(gate) * up
// gate[i] = gelu_tanh(gate[i]) * up[i]
// gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Approximation: tanh(z) ≈ z * (27 + z^2) / (27 + 9*z^2) for small z; use ex2 for larger
