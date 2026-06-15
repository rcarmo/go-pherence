package ptx

const GateUpGELUPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gate_up_gelu(
    .param .u64 param_src,
    .param .u64 param_out,
    .param .u32 param_batch,
    .param .u32 param_intermediate
) {
    .reg .u32 %r<12>;
    .reg .u64 %rd<10>;
    .reg .f32 %f<24>;
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
    shl.b32 %r9, %r5, 1;   // 2*intermediate
    mad.lo.u32 %r10, %r7, %r9, %r8;

    ld.param.u64 %rd0, [param_src];
    ld.param.u64 %rd1, [param_out];

    mul.wide.u32 %rd2, %r10, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3]; // gate
    add.u32 %r10, %r10, %r5;
    mul.wide.u32 %rd4, %r10, 4;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.f32 %f1, [%rd5]; // up

    // gelu_tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    mul.f32 %f2, %f0, %f0;
    mul.f32 %f3, %f2, %f0;
    mov.f32 %f4, 0f3d372713; // 0.044715
    fma.rn.f32 %f5, %f4, %f3, %f0;
    mov.f32 %f6, 0f3f4c422a; // sqrt(2/pi) ~= 0.79788456
    mul.f32 %f7, %f6, %f5;

    // tanh approximation via ex2: tanh(z) = 2/(1+exp(-2z))-1
    mov.f32 %f8, 0fc0000000; // -2.0
    mul.f32 %f9, %f8, %f7;
    mov.f32 %f10, 0f3fb8aa3b; // log2(e)
    mul.f32 %f11, %f9, %f10;
    ex2.approx.ftz.f32 %f12, %f11;
    mov.f32 %f13, 0f3f800000; // 1.0
    add.f32 %f14, %f13, %f12;
    rcp.approx.ftz.f32 %f15, %f14;
    mov.f32 %f16, 0f40000000; // 2.0
    mul.f32 %f17, %f16, %f15;
    sub.f32 %f18, %f17, %f13;
    add.f32 %f19, %f13, %f18;
    mov.f32 %f20, 0f3f000000; // 0.5
    mul.f32 %f21, %f20, %f0;
    mul.f32 %f22, %f21, %f19;
    mul.f32 %f23, %f22, %f1;

    mul.wide.u32 %rd6, %r3, 4;
    add.u64 %rd7, %rd1, %rd6;
    st.global.f32 [%rd7], %f23;

done:
    ret;
}
`
