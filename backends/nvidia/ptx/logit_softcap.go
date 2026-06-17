package ptx

// LogitSoftcapPTX applies x = tanh(x / cap) * cap in-place over n F32 logits.
var LogitSoftcapPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry logit_softcap_f32(
    .param .u64 param_x,
    .param .u32 param_n,
    .param .f32 param_cap
) {
    .reg .u32 %r<6>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<12>;
    .reg .pred %p<3>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [param_n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra done;

    ld.param.u64 %rd0, [param_x];
    ld.param.f32 %f0, [param_cap];
    mul.wide.u32 %rd1, %r3, 4;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.f32 %f1, [%rd2];

    setp.gt.f32 %p1, %f0, 0f00000000;
    @!%p1 bra store;

    div.rn.f32 %f2, %f1, %f0;
    abs.ftz.f32 %f3, %f2;
    setp.eq.f32 %p2, %f3, 0f7f800000;
    @%p2 bra inf_case;

    // tanh(z) = 1 - 2/(1 + exp(2z)); exp via ex2(2z*log2(e)).
    mul.f32 %f4, %f2, 0f4038aa3b;
    ex2.approx.f32 %f5, %f4;
    add.f32 %f6, %f5, 0f3f800000;
    mov.f32 %f7, 0f40000000;
    div.approx.f32 %f8, %f7, %f6;
    mov.f32 %f9, 0f3f800000;
    sub.f32 %f10, %f9, %f8;
    mul.f32 %f1, %f10, %f0;
    bra store;

inf_case:
    neg.f32 %f11, %f0;
    setp.gt.f32 %p2, %f2, 0f00000000;
    selp.f32 %f1, %f0, %f11, %p2;

store:
    st.global.f32 [%rd2], %f1;
done:
    ret;
}
`
