package ptx

// WidenBF16TransposePTX expands compact row-major W[N,K] to F32 Wt[K,N].
// BF16 bit widening is exact and needs no arithmetic approximation.
const WidenBF16TransposePTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry widen_bf16_transpose(.param .u64 SRC, .param .u64 DST, .param .u32 N, .param .u32 K) {
    .reg .u32 %r<12>; .reg .u64 %rd<6>; .reg .pred %p;
    ld.param.u64 %rd0, [SRC]; ld.param.u64 %rd1, [DST];
    ld.param.u32 %r0, [N]; ld.param.u32 %r1, [K];
    mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    mul.lo.u32 %r6, %r0, %r1;
    setp.ge.u32 %p, %r5, %r6; @%p bra done;
    mul.wide.u32 %rd2, %r5, 2; add.u64 %rd3, %rd0, %rd2;
    ld.global.u16 %r7, [%rd3]; shl.b32 %r7, %r7, 16;
    div.u32 %r8, %r5, %r1; rem.u32 %r9, %r5, %r1;
    mad.lo.u32 %r10, %r9, %r0, %r8;
    mul.wide.u32 %rd4, %r10, 4; add.u64 %rd5, %rd1, %rd4;
    st.global.u32 [%rd5], %r7;
done: ret;
}
`

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
