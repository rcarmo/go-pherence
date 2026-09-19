package ptx

// SgemmCompensatedPTX is an explicit accuracy-first projection kernel.
// It does not replace the default SGEMM dispatch or alter its rounding contract.
const SgemmCompensatedPTX = `
.version 7.0
.target sm_80
.address_size 64

// sgemm_nn: C[M,N] = alpha * A[M,K] * B[K,N]
// Row-major, tiled 16x16 with shared memory.
// Kept as the correctness oracle for newer kernels.
.visible .entry sgemm_nn_compensated(
    .param .u64 param_A,
    .param .u64 param_B,
    .param .u64 param_C,
    .param .u32 param_M,
    .param .u32 param_N,
    .param .u32 param_K,
    .param .f32 param_alpha
) {
    .reg .u32 %r<32>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<16>;
    .reg .pred %p<4>;

    .shared .align 4 .f32 sA[256];
    .shared .align 4 .f32 sB[256];

    ld.param.u64 %rd0, [param_A];
    ld.param.u64 %rd1, [param_B];
    ld.param.u64 %rd2, [param_C];
    ld.param.u32 %r0, [param_M];
    ld.param.u32 %r1, [param_N];
    ld.param.u32 %r2, [param_K];
    ld.param.f32 %f0, [param_alpha];

    mov.u32 %r3, %ctaid.y;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %tid.y;
    mov.u32 %r6, %tid.x;

    mad.lo.u32 %r7, %r3, 16, %r5;
    mad.lo.u32 %r8, %r4, 16, %r6;

    mov.f32 %f1, 0.0;
    mov.f32 %f6, 0.0;

    add.u32 %r10, %r2, 15;
    shr.u32 %r10, %r10, 4;
    mov.u32 %r11, 0;

TILE_LOOP:
    setp.ge.u32 %p0, %r11, %r10;
    @%p0 bra TILE_DONE;

    shl.b32 %r12, %r11, 4;

    add.u32 %r13, %r12, %r6;
    setp.lt.u32 %p1, %r7, %r0;
    setp.lt.u32 %p2, %r13, %r2;
    and.pred %p1, %p1, %p2;

    mad.lo.u32 %r14, %r5, 16, %r6;
    mul.wide.u32 %rd3, %r14, 4;
    mov.u64 %rd4, sA;
    add.u64 %rd3, %rd4, %rd3;

    mad.lo.u32 %r15, %r7, %r2, %r13;
    mul.wide.u32 %rd5, %r15, 4;
    add.u64 %rd5, %rd0, %rd5;

    mov.f32 %f2, 0.0;
    @%p1 ld.global.f32 %f2, [%rd5];
    st.shared.f32 [%rd3], %f2;

    add.u32 %r16, %r12, %r5;
    setp.lt.u32 %p1, %r16, %r2;
    setp.lt.u32 %p2, %r8, %r1;
    and.pred %p1, %p1, %p2;

    mov.u64 %rd6, sB;
    add.u64 %rd7, %rd6, %rd3;
    sub.u64 %rd7, %rd7, %rd4;

    mad.lo.u32 %r17, %r16, %r1, %r8;
    mul.wide.u32 %rd8, %r17, 4;
    add.u64 %rd8, %rd1, %rd8;

    mov.f32 %f3, 0.0;
    @%p1 ld.global.f32 %f3, [%rd8];
    st.shared.f32 [%rd7], %f3;

    bar.sync 0;

    mov.u32 %r18, 0;
DOT_LOOP:
    setp.ge.u32 %p1, %r18, 16;
    @%p1 bra DOT_DONE;

    mad.lo.u32 %r19, %r5, 16, %r18;
    mul.wide.u32 %rd9, %r19, 4;
    add.u64 %rd9, %rd4, %rd9;
    ld.shared.f32 %f4, [%rd9];

    mad.lo.u32 %r20, %r18, 16, %r6;
    mul.wide.u32 %rd10, %r20, 4;
    add.u64 %rd10, %rd6, %rd10;
    ld.shared.f32 %f5, [%rd10];

    // Kahan accumulation of rounded products; keep explicit rn ops separate.
    mul.rn.f32 %f7, %f4, %f5;
    sub.rn.f32 %f8, %f7, %f6;
    add.rn.f32 %f9, %f1, %f8;
    sub.rn.f32 %f10, %f9, %f1;
    sub.rn.f32 %f6, %f10, %f8;
    mov.f32 %f1, %f9;

    add.u32 %r18, %r18, 1;
    bra DOT_LOOP;

DOT_DONE:
    bar.sync 0;

    add.u32 %r11, %r11, 1;
    bra TILE_LOOP;

TILE_DONE:
    setp.lt.u32 %p1, %r7, %r0;
    setp.lt.u32 %p2, %r8, %r1;
    and.pred %p1, %p1, %p2;

    mul.f32 %f1, %f1, %f0;

    mad.lo.u32 %r21, %r7, %r1, %r8;
    mul.wide.u32 %rd11, %r21, 4;
    add.u64 %rd11, %rd2, %rd11;

    @%p1 st.global.f32 [%rd11], %f1;
    ret;
}
`
