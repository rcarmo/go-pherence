package ptx

// PTX SGEMM kernels for NVIDIA GPUs (Ampere SM86+).
// Embedded as Go strings and JIT-compiled by the CUDA driver.

const SgemmPTX = `
.version 7.0
.target sm_80
.address_size 64

// sgemm_nn: C[M,N] = alpha * A[M,K] * B[K,N]
// Row-major, tiled 16x16 with shared memory.
// Kept as the correctness oracle for newer kernels.
.visible .entry sgemm_nn(
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

    fma.rn.f32 %f1, %f4, %f5, %f1;

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

const SgemmReg2PTX = `
.version 7.0
.target sm_80
.address_size 64

// sgemm_nn_reg2: 16x16 logical tile, 8x16 threads.
// Each thread computes 2 adjacent output columns.
.visible .entry sgemm_nn_reg2(
    .param .u64 param_A,
    .param .u64 param_B,
    .param .u64 param_C,
    .param .u32 param_M,
    .param .u32 param_N,
    .param .u32 param_K,
    .param .f32 param_alpha
) {
    .reg .u32 %r<64>;
    .reg .u64 %rd<32>;
    .reg .f32 %f<16>;
    .reg .pred %p<8>;

    .shared .align 4 .f32 sA[272]; // 16 x 17
    .shared .align 4 .f32 sB[272]; // 16 x 17

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
    shl.b32 %r8, %r6, 1;
    mad.lo.u32 %r9, %r4, 16, %r8;
    add.u32 %r10, %r9, 1;

    mov.u64 %rd3, sA;
    mov.u64 %rd4, sB;

    mov.f32 %f1, 0.0;
    mov.f32 %f2, 0.0;

    add.u32 %r11, %r2, 15;
    shr.u32 %r11, %r11, 4;
    mov.u32 %r12, 0;

REG2_TILE_LOOP:
    setp.ge.u32 %p0, %r12, %r11;
    @%p0 bra REG2_TILE_DONE;

    shl.b32 %r13, %r12, 4;

    // A tile: each thread loads 2 K values for its output row.
    add.u32 %r14, %r13, %r6;
    add.u32 %r15, %r14, 8;
    mad.lo.u32 %r16, %r5, 17, %r6;
    add.u32 %r17, %r16, 8;

    setp.lt.u32 %p1, %r7, %r0;
    setp.lt.u32 %p2, %r14, %r2;
    and.pred %p3, %p1, %p2;
    mad.lo.u32 %r18, %r7, %r2, %r14;
    mul.wide.u32 %rd5, %r16, 4;
    add.u64 %rd5, %rd3, %rd5;
    mul.wide.u32 %rd6, %r18, 4;
    add.u64 %rd6, %rd0, %rd6;
    mov.f32 %f3, 0.0;
    @%p3 ld.global.f32 %f3, [%rd6];
    st.shared.f32 [%rd5], %f3;

    setp.lt.u32 %p2, %r15, %r2;
    and.pred %p3, %p1, %p2;
    mad.lo.u32 %r18, %r7, %r2, %r15;
    mul.wide.u32 %rd7, %r17, 4;
    add.u64 %rd7, %rd3, %rd7;
    mul.wide.u32 %rd8, %r18, 4;
    add.u64 %rd8, %rd0, %rd8;
    mov.f32 %f4, 0.0;
    @%p3 ld.global.f32 %f4, [%rd8];
    st.shared.f32 [%rd7], %f4;

    // B tile: each thread loads 2 adjacent columns for one K row.
    add.u32 %r19, %r13, %r5;
    mad.lo.u32 %r20, %r5, 17, %r8;
    add.u32 %r21, %r20, 1;

    setp.lt.u32 %p4, %r19, %r2;
    setp.lt.u32 %p5, %r9, %r1;
    and.pred %p6, %p4, %p5;
    mad.lo.u32 %r22, %r19, %r1, %r9;
    mul.wide.u32 %rd9, %r20, 4;
    add.u64 %rd9, %rd4, %rd9;
    mul.wide.u32 %rd10, %r22, 4;
    add.u64 %rd10, %rd1, %rd10;
    mov.f32 %f5, 0.0;
    @%p6 ld.global.f32 %f5, [%rd10];
    st.shared.f32 [%rd9], %f5;

    setp.lt.u32 %p5, %r10, %r1;
    and.pred %p6, %p4, %p5;
    mad.lo.u32 %r22, %r19, %r1, %r10;
    mul.wide.u32 %rd11, %r21, 4;
    add.u64 %rd11, %rd4, %rd11;
    mul.wide.u32 %rd12, %r22, 4;
    add.u64 %rd12, %rd1, %rd12;
    mov.f32 %f6, 0.0;
    @%p6 ld.global.f32 %f6, [%rd12];
    st.shared.f32 [%rd11], %f6;

    bar.sync 0;

    mov.u32 %r30, 0;
REG2_DOT_LOOP:
    setp.ge.u32 %p0, %r30, 16;
    @%p0 bra REG2_DOT_DONE;

    mad.lo.u32 %r31, %r5, 17, %r30;
    mul.wide.u32 %rd13, %r31, 4;
    add.u64 %rd13, %rd3, %rd13;
    ld.shared.f32 %f7, [%rd13];

    mad.lo.u32 %r32, %r30, 17, %r8;
    mul.wide.u32 %rd14, %r32, 4;
    add.u64 %rd14, %rd4, %rd14;
    ld.shared.f32 %f8, [%rd14];
    ld.shared.f32 %f9, [%rd14+4];

    fma.rn.f32 %f1, %f7, %f8, %f1;
    fma.rn.f32 %f2, %f7, %f9, %f2;

    add.u32 %r30, %r30, 1;
    bra REG2_DOT_LOOP;

REG2_DOT_DONE:
    bar.sync 0;

    add.u32 %r12, %r12, 1;
    bra REG2_TILE_LOOP;

REG2_TILE_DONE:
    setp.lt.u32 %p1, %r7, %r0;

    mul.f32 %f1, %f1, %f0;
    mul.f32 %f2, %f2, %f0;

    setp.lt.u32 %p2, %r9, %r1;
    and.pred %p3, %p1, %p2;
    mad.lo.u32 %r40, %r7, %r1, %r9;
    mul.wide.u32 %rd20, %r40, 4;
    add.u64 %rd20, %rd2, %rd20;
    @%p3 st.global.f32 [%rd20], %f1;

    setp.lt.u32 %p2, %r10, %r1;
    and.pred %p3, %p1, %p2;
    mad.lo.u32 %r41, %r7, %r1, %r10;
    mul.wide.u32 %rd21, %r41, 4;
    add.u64 %rd21, %rd2, %rd21;
    @%p3 st.global.f32 [%rd21], %f2;

    ret;
}
`

const SgemmSkinnyPTX = `
.version 7.0
.target sm_80
.address_size 64

// sgemm_nn_skinny: warp-per-output-row kernel for small M.
// 32x4 threads -> logical 4x32 output tile, K tile = 16.
.visible .entry sgemm_nn_skinny(
    .param .u64 param_A,
    .param .u64 param_B,
    .param .u64 param_C,
    .param .u32 param_M,
    .param .u32 param_N,
    .param .u32 param_K,
    .param .f32 param_alpha
) {
    .reg .u32 %r<96>;
    .reg .u64 %rd<48>;
    .reg .f32 %f<16>;
    .reg .pred %p<8>;

    .shared .align 4 .f32 sA[68];   // 4 x 17
    .shared .align 4 .f32 sB[528];  // 16 x 33

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

    mad.lo.u32 %r7, %r3, 4, %r5;
    mad.lo.u32 %r8, %r4, 32, %r6;

    mov.u64 %rd3, sA;
    mov.u64 %rd4, sB;

    mov.f32 %f1, 0.0;

    add.u32 %r9, %r2, 15;
    shr.u32 %r9, %r9, 4;
    mov.u32 %r10, 0;

SKINNY_TILE_LOOP:
    setp.ge.u32 %p0, %r10, %r9;
    @%p0 bra SKINNY_TILE_DONE;

    shl.b32 %r11, %r10, 4;

    // Load A tile: first 16 lanes of each warp load the row fragment.
    setp.lt.u32 %p1, %r6, 16;
    setp.lt.u32 %p2, %r7, %r0;
    and.pred %p3, %p1, %p2;
    add.u32 %r12, %r11, %r6;
    setp.lt.u32 %p4, %r12, %r2;
    and.pred %p3, %p3, %p4;
    mad.lo.u32 %r13, %r5, 17, %r6;
    mul.wide.u32 %rd5, %r13, 4;
    add.u64 %rd5, %rd3, %rd5;
    mad.lo.u32 %r14, %r7, %r2, %r12;
    mul.wide.u32 %rd6, %r14, 4;
    add.u64 %rd6, %rd0, %rd6;
    mov.f32 %f2, 0.0;
    @%p3 ld.global.f32 %f2, [%rd6];
    st.shared.f32 [%rd5], %f2;

    // Load B tile: 128 threads cooperatively cover 16 x 32 values.
    mad.lo.u32 %r20, %r5, 32, %r6;

    mov.u32 %r21, %r20;
    shr.u32 %r22, %r21, 5;
    and.b32 %r23, %r21, 31;
    add.u32 %r24, %r11, %r22;
    mad.lo.u32 %r25, %r22, 33, %r23;
    setp.lt.u32 %p4, %r24, %r2;
    setp.lt.u32 %p5, %r23, 32;
    and.pred %p6, %p4, %p5;
    setp.lt.u32 %p5, %r8, %r1;
    and.pred %p6, %p6, %p5;
    mul.wide.u32 %rd7, %r25, 4;
    add.u64 %rd7, %rd4, %rd7;
    mad.lo.u32 %r26, %r24, %r1, %r8;
    mul.wide.u32 %rd8, %r26, 4;
    add.u64 %rd8, %rd1, %rd8;
    mov.f32 %f3, 0.0;
    @%p6 ld.global.f32 %f3, [%rd8];
    st.shared.f32 [%rd7], %f3;

    add.u32 %r21, %r20, 128;
    shr.u32 %r22, %r21, 5;
    and.b32 %r23, %r21, 31;
    add.u32 %r24, %r11, %r22;
    mad.lo.u32 %r25, %r22, 33, %r23;
    setp.lt.u32 %p4, %r21, 512;
    setp.lt.u32 %p5, %r24, %r2;
    and.pred %p6, %p4, %p5;
    setp.lt.u32 %p5, %r23, 32;
    and.pred %p6, %p6, %p5;
    setp.lt.u32 %p5, %r8, %r1;
    and.pred %p6, %p6, %p5;
    mul.wide.u32 %rd9, %r25, 4;
    add.u64 %rd9, %rd4, %rd9;
    mad.lo.u32 %r26, %r24, %r1, %r8;
    mul.wide.u32 %rd10, %r26, 4;
    add.u64 %rd10, %rd1, %rd10;
    mov.f32 %f4, 0.0;
    @%p6 ld.global.f32 %f4, [%rd10];
    @%p4 st.shared.f32 [%rd9], %f4;

    add.u32 %r21, %r20, 256;
    shr.u32 %r22, %r21, 5;
    and.b32 %r23, %r21, 31;
    add.u32 %r24, %r11, %r22;
    mad.lo.u32 %r25, %r22, 33, %r23;
    setp.lt.u32 %p4, %r21, 512;
    setp.lt.u32 %p5, %r24, %r2;
    and.pred %p6, %p4, %p5;
    setp.lt.u32 %p5, %r23, 32;
    and.pred %p6, %p6, %p5;
    setp.lt.u32 %p5, %r8, %r1;
    and.pred %p6, %p6, %p5;
    mul.wide.u32 %rd11, %r25, 4;
    add.u64 %rd11, %rd4, %rd11;
    mad.lo.u32 %r26, %r24, %r1, %r8;
    mul.wide.u32 %rd12, %r26, 4;
    add.u64 %rd12, %rd1, %rd12;
    mov.f32 %f5, 0.0;
    @%p6 ld.global.f32 %f5, [%rd12];
    @%p4 st.shared.f32 [%rd11], %f5;

    add.u32 %r21, %r20, 384;
    shr.u32 %r22, %r21, 5;
    and.b32 %r23, %r21, 31;
    add.u32 %r24, %r11, %r22;
    mad.lo.u32 %r25, %r22, 33, %r23;
    setp.lt.u32 %p4, %r21, 512;
    setp.lt.u32 %p5, %r24, %r2;
    and.pred %p6, %p4, %p5;
    setp.lt.u32 %p5, %r23, 32;
    and.pred %p6, %p6, %p5;
    setp.lt.u32 %p5, %r8, %r1;
    and.pred %p6, %p6, %p5;
    mul.wide.u32 %rd13, %r25, 4;
    add.u64 %rd13, %rd4, %rd13;
    mad.lo.u32 %r26, %r24, %r1, %r8;
    mul.wide.u32 %rd14, %r26, 4;
    add.u64 %rd14, %rd1, %rd14;
    mov.f32 %f6, 0.0;
    @%p6 ld.global.f32 %f6, [%rd14];
    @%p4 st.shared.f32 [%rd13], %f6;

    bar.sync 0;

    setp.lt.u32 %p1, %r7, %r0;
    setp.lt.u32 %p2, %r8, %r1;
    and.pred %p3, %p1, %p2;
    @!%p3 bra SKINNY_SKIP_DOT;

    mov.u32 %r30, 0;
SKINNY_DOT_LOOP:
    setp.ge.u32 %p0, %r30, 16;
    @%p0 bra SKINNY_DOT_DONE;

    mad.lo.u32 %r31, %r5, 17, %r30;
    mul.wide.u32 %rd20, %r31, 4;
    add.u64 %rd20, %rd3, %rd20;
    ld.shared.f32 %f7, [%rd20];

    mad.lo.u32 %r32, %r30, 33, %r6;
    mul.wide.u32 %rd21, %r32, 4;
    add.u64 %rd21, %rd4, %rd21;
    ld.shared.f32 %f8, [%rd21];

    fma.rn.f32 %f1, %f7, %f8, %f1;

    add.u32 %r30, %r30, 1;
    bra SKINNY_DOT_LOOP;

SKINNY_DOT_DONE:
SKINNY_SKIP_DOT:
    bar.sync 0;

    add.u32 %r10, %r10, 1;
    bra SKINNY_TILE_LOOP;

SKINNY_TILE_DONE:
    setp.lt.u32 %p1, %r7, %r0;
    setp.lt.u32 %p2, %r8, %r1;
    and.pred %p3, %p1, %p2;
    @!%p3 bra SKINNY_RET;

    mul.f32 %f1, %f1, %f0;
    mad.lo.u32 %r40, %r7, %r1, %r8;
    mul.wide.u32 %rd30, %r40, 4;
    add.u64 %rd30, %rd2, %rd30;
    st.global.f32 [%rd30], %f1;

SKINNY_RET:
    ret;
}
`
