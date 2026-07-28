package q4

// GemmQ4PTX: batched INT4 dequant + GEMM kernel.
//
// Launch geometry:
//   - blockDim.x = 128 threads = 4 warps
//   - blockIdx.x  = 4-column output tile
//   - blockIdx.y  = batch row
//
// Each warp computes one output column for one batch row. Lanes stride the K
// dimension by 32, decode packed GPTQ symmetric 4-bit weights + per-group
// scales, accumulate locally, then reduce within the warp via shuffles. Lane 0
// stores the final output value.
var GemmQ4PTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gemm_q4sym(
    .param .u64 input,     // [B, inDim] f32
    .param .u64 qweight,   // [inDim/8, outDim] i32
    .param .u64 gidx,      // [inDim] i32
    .param .u64 scales,    // [groups, outDim] f32
    .param .u64 output,    // [B, outDim] f32
    .param .u32 inDim,
    .param .u32 outDim,
    .param .u32 groups,
    .param .u32 B
) {
    .reg .pred %p<4>;
    .reg .u32 %r<32>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<8>;

    mov.u32 %r0, %ctaid.x;   // 4-column tile index
    mov.u32 %r1, %ctaid.y;   // batch row
    mov.u32 %r2, %tid.x;     // thread id in block

    shr.u32 %r3, %r2, 5;     // warp id [0..3]
    and.b32 %r4, %r2, 31;    // lane id [0..31]

    // col = blockIdx.x * 4 + warpId
    shl.b32 %r5, %r0, 2;
    add.u32 %r6, %r5, %r3;

    ld.param.u32 %r7, [inDim];
    ld.param.u32 %r8, [outDim];
    ld.param.u32 %r9, [B];

    // Bounds checks
    setp.ge.u32 %p0, %r1, %r9;
    @%p0 bra done;
    setp.ge.u32 %p0, %r6, %r8;
    @%p0 bra done;

    // Load base pointers
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [qweight];
    ld.param.u64 %rd2, [gidx];
    ld.param.u64 %rd3, [scales];
    ld.param.u64 %rd4, [output];

    // input_row = input + batch * inDim
    mul.lo.u32 %r10, %r1, %r7;
    mul.wide.u32 %rd5, %r10, 4;
    add.u64 %rd6, %rd0, %rd5;

    // Partial sum for this lane
    mov.f32 %f0, 0f00000000;
    mov.u32 %r10, %r4;       // i = lane

loop:
    setp.ge.u32 %p1, %r10, %r7;
    @%p1 bra reduce;

    // Load packed weight: qweight[(i/8)*outDim + col]
    shr.u32 %r11, %r10, 3;
    mad.lo.u32 %r12, %r11, %r8, %r6;
    mul.wide.u32 %rd7, %r12, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u32 %r13, [%rd8];

    // Extract 4-bit value for i: (packed >> ((i & 7) * 4)) & 0xF
    and.b32 %r14, %r10, 7;
    shl.b32 %r14, %r14, 2;
    shr.u32 %r15, %r13, %r14;
    and.b32 %r15, %r15, 15;
    add.s32 %r15, %r15, -8;
    cvt.rn.f32.s32 %f1, %r15;

    // Load group index and scale
    mul.wide.u32 %rd9, %r10, 4;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.u32 %r16, [%rd10];
    mad.lo.u32 %r17, %r16, %r8, %r6;
    mul.wide.u32 %rd11, %r17, 4;
    add.u64 %rd12, %rd3, %rd11;
    ld.global.f32 %f2, [%rd12];

    // Load input value
    mul.wide.u32 %rd13, %r10, 4;
    add.u64 %rd14, %rd6, %rd13;
    ld.global.f32 %f3, [%rd14];

    // Accumulate
    mul.f32 %f1, %f1, %f2;
    fma.rn.f32 %f0, %f1, %f3, %f0;

    add.u32 %r10, %r10, 32;
    bra loop;

reduce:
    // Warp reduction via shuffle-down
    mov.b32 %r18, %f0;

    shfl.sync.down.b32 %r19|%p2, %r18, 16, 0x1f, 0xffffffff;
    mov.b32 %f4, %r19;
    add.f32 %f0, %f0, %f4;
    mov.b32 %r18, %f0;

    shfl.sync.down.b32 %r19|%p2, %r18, 8, 0x1f, 0xffffffff;
    mov.b32 %f4, %r19;
    add.f32 %f0, %f0, %f4;
    mov.b32 %r18, %f0;

    shfl.sync.down.b32 %r19|%p2, %r18, 4, 0x1f, 0xffffffff;
    mov.b32 %f4, %r19;
    add.f32 %f0, %f0, %f4;
    mov.b32 %r18, %f0;

    shfl.sync.down.b32 %r19|%p2, %r18, 2, 0x1f, 0xffffffff;
    mov.b32 %f4, %r19;
    add.f32 %f0, %f0, %f4;
    mov.b32 %r18, %f0;

    shfl.sync.down.b32 %r19|%p2, %r18, 1, 0x1f, 0xffffffff;
    mov.b32 %f4, %r19;
    add.f32 %f0, %f0, %f4;

    // Lane 0 stores result
    setp.ne.u32 %p3, %r4, 0;
    @%p3 bra done;

    mad.lo.u32 %r20, %r1, %r8, %r6;
    mul.wide.u32 %rd15, %r20, 4;
    add.u64 %rd16, %rd4, %rd15;
    st.global.f32 [%rd16], %f0;

done:
    ret;
}
`
