package fp8

// FP8E4M3GemvF32PTX computes out[row] = dot(dequant_e4m3(W[row,:]), x) * scale[row|0] + bias[row].
// One CUDA block owns one output row; threads decode E4M3 bytes directly and
// reduce into shared memory. Scale is F32 length 1 (per tensor) or OutDim
// (per row). Bias may be null.
const FP8E4M3GemvF32PTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry fp8_e4m3_gemv_f32(
    .param .u64 W,
    .param .u64 SCALE,
    .param .u64 BIAS,
    .param .u64 X,
    .param .u64 O,
    .param .u32 OUT_DIM,
    .param .u32 IN_DIM,
    .param .u32 SCALE_LEN,
    .param .u32 HAS_BIAS
) {
    .reg .pred %p<16>;
    .reg .u32 %r<64>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<32>;
    .shared .align 4 .f32 fp8_sdata[128];

    mov.u32 %r0, %ctaid.x;               // row
    mov.u32 %r1, %tid.x;                 // tid
    mov.u32 %r2, %ntid.x;                // blockDim
    ld.param.u32 %r3, [OUT_DIM];
    setp.ge.u32 %p0, %r0, %r3;
    @%p0 bra done;

    ld.param.u64 %rd0, [W];
    ld.param.u64 %rd1, [SCALE];
    ld.param.u64 %rd2, [BIAS];
    ld.param.u64 %rd3, [X];
    ld.param.u64 %rd4, [O];
    ld.param.u32 %r4, [IN_DIM];
    ld.param.u32 %r5, [SCALE_LEN];
    ld.param.u32 %r21, [HAS_BIAS];

    mov.f32 %f10, 0f00000000;            // sum
    mov.u32 %r6, %r1;                    // col = tid

loop_col:
    setp.ge.u32 %p1, %r6, %r4;
    @%p1 bra reduce_store;

    mad.lo.u32 %r7, %r0, %r4, %r6;       // row*inDim + col
    cvt.u64.u32 %rd5, %r7;
    add.u64 %rd6, %rd0, %rd5;
    ld.global.u8 %r8, [%rd6];            // E4M3 byte

    and.b32 %r9, %r8, 127;               // exp+mant without sign
    setp.eq.u32 %p2, %r9, 127;           // reserved NaN code
    @%p2 bra val_nan;

    and.b32 %r10, %r8, 128;              // sign
    shr.u32 %r11, %r8, 3;
    and.b32 %r11, %r11, 15;              // exp
    and.b32 %r12, %r8, 7;                // mant
    setp.eq.u32 %p3, %r11, 0;
    @%p3 bra val_subnormal;

    add.u32 %r13, %r11, 120;             // exp - 7 + 127
    shl.b32 %r13, %r13, 23;
    shl.b32 %r14, %r12, 20;
    or.b32 %r15, %r13, %r14;
    setp.ne.u32 %p4, %r10, 0;
    @%p4 or.b32 %r15, %r15, 2147483648;
    mov.b32 %f1, %r15;
    bra val_done;

val_subnormal:
    cvt.rn.f32.u32 %f1, %r12;
    mul.f32 %f1, %f1, 0f3B000000;        // mant * 2^-9
    setp.ne.u32 %p4, %r10, 0;
    @%p4 neg.f32 %f1, %f1;
    bra val_done;

val_nan:
    mov.f32 %f1, 0f7FC00000;

val_done:
    mul.wide.u32 %rd7, %r6, 4;
    add.u64 %rd8, %rd3, %rd7;
    ld.global.f32 %f2, [%rd8];
    fma.rn.f32 %f10, %f1, %f2, %f10;
    add.u32 %r6, %r6, %r2;
    bra loop_col;

reduce_store:
    mul.wide.u32 %rd9, %r1, 4;
    mov.u64 %rd10, fp8_sdata;
    add.u64 %rd11, %rd10, %rd9;
    st.shared.f32 [%rd11], %f10;
    bar.sync 0;

    mov.u32 %r30, 64;
red_loop:
    setp.ge.u32 %p5, %r1, %r30;
    @%p5 bra red_skip;
    add.u32 %r31, %r1, %r30;
    setp.ge.u32 %p6, %r31, %r2;
    @%p6 bra red_skip;
    mul.wide.u32 %rd12, %r31, 4;
    add.u64 %rd13, %rd10, %rd12;
    ld.shared.f32 %f11, [%rd11];
    ld.shared.f32 %f12, [%rd13];
    add.f32 %f11, %f11, %f12;
    st.shared.f32 [%rd11], %f11;
red_skip:
    bar.sync 0;
    shr.u32 %r30, %r30, 1;
    setp.gt.u32 %p7, %r30, 0;
    @%p7 bra red_loop;

    setp.ne.u32 %p8, %r1, 0;
    @%p8 bra done;
    ld.shared.f32 %f10, [fp8_sdata];

    // Apply scale: scaleLen==1 uses scale[0], otherwise scale[row].
    setp.eq.u32 %p9, %r5, 1;
    @%p9 bra scale_tensor;
    mov.u32 %r20, %r0;
    bra scale_index_ready;
scale_tensor:
    mov.u32 %r20, 0;
scale_index_ready:
    mul.wide.u32 %rd14, %r20, 4;
    add.u64 %rd15, %rd1, %rd14;
    ld.global.f32 %f4, [%rd15];
    mul.f32 %f10, %f10, %f4;

    // Optional bias, controlled by explicit flag.
    setp.eq.u32 %p10, %r21, 0;
    @%p10 bra store_out;
    mul.wide.u32 %rd16, %r0, 4;
    add.u64 %rd17, %rd2, %rd16;
    ld.global.f32 %f5, [%rd17];
    add.f32 %f10, %f10, %f5;

store_out:
    mul.wide.u32 %rd18, %r0, 4;
    add.u64 %rd19, %rd4, %rd18;
    st.global.f32 [%rd19], %f10;

done:
    ret;
}
`

const FP8E4M3GemmF32PTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry fp8_e4m3_gemm_f32(
    .param .u64 W,
    .param .u64 SCALE,
    .param .u64 BIAS,
    .param .u64 X,
    .param .u64 O,
    .param .u32 OUT_DIM,
    .param .u32 IN_DIM,
    .param .u32 SCALE_LEN,
    .param .u32 HAS_BIAS,
    .param .u32 BATCH
) {
    .reg .pred %p<16>;
    .reg .u32 %r<64>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<32>;

    mov.u32 %r0, %ctaid.x;               // row block, 4 rows per block
    mov.u32 %r22, %ctaid.y;              // batch index
    mov.u32 %r1, %tid.x;                 // tid
    and.b32 %r18, %r1, 31;               // lane
    shr.u32 %r19, %r1, 5;                // warp within block

    ld.param.u32 %r3, [OUT_DIM];
    ld.param.u32 %r23, [BATCH];
    setp.ge.u32 %p0, %r22, %r23;
    @%p0 bra done;

    mad.lo.u32 %r24, %r0, 4, %r19;       // output row for this warp
    setp.ge.u32 %p1, %r24, %r3;
    @%p1 bra done;

    ld.param.u64 %rd0, [W];
    ld.param.u64 %rd1, [SCALE];
    ld.param.u64 %rd2, [BIAS];
    ld.param.u64 %rd3, [X];
    ld.param.u64 %rd4, [O];
    ld.param.u32 %r4, [IN_DIM];
    ld.param.u32 %r5, [SCALE_LEN];
    ld.param.u32 %r21, [HAS_BIAS];

    mov.f32 %f10, 0f00000000;            // sum
    mov.u32 %r6, %r18;                   // col = lane

loop_col:
    setp.ge.u32 %p2, %r6, %r4;
    @%p2 bra reduce_store;

    mad.lo.u32 %r7, %r24, %r4, %r6;      // row*inDim + col
    cvt.u64.u32 %rd5, %r7;
    add.u64 %rd6, %rd0, %rd5;
    ld.global.u8 %r8, [%rd6];            // E4M3 byte

    and.b32 %r9, %r8, 127;               // exp+mant without sign
    setp.eq.u32 %p3, %r9, 127;           // reserved NaN code
    @%p3 bra val_nan;

    and.b32 %r10, %r8, 128;              // sign
    shr.u32 %r11, %r8, 3;
    and.b32 %r11, %r11, 15;              // exp
    and.b32 %r12, %r8, 7;                // mant
    setp.eq.u32 %p4, %r11, 0;
    @%p4 bra val_subnormal;

    add.u32 %r13, %r11, 120;             // exp - 7 + 127
    shl.b32 %r13, %r13, 23;
    shl.b32 %r14, %r12, 20;
    or.b32 %r15, %r13, %r14;
    setp.ne.u32 %p5, %r10, 0;
    @%p5 or.b32 %r15, %r15, 2147483648;
    mov.b32 %f1, %r15;
    bra val_done;

val_subnormal:
    cvt.rn.f32.u32 %f1, %r12;
    mul.f32 %f1, %f1, 0f3B000000;        // mant * 2^-9
    setp.ne.u32 %p5, %r10, 0;
    @%p5 neg.f32 %f1, %f1;
    bra val_done;

val_nan:
    mov.f32 %f1, 0f7FC00000;

val_done:
    mad.lo.u32 %r25, %r22, %r4, %r6;     // batch*inDim + col
    mul.wide.u32 %rd7, %r25, 4;
    add.u64 %rd8, %rd3, %rd7;
    ld.global.f32 %f2, [%rd8];
    fma.rn.f32 %f10, %f1, %f2, %f10;
    add.u32 %r6, %r6, 32;
    bra loop_col;

reduce_store:
    mov.b32 %r16, %f10;

    shfl.sync.down.b32 %r17|%p6, %r16, 16, 0x1f, 0xffffffff;
    mov.b32 %f4, %r17;
    add.f32 %f10, %f10, %f4;
    mov.b32 %r16, %f10;

    shfl.sync.down.b32 %r17|%p6, %r16, 8, 0x1f, 0xffffffff;
    mov.b32 %f4, %r17;
    add.f32 %f10, %f10, %f4;
    mov.b32 %r16, %f10;

    shfl.sync.down.b32 %r17|%p6, %r16, 4, 0x1f, 0xffffffff;
    mov.b32 %f4, %r17;
    add.f32 %f10, %f10, %f4;
    mov.b32 %r16, %f10;

    shfl.sync.down.b32 %r17|%p6, %r16, 2, 0x1f, 0xffffffff;
    mov.b32 %f4, %r17;
    add.f32 %f10, %f10, %f4;
    mov.b32 %r16, %f10;

    shfl.sync.down.b32 %r17|%p6, %r16, 1, 0x1f, 0xffffffff;
    mov.b32 %f4, %r17;
    add.f32 %f10, %f10, %f4;

    setp.ne.u32 %p7, %r18, 0;
    @%p7 bra done;

    // Apply scale: scaleLen==1 uses scale[0], otherwise scale[row].
    setp.eq.u32 %p8, %r5, 1;
    @%p8 bra scale_tensor;
    mov.u32 %r20, %r24;
    bra scale_index_ready;
scale_tensor:
    mov.u32 %r20, 0;
scale_index_ready:
    mul.wide.u32 %rd14, %r20, 4;
    add.u64 %rd15, %rd1, %rd14;
    ld.global.f32 %f4, [%rd15];
    mul.f32 %f10, %f10, %f4;

    // Optional bias, controlled by explicit flag.
    setp.eq.u32 %p9, %r21, 0;
    @%p9 bra store_out;
    mul.wide.u32 %rd16, %r24, 4;
    add.u64 %rd17, %rd2, %rd16;
    ld.global.f32 %f5, [%rd17];
    add.f32 %f10, %f10, %f5;

store_out:
    mad.lo.u32 %r26, %r22, %r3, %r24;    // batch*outDim + row
    mul.wide.u32 %rd18, %r26, 4;
    add.u64 %rd19, %rd4, %rd18;
    st.global.f32 [%rd19], %f10;

done:
    ret;
}
`

// FP8E4M3DequantTransposeF32PTX dequantizes row-major FP8 W[out,in] into
// F32 transposed WT[in,out], applying per-tensor or per-row scale.
const FP8E4M3DequantTransposeF32PTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry fp8_e4m3_dequant_transpose_f32(
    .param .u64 W,
    .param .u64 SCALE,
    .param .u64 WT,
    .param .u32 OUT_DIM,
    .param .u32 IN_DIM,
    .param .u32 SCALE_LEN
) {
    .reg .pred %p<10>;
    .reg .u32 %r<32>;
    .reg .u64 %rd<18>;
    .reg .f32 %f<8>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;      // linear index
    ld.param.u32 %r4, [OUT_DIM];
    ld.param.u32 %r5, [IN_DIM];
    mul.lo.u32 %r6, %r4, %r5;           // total
    setp.ge.u32 %p0, %r3, %r6;
    @%p0 bra done;

    ld.param.u64 %rd0, [W];
    ld.param.u64 %rd1, [SCALE];
    ld.param.u64 %rd2, [WT];
    ld.param.u32 %r20, [SCALE_LEN];

    div.u32 %r7, %r3, %r5;              // row/out
    rem.u32 %r8, %r3, %r5;              // col/in
    cvt.u64.u32 %rd3, %r3;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.u8 %r9, [%rd4];

    and.b32 %r10, %r9, 127;
    setp.eq.u32 %p1, %r10, 127;
    @%p1 bra val_nan;
    and.b32 %r11, %r9, 128;
    shr.u32 %r12, %r9, 3;
    and.b32 %r12, %r12, 15;
    and.b32 %r13, %r9, 7;
    setp.eq.u32 %p2, %r12, 0;
    @%p2 bra val_subnormal;

    add.u32 %r14, %r12, 120;
    shl.b32 %r14, %r14, 23;
    shl.b32 %r15, %r13, 20;
    or.b32 %r16, %r14, %r15;
    setp.ne.u32 %p3, %r11, 0;
    @%p3 or.b32 %r16, %r16, 2147483648;
    mov.b32 %f1, %r16;
    bra val_done;

val_subnormal:
    cvt.rn.f32.u32 %f1, %r13;
    mul.f32 %f1, %f1, 0f3B000000;
    setp.ne.u32 %p3, %r11, 0;
    @%p3 neg.f32 %f1, %f1;
    bra val_done;

val_nan:
    mov.f32 %f1, 0f7FC00000;

val_done:
    setp.eq.u32 %p4, %r20, 1;
    @%p4 mov.u32 %r21, 0;
    @!%p4 mov.u32 %r21, %r7;
    mul.wide.u32 %rd5, %r21, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];
    mul.f32 %f1, %f1, %f2;

    mad.lo.u32 %r22, %r8, %r4, %r7;     // WT[col*out + row]
    mul.wide.u32 %rd7, %r22, 4;
    add.u64 %rd8, %rd2, %rd7;
    st.global.f32 [%rd8], %f1;

done:
    ret;
}
`
