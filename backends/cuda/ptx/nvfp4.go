package ptx

// NVFP4DequantF32PTX materializes ModelOpt/NVFP4 weights to row-major F32.
// One CUDA thread decodes one logical weight element from packed E2M1 FP4 plus
// F8_E4M3FN block scale and scalar weight_scale_2.
const NVFP4DequantF32PTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry nvfp4_dequant_f32(
    .param .u64 W,
    .param .u64 S,
    .param .u64 O,
    .param .f32 SCALE2,
    .param .u32 TOTAL,
    .param .u32 IN_DIM,
    .param .u32 GROUP_SIZE
) {
    .reg .pred %p<8>;
    .reg .u32 %r<40>;
    .reg .u64 %rd<14>;
    .reg .f32 %f<16>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;        // idx
    ld.param.u32 %r4, [TOTAL];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra done;

    ld.param.u64 %rd0, [W];
    ld.param.u64 %rd1, [S];
    ld.param.u64 %rd2, [O];
    ld.param.f32 %f0, [SCALE2];
    ld.param.u32 %r5, [IN_DIM];
    ld.param.u32 %r6, [GROUP_SIZE];

    div.u32 %r7, %r3, %r5;                // row
    rem.u32 %r8, %r3, %r5;                // col
    shr.u32 %r9, %r5, 1;                  // packed bytes per row
    mad.lo.u32 %r10, %r7, %r9, 0;         // row byte offset
    shr.u32 %r26, %r8, 1;                 // col/2 packed byte within row
    add.u32 %r10, %r10, %r26;             // weight byte offset
    cvt.u64.u32 %rd7, %r10;
    add.u64 %rd3, %rd0, %rd7;
    ld.global.u8 %r11, [%rd3];
    and.b32 %r12, %r8, 1;
    setp.ne.u32 %p1, %r12, 0;
    @%p1 shr.u32 %r11, %r11, 4;
    and.b32 %r11, %r11, 15;               // FP4 code

    and.b32 %r13, %r11, 7;                // magnitude code
    and.b32 %r14, %r11, 8;                // sign bit
    mov.f32 %f1, 0f00000000;
    setp.eq.u32 %p2, %r13, 1; @%p2 mov.f32 %f1, 0f3F000000;
    setp.eq.u32 %p2, %r13, 2; @%p2 mov.f32 %f1, 0f3F800000;
    setp.eq.u32 %p2, %r13, 3; @%p2 mov.f32 %f1, 0f3FC00000;
    setp.eq.u32 %p2, %r13, 4; @%p2 mov.f32 %f1, 0f40000000;
    setp.eq.u32 %p2, %r13, 5; @%p2 mov.f32 %f1, 0f40400000;
    setp.eq.u32 %p2, %r13, 6; @%p2 mov.f32 %f1, 0f40800000;
    setp.eq.u32 %p2, %r13, 7; @%p2 mov.f32 %f1, 0f40C00000;
    setp.ne.u32 %p3, %r14, 0;
    @%p3 neg.f32 %f1, %f1;

    div.u32 %r15, %r8, %r6;               // group
    div.u32 %r16, %r5, %r6;               // groups per row
    mad.lo.u32 %r17, %r7, %r16, %r15;
    cvt.u64.u32 %rd8, %r17;
    add.u64 %rd4, %rd1, %rd8;
    ld.global.u8 %r18, [%rd4];            // E4M3FN scale code

    and.b32 %r19, %r18, 127;
    setp.eq.u32 %p4, %r19, 127;
    @%p4 bra scale_nan;
    and.b32 %r20, %r18, 128;              // sign
    shr.u32 %r21, %r18, 3;
    and.b32 %r21, %r21, 15;               // exp
    and.b32 %r22, %r18, 7;                // mant
    setp.eq.u32 %p5, %r21, 0;
    @%p5 bra scale_subnormal;

    add.u32 %r23, %r21, 120;              // exp - 7 + 127
    shl.b32 %r23, %r23, 23;
    shl.b32 %r24, %r22, 20;
    or.b32 %r25, %r23, %r24;
    setp.ne.u32 %p6, %r20, 0;
    @%p6 or.b32 %r25, %r25, 2147483648;
    mov.b32 %f2, %r25;
    bra scale_done;

scale_subnormal:
    cvt.rn.f32.u32 %f2, %r22;
    mul.f32 %f2, %f2, 0f3B000000;         // mant * 2^-9
    setp.ne.u32 %p6, %r20, 0;
    @%p6 neg.f32 %f2, %f2;
    bra scale_done;

scale_nan:
    mov.f32 %f2, 0f7FC00000;

scale_done:
    mul.f32 %f3, %f1, %f2;
    mul.f32 %f3, %f3, %f0;
    mul.wide.u32 %rd5, %r3, 4;
    add.u64 %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f3;

done:
    ret;
}
`

// NVFP4GemvF32PTX computes out[row] = dot(dequant(W[row,:]), x).
const NVFP4GemvF32PTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry nvfp4_gemv_f32(
    .param .u64 W,
    .param .u64 S,
    .param .u64 X,
    .param .u64 O,
    .param .f32 SCALE2,
    .param .u32 OUT_DIM,
    .param .u32 IN_DIM,
    .param .u32 GROUP_SIZE
) {
    .reg .pred %p<12>;
    .reg .u32 %r<64>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<32>;
    .shared .align 4 .f32 nvfp4_sdata[256];

    mov.u32 %r0, %ctaid.x;               // row
    mov.u32 %r1, %tid.x;                 // tid
    mov.u32 %r2, %ntid.x;                // blockDim
    ld.param.u32 %r3, [OUT_DIM];
    setp.ge.u32 %p0, %r0, %r3;
    @%p0 bra done;

    ld.param.u64 %rd0, [W];
    ld.param.u64 %rd1, [S];
    ld.param.u64 %rd2, [X];
    ld.param.u64 %rd3, [O];
    ld.param.f32 %f0, [SCALE2];
    ld.param.u32 %r4, [IN_DIM];
    ld.param.u32 %r5, [GROUP_SIZE];

    mov.f32 %f10, 0f00000000;            // sum
    mov.u32 %r6, %r1;                    // col = tid

loop_col:
    setp.ge.u32 %p1, %r6, %r4;
    @%p1 bra reduce_store;

    shr.u32 %r7, %r4, 1;                 // packed bytes per row
    mad.lo.u32 %r8, %r0, %r7, 0;         // row byte offset
    shr.u32 %r9, %r6, 1;
    add.u32 %r8, %r8, %r9;
    cvt.u64.u32 %rd4, %r8;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.u8 %r10, [%rd5];
    and.b32 %r11, %r6, 1;
    setp.ne.u32 %p2, %r11, 0;
    @%p2 shr.u32 %r10, %r10, 4;
    and.b32 %r10, %r10, 15;

    and.b32 %r12, %r10, 7;
    and.b32 %r13, %r10, 8;
    mov.f32 %f1, 0f00000000;
    setp.eq.u32 %p3, %r12, 1; @%p3 mov.f32 %f1, 0f3F000000;
    setp.eq.u32 %p3, %r12, 2; @%p3 mov.f32 %f1, 0f3F800000;
    setp.eq.u32 %p3, %r12, 3; @%p3 mov.f32 %f1, 0f3FC00000;
    setp.eq.u32 %p3, %r12, 4; @%p3 mov.f32 %f1, 0f40000000;
    setp.eq.u32 %p3, %r12, 5; @%p3 mov.f32 %f1, 0f40400000;
    setp.eq.u32 %p3, %r12, 6; @%p3 mov.f32 %f1, 0f40800000;
    setp.eq.u32 %p3, %r12, 7; @%p3 mov.f32 %f1, 0f40C00000;
    setp.ne.u32 %p4, %r13, 0;
    @%p4 neg.f32 %f1, %f1;

    div.u32 %r14, %r6, %r5;              // group
    div.u32 %r15, %r4, %r5;              // groups per row
    mad.lo.u32 %r16, %r0, %r15, %r14;
    cvt.u64.u32 %rd6, %r16;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.u8 %r17, [%rd7];

    and.b32 %r18, %r17, 127;
    setp.eq.u32 %p5, %r18, 127;
    @%p5 bra scale_nan;
    and.b32 %r19, %r17, 128;
    shr.u32 %r20, %r17, 3;
    and.b32 %r20, %r20, 15;
    and.b32 %r21, %r17, 7;
    setp.eq.u32 %p6, %r20, 0;
    @%p6 bra scale_subnormal;
    add.u32 %r22, %r20, 120;
    shl.b32 %r22, %r22, 23;
    shl.b32 %r23, %r21, 20;
    or.b32 %r24, %r22, %r23;
    setp.ne.u32 %p7, %r19, 0;
    @%p7 or.b32 %r24, %r24, 2147483648;
    mov.b32 %f2, %r24;
    bra scale_done;
scale_subnormal:
    cvt.rn.f32.u32 %f2, %r21;
    mul.f32 %f2, %f2, 0f3B000000;
    setp.ne.u32 %p7, %r19, 0;
    @%p7 neg.f32 %f2, %f2;
    bra scale_done;
scale_nan:
    mov.f32 %f2, 0f7FC00000;
scale_done:
    mul.f32 %f3, %f1, %f2;
    mul.f32 %f3, %f3, %f0;
    mul.wide.u32 %rd8, %r6, 4;
    add.u64 %rd9, %rd2, %rd8;
    ld.global.f32 %f4, [%rd9];
    fma.rn.f32 %f10, %f3, %f4, %f10;
    add.u32 %r6, %r6, %r2;
    bra loop_col;

reduce_store:
    mul.wide.u32 %rd10, %r1, 4;
    mov.u64 %rd11, nvfp4_sdata;
    add.u64 %rd12, %rd11, %rd10;
    st.shared.f32 [%rd12], %f10;
    bar.sync 0;

    mov.u32 %r30, 128;
red_loop:
    setp.ge.u32 %p8, %r1, %r30;
    @%p8 bra red_skip;
    add.u32 %r31, %r1, %r30;
    setp.ge.u32 %p9, %r31, %r2;
    @%p9 bra red_skip;
    mul.wide.u32 %rd13, %r31, 4;
    add.u64 %rd14, %rd11, %rd13;
    ld.shared.f32 %f11, [%rd12];
    ld.shared.f32 %f12, [%rd14];
    add.f32 %f11, %f11, %f12;
    st.shared.f32 [%rd12], %f11;
red_skip:
    bar.sync 0;
    shr.u32 %r30, %r30, 1;
    setp.gt.u32 %p10, %r30, 0;
    @%p10 bra red_loop;

    setp.ne.u32 %p11, %r1, 0;
    @%p11 bra done;
    ld.shared.f32 %f13, [nvfp4_sdata];
    mul.wide.u32 %rd15, %r0, 4;
    add.u64 %rd16, %rd3, %rd15;
    st.global.f32 [%rd16], %f13;

done:
    ret;
}
`
