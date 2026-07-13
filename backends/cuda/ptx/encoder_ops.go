package ptx

// EncoderRowAffinePTX applies a shared affine transform to row-major data.
const EncoderRowAffinePTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry whisper_row_affine_f32(
    .param .u64 X, .param .u64 WEIGHT, .param .u64 BIAS,
    .param .u64 OUT, .param .u32 ROWS, .param .u32 COLS
) {
    .reg .pred %p<2>; .reg .u32 %r<8>; .reg .u64 %rd<12>; .reg .f32 %f<4>;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [ROWS]; ld.param.u32 %r5, [COLS];
    mul.lo.u32 %r6, %r4, %r5; setp.ge.u32 %p0, %r3, %r6; @%p0 bra done;
    rem.u32 %r7, %r3, %r5;
    ld.param.u64 %rd0, [X]; ld.param.u64 %rd1, [WEIGHT]; ld.param.u64 %rd2, [BIAS]; ld.param.u64 %rd3, [OUT];
    mul.wide.u32 %rd4, %r3, 4; mul.wide.u32 %rd5, %r7, 4;
    add.u64 %rd6, %rd0, %rd4; add.u64 %rd7, %rd1, %rd5; add.u64 %rd8, %rd2, %rd5; add.u64 %rd9, %rd3, %rd4;
    ld.global.f32 %f0, [%rd6]; ld.global.f32 %f1, [%rd7]; ld.global.f32 %f2, [%rd8];
    fma.rn.f32 %f3, %f0, %f1, %f2; st.global.f32 [%rd9], %f3;
done: ret;
}`

// EncoderRowBiasPTX adds a shared bias vector to a row-major matrix in-place.
const EncoderRowBiasPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry whisper_row_bias_f32(
    .param .u64 X, .param .u64 BIAS, .param .u32 ROWS, .param .u32 COLS
) {
    .reg .pred %p; .reg .u32 %r<8>; .reg .u64 %rd<8>; .reg .f32 %f<3>;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x; mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [ROWS]; ld.param.u32 %r5, [COLS]; mul.lo.u32 %r6, %r4, %r5;
    setp.ge.u32 %p, %r3, %r6; @%p bra done; rem.u32 %r7, %r3, %r5;
    ld.param.u64 %rd0, [X]; ld.param.u64 %rd1, [BIAS]; mul.wide.u32 %rd2, %r3, 4; mul.wide.u32 %rd3, %r7, 4;
    add.u64 %rd4, %rd0, %rd2; add.u64 %rd5, %rd1, %rd3; ld.global.f32 %f0, [%rd4]; ld.global.f32 %f1, [%rd5];
    add.rn.f32 %f2, %f0, %f1; st.global.f32 [%rd4], %f2;
done: ret;
}`

// EncoderTransposePTX transposes a row-major [rows,cols] F32 matrix.
const EncoderTransposePTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry whisper_transpose_f32(
    .param .u64 IN, .param .u64 OUT, .param .u32 ROWS, .param .u32 COLS
) {
    .reg .pred %p; .reg .u32 %r<10>; .reg .u64 %rd<8>; .reg .f32 %f<2>;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x; mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [ROWS]; ld.param.u32 %r5, [COLS]; mul.lo.u32 %r6, %r4, %r5;
    setp.ge.u32 %p, %r3, %r6; @%p bra done; div.u32 %r7, %r3, %r5; rem.u32 %r8, %r3, %r5; mad.lo.u32 %r9, %r8, %r4, %r7;
    ld.param.u64 %rd0, [IN]; ld.param.u64 %rd1, [OUT]; mul.wide.u32 %rd2, %r3, 4; mul.wide.u32 %rd3, %r9, 4;
    add.u64 %rd4, %rd0, %rd2; add.u64 %rd5, %rd1, %rd3; ld.global.f32 %f0, [%rd4]; st.global.f32 [%rd5], %f0;
done: ret;
}`

// EncoderGELUTanhPTX applies the standard tanh GELU approximation in-place.
const EncoderGELUTanhPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry whisper_gelu_tanh_f32(.param .u64 X, .param .u32 N) {
    .reg .pred %p; .reg .u32 %r<6>; .reg .u64 %rd<5>; .reg .f32 %f<16>;
    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x; mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N]; setp.ge.u32 %p, %r3, %r4; @%p bra done;
    ld.param.u64 %rd0, [X]; mul.wide.u32 %rd1, %r3, 4; add.u64 %rd2, %rd0, %rd1; ld.global.f32 %f0, [%rd2];
    mul.f32 %f1, %f0, %f0; mul.f32 %f2, %f1, %f0; mul.f32 %f2, %f2, 0f3D372713;
    add.f32 %f3, %f0, %f2; mul.f32 %f3, %f3, 0f3F4C422A;
    mul.f32 %f4, %f3, 0f4038AA3B; ex2.approx.f32 %f5, %f4; add.f32 %f6, %f5, 0f3F800000;
    mov.f32 %f7, 0f40000000; div.rn.f32 %f8, %f7, %f6; mov.f32 %f9, 0f3F800000; sub.f32 %f10, %f9, %f8;
    add.f32 %f11, %f10, %f9; mul.f32 %f12, %f11, 0f3F000000; mul.f32 %f13, %f0, %f12; st.global.f32 [%rd2], %f13;
done: ret;
}`
