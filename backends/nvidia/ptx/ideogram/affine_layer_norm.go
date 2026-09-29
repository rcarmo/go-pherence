package ideogram

// AffineLayerNormF32PTX normalizes each row with its own centered variance,
// then applies gamma and beta. One 256-thread block owns a row; a second pass
// avoids cancellation in E[x²]-E[x]² for large-offset inputs.
const AffineLayerNormF32PTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry affine_layer_norm_f32(
    .param .u64 OUT, .param .u64 X, .param .u64 GAMMA, .param .u64 BETA,
    .param .u32 ROWS, .param .u32 COLS, .param .f32 EPS
) {
    .reg .pred %p<8>;
    .reg .u32 %r<24>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<16>;
    .shared .align 4 .f32 partial[256];
    .shared .align 4 .f32 stats[2];

    ld.param.u32 %r0, [ROWS];
    ld.param.u32 %r1, [COLS];
    ld.param.u64 %rd0, [OUT];
    ld.param.u64 %rd1, [X];
    ld.param.u64 %rd2, [GAMMA];
    ld.param.u64 %rd3, [BETA];
    ld.param.f32 %f0, [EPS];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    setp.ge.u32 %p0, %r2, %r0;
    @%p0 bra done;
    mul.lo.u32 %r4, %r2, %r1;
    mul.wide.u32 %rd4, %r3, 4;
    mov.u64 %rd5, partial;
    add.u64 %rd6, %rd5, %rd4;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r5, %r3;
mean_loop:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra mean_reduce;
    add.u32 %r6, %r4, %r5;
    mul.wide.u32 %rd7, %r6, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.f32 %f2, [%rd8];
    add.f32 %f1, %f1, %f2;
    add.u32 %r5, %r5, 256;
    bra mean_loop;
mean_reduce:
    st.shared.f32 [%rd6], %f1;
    bar.sync 0;
    mov.u32 %r7, 128;
mean_reduce_loop:
    setp.ge.u32 %p2, %r3, %r7;
    @%p2 bra mean_reduce_skip;
    add.u32 %r8, %r3, %r7;
    mul.wide.u32 %rd9, %r8, 4;
    add.u64 %rd10, %rd5, %rd9;
    ld.shared.f32 %f3, [%rd6];
    ld.shared.f32 %f4, [%rd10];
    add.f32 %f3, %f3, %f4;
    st.shared.f32 [%rd6], %f3;
mean_reduce_skip:
    bar.sync 0;
    shr.u32 %r7, %r7, 1;
    setp.ne.u32 %p3, %r7, 0;
    @%p3 bra mean_reduce_loop;
    setp.ne.u32 %p4, %r3, 0;
    @%p4 bra mean_ready;
    ld.shared.f32 %f5, [partial];
    cvt.rn.f32.u32 %f6, %r1;
    div.rn.f32 %f5, %f5, %f6;
    st.shared.f32 [stats], %f5;
mean_ready:
    bar.sync 0;
    ld.shared.f32 %f5, [stats];
    mov.f32 %f1, 0f00000000;
    mov.u32 %r5, %r3;
variance_loop:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra variance_reduce;
    add.u32 %r6, %r4, %r5;
    mul.wide.u32 %rd7, %r6, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.f32 %f2, [%rd8];
    sub.f32 %f2, %f2, %f5;
    fma.rn.f32 %f1, %f2, %f2, %f1;
    add.u32 %r5, %r5, 256;
    bra variance_loop;
variance_reduce:
    st.shared.f32 [%rd6], %f1;
    bar.sync 0;
    mov.u32 %r7, 128;
variance_reduce_loop:
    setp.ge.u32 %p2, %r3, %r7;
    @%p2 bra variance_reduce_skip;
    add.u32 %r8, %r3, %r7;
    mul.wide.u32 %rd9, %r8, 4;
    add.u64 %rd10, %rd5, %rd9;
    ld.shared.f32 %f3, [%rd6];
    ld.shared.f32 %f4, [%rd10];
    add.f32 %f3, %f3, %f4;
    st.shared.f32 [%rd6], %f3;
variance_reduce_skip:
    bar.sync 0;
    shr.u32 %r7, %r7, 1;
    setp.ne.u32 %p3, %r7, 0;
    @%p3 bra variance_reduce_loop;
    setp.ne.u32 %p4, %r3, 0;
    @%p4 bra variance_ready;
    ld.shared.f32 %f7, [partial];
    cvt.rn.f32.u32 %f6, %r1;
    div.rn.f32 %f7, %f7, %f6;
    add.f32 %f7, %f7, %f0;
    sqrt.rn.f32 %f7, %f7;
    rcp.rn.f32 %f7, %f7;
    st.shared.f32 [stats+4], %f7;
variance_ready:
    bar.sync 0;
    ld.shared.f32 %f7, [stats+4];
    mov.u32 %r5, %r3;
output_loop:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra done;
    add.u32 %r6, %r4, %r5;
    mul.wide.u32 %rd7, %r6, 4;
    add.u64 %rd8, %rd1, %rd7;
    add.u64 %rd11, %rd0, %rd7;
    mul.wide.u32 %rd12, %r5, 4;
    add.u64 %rd13, %rd2, %rd12;
    add.u64 %rd14, %rd3, %rd12;
    ld.global.f32 %f2, [%rd8];
    ld.global.f32 %f8, [%rd13];
    ld.global.f32 %f9, [%rd14];
    sub.f32 %f2, %f2, %f5;
    mul.f32 %f2, %f2, %f7;
    fma.rn.f32 %f2, %f2, %f8, %f9;
    st.global.f32 [%rd11], %f2;
    add.u32 %r5, %r5, 256;
    bra output_loop;
done:
    ret;
}
`
