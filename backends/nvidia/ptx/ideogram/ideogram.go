package ideogram

// IdeogramCFGStepPTX fuses asymmetric CFG and FlowMatch Euler update:
//
//	out[i] = latents[i] + sigma * (uncond[i] + guidance * (cond[i] - uncond[i]))
//
// It is a simple full-tensor vector kernel used by the Ideogram denoise loop
// once conditional and unconditional DiT velocities are available.
const IdeogramCFGStepPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry ideogram_cfg_step_f32(
    .param .u64 LATENTS,
    .param .u64 COND,
    .param .u64 UNCOND,
    .param .u64 OUT,
    .param .f32 GUIDANCE,
    .param .f32 SIGMA,
    .param .u32 N
) {
    .reg .pred %p<2>;
    .reg .u32 %r<8>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<12>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra done;

    ld.param.u64 %rd0, [LATENTS];
    ld.param.u64 %rd1, [COND];
    ld.param.u64 %rd2, [UNCOND];
    ld.param.u64 %rd3, [OUT];
    ld.param.f32 %f0, [GUIDANCE];
    ld.param.f32 %f1, [SIGMA];

    mul.wide.u32 %rd4, %r3, 4;
    add.u64 %rd5, %rd0, %rd4;
    add.u64 %rd6, %rd1, %rd4;
    add.u64 %rd7, %rd2, %rd4;
    add.u64 %rd8, %rd3, %rd4;

    ld.global.f32 %f2, [%rd5];      // latent
    ld.global.f32 %f3, [%rd6];      // cond
    ld.global.f32 %f4, [%rd7];      // uncond
    sub.f32 %f5, %f3, %f4;
    fma.rn.f32 %f6, %f0, %f5, %f4;  // guided
    fma.rn.f32 %f7, %f1, %f6, %f2;  // stepped
    st.global.f32 [%rd8], %f7;

done:
    ret;
}
`

// IdeogramLayerNormNoAffinePTX computes row-wise non-affine LayerNorm:
//
//	y[row,col] = (x[row,col] - mean(row)) * rsqrt(var(row) + eps)
//
// One CUDA block owns one row. Threads reduce both sum(x) and sum(x*x), then
// write normalized elements with a strided loop.
const IdeogramLayerNormNoAffinePTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry ideogram_layer_norm_no_affine_f32(
    .param .u64 X,
    .param .u64 O,
    .param .u32 ROWS,
    .param .u32 COLS,
    .param .f32 EPS
) {
    .reg .pred %p<10>;
    .reg .u32 %r<32>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<32>;
    .shared .align 4 .f32 ln_sum[256];
    .shared .align 4 .f32 ln_sumsq[256];

    mov.u32 %r0, %ctaid.x;              // row
    mov.u32 %r1, %tid.x;                // tid
    mov.u32 %r2, %ntid.x;               // blockDim
    ld.param.u32 %r3, [ROWS];
    setp.ge.u32 %p0, %r0, %r3;
    @%p0 bra done;

    ld.param.u64 %rd0, [X];
    ld.param.u64 %rd1, [O];
    ld.param.u32 %r4, [COLS];
    ld.param.f32 %f0, [EPS];

    mov.f32 %f1, 0f00000000;            // sum
    mov.f32 %f2, 0f00000000;            // sumsq
    mov.u32 %r5, %r1;                   // col = tid
    mad.lo.u32 %r6, %r0, %r4, 0;        // row offset elements

sum_loop:
    setp.ge.u32 %p1, %r5, %r4;
    @%p1 bra reduce;
    add.u32 %r7, %r6, %r5;
    mul.wide.u32 %rd2, %r7, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f3, [%rd3];
    add.f32 %f1, %f1, %f3;
    fma.rn.f32 %f2, %f3, %f3, %f2;
    add.u32 %r5, %r5, %r2;
    bra sum_loop;

reduce:
    mul.wide.u32 %rd4, %r1, 4;
    mov.u64 %rd5, ln_sum;
    mov.u64 %rd6, ln_sumsq;
    add.u64 %rd7, %rd5, %rd4;
    add.u64 %rd8, %rd6, %rd4;
    st.shared.f32 [%rd7], %f1;
    st.shared.f32 [%rd8], %f2;
    bar.sync 0;

    mov.u32 %r20, 128;
red_loop:
    setp.ge.u32 %p2, %r1, %r20;
    @%p2 bra red_skip;
    add.u32 %r21, %r1, %r20;
    setp.ge.u32 %p3, %r21, %r2;
    @%p3 bra red_skip;
    mul.wide.u32 %rd9, %r21, 4;
    add.u64 %rd10, %rd5, %rd9;
    add.u64 %rd11, %rd6, %rd9;
    ld.shared.f32 %f4, [%rd7];
    ld.shared.f32 %f5, [%rd10];
    add.f32 %f4, %f4, %f5;
    st.shared.f32 [%rd7], %f4;
    ld.shared.f32 %f6, [%rd8];
    ld.shared.f32 %f7, [%rd11];
    add.f32 %f6, %f6, %f7;
    st.shared.f32 [%rd8], %f6;
red_skip:
    bar.sync 0;
    shr.u32 %r20, %r20, 1;
    setp.gt.u32 %p4, %r20, 0;
    @%p4 bra red_loop;

    setp.ne.u32 %p5, %r1, 0;
    @%p5 bra wait_stats;
    ld.shared.f32 %f8, [ln_sum];
    ld.shared.f32 %f9, [ln_sumsq];
    cvt.rn.f32.u32 %f10, %r4;
    div.rn.f32 %f11, %f8, %f10;         // mean
    div.rn.f32 %f12, %f9, %f10;         // mean square
    mul.f32 %f13, %f11, %f11;
    sub.f32 %f14, %f12, %f13;           // variance = mean(x^2) - mean^2
    add.f32 %f14, %f14, %f0;
    sqrt.rn.f32 %f15, %f14;
    rcp.rn.f32 %f16, %f15;
    st.shared.f32 [ln_sum], %f11;
    st.shared.f32 [ln_sumsq], %f16;

wait_stats:
    bar.sync 0;
    ld.shared.f32 %f20, [ln_sum];       // mean
    ld.shared.f32 %f21, [ln_sumsq];     // inv std

    mov.u32 %r22, %r1;
out_loop:
    setp.ge.u32 %p6, %r22, %r4;
    @%p6 bra done;
    add.u32 %r23, %r6, %r22;
    mul.wide.u32 %rd12, %r23, 4;
    add.u64 %rd13, %rd0, %rd12;
    add.u64 %rd14, %rd1, %rd12;
    ld.global.f32 %f22, [%rd13];
    sub.f32 %f23, %f22, %f20;
    mul.f32 %f24, %f23, %f21;
    st.global.f32 [%rd14], %f24;
    add.u32 %r22, %r22, %r2;
    bra out_loop;

done:
    ret;
}
`

// IdeogramAdaLNTransformPTX transforms one DiT block adaLN modulation vector
// in-place. The input layout is [scale_msa, gate_msa, scale_mlp, gate_mlp],
// each length emb. It writes:
//
//	scale_* = 1 + scale_*
//	gate_*  = tanh(gate_*)
//
// The tanh implementation matches the existing PTX activation style using
// tanh(x) = 1 - 2/(1 + exp(2x)).
const IdeogramAdaLNTransformPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry ideogram_adaln_transform_f32(
    .param .u64 MOD,
    .param .u32 EMB
) {
    .reg .pred %p<4>;
    .reg .u32 %r<16>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<24>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;      // i
    ld.param.u32 %r4, [EMB];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra done;

    ld.param.u64 %rd0, [MOD];
    mul.wide.u32 %rd1, %r3, 4;

    // scale_msa[i] += 1
    add.u64 %rd2, %rd0, %rd1;
    ld.global.f32 %f0, [%rd2];
    add.f32 %f0, %f0, 0f3F800000;
    st.global.f32 [%rd2], %f0;

    // scale_mlp[i] += 1 (offset 2*emb)
    mul.lo.u32 %r5, %r4, 2;
    add.u32 %r6, %r5, %r3;
    mul.wide.u32 %rd3, %r6, 4;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];
    add.f32 %f1, %f1, 0f3F800000;
    st.global.f32 [%rd4], %f1;

    // gate_msa[i] = tanh(gate_msa[i]) (offset emb)
    add.u32 %r7, %r4, %r3;
    mul.wide.u32 %rd5, %r7, 4;
    add.u64 %rd6, %rd0, %rd5;
    ld.global.f32 %f2, [%rd6];
    mul.f32 %f3, %f2, 0f4038AA3B;       // 2*x*log2(e)
    ex2.approx.f32 %f3, %f3;            // exp(2x)
    add.f32 %f4, %f3, 0f3F800000;
    mov.f32 %f5, 0f40000000;
    div.approx.f32 %f5, %f5, %f4;
    mov.f32 %f6, 0f3F800000;
    sub.f32 %f6, %f6, %f5;
    st.global.f32 [%rd6], %f6;

    // gate_mlp[i] = tanh(gate_mlp[i]) (offset 3*emb)
    mul.lo.u32 %r8, %r4, 3;
    add.u32 %r9, %r8, %r3;
    mul.wide.u32 %rd7, %r9, 4;
    add.u64 %rd8, %rd0, %rd7;
    ld.global.f32 %f7, [%rd8];
    mul.f32 %f8, %f7, 0f4038AA3B;
    ex2.approx.f32 %f8, %f8;
    add.f32 %f9, %f8, 0f3F800000;
    mov.f32 %f10, 0f40000000;
    div.approx.f32 %f10, %f10, %f9;
    mov.f32 %f11, 0f3F800000;
    sub.f32 %f11, %f11, %f10;
    st.global.f32 [%rd8], %f11;

done:
    ret;
}
`

// IdeogramGatedResidualPTX computes hidden[i] += gate[i] * update[i]. It is
// used for adaLN-gated attention and MLP residuals after post-norm.
const IdeogramGatedResidualPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry ideogram_gated_residual_f32(
    .param .u64 HIDDEN,
    .param .u64 UPDATE,
    .param .u64 GATE,
    .param .u32 N
) {
    .reg .pred %p<2>;
    .reg .u32 %r<8>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<8>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra done;

    ld.param.u64 %rd0, [HIDDEN];
    ld.param.u64 %rd1, [UPDATE];
    ld.param.u64 %rd2, [GATE];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd3;

    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    fma.rn.f32 %f3, %f2, %f1, %f0;
    st.global.f32 [%rd4], %f3;

done:
    ret;
}
`
