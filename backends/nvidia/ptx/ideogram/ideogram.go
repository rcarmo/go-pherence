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

// IdeogramMRoPEPTX applies precomputed Ideogram MRoPE tables to a row-major
// [tokens, heads, head_dim] tensor in place. Cos/sin are [tokens, head_dim/2]
// and are shared across heads. The rotation is NeoX rotate-half:
//
//	y[j]      = x[j]*cos - x[j+half]*sin
//	y[j+half] = x[j+half]*cos + x[j]*sin
const IdeogramMRoPEPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry ideogram_mrope_f32(
    .param .u64 X,
    .param .u64 COS,
    .param .u64 SIN,
    .param .u32 TOKENS,
    .param .u32 HEADS,
    .param .u32 HEAD_DIM
) {
    .reg .pred %p<4>;
    .reg .u32 %r<48>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<16>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;      // linear pair index

    ld.param.u32 %r4, [TOKENS];
    ld.param.u32 %r5, [HEADS];
    ld.param.u32 %r6, [HEAD_DIM];
    shr.u32 %r7, %r6, 1;                // half
    mul.lo.u32 %r8, %r4, %r5;
    mul.lo.u32 %r9, %r8, %r7;           // total pairs
    setp.ge.u32 %p0, %r3, %r9;
    @%p0 bra done;

    ld.param.u64 %rd0, [X];
    ld.param.u64 %rd1, [COS];
    ld.param.u64 %rd2, [SIN];

    rem.u32 %r10, %r3, %r7;             // pair j
    div.u32 %r11, %r3, %r7;             // token*heads + head
    rem.u32 %r12, %r11, %r5;            // head
    div.u32 %r13, %r11, %r5;            // token

    // Base element = ((token*heads + head) * head_dim)
    mad.lo.u32 %r14, %r13, %r5, %r12;
    mul.lo.u32 %r15, %r14, %r6;
    add.u32 %r16, %r15, %r10;           // offset first half
    add.u32 %r17, %r16, %r7;            // offset second half

    mul.wide.u32 %rd3, %r16, 4;
    add.u64 %rd4, %rd0, %rd3;
    mul.wide.u32 %rd5, %r17, 4;
    add.u64 %rd6, %rd0, %rd5;

    // table offset = token*half + pair
    mad.lo.u32 %r18, %r13, %r7, %r10;
    mul.wide.u32 %rd7, %r18, 4;
    add.u64 %rd8, %rd1, %rd7;
    add.u64 %rd9, %rd2, %rd7;

    ld.global.f32 %f0, [%rd4];          // x1
    ld.global.f32 %f1, [%rd6];          // x2
    ld.global.f32 %f2, [%rd8];          // cos
    ld.global.f32 %f3, [%rd9];          // sin

    mul.f32 %f4, %f0, %f2;
    mul.f32 %f5, %f1, %f3;
    sub.f32 %f6, %f4, %f5;

    mul.f32 %f7, %f1, %f2;
    fma.rn.f32 %f8, %f0, %f3, %f7;

    st.global.f32 [%rd4], %f6;
    st.global.f32 [%rd6], %f8;

done:
    ret;
}
`

// IdeogramAttentionScoresPTX computes full non-causal attention scores for a
// token-major Q/K tensor: Q,K are [tokens, heads, head_dim], scores are
// [heads, tokens, tokens]. One thread computes one (head, query, key) dot.
const IdeogramAttentionScoresPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry ideogram_attention_scores_f32(
    .param .u64 Q,
    .param .u64 K,
    .param .u64 SCORES,
    .param .u32 TOKENS,
    .param .u32 HEADS,
    .param .u32 HEAD_DIM,
    .param .f32 SCALE
) {
    .reg .pred %p<4>;
    .reg .u32 %r<48>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<16>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;      // linear score index

    ld.param.u32 %r4, [TOKENS];
    ld.param.u32 %r5, [HEADS];
    ld.param.u32 %r6, [HEAD_DIM];
    mul.lo.u32 %r7, %r4, %r4;
    mul.lo.u32 %r8, %r7, %r5;           // total scores
    setp.ge.u32 %p0, %r3, %r8;
    @%p0 bra done;

    ld.param.u64 %rd0, [Q];
    ld.param.u64 %rd1, [K];
    ld.param.u64 %rd2, [SCORES];
    ld.param.f32 %f0, [SCALE];

    rem.u32 %r9, %r3, %r4;              // key token tj
    div.u32 %r10, %r3, %r4;
    rem.u32 %r11, %r10, %r4;            // query token ti
    div.u32 %r12, %r10, %r4;            // head h

    mov.f32 %f1, 0f00000000;            // dot
    mov.u32 %r13, 0;                    // d
score_loop:
    setp.ge.u32 %p1, %r13, %r6;
    @%p1 bra store_score;

    // q offset = ((ti*heads + h) * headDim + d)
    mad.lo.u32 %r14, %r11, %r5, %r12;
    mad.lo.u32 %r15, %r14, %r6, %r13;
    mul.wide.u32 %rd3, %r15, 4;
    add.u64 %rd4, %rd0, %rd3;

    // k offset = ((tj*heads + h) * headDim + d)
    mad.lo.u32 %r16, %r9, %r5, %r12;
    mad.lo.u32 %r17, %r16, %r6, %r13;
    mul.wide.u32 %rd5, %r17, 4;
    add.u64 %rd6, %rd1, %rd5;

    ld.global.f32 %f2, [%rd4];
    ld.global.f32 %f3, [%rd6];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r13, %r13, 1;
    bra score_loop;

store_score:
    mul.f32 %f1, %f1, %f0;
    mul.wide.u32 %rd7, %r3, 4;
    add.u64 %rd8, %rd2, %rd7;
    st.global.f32 [%rd8], %f1;

done:
    ret;
}
`

// IdeogramAttentionValuesPTX computes O = softmax(scores) * V. Probabilities
// are [heads, tokens, tokens]; V and O are token-major [tokens, heads,
// head_dim]. One thread computes one output element.
const IdeogramAttentionValuesPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry ideogram_attention_values_f32(
    .param .u64 PROBS,
    .param .u64 V,
    .param .u64 O,
    .param .u32 TOKENS,
    .param .u32 HEADS,
    .param .u32 HEAD_DIM
) {
    .reg .pred %p<4>;
    .reg .u32 %r<48>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<16>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;      // output element index

    ld.param.u32 %r4, [TOKENS];
    ld.param.u32 %r5, [HEADS];
    ld.param.u32 %r6, [HEAD_DIM];
    mul.lo.u32 %r7, %r4, %r5;
    mul.lo.u32 %r8, %r7, %r6;           // total output elems
    setp.ge.u32 %p0, %r3, %r8;
    @%p0 bra done;

    ld.param.u64 %rd0, [PROBS];
    ld.param.u64 %rd1, [V];
    ld.param.u64 %rd2, [O];

    rem.u32 %r9, %r3, %r6;              // d
    div.u32 %r10, %r3, %r6;
    rem.u32 %r11, %r10, %r5;            // h
    div.u32 %r12, %r10, %r5;            // query token ti

    mov.f32 %f1, 0f00000000;            // sum
    mov.u32 %r13, 0;                    // key token tj
value_loop:
    setp.ge.u32 %p1, %r13, %r4;
    @%p1 bra store_value;

    // prob offset = (h*tokens + ti)*tokens + tj
    mad.lo.u32 %r14, %r11, %r4, %r12;
    mad.lo.u32 %r15, %r14, %r4, %r13;
    mul.wide.u32 %rd3, %r15, 4;
    add.u64 %rd4, %rd0, %rd3;

    // v offset = ((tj*heads + h) * headDim + d)
    mad.lo.u32 %r16, %r13, %r5, %r11;
    mad.lo.u32 %r17, %r16, %r6, %r9;
    mul.wide.u32 %rd5, %r17, 4;
    add.u64 %rd6, %rd1, %rd5;

    ld.global.f32 %f2, [%rd4];
    ld.global.f32 %f3, [%rd6];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r13, %r13, 1;
    bra value_loop;

store_value:
    mul.wide.u32 %rd7, %r3, 4;
    add.u64 %rd8, %rd2, %rd7;
    st.global.f32 [%rd8], %f1;

done:
    ret;
}
`

// IdeogramLatentDenormPTX applies per-channel latent denormalization in place:
// x[token, channel] = x[token, channel] * scale[channel] + shift[channel].
const IdeogramLatentDenormPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry ideogram_latent_denorm_f32(
    .param .u64 x,
    .param .u64 scale,
    .param .u64 shift,
    .param .u32 n,
    .param .u32 channels
)
{
    .reg .pred %p<3>;
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [scale];
    ld.param.u64 %rd3, [shift];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r2, [channels];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra DONE;

    rem.u32 %r7, %r6, %r2;
    mul.wide.u32 %rd4, %r6, 4;
    add.u64 %rd5, %rd1, %rd4;
    mul.wide.u32 %rd6, %r7, 4;
    add.u64 %rd7, %rd2, %rd6;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd7];
    add.u64 %rd7, %rd3, %rd6;
    ld.global.f32 %f3, [%rd7];
    fma.rn.f32 %f4, %f1, %f2, %f3;
    st.global.f32 [%rd5], %f4;
DONE:
    ret;
}`

// IdeogramRGBClampF32PTX converts CHW RGB float values in [-1,1] to clamped
// interleaved RGB float values in [0,255]. The host can then round/cast to u8.
const IdeogramRGBClampF32PTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry ideogram_rgb_clamp_f32(
    .param .u64 out,
    .param .u64 in,
    .param .u32 hw
)
{
    .reg .pred %p<5>;
    .reg .u32 %r<8>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [in];
    ld.param.u32 %r1, [hw];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.u32 %r5, %r3, %r4, %r2;
    mul.lo.u32 %r6, %r1, 3;
    setp.ge.u32 %p1, %r5, %r6;
    @%p1 bra DONE;

    rem.u32 %r7, %r5, 3;        // channel in interleaved output
    div.u32 %r2, %r5, 3;        // pixel
    mad.lo.u32 %r3, %r7, %r1, %r2; // CHW input index

    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    mul.rn.f32 %f2, %f1, 0f3f000000; // 0.5
    add.rn.f32 %f3, %f2, 0f3f000000; // +0.5
    mul.rn.f32 %f4, %f3, 0f437f0000; // *255
    max.f32 %f5, %f4, 0f00000000;
    min.f32 %f6, %f5, 0f437f0000;

    mul.wide.u32 %rd5, %r5, 4;
    add.u64 %rd6, %rd1, %rd5;
    st.global.f32 [%rd6], %f6;
DONE:
    ret;
}`

// IdeogramUpsampleNearestPTX upsamples CHW feature maps by nearest-neighbour.
const IdeogramUpsampleNearestPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry ideogram_upsample_nearest_f32(
    .param .u64 out,
    .param .u64 in,
    .param .u32 c,
    .param .u32 h,
    .param .u32 w,
    .param .u32 factor,
    .param .u32 total
)
{
    .reg .pred %p<2>;
    .reg .u32 %r<18>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [in];
    ld.param.u32 %r1, [c];
    ld.param.u32 %r2, [h];
    ld.param.u32 %r3, [w];
    ld.param.u32 %r4, [factor];
    ld.param.u32 %r5, [total];

    mov.u32 %r6, %tid.x;
    mov.u32 %r7, %ctaid.x;
    mov.u32 %r8, %ntid.x;
    mad.lo.u32 %r9, %r7, %r8, %r6;
    setp.ge.u32 %p1, %r9, %r5;
    @%p1 bra DONE;

    mul.lo.u32 %r10, %r2, %r4; // outH
    mul.lo.u32 %r11, %r3, %r4; // outW
    div.u32 %r12, %r9, %r11;   // c*outH + y
    rem.u32 %r13, %r9, %r11;   // x
    div.u32 %r14, %r12, %r10;  // ch
    rem.u32 %r15, %r12, %r10;  // y
    div.u32 %r16, %r15, %r4;   // sy
    div.u32 %r17, %r13, %r4;   // sx
    mul.lo.u32 %r6, %r14, %r2;
    add.u32 %r6, %r6, %r16;
    mul.lo.u32 %r6, %r6, %r3;
    add.u32 %r6, %r6, %r17;

    mul.wide.u32 %rd3, %r6, 4;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    mul.wide.u32 %rd5, %r9, 4;
    add.u64 %rd6, %rd1, %rd5;
    st.global.f32 [%rd6], %f1;
DONE:
    ret;
}`

// IdeogramUnpatchifyPTX converts token-major patchified latents to CHW feature map.
const IdeogramUnpatchifyPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry ideogram_unpatchify_f32(
    .param .u64 out,
    .param .u64 tokens,
    .param .u32 gridH,
    .param .u32 gridW,
    .param .u32 inChannels,
    .param .u32 latentChannels,
    .param .u32 patchH,
    .param .u32 patchW,
    .param .u32 total
)
{
    .reg .pred %p<2>;
    .reg .u32 %r<30>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [tokens];
    ld.param.u32 %r1, [gridH];
    ld.param.u32 %r2, [gridW];
    ld.param.u32 %r3, [inChannels];
    ld.param.u32 %r4, [latentChannels];
    ld.param.u32 %r5, [patchH];
    ld.param.u32 %r6, [patchW];
    ld.param.u32 %r7, [total];

    mov.u32 %r8, %tid.x;
    mov.u32 %r9, %ctaid.x;
    mov.u32 %r10, %ntid.x;
    mad.lo.u32 %r11, %r9, %r10, %r8; // output linear CHW idx
    setp.ge.u32 %p1, %r11, %r7;
    @%p1 bra DONE;

    mul.lo.u32 %r12, %r1, %r5; // H
    mul.lo.u32 %r13, %r2, %r6; // W
    mul.lo.u32 %r14, %r12, %r13; // HW
    div.u32 %r15, %r11, %r14; // ch
    rem.u32 %r16, %r11, %r14; // yW+x
    div.u32 %r17, %r16, %r13; // y
    rem.u32 %r18, %r16, %r13; // x
    div.u32 %r19, %r17, %r5;  // grid row
    rem.u32 %r20, %r17, %r5;  // py
    div.u32 %r21, %r18, %r6;  // grid col
    rem.u32 %r22, %r18, %r6;  // px
    mul.lo.u32 %r23, %r19, %r2;
    add.u32 %r23, %r23, %r21; // token index
    mul.lo.u32 %r24, %r20, %r6;
    add.u32 %r24, %r24, %r22;
    mul.lo.u32 %r24, %r24, %r4;
    add.u32 %r24, %r24, %r15; // inner token channel
    mul.lo.u32 %r25, %r23, %r3;
    add.u32 %r25, %r25, %r24; // token offset

    mul.wide.u32 %rd3, %r25, 4;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    mul.wide.u32 %rd5, %r11, 4;
    add.u64 %rd6, %rd1, %rd5;
    st.global.f32 [%rd6], %f1;
DONE:
    ret;
}`
