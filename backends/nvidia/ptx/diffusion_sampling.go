package ptx

// DiffusionDenseSamplePTX samples dense DiffusionGemma canvas logits on-device.
// One block per canvas position. It mirrors llama.cpp's diffusion_dense_sample_kernel:
// argmax over logits*temp_inv, entropy of softmax(logits*temp_inv), and one
// multinomial draw using a pre-drawn uniform per position.
var DiffusionDenseSamplePTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry diffusion_dense_sample_f32(
    .param .u64 logits,
    .param .u64 u,
    .param .u64 argmax,
    .param .u64 entropy,
    .param .u64 sampled,
    .param .u32 n_vocab,
    .param .f32 inv_temp
) {
    .reg .u32 %r<38>;
    .reg .u64 %rd<22>;
    .reg .f32 %f<24>;
    .reg .pred %p;
    .shared .align 4 .f32 s_val[256];
    .shared .align 4 .f32 s_sum[256];
    .shared .align 4 .u32 s_idx[256];
    .shared .align 4 .u32 s_tok;

    mov.u32 %r0, %ctaid.x;    // row
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [n_vocab];
    ld.param.f32 %f20, [inv_temp];
    ld.param.u64 %rd0, [logits];
    ld.param.u64 %rd1, [u];
    ld.param.u64 %rd2, [argmax];
    ld.param.u64 %rd3, [entropy];
    ld.param.u64 %rd4, [sampled];

    // row_logits = logits + row*n_vocab
    mul.lo.u32 %r3, %r0, %r2;
    mul.wide.u32 %rd5, %r3, 4;
    add.u64 %rd6, %rd0, %rd5;

    // local max / argmax over scaled logits
    mov.u32 %r4, %r1;
    mov.f32 %f0, 0fFF800000; // -inf
    mov.u32 %r5, 0;
max_loop:
    setp.ge.u32 %p, %r4, %r2;
    @%p bra max_done;
    mul.wide.u32 %rd7, %r4, 4;
    add.u64 %rd8, %rd6, %rd7;
    ld.global.f32 %f1, [%rd8];
    mul.f32 %f1, %f1, %f20;
    setp.gt.f32 %p, %f1, %f0;
    @!%p bra max_skip;
    mov.f32 %f0, %f1;
    mov.u32 %r5, %r4;
max_skip:
    add.u32 %r4, %r4, 256;
    bra max_loop;
max_done:
    mov.u64 %rd9, s_val;
    mov.u64 %rd10, s_idx;
    mul.wide.u32 %rd11, %r1, 4;
    add.u64 %rd12, %rd9, %rd11;
    add.u64 %rd13, %rd10, %rd11;
    st.shared.f32 [%rd12], %f0;
    st.shared.u32 [%rd13], %r5;
    bar.sync 0;

    mov.u32 %r6, 128;
max_red_loop:
    setp.lt.u32 %p, %r6, 1;
    @%p bra max_red_done;
    setp.ge.u32 %p, %r1, %r6;
    @%p bra max_red_skip;
    add.u32 %r7, %r1, %r6;
    mul.wide.u32 %rd14, %r7, 4;
    add.u64 %rd15, %rd9, %rd14;
    add.u64 %rd16, %rd10, %rd14;
    ld.shared.f32 %f2, [%rd15];
    ld.shared.f32 %f3, [%rd12];
    setp.gt.f32 %p, %f2, %f3;
    @!%p bra max_red_skip;
    ld.shared.u32 %r8, [%rd16];
    st.shared.f32 [%rd12], %f2;
    st.shared.u32 [%rd13], %r8;
max_red_skip:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    bra max_red_loop;
max_red_done:
    ld.shared.f32 %f4, [s_val]; // max_l
    ld.shared.u32 %r9, [s_idx]; // amax

    // compute Z=sum exp(x-max), T=sum (x-max)*exp(x-max)
    mov.u32 %r4, %r1;
    mov.f32 %f5, 0f00000000;
    mov.f32 %f6, 0f00000000;
sum_loop:
    setp.ge.u32 %p, %r4, %r2;
    @%p bra sum_done;
    mul.wide.u32 %rd7, %r4, 4;
    add.u64 %rd8, %rd6, %rd7;
    ld.global.f32 %f1, [%rd8];
    mul.f32 %f1, %f1, %f20;
    sub.f32 %f7, %f1, %f4;
    ex2.approx.ftz.f32 %f8, %f7;      // exp2(d)
    mul.f32 %f9, %f7, 0f3fb8aa3b;     // d / ln(2)? placeholder not used
    // Use ex2(d/log(2)): approximate exp(d). 1/log(2)=1.4426950408889634
    mul.f32 %f10, %f7, 0f3fb8aa3b;
    ex2.approx.ftz.f32 %f8, %f10;
    add.f32 %f5, %f5, %f8;
    fma.rn.f32 %f6, %f7, %f8, %f6;
    add.u32 %r4, %r4, 256;
    bra sum_loop;
sum_done:
    mov.u64 %rd17, s_sum;
    add.u64 %rd18, %rd17, %rd11;
    st.shared.f32 [%rd18], %f5;
    st.shared.f32 [%rd12], %f6;
    bar.sync 0;
    mov.u32 %r6, 128;
sum_red_loop:
    setp.lt.u32 %p, %r6, 1;
    @%p bra sum_red_done;
    setp.ge.u32 %p, %r1, %r6;
    @%p bra sum_red_skip;
    add.u32 %r7, %r1, %r6;
    mul.wide.u32 %rd14, %r7, 4;
    add.u64 %rd15, %rd17, %rd14;
    add.u64 %rd16, %rd9, %rd14;
    ld.shared.f32 %f11, [%rd15];
    ld.shared.f32 %f12, [%rd18];
    add.f32 %f12, %f12, %f11;
    st.shared.f32 [%rd18], %f12;
    ld.shared.f32 %f13, [%rd16];
    ld.shared.f32 %f14, [%rd12];
    add.f32 %f14, %f14, %f13;
    st.shared.f32 [%rd12], %f14;
sum_red_skip:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    bra sum_red_loop;
sum_red_done:
    setp.ne.u32 %p, %r1, 0;
    @%p bra sample_prep;
    ld.shared.f32 %f15, [s_sum]; // z
    ld.shared.f32 %f16, [s_val]; // t
    lg2.approx.ftz.f32 %f17, %f15;
    mul.f32 %f17, %f17, 0f3f317218; // ln(2)
    div.rn.f32 %f18, %f16, %f15;
    sub.f32 %f19, %f17, %f18;
    mul.wide.u32 %rd7, %r0, 4;
    add.u64 %rd8, %rd2, %rd7;
    st.global.u32 [%rd8], %r9;
    add.u64 %rd8, %rd3, %rd7;
    st.global.f32 [%rd8], %f19;

sample_prep:
    bar.sync 0;
    // r = u[row] * z
    ld.shared.f32 %f15, [s_sum];
    mul.wide.u32 %rd7, %r0, 4;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.f32 %f21, [%rd8];
    mul.f32 %f21, %f21, %f15;

    // contiguous slice per thread; sum slice exp(d)
    add.u32 %r20, %r2, 255;
    shr.u32 %r20, %r20, 8; // chunk = ceil(vocab/256)
    mul.lo.u32 %r21, %r1, %r20; // beg
    add.u32 %r22, %r21, %r20;   // end
    min.u32 %r22, %r22, %r2;
    mov.u32 %r4, %r21;
    mov.f32 %f22, 0f00000000;
slice_loop:
    setp.ge.u32 %p, %r4, %r22;
    @%p bra slice_done;
    mul.wide.u32 %rd7, %r4, 4;
    add.u64 %rd8, %rd6, %rd7;
    ld.global.f32 %f1, [%rd8];
    mul.f32 %f1, %f1, %f20;
    sub.f32 %f7, %f1, %f4;
    mul.f32 %f10, %f7, 0f3fb8aa3b;
    ex2.approx.ftz.f32 %f8, %f10;
    add.f32 %f22, %f22, %f8;
    add.u32 %r4, %r4, 1;
    bra slice_loop;
slice_done:
    st.shared.f32 [%rd18], %f22;
    bar.sync 0;

    // thread 0 finds crossing slice
    @%p bra skip_dummy;
skip_dummy:
    setp.ne.u32 %p, %r1, 0;
    @%p bra scan_done;
    st.shared.u32 [s_tok], %r2;
    mov.f32 %f23, 0f00000000;
    mov.u32 %r23, 0;
scan_loop:
    setp.ge.u32 %p, %r23, 256;
    @%p bra scan_done;
    mul.wide.u32 %rd14, %r23, 4;
    add.u64 %rd15, %rd17, %rd14;
    ld.shared.f32 %f11, [%rd15];
    add.f32 %f12, %f23, %f11;
    setp.ge.f32 %p, %f12, %f21;
    @%p bra found_slice;
    mov.f32 %f23, %f12;
    add.u32 %r23, %r23, 1;
    bra scan_loop;
found_slice:
    st.shared.u32 [s_idx], %r23;
    st.shared.f32 [s_val], %f23;
scan_done:
    bar.sync 0;

sample_walk:
    ld.shared.u32 %r23, [s_idx];
    setp.ne.u32 %p, %r1, %r23;
    @%p bra sample_done;
    ld.shared.f32 %f23, [s_val];
    mov.u32 %r4, %r21;
walk_loop:
    setp.ge.u32 %p, %r4, %r22;
    @%p bra sample_done;
    mul.wide.u32 %rd7, %r4, 4;
    add.u64 %rd8, %rd6, %rd7;
    ld.global.f32 %f1, [%rd8];
    mul.f32 %f1, %f1, %f20;
    sub.f32 %f7, %f1, %f4;
    mul.f32 %f10, %f7, 0f3fb8aa3b;
    ex2.approx.ftz.f32 %f8, %f10;
    add.f32 %f23, %f23, %f8;
    setp.ge.f32 %p, %f23, %f21;
    @%p bra set_sample;
    add.u32 %r4, %r4, 1;
    bra walk_loop;
set_sample:
    st.shared.u32 [s_tok], %r4;
sample_done:
    bar.sync 0;
    setp.ne.u32 %p, %r1, 0;
    @%p bra done;
    ld.shared.u32 %r24, [s_tok];
    setp.lt.u32 %p, %r24, %r2;
    @%p bra store_sample;
    sub.u32 %r24, %r2, 1;
store_sample:
    mul.wide.u32 %rd7, %r0, 4;
    add.u64 %rd8, %rd4, %rd7;
    st.global.u32 [%rd8], %r24;
done:
    ret;
}
`

// DiffusionSoftmaxRowsPTX computes probs[row,v] = softmax(logits[row,*] * inv_temp)[v].
// One block per row, 256 threads per block, supports very wide vocab rows.
var DiffusionSoftmaxRowsPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry diffusion_softmax_rows_f32(
    .param .u64 logits,
    .param .u64 probs,
    .param .u32 n_vocab,
    .param .f32 inv_temp
) {
    .reg .u32 %r<20>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<18>;
    .reg .pred %p;
    .shared .align 4 .f32 s_val[256];
    .shared .align 4 .f32 s_sum[256];

    mov.u32 %r0, %ctaid.x; // row
    mov.u32 %r1, %tid.x;
    ld.param.u32 %r2, [n_vocab];
    ld.param.f32 %f15, [inv_temp];
    ld.param.u64 %rd0, [logits];
    ld.param.u64 %rd1, [probs];

    mul.lo.u32 %r3, %r0, %r2;
    mul.wide.u32 %rd2, %r3, 4;
    add.u64 %rd3, %rd0, %rd2; // row logits
    add.u64 %rd4, %rd1, %rd2; // row probs

    mov.u32 %r4, %r1;
    mov.f32 %f0, 0fFF800000;
max_loop:
    setp.ge.u32 %p, %r4, %r2;
    @%p bra max_done;
    mul.wide.u32 %rd5, %r4, 4;
    add.u64 %rd6, %rd3, %rd5;
    ld.global.f32 %f1, [%rd6];
    mul.f32 %f1, %f1, %f15;
    max.f32 %f0, %f0, %f1;
    add.u32 %r4, %r4, 256;
    bra max_loop;
max_done:
    mov.u64 %rd7, s_val;
    mul.wide.u32 %rd8, %r1, 4;
    add.u64 %rd9, %rd7, %rd8;
    st.shared.f32 [%rd9], %f0;
    bar.sync 0;
    mov.u32 %r5, 128;
max_red:
    setp.lt.u32 %p, %r5, 1;
    @%p bra max_red_done;
    setp.ge.u32 %p, %r1, %r5;
    @%p bra max_red_skip;
    add.u32 %r6, %r1, %r5;
    mul.wide.u32 %rd10, %r6, 4;
    add.u64 %rd11, %rd7, %rd10;
    ld.shared.f32 %f2, [%rd11];
    ld.shared.f32 %f3, [%rd9];
    max.f32 %f3, %f3, %f2;
    st.shared.f32 [%rd9], %f3;
max_red_skip:
    bar.sync 0;
    shr.u32 %r5, %r5, 1;
    bra max_red;
max_red_done:
    ld.shared.f32 %f4, [s_val]; // row max

    mov.u32 %r4, %r1;
    mov.f32 %f5, 0f00000000;
sum_loop:
    setp.ge.u32 %p, %r4, %r2;
    @%p bra sum_done;
    mul.wide.u32 %rd5, %r4, 4;
    add.u64 %rd6, %rd3, %rd5;
    ld.global.f32 %f1, [%rd6];
    mul.f32 %f1, %f1, %f15;
    sub.f32 %f6, %f1, %f4;
    mul.f32 %f7, %f6, 0f3fb8aa3b; // 1/log(2)
    ex2.approx.ftz.f32 %f8, %f7;
    add.f32 %f5, %f5, %f8;
    add.u32 %r4, %r4, 256;
    bra sum_loop;
sum_done:
    mov.u64 %rd12, s_sum;
    add.u64 %rd13, %rd12, %rd8;
    st.shared.f32 [%rd13], %f5;
    bar.sync 0;
    mov.u32 %r5, 128;
sum_red:
    setp.lt.u32 %p, %r5, 1;
    @%p bra sum_red_done;
    setp.ge.u32 %p, %r1, %r5;
    @%p bra sum_red_skip;
    add.u32 %r6, %r1, %r5;
    mul.wide.u32 %rd10, %r6, 4;
    add.u64 %rd11, %rd12, %rd10;
    ld.shared.f32 %f2, [%rd11];
    ld.shared.f32 %f3, [%rd13];
    add.f32 %f3, %f3, %f2;
    st.shared.f32 [%rd13], %f3;
sum_red_skip:
    bar.sync 0;
    shr.u32 %r5, %r5, 1;
    bra sum_red;
sum_red_done:
    ld.shared.f32 %f9, [s_sum];

    mov.u32 %r4, %r1;
write_loop:
    setp.ge.u32 %p, %r4, %r2;
    @%p bra done;
    mul.wide.u32 %rd5, %r4, 4;
    add.u64 %rd6, %rd3, %rd5;
    ld.global.f32 %f1, [%rd6];
    mul.f32 %f1, %f1, %f15;
    sub.f32 %f6, %f1, %f4;
    mul.f32 %f7, %f6, 0f3fb8aa3b;
    ex2.approx.ftz.f32 %f8, %f7;
    div.rn.f32 %f10, %f8, %f9;
    add.u64 %rd14, %rd4, %rd5;
    st.global.f32 [%rd14], %f10;
    add.u32 %r4, %r4, 256;
    bra write_loop;
done:
    ret;
}
`
