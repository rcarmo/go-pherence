package ptx

// AttentivePoolPTX is the GPU kernel for attentive statistics pooling.
// Computes attention-weighted mean and standard deviation across the time dimension.
// Grid: (channels, 1, 1), Block: (1..256, 1, 1). Thread 0 computes one channel.
// Layouts: h[channels, length], attn_w[attn_dim, channels], out[2*channels].
// This is a correctness-first scalar GPU implementation. It recomputes the
// attention softmax per output channel to keep the entry self-contained; a future
// optimized kernel can split score computation/reduction without changing the
// contract.
const AttentivePoolPTX = `
.version 7.0
.target sm_70
.address_size 64

.visible .entry attentive_stat_pool(
    .param .u64 out_ptr,
    .param .u64 h_ptr,
    .param .u64 attn_w_ptr,
    .param .u64 attn_b_ptr,
    .param .u64 v_ptr,
    .param .f32 v_bias,
    .param .u32 channels,
    .param .u32 length,
    .param .u32 attn_dim
) {
    .reg .pred %p_tid, %p_c_oob, %p_t_done, %p_a_done, %p_ch_done;
    .reg .u32 %tidx, %c, %t, %a, %ch, %channels_u, %length_u, %attn_dim_u;
    .reg .u32 %idx, %hidx, %widx;
    .reg .u64 %outp, %hp, %wp, %bp, %vp, %addr, %off;
    .reg .f32 %v_bias_r, %score, %inner, %w_attn, %b_attn, %v_attn;
    .reg .f32 %hval, %maxv, %sumw, %weight, %mean, %second, %var, %std;
    .reg .f32 %neg2, %tmp, %expv, %num, %den, %tanhv;

    ld.param.u64 %outp, [out_ptr];
    ld.param.u64 %hp, [h_ptr];
    ld.param.u64 %wp, [attn_w_ptr];
    ld.param.u64 %bp, [attn_b_ptr];
    ld.param.u64 %vp, [v_ptr];
    ld.param.f32 %v_bias_r, [v_bias];
    ld.param.u32 %channels_u, [channels];
    ld.param.u32 %length_u, [length];
    ld.param.u32 %attn_dim_u, [attn_dim];

    mov.u32 %tidx, %tid.x;
    setp.ne.u32 %p_tid, %tidx, 0;
    @%p_tid bra DONE;
    mov.u32 %c, %ctaid.x;
    setp.ge.u32 %p_c_oob, %c, %channels_u;
    @%p_c_oob bra DONE;
    mov.f32 %neg2, 0fc0000000; // -2.0f

    // Pass 1: max attention score over time.
    mov.f32 %maxv, 0ff0000000; // -inf
    mov.u32 %t, 0;
MAX_T_LOOP:
    setp.ge.u32 %p_t_done, %t, %length_u;
    @%p_t_done bra MAX_DONE;
    mov.f32 %score, %v_bias_r;
    mov.u32 %a, 0;
MAX_A_LOOP:
    setp.ge.u32 %p_a_done, %a, %attn_dim_u;
    @%p_a_done bra MAX_A_DONE;
    mov.f32 %inner, 0f00000000;
    mov.u32 %ch, 0;
MAX_CH_LOOP:
    setp.ge.u32 %p_ch_done, %ch, %channels_u;
    @%p_ch_done bra MAX_CH_DONE;
    mad.lo.u32 %hidx, %ch, %length_u, %t;
    mul.wide.u32 %off, %hidx, 4;
    add.u64 %addr, %hp, %off;
    ld.global.f32 %hval, [%addr];
    mad.lo.u32 %widx, %a, %channels_u, %ch;
    mul.wide.u32 %off, %widx, 4;
    add.u64 %addr, %wp, %off;
    ld.global.f32 %w_attn, [%addr];
    fma.rn.f32 %inner, %w_attn, %hval, %inner;
    add.u32 %ch, %ch, 1;
    bra MAX_CH_LOOP;
MAX_CH_DONE:
    mul.wide.u32 %off, %a, 4;
    add.u64 %addr, %bp, %off;
    ld.global.f32 %b_attn, [%addr];
    add.rn.f32 %inner, %inner, %b_attn;
    // tanh(inner) ~= (1 - exp(-2x)) / (1 + exp(-2x))
    mul.rn.f32 %tmp, %inner, %neg2;
    mul.rn.f32 %tmp, %tmp, 0f3fb8aa3b; // log2(e)
    ex2.approx.ftz.f32 %expv, %tmp;
    sub.rn.f32 %num, 1.0, %expv;
    add.rn.f32 %den, 1.0, %expv;
    div.rn.f32 %tanhv, %num, %den;
    mul.wide.u32 %off, %a, 4;
    add.u64 %addr, %vp, %off;
    ld.global.f32 %v_attn, [%addr];
    fma.rn.f32 %score, %v_attn, %tanhv, %score;
    add.u32 %a, %a, 1;
    bra MAX_A_LOOP;
MAX_A_DONE:
    max.f32 %maxv, %maxv, %score;
    add.u32 %t, %t, 1;
    bra MAX_T_LOOP;
MAX_DONE:

    // Pass 2: softmax denominator, weighted mean and weighted second moment.
    mov.f32 %sumw, 0f00000000;
    mov.f32 %mean, 0f00000000;
    mov.f32 %second, 0f00000000;
    mov.u32 %t, 0;
SUM_T_LOOP:
    setp.ge.u32 %p_t_done, %t, %length_u;
    @%p_t_done bra STORE;
    mov.f32 %score, %v_bias_r;
    mov.u32 %a, 0;
SUM_A_LOOP:
    setp.ge.u32 %p_a_done, %a, %attn_dim_u;
    @%p_a_done bra SUM_A_DONE;
    mov.f32 %inner, 0f00000000;
    mov.u32 %ch, 0;
SUM_CH_LOOP:
    setp.ge.u32 %p_ch_done, %ch, %channels_u;
    @%p_ch_done bra SUM_CH_DONE;
    mad.lo.u32 %hidx, %ch, %length_u, %t;
    mul.wide.u32 %off, %hidx, 4;
    add.u64 %addr, %hp, %off;
    ld.global.f32 %hval, [%addr];
    mad.lo.u32 %widx, %a, %channels_u, %ch;
    mul.wide.u32 %off, %widx, 4;
    add.u64 %addr, %wp, %off;
    ld.global.f32 %w_attn, [%addr];
    fma.rn.f32 %inner, %w_attn, %hval, %inner;
    add.u32 %ch, %ch, 1;
    bra SUM_CH_LOOP;
SUM_CH_DONE:
    mul.wide.u32 %off, %a, 4;
    add.u64 %addr, %bp, %off;
    ld.global.f32 %b_attn, [%addr];
    add.rn.f32 %inner, %inner, %b_attn;
    mul.rn.f32 %tmp, %inner, %neg2;
    mul.rn.f32 %tmp, %tmp, 0f3fb8aa3b;
    ex2.approx.ftz.f32 %expv, %tmp;
    sub.rn.f32 %num, 1.0, %expv;
    add.rn.f32 %den, 1.0, %expv;
    div.rn.f32 %tanhv, %num, %den;
    mul.wide.u32 %off, %a, 4;
    add.u64 %addr, %vp, %off;
    ld.global.f32 %v_attn, [%addr];
    fma.rn.f32 %score, %v_attn, %tanhv, %score;
    add.u32 %a, %a, 1;
    bra SUM_A_LOOP;
SUM_A_DONE:
    sub.rn.f32 %tmp, %score, %maxv;
    mul.rn.f32 %tmp, %tmp, 0f3fb8aa3b;
    ex2.approx.ftz.f32 %weight, %tmp;
    add.rn.f32 %sumw, %sumw, %weight;
    mad.lo.u32 %hidx, %c, %length_u, %t;
    mul.wide.u32 %off, %hidx, 4;
    add.u64 %addr, %hp, %off;
    ld.global.f32 %hval, [%addr];
    fma.rn.f32 %mean, %weight, %hval, %mean;
    mul.rn.f32 %tmp, %hval, %hval;
    fma.rn.f32 %second, %weight, %tmp, %second;
    add.u32 %t, %t, 1;
    bra SUM_T_LOOP;

STORE:
    div.rn.f32 %mean, %mean, %sumw;
    div.rn.f32 %second, %second, %sumw;
    mul.rn.f32 %tmp, %mean, %mean;
    sub.rn.f32 %var, %second, %tmp;
    max.f32 %var, %var, 0f00000000;
    sqrt.rn.f32 %std, %var;
    mul.wide.u32 %off, %c, 4;
    add.u64 %addr, %outp, %off;
    st.global.f32 [%addr], %mean;
    add.u32 %idx, %channels_u, %c;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %addr, %outp, %off;
    st.global.f32 [%addr], %std;
DONE:
    ret;
}
`
