package ptx

// AttentionFullPTX computes non-causal attention with one 128-thread block per
// (head, query). Scores for up to 2048 keys stay in shared memory. Layout is
// [sequence, heads, head_dim]; Whisper uses head_dim=64.
const AttentionFullPTX = `
.version 7.0
.target sm_70
.address_size 64

.visible .entry attention_full(
    .param .u64 out_ptr, .param .u64 q_ptr, .param .u64 k_ptr, .param .u64 v_ptr,
    .param .u32 seq_q, .param .u32 seq_kv, .param .u32 num_heads,
    .param .u32 head_dim, .param .f32 scale
) {
    .reg .pred %p<8>;
    .reg .u32 %r<24>;
    .reg .u64 %rd<12>;
    .reg .f32 %f<16>;
    .shared .align 4 .f32 q_shared[128];
    .shared .align 4 .f32 scores_shared[2048];
    .shared .align 4 .f32 reduce_shared[128];

    ld.param.u64 %rd0, [out_ptr];
    ld.param.u64 %rd1, [q_ptr];
    ld.param.u64 %rd2, [k_ptr];
    ld.param.u64 %rd3, [v_ptr];
    ld.param.u32 %r0, [seq_q];
    ld.param.u32 %r1, [seq_kv];
    ld.param.u32 %r2, [num_heads];
    ld.param.u32 %r3, [head_dim];
    ld.param.f32 %f0, [scale];
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ctaid.y;
    mov.u32 %r17, q_shared;
    mov.u32 %r18, scores_shared;
    mov.u32 %r19, reduce_shared;
    setp.ge.u32 %p0, %r5, %r2;
    @%p0 bra DONE;
    setp.ge.u32 %p0, %r6, %r0;
    @%p0 bra DONE;
    setp.gt.u32 %p0, %r3, 128;
    @%p0 bra DONE;
    setp.gt.u32 %p0, %r1, 2048;
    @%p0 bra DONE;

    // q_base = (query * heads + head) * head_dim
    mad.lo.u32 %r7, %r6, %r2, %r5;
    mul.lo.u32 %r7, %r7, %r3;
    setp.ge.u32 %p0, %r4, %r3;
    @%p0 bra Q_LOADED;
    add.u32 %r8, %r7, %r4;
    mul.wide.u32 %rd4, %r8, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];
    mul.lo.u32 %r15, %r4, 4;
    add.u32 %r20, %r17, %r15;
    st.shared.f32 [%r20], %f1;
Q_LOADED:
    bar.sync 0;

    // Threads independently compute strided Q.K scores.
    mov.f32 %f2, 0fFF800000;
    mov.u32 %r9, %r4;
SCORE_LOOP:
    setp.ge.u32 %p0, %r9, %r1;
    @%p0 bra SCORE_DONE;
    mad.lo.u32 %r10, %r9, %r2, %r5;
    mul.lo.u32 %r10, %r10, %r3;
    mov.f32 %f3, 0f00000000;
    mov.u32 %r11, 0;
DOT_LOOP:
    setp.ge.u32 %p0, %r11, %r3;
    @%p0 bra DOT_DONE;
    mul.lo.u32 %r15, %r11, 4;
    add.u32 %r20, %r17, %r15;
    ld.shared.f32 %f4, [%r20];
    add.u32 %r12, %r10, %r11;
    mul.wide.u32 %rd5, %r12, 4;
    add.u64 %rd6, %rd2, %rd5;
    ld.global.f32 %f5, [%rd6];
    fma.rn.f32 %f3, %f4, %f5, %f3;
    add.u32 %r11, %r11, 1;
    bra DOT_LOOP;
DOT_DONE:
    mul.rn.f32 %f3, %f3, %f0;
    mul.lo.u32 %r15, %r9, 4;
    add.u32 %r20, %r18, %r15;
    st.shared.f32 [%r20], %f3;
    max.f32 %f2, %f2, %f3;
    add.u32 %r9, %r9, 128;
    bra SCORE_LOOP;
SCORE_DONE:
    mul.lo.u32 %r15, %r4, 4;
    add.u32 %r20, %r19, %r15;
    st.shared.f32 [%r20], %f2;
    bar.sync 0;

    mov.u32 %r13, 64;
MAX_REDUCE:
    setp.eq.u32 %p0, %r13, 0;
    @%p0 bra MAX_READY;
    setp.ge.u32 %p0, %r4, %r13;
    @%p0 bra MAX_SKIP;
    mul.lo.u32 %r15, %r4, 4;
    add.u32 %r20, %r19, %r15;
    ld.shared.f32 %f2, [%r20];
    add.u32 %r14, %r4, %r13;
    mul.lo.u32 %r16, %r14, 4;
    add.u32 %r21, %r19, %r16;
    ld.shared.f32 %f6, [%r21];
    max.f32 %f2, %f2, %f6;
    st.shared.f32 [%r20], %f2;
MAX_SKIP:
    bar.sync 0;
    shr.u32 %r13, %r13, 1;
    bra MAX_REDUCE;
MAX_READY:
    ld.shared.f32 %f2, [%r19];

    // Exponentiate scores in place and reduce their sum.
    mov.f32 %f7, 0f00000000;
    mov.u32 %r9, %r4;
EXP_LOOP:
    setp.ge.u32 %p0, %r9, %r1;
    @%p0 bra EXP_DONE;
    mul.lo.u32 %r15, %r9, 4;
    add.u32 %r20, %r18, %r15;
    ld.shared.f32 %f3, [%r20];
    sub.rn.f32 %f3, %f3, %f2;
    mul.rn.f32 %f3, %f3, 0f3FB8AA3B;
    ex2.approx.ftz.f32 %f8, %f3;
    st.shared.f32 [%r20], %f8;
    add.rn.f32 %f7, %f7, %f8;
    add.u32 %r9, %r9, 128;
    bra EXP_LOOP;
EXP_DONE:
    mul.lo.u32 %r15, %r4, 4;
    add.u32 %r20, %r19, %r15;
    st.shared.f32 [%r20], %f7;
    bar.sync 0;
    mov.u32 %r13, 64;
SUM_REDUCE:
    setp.eq.u32 %p0, %r13, 0;
    @%p0 bra SUM_READY;
    setp.ge.u32 %p0, %r4, %r13;
    @%p0 bra SUM_SKIP;
    mul.lo.u32 %r15, %r4, 4;
    add.u32 %r20, %r19, %r15;
    ld.shared.f32 %f7, [%r20];
    add.u32 %r14, %r4, %r13;
    mul.lo.u32 %r16, %r14, 4;
    add.u32 %r21, %r19, %r16;
    ld.shared.f32 %f6, [%r21];
    add.rn.f32 %f7, %f7, %f6;
    st.shared.f32 [%r20], %f7;
SUM_SKIP:
    bar.sync 0;
    shr.u32 %r13, %r13, 1;
    bra SUM_REDUCE;
SUM_READY:
    ld.shared.f32 %f7, [%r19];

    // One thread per output dimension accumulates softmax(score) * V.
    setp.ge.u32 %p0, %r4, %r3;
    @%p0 bra DONE;
    mov.f32 %f9, 0f00000000;
    mov.u32 %r9, 0;
VALUE_LOOP:
    setp.ge.u32 %p0, %r9, %r1;
    @%p0 bra VALUE_DONE;
    mul.lo.u32 %r15, %r9, 4;
    add.u32 %r20, %r18, %r15;
    ld.shared.f32 %f8, [%r20];
    mad.lo.u32 %r10, %r9, %r2, %r5;
    mul.lo.u32 %r10, %r10, %r3;
    add.u32 %r10, %r10, %r4;
    mul.wide.u32 %rd5, %r10, 4;
    add.u64 %rd6, %rd3, %rd5;
    ld.global.f32 %f10, [%rd6];
    fma.rn.f32 %f9, %f8, %f10, %f9;
    add.u32 %r9, %r9, 1;
    bra VALUE_LOOP;
VALUE_DONE:
    div.rn.f32 %f9, %f9, %f7;
    add.u32 %r8, %r7, %r4;
    mul.wide.u32 %rd4, %r8, 4;
    add.u64 %rd5, %rd0, %rd4;
    st.global.f32 [%rd5], %f9;
DONE:
    ret;
}
`

// CrossAttentionPTX is a cross-attention kernel (Q from decoder, K/V from encoder).
// Layout is [seq, num_heads, head_dim]. Grid: (num_heads, dec_len, 1).
const CrossAttentionPTX = `
.version 7.0
.target sm_70
.address_size 64

.visible .entry cross_attention(
    .param .u64 out_ptr,
    .param .u64 q_ptr,
    .param .u64 k_ptr,
    .param .u64 v_ptr,
    .param .u32 dec_len,
    .param .u32 enc_len,
    .param .u32 num_heads,
    .param .u32 head_dim,
    .param .f32 scale
) {
    .reg .pred %p_tid, %p_q_oob, %p_h_oob, %p_done, %p_d_done;
    .reg .u32 %tidx, %h, %qpos, %seqQ, %seqKV, %heads, %hdim, %kv, %d;
    .reg .u32 %q_base, %kv_base, %idx, %out_idx;
    .reg .u64 %outp, %qp, %kp, %vp, %addr, %off;
    .reg .f32 %scalev, %dot, %score, %maxv, %sumw, %w, %acc, %qv, %kvv, %vv, %tmp;

    ld.param.u64 %outp, [out_ptr];
    ld.param.u64 %qp, [q_ptr];
    ld.param.u64 %kp, [k_ptr];
    ld.param.u64 %vp, [v_ptr];
    ld.param.u32 %seqQ, [dec_len];
    ld.param.u32 %seqKV, [enc_len];
    ld.param.u32 %heads, [num_heads];
    ld.param.u32 %hdim, [head_dim];
    ld.param.f32 %scalev, [scale];

    mov.u32 %tidx, %tid.x;
    setp.ne.u32 %p_tid, %tidx, 0;
    @%p_tid bra DONE;
    mov.u32 %h, %ctaid.x;
    mov.u32 %qpos, %ctaid.y;
    setp.ge.u32 %p_h_oob, %h, %heads;
    @%p_h_oob bra DONE;
    setp.ge.u32 %p_q_oob, %qpos, %seqQ;
    @%p_q_oob bra DONE;
    mul.lo.u32 %q_base, %qpos, %heads;
    add.u32 %q_base, %q_base, %h;
    mul.lo.u32 %q_base, %q_base, %hdim;

    mov.f32 %maxv, 0ff0000000;
    mov.u32 %kv, 0;
MAX_LOOP:
    setp.ge.u32 %p_done, %kv, %seqKV;
    @%p_done bra MAX_DONE;
    mul.lo.u32 %kv_base, %kv, %heads;
    add.u32 %kv_base, %kv_base, %h;
    mul.lo.u32 %kv_base, %kv_base, %hdim;
    mov.f32 %dot, 0f00000000;
    mov.u32 %d, 0;
DOT_MAX_LOOP:
    setp.ge.u32 %p_d_done, %d, %hdim;
    @%p_d_done bra DOT_MAX_DONE;
    add.u32 %idx, %q_base, %d;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %addr, %qp, %off;
    ld.global.f32 %qv, [%addr];
    add.u32 %idx, %kv_base, %d;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %addr, %kp, %off;
    ld.global.f32 %kvv, [%addr];
    fma.rn.f32 %dot, %qv, %kvv, %dot;
    add.u32 %d, %d, 1;
    bra DOT_MAX_LOOP;
DOT_MAX_DONE:
    mul.rn.f32 %score, %dot, %scalev;
    max.f32 %maxv, %maxv, %score;
    add.u32 %kv, %kv, 1;
    bra MAX_LOOP;
MAX_DONE:

    mov.u32 %d, 0;
OUT_D_LOOP:
    setp.ge.u32 %p_d_done, %d, %hdim;
    @%p_d_done bra DONE;
    mov.f32 %sumw, 0f00000000;
    mov.f32 %acc, 0f00000000;
    mov.u32 %kv, 0;
KV_OUT_LOOP:
    setp.ge.u32 %p_done, %kv, %seqKV;
    @%p_done bra STORE_D;
    mul.lo.u32 %kv_base, %kv, %heads;
    add.u32 %kv_base, %kv_base, %h;
    mul.lo.u32 %kv_base, %kv_base, %hdim;
    mov.f32 %dot, 0f00000000;
    mov.u32 %idx, 0;
DOT_OUT_LOOP:
    setp.ge.u32 %p_done, %idx, %hdim;
    @%p_done bra DOT_OUT_DONE;
    add.u32 %out_idx, %q_base, %idx;
    mul.wide.u32 %off, %out_idx, 4;
    add.u64 %addr, %qp, %off;
    ld.global.f32 %qv, [%addr];
    add.u32 %out_idx, %kv_base, %idx;
    mul.wide.u32 %off, %out_idx, 4;
    add.u64 %addr, %kp, %off;
    ld.global.f32 %kvv, [%addr];
    fma.rn.f32 %dot, %qv, %kvv, %dot;
    add.u32 %idx, %idx, 1;
    bra DOT_OUT_LOOP;
DOT_OUT_DONE:
    mul.rn.f32 %score, %dot, %scalev;
    sub.rn.f32 %score, %score, %maxv;
    mul.rn.f32 %tmp, %score, 0f3fb8aa3b;
    ex2.approx.ftz.f32 %w, %tmp;
    add.rn.f32 %sumw, %sumw, %w;
    add.u32 %out_idx, %kv_base, %d;
    mul.wide.u32 %off, %out_idx, 4;
    add.u64 %addr, %vp, %off;
    ld.global.f32 %vv, [%addr];
    fma.rn.f32 %acc, %w, %vv, %acc;
    add.u32 %kv, %kv, 1;
    bra KV_OUT_LOOP;
STORE_D:
    div.rn.f32 %acc, %acc, %sumw;
    add.u32 %out_idx, %q_base, %d;
    mul.wide.u32 %off, %out_idx, 4;
    add.u64 %addr, %outp, %off;
    st.global.f32 [%addr], %acc;
    add.u32 %d, %d, 1;
    bra OUT_D_LOOP;
DONE:
    ret;
}
`
