package ptx

// AttentionFullPTX is a non-causal (bidirectional) attention kernel for encoder.
// Grid: (num_heads, seq_q, 1), Block: (1..32, 1, 1). Thread 0 computes one
// (head, query_position) row. Layout is [seq, num_heads, head_dim].
// This is a correctness-first scalar GPU implementation of QKᵀ -> softmax -> V;
// future kernels can replace the body with tiled/flash attention without changing
// the entry contract.
const AttentionFullPTX = `
.version 7.0
.target sm_70
.address_size 64

.visible .entry attention_full(
    .param .u64 out_ptr,
    .param .u64 q_ptr,
    .param .u64 k_ptr,
    .param .u64 v_ptr,
    .param .u32 seq_q,
    .param .u32 seq_kv,
    .param .u32 num_heads,
    .param .u32 head_dim,
    .param .f32 scale
) {
    .reg .pred %p_tid, %p_q_oob, %p_h_oob, %p_done, %p_d_done;
    .reg .u32 %tid, %h, %qpos, %seqQ, %seqKV, %heads, %hdim, %kv, %d;
    .reg .u32 %q_base, %kv_base, %idx, %out_idx;
    .reg .u64 %outp, %qp, %kp, %vp, %addr, %off;
    .reg .f32 %scalev, %dot, %score, %maxv, %sumw, %w, %acc, %qv, %kvv, %vv, %tmp;

    ld.param.u64 %outp, [out_ptr];
    ld.param.u64 %qp, [q_ptr];
    ld.param.u64 %kp, [k_ptr];
    ld.param.u64 %vp, [v_ptr];
    ld.param.u32 %seqQ, [seq_q];
    ld.param.u32 %seqKV, [seq_kv];
    ld.param.u32 %heads, [num_heads];
    ld.param.u32 %hdim, [head_dim];
    ld.param.f32 %scalev, [scale];

    mov.u32 %tid, %tid.x;
    setp.ne.u32 %p_tid, %tid, 0;
    @%p_tid bra DONE;
    mov.u32 %h, %ctaid.x;
    mov.u32 %qpos, %ctaid.y;
    setp.ge.u32 %p_h_oob, %h, %heads;
    @%p_h_oob bra DONE;
    setp.ge.u32 %p_q_oob, %qpos, %seqQ;
    @%p_q_oob bra DONE;

    // q_base = (qpos * heads + h) * hdim
    mul.lo.u32 %q_base, %qpos, %heads;
    add.u32 %q_base, %q_base, %h;
    mul.lo.u32 %q_base, %q_base, %hdim;

    mov.f32 %maxv, 0ff0000000; // -inf
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
    mul.rn.f32 %tmp, %score, 0f3fb8aa3b; // log2(e)
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
    .reg .u32 %tid, %h, %qpos, %seqQ, %seqKV, %heads, %hdim, %kv, %d;
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

    mov.u32 %tid, %tid.x;
    setp.ne.u32 %p_tid, %tid, 0;
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
