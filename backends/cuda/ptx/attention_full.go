package ptx

// AttentionFullPTX is a non-causal (bidirectional) attention kernel for encoder.
// Same as causal attention but without the triangular mask.
// Grid: (num_heads, seq_q, 1), Block: (32, 1, 1) or (seq_kv rounded, 1, 1)
const AttentionFullPTX = `
.version 7.0
.target sm_70
.address_size 64

// Non-causal multi-head attention
// Each thread-block handles one (head, query_position) pair.
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
    // TODO: implement flash-attention style tiled non-causal attention
    ret;
}
`

// CrossAttentionPTX is a cross-attention kernel (Q from decoder, K/V from encoder).
// Functionally identical to AttentionFull but may have different seq lengths for Q vs K/V.
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
    // TODO: implement tiled cross-attention
    ret;
}
`
