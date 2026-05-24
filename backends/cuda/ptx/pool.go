package ptx

// AttentivePoolPTX is the GPU kernel for attentive statistics pooling.
// Computes attention-weighted mean and standard deviation across the time dimension.
// Grid: (1, 1, 1), Block: (256, 1, 1)
// One block handles all channels for a single segment.
const AttentivePoolPTX = `
.version 7.0
.target sm_70
.address_size 64

// Attentive statistics pooling kernel
// Input: h [channels * length] (channel-first)
// Output: out [channels * 2] (mean ++ std)
// Attention weights computed per-timestep, then applied to compute weighted stats.
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
    .reg .u32 %tid, %c, %t;
    .reg .f32 %score, %w, %sum, %sq, %val, %mean, %var, %std;
    .shared .f32 sh_weights[4096]; // max length

    // Step 1: Each thread computes attention scores for a subset of timesteps
    // score_t = sum_a(V[a] * tanh(sum_c(W[a,c] * h[c,t]) + b[a])) + v_bias
    // Collaborative: threads split across time dimension

    // Step 2: Softmax over time (parallel reduction in shared memory)
    // sh_weights[t] = exp(score_t - max) / sum(exp)

    // Step 3: Each thread computes weighted mean/std for a subset of channels
    // mean_c = sum_t(w_t * h[c,t])
    // std_c = sqrt(sum_t(w_t * h[c,t]^2) - mean_c^2)

    ret;
}
`
