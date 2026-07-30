package ptx

// Split-KV decode attention candidate PTX kernels.
//
// The candidate keeps score materialization chunk-local: one partial block per
// (query head, KV chunk) computes a chunk max, exp sum, and weighted V vector
// for queryLen=1. A merge block per head then combines chunk partials via the
// standard stable softmax identity.
const AttentionSplitKVPartialPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gqa_attention_splitkv_partial(
    .param .u64 param_q,
    .param .u64 param_k,
    .param .u64 param_v,
    .param .u64 param_partial_max,
    .param .u64 param_partial_sum,
    .param .u64 param_partial_out,
    .param .u32 param_seqLen,
    .param .u32 param_nHeads,
    .param .u32 param_nKVHeads,
    .param .u32 param_headDim,
    .param .u32 param_nChunks,
    .param .f32 param_scale
) {
    .reg .pred %p<8>;
    .reg .u32 %r<48>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<24>;

    .shared .align 4 .f32 scores[256];
    .shared .align 4 .f32 chunk_max[1];
    .shared .align 4 .f32 chunk_sum[1];

    mov.u32 %r0, %ctaid.x;    // head
    mov.u32 %r1, %ctaid.y;    // chunk
    mov.u32 %r2, %tid.x;      // tid / token within chunk / dim lane

    ld.param.u32 %r3, [param_seqLen];
    ld.param.u32 %r4, [param_nHeads];
    ld.param.u32 %r5, [param_nKVHeads];
    ld.param.u32 %r6, [param_headDim];
    ld.param.u32 %r7, [param_nChunks];
    ld.param.f32 %f1, [param_scale];

    setp.ge.u32 %p0, %r0, %r4;
    @%p0 bra splitkv_partial_done;
    setp.ge.u32 %p1, %r1, %r7;
    @%p1 bra splitkv_partial_done;

    div.u32 %r8, %r4, %r5;    // heads per KV head
    div.u32 %r9, %r0, %r8;    // kv head index

    mul.lo.u32 %r10, %r1, 256; // chunkStart
    add.u32 %r11, %r10, %r2;   // seq index for this thread

    ld.param.u64 %rd0, [param_q];
    ld.param.u64 %rd1, [param_k];
    ld.param.u64 %rd2, [param_v];
    ld.param.u64 %rd3, [param_partial_max];
    ld.param.u64 %rd4, [param_partial_sum];
    ld.param.u64 %rd5, [param_partial_out];

    mul.lo.u32 %r12, %r0, %r6; // q base
    mul.lo.u32 %r13, %r5, %r6; // kvDim
    mul.lo.u32 %r14, %r9, %r6; // kv head base within row

    mov.f32 %f0, 0fFF800000;   // default score = -inf for tail threads
    setp.ge.u32 %p2, %r11, %r3;
    @%p2 bra splitkv_partial_store_score;

    mul.lo.u32 %r15, %r11, %r13;
    add.u32 %r15, %r15, %r14;  // k base for this token

    mov.f32 %f2, 0f00000000;
    mov.u32 %r16, 0;
splitkv_partial_dot_loop:
    setp.ge.u32 %p3, %r16, %r6;
    @%p3 bra splitkv_partial_dot_done;

    add.u32 %r17, %r12, %r16;
    mul.wide.u32 %rd6, %r17, 4;
    add.u64 %rd7, %rd0, %rd6;
    ld.global.f32 %f3, [%rd7];

    add.u32 %r18, %r15, %r16;
    mul.wide.u32 %rd8, %r18, 4;
    add.u64 %rd9, %rd1, %rd8;
    ld.global.f32 %f4, [%rd9];

    fma.rn.f32 %f2, %f3, %f4, %f2;
    add.u32 %r16, %r16, 1;
    bra splitkv_partial_dot_loop;
splitkv_partial_dot_done:
    mul.f32 %f0, %f2, %f1;

splitkv_partial_store_score:
    mov.u64 %rd10, scores;
    mul.wide.u32 %rd11, %r2, 4;
    add.u64 %rd12, %rd10, %rd11;
    st.shared.f32 [%rd12], %f0;
    bar.sync 0;

    setp.ne.u32 %p4, %r2, 0;
    @%p4 bra splitkv_partial_after_reduce;

    sub.u32 %r19, %r3, %r10;     // remaining tokens from chunkStart
    setp.gt.u32 %p5, %r19, 256;
    @%p5 mov.u32 %r19, 256;

    mov.f32 %f5, 0fFF800000;
    mov.u32 %r20, 0;
splitkv_partial_max_loop:
    setp.ge.u32 %p6, %r20, %r19;
    @%p6 bra splitkv_partial_max_done;
    mul.wide.u32 %rd13, %r20, 4;
    add.u64 %rd14, %rd10, %rd13;
    ld.shared.f32 %f6, [%rd14];
    max.f32 %f5, %f5, %f6;
    add.u32 %r20, %r20, 1;
    bra splitkv_partial_max_loop;
splitkv_partial_max_done:
    mov.u64 %rd15, chunk_max;
    st.shared.f32 [%rd15], %f5;

    mov.f32 %f7, 0f00000000;
    mov.u32 %r20, 0;
splitkv_partial_sum_loop:
    setp.ge.u32 %p6, %r20, %r19;
    @%p6 bra splitkv_partial_sum_done;
    mul.wide.u32 %rd13, %r20, 4;
    add.u64 %rd14, %rd10, %rd13;
    ld.shared.f32 %f6, [%rd14];
    sub.f32 %f6, %f6, %f5;
    mul.f32 %f6, %f6, 0f3FB8AA3B;
    ex2.approx.f32 %f6, %f6;
    add.f32 %f7, %f7, %f6;
    st.shared.f32 [%rd14], %f6;
    add.u32 %r20, %r20, 1;
    bra splitkv_partial_sum_loop;
splitkv_partial_sum_done:
    mov.u64 %rd16, chunk_sum;
    st.shared.f32 [%rd16], %f7;

splitkv_partial_after_reduce:
    bar.sync 0;

    mul.lo.u32 %r21, %r0, %r7;
    add.u32 %r21, %r21, %r1;     // flat (head,chunk) index

    setp.ne.u32 %p4, %r2, 0;
    @%p4 bra splitkv_partial_dims;
    mul.wide.u32 %rd17, %r21, 4;
    add.u64 %rd18, %rd3, %rd17;
    ld.shared.f32 %f8, [%rd15];
    st.global.f32 [%rd18], %f8;
    add.u64 %rd19, %rd4, %rd17;
    ld.shared.f32 %f9, [%rd16];
    st.global.f32 [%rd19], %f9;

splitkv_partial_dims:
    mov.u32 %r22, %r2;
splitkv_partial_dim_loop:
    setp.ge.u32 %p7, %r22, %r6;
    @%p7 bra splitkv_partial_done;

    sub.u32 %r19, %r3, %r10;
    setp.gt.u32 %p5, %r19, 256;
    @%p5 mov.u32 %r19, 256;

    mov.f32 %f10, 0f00000000;
    mov.u32 %r23, 0;
splitkv_partial_v_loop:
    setp.ge.u32 %p6, %r23, %r19;
    @%p6 bra splitkv_partial_v_done;

    mul.wide.u32 %rd13, %r23, 4;
    add.u64 %rd14, %rd10, %rd13;
    ld.shared.f32 %f11, [%rd14];

    add.u32 %r24, %r10, %r23;
    mul.lo.u32 %r25, %r24, %r13;
    add.u32 %r25, %r25, %r14;
    add.u32 %r25, %r25, %r22;
    mul.wide.u32 %rd20, %r25, 4;
    add.u64 %rd21, %rd2, %rd20;
    ld.global.f32 %f12, [%rd21];

    fma.rn.f32 %f10, %f11, %f12, %f10;
    add.u32 %r23, %r23, 1;
    bra splitkv_partial_v_loop;
splitkv_partial_v_done:

    mul.lo.u32 %r26, %r21, %r6;
    add.u32 %r26, %r26, %r22;
    mul.wide.u32 %rd22, %r26, 4;
    add.u64 %rd23, %rd5, %rd22;
    st.global.f32 [%rd23], %f10;

    add.u32 %r22, %r22, 256;
    bra splitkv_partial_dim_loop;

splitkv_partial_done:
    ret;
}
`

const AttentionSplitKVMergePTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gqa_attention_splitkv_merge(
    .param .u64 param_partial_max,
    .param .u64 param_partial_sum,
    .param .u64 param_partial_out,
    .param .u64 param_out,
    .param .u32 param_nHeads,
    .param .u32 param_headDim,
    .param .u32 param_nChunks
) {
    .reg .pred %p<6>;
    .reg .u32 %r<32>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<20>;

    .shared .align 4 .f32 global_max[1];
    .shared .align 4 .f32 global_sum[1];

    mov.u32 %r0, %ctaid.x;   // head
    mov.u32 %r1, %tid.x;     // tid / dim lane

    ld.param.u32 %r2, [param_nHeads];
    ld.param.u32 %r3, [param_headDim];
    ld.param.u32 %r4, [param_nChunks];

    setp.ge.u32 %p0, %r0, %r2;
    @%p0 bra splitkv_merge_done;

    ld.param.u64 %rd0, [param_partial_max];
    ld.param.u64 %rd1, [param_partial_sum];
    ld.param.u64 %rd2, [param_partial_out];
    ld.param.u64 %rd3, [param_out];

    mul.lo.u32 %r5, %r0, %r4; // flat head chunk base

    setp.ne.u32 %p1, %r1, 0;
    @%p1 bra splitkv_merge_after_reduce;

    mov.f32 %f0, 0fFF800000;
    mov.u32 %r6, 0;
splitkv_merge_max_loop:
    setp.ge.u32 %p2, %r6, %r4;
    @%p2 bra splitkv_merge_max_done;
    add.u32 %r7, %r5, %r6;
    mul.wide.u32 %rd4, %r7, 4;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.f32 %f1, [%rd5];
    max.f32 %f0, %f0, %f1;
    add.u32 %r6, %r6, 1;
    bra splitkv_merge_max_loop;
splitkv_merge_max_done:
    mov.u64 %rd6, global_max;
    st.shared.f32 [%rd6], %f0;

    mov.f32 %f2, 0f00000000;
    mov.u32 %r6, 0;
splitkv_merge_sum_loop:
    setp.ge.u32 %p2, %r6, %r4;
    @%p2 bra splitkv_merge_sum_done;
    add.u32 %r7, %r5, %r6;
    mul.wide.u32 %rd4, %r7, 4;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.f32 %f1, [%rd5];
    sub.f32 %f3, %f1, %f0;
    mul.f32 %f3, %f3, 0f3FB8AA3B;
    ex2.approx.f32 %f3, %f3;
    add.u64 %rd7, %rd1, %rd4;
    ld.global.f32 %f4, [%rd7];
    fma.rn.f32 %f2, %f3, %f4, %f2;
    add.u32 %r6, %r6, 1;
    bra splitkv_merge_sum_loop;
splitkv_merge_sum_done:
    mov.u64 %rd8, global_sum;
    st.shared.f32 [%rd8], %f2;

splitkv_merge_after_reduce:
    bar.sync 0;

    mov.u32 %r8, %r1;
splitkv_merge_dim_loop:
    setp.ge.u32 %p3, %r8, %r3;
    @%p3 bra splitkv_merge_done;

    ld.shared.f32 %f5, [%rd6];
    ld.shared.f32 %f6, [%rd8];
    mov.f32 %f7, 0f00000000;
    mov.u32 %r6, 0;
splitkv_merge_accum_loop:
    setp.ge.u32 %p4, %r6, %r4;
    @%p4 bra splitkv_merge_accum_done;

    add.u32 %r7, %r5, %r6;
    mul.wide.u32 %rd4, %r7, 4;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.f32 %f1, [%rd5];
    sub.f32 %f3, %f1, %f5;
    mul.f32 %f3, %f3, 0f3FB8AA3B;
    ex2.approx.f32 %f3, %f3;

    mul.lo.u32 %r9, %r7, %r3;
    add.u32 %r9, %r9, %r8;
    mul.wide.u32 %rd9, %r9, 4;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.f32 %f8, [%rd10];

    fma.rn.f32 %f7, %f3, %f8, %f7;
    add.u32 %r6, %r6, 1;
    bra splitkv_merge_accum_loop;
splitkv_merge_accum_done:

    div.rn.f32 %f7, %f7, %f6;
    mul.lo.u32 %r10, %r0, %r3;
    add.u32 %r10, %r10, %r8;
    mul.wide.u32 %rd11, %r10, 4;
    add.u64 %rd12, %rd3, %rd11;
    st.global.f32 [%rd12], %f7;

    add.u32 %r8, %r8, 256;
    bra splitkv_merge_dim_loop;

splitkv_merge_done:
    ret;
}
`
