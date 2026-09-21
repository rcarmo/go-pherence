package ptx

// IndependentBranchAttentionPTX computes one query row per branch over a
// shared immutable trunk plus one branch-local suffix. The active branch index
// is explicit, so no row can address another sibling's suffix.
const IndependentBranchAttentionPTX = `.version 7.0
.target sm_80
.address_size 64

.visible .entry gqa_attention_independent(
    .param .u64 param_q,
    .param .u64 param_trunk_k,
    .param .u64 param_trunk_v,
    .param .u64 param_suffix_k,
    .param .u64 param_suffix_v,
    .param .u64 param_active,
    .param .u64 param_out,
    .param .u32 param_batch,
    .param .u32 param_trunk_len,
    .param .u32 param_suffix_stride,
    .param .u32 param_seq_start,
    .param .u32 param_seq_len,
    .param .u32 param_n_heads,
    .param .u32 param_n_kv_heads,
    .param .u32 param_head_dim,
    .param .f32 param_scale
) {
    .reg .u32 %r<48>;
    .reg .u64 %rd<28>;
    .reg .f32 %f<16>;
    .reg .pred %p<7>;
    .shared .align 4 .f32 scores[2048];

    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ctaid.y; mov.u32 %r2, %tid.x;
    ld.param.u32 %r3, [param_batch]; ld.param.u32 %r4, [param_trunk_len];
    ld.param.u32 %r5, [param_suffix_stride]; ld.param.u32 %r6, [param_seq_start];
    ld.param.u32 %r7, [param_seq_len]; ld.param.u32 %r8, [param_n_heads];
    ld.param.u32 %r9, [param_n_kv_heads]; ld.param.u32 %r10, [param_head_dim];
    setp.ge.u32 %p0, %r0, %r8; setp.ge.u32 %p1, %r1, %r3; or.pred %p0, %p0, %p1; @%p0 bra done;

    ld.param.u64 %rd0, [param_q]; ld.param.u64 %rd1, [param_trunk_k]; ld.param.u64 %rd2, [param_trunk_v];
    ld.param.u64 %rd3, [param_suffix_k]; ld.param.u64 %rd4, [param_suffix_v]; ld.param.u64 %rd5, [param_active]; ld.param.u64 %rd6, [param_out];
    mul.wide.u32 %rd7, %r1, 4; add.u64 %rd8, %rd5, %rd7; ld.global.u32 %r11, [%rd8];
    div.u32 %r12, %r8, %r9; div.u32 %r13, %r0, %r12;
    mul.lo.u32 %r14, %r8, %r10; mad.lo.u32 %r15, %r1, %r14, 0; mad.lo.u32 %r15, %r0, %r10, %r15;
    mul.lo.u32 %r16, %r9, %r10; mul.lo.u32 %r17, %r11, %r5; mul.lo.u32 %r17, %r17, %r16;
    ld.param.f32 %f1, [param_scale]; mov.u32 %r18, %r2;

score_loop:
    setp.ge.u32 %p2, %r18, %r7; @%p2 bra score_done;
    add.u32 %r19, %r6, %r18; mul.lo.u32 %r20, %r19, %r16; mad.lo.u32 %r20, %r13, %r10, %r20;
    setp.lt.u32 %p3, %r19, %r4; @%p3 bra score_trunk;
    sub.u32 %r21, %r19, %r4; mad.lo.u32 %r20, %r21, %r16, %r17; mad.lo.u32 %r20, %r13, %r10, %r20;
    mov.u64 %rd9, %rd3; bra score_ptr;
score_trunk:
    mov.u64 %rd9, %rd1;
score_ptr:
    mov.f32 %f2, 0f00000000; mov.u32 %r22, 0;
dot_loop:
    setp.ge.u32 %p4, %r22, %r10; @%p4 bra dot_done;
    add.u32 %r23, %r15, %r22; mul.wide.u32 %rd10, %r23, 4; add.u64 %rd11, %rd0, %rd10; ld.global.f32 %f3, [%rd11];
    add.u32 %r24, %r20, %r22; mul.wide.u32 %rd12, %r24, 4; add.u64 %rd13, %rd9, %rd12; ld.global.f32 %f4, [%rd13];
    fma.rn.f32 %f2, %f3, %f4, %f2; add.u32 %r22, %r22, 1; bra dot_loop;
dot_done:
    mul.f32 %f2, %f2, %f1; mov.u64 %rd14, scores; mul.wide.u32 %rd15, %r18, 4; add.u64 %rd16, %rd14, %rd15; st.shared.f32 [%rd16], %f2;
    add.u32 %r18, %r18, 256; bra score_loop;
score_done:
    bar.sync 0; setp.ne.u32 %p2, %r2, 0; @%p2 bra softmax_done;
    mov.f32 %f5, 0fFF800000; mov.u32 %r18, 0;
max_loop:
    setp.ge.u32 %p4, %r18, %r7; @%p4 bra max_done; mul.wide.u32 %rd15, %r18, 4; add.u64 %rd16, %rd14, %rd15; ld.shared.f32 %f6, [%rd16]; max.f32 %f5, %f5, %f6; add.u32 %r18, %r18, 1; bra max_loop;
max_done:
    mov.f32 %f7, 0f00000000; mov.u32 %r18, 0;
exp_loop:
    setp.ge.u32 %p4, %r18, %r7; @%p4 bra exp_done; mul.wide.u32 %rd15, %r18, 4; add.u64 %rd16, %rd14, %rd15; ld.shared.f32 %f6, [%rd16]; sub.f32 %f6, %f6, %f5; mul.f32 %f6, %f6, 0f3FB8AA3B; ex2.approx.f32 %f6, %f6; add.f32 %f7, %f7, %f6; st.shared.f32 [%rd16], %f6; add.u32 %r18, %r18, 1; bra exp_loop;
exp_done:
    mov.f32 %f8, 0f3F800000; div.rn.f32 %f7, %f8, %f7; mov.u32 %r18, 0;
norm_loop:
    setp.ge.u32 %p4, %r18, %r7; @%p4 bra norm_done; mul.wide.u32 %rd15, %r18, 4; add.u64 %rd16, %rd14, %rd15; ld.shared.f32 %f6, [%rd16]; mul.f32 %f6, %f6, %f7; st.shared.f32 [%rd16], %f6; add.u32 %r18, %r18, 1; bra norm_loop;
norm_done:
softmax_done:
    bar.sync 0; mov.u32 %r22, %r2;
vsum_loop:
    setp.ge.u32 %p2, %r22, %r10; @%p2 bra done; mov.f32 %f9, 0f00000000; mov.u32 %r18, 0;
vsum_inner:
    setp.ge.u32 %p4, %r18, %r7; @%p4 bra vsum_done;
    mul.wide.u32 %rd15, %r18, 4; add.u64 %rd16, %rd14, %rd15; ld.shared.f32 %f10, [%rd16];
    add.u32 %r19, %r6, %r18; mul.lo.u32 %r20, %r19, %r16; mad.lo.u32 %r20, %r13, %r10, %r20;
    setp.lt.u32 %p5, %r19, %r4; @%p5 bra value_trunk;
    sub.u32 %r21, %r19, %r4; mad.lo.u32 %r20, %r21, %r16, %r17; mad.lo.u32 %r20, %r13, %r10, %r20; mov.u64 %rd17, %rd4; bra value_ptr;
value_trunk:
    mov.u64 %rd17, %rd2;
value_ptr:
    add.u32 %r20, %r20, %r22; mul.wide.u32 %rd18, %r20, 4; add.u64 %rd19, %rd17, %rd18; ld.global.f32 %f11, [%rd19]; fma.rn.f32 %f9, %f10, %f11, %f9;
    add.u32 %r18, %r18, 1; bra vsum_inner;
vsum_done:
    add.u32 %r23, %r15, %r22; mul.wide.u32 %rd20, %r23, 4; add.u64 %rd21, %rd6, %rd20; st.global.f32 [%rd21], %f9; add.u32 %r22, %r22, 256; bra vsum_loop;
done:
    ret;
}
`
