package ptx

// RoPEPartialSequencePTX applies partial RoPE to packed rows at consecutive
// absolute positions. It is the prompt-ubatch counterpart to rope_partial.
const RoPEPartialSequencePTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry rope_partial_sequence(
 .param .u64 X,.param .u64 CS,.param .u32 ROWS,.param .u32 POS0,
 .param .u32 HEADS,.param .u32 DIM,.param .u32 ROT) {
 .reg .u32 %r<24>; .reg .u64 %rd<12>; .reg .f32 %f<12>; .reg .pred %p;
 mov.u32 %r0,%ctaid.x; mov.u32 %r1,%ntid.x; mov.u32 %r2,%tid.x; mad.lo.u32 %r3,%r0,%r1,%r2;
 ld.param.u32 %r4,[ROWS]; ld.param.u32 %r5,[POS0]; ld.param.u32 %r6,[HEADS]; ld.param.u32 %r7,[DIM]; ld.param.u32 %r8,[ROT];
 mul.lo.u32 %r9,%r6,%r8; mul.lo.u32 %r10,%r4,%r9; setp.ge.u32 %p,%r3,%r10; @%p bra done;
 div.u32 %r11,%r3,%r9; rem.u32 %r12,%r3,%r9; div.u32 %r13,%r12,%r8; rem.u32 %r14,%r12,%r8;
 mul.lo.u32 %r15,%r6,%r7; mad.lo.u32 %r16,%r11,%r15,0; mad.lo.u32 %r16,%r13,%r7,%r16; add.u32 %r16,%r16,%r14; add.u32 %r17,%r16,%r8;
 ld.param.u64 %rd0,[X]; mul.wide.u32 %rd1,%r16,4; add.u64 %rd2,%rd0,%rd1; mul.wide.u32 %rd3,%r17,4; add.u64 %rd4,%rd0,%rd3; ld.global.f32 %f0,[%rd2]; ld.global.f32 %f1,[%rd4];
 add.u32 %r18,%r5,%r11; mul.lo.u32 %r19,%r18,%r8; add.u32 %r19,%r19,%r14; shl.b32 %r19,%r19,1; ld.param.u64 %rd5,[CS]; mul.wide.u32 %rd6,%r19,4; add.u64 %rd7,%rd5,%rd6; ld.global.f32 %f2,[%rd7]; ld.global.f32 %f3,[%rd7+4];
 neg.f32 %f4,%f3; mul.f32 %f5,%f1,%f4; fma.rn.f32 %f6,%f0,%f2,%f5; mul.f32 %f7,%f1,%f2; fma.rn.f32 %f8,%f0,%f3,%f7; st.global.f32 [%rd2],%f6; st.global.f32 [%rd4],%f8;
done: ret;
}`

// CausalBatchAttentionPTX computes GQA for packed query rows over a shared
// physical KV sequence. POS0 is the first query's absolute position; WINDOW=0
// means full causal attention.
const CausalBatchAttentionPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry gqa_attention_causal_batch(
 .param .u64 Q,.param .u64 K,.param .u64 V,.param .u64 O,
 .param .u32 ROWS,.param .u32 POS0,.param .u32 KV_LEN,.param .u32 WINDOW,
 .param .u32 HEADS,.param .u32 KVHEADS,.param .u32 DIM,.param .f32 SCALE) {
 .reg .u32 %r<44>; .reg .u64 %rd<24>; .reg .f32 %f<16>; .reg .pred %p<7>;
 .shared .align 4 .f32 scores[2048];
 mov.u32 %r0,%ctaid.x; mov.u32 %r1,%ctaid.y; mov.u32 %r2,%tid.x;
 ld.param.u32 %r3,[ROWS]; ld.param.u32 %r4,[POS0]; ld.param.u32 %r5,[KV_LEN]; ld.param.u32 %r6,[WINDOW]; ld.param.u32 %r7,[HEADS]; ld.param.u32 %r8,[KVHEADS]; ld.param.u32 %r9,[DIM];
 setp.ge.u32 %p0,%r0,%r7; setp.ge.u32 %p1,%r1,%r3; or.pred %p0,%p0,%p1; @%p0 bra done;
 ld.param.u64 %rd0,[Q]; ld.param.u64 %rd1,[K]; ld.param.u64 %rd2,[V]; ld.param.u64 %rd3,[O]; ld.param.f32 %f0,[SCALE];
 div.u32 %r10,%r7,%r8; div.u32 %r11,%r0,%r10; mul.lo.u32 %r12,%r7,%r9; mul.lo.u32 %r13,%r8,%r9; mad.lo.u32 %r14,%r1,%r12,0; mad.lo.u32 %r14,%r0,%r9,%r14;
 add.u32 %r15,%r4,%r1; add.u32 %r16,%r15,1; min.u32 %r16,%r16,%r5; mov.u32 %r17,0; setp.eq.u32 %p2,%r6,0; @%p2 bra start_done; setp.le.u32 %p3,%r16,%r6; @%p3 bra start_done; sub.u32 %r17,%r16,%r6;
start_done: sub.u32 %r18,%r16,%r17; mov.u32 %r19,%r2;
score_loop: setp.ge.u32 %p4,%r19,%r18; @%p4 bra score_done; add.u32 %r20,%r17,%r19; mad.lo.u32 %r21,%r20,%r13,0; mad.lo.u32 %r21,%r11,%r9,%r21; mov.f32 %f1,0f00000000; mov.u32 %r22,0;
dot_loop: setp.ge.u32 %p5,%r22,%r9; @%p5 bra dot_done; add.u32 %r23,%r14,%r22; mul.wide.u32 %rd4,%r23,4; add.u64 %rd5,%rd0,%rd4; ld.global.f32 %f2,[%rd5]; add.u32 %r24,%r21,%r22; mul.wide.u32 %rd6,%r24,4; add.u64 %rd7,%rd1,%rd6; ld.global.f32 %f3,[%rd7]; fma.rn.f32 %f1,%f2,%f3,%f1; add.u32 %r22,%r22,1; bra dot_loop;
dot_done: mul.f32 %f1,%f1,%f0; mov.u64 %rd8,scores; mul.wide.u32 %rd9,%r19,4; add.u64 %rd10,%rd8,%rd9; st.shared.f32 [%rd10],%f1; add.u32 %r19,%r19,256; bra score_loop;
score_done: bar.sync 0; setp.ne.u32 %p4,%r2,0; @%p4 bra soft_done; mov.f32 %f4,0fFF800000; mov.u32 %r19,0;
max_loop: setp.ge.u32 %p5,%r19,%r18; @%p5 bra max_done; mul.wide.u32 %rd9,%r19,4; add.u64 %rd10,%rd8,%rd9; ld.shared.f32 %f5,[%rd10]; max.f32 %f4,%f4,%f5; add.u32 %r19,%r19,1; bra max_loop;
max_done: mov.f32 %f6,0f00000000; mov.u32 %r19,0;
exp_loop: setp.ge.u32 %p5,%r19,%r18; @%p5 bra exp_done; mul.wide.u32 %rd9,%r19,4; add.u64 %rd10,%rd8,%rd9; ld.shared.f32 %f5,[%rd10]; sub.f32 %f5,%f5,%f4; mul.f32 %f5,%f5,0f3FB8AA3B; ex2.approx.f32 %f5,%f5; add.f32 %f6,%f6,%f5; st.shared.f32 [%rd10],%f5; add.u32 %r19,%r19,1; bra exp_loop;
exp_done: mov.f32 %f7,0f3F800000; div.rn.f32 %f6,%f7,%f6; mov.u32 %r19,0;
norm_loop: setp.ge.u32 %p5,%r19,%r18; @%p5 bra soft_done; mul.wide.u32 %rd9,%r19,4; add.u64 %rd10,%rd8,%rd9; ld.shared.f32 %f5,[%rd10]; mul.f32 %f5,%f5,%f6; st.shared.f32 [%rd10],%f5; add.u32 %r19,%r19,1; bra norm_loop;
soft_done: bar.sync 0; mov.u32 %r22,%r2;
v_loop: setp.ge.u32 %p4,%r22,%r9; @%p4 bra done; mov.f32 %f8,0f00000000; mov.u32 %r19,0;
v_inner: setp.ge.u32 %p5,%r19,%r18; @%p5 bra v_done; mul.wide.u32 %rd9,%r19,4; add.u64 %rd10,%rd8,%rd9; ld.shared.f32 %f9,[%rd10]; add.u32 %r20,%r17,%r19; mad.lo.u32 %r21,%r20,%r13,0; mad.lo.u32 %r21,%r11,%r9,%r21; add.u32 %r21,%r21,%r22; mul.wide.u32 %rd11,%r21,4; add.u64 %rd12,%rd2,%rd11; ld.global.f32 %f10,[%rd12]; fma.rn.f32 %f8,%f9,%f10,%f8; add.u32 %r19,%r19,1; bra v_inner;
v_done: add.u32 %r23,%r14,%r22; mul.wide.u32 %rd13,%r23,4; add.u64 %rd14,%rd3,%rd13; st.global.f32 [%rd14],%f8; add.u32 %r22,%r22,256; bra v_loop;
done: ret;
}`
