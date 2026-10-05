package ptx

// TrainingAttentionBackwardPTX uses dense [H,S,S] probability/aux storage and
// [S,H,D] Q/K/V/dO. Causal/context invalid derivatives are explicitly zero.
// mode0=dP,1=scaled dScore,2=dQ,3=dK,4=dV; no atomics, one output owner.
const TrainingAttentionBackwardPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry training_attention_backward(
 .param .u64 Q,.param .u64 K,.param .u64 V,.param .u64 P,.param .u64 DY,.param .u64 AUX,.param .u64 OUT,
 .param .u32 S,.param .u32 H,.param .u32 D,.param .u32 CONTEXT,.param .u32 MODE,.param .f32 SCALE
){
 .reg .u32 %r<30>; .reg .u64 %d<15>; .reg .f32 %f<10>; .reg .pred %p<4>;
 mov.u32 %r0,%ctaid.x;mov.u32 %r1,%ntid.x;mov.u32 %r2,%tid.x;mad.lo.u32 %r3,%r0,%r1,%r2;
 ld.param.u32 %r4,[S];ld.param.u32 %r5,[H];ld.param.u32 %r6,[D];ld.param.u32 %r7,[CONTEXT];ld.param.u32 %r8,[MODE];ld.param.f32 %f9,[SCALE];
 ld.param.u64 %d0,[Q];ld.param.u64 %d1,[K];ld.param.u64 %d2,[V];ld.param.u64 %d3,[P];ld.param.u64 %d4,[DY];ld.param.u64 %d5,[AUX];ld.param.u64 %d6,[OUT];
 mul.lo.u32 %r9,%r5,%r6;mul.lo.u32 %r10,%r4,%r4;
 mov.f32 %f0,0f00000000;mov.u32 %r11,0;
 setp.eq.u32 %p0,%r8,0;@%p0 bra dp;
 setp.eq.u32 %p0,%r8,1;@%p0 bra ds;
 // QKV derivative output coordinates row/head/dim.
 mul.lo.u32 %r12,%r4,%r9;setp.ge.u32 %p0,%r3,%r12;@%p0 bra done;
 div.u32 %r13,%r3,%r9;rem.u32 %r14,%r3,%r9;div.u32 %r15,%r14,%r6;rem.u32 %r16,%r14,%r6;
 setp.eq.u32 %p0,%r8,2;@%p0 bra dq;
 // dK/dV: fixed key row, valid query>=key; context query-key<C.
 mov.u32 %r11,%r13;
 kvloop:
 setp.ge.u32 %p0,%r11,%r4;@%p0 bra store;
 sub.u32 %r17,%r11,%r13;setp.eq.u32 %p1,%r7,0;setp.ge.u32 %p2,%r17,%r7;and.pred %p3,%p2,!%p1;@%p3 bra store;
 mad.lo.u32 %r18,%r15,%r4,%r11;mad.lo.u32 %r18,%r18,%r4,%r13;
 mul.wide.u32 %d7,%r18,4;
 mad.lo.u32 %r19,%r11,%r9,%r14;mul.wide.u32 %d8,%r19,4;
 setp.eq.u32 %p0,%r8,3;@%p0 bra dkload;
 add.u64 %d9,%d3,%d7;add.u64 %d10,%d4,%d8;bra kvload;
 dkload:add.u64 %d9,%d5,%d7;add.u64 %d10,%d0,%d8;
 kvload:ld.global.f32 %f1,[%d9];ld.global.f32 %f2,[%d10];mul.rn.f32 %f3,%f1,%f2;add.rn.f32 %f0,%f0,%f3;
 add.u32 %r11,%r11,1;bra kvloop;
 dq:
 // fixed query: key starts at max(0,query-context+1).
 setp.eq.u32 %p0,%r7,0;@%p0 bra dqloop;
 add.u32 %r17,%r13,1;setp.gt.u32 %p0,%r17,%r7;@%p0 sub.u32 %r11,%r17,%r7;
 dqloop:
 setp.gt.u32 %p0,%r11,%r13;@%p0 bra store;
 mad.lo.u32 %r18,%r15,%r4,%r13;mad.lo.u32 %r18,%r18,%r4,%r11;
 mul.wide.u32 %d7,%r18,4;add.u64 %d9,%d5,%d7;ld.global.f32 %f1,[%d9];
 mad.lo.u32 %r19,%r11,%r9,%r14;mul.wide.u32 %d8,%r19,4;add.u64 %d10,%d1,%d8;ld.global.f32 %f2,[%d10];
 mul.rn.f32 %f3,%f1,%f2;add.rn.f32 %f0,%f0,%f3;add.u32 %r11,%r11,1;bra dqloop;
 dp:
 mul.lo.u32 %r12,%r5,%r10;setp.ge.u32 %p0,%r3,%r12;@%p0 bra done;
 rem.u32 %r13,%r3,%r4;div.u32 %r14,%r3,%r4;rem.u32 %r15,%r14,%r4;div.u32 %r16,%r3,%r10;
 // key=r13,query=r15,head=r16.
 setp.gt.u32 %p0,%r13,%r15;@%p0 bra store;
 sub.u32 %r17,%r15,%r13;setp.eq.u32 %p1,%r7,0;setp.ge.u32 %p2,%r17,%r7;and.pred %p3,%p2,!%p1;@%p3 bra store;
 mad.lo.u32 %r18,%r15,%r9,0;mad.lo.u32 %r18,%r16,%r6,%r18;
 mad.lo.u32 %r19,%r13,%r9,0;mad.lo.u32 %r19,%r16,%r6,%r19;
 dploop:
 setp.ge.u32 %p0,%r11,%r6;@%p0 bra store;
 add.u32 %r20,%r18,%r11;mul.wide.u32 %d7,%r20,4;add.u64 %d9,%d4,%d7;
 add.u32 %r20,%r19,%r11;mul.wide.u32 %d8,%r20,4;add.u64 %d10,%d2,%d8;
 ld.global.f32 %f1,[%d9];ld.global.f32 %f2,[%d10];mul.rn.f32 %f3,%f1,%f2;add.rn.f32 %f0,%f0,%f3;
 add.u32 %r11,%r11,1;bra dploop;
 ds:
 // One invocation owns full query/head row in dense dScore.
 mul.lo.u32 %r12,%r5,%r4;setp.ge.u32 %p0,%r3,%r12;@%p0 bra done;
 rem.u32 %r13,%r3,%r4;mul.lo.u32 %r14,%r3,%r4;
 dsreduce:
 setp.gt.u32 %p0,%r11,%r13;@%p0 bra dsapply;
 sub.u32 %r17,%r13,%r11;setp.eq.u32 %p1,%r7,0;setp.ge.u32 %p2,%r17,%r7;and.pred %p3,%p2,!%p1;@%p3 bra dsnext;
 add.u32 %r18,%r14,%r11;mul.wide.u32 %d7,%r18,4;add.u64 %d9,%d3,%d7;add.u64 %d10,%d5,%d7;
 ld.global.f32 %f1,[%d9];ld.global.f32 %f2,[%d10];mul.rn.f32 %f3,%f1,%f2;add.rn.f32 %f0,%f0,%f3;
 dsnext:add.u32 %r11,%r11,1;bra dsreduce;
 dsapply:mov.u32 %r11,0;
 dsloop:
 setp.ge.u32 %p0,%r11,%r4;@%p0 bra done;
 mov.f32 %f4,0f00000000;setp.gt.u32 %p0,%r11,%r13;@%p0 bra dsstore;
 sub.u32 %r17,%r13,%r11;setp.eq.u32 %p1,%r7,0;setp.ge.u32 %p2,%r17,%r7;and.pred %p3,%p2,!%p1;@%p3 bra dsstore;
 add.u32 %r18,%r14,%r11;mul.wide.u32 %d7,%r18,4;add.u64 %d9,%d3,%d7;add.u64 %d10,%d5,%d7;
 ld.global.f32 %f1,[%d9];ld.global.f32 %f2,[%d10];sub.rn.f32 %f2,%f2,%f0;mul.rn.f32 %f4,%f1,%f2;mul.rn.f32 %f4,%f4,%f9;
 dsstore:
 add.u32 %r18,%r14,%r11;mul.wide.u32 %d7,%r18,4;add.u64 %d9,%d6,%d7;st.global.f32 [%d9],%f4;
 add.u32 %r11,%r11,1;bra dsloop;
 store:mul.wide.u32 %d7,%r3,4;add.u64 %d9,%d6,%d7;st.global.f32 [%d9],%f0;
 done:ret;
}
`
