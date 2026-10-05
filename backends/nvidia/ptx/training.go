package ptx

// TrainingSiLUBackwardPTX uses a stable sigmoid and one owner per dX element.
// ex2.approx is an explicit device approximation requiring numerical tests.
const TrainingSiLUBackwardPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry training_silu_backward(.param .u64 X,.param .u64 DY,.param .u64 DX,.param .u32 N){
 .reg .u32 %r<5>; .reg .u64 %d<6>; .reg .f32 %f<12>; .reg .pred %p;
 mov.u32 %r0,%ctaid.x;mov.u32 %r1,%ntid.x;mov.u32 %r2,%tid.x;
 mad.lo.u32 %r3,%r0,%r1,%r2;ld.param.u32 %r4,[N];setp.ge.u32 %p,%r3,%r4;@%p bra done;
 ld.param.u64 %d0,[X];ld.param.u64 %d1,[DY];ld.param.u64 %d2,[DX];
 mul.wide.u32 %d3,%r3,4;add.u64 %d0,%d0,%d3;add.u64 %d1,%d1,%d3;add.u64 %d2,%d2,%d3;
 ld.global.f32 %f0,[%d0];ld.global.f32 %f1,[%d1];
 abs.f32 %f2,%f0;neg.f32 %f2,%f2;mul.rn.f32 %f2,%f2,0f3FB8AA3B;ex2.approx.f32 %f2,%f2;
 add.rn.f32 %f3,0f3F800000,%f2;div.rn.f32 %f4,0f3F800000,%f3;
 setp.lt.f32 %p,%f0,0f00000000;@%p div.rn.f32 %f4,%f2,%f3;
 sub.rn.f32 %f5,0f3F800000,%f4;mul.rn.f32 %f5,%f0,%f5;add.rn.f32 %f5,0f3F800000,%f5;
 mul.rn.f32 %f5,%f4,%f5;mul.rn.f32 %f5,%f1,%f5;st.global.f32 [%d2],%f5;
 done:ret;
}
`

// TrainingAdamWPTX performs one caller-admitted F32 AdamW/EMA update. Constants
// and buffers match backends/training.AdamWStep. Separate rn instructions keep
// the CPU expression's evaluation order; no fused multiply-add is requested.
const TrainingAdamWPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry training_adamw(
 .param .u64 P,.param .u64 G,.param .u64 M,.param .u64 V,.param .u64 E,
 .param .u32 N,.param .f32 LR,.param .f32 B1,.param .f32 B2,
 .param .f32 EPS,.param .f32 DECAY,.param .f32 C1,.param .f32 C2,
 .param .f32 EMA,.param .f32 SCALE
){
 .reg .u32 %r<5>; .reg .u64 %d<7>; .reg .f32 %f<25>; .reg .pred %p;
 mov.u32 %r0,%ctaid.x; mov.u32 %r1,%ntid.x; mov.u32 %r2,%tid.x;
 mad.lo.u32 %r3,%r0,%r1,%r2; ld.param.u32 %r4,[N];
 setp.ge.u32 %p,%r3,%r4; @%p bra done;
 ld.param.u64 %d0,[P]; ld.param.u64 %d1,[G]; ld.param.u64 %d2,[M];
 ld.param.u64 %d3,[V]; ld.param.u64 %d4,[E];
 mul.wide.u32 %d5,%r3,4;
 add.u64 %d0,%d0,%d5; add.u64 %d1,%d1,%d5; add.u64 %d2,%d2,%d5;
 add.u64 %d3,%d3,%d5; add.u64 %d4,%d4,%d5;
 ld.param.f32 %f0,[LR]; ld.param.f32 %f1,[B1]; ld.param.f32 %f2,[B2];
 ld.param.f32 %f3,[EPS]; ld.param.f32 %f4,[DECAY]; ld.param.f32 %f5,[C1];
 ld.param.f32 %f6,[C2]; ld.param.f32 %f7,[EMA]; ld.param.f32 %f8,[SCALE];
 ld.global.f32 %f9,[%d1]; mul.rn.f32 %f9,%f9,%f8;
 ld.global.f32 %f10,[%d2]; ld.global.f32 %f11,[%d3];
 mul.rn.f32 %f10,%f10,%f1; sub.rn.f32 %f12,0f3F800000,%f1;
 mul.rn.f32 %f12,%f12,%f9; add.rn.f32 %f10,%f10,%f12;
 mul.rn.f32 %f11,%f11,%f2; sub.rn.f32 %f12,0f3F800000,%f2;
 mul.rn.f32 %f12,%f12,%f9; mul.rn.f32 %f12,%f12,%f9;
 add.rn.f32 %f11,%f11,%f12;
 div.rn.f32 %f13,%f10,%f5; div.rn.f32 %f14,%f11,%f6;
 sqrt.rn.f32 %f14,%f14; add.rn.f32 %f14,%f14,%f3;
 mul.rn.f32 %f13,%f13,%f0; div.rn.f32 %f13,%f13,%f14;
 ld.global.f32 %f15,[%d0]; mul.rn.f32 %f16,%f0,%f4;
 sub.rn.f32 %f16,0f3F800000,%f16; mul.rn.f32 %f15,%f15,%f16;
 sub.rn.f32 %f15,%f15,%f13;
 st.global.f32 [%d0],%f15; st.global.f32 [%d2],%f10; st.global.f32 [%d3],%f11;
 setp.le.f32 %p,%f7,0f00000000; @%p bra done;
 ld.global.f32 %f17,[%d4]; mul.rn.f32 %f17,%f17,%f7;
 sub.rn.f32 %f18,0f3F800000,%f7; mul.rn.f32 %f18,%f18,%f15;
 add.rn.f32 %f17,%f17,%f18; st.global.f32 [%d4],%f17;
 done: ret;
}
`

// TrainingLinearBackwardPTX uses an independent owner per dX/dW/dBias output.
// mode=0/1/2; dW and dBias overwrite (branch accumulation is caller-managed).
const TrainingLinearBackwardPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry training_linear_backward(
 .param .u64 X,.param .u64 W,.param .u64 DY,.param .u64 OUT,
 .param .u32 M,.param .u32 K,.param .u32 N,.param .u32 MODE
){
 .reg .u32 %r<18>; .reg .u64 %d<10>; .reg .f32 %f<4>; .reg .pred %p<3>;
 mov.u32 %r0,%ctaid.x; mov.u32 %r1,%ntid.x; mov.u32 %r2,%tid.x;
 mad.lo.u32 %r3,%r0,%r1,%r2;
 ld.param.u32 %r4,[M]; ld.param.u32 %r5,[K]; ld.param.u32 %r6,[N];
 ld.param.u32 %r7,[MODE]; mov.u32 %r8,0; mov.f32 %f0,0f00000000;
 ld.param.u64 %d0,[X]; ld.param.u64 %d1,[W]; ld.param.u64 %d2,[DY]; ld.param.u64 %d3,[OUT];
 setp.eq.u32 %p0,%r7,0; @%p0 bra dx;
 setp.eq.u32 %p0,%r7,1; @%p0 bra dw;
 setp.ge.u32 %p0,%r3,%r6; @%p0 bra done;
 biasloop:
 setp.ge.u32 %p0,%r8,%r4; @%p0 bra store;
 mad.lo.u32 %r9,%r8,%r6,%r3; mul.wide.u32 %d4,%r9,4;
 add.u64 %d5,%d2,%d4; ld.global.f32 %f1,[%d5]; add.rn.f32 %f0,%f0,%f1;
 add.u32 %r8,%r8,1; bra biasloop;
 dx:
 mul.lo.u32 %r9,%r4,%r5; setp.ge.u32 %p0,%r3,%r9; @%p0 bra done;
 div.u32 %r10,%r3,%r5; rem.u32 %r11,%r3,%r5;
 dxloop:
 setp.ge.u32 %p0,%r8,%r6; @%p0 bra store;
 mad.lo.u32 %r12,%r10,%r6,%r8; mad.lo.u32 %r13,%r8,%r5,%r11;
 mul.wide.u32 %d4,%r12,4; mul.wide.u32 %d6,%r13,4;
 add.u64 %d5,%d2,%d4; add.u64 %d7,%d1,%d6;
 ld.global.f32 %f1,[%d5]; ld.global.f32 %f2,[%d7];
 mul.rn.f32 %f3,%f1,%f2; add.rn.f32 %f0,%f0,%f3;
 add.u32 %r8,%r8,1; bra dxloop;
 dw:
 mul.lo.u32 %r9,%r6,%r5; setp.ge.u32 %p0,%r3,%r9; @%p0 bra done;
 div.u32 %r10,%r3,%r5; rem.u32 %r11,%r3,%r5;
 dwloop:
 setp.ge.u32 %p0,%r8,%r4; @%p0 bra store;
 mad.lo.u32 %r12,%r8,%r6,%r10; mad.lo.u32 %r13,%r8,%r5,%r11;
 mul.wide.u32 %d4,%r12,4; mul.wide.u32 %d6,%r13,4;
 add.u64 %d5,%d2,%d4; add.u64 %d7,%d0,%d6;
 ld.global.f32 %f1,[%d5]; ld.global.f32 %f2,[%d7];
 mul.rn.f32 %f3,%f1,%f2; add.rn.f32 %f0,%f0,%f3;
 add.u32 %r8,%r8,1; bra dwloop;
 store:
 mul.wide.u32 %d8,%r3,4; add.u64 %d9,%d3,%d8; st.global.f32 [%d9],%f0;
 done: ret;
}
`
