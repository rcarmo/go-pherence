package ptx

// TrainingSiLUDualBackwardPTX evaluates exact analytic reverse-over-forward
// SiLU derivative expressions with the documented device exp approximation.
const TrainingSiLUDualBackwardPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry training_silu_dual_backward(.param .u64 X,.param .u64 XT,.param .u64 GV,.param .u64 GT,.param .u64 DX,.param .u64 DT,.param .u32 N){
 .reg .u32 %r<5>; .reg .u64 %d<9>; .reg .f32 %f<17>; .reg .pred %p;
 mov.u32 %r0,%ctaid.x;mov.u32 %r1,%ntid.x;mov.u32 %r2,%tid.x;mad.lo.u32 %r3,%r0,%r1,%r2;
 ld.param.u32 %r4,[N];setp.ge.u32 %p,%r3,%r4;@%p bra done;
 ld.param.u64 %d0,[X];ld.param.u64 %d1,[XT];ld.param.u64 %d2,[GV];ld.param.u64 %d3,[GT];ld.param.u64 %d4,[DX];ld.param.u64 %d5,[DT];
 mul.wide.u32 %d6,%r3,4;add.u64 %d0,%d0,%d6;add.u64 %d1,%d1,%d6;add.u64 %d2,%d2,%d6;add.u64 %d3,%d3,%d6;add.u64 %d4,%d4,%d6;add.u64 %d5,%d5,%d6;
 ld.global.f32 %f0,[%d0];ld.global.f32 %f1,[%d1];ld.global.f32 %f2,[%d2];ld.global.f32 %f3,[%d3];
 abs.f32 %f4,%f0;neg.f32 %f4,%f4;mul.rn.f32 %f4,%f4,0f3FB8AA3B;ex2.approx.f32 %f4,%f4;
 add.rn.f32 %f5,0f3F800000,%f4;div.rn.f32 %f6,0f3F800000,%f5;setp.lt.f32 %p,%f0,0f00000000;@%p div.rn.f32 %f6,%f4,%f5;
 sub.rn.f32 %f7,0f3F800000,%f6;mul.rn.f32 %f8,%f0,%f7;add.rn.f32 %f8,0f3F800000,%f8;mul.rn.f32 %f8,%f6,%f8;
 mul.rn.f32 %f9,%f6,%f7;mul.rn.f32 %f10,0f40000000,%f6;sub.rn.f32 %f10,0f3F800000,%f10;mul.rn.f32 %f10,%f0,%f10;add.rn.f32 %f10,0f40000000,%f10;mul.rn.f32 %f9,%f9,%f10;
 mul.rn.f32 %f11,%f2,%f8;mul.rn.f32 %f12,%f3,%f9;mul.rn.f32 %f12,%f12,%f1;add.rn.f32 %f11,%f11,%f12;st.global.f32 [%d4],%f11;
 mul.rn.f32 %f13,%f3,%f8;st.global.f32 [%d5],%f13;
 done:ret;
}
`
