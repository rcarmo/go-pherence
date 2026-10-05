package ptx

// TrainingLayerNormBackwardPTX consumes pre-affine normalized Xhat and cached
// invstd. mode0 owns a complete row; modes1/2 own dWeight/dBias columns.
const TrainingLayerNormBackwardPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry training_layernorm_backward(
 .param .u64 X,.param .u64 INV,.param .u64 W,.param .u64 DY,.param .u64 OUT,
 .param .u32 M,.param .u32 N,.param .u32 MODE
){
 .reg .u32 %r<15>; .reg .u64 %d<14>; .reg .f32 %f<14>; .reg .pred %p<2>;
 mov.u32 %r0,%ctaid.x;mov.u32 %r1,%ntid.x;mov.u32 %r2,%tid.x;
 mad.lo.u32 %r3,%r0,%r1,%r2;
 ld.param.u32 %r4,[M];ld.param.u32 %r5,[N];ld.param.u32 %r6,[MODE];
 ld.param.u64 %d0,[X];ld.param.u64 %d1,[INV];ld.param.u64 %d2,[W];ld.param.u64 %d3,[DY];ld.param.u64 %d4,[OUT];
 mov.f32 %f0,0f00000000;mov.f32 %f1,0f00000000;mov.u32 %r7,0;
 setp.eq.u32 %p0,%r6,0;@%p0 bra dx;
 setp.ge.u32 %p0,%r3,%r5;@%p0 bra done;
 paramloop:
 setp.ge.u32 %p0,%r7,%r4;@%p0 bra paramstore;
 mad.lo.u32 %r8,%r7,%r5,%r3;mul.wide.u32 %d5,%r8,4;add.u64 %d6,%d3,%d5;ld.global.f32 %f2,[%d6];
 setp.eq.u32 %p0,%r6,1;@!%p0 bra paramadd;
 add.u64 %d6,%d0,%d5;ld.global.f32 %f3,[%d6];mul.rn.f32 %f2,%f2,%f3;
 paramadd:add.rn.f32 %f0,%f0,%f2;add.u32 %r7,%r7,1;bra paramloop;
 paramstore:
 mul.wide.u32 %d5,%r3,4;add.u64 %d6,%d4,%d5;st.global.f32 [%d6],%f0;bra done;
 dx:
 setp.ge.u32 %p0,%r3,%r4;@%p0 bra done;
 mul.lo.u32 %r9,%r3,%r5;
 dxreduce:
 setp.ge.u32 %p0,%r7,%r5;@%p0 bra dxapply;
 add.u32 %r8,%r9,%r7;mul.wide.u32 %d5,%r8,4;mul.wide.u32 %d7,%r7,4;
 add.u64 %d6,%d3,%d5;ld.global.f32 %f2,[%d6];add.u64 %d8,%d2,%d7;ld.global.f32 %f3,[%d8];
 mul.rn.f32 %f2,%f2,%f3;add.rn.f32 %f0,%f0,%f2;
 add.u64 %d6,%d0,%d5;ld.global.f32 %f4,[%d6];mul.rn.f32 %f4,%f4,%f2;add.rn.f32 %f1,%f1,%f4;
 add.u32 %r7,%r7,1;bra dxreduce;
 dxapply:
 cvt.rn.f32.u32 %f5,%r5;mul.wide.u32 %d5,%r3,4;add.u64 %d6,%d1,%d5;ld.global.f32 %f6,[%d6];mov.u32 %r7,0;
 dxloop:
 setp.ge.u32 %p0,%r7,%r5;@%p0 bra done;
 add.u32 %r8,%r9,%r7;mul.wide.u32 %d5,%r8,4;mul.wide.u32 %d7,%r7,4;
 add.u64 %d6,%d3,%d5;ld.global.f32 %f2,[%d6];add.u64 %d8,%d2,%d7;ld.global.f32 %f3,[%d8];mul.rn.f32 %f2,%f2,%f3;
 add.u64 %d6,%d0,%d5;ld.global.f32 %f4,[%d6];mul.rn.f32 %f4,%f4,%f1;div.rn.f32 %f4,%f4,%f5;
 div.rn.f32 %f7,%f0,%f5;sub.rn.f32 %f2,%f2,%f7;sub.rn.f32 %f2,%f2,%f4;mul.rn.f32 %f2,%f2,%f6;
 add.u64 %d6,%d4,%d5;st.global.f32 [%d6],%f2;add.u32 %r7,%r7,1;bra dxloop;
 done:ret;
}
`
