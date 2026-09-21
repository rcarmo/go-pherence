package q4

// GemmQ4KBatch8PTX reuses each decoded Q4_K weight across up to eight input
// rows. Four warps compute four output rows per block; warp shuffles avoid the
// block-wide reductions used by the batch-1 oracle.
const GemmQ4KBatch8PTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry gemm_q4_k_batch8(
 .param .u64 X,.param .u64 Q,.param .u64 SCALES,.param .u64 MINS,.param .u64 O,
 .param .u32 K,.param .u32 N,.param .u32 B) {
 .reg .pred %p<12>; .reg .u32 %r<64>; .reg .u64 %rd<32>; .reg .f32 %f<48>;
 mov.u32 %r0,%ctaid.x; mov.u32 %r1,%ctaid.y; mov.u32 %r2,%tid.x;
 shr.u32 %r3,%r2,5; and.b32 %r4,%r2,31; shl.b32 %r5,%r0,2; add.u32 %r5,%r5,%r3;
 ld.param.u32 %r6,[K]; ld.param.u32 %r7,[N]; ld.param.u32 %r8,[B];
 setp.ge.u32 %p0,%r5,%r7; @%p0 bra done; shl.b32 %r9,%r1,3; setp.ge.u32 %p1,%r9,%r8; @%p1 bra done;
 ld.param.u64 %rd0,[X]; ld.param.u64 %rd1,[Q]; ld.param.u64 %rd2,[SCALES]; ld.param.u64 %rd3,[MINS]; ld.param.u64 %rd4,[O];
 shr.u32 %r10,%r6,8; mov.f32 %f0,0f00000000; mov.f32 %f1,0f00000000; mov.f32 %f2,0f00000000; mov.f32 %f3,0f00000000; mov.f32 %f4,0f00000000; mov.f32 %f5,0f00000000; mov.f32 %f6,0f00000000; mov.f32 %f7,0f00000000; mov.u32 %r11,%r4;
k_loop: setp.ge.u32 %p2,%r11,%r6; @%p2 bra reduce;
 shr.u32 %r12,%r11,8; and.b32 %r13,%r11,255; shr.u32 %r14,%r13,5; and.b32 %r15,%r13,31; shr.u32 %r16,%r14,1; mad.lo.u32 %r17,%r16,32,%r15; mad.lo.u32 %r18,%r5,%r10,%r12; shl.b32 %r19,%r18,7; add.u32 %r19,%r19,%r17; cvt.u64.u32 %rd5,%r19; add.u64 %rd6,%rd1,%rd5; ld.global.u8 %r20,[%rd6]; and.b32 %r21,%r14,1; setp.eq.u32 %p3,%r21,0; @%p3 bra qlo; shr.u32 %r22,%r20,4; bra qdone; qlo: and.b32 %r22,%r20,15; qdone:
 shl.b32 %r23,%r18,3; add.u32 %r23,%r23,%r14; mul.wide.u32 %rd7,%r23,4; add.u64 %rd8,%rd2,%rd7; add.u64 %rd9,%rd3,%rd7; ld.global.f32 %f8,[%rd8]; ld.global.f32 %f9,[%rd9]; cvt.rn.f32.u32 %f10,%r22; neg.f32 %f9,%f9; fma.rn.f32 %f11,%f10,%f8,%f9;
 mad.lo.u32 %r24,%r9,%r6,%r11; mul.wide.u32 %rd10,%r24,4; add.u64 %rd11,%rd0,%rd10; mul.wide.u32 %rd12,%r6,4;
 mov.f32 %f12,0f00000000; add.u32 %r25,%r9,0; setp.lt.u32 %p4,%r25,%r8; @%p4 ld.global.f32 %f12,[%rd11]; fma.rn.f32 %f0,%f11,%f12,%f0;
 add.u64 %rd11,%rd11,%rd12; mov.f32 %f13,0f00000000; add.u32 %r25,%r9,1; setp.lt.u32 %p4,%r25,%r8; @%p4 ld.global.f32 %f13,[%rd11]; fma.rn.f32 %f1,%f11,%f13,%f1;
 add.u64 %rd11,%rd11,%rd12; mov.f32 %f14,0f00000000; add.u32 %r25,%r9,2; setp.lt.u32 %p4,%r25,%r8; @%p4 ld.global.f32 %f14,[%rd11]; fma.rn.f32 %f2,%f11,%f14,%f2;
 add.u64 %rd11,%rd11,%rd12; mov.f32 %f15,0f00000000; add.u32 %r25,%r9,3; setp.lt.u32 %p4,%r25,%r8; @%p4 ld.global.f32 %f15,[%rd11]; fma.rn.f32 %f3,%f11,%f15,%f3;
 add.u64 %rd11,%rd11,%rd12; mov.f32 %f16,0f00000000; add.u32 %r25,%r9,4; setp.lt.u32 %p4,%r25,%r8; @%p4 ld.global.f32 %f16,[%rd11]; fma.rn.f32 %f4,%f11,%f16,%f4;
 add.u64 %rd11,%rd11,%rd12; mov.f32 %f17,0f00000000; add.u32 %r25,%r9,5; setp.lt.u32 %p4,%r25,%r8; @%p4 ld.global.f32 %f17,[%rd11]; fma.rn.f32 %f5,%f11,%f17,%f5;
 add.u64 %rd11,%rd11,%rd12; mov.f32 %f18,0f00000000; add.u32 %r25,%r9,6; setp.lt.u32 %p4,%r25,%r8; @%p4 ld.global.f32 %f18,[%rd11]; fma.rn.f32 %f6,%f11,%f18,%f6;
 add.u64 %rd11,%rd11,%rd12; mov.f32 %f19,0f00000000; add.u32 %r25,%r9,7; setp.lt.u32 %p4,%r25,%r8; @%p4 ld.global.f32 %f19,[%rd11]; fma.rn.f32 %f7,%f11,%f19,%f7;
 add.u32 %r11,%r11,32; bra k_loop;
reduce:
 mov.u32 %r30,16;
red_loop: setp.eq.u32 %p5,%r30,0; @%p5 bra store;
 mov.b32 %r31,%f0; shfl.sync.down.b32 %r32|%p6,%r31,%r30,0x1f,0xffffffff; mov.b32 %f20,%r32; add.f32 %f0,%f0,%f20;
 mov.b32 %r31,%f1; shfl.sync.down.b32 %r32|%p6,%r31,%r30,0x1f,0xffffffff; mov.b32 %f20,%r32; add.f32 %f1,%f1,%f20;
 mov.b32 %r31,%f2; shfl.sync.down.b32 %r32|%p6,%r31,%r30,0x1f,0xffffffff; mov.b32 %f20,%r32; add.f32 %f2,%f2,%f20;
 mov.b32 %r31,%f3; shfl.sync.down.b32 %r32|%p6,%r31,%r30,0x1f,0xffffffff; mov.b32 %f20,%r32; add.f32 %f3,%f3,%f20;
 mov.b32 %r31,%f4; shfl.sync.down.b32 %r32|%p6,%r31,%r30,0x1f,0xffffffff; mov.b32 %f20,%r32; add.f32 %f4,%f4,%f20;
 mov.b32 %r31,%f5; shfl.sync.down.b32 %r32|%p6,%r31,%r30,0x1f,0xffffffff; mov.b32 %f20,%r32; add.f32 %f5,%f5,%f20;
 mov.b32 %r31,%f6; shfl.sync.down.b32 %r32|%p6,%r31,%r30,0x1f,0xffffffff; mov.b32 %f20,%r32; add.f32 %f6,%f6,%f20;
 mov.b32 %r31,%f7; shfl.sync.down.b32 %r32|%p6,%r31,%r30,0x1f,0xffffffff; mov.b32 %f20,%r32; add.f32 %f7,%f7,%f20;
 shr.u32 %r30,%r30,1; bra red_loop;
store: setp.ne.u32 %p7,%r4,0; @%p7 bra done; mad.lo.u32 %r33,%r9,%r7,%r5; mul.wide.u32 %rd13,%r33,4; add.u64 %rd14,%rd4,%rd13; mul.wide.u32 %rd15,%r7,4;
 add.u32 %r25,%r9,0; setp.lt.u32 %p8,%r25,%r8; @%p8 st.global.f32 [%rd14],%f0;
 add.u64 %rd14,%rd14,%rd15; add.u32 %r25,%r9,1; setp.lt.u32 %p8,%r25,%r8; @%p8 st.global.f32 [%rd14],%f1;
 add.u64 %rd14,%rd14,%rd15; add.u32 %r25,%r9,2; setp.lt.u32 %p8,%r25,%r8; @%p8 st.global.f32 [%rd14],%f2;
 add.u64 %rd14,%rd14,%rd15; add.u32 %r25,%r9,3; setp.lt.u32 %p8,%r25,%r8; @%p8 st.global.f32 [%rd14],%f3;
 add.u64 %rd14,%rd14,%rd15; add.u32 %r25,%r9,4; setp.lt.u32 %p8,%r25,%r8; @%p8 st.global.f32 [%rd14],%f4;
 add.u64 %rd14,%rd14,%rd15; add.u32 %r25,%r9,5; setp.lt.u32 %p8,%r25,%r8; @%p8 st.global.f32 [%rd14],%f5;
 add.u64 %rd14,%rd14,%rd15; add.u32 %r25,%r9,6; setp.lt.u32 %p8,%r25,%r8; @%p8 st.global.f32 [%rd14],%f6;
 add.u64 %rd14,%rd14,%rd15; add.u32 %r25,%r9,7; setp.lt.u32 %p8,%r25,%r8; @%p8 st.global.f32 [%rd14],%f7;
done: ret;
}`
