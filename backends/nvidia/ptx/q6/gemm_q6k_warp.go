package q6

// GemmQ6KWarpPTX computes four output rows per 128-thread block. Each warp
// reduces one output with shuffle instructions; batch rows are grid.y.
const GemmQ6KWarpPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry gemm_q6_k_warp(
 .param .u64 X,.param .u64 RAW,.param .u64 O,
 .param .u32 K,.param .u32 N,.param .u32 B) {
 .reg .pred %p<8>; .reg .u32 %r<64>; .reg .u64 %rd<28>; .reg .f32 %f<16>; .reg .b16 %h<2>;
 mov.u32 %r0,%ctaid.x; mov.u32 %r1,%ctaid.y; mov.u32 %r2,%tid.x; shr.u32 %r3,%r2,5; and.b32 %r4,%r2,31; shl.b32 %r5,%r0,2; add.u32 %r5,%r5,%r3;
 ld.param.u32 %r6,[K]; ld.param.u32 %r7,[N]; ld.param.u32 %r8,[B]; setp.ge.u32 %p0,%r5,%r7; setp.ge.u32 %p1,%r1,%r8; or.pred %p0,%p0,%p1; @%p0 bra done;
 ld.param.u64 %rd0,[X]; ld.param.u64 %rd1,[RAW]; ld.param.u64 %rd2,[O]; shr.u32 %r9,%r6,8; mul.lo.u32 %r10,%r5,%r9; mul.lo.u32 %r11,%r1,%r6; mov.f32 %f0,0f00000000; mov.u32 %r12,%r4;
k_loop: setp.ge.u32 %p2,%r12,%r6; @%p2 bra reduce; shr.u32 %r13,%r12,8; and.b32 %r14,%r12,255; shr.u32 %r15,%r14,7; shr.u32 %r16,%r14,5; and.b32 %r16,%r16,3; and.b32 %r17,%r14,31; add.u32 %r18,%r10,%r13; mul.lo.u32 %r19,%r18,210;
 shl.b32 %r20,%r15,6; and.b32 %r21,%r16,1; shl.b32 %r21,%r21,5; add.u32 %r20,%r20,%r21; add.u32 %r20,%r20,%r17; add.u32 %r20,%r20,%r19; cvt.u64.u32 %rd3,%r20; add.u64 %rd4,%rd1,%rd3; ld.global.u8 %r22,[%rd4]; setp.lt.u32 %p3,%r16,2; @%p3 bra ql_lo; shr.u32 %r23,%r22,4; bra ql_done; ql_lo: and.b32 %r23,%r22,15; ql_done:
 shl.b32 %r24,%r15,5; add.u32 %r24,%r24,%r17; add.u32 %r24,%r24,%r19; add.u32 %r24,%r24,128; cvt.u64.u32 %rd5,%r24; add.u64 %rd6,%rd1,%rd5; ld.global.u8 %r25,[%rd6]; shl.b32 %r26,%r16,1; shr.u32 %r27,%r25,%r26; and.b32 %r27,%r27,3; shl.b32 %r27,%r27,4; or.b32 %r28,%r23,%r27; add.s32 %r28,%r28,-32;
 shl.b32 %r29,%r15,3; shr.u32 %r30,%r17,4; shl.b32 %r31,%r16,1; add.u32 %r29,%r29,%r30; add.u32 %r29,%r29,%r31; add.u32 %r29,%r29,%r19; add.u32 %r29,%r29,192; cvt.u64.u32 %rd7,%r29; add.u64 %rd8,%rd1,%rd7; ld.global.s8 %r32,[%rd8]; add.u32 %r33,%r19,208; cvt.u64.u32 %rd9,%r33; add.u64 %rd10,%rd1,%rd9; ld.global.b16 %h0,[%rd10]; cvt.f32.f16 %f1,%h0; cvt.rn.f32.s32 %f2,%r32; cvt.rn.f32.s32 %f3,%r28; mul.f32 %f4,%f1,%f2; mul.f32 %f4,%f4,%f3;
 add.u32 %r34,%r11,%r12; mul.wide.u32 %rd11,%r34,4; add.u64 %rd12,%rd0,%rd11; ld.global.f32 %f5,[%rd12]; fma.rn.f32 %f0,%f4,%f5,%f0; add.u32 %r12,%r12,32; bra k_loop;
reduce: mov.u32 %r40,16;
red_loop: setp.eq.u32 %p4,%r40,0; @%p4 bra store; mov.b32 %r41,%f0; shfl.sync.down.b32 %r42|%p5,%r41,%r40,0x1f,0xffffffff; mov.b32 %f6,%r42; add.f32 %f0,%f0,%f6; shr.u32 %r40,%r40,1; bra red_loop;
store: setp.ne.u32 %p6,%r4,0; @%p6 bra done; mad.lo.u32 %r43,%r1,%r7,%r5; mul.wide.u32 %rd13,%r43,4; add.u64 %rd14,%rd2,%rd13; st.global.f32 [%rd14],%f0;
done: ret;
}`
