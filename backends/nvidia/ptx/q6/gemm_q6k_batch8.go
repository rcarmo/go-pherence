package q6

// GemmQ6KBatch8PTX reuses each decoded Q6_K weight across up to eight input
// rows. Four warps compute four output rows per block.
const GemmQ6KBatch8PTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry gemm_q6_k_batch8(
 .param .u64 X,.param .u64 RAW,.param .u64 O,
 .param .u32 K,.param .u32 N,.param .u32 B) {
 .reg .pred %p<10>; .reg .u32 %r<72>; .reg .u64 %rd<36>; .reg .f32 %f<40>; .reg .b16 %h<2>;
 mov.u32 %r0,%ctaid.x; mov.u32 %r1,%ctaid.y; mov.u32 %r2,%tid.x; shr.u32 %r3,%r2,5; and.b32 %r4,%r2,31; shl.b32 %r5,%r0,2; add.u32 %r5,%r5,%r3;
 ld.param.u32 %r6,[K]; ld.param.u32 %r7,[N]; ld.param.u32 %r8,[B]; setp.ge.u32 %p0,%r5,%r7; @%p0 bra done; shl.b32 %r9,%r1,3; setp.ge.u32 %p1,%r9,%r8; @%p1 bra done;
 ld.param.u64 %rd0,[X]; ld.param.u64 %rd1,[RAW]; ld.param.u64 %rd2,[O]; shr.u32 %r10,%r6,8; mul.lo.u32 %r11,%r5,%r10;
 mov.f32 %f0,0f00000000; mov.f32 %f1,0f00000000; mov.f32 %f2,0f00000000; mov.f32 %f3,0f00000000; mov.f32 %f4,0f00000000; mov.f32 %f5,0f00000000; mov.f32 %f6,0f00000000; mov.f32 %f7,0f00000000; mov.u32 %r12,%r4;
k_loop: setp.ge.u32 %p2,%r12,%r6; @%p2 bra reduce; shr.u32 %r13,%r12,8; and.b32 %r14,%r12,255; shr.u32 %r15,%r14,7; shr.u32 %r16,%r14,5; and.b32 %r16,%r16,3; and.b32 %r17,%r14,31; add.u32 %r18,%r11,%r13; mul.lo.u32 %r19,%r18,210;
 shl.b32 %r20,%r15,6; and.b32 %r21,%r16,1; shl.b32 %r21,%r21,5; add.u32 %r20,%r20,%r21; add.u32 %r20,%r20,%r17; add.u32 %r20,%r20,%r19; cvt.u64.u32 %rd3,%r20; add.u64 %rd4,%rd1,%rd3; ld.global.u8 %r22,[%rd4]; setp.lt.u32 %p3,%r16,2; @%p3 bra ql_lo; shr.u32 %r23,%r22,4; bra ql_done; ql_lo: and.b32 %r23,%r22,15; ql_done:
 shl.b32 %r24,%r15,5; add.u32 %r24,%r24,%r17; add.u32 %r24,%r24,%r19; add.u32 %r24,%r24,128; cvt.u64.u32 %rd5,%r24; add.u64 %rd6,%rd1,%rd5; ld.global.u8 %r25,[%rd6]; shl.b32 %r26,%r16,1; shr.u32 %r27,%r25,%r26; and.b32 %r27,%r27,3; shl.b32 %r27,%r27,4; or.b32 %r28,%r23,%r27; add.s32 %r28,%r28,-32;
 shl.b32 %r29,%r15,3; shr.u32 %r30,%r17,4; shl.b32 %r31,%r16,1; add.u32 %r29,%r29,%r30; add.u32 %r29,%r29,%r31; add.u32 %r29,%r29,%r19; add.u32 %r29,%r29,192; cvt.u64.u32 %rd7,%r29; add.u64 %rd8,%rd1,%rd7; ld.global.s8 %r32,[%rd8]; add.u32 %r33,%r19,208; cvt.u64.u32 %rd9,%r33; add.u64 %rd10,%rd1,%rd9; ld.global.b16 %h0,[%rd10]; cvt.f32.f16 %f8,%h0; cvt.rn.f32.s32 %f9,%r32; cvt.rn.f32.s32 %f10,%r28; mul.f32 %f11,%f8,%f9; mul.f32 %f11,%f11,%f10;
 mad.lo.u32 %r34,%r9,%r6,%r12; mul.wide.u32 %rd11,%r34,4; add.u64 %rd12,%rd0,%rd11; mul.wide.u32 %rd13,%r6,4;
 mov.f32 %f12,0f00000000; add.u32 %r35,%r9,0; setp.lt.u32 %p4,%r35,%r8; @%p4 ld.global.f32 %f12,[%rd12]; fma.rn.f32 %f0,%f11,%f12,%f0;
 add.u64 %rd12,%rd12,%rd13; mov.f32 %f13,0f00000000; add.u32 %r35,%r9,1; setp.lt.u32 %p4,%r35,%r8; @%p4 ld.global.f32 %f13,[%rd12]; fma.rn.f32 %f1,%f11,%f13,%f1;
 add.u64 %rd12,%rd12,%rd13; mov.f32 %f14,0f00000000; add.u32 %r35,%r9,2; setp.lt.u32 %p4,%r35,%r8; @%p4 ld.global.f32 %f14,[%rd12]; fma.rn.f32 %f2,%f11,%f14,%f2;
 add.u64 %rd12,%rd12,%rd13; mov.f32 %f15,0f00000000; add.u32 %r35,%r9,3; setp.lt.u32 %p4,%r35,%r8; @%p4 ld.global.f32 %f15,[%rd12]; fma.rn.f32 %f3,%f11,%f15,%f3;
 add.u64 %rd12,%rd12,%rd13; mov.f32 %f16,0f00000000; add.u32 %r35,%r9,4; setp.lt.u32 %p4,%r35,%r8; @%p4 ld.global.f32 %f16,[%rd12]; fma.rn.f32 %f4,%f11,%f16,%f4;
 add.u64 %rd12,%rd12,%rd13; mov.f32 %f17,0f00000000; add.u32 %r35,%r9,5; setp.lt.u32 %p4,%r35,%r8; @%p4 ld.global.f32 %f17,[%rd12]; fma.rn.f32 %f5,%f11,%f17,%f5;
 add.u64 %rd12,%rd12,%rd13; mov.f32 %f18,0f00000000; add.u32 %r35,%r9,6; setp.lt.u32 %p4,%r35,%r8; @%p4 ld.global.f32 %f18,[%rd12]; fma.rn.f32 %f6,%f11,%f18,%f6;
 add.u64 %rd12,%rd12,%rd13; mov.f32 %f19,0f00000000; add.u32 %r35,%r9,7; setp.lt.u32 %p4,%r35,%r8; @%p4 ld.global.f32 %f19,[%rd12]; fma.rn.f32 %f7,%f11,%f19,%f7;
 add.u32 %r12,%r12,32; bra k_loop;
reduce: mov.u32 %r40,16;
red_loop: setp.eq.u32 %p5,%r40,0; @%p5 bra store;
 mov.b32 %r41,%f0; shfl.sync.down.b32 %r42|%p6,%r41,%r40,0x1f,0xffffffff; mov.b32 %f20,%r42; add.f32 %f0,%f0,%f20;
 mov.b32 %r41,%f1; shfl.sync.down.b32 %r42|%p6,%r41,%r40,0x1f,0xffffffff; mov.b32 %f20,%r42; add.f32 %f1,%f1,%f20;
 mov.b32 %r41,%f2; shfl.sync.down.b32 %r42|%p6,%r41,%r40,0x1f,0xffffffff; mov.b32 %f20,%r42; add.f32 %f2,%f2,%f20;
 mov.b32 %r41,%f3; shfl.sync.down.b32 %r42|%p6,%r41,%r40,0x1f,0xffffffff; mov.b32 %f20,%r42; add.f32 %f3,%f3,%f20;
 mov.b32 %r41,%f4; shfl.sync.down.b32 %r42|%p6,%r41,%r40,0x1f,0xffffffff; mov.b32 %f20,%r42; add.f32 %f4,%f4,%f20;
 mov.b32 %r41,%f5; shfl.sync.down.b32 %r42|%p6,%r41,%r40,0x1f,0xffffffff; mov.b32 %f20,%r42; add.f32 %f5,%f5,%f20;
 mov.b32 %r41,%f6; shfl.sync.down.b32 %r42|%p6,%r41,%r40,0x1f,0xffffffff; mov.b32 %f20,%r42; add.f32 %f6,%f6,%f20;
 mov.b32 %r41,%f7; shfl.sync.down.b32 %r42|%p6,%r41,%r40,0x1f,0xffffffff; mov.b32 %f20,%r42; add.f32 %f7,%f7,%f20; shr.u32 %r40,%r40,1; bra red_loop;
store: setp.ne.u32 %p7,%r4,0; @%p7 bra done; mad.lo.u32 %r43,%r9,%r7,%r5; mul.wide.u32 %rd14,%r43,4; add.u64 %rd15,%rd2,%rd14; mul.wide.u32 %rd16,%r7,4;
 add.u32 %r35,%r9,0; setp.lt.u32 %p8,%r35,%r8; @%p8 st.global.f32 [%rd15],%f0;
 add.u64 %rd15,%rd15,%rd16; add.u32 %r35,%r9,1; setp.lt.u32 %p8,%r35,%r8; @%p8 st.global.f32 [%rd15],%f1;
 add.u64 %rd15,%rd15,%rd16; add.u32 %r35,%r9,2; setp.lt.u32 %p8,%r35,%r8; @%p8 st.global.f32 [%rd15],%f2;
 add.u64 %rd15,%rd15,%rd16; add.u32 %r35,%r9,3; setp.lt.u32 %p8,%r35,%r8; @%p8 st.global.f32 [%rd15],%f3;
 add.u64 %rd15,%rd15,%rd16; add.u32 %r35,%r9,4; setp.lt.u32 %p8,%r35,%r8; @%p8 st.global.f32 [%rd15],%f4;
 add.u64 %rd15,%rd15,%rd16; add.u32 %r35,%r9,5; setp.lt.u32 %p8,%r35,%r8; @%p8 st.global.f32 [%rd15],%f5;
 add.u64 %rd15,%rd15,%rd16; add.u32 %r35,%r9,6; setp.lt.u32 %p8,%r35,%r8; @%p8 st.global.f32 [%rd15],%f6;
 add.u64 %rd15,%rd15,%rd16; add.u32 %r35,%r9,7; setp.lt.u32 %p8,%r35,%r8; @%p8 st.global.f32 [%rd15],%f7;
done: ret;
}`
