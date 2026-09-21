package q5

// GemmQ5KWarpPTX computes four output rows per block with one warp per row.
const GemmQ5KWarpPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry gemm_q5_k_warp(.param .u64 X,.param .u64 RAW,.param .u64 O,.param .u32 K,.param .u32 N,.param .u32 B){
 .reg .pred %p<10>; .reg .u32 %r<72>; .reg .u64 %rd<36>; .reg .f32 %f<20>; .reg .b16 %h<4>;
 mov.u32 %r0,%ctaid.x;mov.u32 %r1,%ctaid.y;mov.u32 %r2,%tid.x;shr.u32 %r3,%r2,5;and.b32 %r4,%r2,31;shl.b32 %r5,%r0,2;add.u32 %r5,%r5,%r3;ld.param.u32 %r6,[K];ld.param.u32 %r7,[N];ld.param.u32 %r8,[B];setp.ge.u32 %p0,%r5,%r7;setp.ge.u32 %p1,%r1,%r8;or.pred %p0,%p0,%p1;@%p0 bra done;
 ld.param.u64 %rd0,[X];ld.param.u64 %rd1,[RAW];ld.param.u64 %rd2,[O];shr.u32 %r9,%r6,8;mul.lo.u32 %r10,%r5,%r9;mul.lo.u32 %r11,%r1,%r6;mov.f32 %f0,0f00000000;mov.u32 %r12,%r4;
kloop:setp.ge.u32 %p2,%r12,%r6;@%p2 bra reduce;shr.u32 %r13,%r12,8;and.b32 %r14,%r12,255;shr.u32 %r15,%r14,6;shr.u32 %r16,%r14,5;and.b32 %r17,%r14,31;and.b32 %r18,%r16,1;add.u32 %r19,%r10,%r13;mul.lo.u32 %r20,%r19,176;cvt.u64.u32 %rd3,%r20;add.u64 %rd4,%rd1,%rd3;ld.global.b16 %h0,[%rd4];cvt.f32.f16 %f1,%h0;ld.global.b16 %h1,[%rd4+2];cvt.f32.f16 %f2,%h1;
 add.u64 %rd5,%rd4,4;cvt.u64.u32 %rd6,%r16;add.u64 %rd7,%rd5,%rd6;ld.global.u8 %r21,[%rd7];setp.lt.u32 %p3,%r16,4;@%p3 bra slow;add.u32 %r22,%r16,-4;add.u32 %r23,%r16,4;cvt.u64.u32 %rd8,%r23;add.u64 %rd9,%rd5,%rd8;ld.global.u8 %r24,[%rd9];and.b32 %r25,%r24,15;cvt.u64.u32 %rd10,%r22;add.u64 %rd11,%rd5,%rd10;ld.global.u8 %r26,[%rd11];shr.u32 %r26,%r26,6;shl.b32 %r26,%r26,4;or.b32 %r27,%r25,%r26;add.u32 %r28,%r22,4;cvt.u64.u32 %rd12,%r28;add.u64 %rd13,%rd5,%rd12;ld.global.u8 %r29,[%rd13];shr.u32 %r30,%r24,4;shr.u32 %r31,%r29,6;shl.b32 %r31,%r31,4;or.b32 %r32,%r30,%r31;bra sdone;
slow:and.b32 %r27,%r21,63;add.u32 %r28,%r16,4;cvt.u64.u32 %rd12,%r28;add.u64 %rd13,%rd5,%rd12;ld.global.u8 %r29,[%rd13];and.b32 %r32,%r29,63;
sdone:cvt.rn.f32.u32 %f3,%r27;cvt.rn.f32.u32 %f4,%r32;mul.f32 %f3,%f3,%f1;mul.f32 %f4,%f4,%f2;add.u32 %r33,%r20,16;add.u32 %r33,%r33,%r17;cvt.u64.u32 %rd14,%r33;add.u64 %rd15,%rd1,%rd14;ld.global.u8 %r34,[%rd15];mov.u32 %r35,1;shl.b32 %r36,%r15,1;add.u32 %r36,%r36,%r18;shl.b32 %r35,%r35,%r36;and.b32 %r37,%r34,%r35;shl.b32 %r38,%r15,5;add.u32 %r38,%r38,%r17;add.u32 %r38,%r38,%r20;add.u32 %r38,%r38,48;cvt.u64.u32 %rd16,%r38;add.u64 %rd17,%rd1,%rd16;ld.global.u8 %r39,[%rd17];setp.eq.u32 %p4,%r18,0;@%p4 bra qlo;shr.u32 %r40,%r39,4;bra qdone;qlo:and.b32 %r40,%r39,15;qdone:setp.eq.u32 %p5,%r37,0;@%p5 bra highdone;add.u32 %r40,%r40,16;highdone:cvt.rn.f32.u32 %f5,%r40;neg.f32 %f4,%f4;fma.rn.f32 %f6,%f5,%f3,%f4;add.u32 %r41,%r11,%r12;mul.wide.u32 %rd18,%r41,4;add.u64 %rd19,%rd0,%rd18;ld.global.f32 %f7,[%rd19];fma.rn.f32 %f0,%f6,%f7,%f0;add.u32 %r12,%r12,32;bra kloop;
reduce:mov.u32 %r50,16;red:setp.eq.u32 %p6,%r50,0;@%p6 bra store;mov.b32 %r51,%f0;shfl.sync.down.b32 %r52|%p7,%r51,%r50,0x1f,0xffffffff;mov.b32 %f8,%r52;add.f32 %f0,%f0,%f8;shr.u32 %r50,%r50,1;bra red;
store:setp.ne.u32 %p8,%r4,0;@%p8 bra done;mad.lo.u32 %r53,%r1,%r7,%r5;mul.wide.u32 %rd20,%r53,4;add.u64 %rd21,%rd2,%rd20;st.global.f32 [%rd21],%f0;done:ret;}`
