//go:build amd64

#include "textflag.h"

TEXT ·_go_rope_partial_avx2(SB), $128-40

    MOVQ x+0(FP), DI
    MOVQ freq+8(FP), SI
    MOVQ heads+16(FP), DX
    MOVQ headDim+24(FP), CX
    MOVQ rotHalf+32(FP), R8
    ADDQ $128, SP

    BYTE $0x55                   // push    rbp
    WORD $0x5741                 // push    r15
    WORD $0x5641                 // push    r14
    WORD $0x5541                 // push    r13
    WORD $0x5441                 // push    r12
    BYTE $0x53                   // push    rbx
    LONG $0x50ec8348             // sub    rsp, 80
    WORD $0xd285                 // test    edx, edx
	JLE LBB0_19
    WORD $0x8949; BYTE $0xf2     // mov    r10, rsi
    WORD $0x8949; BYTE $0xfb     // mov    r11, rdi
    WORD $0x6349; BYTE $0xc0     // movsxd    rax, r8d
    WORD $0xf883; BYTE $0x08     // cmp    eax, 8
	JGE LBB0_2
    WORD $0x8545; BYTE $0xc0     // test    r8d, r8d
	JLE LBB0_19
    WORD $0x6348; BYTE $0xc9     // movsxd    rcx, ecx
    WORD $0xd289                 // mov    edx, edx
    LONG $0x18c38349             // add    r11, 24
    LONG $0x02e1c148             // shl    rcx, 2
	JMP LBB0_11
LBB0_18:
    WORD $0x0149; BYTE $0xcb     // add    r11, rcx
    WORD $0xff48; BYTE $0xca     // dec    rdx
	JE LBB0_19
LBB0_11:
    LONG $0x107ac1c4; BYTE $0x02 // vmovss    xmm0, dword [r10]
    LONG $0x107ac1c4; WORD $0x044a // vmovss    xmm1, dword [r10 + 4]
    LONG $0x107ac1c4; WORD $0xe853 // vmovss    xmm2, dword [r11 - 24]
    LONG $0x107ac1c4; WORD $0x835c; BYTE $0xe8 // vmovss    xmm3, dword [r11 + 4*rax - 24]
    LONG $0xe259fac5             // vmulss    xmm4, xmm0, xmm2
    LONG $0xeb59f2c5             // vmulss    xmm5, xmm1, xmm3
    LONG $0xe55cdac5             // vsubss    xmm4, xmm4, xmm5
    LONG $0x117ac1c4; WORD $0xe863 // vmovss    dword [r11 - 24], xmm4
    LONG $0xca59f2c5             // vmulss    xmm1, xmm1, xmm2
    LONG $0xc359fac5             // vmulss    xmm0, xmm0, xmm3
    LONG $0xc058f2c5             // vaddss    xmm0, xmm1, xmm0
    LONG $0x117ac1c4; WORD $0x8344; BYTE $0xe8 // vmovss    dword [r11 + 4*rax - 24], xmm0
    LONG $0x01f88341             // cmp    r8d, 1
	JE LBB0_18
    LONG $0x107ac1c4; WORD $0x0842 // vmovss    xmm0, dword [r10 + 8]
    LONG $0x107ac1c4; WORD $0x0c4a // vmovss    xmm1, dword [r10 + 12]
    LONG $0x107ac1c4; WORD $0xec53 // vmovss    xmm2, dword [r11 - 20]
    LONG $0x107ac1c4; WORD $0x835c; BYTE $0xec // vmovss    xmm3, dword [r11 + 4*rax - 20]
    LONG $0xe259fac5             // vmulss    xmm4, xmm0, xmm2
    LONG $0xeb59f2c5             // vmulss    xmm5, xmm1, xmm3
    LONG $0xe55cdac5             // vsubss    xmm4, xmm4, xmm5
    LONG $0x117ac1c4; WORD $0xec63 // vmovss    dword [r11 - 20], xmm4
    LONG $0xca59f2c5             // vmulss    xmm1, xmm1, xmm2
    LONG $0xc359fac5             // vmulss    xmm0, xmm0, xmm3
    LONG $0xc058f2c5             // vaddss    xmm0, xmm1, xmm0
    LONG $0x117ac1c4; WORD $0x8344; BYTE $0xec // vmovss    dword [r11 + 4*rax - 20], xmm0
    LONG $0x02f88341             // cmp    r8d, 2
	JE LBB0_18
    LONG $0x107ac1c4; WORD $0x1042 // vmovss    xmm0, dword [r10 + 16]
    LONG $0x107ac1c4; WORD $0x144a // vmovss    xmm1, dword [r10 + 20]
    LONG $0x107ac1c4; WORD $0xf053 // vmovss    xmm2, dword [r11 - 16]
    LONG $0x107ac1c4; WORD $0x835c; BYTE $0xf0 // vmovss    xmm3, dword [r11 + 4*rax - 16]
    LONG $0xe259fac5             // vmulss    xmm4, xmm0, xmm2
    LONG $0xeb59f2c5             // vmulss    xmm5, xmm1, xmm3
    LONG $0xe55cdac5             // vsubss    xmm4, xmm4, xmm5
    LONG $0x117ac1c4; WORD $0xf063 // vmovss    dword [r11 - 16], xmm4
    LONG $0xca59f2c5             // vmulss    xmm1, xmm1, xmm2
    LONG $0xc359fac5             // vmulss    xmm0, xmm0, xmm3
    LONG $0xc058f2c5             // vaddss    xmm0, xmm1, xmm0
    LONG $0x117ac1c4; WORD $0x8344; BYTE $0xf0 // vmovss    dword [r11 + 4*rax - 16], xmm0
    LONG $0x03f88341             // cmp    r8d, 3
	JE LBB0_18
    LONG $0x107ac1c4; WORD $0x1842 // vmovss    xmm0, dword [r10 + 24]
    LONG $0x107ac1c4; WORD $0x1c4a // vmovss    xmm1, dword [r10 + 28]
    LONG $0x107ac1c4; WORD $0xf453 // vmovss    xmm2, dword [r11 - 12]
    LONG $0x107ac1c4; WORD $0x835c; BYTE $0xf4 // vmovss    xmm3, dword [r11 + 4*rax - 12]
    LONG $0xe259fac5             // vmulss    xmm4, xmm0, xmm2
    LONG $0xeb59f2c5             // vmulss    xmm5, xmm1, xmm3
    LONG $0xe55cdac5             // vsubss    xmm4, xmm4, xmm5
    LONG $0x117ac1c4; WORD $0xf463 // vmovss    dword [r11 - 12], xmm4
    LONG $0xca59f2c5             // vmulss    xmm1, xmm1, xmm2
    LONG $0xc359fac5             // vmulss    xmm0, xmm0, xmm3
    LONG $0xc058f2c5             // vaddss    xmm0, xmm1, xmm0
    LONG $0x117ac1c4; WORD $0x8344; BYTE $0xf4 // vmovss    dword [r11 + 4*rax - 12], xmm0
    LONG $0x04f88341             // cmp    r8d, 4
	JE LBB0_18
    LONG $0x107ac1c4; WORD $0x2042 // vmovss    xmm0, dword [r10 + 32]
    LONG $0x107ac1c4; WORD $0x244a // vmovss    xmm1, dword [r10 + 36]
    LONG $0x107ac1c4; WORD $0xf853 // vmovss    xmm2, dword [r11 - 8]
    LONG $0x107ac1c4; WORD $0x835c; BYTE $0xf8 // vmovss    xmm3, dword [r11 + 4*rax - 8]
    LONG $0xe259fac5             // vmulss    xmm4, xmm0, xmm2
    LONG $0xeb59f2c5             // vmulss    xmm5, xmm1, xmm3
    LONG $0xe55cdac5             // vsubss    xmm4, xmm4, xmm5
    LONG $0x117ac1c4; WORD $0xf863 // vmovss    dword [r11 - 8], xmm4
    LONG $0xca59f2c5             // vmulss    xmm1, xmm1, xmm2
    LONG $0xc359fac5             // vmulss    xmm0, xmm0, xmm3
    LONG $0xc058f2c5             // vaddss    xmm0, xmm1, xmm0
    LONG $0x117ac1c4; WORD $0x8344; BYTE $0xf8 // vmovss    dword [r11 + 4*rax - 8], xmm0
    LONG $0x05f88341             // cmp    r8d, 5
	JE LBB0_18
    LONG $0x107ac1c4; WORD $0x2842 // vmovss    xmm0, dword [r10 + 40]
    LONG $0x107ac1c4; WORD $0x2c4a // vmovss    xmm1, dword [r10 + 44]
    LONG $0x107ac1c4; WORD $0xfc53 // vmovss    xmm2, dword [r11 - 4]
    LONG $0x107ac1c4; WORD $0x835c; BYTE $0xfc // vmovss    xmm3, dword [r11 + 4*rax - 4]
    LONG $0xe259fac5             // vmulss    xmm4, xmm0, xmm2
    LONG $0xeb59f2c5             // vmulss    xmm5, xmm1, xmm3
    LONG $0xe55cdac5             // vsubss    xmm4, xmm4, xmm5
    LONG $0x117ac1c4; WORD $0xfc63 // vmovss    dword [r11 - 4], xmm4
    LONG $0xca59f2c5             // vmulss    xmm1, xmm1, xmm2
    LONG $0xc359fac5             // vmulss    xmm0, xmm0, xmm3
    LONG $0xc058f2c5             // vaddss    xmm0, xmm1, xmm0
    LONG $0x117ac1c4; WORD $0x8344; BYTE $0xfc // vmovss    dword [r11 + 4*rax - 4], xmm0
    LONG $0x06f88341             // cmp    r8d, 6
	JE LBB0_18
    LONG $0x107ac1c4; WORD $0x3042 // vmovss    xmm0, dword [r10 + 48]
    LONG $0x107ac1c4; WORD $0x344a // vmovss    xmm1, dword [r10 + 52]
    LONG $0x107ac1c4; BYTE $0x13 // vmovss    xmm2, dword [r11]
    LONG $0x107ac1c4; WORD $0x831c // vmovss    xmm3, dword [r11 + 4*rax]
    LONG $0xe259fac5             // vmulss    xmm4, xmm0, xmm2
    LONG $0xeb59f2c5             // vmulss    xmm5, xmm1, xmm3
    LONG $0xe55cdac5             // vsubss    xmm4, xmm4, xmm5
    LONG $0x117ac1c4; BYTE $0x23 // vmovss    dword [r11], xmm4
    LONG $0xca59f2c5             // vmulss    xmm1, xmm1, xmm2
    LONG $0xc359fac5             // vmulss    xmm0, xmm0, xmm3
    LONG $0xc058f2c5             // vaddss    xmm0, xmm1, xmm0
    LONG $0x117ac1c4; WORD $0x8304 // vmovss    dword [r11 + 4*rax], xmm0
	JMP LBB0_18
LBB0_2:
    WORD $0x6348; BYTE $0xc9     // movsxd    rcx, ecx
    WORD $0xd289                 // mov    edx, edx
    LONG $0x24548948; BYTE $0x40 // mov    qword [rsp + 64], rdx
    LONG $0x02e1c148             // shl    rcx, 2
    LONG $0x244c8948; BYTE $0x48 // mov    qword [rsp + 72], rcx
    LONG $0xf8488d48             // lea    rcx, [rax - 8]
    WORD $0x8948; BYTE $0xca     // mov    rdx, rcx
    LONG $0x03eac148             // shr    rdx, 3
    LONG $0xf8e18348             // and    rcx, -8
    LONG $0x09c18348             // add    rcx, 9
    WORD $0x3948; BYTE $0xc1     // cmp    rcx, rax
    LONG $0xc84e0f48             // cmovle    rcx, rax
    QUAD $0x000000008d3c8d48     // lea    rdi, [4*rcx]
    WORD $0x8948; BYTE $0xd6     // mov    rsi, rdx
    LONG $0x05e6c148             // shl    rsi, 5
    QUAD $0x00000000850c8d4c     // lea    r9, [4*rax]
    QUAD $0x00000000cd1c8d48     // lea    rbx, [8*rcx]
    LONG $0x06e2c148             // shl    rdx, 6
    WORD $0x2948; BYTE $0xd3     // sub    rbx, rdx
    LONG $0x245c8948; BYTE $0x18 // mov    qword [rsp + 24], rbx
    WORD $0x8944; BYTE $0xc2     // mov    edx, r8d
    LONG $0xfff8e281; WORD $0x7fff // and    edx, 2147483640
    LONG $0x015a8d48             // lea    rbx, [rdx + 1]
    WORD $0x3948; BYTE $0xc3     // cmp    rbx, rax
    LONG $0xd84e0f48             // cmovle    rbx, rax
    WORD $0x8949; BYTE $0xde     // mov    r14, rbx
    WORD $0x2949; BYTE $0xd6     // sub    r14, rdx
    LONG $0x2474894c; BYTE $0x28 // mov    qword [rsp + 40], r14
    LONG $0x833c8d4d             // lea    r15, [r11 + 4*rax]
    LONG $0x890c8d49             // lea    rcx, [r9 + 4*rcx]
    WORD $0x2948; BYTE $0xf1     // sub    rcx, rsi
    LONG $0xe0c18348             // add    rcx, -32
    LONG $0x244c8948; BYTE $0x30 // mov    qword [rsp + 48], rcx
    WORD $0x2948; BYTE $0xf7     // sub    rdi, rsi
    LONG $0xe0c78348             // add    rdi, -32
    LONG $0x247c8948; BYTE $0x38 // mov    qword [rsp + 56], rdi
    QUAD $0xfffffffffff8bd49; WORD $0x7fff // mov    r13, 9223372036854775800
    LONG $0x245c8948; BYTE $0x10 // mov    qword [rsp + 16], rbx
    WORD $0x2149; BYTE $0xdd     // and    r13, rbx
    WORD $0x2949; BYTE $0xd5     // sub    r13, rdx
    WORD $0xed31                 // xor    ebp, ebp
    LONG $0x24448944; BYTE $0x04 // mov    dword [rsp + 4], r8d
	JMP LBB0_3
LBB0_28:
    WORD $0xff48; BYTE $0xc5     // inc    rbp
    LONG $0x244c8b48; BYTE $0x48 // mov    rcx, qword [rsp + 72]
    WORD $0x0149; BYTE $0xcb     // add    r11, rcx
    WORD $0x0149; BYTE $0xcf     // add    r15, rcx
    LONG $0x246c3b48; BYTE $0x40 // cmp    rbp, qword [rsp + 64]
	JE LBB0_19
LBB0_3:
    WORD $0x894c; BYTE $0xdf     // mov    rdi, r11
    WORD $0x3145; BYTE $0xf6     // xor    r14d, r14d
    LONG $0x244c8b4c; BYTE $0x38 // mov    r9, qword [rsp + 56]
    QUAD $0x0000000085148d48     // lea    rdx, [4*rax]
    LONG $0x24648b4c; BYTE $0x30 // mov    r12, qword [rsp + 48]
    WORD $0x894c; BYTE $0xd6     // mov    rsi, r10
    WORD $0xc931                 // xor    ecx, ecx
LBB0_4:
    WORD $0x8948; BYTE $0xcb     // mov    rbx, rcx
    LONG $0x107cc1c4; WORD $0xca04 // vmovups    ymm0, yword [r10 + 8*rcx]
    LONG $0x107cc1c4; WORD $0xca4c; BYTE $0x20 // vmovups    ymm1, yword [r10 + 8*rcx + 32]
    LONG $0xd1c6fcc5; BYTE $0x88 // vshufps    ymm2, ymm0, ymm1, 136
    LONG $0x01fde3c4; WORD $0xd8d2 // vpermpd    ymm2, ymm2, 216
    LONG $0xc1c6fcc5; BYTE $0xdd // vshufps    ymm0, ymm0, ymm1, 221
    LONG $0x01fde3c4; WORD $0xd8c0 // vpermpd    ymm0, ymm0, 216
    LONG $0x107cc1c4; WORD $0x8b0c // vmovups    ymm1, yword [r11 + 4*rcx]
    LONG $0x107cc1c4; WORD $0x8f1c // vmovups    ymm3, yword [r15 + 4*rcx]
    LONG $0xe259f4c5             // vmulps    ymm4, ymm1, ymm2
    LONG $0xe859e4c5             // vmulps    ymm5, ymm3, ymm0
    LONG $0xe55cdcc5             // vsubps    ymm4, ymm4, ymm5
    LONG $0x117cc1c4; WORD $0x8b24 // vmovups    yword [r11 + 4*rcx], ymm4
    LONG $0xc059f4c5             // vmulps    ymm0, ymm1, ymm0
    LONG $0xca59e4c5             // vmulps    ymm1, ymm3, ymm2
    LONG $0xc158fcc5             // vaddps    ymm0, ymm0, ymm1
    LONG $0x117cc1c4; WORD $0x8f04 // vmovups    yword [r15 + 4*rcx], ymm0
    LONG $0x08c18348             // add    rcx, 8
    LONG $0x40c68348             // add    rsi, 64
    LONG $0x20c48349             // add    r12, 32
    LONG $0x20c28348             // add    rdx, 32
    LONG $0x20c18349             // add    r9, 32
    LONG $0x20c68349             // add    r14, 32
    LONG $0x20c78348             // add    rdi, 32
    LONG $0x10c38348             // add    rbx, 16
    WORD $0x3948; BYTE $0xc3     // cmp    rbx, rax
	JLE LBB0_4
    WORD $0x3941; BYTE $0xc8     // cmp    r8d, ecx
	JLE LBB0_28
    LONG $0x247c8348; WORD $0x1028 // cmp    qword [rsp + 40], 16
	JB LBB0_27
    WORD $0x014d; BYTE $0xde     // add    r14, r11
    WORD $0x014d; BYTE $0xd9     // add    r9, r11
    WORD $0x014c; BYTE $0xda     // add    rdx, r11
    WORD $0x014d; BYTE $0xdc     // add    r12, r11
    LONG $0x24448b4c; BYTE $0x18 // mov    r8, qword [rsp + 24]
    LONG $0x061c8d4a             // lea    rbx, [rsi + r8]
    LONG $0xbcc38348             // add    rbx, -68
    LONG $0x245c8948; BYTE $0x08 // mov    qword [rsp + 8], rbx
    WORD $0x0149; BYTE $0xf0     // add    r8, rsi
    LONG $0xc0c08349             // add    r8, -64
    LONG $0x2444894c; BYTE $0x20 // mov    qword [rsp + 32], r8
    WORD $0x394d; BYTE $0xe6     // cmp    r14, r12
    LONG $0x2454920f; BYTE $0x03 // setb    byte [rsp + 3]
    WORD $0x394c; BYTE $0xca     // cmp    rdx, r9
    LONG $0x2454920f; BYTE $0x02 // setb    byte [rsp + 2]
    WORD $0x3949; BYTE $0xde     // cmp    r14, rbx
    WORD $0x920f; BYTE $0xd3     // setb    bl
    WORD $0x394c; BYTE $0xce     // cmp    rsi, r9
    LONG $0x2454920f; BYTE $0x01 // setb    byte [rsp + 1]
    WORD $0x394d; BYTE $0xc6     // cmp    r14, r8
    LONG $0xd6920f41             // setb    r14b
    LONG $0x04468d4c             // lea    r8, [rsi + 4]
    WORD $0x394d; BYTE $0xc8     // cmp    r8, r9
    LONG $0x2414920f             // setb    byte [rsp]
    LONG $0x24543b48; BYTE $0x08 // cmp    rdx, qword [rsp + 8]
    LONG $0xd1920f41             // setb    r9b
    WORD $0x394c; BYTE $0xe6     // cmp    rsi, r12
    LONG $0x2454920f; BYTE $0x08 // setb    byte [rsp + 8]
    LONG $0x24543b48; BYTE $0x20 // cmp    rdx, qword [rsp + 32]
    WORD $0x920f; BYTE $0xd2     // setb    dl
    LONG $0x04468d4c             // lea    r8, [rsi + 4]
    WORD $0x394d; BYTE $0xe0     // cmp    r8, r12
    LONG $0xd4920f41             // setb    r12b
    LONG $0x44b60f44; WORD $0x0224 // movzx    r8d, byte [rsp + 2]
    LONG $0x24448444; BYTE $0x03 // test    byte [rsp + 3], r8b
	JNE LBB0_8
    LONG $0x01245c22             // and    bl, byte [rsp + 1]
    LONG $0x24448b44; BYTE $0x04 // mov    r8d, dword [rsp + 4]
	JNE LBB0_27
    LONG $0x24342244             // and    r14b, byte [rsp]
	JNE LBB0_27
    LONG $0x244c2244; BYTE $0x08 // and    r9b, byte [rsp + 8]
	JNE LBB0_27
    WORD $0x2044; BYTE $0xe2     // and    dl, r12b
	JNE LBB0_27
    WORD $0x014c; BYTE $0xe9     // add    rcx, r13
    QUAD $0x0000000085148d48     // lea    rdx, [4*rax]
    WORD $0x0148; BYTE $0xfa     // add    rdx, rdi
    WORD $0x3145; BYTE $0xc9     // xor    r9d, r9d
LBB0_25:
    LONG $0x107ca1c4; WORD $0xce04 // vmovups    ymm0, yword [rsi + 8*r9]
    LONG $0x107ca1c4; WORD $0xce4c; BYTE $0x20 // vmovups    ymm1, yword [rsi + 8*r9 + 32]
    LONG $0xd1c6fcc5; BYTE $0x88 // vshufps    ymm2, ymm0, ymm1, 136
    LONG $0x01fde3c4; WORD $0xd8d2 // vpermpd    ymm2, ymm2, 216
    LONG $0xc1c6fcc5; BYTE $0xdd // vshufps    ymm0, ymm0, ymm1, 221
    LONG $0x01fde3c4; WORD $0xd8c0 // vpermpd    ymm0, ymm0, 216
    LONG $0x107ca1c4; WORD $0x8f0c // vmovups    ymm1, yword [rdi + 4*r9]
    LONG $0x107ca1c4; WORD $0x8a1c // vmovups    ymm3, yword [rdx + 4*r9]
    LONG $0xe159ecc5             // vmulps    ymm4, ymm2, ymm1
    LONG $0xeb59fcc5             // vmulps    ymm5, ymm0, ymm3
    LONG $0xe55cdcc5             // vsubps    ymm4, ymm4, ymm5
    LONG $0x117ca1c4; WORD $0x8f24 // vmovups    yword [rdi + 4*r9], ymm4
    LONG $0xc159fcc5             // vmulps    ymm0, ymm0, ymm1
    LONG $0xcb59ecc5             // vmulps    ymm1, ymm2, ymm3
    LONG $0xc158fcc5             // vaddps    ymm0, ymm0, ymm1
    LONG $0x117ca1c4; WORD $0x8a04 // vmovups    yword [rdx + 4*r9], ymm0
    LONG $0x08c18349             // add    r9, 8
    WORD $0x394d; BYTE $0xcd     // cmp    r13, r9
	JNE LBB0_25
    LONG $0x102444f6; BYTE $0x07 // test    byte [rsp + 16], 7
	JE LBB0_28
LBB0_27:
    LONG $0x107ac1c4; WORD $0xca04 // vmovss    xmm0, dword [r10 + 8*rcx]
    LONG $0x107ac1c4; WORD $0xca4c; BYTE $0x04 // vmovss    xmm1, dword [r10 + 8*rcx + 4]
    LONG $0x107ac1c4; WORD $0x8b14 // vmovss    xmm2, dword [r11 + 4*rcx]
    LONG $0x107ac1c4; WORD $0x8f1c // vmovss    xmm3, dword [r15 + 4*rcx]
    LONG $0xe259fac5             // vmulss    xmm4, xmm0, xmm2
    LONG $0xeb59f2c5             // vmulss    xmm5, xmm1, xmm3
    LONG $0xe55cdac5             // vsubss    xmm4, xmm4, xmm5
    LONG $0x117ac1c4; WORD $0x8b24 // vmovss    dword [r11 + 4*rcx], xmm4
    LONG $0xca59f2c5             // vmulss    xmm1, xmm1, xmm2
    LONG $0xc359fac5             // vmulss    xmm0, xmm0, xmm3
    LONG $0xc058f2c5             // vaddss    xmm0, xmm1, xmm0
    LONG $0x117ac1c4; WORD $0x8f04 // vmovss    dword [r15 + 4*rcx], xmm0
    WORD $0xff48; BYTE $0xc1     // inc    rcx
    WORD $0x3948; BYTE $0xc1     // cmp    rcx, rax
	JL LBB0_27
	JMP LBB0_28
LBB0_8:
    LONG $0x24448b44; BYTE $0x04 // mov    r8d, dword [rsp + 4]
	JMP LBB0_27
LBB0_19:
    LONG $0x50c48348             // add    rsp, 80
    BYTE $0x5b                   // pop    rbx
    WORD $0x5c41                 // pop    r12
    WORD $0x5d41                 // pop    r13
    WORD $0x5e41                 // pop    r14
    WORD $0x5f41                 // pop    r15
    BYTE $0x5d                   // pop    rbp
    VZEROUPPER
    SUBQ $128, SP
    RET
