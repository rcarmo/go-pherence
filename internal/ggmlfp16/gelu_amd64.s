//go:build amd64

#include "textflag.h"

DATA LCDATA1<>+0x000(SB)/8, $0x007fffff7fffffff
DATA LCDATA1<>+0x008(SB)/8, $0xffffff9000001000
DATA LCDATA1<>+0x010(SB)/8, $0x0000001e38800000
DATA LCDATA1<>+0x018(SB)/8, $0x0000007e00007c00
DATA LCDATA1<>+0x020(SB)/8, $0x0000007d33000000
DATA LCDATA1<>+0x028(SB)/8, $0x0080000000000001
DATA LCDATA1<>+0x030(SB)/8, $0xc120000000008000
DATA LCDATA1<>+0x038(SB)/8, $0x0000000041200000
GLOBL LCDATA1<>(SB), 8, $64

TEXT ·_go_gelu_fp16_mul_avx2(SB), $168-40

    MOVQ dst+0(FP), DI
    MOVQ gate+8(FP), SI
    MOVQ up+16(FP), DX
    MOVQ n+24(FP), CX
    MOVQ table+32(FP), R8
    LEAQ LCDATA1<>(SB), R11
    ADDQ $168, SP

    WORD $0xf983; BYTE $0x08     // cmp    ecx, 8
	JL LBB0_4
    LONG $0xa8ec8148; WORD $0x0000; BYTE $0x00 // sub    rsp, 168
    WORD $0xc889                 // mov    eax, ecx
    LONG $0x187dc2c4; BYTE $0x03 // vbroadcastss    ymm0, dword 0[r11] /* [rip + .LCPI0_0] */
    QUAD $0x000080248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 128], ymm0
    LONG $0x187dc2c4; WORD $0x0443 // vbroadcastss    ymm0, dword 4[r11] /* [rip + .LCPI0_1] */
    LONG $0x4411fcc5; WORD $0x6024 // vmovups    yword [rsp + 96], ymm0
    LONG $0x187dc2c4; WORD $0x0843 // vbroadcastss    ymm0, dword 8[r11] /* [rip + .LCPI0_2] */
    LONG $0x4411fcc5; WORD $0x4024 // vmovups    yword [rsp + 64], ymm0
    LONG $0x187dc2c4; WORD $0x0c43 // vbroadcastss    ymm0, dword 12[r11] /* [rip + .LCPI0_3] */
    LONG $0x4411fcc5; WORD $0x2024 // vmovups    yword [rsp + 32], ymm0
    LONG $0x000008b9; BYTE $0x00 // mov    ecx, 8
    LONG $0x187dc2c4; WORD $0x1043 // vbroadcastss    ymm0, dword 16[r11] /* [rip + .LCPI0_4] */
    LONG $0x0411fcc5; BYTE $0x24 // vmovups    yword [rsp], ymm0
    LONG $0x587dc2c4; WORD $0x1473 // vpbroadcastd    ymm6, dword 20[r11] /* [rip + .LCPI0_5] */
    LONG $0x187dc2c4; WORD $0x187b // vbroadcastss    ymm7, dword 24[r11] /* [rip + .LCPI0_6] */
    LONG $0x587d42c4; WORD $0x1c43 // vpbroadcastd    ymm8, dword 28[r11] /* [rip + .LCPI0_7] */
    LONG $0x587d42c4; WORD $0x204b // vpbroadcastd    ymm9, dword 32[r11] /* [rip + .LCPI0_8] */
    LONG $0x587d42c4; WORD $0x2453 // vpbroadcastd    ymm10, dword 36[r11] /* [rip + .LCPI0_9] */
    LONG $0x587d42c4; WORD $0x285b // vpbroadcastd    ymm11, dword 40[r11] /* [rip + .LCPI0_10] */
    LONG $0x587d42c4; WORD $0x2c63 // vpbroadcastd    ymm12, dword 44[r11] /* [rip + .LCPI0_11] */
    LONG $0x587d42c4; WORD $0x306b // vpbroadcastd    ymm13, dword 48[r11] /* [rip + .LCPI0_12] */
    LONG $0xe4efd9c5             // vpxor    xmm4, xmm4, xmm4
LBB0_2:
    LONG $0x746f7ec5; WORD $0xe08e // vmovdqu    ymm14, yword [rsi + 4*rcx - 32]
    QUAD $0x00008024bcdb0dc5; BYTE $0x00 // vpand    ymm15, ymm14, yword [rsp + 128]
    LONG $0x727dc1c4; WORD $0x17d7 // vpsrld    ymm0, ymm15, 23
    LONG $0x4cdb8dc5; WORD $0x6024 // vpand    ymm1, ymm14, yword [rsp + 96]
    LONG $0x54fef5c5; WORD $0x4024 // vpaddd    ymm2, ymm1, yword [rsp + 64]
    LONG $0xd272e5c5; BYTE $0x17 // vpsrld    ymm3, ymm2, 23
    LONG $0x6cfefdc5; WORD $0x2024 // vpaddd    ymm5, ymm0, yword [rsp + 32]
    LONG $0xebfed5c5             // vpaddd    ymm5, ymm5, ymm3
    LONG $0xdc76e5c5             // vpcmpeqd    ymm3, ymm3, ymm4
    LONG $0xd272edc5; BYTE $0x0d // vpsrld    ymm2, ymm2, 13
    LONG $0xd2dbe5c5             // vpand    ymm2, ymm3, ymm2
    LONG $0xf572e5c5; BYTE $0x0a // vpslld    ymm3, ymm5, 10
    LONG $0xd2ebe5c5             // vpor    ymm2, ymm3, ymm2
    LONG $0x3f05e2c4; WORD $0x241c // vpmaxud    ymm3, ymm15, yword [rsp]
    LONG $0xee66d5c5             // vpcmpgtd    ymm5, ymm5, ymm6
    LONG $0x4a6de3c4; WORD $0x50d7 // vblendvps    ymm2, ymm2, ymm7, ymm5
    LONG $0xdb7685c5             // vpcmpeqd    ymm3, ymm15, ymm3
    LONG $0x3f05c2c4; BYTE $0xe9 // vpmaxud    ymm5, ymm15, ymm9
    LONG $0xed7685c5             // vpcmpeqd    ymm5, ymm15, ymm5
    LONG $0xf8fa2dc5             // vpsubd    ymm15, ymm10, ymm0
    LONG $0x472542c4; BYTE $0xff // vpsllvd    ymm15, ymm11, ymm15
    LONG $0xc9fe9dc5             // vpaddd    ymm1, ymm12, ymm1
    LONG $0xc9fe85c5             // vpaddd    ymm1, ymm15, ymm1
    LONG $0xc0fabdc5             // vpsubd    ymm0, ymm8, ymm0
    LONG $0x4575e2c4; BYTE $0xc0 // vpsrlvd    ymm0, ymm1, ymm0
    LONG $0xc0dbd5c5             // vpand    ymm0, ymm5, ymm0
    LONG $0x4a7de3c4; WORD $0x30c2 // vblendvps    ymm0, ymm0, ymm2, ymm3
    LONG $0x7275c1c4; WORD $0x10d6 // vpsrld    ymm1, ymm14, 16
    LONG $0xc9db95c5             // vpand    ymm1, ymm13, ymm1
    LONG $0xc156fcc5             // vorps    ymm0, ymm0, ymm1
    LONG $0xc9eff1c5             // vpxor    xmm1, xmm1, xmm1
    LONG $0xd276edc5             // vpcmpeqd    ymm2, ymm2, ymm2
    LONG $0x906dc2c4; WORD $0x400c // vpgatherdd    ymm1, dword [r8 + 2*ymm0], ymm2
    LONG $0x0e75e3c4; WORD $0xaac4 // vpblendw    ymm0, ymm1, ymm4, 170
    LONG $0x397de3c4; WORD $0x01c1 // vextracti128    xmm1, ymm0, 1
    LONG $0x2b79e2c4; BYTE $0xc1 // vpackusdw    xmm0, xmm0, xmm1
    LONG $0x137de2c4; BYTE $0xc0 // vcvtph2ps    ymm0, xmm0
    LONG $0x187dc2c4; WORD $0x344b // vbroadcastss    ymm1, dword 52[r11] /* [rip + .LCPI0_13] */
    LONG $0xc9c28cc5; BYTE $0x06 // vcmpnleps    ymm1, ymm14, ymm1
    LONG $0xc054f4c5             // vandps    ymm0, ymm1, ymm0
    LONG $0x187dc2c4; WORD $0x384b // vbroadcastss    ymm1, dword 56[r11] /* [rip + .LCPI0_14] */
    LONG $0xc274c1c4; WORD $0x02ce // vcmpleps    ymm1, ymm1, ymm14
    LONG $0x4a7dc3c4; WORD $0x10c6 // vblendvps    ymm0, ymm0, ymm14, ymm1
    LONG $0x4459fcc5; WORD $0xe08a // vmulps    ymm0, ymm0, yword [rdx + 4*rcx - 32]
    LONG $0x4411fcc5; WORD $0xe08f // vmovups    yword [rdi + 4*rcx - 32], ymm0
    LONG $0x08c18348             // add    rcx, 8
    WORD $0x3948; BYTE $0xc1     // cmp    rcx, rax
	JBE LBB0_2
    LONG $0xa8c48148; WORD $0x0000; BYTE $0x00 // add    rsp, 168
LBB0_4:
    VZEROUPPER
    SUBQ $168, SP
    RET
