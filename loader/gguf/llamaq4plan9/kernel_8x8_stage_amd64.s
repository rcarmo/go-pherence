//go:build amd64

#include "textflag.h"

DATA LCDATA1<>+0x000(SB)/8, $0x0f0f0f0f88888888
DATA LCDATA1<>+0x008(SB)/8, $0x0000000000000000
DATA LCDATA1<>+0x010(SB)/8, $0x0000000000000000
DATA LCDATA1<>+0x018(SB)/8, $0x0000000000000000
DATA LCDATA1<>+0x020(SB)/8, $0x0101010101010101
DATA LCDATA1<>+0x028(SB)/8, $0x0101010101010101
DATA LCDATA1<>+0x030(SB)/8, $0x0101010101010101
DATA LCDATA1<>+0x038(SB)/8, $0x0101010101010101
DATA LCDATA1<>+0x040(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA LCDATA1<>+0x048(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA LCDATA1<>+0x050(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA LCDATA1<>+0x058(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA LCDATA1<>+0x060(SB)/8, $0x000000000000010f
GLOBL LCDATA1<>(SB), 8, $104

TEXT ·_go_llama_q4_0_q8_0_8x8_stage(SB), $592-32

    MOVQ q4+0(FP), DI
    MOVQ q8+8(FP), SI
    MOVQ blocks+16(FP), DX
    MOVQ out+24(FP), CX
    ADDQ $592, SP
    PUSHQ BP
    LEAQ LCDATA1<>(SB), BP

    LONG $0x48ec8148; WORD $0x0002; BYTE $0x00 // sub    rsp, 584
    WORD $0xd285                 // test    edx, edx
	JLE LBB0_1
    WORD $0xd089                 // mov    eax, edx
    WORD $0x8941; BYTE $0xd0     // mov    r8d, edx
    WORD $0xe2c1; BYTE $0x07     // shl    edx, 7
    WORD $0x048d; BYTE $0xc2     // lea    eax, [rdx + 8*rax]
    LONG $0x04e0c149             // shl    r8, 4
    LONG $0xc0148d4b             // lea    rdx, [r8 + 8*r8]
    LONG $0xc057f8c5             // vxorps    xmm0, xmm0, xmm0
    LONG $0x4411fcc5; WORD $0x4024 // vmovups    yword [rsp + 64], ymm0
    LONG $0x187de2c4; WORD $0x0045 // vbroadcastss    ymm0, dword 0[rbp] /* [rip + .LCPI0_0] */
    QUAD $0x000140248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 320], ymm0
    WORD $0x3145; BYTE $0xc0     // xor    r8d, r8d
    LONG $0x187de2c4; WORD $0x0445 // vbroadcastss    ymm0, dword 4[rbp] /* [rip + .LCPI0_1] */
    QUAD $0x000120248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 288], ymm0
    LONG $0xc057f8c5             // vxorps    xmm0, xmm0, xmm0
    LONG $0x4411fcc5; WORD $0x2024 // vmovups    yword [rsp + 32], ymm0
    LONG $0xef2941c4; BYTE $0xd2 // vpxor    xmm10, xmm10, xmm10
    LONG $0x573041c4; BYTE $0xc9 // vxorps    xmm9, xmm9, xmm9
    LONG $0xef3941c4; BYTE $0xc0 // vpxor    xmm8, xmm8, xmm8
    LONG $0xff57c0c5             // vxorps    xmm7, xmm7, xmm7
    LONG $0xe457d8c5             // vxorps    xmm4, xmm4, xmm4
LBB0_3:
    QUAD $0x0001c024947f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 448], ymm10
    QUAD $0x0001e0248c117cc5; BYTE $0x00 // vmovups    yword [rsp + 480], ymm9
    QUAD $0x00020024847f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 512], ymm8
    QUAD $0x00022024bc11fcc5; BYTE $0x00 // vmovups    yword [rsp + 544], ymm7
    QUAD $0x0000e024a411fcc5; BYTE $0x00 // vmovups    yword [rsp + 224], ymm4
    QUAD $0x000100248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 256], ymm0
    QUAD $0x000140248c6f7ec5; BYTE $0x00 // vmovdqu    ymm9, yword [rsp + 320]
    LONG $0xef35a1c4; WORD $0x0744; BYTE $0x10 // vpxor    ymm0, ymm9, yword [rdi + r8 + 16]
    LONG $0xef35a1c4; WORD $0x074c; BYTE $0x30 // vpxor    ymm1, ymm9, yword [rdi + r8 + 48]
    LONG $0x387de3c4; WORD $0x01d1 // vinserti128    ymm2, ymm0, xmm1, 1
    QUAD $0x00008024947ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 128], ymm2
    LONG $0x467de3c4; WORD $0x31c1 // vperm2i128    ymm0, ymm0, ymm1, 49
    LONG $0x447ffec5; WORD $0x6024 // vmovdqu    yword [rsp + 96], ymm0
    QUAD $0x00012024b46ffec5; BYTE $0x00 // vmovdqu    ymm6, yword [rsp + 288]
    LONG $0xcedbedc5             // vpand    ymm1, ymm2, ymm6
    QUAD $0x0000a0248c7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 160], ymm1
    LONG $0xdedbfdc5             // vpand    ymm3, ymm0, ymm6
    LONG $0x1c7ffec5; BYTE $0x24 // vmovdqu    yword [rsp], ymm3
    LONG $0x566ffec5; BYTE $0x08 // vmovdqu    ymm2, yword [rsi + 8]
    LONG $0xc270f9c5; BYTE $0xa0 // vpshufd    xmm0, xmm2, 160
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0xc970fdc5; BYTE $0x88 // vpshufd    ymm1, ymm1, 136
    LONG $0xef2941c4; BYTE $0xd2 // vpxor    xmm10, xmm10, xmm10
    LONG $0xdb70fdc5; BYTE $0x88 // vpshufd    ymm3, ymm3, 136
    LONG $0xef3941c4; BYTE $0xc0 // vpxor    xmm8, xmm8, xmm8
    LONG $0x507562c4; BYTE $0xd0                                 // {vex}    vpdpbusd    ymm10, ymm1, ymm0
    LONG $0xfa70fdc5; BYTE $0xa0 // vpshufd    ymm7, ymm2, 160
    LONG $0x00fde3c4; WORD $0xeeff // vpermq    ymm7, ymm7, 238
    LONG $0xef1941c4; BYTE $0xe4 // vpxor    xmm12, xmm12, xmm12
    LONG $0x506562c4; BYTE $0xc0                                 // {vex}    vpdpbusd    ymm8, ymm3, ymm0
    LONG $0x507562c4; BYTE $0xe7                                 // {vex}    vpdpbusd    ymm12, ymm1, ymm7
    LONG $0xef2141c4; BYTE $0xdb // vpxor    xmm11, xmm11, xmm11
    LONG $0x646ffec5; WORD $0x0806 // vmovdqu    ymm4, yword [rsi + rax + 8]
    LONG $0x506562c4; BYTE $0xdf                                 // {vex}    vpdpbusd    ymm11, ymm3, ymm7
    LONG $0xc470f9c5; BYTE $0xa0 // vpshufd    xmm0, xmm4, 160
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0xef0941c4; BYTE $0xf6 // vpxor    xmm14, xmm14, xmm14
    LONG $0x507562c4; BYTE $0xf0                                 // {vex}    vpdpbusd    ymm14, ymm1, ymm0
    LONG $0xef0141c4; BYTE $0xff // vpxor    xmm15, xmm15, xmm15
    LONG $0xfc70fdc5; BYTE $0xa0 // vpshufd    ymm7, ymm4, 160
    LONG $0x00fde3c4; WORD $0xeeff // vpermq    ymm7, ymm7, 238
    LONG $0x506562c4; BYTE $0xf8                                 // {vex}    vpdpbusd    ymm15, ymm3, ymm0
    LONG $0xedefd1c5             // vpxor    xmm5, xmm5, xmm5
    LONG $0x5075e2c4; BYTE $0xef                                 // {vex}    vpdpbusd    ymm5, ymm1, ymm7
    LONG $0xef1141c4; BYTE $0xed // vpxor    xmm13, xmm13, xmm13
    LONG $0x506562c4; BYTE $0xef                                 // {vex}    vpdpbusd    ymm13, ymm3, ymm7
    LONG $0x7e6ffec5; BYTE $0x28 // vmovdqu    ymm7, yword [rsi + 40]
    LONG $0xda6ffdc5             // vmovdqa    ymm3, ymm2
    QUAD $0x0000c024947ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 192], ymm2
    LONG $0xc370f9c5; BYTE $0xf5 // vpshufd    xmm0, xmm3, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    QUAD $0x0000a0248c70fdc5; WORD $0xdd00 // vpshufd    ymm1, yword [rsp + 160], 221
    LONG $0x507562c4; BYTE $0xd0                                 // {vex}    vpdpbusd    ymm10, ymm1, ymm0
    LONG $0x1470fdc5; WORD $0xdd24 // vpshufd    ymm2, yword [rsp], 221
    LONG $0x506d62c4; BYTE $0xc0                                 // {vex}    vpdpbusd    ymm8, ymm2, ymm0
    LONG $0xc370fdc5; BYTE $0xf5 // vpshufd    ymm0, ymm3, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x507562c4; BYTE $0xe0                                 // {vex}    vpdpbusd    ymm12, ymm1, ymm0
    LONG $0x506d62c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm11, ymm2, ymm0
    QUAD $0x0001a024a47ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 416], ymm4
    LONG $0xc470f9c5; BYTE $0xf5 // vpshufd    xmm0, xmm4, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0x507562c4; BYTE $0xf0                                 // {vex}    vpdpbusd    ymm14, ymm1, ymm0
    LONG $0x506d62c4; BYTE $0xf8                                 // {vex}    vpdpbusd    ymm15, ymm2, ymm0
    LONG $0xc470fdc5; BYTE $0xf5 // vpshufd    ymm0, ymm4, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x5075e2c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm5, ymm1, ymm0
    LONG $0xef35a1c4; WORD $0x074c; BYTE $0x50 // vpxor    ymm1, ymm9, yword [rdi + r8 + 80]
    LONG $0x506d62c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm13, ymm2, ymm0
    LONG $0xef35a1c4; WORD $0x0744; BYTE $0x70 // vpxor    ymm0, ymm9, yword [rdi + r8 + 112]
    LONG $0x3875e3c4; WORD $0x01d0 // vinserti128    ymm2, ymm1, xmm0, 1
    LONG $0x147ffec5; BYTE $0x24 // vmovdqu    yword [rsp], ymm2
    LONG $0x4675e3c4; WORD $0x31d8 // vperm2i128    ymm3, ymm1, ymm0, 49
    QUAD $0x000160249c7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 352], ymm3
    LONG $0xc770f9c5; BYTE $0xa0 // vpshufd    xmm0, xmm7, 160
    LONG $0x00fd63c4; WORD $0x44c8 // vpermq    ymm9, ymm0, 68
    LONG $0xcedbedc5             // vpand    ymm1, ymm2, ymm6
    LONG $0xc170fdc5; BYTE $0x88 // vpshufd    ymm0, ymm1, 136
    LONG $0x507d42c4; BYTE $0xd1                                 // {vex}    vpdpbusd    ymm10, ymm0, ymm9
    LONG $0xd6dbe5c5             // vpand    ymm2, ymm3, ymm6
    LONG $0xda70fdc5; BYTE $0x88 // vpshufd    ymm3, ymm2, 136
    LONG $0x506542c4; BYTE $0xc1                                 // {vex}    vpdpbusd    ymm8, ymm3, ymm9
    LONG $0xcf707dc5; BYTE $0xa0 // vpshufd    ymm9, ymm7, 160
    LONG $0x00fd43c4; WORD $0xeec9 // vpermq    ymm9, ymm9, 238
    LONG $0x507d42c4; BYTE $0xe1                                 // {vex}    vpdpbusd    ymm12, ymm0, ymm9
    LONG $0x506542c4; BYTE $0xd9                                 // {vex}    vpdpbusd    ymm11, ymm3, ymm9
    LONG $0x646ffec5; WORD $0x2806 // vmovdqu    ymm4, yword [rsi + rax + 40]
    LONG $0xcc7079c5; BYTE $0xa0 // vpshufd    xmm9, xmm4, 160
    LONG $0x00fd43c4; WORD $0x44c9 // vpermq    ymm9, ymm9, 68
    LONG $0x507d42c4; BYTE $0xf1                                 // {vex}    vpdpbusd    ymm14, ymm0, ymm9
    LONG $0x506542c4; BYTE $0xf9                                 // {vex}    vpdpbusd    ymm15, ymm3, ymm9
    LONG $0xcc707dc5; BYTE $0xa0 // vpshufd    ymm9, ymm4, 160
    LONG $0x00fd43c4; WORD $0xeec9 // vpermq    ymm9, ymm9, 238
    LONG $0x507dc2c4; BYTE $0xe9                                 // {vex}    vpdpbusd    ymm5, ymm0, ymm9
    LONG $0x506542c4; BYTE $0xe9                                 // {vex}    vpdpbusd    ymm13, ymm3, ymm9
    QUAD $0x00018024bc7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 384], ymm7
    LONG $0xc770f9c5; BYTE $0xf5 // vpshufd    xmm0, xmm7, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0xc970fdc5; BYTE $0xdd // vpshufd    ymm1, ymm1, 221
    LONG $0x507562c4; BYTE $0xd0                                 // {vex}    vpdpbusd    ymm10, ymm1, ymm0
    LONG $0xd270fdc5; BYTE $0xdd // vpshufd    ymm2, ymm2, 221
    LONG $0x506d62c4; BYTE $0xc0                                 // {vex}    vpdpbusd    ymm8, ymm2, ymm0
    LONG $0xc770fdc5; BYTE $0xf5 // vpshufd    ymm0, ymm7, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x507562c4; BYTE $0xe0                                 // {vex}    vpdpbusd    ymm12, ymm1, ymm0
    LONG $0x506d62c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm11, ymm2, ymm0
    QUAD $0x0000a024a47ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 160], ymm4
    LONG $0xc470f9c5; BYTE $0xf5 // vpshufd    xmm0, xmm4, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0x507562c4; BYTE $0xf0                                 // {vex}    vpdpbusd    ymm14, ymm1, ymm0
    LONG $0x506d62c4; BYTE $0xf8                                 // {vex}    vpdpbusd    ymm15, ymm2, ymm0
    LONG $0xc470fdc5; BYTE $0xf5 // vpshufd    ymm0, ymm4, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x5075e2c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm5, ymm1, ymm0
    LONG $0x506d62c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm13, ymm2, ymm0
    QUAD $0x00008024846ffec5; BYTE $0x00 // vmovdqu    ymm0, yword [rsp + 128]
    LONG $0xd071fdc5; BYTE $0x04 // vpsrlw    ymm0, ymm0, 4
    LONG $0x4c6ffec5; WORD $0x6024 // vmovdqu    ymm1, yword [rsp + 96]
    LONG $0xd171f5c5; BYTE $0x04 // vpsrlw    ymm1, ymm1, 4
    LONG $0x787de2c4; WORD $0x607d // vpbroadcastb    ymm7, byte 96[rbp] /* [rip + .LCPI0_4] */
    LONG $0xd7dbfdc5             // vpand    ymm2, ymm0, ymm7
    LONG $0xdfdbf5c5             // vpand    ymm3, ymm1, ymm7
    LONG $0x666ffec5; BYTE $0x48 // vmovdqu    ymm4, yword [rsi + 72]
    LONG $0xc470f9c5; BYTE $0xa0 // vpshufd    xmm0, xmm4, 160
    LONG $0x00fd63c4; WORD $0x44c8 // vpermq    ymm9, ymm0, 68
    LONG $0xca70fdc5; BYTE $0x88 // vpshufd    ymm1, ymm2, 136
    LONG $0x507542c4; BYTE $0xd1                                 // {vex}    vpdpbusd    ymm10, ymm1, ymm9
    LONG $0xc370fdc5; BYTE $0x88 // vpshufd    ymm0, ymm3, 136
    LONG $0x507d42c4; BYTE $0xc1                                 // {vex}    vpdpbusd    ymm8, ymm0, ymm9
    LONG $0xcc707dc5; BYTE $0xa0 // vpshufd    ymm9, ymm4, 160
    LONG $0x00fd43c4; WORD $0xeec9 // vpermq    ymm9, ymm9, 238
    LONG $0x507542c4; BYTE $0xe1                                 // {vex}    vpdpbusd    ymm12, ymm1, ymm9
    LONG $0x507d42c4; BYTE $0xd9                                 // {vex}    vpdpbusd    ymm11, ymm0, ymm9
    LONG $0x4c6f7ec5; WORD $0x4806 // vmovdqu    ymm9, yword [rsi + rax + 72]
    LONG $0x7079c1c4; WORD $0xa0f1 // vpshufd    xmm6, xmm9, 160
    LONG $0x00fde3c4; WORD $0x44f6 // vpermq    ymm6, ymm6, 68
    LONG $0x507562c4; BYTE $0xf6                                 // {vex}    vpdpbusd    ymm14, ymm1, ymm6
    LONG $0x507d62c4; BYTE $0xfe                                 // {vex}    vpdpbusd    ymm15, ymm0, ymm6
    LONG $0x707dc1c4; WORD $0xa0f1 // vpshufd    ymm6, ymm9, 160
    LONG $0x00fde3c4; WORD $0xeef6 // vpermq    ymm6, ymm6, 238
    LONG $0x5075e2c4; BYTE $0xee                                 // {vex}    vpdpbusd    ymm5, ymm1, ymm6
    LONG $0x507d62c4; BYTE $0xee                                 // {vex}    vpdpbusd    ymm13, ymm0, ymm6
    LONG $0x647ffec5; WORD $0x6024 // vmovdqu    yword [rsp + 96], ymm4
    LONG $0xc470f9c5; BYTE $0xf5 // vpshufd    xmm0, xmm4, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0xca70fdc5; BYTE $0xdd // vpshufd    ymm1, ymm2, 221
    LONG $0x507562c4; BYTE $0xd0                                 // {vex}    vpdpbusd    ymm10, ymm1, ymm0
    LONG $0xd370fdc5; BYTE $0xdd // vpshufd    ymm2, ymm3, 221
    LONG $0x506d62c4; BYTE $0xc0                                 // {vex}    vpdpbusd    ymm8, ymm2, ymm0
    LONG $0xc470fdc5; BYTE $0xf5 // vpshufd    ymm0, ymm4, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x507562c4; BYTE $0xe0                                 // {vex}    vpdpbusd    ymm12, ymm1, ymm0
    LONG $0x506d62c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm11, ymm2, ymm0
    QUAD $0x000080248c7f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 128], ymm9
    LONG $0x7079c1c4; WORD $0xf5c1 // vpshufd    xmm0, xmm9, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0x507562c4; BYTE $0xf0                                 // {vex}    vpdpbusd    ymm14, ymm1, ymm0
    LONG $0x506d62c4; BYTE $0xf8                                 // {vex}    vpdpbusd    ymm15, ymm2, ymm0
    LONG $0x707dc1c4; WORD $0xf5c1 // vpshufd    ymm0, ymm9, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x5075e2c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm5, ymm1, ymm0
    LONG $0xe56ffdc5             // vmovdqa    ymm4, ymm5
    LONG $0x506d62c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm13, ymm2, ymm0
    LONG $0x046ffec5; BYTE $0x24 // vmovdqu    ymm0, yword [rsp]
    LONG $0xd071fdc5; BYTE $0x04 // vpsrlw    ymm0, ymm0, 4
    QUAD $0x000160248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 352]
    LONG $0xd171edc5; BYTE $0x04 // vpsrlw    ymm2, ymm1, 4
    LONG $0xcfdbfdc5             // vpand    ymm1, ymm0, ymm7
    LONG $0xd7dbedc5             // vpand    ymm2, ymm2, ymm7
    LONG $0x147ffec5; BYTE $0x24 // vmovdqu    yword [rsp], ymm2
    LONG $0x7e6ffec5; BYTE $0x68 // vmovdqu    ymm7, yword [rsi + 104]
    LONG $0xc770f9c5; BYTE $0xa0 // vpshufd    xmm0, xmm7, 160
    LONG $0x00fde3c4; WORD $0x44f0 // vpermq    ymm6, ymm0, 68
    LONG $0xc170fdc5; BYTE $0x88 // vpshufd    ymm0, ymm1, 136
    LONG $0x507d62c4; BYTE $0xd6                                 // {vex}    vpdpbusd    ymm10, ymm0, ymm6
    LONG $0xda70fdc5; BYTE $0x88 // vpshufd    ymm3, ymm2, 136
    LONG $0x506562c4; BYTE $0xc6                                 // {vex}    vpdpbusd    ymm8, ymm3, ymm6
    LONG $0xf770fdc5; BYTE $0xa0 // vpshufd    ymm6, ymm7, 160
    LONG $0x00fde3c4; WORD $0xeef6 // vpermq    ymm6, ymm6, 238
    LONG $0x507d62c4; BYTE $0xe6                                 // {vex}    vpdpbusd    ymm12, ymm0, ymm6
    LONG $0x506562c4; BYTE $0xde                                 // {vex}    vpdpbusd    ymm11, ymm3, ymm6
    LONG $0x746ffec5; WORD $0x6806 // vmovdqu    ymm6, yword [rsi + rax + 104]
    LONG $0xce7079c5; BYTE $0xa0 // vpshufd    xmm9, xmm6, 160
    LONG $0x00fd43c4; WORD $0x44c9 // vpermq    ymm9, ymm9, 68
    LONG $0x507d42c4; BYTE $0xf1                                 // {vex}    vpdpbusd    ymm14, ymm0, ymm9
    LONG $0x506542c4; BYTE $0xf9                                 // {vex}    vpdpbusd    ymm15, ymm3, ymm9
    LONG $0xce707dc5; BYTE $0xa0 // vpshufd    ymm9, ymm6, 160
    LONG $0x00fd43c4; WORD $0xeec9 // vpermq    ymm9, ymm9, 238
    LONG $0x507dc2c4; BYTE $0xe1                                 // {vex}    vpdpbusd    ymm4, ymm0, ymm9
    LONG $0x506542c4; BYTE $0xe9                                 // {vex}    vpdpbusd    ymm13, ymm3, ymm9
    LONG $0xc0eff9c5             // vpxor    xmm0, xmm0, xmm0
    LONG $0x787de2c4; WORD $0x6155 // vpbroadcastb    ymm2, byte 97[rbp] /* [rip + .LCPI0_5] */
    QUAD $0x00c02484506de2c4; WORD $0x0000                       // {vex}    vpdpbusd    ymm0, ymm2, YMMWORD PTR [rsp + 192]
    QUAD $0x01802484506de2c4; WORD $0x0000                       // {vex}    vpdpbusd    ymm0, ymm2, YMMWORD PTR [rsp + 384]
    LONG $0x506de2c4; WORD $0x2444; BYTE $0x60                   // {vex}    vpdpbusd    ymm0, ymm2, YMMWORD PTR [rsp + 96]
    LONG $0xdf70f9c5; BYTE $0xf5 // vpshufd    xmm3, xmm7, 245
    LONG $0x00fde3c4; WORD $0x44db // vpermq    ymm3, ymm3, 68
    LONG $0xe970fdc5; BYTE $0xdd // vpshufd    ymm5, ymm1, 221
    LONG $0x505562c4; BYTE $0xd3                                 // {vex}    vpdpbusd    ymm10, ymm5, ymm3
    LONG $0x0c707dc5; WORD $0xdd24 // vpshufd    ymm9, yword [rsp], 221
    LONG $0x503562c4; BYTE $0xc3                                 // {vex}    vpdpbusd    ymm8, ymm9, ymm3
    LONG $0xcf70fdc5; BYTE $0xf5 // vpshufd    ymm1, ymm7, 245
    LONG $0x00fde3c4; WORD $0xeec9 // vpermq    ymm1, ymm1, 238
    LONG $0x505562c4; BYTE $0xe1                                 // {vex}    vpdpbusd    ymm12, ymm5, ymm1
    LONG $0x503562c4; BYTE $0xd9                                 // {vex}    vpdpbusd    ymm11, ymm9, ymm1
    LONG $0x506de2c4; BYTE $0xc7                                 // {vex}    vpdpbusd    ymm0, ymm2, ymm7
    LONG $0xce70f9c5; BYTE $0xf5 // vpshufd    xmm1, xmm6, 245
    LONG $0x00fde3c4; WORD $0x44c9 // vpermq    ymm1, ymm1, 68
    LONG $0x505562c4; BYTE $0xf1                                 // {vex}    vpdpbusd    ymm14, ymm5, ymm1
    LONG $0x503562c4; BYTE $0xf9                                 // {vex}    vpdpbusd    ymm15, ymm9, ymm1
    LONG $0x137da2c4; WORD $0x070c // vcvtph2ps    ymm1, oword [rdi + r8]
    LONG $0x027de2c4; BYTE $0xc0 // vphaddd    ymm0, ymm0, ymm0
    LONG $0xd670fdc5; BYTE $0xf5 // vpshufd    ymm2, ymm6, 245
    LONG $0x00fde3c4; WORD $0xeeda // vpermq    ymm3, ymm2, 238
    LONG $0x5055e2c4; BYTE $0xe3                                 // {vex}    vpdpbusd    ymm4, ymm5, ymm3
    QUAD $0x0000c024a47ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 192], ymm4
    LONG $0x1379e2c4; BYTE $0x16 // vcvtph2ps    xmm2, qword [rsi]
    LONG $0x503562c4; BYTE $0xeb                                 // {vex}    vpdpbusd    ymm13, ymm9, ymm3
    QUAD $0x0001e0248c107cc5; BYTE $0x00 // vmovups    ymm9, yword [rsp + 480]
    LONG $0xf072fdc5; BYTE $0x03 // vpslld    ymm0, ymm0, 3
    LONG $0x587de2c4; BYTE $0xd8 // vpbroadcastd    ymm3, xmm0
    LONG $0x6c2dc1c4; BYTE $0xe8 // vpunpcklqdq    ymm5, ymm10, ymm8
    LONG $0xdbfad5c5             // vpsubd    ymm3, ymm5, ymm3
    LONG $0x187de2c4; BYTE $0xea // vbroadcastss    ymm5, xmm2
    LONG $0xe959d4c5             // vmulps    ymm5, ymm5, ymm1
    LONG $0xdb5bfcc5             // vcvtdq2ps    ymm3, ymm3
    LONG $0x7c10fcc5; WORD $0x4024 // vmovups    ymm7, yword [rsp + 64]
    LONG $0xb865e2c4; BYTE $0xfd // vfmadd231ps    ymm7, ymm3, ymm5
    LONG $0x7c11fcc5; WORD $0x4024 // vmovups    yword [rsp + 64], ymm7
    QUAD $0x00022024bc10fcc5; BYTE $0x00 // vmovups    ymm7, yword [rsp + 544]
    LONG $0x6d2dc1c4; BYTE $0xd8 // vpunpckhqdq    ymm3, ymm10, ymm8
    QUAD $0x0001c02494107cc5; BYTE $0x00 // vmovups    ymm10, yword [rsp + 448]
    LONG $0xe070f9c5; BYTE $0x55 // vpshufd    xmm4, xmm0, 85
    LONG $0x587de2c4; BYTE $0xe4 // vpbroadcastd    ymm4, xmm4
    LONG $0xdcfae5c5             // vpsubd    ymm3, ymm3, ymm4
    LONG $0xe216fac5             // vmovshdup    xmm4, xmm2
    LONG $0x187de2c4; BYTE $0xe4 // vbroadcastss    ymm4, xmm4
    LONG $0xdb5bfcc5             // vcvtdq2ps    ymm3, ymm3
    LONG $0xe159dcc5             // vmulps    ymm4, ymm4, ymm1
    LONG $0x6c10fcc5; WORD $0x2024 // vmovups    ymm5, yword [rsp + 32]
    LONG $0xb865e2c4; BYTE $0xec // vfmadd231ps    ymm5, ymm3, ymm4
    LONG $0x6c11fcc5; WORD $0x2024 // vmovups    yword [rsp + 32], ymm5
    LONG $0x397de3c4; WORD $0x01c3 // vextracti128    xmm3, ymm0, 1
    LONG $0x6c1dc1c4; BYTE $0xe3 // vpunpcklqdq    ymm4, ymm12, ymm11
    LONG $0x587de2c4; BYTE $0xdb // vpbroadcastd    ymm3, xmm3
    LONG $0xdbfaddc5             // vpsubd    ymm3, ymm4, ymm3
    LONG $0xe2c6e9c5; BYTE $0x01 // vshufpd    xmm4, xmm2, xmm2, 1
    LONG $0xdb5bfcc5             // vcvtdq2ps    ymm3, ymm3
    LONG $0x187de2c4; BYTE $0xe4 // vbroadcastss    ymm4, xmm4
    LONG $0xe159dcc5             // vmulps    ymm4, ymm4, ymm1
    LONG $0xb86562c4; BYTE $0xd4 // vfmadd231ps    ymm10, ymm3, ymm4
    LONG $0xdb57e0c5             // vxorps    xmm3, xmm3, xmm3
    LONG $0x787de2c4; WORD $0x6165 // vpbroadcastb    ymm4, byte 97[rbp] /* [rip + .LCPI0_5] */
    QUAD $0x01a0249c505de2c4; WORD $0x0000                       // {vex}    vpdpbusd    ymm3, ymm4, YMMWORD PTR [rsp + 416]
    QUAD $0x00a0249c505de2c4; WORD $0x0000                       // {vex}    vpdpbusd    ymm3, ymm4, YMMWORD PTR [rsp + 160]
    QUAD $0x0080249c505de2c4; WORD $0x0000                       // {vex}    vpdpbusd    ymm3, ymm4, YMMWORD PTR [rsp + 128]
    LONG $0x505de2c4; BYTE $0xde                                 // {vex}    vpdpbusd    ymm3, ymm4, ymm6
    LONG $0x6d1dc1c4; BYTE $0xe3 // vpunpckhqdq    ymm4, ymm12, ymm11
    QUAD $0x0002002484107cc5; BYTE $0x00 // vmovups    ymm8, yword [rsp + 512]
    LONG $0xc070fdc5; BYTE $0x55 // vpshufd    ymm0, ymm0, 85
    LONG $0x00fde3c4; WORD $0xaac0 // vpermq    ymm0, ymm0, 170
    LONG $0xc0faddc5             // vpsubd    ymm0, ymm4, ymm0
    LONG $0xd2c6e8c5; BYTE $0xff // vshufps    xmm2, xmm2, xmm2, 255
    LONG $0x187de2c4; BYTE $0xd2 // vbroadcastss    ymm2, xmm2
    LONG $0xd159ecc5             // vmulps    ymm2, ymm2, ymm1
    LONG $0xc05bfcc5             // vcvtdq2ps    ymm0, ymm0
    LONG $0x0265e2c4; BYTE $0xdb // vphaddd    ymm3, ymm3, ymm3
    LONG $0xb87d62c4; BYTE $0xca // vfmadd231ps    ymm9, ymm0, ymm2
    LONG $0x1379e2c4; WORD $0x0604 // vcvtph2ps    xmm0, qword [rsi + rax]
    LONG $0xf372edc5; BYTE $0x03 // vpslld    ymm2, ymm3, 3
    LONG $0x6c0dc1c4; BYTE $0xdf // vpunpcklqdq    ymm3, ymm14, ymm15
    LONG $0x587de2c4; BYTE $0xe2 // vpbroadcastd    ymm4, xmm2
    LONG $0xdcfae5c5             // vpsubd    ymm3, ymm3, ymm4
    LONG $0x187de2c4; BYTE $0xe0 // vbroadcastss    ymm4, xmm0
    LONG $0xdb5bfcc5             // vcvtdq2ps    ymm3, ymm3
    LONG $0xe159dcc5             // vmulps    ymm4, ymm4, ymm1
    LONG $0xb86562c4; BYTE $0xc4 // vfmadd231ps    ymm8, ymm3, ymm4
    LONG $0x6d0dc1c4; BYTE $0xdf // vpunpckhqdq    ymm3, ymm14, ymm15
    LONG $0xe270f9c5; BYTE $0x55 // vpshufd    xmm4, xmm2, 85
    LONG $0x587de2c4; BYTE $0xe4 // vpbroadcastd    ymm4, xmm4
    LONG $0xdcfae5c5             // vpsubd    ymm3, ymm3, ymm4
    LONG $0xe016fac5             // vmovshdup    xmm4, xmm0
    LONG $0xdb5bfcc5             // vcvtdq2ps    ymm3, ymm3
    LONG $0x187de2c4; BYTE $0xe4 // vbroadcastss    ymm4, xmm4
    LONG $0xe159dcc5             // vmulps    ymm4, ymm4, ymm1
    LONG $0xb865e2c4; BYTE $0xfc // vfmadd231ps    ymm7, ymm3, ymm4
    QUAD $0x0000c024ac6ffec5; BYTE $0x00 // vmovdqu    ymm5, yword [rsp + 192]
    LONG $0x6c55c1c4; BYTE $0xdd // vpunpcklqdq    ymm3, ymm5, ymm13
    LONG $0x397de3c4; WORD $0x01d4 // vextracti128    xmm4, ymm2, 1
    LONG $0x587de2c4; BYTE $0xe4 // vpbroadcastd    ymm4, xmm4
    LONG $0xdcfae5c5             // vpsubd    ymm3, ymm3, ymm4
    LONG $0xdb5bfcc5             // vcvtdq2ps    ymm3, ymm3
    LONG $0xe0c6f9c5; BYTE $0x01 // vshufpd    xmm4, xmm0, xmm0, 1
    LONG $0x187de2c4; BYTE $0xe4 // vbroadcastss    ymm4, xmm4
    LONG $0xe159dcc5             // vmulps    ymm4, ymm4, ymm1
    QUAD $0x0000e024b410fcc5; BYTE $0x00 // vmovups    ymm6, yword [rsp + 224]
    LONG $0xb865e2c4; BYTE $0xf4 // vfmadd231ps    ymm6, ymm3, ymm4
    QUAD $0x0000e024b411fcc5; BYTE $0x00 // vmovups    yword [rsp + 224], ymm6
    QUAD $0x0000e024a410fcc5; BYTE $0x00 // vmovups    ymm4, yword [rsp + 224]
    LONG $0x6d55c1c4; BYTE $0xdd // vpunpckhqdq    ymm3, ymm5, ymm13
    LONG $0xd270fdc5; BYTE $0x55 // vpshufd    ymm2, ymm2, 85
    LONG $0x00fde3c4; WORD $0xaad2 // vpermq    ymm2, ymm2, 170
    LONG $0xd2fae5c5             // vpsubd    ymm2, ymm3, ymm2
    LONG $0xc0c6f8c5; BYTE $0xff // vshufps    xmm0, xmm0, xmm0, 255
    LONG $0x187de2c4; BYTE $0xc0 // vbroadcastss    ymm0, xmm0
    LONG $0xc159fcc5             // vmulps    ymm0, ymm0, ymm1
    LONG $0xca5bfcc5             // vcvtdq2ps    ymm1, ymm2
    QUAD $0x000100249410fcc5; BYTE $0x00 // vmovups    ymm2, yword [rsp + 256]
    LONG $0xb875e2c4; BYTE $0xd0 // vfmadd231ps    ymm2, ymm1, ymm0
    QUAD $0x000100249411fcc5; BYTE $0x00 // vmovups    yword [rsp + 256], ymm2
    QUAD $0x000100248410fcc5; BYTE $0x00 // vmovups    ymm0, yword [rsp + 256]
    LONG $0x88c68148; WORD $0x0000; BYTE $0x00 // add    rsi, 136
    LONG $0x90c08149; WORD $0x0000; BYTE $0x00 // add    r8, 144
    WORD $0x394c; BYTE $0xc2     // cmp    rdx, r8
	JNE LBB0_3
	JMP LBB0_4
LBB0_1:
    LONG $0xc057f8c5             // vxorps    xmm0, xmm0, xmm0
    LONG $0xe457d8c5             // vxorps    xmm4, xmm4, xmm4
    LONG $0xff57c0c5             // vxorps    xmm7, xmm7, xmm7
    LONG $0xef3941c4; BYTE $0xc0 // vpxor    xmm8, xmm8, xmm8
    LONG $0x573041c4; BYTE $0xc9 // vxorps    xmm9, xmm9, xmm9
    LONG $0xef2941c4; BYTE $0xd2 // vpxor    xmm10, xmm10, xmm10
    LONG $0xc957f0c5             // vxorps    xmm1, xmm1, xmm1
    LONG $0x4c11fcc5; WORD $0x2024 // vmovups    yword [rsp + 32], ymm1
    LONG $0x4c11fcc5; WORD $0x4024 // vmovups    yword [rsp + 64], ymm1
LBB0_4:
    LONG $0x4c10fcc5; WORD $0x4024 // vmovups    ymm1, yword [rsp + 64]
    LONG $0x0911fcc5             // vmovups    yword [rcx], ymm1
    LONG $0x4c10fcc5; WORD $0x2024 // vmovups    ymm1, yword [rsp + 32]
    LONG $0x4911fcc5; BYTE $0x20 // vmovups    yword [rcx + 32], ymm1
    LONG $0x517f7ec5; BYTE $0x40 // vmovdqu    yword [rcx + 64], ymm10
    LONG $0x49117cc5; BYTE $0x60 // vmovups    yword [rcx + 96], ymm9
    QUAD $0x00000080817f7ec5     // vmovdqu    yword [rcx + 128], ymm8
    QUAD $0x000000a0b911fcc5     // vmovups    yword [rcx + 160], ymm7
    QUAD $0x000000c0a111fcc5     // vmovups    yword [rcx + 192], ymm4
    QUAD $0x000000e08111fcc5     // vmovups    yword [rcx + 224], ymm0
    LONG $0x48c48148; WORD $0x0002; BYTE $0x00 // add    rsp, 584
    VZEROUPPER
    POPQ BP
    SUBQ $592, SP
    RET
