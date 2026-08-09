//go:build amd64

#include "textflag.h"

TEXT ·_go_llama_q4_0_q8_0_8x16_pair_direct(SB), $824-48

    MOVQ q4+0(FP), DI
    MOVQ q8+8(FP), SI
    MOVQ blocks+16(FP), DX
    MOVQ out+24(FP), CX
    MOVQ stride+32(FP), R8
    MOVQ constants+40(FP), R9
    ADDQ $824, SP

    BYTE $0x55                   // push    rbp
    WORD $0x5741                 // push    r15
    WORD $0x5641                 // push    r14
    WORD $0x5541                 // push    r13
    WORD $0x5441                 // push    r12
    BYTE $0x53                   // push    rbx
    LONG $0x08ec8148; WORD $0x0003; BYTE $0x00 // sub    rsp, 776
    WORD $0xd089                 // mov    eax, edx
    LONG $0xc057f8c5             // vxorps    xmm0, xmm0, xmm0
    LONG $0xdbefe1c5             // vpxor    xmm3, xmm3, xmm3
    LONG $0xf657c8c5             // vxorps    xmm6, xmm6, xmm6
    LONG $0x573841c4; BYTE $0xc0 // vxorps    xmm8, xmm8, xmm8
    LONG $0x573041c4; BYTE $0xc9 // vxorps    xmm9, xmm9, xmm9
    LONG $0xc957f0c5             // vxorps    xmm1, xmm1, xmm1
    QUAD $0x0000c0248c11fcc5; BYTE $0x00 // vmovups    yword [rsp + 192], ymm1
    QUAD $0x0000a0248c11fcc5; BYTE $0x00 // vmovups    yword [rsp + 160], ymm1
    LONG $0x4c11fcc5; WORD $0x2024 // vmovups    yword [rsp + 32], ymm1
    WORD $0xd285                 // test    edx, edx
	JLE LBB0_3
    LONG $0x107cc1c4; BYTE $0x01 // vmovups    ymm0, yword [r9]
    QUAD $0x000160248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 352], ymm0
    LONG $0x107cc1c4; WORD $0x2041 // vmovups    ymm0, yword [r9 + 32]
    QUAD $0x000240248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 576], ymm0
    LONG $0x107cc1c4; WORD $0x4041 // vmovups    ymm0, yword [r9 + 64]
    QUAD $0x000220248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 544], ymm0
    LONG $0x107cc1c4; WORD $0x6041 // vmovups    ymm0, yword [r9 + 96]
    QUAD $0x000200248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 512], ymm0
    QUAD $0x00008081107cc1c4; BYTE $0x00 // vmovups    ymm0, yword [r9 + 128]
    QUAD $0x000180248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 384], ymm0
    QUAD $0x0000a081107cc1c4; BYTE $0x00 // vmovups    ymm0, yword [r9 + 160]
    QUAD $0x0001e0248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 480], ymm0
    QUAD $0x0000c081107cc1c4; BYTE $0x00 // vmovups    ymm0, yword [r9 + 192]
    QUAD $0x0001c0248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 448], ymm0
    QUAD $0x0000e081107cc1c4; BYTE $0x00 // vmovups    ymm0, yword [r9 + 224]
    QUAD $0x0001a0248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 416], ymm0
    WORD $0x8941; BYTE $0xd2     // mov    r10d, edx
    LONG $0x07e2c141             // shl    r10d, 7
    LONG $0xd2148d45             // lea    r10d, [r10 + 8*rdx]
    WORD $0x8949; BYTE $0xc3     // mov    r11, rax
    LONG $0x04e3c149             // shl    r11, 4
    LONG $0xdb1c8d4f             // lea    r11, [r11 + 8*r11]
    LONG $0xc057f8c5             // vxorps    xmm0, xmm0, xmm0
    LONG $0x4411fcc5; WORD $0x2024 // vmovups    yword [rsp + 32], ymm0
    WORD $0xdb31                 // xor    ebx, ebx
    WORD $0x8949; BYTE $0xf6     // mov    r14, rsi
    QUAD $0x0000a0248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 160], ymm0
    QUAD $0x0000c0248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 192], ymm0
LBB0_2:
    QUAD $0x0002a0248c117cc5; BYTE $0x00 // vmovups    yword [rsp + 672], ymm9
    QUAD $0x0002c02484117cc5; BYTE $0x00 // vmovups    yword [rsp + 704], ymm8
    QUAD $0x0002e024b411fcc5; BYTE $0x00 // vmovups    yword [rsp + 736], ymm6
    QUAD $0x0000e0249c7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 224], ymm3
    QUAD $0x000100248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 256], ymm0
    QUAD $0x00024024846f7ec5; BYTE $0x00 // vmovdqu    ymm8, yword [rsp + 576]
    LONG $0x44efbdc5; WORD $0x101f // vpxor    ymm0, ymm8, yword [rdi + rbx + 16]
    LONG $0x4cefbdc5; WORD $0x301f // vpxor    ymm1, ymm8, yword [rdi + rbx + 48]
    QUAD $0x00020024b46f7ec5; BYTE $0x00 // vmovdqu    ymm14, yword [rsp + 512]
    LONG $0x360de2c4; BYTE $0xd1 // vpermd    ymm2, ymm14, ymm1
    LONG $0x027de3c4; WORD $0xf0d2 // vpblendd    ymm2, ymm0, ymm2, 240
    LONG $0x147ffec5; BYTE $0x24 // vmovdqu    yword [rsp], ymm2
    LONG $0x360de2c4; BYTE $0xc0 // vpermd    ymm0, ymm14, ymm0
    LONG $0x027de3c4; WORD $0xf0d9 // vpblendd    ymm3, ymm0, ymm1, 240
    LONG $0x5c7ffec5; WORD $0x6024 // vmovdqu    yword [rsp + 96], ymm3
    QUAD $0x00016024a46ffec5; BYTE $0x00 // vmovdqu    ymm4, yword [rsp + 352]
    LONG $0xecdbedc5             // vpand    ymm5, ymm2, ymm4
    LONG $0x6f7ec1c4; WORD $0x0846 // vmovdqu    ymm0, yword [r14 + 8]
    LONG $0xc870f9c5; BYTE $0xa0 // vpshufd    xmm1, xmm0, 160
    LONG $0xd4dbe5c5             // vpand    ymm2, ymm3, ymm4
    LONG $0x00fde3c4; WORD $0x44c9 // vpermq    ymm1, ymm1, 68
    LONG $0xe570fdc5; BYTE $0x88 // vpshufd    ymm4, ymm5, 136
    LONG $0xdd6ffdc5             // vmovdqa    ymm3, ymm5
    LONG $0xef2941c4; BYTE $0xd2 // vpxor    xmm10, xmm10, xmm10
    LONG $0x505d62c4; BYTE $0xd1                                 // {vex}    vpdpbusd    ymm10, ymm4, ymm1
    LONG $0xf270fdc5; BYTE $0x88 // vpshufd    ymm6, ymm2, 136
    LONG $0xedefd1c5             // vpxor    xmm5, xmm5, xmm5
    LONG $0xf870fdc5; BYTE $0xa0 // vpshufd    ymm7, ymm0, 160
    LONG $0x504de2c4; BYTE $0xe9                                 // {vex}    vpdpbusd    ymm5, ymm6, ymm1
    LONG $0x00fde3c4; WORD $0xeecf // vpermq    ymm1, ymm7, 238
    LONG $0xef1141c4; BYTE $0xed // vpxor    xmm13, xmm13, xmm13
    LONG $0xef1941c4; BYTE $0xe4 // vpxor    xmm12, xmm12, xmm12
    LONG $0x505d62c4; BYTE $0xe9                                 // {vex}    vpdpbusd    ymm13, ymm4, ymm1
    LONG $0x6f7e01c4; WORD $0x167c; BYTE $0x08 // vmovdqu    ymm15, yword [r14 + r10 + 8]
    LONG $0x7079c1c4; WORD $0xa0ff // vpshufd    xmm7, xmm15, 160
    LONG $0x00fd63c4; WORD $0x44cf // vpermq    ymm9, ymm7, 68
    LONG $0x504d62c4; BYTE $0xe1                                 // {vex}    vpdpbusd    ymm12, ymm6, ymm1
    LONG $0xef2141c4; BYTE $0xdb // vpxor    xmm11, xmm11, xmm11
    LONG $0x505d42c4; BYTE $0xd9                                 // {vex}    vpdpbusd    ymm11, ymm4, ymm9
    LONG $0xffefc1c5             // vpxor    xmm7, xmm7, xmm7
    LONG $0x504dc2c4; BYTE $0xf9                                 // {vex}    vpdpbusd    ymm7, ymm6, ymm9
    LONG $0x707dc1c4; WORD $0xa0cf // vpshufd    ymm1, ymm15, 160
    LONG $0x00fde3c4; WORD $0xeec9 // vpermq    ymm1, ymm1, 238
    LONG $0xef3141c4; BYTE $0xc9 // vpxor    xmm9, xmm9, xmm9
    LONG $0x505d62c4; BYTE $0xc9                                 // {vex}    vpdpbusd    ymm9, ymm4, ymm1
    LONG $0xe4efd9c5             // vpxor    xmm4, xmm4, xmm4
    LONG $0x504de2c4; BYTE $0xe1                                 // {vex}    vpdpbusd    ymm4, ymm6, ymm1
    LONG $0xf46ffdc5             // vmovdqa    ymm6, ymm4
    LONG $0xe06ffdc5             // vmovdqa    ymm4, ymm0
    QUAD $0x00008024847ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 128], ymm0
    LONG $0xcc70f9c5; BYTE $0xf5 // vpshufd    xmm1, xmm4, 245
    LONG $0x00fde3c4; WORD $0x44c9 // vpermq    ymm1, ymm1, 68
    LONG $0xc370fdc5; BYTE $0xdd // vpshufd    ymm0, ymm3, 221
    LONG $0x507d62c4; BYTE $0xd1                                 // {vex}    vpdpbusd    ymm10, ymm0, ymm1
    LONG $0xd270fdc5; BYTE $0xdd // vpshufd    ymm2, ymm2, 221
    LONG $0x506de2c4; BYTE $0xe9                                 // {vex}    vpdpbusd    ymm5, ymm2, ymm1
    LONG $0xcc70fdc5; BYTE $0xf5 // vpshufd    ymm1, ymm4, 245
    LONG $0x00fde3c4; WORD $0xeec9 // vpermq    ymm1, ymm1, 238
    LONG $0x507d62c4; BYTE $0xe9                                 // {vex}    vpdpbusd    ymm13, ymm0, ymm1
    LONG $0x506d62c4; BYTE $0xe1                                 // {vex}    vpdpbusd    ymm12, ymm2, ymm1
    QUAD $0x00028024bc7f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 640], ymm15
    LONG $0x7079c1c4; WORD $0xf5cf // vpshufd    xmm1, xmm15, 245
    LONG $0x00fde3c4; WORD $0x44c9 // vpermq    ymm1, ymm1, 68
    LONG $0x507d62c4; BYTE $0xd9                                 // {vex}    vpdpbusd    ymm11, ymm0, ymm1
    LONG $0x506de2c4; BYTE $0xf9                                 // {vex}    vpdpbusd    ymm7, ymm2, ymm1
    LONG $0x707dc1c4; WORD $0xf5cf // vpshufd    ymm1, ymm15, 245
    LONG $0x64efbdc5; WORD $0x501f // vpxor    ymm4, ymm8, yword [rdi + rbx + 80]
    LONG $0x00fde3c4; WORD $0xeec9 // vpermq    ymm1, ymm1, 238
    LONG $0x507d62c4; BYTE $0xc9                                 // {vex}    vpdpbusd    ymm9, ymm0, ymm1
    LONG $0x44efbdc5; WORD $0x701f // vpxor    ymm0, ymm8, yword [rdi + rbx + 112]
    LONG $0x506de2c4; BYTE $0xf1                                 // {vex}    vpdpbusd    ymm6, ymm2, ymm1
    LONG $0xfe6f7dc5             // vmovdqa    ymm15, ymm6
    LONG $0x6f7ec1c4; WORD $0x2856 // vmovdqu    ymm2, yword [r14 + 40]
    LONG $0x360de2c4; BYTE $0xc8 // vpermd    ymm1, ymm14, ymm0
    LONG $0x025de3c4; WORD $0xf0d9 // vpblendd    ymm3, ymm4, ymm1, 240
    QUAD $0x000120249c7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 288], ymm3
    LONG $0x360de2c4; BYTE $0xcc // vpermd    ymm1, ymm14, ymm4
    LONG $0x0275e3c4; WORD $0xf0e0 // vpblendd    ymm4, ymm1, ymm0, 240
    QUAD $0x00026024a47ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 608], ymm4
    LONG $0xc270f9c5; BYTE $0xa0 // vpshufd    xmm0, xmm2, 160
    LONG $0xc26f7dc5             // vmovdqa    ymm8, ymm2
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    QUAD $0x000160248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 352]
    LONG $0xf1dbe5c5             // vpand    ymm6, ymm3, ymm1
    LONG $0xd670fdc5; BYTE $0x88 // vpshufd    ymm2, ymm6, 136
    LONG $0x506d62c4; BYTE $0xd0                                 // {vex}    vpdpbusd    ymm10, ymm2, ymm0
    LONG $0xe1dbddc5             // vpand    ymm4, ymm4, ymm1
    LONG $0xd96ffdc5             // vmovdqa    ymm3, ymm1
    LONG $0xcc70fdc5; BYTE $0x88 // vpshufd    ymm1, ymm4, 136
    LONG $0x5075e2c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm5, ymm1, ymm0
    LONG $0x707dc1c4; WORD $0xa0c0 // vpshufd    ymm0, ymm8, 160
    LONG $0x6f7d41c4; BYTE $0xf0 // vmovdqa    ymm14, ymm8
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x506d62c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm13, ymm2, ymm0
    LONG $0x507562c4; BYTE $0xe0                                 // {vex}    vpdpbusd    ymm12, ymm1, ymm0
    LONG $0x6f7e01c4; WORD $0x1644; BYTE $0x28 // vmovdqu    ymm8, yword [r14 + r10 + 40]
    LONG $0x7079c1c4; WORD $0xa0c0 // vpshufd    xmm0, xmm8, 160
    QUAD $0x00014024847f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 320], ymm8
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0x506d62c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm11, ymm2, ymm0
    LONG $0x5075e2c4; BYTE $0xf8                                 // {vex}    vpdpbusd    ymm7, ymm1, ymm0
    LONG $0x707dc1c4; WORD $0xa0c0 // vpshufd    ymm0, ymm8, 160
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x506d62c4; BYTE $0xc8                                 // {vex}    vpdpbusd    ymm9, ymm2, ymm0
    LONG $0x507562c4; BYTE $0xf8                                 // {vex}    vpdpbusd    ymm15, ymm1, ymm0
    LONG $0x747f7ec5; WORD $0x4024 // vmovdqu    yword [rsp + 64], ymm14
    LONG $0x7079c1c4; WORD $0xf5c6 // vpshufd    xmm0, xmm14, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0xce70fdc5; BYTE $0xdd // vpshufd    ymm1, ymm6, 221
    LONG $0x507562c4; BYTE $0xd0                                 // {vex}    vpdpbusd    ymm10, ymm1, ymm0
    LONG $0xd470fdc5; BYTE $0xdd // vpshufd    ymm2, ymm4, 221
    LONG $0x506de2c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm5, ymm2, ymm0
    LONG $0x707dc1c4; WORD $0xf5c6 // vpshufd    ymm0, ymm14, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x507562c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm13, ymm1, ymm0
    LONG $0x506d62c4; BYTE $0xe0                                 // {vex}    vpdpbusd    ymm12, ymm2, ymm0
    LONG $0x7079c1c4; WORD $0xf5c0 // vpshufd    xmm0, xmm8, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0x507562c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm11, ymm1, ymm0
    LONG $0x506de2c4; BYTE $0xf8                                 // {vex}    vpdpbusd    ymm7, ymm2, ymm0
    LONG $0x707dc1c4; WORD $0xf5c0 // vpshufd    ymm0, ymm8, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x507562c4; BYTE $0xc8                                 // {vex}    vpdpbusd    ymm9, ymm1, ymm0
    LONG $0x506d62c4; BYTE $0xf8                                 // {vex}    vpdpbusd    ymm15, ymm2, ymm0
    LONG $0x046ffec5; BYTE $0x24 // vmovdqu    ymm0, yword [rsp]
    LONG $0xd071fdc5; BYTE $0x04 // vpsrlw    ymm0, ymm0, 4
    LONG $0x4c6ffec5; WORD $0x6024 // vmovdqu    ymm1, yword [rsp + 96]
    LONG $0xd171f5c5; BYTE $0x04 // vpsrlw    ymm1, ymm1, 4
    LONG $0xe0dbe5c5             // vpand    ymm4, ymm3, ymm0
    LONG $0xd1dbe5c5             // vpand    ymm2, ymm3, ymm1
    LONG $0x6f7e41c4; WORD $0x4846 // vmovdqu    ymm8, yword [r14 + 72]
    LONG $0x7079c1c4; WORD $0xa0c0 // vpshufd    xmm0, xmm8, 160
    LONG $0x00fde3c4; WORD $0x44f0 // vpermq    ymm6, ymm0, 68
    LONG $0xcc70fdc5; BYTE $0x88 // vpshufd    ymm1, ymm4, 136
    LONG $0x507562c4; BYTE $0xd6                                 // {vex}    vpdpbusd    ymm10, ymm1, ymm6
    LONG $0xc270fdc5; BYTE $0x88 // vpshufd    ymm0, ymm2, 136
    LONG $0x507de2c4; BYTE $0xee                                 // {vex}    vpdpbusd    ymm5, ymm0, ymm6
    LONG $0x707dc1c4; WORD $0xa0f0 // vpshufd    ymm6, ymm8, 160
    LONG $0x00fde3c4; WORD $0xeef6 // vpermq    ymm6, ymm6, 238
    LONG $0x507562c4; BYTE $0xee                                 // {vex}    vpdpbusd    ymm13, ymm1, ymm6
    LONG $0x507d62c4; BYTE $0xe6                                 // {vex}    vpdpbusd    ymm12, ymm0, ymm6
    LONG $0x6f7e01c4; WORD $0x1674; BYTE $0x48 // vmovdqu    ymm14, yword [r14 + r10 + 72]
    LONG $0x7079c1c4; WORD $0xa0f6 // vpshufd    xmm6, xmm14, 160
    LONG $0x00fde3c4; WORD $0x44f6 // vpermq    ymm6, ymm6, 68
    LONG $0x507562c4; BYTE $0xde                                 // {vex}    vpdpbusd    ymm11, ymm1, ymm6
    LONG $0x507de2c4; BYTE $0xfe                                 // {vex}    vpdpbusd    ymm7, ymm0, ymm6
    LONG $0x707dc1c4; WORD $0xa0f6 // vpshufd    ymm6, ymm14, 160
    LONG $0x00fde3c4; WORD $0xeef6 // vpermq    ymm6, ymm6, 238
    LONG $0x507562c4; BYTE $0xce                                 // {vex}    vpdpbusd    ymm9, ymm1, ymm6
    LONG $0x507d62c4; BYTE $0xfe                                 // {vex}    vpdpbusd    ymm15, ymm0, ymm6
    LONG $0x7079c1c4; WORD $0xf5c0 // vpshufd    xmm0, xmm8, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0xcc70fdc5; BYTE $0xdd // vpshufd    ymm1, ymm4, 221
    LONG $0x507562c4; BYTE $0xd0                                 // {vex}    vpdpbusd    ymm10, ymm1, ymm0
    LONG $0xd270fdc5; BYTE $0xdd // vpshufd    ymm2, ymm2, 221
    LONG $0x506de2c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm5, ymm2, ymm0
    LONG $0x707dc1c4; WORD $0xf5c0 // vpshufd    ymm0, ymm8, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x507562c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm13, ymm1, ymm0
    LONG $0x506d62c4; BYTE $0xe0                                 // {vex}    vpdpbusd    ymm12, ymm2, ymm0
    LONG $0x747f7ec5; WORD $0x6024 // vmovdqu    yword [rsp + 96], ymm14
    LONG $0x7079c1c4; WORD $0xf5c6 // vpshufd    xmm0, xmm14, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0x507562c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm11, ymm1, ymm0
    LONG $0x1c7f7ec5; BYTE $0x24 // vmovdqu    yword [rsp], ymm11
    LONG $0x506de2c4; BYTE $0xf8                                 // {vex}    vpdpbusd    ymm7, ymm2, ymm0
    LONG $0x707dc1c4; WORD $0xf5c6 // vpshufd    ymm0, ymm14, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x507562c4; BYTE $0xc8                                 // {vex}    vpdpbusd    ymm9, ymm1, ymm0
    LONG $0x6f7d41c4; BYTE $0xf1 // vmovdqa    ymm14, ymm9
    LONG $0x506d62c4; BYTE $0xf8                                 // {vex}    vpdpbusd    ymm15, ymm2, ymm0
    QUAD $0x00012024846ffec5; BYTE $0x00 // vmovdqu    ymm0, yword [rsp + 288]
    LONG $0xd071fdc5; BYTE $0x04 // vpsrlw    ymm0, ymm0, 4
    QUAD $0x000260248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 608]
    LONG $0xd171f5c5; BYTE $0x04 // vpsrlw    ymm1, ymm1, 4
    LONG $0xe0dbe5c5             // vpand    ymm4, ymm3, ymm0
    LONG $0xc9dbe5c5             // vpand    ymm1, ymm3, ymm1
    QUAD $0x000120248c7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 288], ymm1
    LONG $0x6f7e41c4; WORD $0x685e // vmovdqu    ymm11, yword [r14 + 104]
    LONG $0x7079c1c4; WORD $0xa0c3 // vpshufd    xmm0, xmm11, 160
    LONG $0x00fde3c4; WORD $0x44f0 // vpermq    ymm6, ymm0, 68
    LONG $0xc470fdc5; BYTE $0x88 // vpshufd    ymm0, ymm4, 136
    LONG $0x507d62c4; BYTE $0xd6                                 // {vex}    vpdpbusd    ymm10, ymm0, ymm6
    LONG $0xc970fdc5; BYTE $0x88 // vpshufd    ymm1, ymm1, 136
    LONG $0x5075e2c4; BYTE $0xee                                 // {vex}    vpdpbusd    ymm5, ymm1, ymm6
    LONG $0x707dc1c4; WORD $0xa0f3 // vpshufd    ymm6, ymm11, 160
    LONG $0x00fde3c4; WORD $0xeef6 // vpermq    ymm6, ymm6, 238
    LONG $0x507d62c4; BYTE $0xee                                 // {vex}    vpdpbusd    ymm13, ymm0, ymm6
    LONG $0x507562c4; BYTE $0xe6                                 // {vex}    vpdpbusd    ymm12, ymm1, ymm6
    LONG $0x6f7e01c4; WORD $0x164c; BYTE $0x68 // vmovdqu    ymm9, yword [r14 + r10 + 104]
    LONG $0x7079c1c4; WORD $0xa0f1 // vpshufd    xmm6, xmm9, 160
    LONG $0x00fde3c4; WORD $0x44f6 // vpermq    ymm6, ymm6, 68
    LONG $0x1c6ffec5; BYTE $0x24 // vmovdqu    ymm3, yword [rsp]
    LONG $0x507de2c4; BYTE $0xde                                 // {vex}    vpdpbusd    ymm3, ymm0, ymm6
    LONG $0x5075e2c4; BYTE $0xfe                                 // {vex}    vpdpbusd    ymm7, ymm1, ymm6
    LONG $0x707dc1c4; WORD $0xa0f1 // vpshufd    ymm6, ymm9, 160
    LONG $0x00fde3c4; WORD $0xeef6 // vpermq    ymm6, ymm6, 238
    LONG $0x507d62c4; BYTE $0xf6                                 // {vex}    vpdpbusd    ymm14, ymm0, ymm6
    LONG $0x507562c4; BYTE $0xfe                                 // {vex}    vpdpbusd    ymm15, ymm1, ymm6
    LONG $0xc0eff9c5             // vpxor    xmm0, xmm0, xmm0
    QUAD $0x00022024946ffec5; BYTE $0x00 // vmovdqu    ymm2, yword [rsp + 544]
    QUAD $0x00802484506de2c4; WORD $0x0000                       // {vex}    vpdpbusd    ymm0, ymm2, YMMWORD PTR [rsp + 128]
    LONG $0x506de2c4; WORD $0x2444; BYTE $0x40                   // {vex}    vpdpbusd    ymm0, ymm2, YMMWORD PTR [rsp + 64]
    LONG $0x506dc2c4; BYTE $0xc0                                 // {vex}    vpdpbusd    ymm0, ymm2, ymm8
    LONG $0x7079c1c4; WORD $0xf5cb // vpshufd    xmm1, xmm11, 245
    LONG $0x00fde3c4; WORD $0x44c9 // vpermq    ymm1, ymm1, 68
    LONG $0xf470fdc5; BYTE $0xdd // vpshufd    ymm6, ymm4, 221
    LONG $0x504d62c4; BYTE $0xd1                                 // {vex}    vpdpbusd    ymm10, ymm6, ymm1
    QUAD $0x0001202484707dc5; WORD $0xdd00 // vpshufd    ymm8, yword [rsp + 288], 221
    LONG $0x503de2c4; BYTE $0xe9                                 // {vex}    vpdpbusd    ymm5, ymm8, ymm1
    LONG $0x707dc1c4; WORD $0xf5cb // vpshufd    ymm1, ymm11, 245
    LONG $0x00fde3c4; WORD $0xeec9 // vpermq    ymm1, ymm1, 238
    LONG $0x504d62c4; BYTE $0xe9                                 // {vex}    vpdpbusd    ymm13, ymm6, ymm1
    LONG $0x503d62c4; BYTE $0xe1                                 // {vex}    vpdpbusd    ymm12, ymm8, ymm1
    LONG $0x506dc2c4; BYTE $0xc3                                 // {vex}    vpdpbusd    ymm0, ymm2, ymm11
    LONG $0x7079c1c4; WORD $0xf5c9 // vpshufd    xmm1, xmm9, 245
    LONG $0x00fde3c4; WORD $0x44c9 // vpermq    ymm1, ymm1, 68
    LONG $0x504de2c4; BYTE $0xd9                                 // {vex}    vpdpbusd    ymm3, ymm6, ymm1
    LONG $0x1c7ffec5; BYTE $0x24 // vmovdqu    yword [rsp], ymm3
    LONG $0x503de2c4; BYTE $0xf9                                 // {vex}    vpdpbusd    ymm7, ymm8, ymm1
    LONG $0x137de2c4; WORD $0x1f24 // vcvtph2ps    ymm4, oword [rdi + rbx]
    LONG $0x027d62c4; BYTE $0xd8 // vphaddd    ymm11, ymm0, ymm0
    LONG $0x707dc1c4; WORD $0xf5c1 // vpshufd    ymm0, ymm9, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x504d62c4; BYTE $0xf0                                 // {vex}    vpdpbusd    ymm14, ymm6, ymm0
    QUAD $0x00008024b47f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 128], ymm14
    LONG $0x1379c2c4; BYTE $0x36 // vcvtph2ps    xmm6, qword [r14]
    LONG $0x503d62c4; BYTE $0xf8                                 // {vex}    vpdpbusd    ymm15, ymm8, ymm0
    QUAD $0x0002c02484107cc5; BYTE $0x00 // vmovups    ymm8, yword [rsp + 704]
    QUAD $0x00018024846ffec5; BYTE $0x00 // vmovdqu    ymm0, yword [rsp + 384]
    LONG $0x367dc2c4; BYTE $0xc3 // vpermd    ymm0, ymm0, ymm11
    LONG $0xf072fdc5; BYTE $0x03 // vpslld    ymm0, ymm0, 3
    LONG $0xcd6cadc5             // vpunpcklqdq    ymm1, ymm10, ymm5
    LONG $0xc0faf5c5             // vpsubd    ymm0, ymm1, ymm0
    LONG $0x187de2c4; BYTE $0xce // vbroadcastss    ymm1, xmm6
    LONG $0xcc59f4c5             // vmulps    ymm1, ymm1, ymm4
    LONG $0xc05bfcc5             // vcvtdq2ps    ymm0, ymm0
    LONG $0x74107cc5; WORD $0x2024 // vmovups    ymm14, yword [rsp + 32]
    LONG $0xb87d62c4; BYTE $0xf1 // vfmadd231ps    ymm14, ymm0, ymm1
    LONG $0x74117cc5; WORD $0x2024 // vmovups    yword [rsp + 32], ymm14
    LONG $0xc56dadc5             // vpunpckhqdq    ymm0, ymm10, ymm5
    QUAD $0x0001e024ac6ffec5; BYTE $0x00 // vmovdqu    ymm5, yword [rsp + 480]
    LONG $0x3655c2c4; BYTE $0xcb // vpermd    ymm1, ymm5, ymm11
    LONG $0xf172f5c5; BYTE $0x03 // vpslld    ymm1, ymm1, 3
    LONG $0xc1fafdc5             // vpsubd    ymm0, ymm0, ymm1
    LONG $0xce16fac5             // vmovshdup    xmm1, xmm6
    LONG $0x187de2c4; BYTE $0xc9 // vbroadcastss    ymm1, xmm1
    LONG $0xc05bfcc5             // vcvtdq2ps    ymm0, ymm0
    LONG $0xcc59f4c5             // vmulps    ymm1, ymm1, ymm4
    QUAD $0x0000a0249c10fcc5; BYTE $0x00 // vmovups    ymm3, yword [rsp + 160]
    LONG $0xb87de2c4; BYTE $0xd9 // vfmadd231ps    ymm3, ymm0, ymm1
    QUAD $0x0000a0249c11fcc5; BYTE $0x00 // vmovups    yword [rsp + 160], ymm3
    QUAD $0x0001c024b46f7ec5; BYTE $0x00 // vmovdqu    ymm14, yword [rsp + 448]
    LONG $0x360dc2c4; BYTE $0xc3 // vpermd    ymm0, ymm14, ymm11
    LONG $0x6c15c1c4; BYTE $0xcc // vpunpcklqdq    ymm1, ymm13, ymm12
    LONG $0xf072fdc5; BYTE $0x03 // vpslld    ymm0, ymm0, 3
    LONG $0xc0faf5c5             // vpsubd    ymm0, ymm1, ymm0
    LONG $0xcec6c9c5; BYTE $0x01 // vshufpd    xmm1, xmm6, xmm6, 1
    LONG $0xc05bfcc5             // vcvtdq2ps    ymm0, ymm0
    LONG $0x187de2c4; BYTE $0xc9 // vbroadcastss    ymm1, xmm1
    LONG $0xcc59f4c5             // vmulps    ymm1, ymm1, ymm4
    QUAD $0x0000c0249c10fcc5; BYTE $0x00 // vmovups    ymm3, yword [rsp + 192]
    LONG $0xb87de2c4; BYTE $0xd9 // vfmadd231ps    ymm3, ymm0, ymm1
    QUAD $0x0000c0249c11fcc5; BYTE $0x00 // vmovups    yword [rsp + 192], ymm3
    LONG $0xc057f8c5             // vxorps    xmm0, xmm0, xmm0
    QUAD $0x02802484506de2c4; WORD $0x0000                       // {vex}    vpdpbusd    ymm0, ymm2, YMMWORD PTR [rsp + 640]
    QUAD $0x01402484506de2c4; WORD $0x0000                       // {vex}    vpdpbusd    ymm0, ymm2, YMMWORD PTR [rsp + 320]
    LONG $0x506de2c4; WORD $0x2444; BYTE $0x60                   // {vex}    vpdpbusd    ymm0, ymm2, YMMWORD PTR [rsp + 96]
    LONG $0x506dc2c4; BYTE $0xc1                                 // {vex}    vpdpbusd    ymm0, ymm2, ymm9
    QUAD $0x0002a0248c107cc5; BYTE $0x00 // vmovups    ymm9, yword [rsp + 672]
    LONG $0x6d15c1c4; BYTE $0xcc // vpunpckhqdq    ymm1, ymm13, ymm12
    QUAD $0x0001a024946f7ec5; BYTE $0x00 // vmovdqu    ymm10, yword [rsp + 416]
    LONG $0x362dc2c4; BYTE $0xd3 // vpermd    ymm2, ymm10, ymm11
    LONG $0xf272edc5; BYTE $0x03 // vpslld    ymm2, ymm2, 3
    LONG $0xcafaf5c5             // vpsubd    ymm1, ymm1, ymm2
    LONG $0xd6c6c8c5; BYTE $0xff // vshufps    xmm2, xmm6, xmm6, 255
    QUAD $0x0002e024b410fcc5; BYTE $0x00 // vmovups    ymm6, yword [rsp + 736]
    LONG $0x187de2c4; BYTE $0xd2 // vbroadcastss    ymm2, xmm2
    LONG $0xd459ecc5             // vmulps    ymm2, ymm2, ymm4
    LONG $0xd95bfcc5             // vcvtdq2ps    ymm3, ymm1
    LONG $0x027de2c4; BYTE $0xc8 // vphaddd    ymm1, ymm0, ymm0
    LONG $0xb86562c4; BYTE $0xca // vfmadd231ps    ymm9, ymm3, ymm2
    LONG $0x137982c4; WORD $0x1604 // vcvtph2ps    xmm0, qword [r14 + r10]
    QUAD $0x00018024946ffec5; BYTE $0x00 // vmovdqu    ymm2, yword [rsp + 384]
    LONG $0x366de2c4; BYTE $0xd1 // vpermd    ymm2, ymm2, ymm1
    LONG $0x1c6f7ec5; BYTE $0x24 // vmovdqu    ymm11, yword [rsp]
    LONG $0xdf6ca5c5             // vpunpcklqdq    ymm3, ymm11, ymm7
    LONG $0xf272edc5; BYTE $0x03 // vpslld    ymm2, ymm2, 3
    LONG $0xd2fae5c5             // vpsubd    ymm2, ymm3, ymm2
    LONG $0x187de2c4; BYTE $0xd8 // vbroadcastss    ymm3, xmm0
    LONG $0xd25bfcc5             // vcvtdq2ps    ymm2, ymm2
    LONG $0xdc59e4c5             // vmulps    ymm3, ymm3, ymm4
    LONG $0xb86d62c4; BYTE $0xc3 // vfmadd231ps    ymm8, ymm2, ymm3
    LONG $0xd76da5c5             // vpunpckhqdq    ymm2, ymm11, ymm7
    LONG $0x3655e2c4; BYTE $0xd9 // vpermd    ymm3, ymm5, ymm1
    LONG $0xf372e5c5; BYTE $0x03 // vpslld    ymm3, ymm3, 3
    LONG $0xd3faedc5             // vpsubd    ymm2, ymm2, ymm3
    LONG $0xd816fac5             // vmovshdup    xmm3, xmm0
    LONG $0xd25bfcc5             // vcvtdq2ps    ymm2, ymm2
    LONG $0x187de2c4; BYTE $0xdb // vbroadcastss    ymm3, xmm3
    LONG $0xdc59e4c5             // vmulps    ymm3, ymm3, ymm4
    LONG $0xb86de2c4; BYTE $0xf3 // vfmadd231ps    ymm6, ymm2, ymm3
    QUAD $0x00008024bc6ffec5; BYTE $0x00 // vmovdqu    ymm7, yword [rsp + 128]
    LONG $0x6c45c1c4; BYTE $0xd7 // vpunpcklqdq    ymm2, ymm7, ymm15
    LONG $0x360de2c4; BYTE $0xd9 // vpermd    ymm3, ymm14, ymm1
    LONG $0xf372e5c5; BYTE $0x03 // vpslld    ymm3, ymm3, 3
    LONG $0xd3faedc5             // vpsubd    ymm2, ymm2, ymm3
    LONG $0xd25bfcc5             // vcvtdq2ps    ymm2, ymm2
    LONG $0xd8c6f9c5; BYTE $0x01 // vshufpd    xmm3, xmm0, xmm0, 1
    LONG $0x187de2c4; BYTE $0xdb // vbroadcastss    ymm3, xmm3
    LONG $0xdc59e4c5             // vmulps    ymm3, ymm3, ymm4
    QUAD $0x0000e024ac10fcc5; BYTE $0x00 // vmovups    ymm5, yword [rsp + 224]
    LONG $0xb86de2c4; BYTE $0xeb // vfmadd231ps    ymm5, ymm2, ymm3
    QUAD $0x0000e024ac11fcc5; BYTE $0x00 // vmovups    yword [rsp + 224], ymm5
    QUAD $0x0000e0249c6ffec5; BYTE $0x00 // vmovdqu    ymm3, yword [rsp + 224]
    LONG $0x6d45c1c4; BYTE $0xd7 // vpunpckhqdq    ymm2, ymm7, ymm15
    LONG $0x362de2c4; BYTE $0xc9 // vpermd    ymm1, ymm10, ymm1
    LONG $0xf172f5c5; BYTE $0x03 // vpslld    ymm1, ymm1, 3
    LONG $0xc9faedc5             // vpsubd    ymm1, ymm2, ymm1
    LONG $0xc0c6f8c5; BYTE $0xff // vshufps    xmm0, xmm0, xmm0, 255
    LONG $0x187de2c4; BYTE $0xc0 // vbroadcastss    ymm0, xmm0
    LONG $0xc459fcc5             // vmulps    ymm0, ymm0, ymm4
    LONG $0xc95bfcc5             // vcvtdq2ps    ymm1, ymm1
    QUAD $0x000100249410fcc5; BYTE $0x00 // vmovups    ymm2, yword [rsp + 256]
    LONG $0xb875e2c4; BYTE $0xd0 // vfmadd231ps    ymm2, ymm1, ymm0
    QUAD $0x000100249411fcc5; BYTE $0x00 // vmovups    yword [rsp + 256], ymm2
    QUAD $0x000100248410fcc5; BYTE $0x00 // vmovups    ymm0, yword [rsp + 256]
    LONG $0x88c68149; WORD $0x0000; BYTE $0x00 // add    r14, 136
    LONG $0x90c38148; WORD $0x0000; BYTE $0x00 // add    rbx, 144
    WORD $0x3949; BYTE $0xdb     // cmp    r11, rbx
	JNE LBB0_2
LBB0_3:
    LONG $0x001c8d47             // lea    r11d, [r8 + r8]
    QUAD $0x00000000852c8d42     // lea    ebp, [4*r8]
    QUAD $0x00000000c5148d46     // lea    r10d, [8*r8]
    WORD $0x8945; BYTE $0xd5     // mov    r13d, r10d
    WORD $0x2945; BYTE $0xc5     // sub    r13d, r8d
    LONG $0x4c10fcc5; WORD $0x2024 // vmovups    ymm1, yword [rsp + 32]
    LONG $0x0911fcc5             // vmovups    yword [rcx], ymm1
    WORD $0x634d; BYTE $0xc0     // movsxd    r8, r8d
    QUAD $0x0000a0248c10fcc5; BYTE $0x00 // vmovups    ymm1, yword [rsp + 160]
    LONG $0x117ca1c4; WORD $0x810c // vmovups    yword [rcx + 4*r8], ymm1
    WORD $0x634d; BYTE $0xdb     // movsxd    r11, r11d
    QUAD $0x0000c0248c10fcc5; BYTE $0x00 // vmovups    ymm1, yword [rsp + 192]
    LONG $0x117ca1c4; WORD $0x990c // vmovups    yword [rcx + 4*r11], ymm1
    LONG $0x401c8d43             // lea    ebx, [r8 + 2*r8]
    WORD $0x6348; BYTE $0xdb     // movsxd    rbx, ebx
    LONG $0x0c117cc5; BYTE $0x99 // vmovups    yword [rcx + 4*rbx], ymm9
    WORD $0x634c; BYTE $0xf5     // movsxd    r14, ebp
    LONG $0x117c21c4; WORD $0xb104 // vmovups    yword [rcx + 4*r14], ymm8
    LONG $0x802c8d43             // lea    ebp, [r8 + 4*r8]
    WORD $0x634c; BYTE $0xfd     // movsxd    r15, ebp
    LONG $0x117ca1c4; WORD $0xb934 // vmovups    yword [rcx + 4*r15], ymm6
    LONG $0x5b2c8d43             // lea    ebp, [r11 + 2*r11]
    WORD $0x634c; BYTE $0xe5     // movsxd    r12, ebp
    LONG $0x7f7ea1c4; WORD $0xa11c // vmovdqu    yword [rcx + 4*r12], ymm3
    WORD $0x634d; BYTE $0xed     // movsxd    r13, r13d
    LONG $0x117ca1c4; WORD $0xa904 // vmovups    yword [rcx + 4*r13], ymm0
    LONG $0xe457d8c5             // vxorps    xmm4, xmm4, xmm4
    LONG $0xf657c8c5             // vxorps    xmm6, xmm6, xmm6
    LONG $0xffefc1c5             // vpxor    xmm7, xmm7, xmm7
    LONG $0x573841c4; BYTE $0xc0 // vxorps    xmm8, xmm8, xmm8
    LONG $0xc057f8c5             // vxorps    xmm0, xmm0, xmm0
    LONG $0x0411fcc5; BYTE $0x24 // vmovups    yword [rsp], ymm0
    LONG $0x4411fcc5; WORD $0x2024 // vmovups    yword [rsp + 32], ymm0
    LONG $0xef1141c4; BYTE $0xed // vpxor    xmm13, xmm13, xmm13
    WORD $0xd285                 // test    edx, edx
	JLE LBB0_6
    WORD $0x6348; BYTE $0xea     // movsxd    rbp, edx
    LONG $0x10ed6948; WORD $0x0001; BYTE $0x00 // imul    rbp, rbp, 272
    WORD $0x0148; BYTE $0xee     // add    rsi, rbp
    LONG $0x107cc1c4; BYTE $0x01 // vmovups    ymm0, yword [r9]
    QUAD $0x0000c0248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 192], ymm0
    LONG $0x107cc1c4; WORD $0x2041 // vmovups    ymm0, yword [r9 + 32]
    QUAD $0x000220248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 544], ymm0
    LONG $0x107cc1c4; WORD $0x4041 // vmovups    ymm0, yword [r9 + 64]
    QUAD $0x000200248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 512], ymm0
    LONG $0x107cc1c4; WORD $0x6041 // vmovups    ymm0, yword [r9 + 96]
    QUAD $0x0001e0248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 480], ymm0
    QUAD $0x00008081107cc1c4; BYTE $0x00 // vmovups    ymm0, yword [r9 + 128]
    QUAD $0x0000a0248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 160], ymm0
    QUAD $0x0000a081107cc1c4; BYTE $0x00 // vmovups    ymm0, yword [r9 + 160]
    QUAD $0x0001c0248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 448], ymm0
    QUAD $0x0000c081107cc1c4; BYTE $0x00 // vmovups    ymm0, yword [r9 + 192]
    QUAD $0x000160248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 352], ymm0
    QUAD $0x0000e081107cc1c4; BYTE $0x00 // vmovups    ymm0, yword [r9 + 224]
    QUAD $0x0001a0248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 416], ymm0
    WORD $0x8941; BYTE $0xd1     // mov    r9d, edx
    WORD $0xe2c1; BYTE $0x07     // shl    edx, 7
    LONG $0xca148d42             // lea    edx, [rdx + 8*r9]
    LONG $0x04e0c148             // shl    rax, 4
    LONG $0xc0048d48             // lea    rax, [rax + 8*rax]
    WORD $0x3145; BYTE $0xc9     // xor    r9d, r9d
    LONG $0xc057f8c5             // vxorps    xmm0, xmm0, xmm0
    LONG $0x4411fcc5; WORD $0x2024 // vmovups    yword [rsp + 32], ymm0
    LONG $0x0411fcc5; BYTE $0x24 // vmovups    yword [rsp], ymm0
LBB0_5:
    QUAD $0x0002a02484117cc5; BYTE $0x00 // vmovups    yword [rsp + 672], ymm8
    QUAD $0x0002c024bc7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 704], ymm7
    QUAD $0x0002e024ac7f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 736], ymm13
    QUAD $0x00018024b411fcc5; BYTE $0x00 // vmovups    yword [rsp + 384], ymm6
    QUAD $0x0000e024a411fcc5; BYTE $0x00 // vmovups    yword [rsp + 224], ymm4
    QUAD $0x000100248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 256], ymm0
    QUAD $0x00022024ac6f7ec5; BYTE $0x00 // vmovdqu    ymm13, yword [rsp + 544]
    LONG $0xef15a1c4; WORD $0x0f44; BYTE $0x10 // vpxor    ymm0, ymm13, yword [rdi + r9 + 16]
    LONG $0xef15a1c4; WORD $0x0f4c; BYTE $0x30 // vpxor    ymm1, ymm13, yword [rdi + r9 + 48]
    QUAD $0x0001e024bc6f7ec5; BYTE $0x00 // vmovdqu    ymm15, yword [rsp + 480]
    LONG $0x3605e2c4; BYTE $0xd1 // vpermd    ymm2, ymm15, ymm1
    LONG $0x027de3c4; WORD $0xf0d2 // vpblendd    ymm2, ymm0, ymm2, 240
    LONG $0x547ffec5; WORD $0x6024 // vmovdqu    yword [rsp + 96], ymm2
    LONG $0x3605e2c4; BYTE $0xc0 // vpermd    ymm0, ymm15, ymm0
    LONG $0x027de3c4; WORD $0xf0d9 // vpblendd    ymm3, ymm0, ymm1, 240
    LONG $0x5c7ffec5; WORD $0x4024 // vmovdqu    yword [rsp + 64], ymm3
    QUAD $0x0000c024a46ffec5; BYTE $0x00 // vmovdqu    ymm4, yword [rsp + 192]
    LONG $0xccdbedc5             // vpand    ymm1, ymm2, ymm4
    LONG $0x466ffec5; BYTE $0x08 // vmovdqu    ymm0, yword [rsi + 8]
    LONG $0xd070f9c5; BYTE $0xa0 // vpshufd    xmm2, xmm0, 160
    LONG $0xdcdbe5c5             // vpand    ymm3, ymm3, ymm4
    QUAD $0x000140249c7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 320], ymm3
    LONG $0x00fde3c4; WORD $0x44d2 // vpermq    ymm2, ymm2, 68
    LONG $0xe170fdc5; BYTE $0x88 // vpshufd    ymm4, ymm1, 136
    LONG $0xed57d0c5             // vxorps    xmm5, xmm5, xmm5
    LONG $0x505de2c4; BYTE $0xea                                 // {vex}    vpdpbusd    ymm5, ymm4, ymm2
    LONG $0xfb70fdc5; BYTE $0x88 // vpshufd    ymm7, ymm3, 136
    LONG $0xf657c8c5             // vxorps    xmm6, xmm6, xmm6
    LONG $0xd870fdc5; BYTE $0xa0 // vpshufd    ymm3, ymm0, 160
    LONG $0x5045e2c4; BYTE $0xf2                                 // {vex}    vpdpbusd    ymm6, ymm7, ymm2
    LONG $0x00fde3c4; WORD $0xeed3 // vpermq    ymm2, ymm3, 238
    LONG $0xef0941c4; BYTE $0xf6 // vpxor    xmm14, xmm14, xmm14
    LONG $0x573041c4; BYTE $0xc9 // vxorps    xmm9, xmm9, xmm9
    LONG $0x505d62c4; BYTE $0xf2                                 // {vex}    vpdpbusd    ymm14, ymm4, ymm2
    LONG $0x5c6ffec5; WORD $0x0816 // vmovdqu    ymm3, yword [rsi + rdx + 8]
    LONG $0xc37079c5; BYTE $0xa0 // vpshufd    xmm8, xmm3, 160
    LONG $0x00fd43c4; WORD $0x44c0 // vpermq    ymm8, ymm8, 68
    LONG $0x504562c4; BYTE $0xca                                 // {vex}    vpdpbusd    ymm9, ymm7, ymm2
    LONG $0xef1941c4; BYTE $0xe4 // vpxor    xmm12, xmm12, xmm12
    LONG $0x505d42c4; BYTE $0xe0                                 // {vex}    vpdpbusd    ymm12, ymm4, ymm8
    LONG $0xef2941c4; BYTE $0xd2 // vpxor    xmm10, xmm10, xmm10
    LONG $0x504542c4; BYTE $0xd0                                 // {vex}    vpdpbusd    ymm10, ymm7, ymm8
    LONG $0xc36f7dc5             // vmovdqa    ymm8, ymm3
    QUAD $0x000280249c7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 640], ymm3
    LONG $0xd370fdc5; BYTE $0xa0 // vpshufd    ymm2, ymm3, 160
    LONG $0x00fde3c4; WORD $0xeed2 // vpermq    ymm2, ymm2, 238
    LONG $0xdbefe1c5             // vpxor    xmm3, xmm3, xmm3
    LONG $0x505de2c4; BYTE $0xda                                 // {vex}    vpdpbusd    ymm3, ymm4, ymm2
    LONG $0xef2141c4; BYTE $0xdb // vpxor    xmm11, xmm11, xmm11
    LONG $0x504562c4; BYTE $0xda                                 // {vex}    vpdpbusd    ymm11, ymm7, ymm2
    LONG $0xe06ffdc5             // vmovdqa    ymm4, ymm0
    QUAD $0x00008024847ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 128], ymm0
    LONG $0xd470f9c5; BYTE $0xf5 // vpshufd    xmm2, xmm4, 245
    LONG $0x00fde3c4; WORD $0x44d2 // vpermq    ymm2, ymm2, 68
    LONG $0xc970fdc5; BYTE $0xdd // vpshufd    ymm1, ymm1, 221
    LONG $0x5075e2c4; BYTE $0xea                                 // {vex}    vpdpbusd    ymm5, ymm1, ymm2
    QUAD $0x000140248470fdc5; WORD $0xdd00 // vpshufd    ymm0, yword [rsp + 320], 221
    LONG $0x507de2c4; BYTE $0xf2                                 // {vex}    vpdpbusd    ymm6, ymm0, ymm2
    LONG $0xd470fdc5; BYTE $0xf5 // vpshufd    ymm2, ymm4, 245
    LONG $0x00fde3c4; WORD $0xeed2 // vpermq    ymm2, ymm2, 238
    LONG $0x507562c4; BYTE $0xf2                                 // {vex}    vpdpbusd    ymm14, ymm1, ymm2
    LONG $0x507d62c4; BYTE $0xca                                 // {vex}    vpdpbusd    ymm9, ymm0, ymm2
    LONG $0x7079c1c4; WORD $0xf5d0 // vpshufd    xmm2, xmm8, 245
    LONG $0x00fde3c4; WORD $0x44d2 // vpermq    ymm2, ymm2, 68
    LONG $0x507562c4; BYTE $0xe2                                 // {vex}    vpdpbusd    ymm12, ymm1, ymm2
    LONG $0x507d62c4; BYTE $0xd2                                 // {vex}    vpdpbusd    ymm10, ymm0, ymm2
    LONG $0x707dc1c4; WORD $0xf5d0 // vpshufd    ymm2, ymm8, 245
    LONG $0xef15a1c4; WORD $0x0f64; BYTE $0x50 // vpxor    ymm4, ymm13, yword [rdi + r9 + 80]
    LONG $0x00fde3c4; WORD $0xeed2 // vpermq    ymm2, ymm2, 238
    LONG $0x5075e2c4; BYTE $0xda                                 // {vex}    vpdpbusd    ymm3, ymm1, ymm2
    LONG $0xef15a1c4; WORD $0x0f4c; BYTE $0x70 // vpxor    ymm1, ymm13, yword [rdi + r9 + 112]
    LONG $0x507d62c4; BYTE $0xda                                 // {vex}    vpdpbusd    ymm11, ymm0, ymm2
    LONG $0x7e6ffec5; BYTE $0x28 // vmovdqu    ymm7, yword [rsi + 40]
    LONG $0x3605e2c4; BYTE $0xc1 // vpermd    ymm0, ymm15, ymm1
    LONG $0x025d63c4; WORD $0xf0c0 // vpblendd    ymm8, ymm4, ymm0, 240
    QUAD $0x00026024847f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 608], ymm8
    LONG $0x3605e2c4; BYTE $0xc4 // vpermd    ymm0, ymm15, ymm4
    LONG $0x027d63c4; WORD $0xf0e9 // vpblendd    ymm13, ymm0, ymm1, 240
    QUAD $0x00024024ac7f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 576], ymm13
    LONG $0xc770f9c5; BYTE $0xa0 // vpshufd    xmm0, xmm7, 160
    LONG $0x00fde3c4; WORD $0x44d0 // vpermq    ymm2, ymm0, 68
    QUAD $0x0000c0248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 192]
    LONG $0xe1dbbdc5             // vpand    ymm4, ymm8, ymm1
    LONG $0xc470fdc5; BYTE $0x88 // vpshufd    ymm0, ymm4, 136
    LONG $0x507de2c4; BYTE $0xea                                 // {vex}    vpdpbusd    ymm5, ymm0, ymm2
    LONG $0xf9db15c5             // vpand    ymm15, ymm13, ymm1
    LONG $0x707d41c4; WORD $0x88c7 // vpshufd    ymm8, ymm15, 136
    LONG $0x503de2c4; BYTE $0xf2                                 // {vex}    vpdpbusd    ymm6, ymm8, ymm2
    LONG $0xd770fdc5; BYTE $0xa0 // vpshufd    ymm2, ymm7, 160
    LONG $0x00fde3c4; WORD $0xeed2 // vpermq    ymm2, ymm2, 238
    LONG $0x507d62c4; BYTE $0xf2                                 // {vex}    vpdpbusd    ymm14, ymm0, ymm2
    LONG $0x503d62c4; BYTE $0xca                                 // {vex}    vpdpbusd    ymm9, ymm8, ymm2
    LONG $0x6c6f7ec5; WORD $0x2816 // vmovdqu    ymm13, yword [rsi + rdx + 40]
    LONG $0x7079c1c4; WORD $0xa0d5 // vpshufd    xmm2, xmm13, 160
    QUAD $0x00014024ac7f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 320], ymm13
    LONG $0x00fde3c4; WORD $0x44d2 // vpermq    ymm2, ymm2, 68
    LONG $0x507d62c4; BYTE $0xe2                                 // {vex}    vpdpbusd    ymm12, ymm0, ymm2
    LONG $0x503d62c4; BYTE $0xd2                                 // {vex}    vpdpbusd    ymm10, ymm8, ymm2
    LONG $0x707dc1c4; WORD $0xa0d5 // vpshufd    ymm2, ymm13, 160
    LONG $0x00fde3c4; WORD $0xeed2 // vpermq    ymm2, ymm2, 238
    LONG $0x507de2c4; BYTE $0xda                                 // {vex}    vpdpbusd    ymm3, ymm0, ymm2
    LONG $0x503d62c4; BYTE $0xda                                 // {vex}    vpdpbusd    ymm11, ymm8, ymm2
    QUAD $0x00012024bc7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 288], ymm7
    LONG $0xc770f9c5; BYTE $0xf5 // vpshufd    xmm0, xmm7, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0xd470fdc5; BYTE $0xdd // vpshufd    ymm2, ymm4, 221
    LONG $0x506de2c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm5, ymm2, ymm0
    LONG $0x707dc1c4; WORD $0xdde7 // vpshufd    ymm4, ymm15, 221
    LONG $0x505de2c4; BYTE $0xf0                                 // {vex}    vpdpbusd    ymm6, ymm4, ymm0
    LONG $0xc770fdc5; BYTE $0xf5 // vpshufd    ymm0, ymm7, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x506d62c4; BYTE $0xf0                                 // {vex}    vpdpbusd    ymm14, ymm2, ymm0
    LONG $0x505d62c4; BYTE $0xc8                                 // {vex}    vpdpbusd    ymm9, ymm4, ymm0
    LONG $0x7079c1c4; WORD $0xf5c5 // vpshufd    xmm0, xmm13, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0x506d62c4; BYTE $0xe0                                 // {vex}    vpdpbusd    ymm12, ymm2, ymm0
    LONG $0x505d62c4; BYTE $0xd0                                 // {vex}    vpdpbusd    ymm10, ymm4, ymm0
    LONG $0x707dc1c4; WORD $0xf5c5 // vpshufd    ymm0, ymm13, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x506de2c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm3, ymm2, ymm0
    LONG $0x505d62c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm11, ymm4, ymm0
    LONG $0x446ffec5; WORD $0x6024 // vmovdqu    ymm0, yword [rsp + 96]
    LONG $0xd071fdc5; BYTE $0x04 // vpsrlw    ymm0, ymm0, 4
    LONG $0x546ffec5; WORD $0x4024 // vmovdqu    ymm2, yword [rsp + 64]
    LONG $0xd271edc5; BYTE $0x04 // vpsrlw    ymm2, ymm2, 4
    LONG $0xe96f7dc5             // vmovdqa    ymm13, ymm1
    LONG $0xe0dbf5c5             // vpand    ymm4, ymm1, ymm0
    LONG $0xfadbf5c5             // vpand    ymm7, ymm1, ymm2
    LONG $0x7c7ffec5; WORD $0x4024 // vmovdqu    yword [rsp + 64], ymm7
    LONG $0x7e6f7ec5; BYTE $0x48 // vmovdqu    ymm15, yword [rsi + 72]
    LONG $0x7079c1c4; WORD $0xa0c7 // vpshufd    xmm0, xmm15, 160
    LONG $0x00fde3c4; WORD $0x44d0 // vpermq    ymm2, ymm0, 68
    LONG $0xcc70fdc5; BYTE $0x88 // vpshufd    ymm1, ymm4, 136
    LONG $0x5075e2c4; BYTE $0xea                                 // {vex}    vpdpbusd    ymm5, ymm1, ymm2
    LONG $0xc7707dc5; BYTE $0x88 // vpshufd    ymm8, ymm7, 136
    LONG $0x503de2c4; BYTE $0xf2                                 // {vex}    vpdpbusd    ymm6, ymm8, ymm2
    LONG $0x707dc1c4; WORD $0xa0d7 // vpshufd    ymm2, ymm15, 160
    LONG $0x00fde3c4; WORD $0xeed2 // vpermq    ymm2, ymm2, 238
    LONG $0x507562c4; BYTE $0xf2                                 // {vex}    vpdpbusd    ymm14, ymm1, ymm2
    LONG $0x503d62c4; BYTE $0xca                                 // {vex}    vpdpbusd    ymm9, ymm8, ymm2
    LONG $0x7c6ffec5; WORD $0x4816 // vmovdqu    ymm7, yword [rsi + rdx + 72]
    LONG $0xd770f9c5; BYTE $0xa0 // vpshufd    xmm2, xmm7, 160
    LONG $0x00fde3c4; WORD $0x44d2 // vpermq    ymm2, ymm2, 68
    LONG $0x507562c4; BYTE $0xe2                                 // {vex}    vpdpbusd    ymm12, ymm1, ymm2
    LONG $0x503d62c4; BYTE $0xd2                                 // {vex}    vpdpbusd    ymm10, ymm8, ymm2
    LONG $0xd770fdc5; BYTE $0xa0 // vpshufd    ymm2, ymm7, 160
    LONG $0x00fde3c4; WORD $0xeed2 // vpermq    ymm2, ymm2, 238
    LONG $0x5075e2c4; BYTE $0xda                                 // {vex}    vpdpbusd    ymm3, ymm1, ymm2
    LONG $0x503d62c4; BYTE $0xda                                 // {vex}    vpdpbusd    ymm11, ymm8, ymm2
    LONG $0x7079c1c4; WORD $0xf5c7 // vpshufd    xmm0, xmm15, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0xd470fdc5; BYTE $0xdd // vpshufd    ymm2, ymm4, 221
    LONG $0x506de2c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm5, ymm2, ymm0
    LONG $0x6470fdc5; WORD $0x4024; BYTE $0xdd // vpshufd    ymm4, yword [rsp + 64], 221
    LONG $0x505de2c4; BYTE $0xf0                                 // {vex}    vpdpbusd    ymm6, ymm4, ymm0
    LONG $0x707dc1c4; WORD $0xf5c7 // vpshufd    ymm0, ymm15, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x506d62c4; BYTE $0xf0                                 // {vex}    vpdpbusd    ymm14, ymm2, ymm0
    LONG $0x505d62c4; BYTE $0xc8                                 // {vex}    vpdpbusd    ymm9, ymm4, ymm0
    LONG $0x7c7ffec5; WORD $0x6024 // vmovdqu    yword [rsp + 96], ymm7
    LONG $0xc770f9c5; BYTE $0xf5 // vpshufd    xmm0, xmm7, 245
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0x506d62c4; BYTE $0xe0                                 // {vex}    vpdpbusd    ymm12, ymm2, ymm0
    LONG $0x505d62c4; BYTE $0xd0                                 // {vex}    vpdpbusd    ymm10, ymm4, ymm0
    LONG $0xc770fdc5; BYTE $0xf5 // vpshufd    ymm0, ymm7, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x506de2c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm3, ymm2, ymm0
    LONG $0x505d62c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm11, ymm4, ymm0
    QUAD $0x00026024846ffec5; BYTE $0x00 // vmovdqu    ymm0, yword [rsp + 608]
    LONG $0xd071fdc5; BYTE $0x04 // vpsrlw    ymm0, ymm0, 4
    QUAD $0x000240248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 576]
    LONG $0xd171f5c5; BYTE $0x04 // vpsrlw    ymm1, ymm1, 4
    LONG $0xf8db95c5             // vpand    ymm7, ymm13, ymm0
    LONG $0xd1db95c5             // vpand    ymm2, ymm13, ymm1
    LONG $0x547ffec5; WORD $0x4024 // vmovdqu    yword [rsp + 64], ymm2
    LONG $0x666ffec5; BYTE $0x68 // vmovdqu    ymm4, yword [rsi + 104]
    LONG $0xc470f9c5; BYTE $0xa0 // vpshufd    xmm0, xmm4, 160
    LONG $0x00fd63c4; WORD $0x44c0 // vpermq    ymm8, ymm0, 68
    LONG $0xcf70fdc5; BYTE $0x88 // vpshufd    ymm1, ymm7, 136
    LONG $0x5075c2c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm5, ymm1, ymm8
    LONG $0xc270fdc5; BYTE $0x88 // vpshufd    ymm0, ymm2, 136
    LONG $0x507dc2c4; BYTE $0xf0                                 // {vex}    vpdpbusd    ymm6, ymm0, ymm8
    LONG $0xc4707dc5; BYTE $0xa0 // vpshufd    ymm8, ymm4, 160
    LONG $0x00fd43c4; WORD $0xeec0 // vpermq    ymm8, ymm8, 238
    LONG $0x507542c4; BYTE $0xf0                                 // {vex}    vpdpbusd    ymm14, ymm1, ymm8
    LONG $0x507d42c4; BYTE $0xc8                                 // {vex}    vpdpbusd    ymm9, ymm0, ymm8
    LONG $0x446f7ec5; WORD $0x6816 // vmovdqu    ymm8, yword [rsi + rdx + 104]
    LONG $0x707941c4; WORD $0xa0e8 // vpshufd    xmm13, xmm8, 160
    LONG $0x00fd43c4; WORD $0x44ed // vpermq    ymm13, ymm13, 68
    LONG $0x507542c4; BYTE $0xe5                                 // {vex}    vpdpbusd    ymm12, ymm1, ymm13
    LONG $0x507d42c4; BYTE $0xd5                                 // {vex}    vpdpbusd    ymm10, ymm0, ymm13
    LONG $0x707d41c4; WORD $0xa0e8 // vpshufd    ymm13, ymm8, 160
    LONG $0x00fd43c4; WORD $0xeeed // vpermq    ymm13, ymm13, 238
    LONG $0x5075c2c4; BYTE $0xdd                                 // {vex}    vpdpbusd    ymm3, ymm1, ymm13
    LONG $0xd36ffdc5             // vmovdqa    ymm2, ymm3
    LONG $0x507d42c4; BYTE $0xdd                                 // {vex}    vpdpbusd    ymm11, ymm0, ymm13
    LONG $0xc0eff9c5             // vpxor    xmm0, xmm0, xmm0
    QUAD $0x000200249c6ffec5; BYTE $0x00 // vmovdqu    ymm3, yword [rsp + 512]
    QUAD $0x008024845065e2c4; WORD $0x0000                       // {vex}    vpdpbusd    ymm0, ymm3, YMMWORD PTR [rsp + 128]
    QUAD $0x012024845065e2c4; WORD $0x0000                       // {vex}    vpdpbusd    ymm0, ymm3, YMMWORD PTR [rsp + 288]
    LONG $0x5065c2c4; BYTE $0xc7                                 // {vex}    vpdpbusd    ymm0, ymm3, ymm15
    LONG $0xcc70f9c5; BYTE $0xf5 // vpshufd    xmm1, xmm4, 245
    LONG $0x00fde3c4; WORD $0x44c9 // vpermq    ymm1, ymm1, 68
    LONG $0xef707dc5; BYTE $0xdd // vpshufd    ymm13, ymm7, 221
    LONG $0x5015e2c4; BYTE $0xe9                                 // {vex}    vpdpbusd    ymm5, ymm13, ymm1
    LONG $0x7c707dc5; WORD $0x4024; BYTE $0xdd // vpshufd    ymm15, yword [rsp + 64], 221
    LONG $0x5005e2c4; BYTE $0xf1                                 // {vex}    vpdpbusd    ymm6, ymm15, ymm1
    LONG $0xcc70fdc5; BYTE $0xf5 // vpshufd    ymm1, ymm4, 245
    LONG $0x00fde3c4; WORD $0xeec9 // vpermq    ymm1, ymm1, 238
    LONG $0x501562c4; BYTE $0xf1                                 // {vex}    vpdpbusd    ymm14, ymm13, ymm1
    LONG $0x500562c4; BYTE $0xc9                                 // {vex}    vpdpbusd    ymm9, ymm15, ymm1
    LONG $0x5065e2c4; BYTE $0xc4                                 // {vex}    vpdpbusd    ymm0, ymm3, ymm4
    LONG $0x7079c1c4; WORD $0xf5c8 // vpshufd    xmm1, xmm8, 245
    LONG $0x00fde3c4; WORD $0x44c9 // vpermq    ymm1, ymm1, 68
    LONG $0x501562c4; BYTE $0xe1                                 // {vex}    vpdpbusd    ymm12, ymm13, ymm1
    LONG $0x500562c4; BYTE $0xd1                                 // {vex}    vpdpbusd    ymm10, ymm15, ymm1
    LONG $0x137da2c4; WORD $0x0f0c // vcvtph2ps    ymm1, oword [rdi + r9]
    LONG $0x027de2c4; BYTE $0xe0 // vphaddd    ymm4, ymm0, ymm0
    LONG $0x707dc1c4; WORD $0xf5c0 // vpshufd    ymm0, ymm8, 245
    LONG $0x00fde3c4; WORD $0xeec0 // vpermq    ymm0, ymm0, 238
    LONG $0x5015e2c4; BYTE $0xd0                                 // {vex}    vpdpbusd    ymm2, ymm13, ymm0
    QUAD $0x00008024947ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 128], ymm2
    QUAD $0x0002e024ac107cc5; BYTE $0x00 // vmovups    ymm13, yword [rsp + 736]
    LONG $0x1379e2c4; BYTE $0x3e // vcvtph2ps    xmm7, qword [rsi]
    LONG $0x500562c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm11, ymm15, ymm0
    QUAD $0x0000a024846ffec5; BYTE $0x00 // vmovdqu    ymm0, yword [rsp + 160]
    LONG $0x367de2c4; BYTE $0xc4 // vpermd    ymm0, ymm0, ymm4
    LONG $0xf072fdc5; BYTE $0x03 // vpslld    ymm0, ymm0, 3
    LONG $0xd66cd5c5             // vpunpcklqdq    ymm2, ymm5, ymm6
    LONG $0xc0faedc5             // vpsubd    ymm0, ymm2, ymm0
    LONG $0x187de2c4; BYTE $0xd7 // vbroadcastss    ymm2, xmm7
    LONG $0xd159ecc5             // vmulps    ymm2, ymm2, ymm1
    LONG $0xc05bfcc5             // vcvtdq2ps    ymm0, ymm0
    LONG $0xb87d62c4; BYTE $0xea // vfmadd231ps    ymm13, ymm0, ymm2
    LONG $0xc66dd5c5             // vpunpckhqdq    ymm0, ymm5, ymm6
    QUAD $0x00018024b410fcc5; BYTE $0x00 // vmovups    ymm6, yword [rsp + 384]
    QUAD $0x0001c024ac6ffec5; BYTE $0x00 // vmovdqu    ymm5, yword [rsp + 448]
    LONG $0x3655e2c4; BYTE $0xd4 // vpermd    ymm2, ymm5, ymm4
    LONG $0xf272edc5; BYTE $0x03 // vpslld    ymm2, ymm2, 3
    LONG $0xc2fafdc5             // vpsubd    ymm0, ymm0, ymm2
    LONG $0xd716fac5             // vmovshdup    xmm2, xmm7
    LONG $0x187de2c4; BYTE $0xd2 // vbroadcastss    ymm2, xmm2
    LONG $0xc05bfcc5             // vcvtdq2ps    ymm0, ymm0
    LONG $0xd159ecc5             // vmulps    ymm2, ymm2, ymm1
    LONG $0x7c107cc5; WORD $0x2024 // vmovups    ymm15, yword [rsp + 32]
    LONG $0xb87d62c4; BYTE $0xfa // vfmadd231ps    ymm15, ymm0, ymm2
    LONG $0x7c117cc5; WORD $0x2024 // vmovups    yword [rsp + 32], ymm15
    QUAD $0x00016024846ffec5; BYTE $0x00 // vmovdqu    ymm0, yword [rsp + 352]
    LONG $0x367de2c4; BYTE $0xc4 // vpermd    ymm0, ymm0, ymm4
    LONG $0x6c0dc1c4; BYTE $0xd1 // vpunpcklqdq    ymm2, ymm14, ymm9
    LONG $0xf072fdc5; BYTE $0x03 // vpslld    ymm0, ymm0, 3
    LONG $0xc0faedc5             // vpsubd    ymm0, ymm2, ymm0
    LONG $0xd7c6c1c5; BYTE $0x01 // vshufpd    xmm2, xmm7, xmm7, 1
    LONG $0xc05bfcc5             // vcvtdq2ps    ymm0, ymm0
    LONG $0x187de2c4; BYTE $0xd2 // vbroadcastss    ymm2, xmm2
    LONG $0xd159ecc5             // vmulps    ymm2, ymm2, ymm1
    LONG $0x3c107cc5; BYTE $0x24 // vmovups    ymm15, yword [rsp]
    LONG $0xb87d62c4; BYTE $0xfa // vfmadd231ps    ymm15, ymm0, ymm2
    LONG $0x3c117cc5; BYTE $0x24 // vmovups    yword [rsp], ymm15
    LONG $0xc057f8c5             // vxorps    xmm0, xmm0, xmm0
    QUAD $0x028024845065e2c4; WORD $0x0000                       // {vex}    vpdpbusd    ymm0, ymm3, YMMWORD PTR [rsp + 640]
    QUAD $0x014024845065e2c4; WORD $0x0000                       // {vex}    vpdpbusd    ymm0, ymm3, YMMWORD PTR [rsp + 320]
    LONG $0x5065e2c4; WORD $0x2444; BYTE $0x60                   // {vex}    vpdpbusd    ymm0, ymm3, YMMWORD PTR [rsp + 96]
    LONG $0x5065c2c4; BYTE $0xc0                                 // {vex}    vpdpbusd    ymm0, ymm3, ymm8
    QUAD $0x0002a02484107cc5; BYTE $0x00 // vmovups    ymm8, yword [rsp + 672]
    LONG $0x6d0dc1c4; BYTE $0xd1 // vpunpckhqdq    ymm2, ymm14, ymm9
    QUAD $0x0001a0248c6f7ec5; BYTE $0x00 // vmovdqu    ymm9, yword [rsp + 416]
    LONG $0x3635e2c4; BYTE $0xdc // vpermd    ymm3, ymm9, ymm4
    LONG $0xf372e5c5; BYTE $0x03 // vpslld    ymm3, ymm3, 3
    LONG $0xd3faedc5             // vpsubd    ymm2, ymm2, ymm3
    LONG $0xdfc6c0c5; BYTE $0xff // vshufps    xmm3, xmm7, xmm7, 255
    QUAD $0x0002c024bc10fcc5; BYTE $0x00 // vmovups    ymm7, yword [rsp + 704]
    LONG $0x187de2c4; BYTE $0xdb // vbroadcastss    ymm3, xmm3
    LONG $0xd959e4c5             // vmulps    ymm3, ymm3, ymm1
    LONG $0xe25bfcc5             // vcvtdq2ps    ymm4, ymm2
    LONG $0x027de2c4; BYTE $0xd0 // vphaddd    ymm2, ymm0, ymm0
    LONG $0xb85d62c4; BYTE $0xc3 // vfmadd231ps    ymm8, ymm4, ymm3
    LONG $0x1379e2c4; WORD $0x1604 // vcvtph2ps    xmm0, qword [rsi + rdx]
    QUAD $0x0000a0249c6ffec5; BYTE $0x00 // vmovdqu    ymm3, yword [rsp + 160]
    LONG $0x3665e2c4; BYTE $0xda // vpermd    ymm3, ymm3, ymm2
    LONG $0x6c1dc1c4; BYTE $0xe2 // vpunpcklqdq    ymm4, ymm12, ymm10
    LONG $0xf372e5c5; BYTE $0x03 // vpslld    ymm3, ymm3, 3
    LONG $0xdbfaddc5             // vpsubd    ymm3, ymm4, ymm3
    LONG $0x187de2c4; BYTE $0xe0 // vbroadcastss    ymm4, xmm0
    LONG $0xdb5bfcc5             // vcvtdq2ps    ymm3, ymm3
    LONG $0xe159dcc5             // vmulps    ymm4, ymm4, ymm1
    LONG $0xb865e2c4; BYTE $0xfc // vfmadd231ps    ymm7, ymm3, ymm4
    LONG $0x6d1dc1c4; BYTE $0xda // vpunpckhqdq    ymm3, ymm12, ymm10
    LONG $0x3655e2c4; BYTE $0xe2 // vpermd    ymm4, ymm5, ymm2
    LONG $0xf472ddc5; BYTE $0x03 // vpslld    ymm4, ymm4, 3
    LONG $0xdcfae5c5             // vpsubd    ymm3, ymm3, ymm4
    LONG $0xe016fac5             // vmovshdup    xmm4, xmm0
    LONG $0xdb5bfcc5             // vcvtdq2ps    ymm3, ymm3
    LONG $0x187de2c4; BYTE $0xe4 // vbroadcastss    ymm4, xmm4
    LONG $0xe159dcc5             // vmulps    ymm4, ymm4, ymm1
    LONG $0xb865e2c4; BYTE $0xf4 // vfmadd231ps    ymm6, ymm3, ymm4
    QUAD $0x00008024946f7ec5; BYTE $0x00 // vmovdqu    ymm10, yword [rsp + 128]
    LONG $0x6c2dc1c4; BYTE $0xdb // vpunpcklqdq    ymm3, ymm10, ymm11
    QUAD $0x00016024a46ffec5; BYTE $0x00 // vmovdqu    ymm4, yword [rsp + 352]
    LONG $0x365de2c4; BYTE $0xe2 // vpermd    ymm4, ymm4, ymm2
    LONG $0xf472ddc5; BYTE $0x03 // vpslld    ymm4, ymm4, 3
    LONG $0xdcfae5c5             // vpsubd    ymm3, ymm3, ymm4
    LONG $0xdb5bfcc5             // vcvtdq2ps    ymm3, ymm3
    LONG $0xe0c6f9c5; BYTE $0x01 // vshufpd    xmm4, xmm0, xmm0, 1
    LONG $0x187de2c4; BYTE $0xe4 // vbroadcastss    ymm4, xmm4
    LONG $0xe159dcc5             // vmulps    ymm4, ymm4, ymm1
    QUAD $0x0000e024ac10fcc5; BYTE $0x00 // vmovups    ymm5, yword [rsp + 224]
    LONG $0xb865e2c4; BYTE $0xec // vfmadd231ps    ymm5, ymm3, ymm4
    QUAD $0x0000e024ac11fcc5; BYTE $0x00 // vmovups    yword [rsp + 224], ymm5
    QUAD $0x0000e024a410fcc5; BYTE $0x00 // vmovups    ymm4, yword [rsp + 224]
    LONG $0x6d2dc1c4; BYTE $0xdb // vpunpckhqdq    ymm3, ymm10, ymm11
    LONG $0x3635e2c4; BYTE $0xd2 // vpermd    ymm2, ymm9, ymm2
    LONG $0xf272edc5; BYTE $0x03 // vpslld    ymm2, ymm2, 3
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
    LONG $0x90c18149; WORD $0x0000; BYTE $0x00 // add    r9, 144
    WORD $0x394c; BYTE $0xc8     // cmp    rax, r9
	JNE LBB0_5
LBB0_6:
    WORD $0x6349; BYTE $0xc2     // movsxd    rax, r10d
    LONG $0x81148d48             // lea    rdx, [rcx + 4*rax]
    LONG $0x2c7f7ec5; BYTE $0x81 // vmovdqu    yword [rcx + 4*rax], ymm13
    LONG $0x4c10fcc5; WORD $0x2024 // vmovups    ymm1, yword [rsp + 32]
    LONG $0x117ca1c4; WORD $0x820c // vmovups    yword [rdx + 4*r8], ymm1
    LONG $0x0c10fcc5; BYTE $0x24 // vmovups    ymm1, yword [rsp]
    LONG $0x117ca1c4; WORD $0x9a0c // vmovups    yword [rdx + 4*r11], ymm1
    LONG $0x04117cc5; BYTE $0x9a // vmovups    yword [rdx + 4*rbx], ymm8
    LONG $0x7f7ea1c4; WORD $0xb23c // vmovdqu    yword [rdx + 4*r14], ymm7
    LONG $0x117ca1c4; WORD $0xba34 // vmovups    yword [rdx + 4*r15], ymm6
    LONG $0x117ca1c4; WORD $0xa224 // vmovups    yword [rdx + 4*r12], ymm4
    LONG $0x117ca1c4; WORD $0xaa04 // vmovups    yword [rdx + 4*r13], ymm0
    LONG $0x08c48148; WORD $0x0003; BYTE $0x00 // add    rsp, 776
    BYTE $0x5b                   // pop    rbx
    WORD $0x5c41                 // pop    r12
    WORD $0x5d41                 // pop    r13
    WORD $0x5e41                 // pop    r14
    WORD $0x5f41                 // pop    r15
    BYTE $0x5d                   // pop    rbp
    VZEROUPPER
    SUBQ $824, SP
    RET
