//go:build amd64

#include "textflag.h"

DATA LCDATA1<>+0x000(SB)/8, $0x0f0f0f0f88888888
DATA LCDATA1<>+0x008(SB)/8, $0x0000000000000000
DATA LCDATA1<>+0x010(SB)/8, $0x0000000000000000
DATA LCDATA1<>+0x018(SB)/8, $0x0000000000000000
DATA LCDATA1<>+0x020(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA LCDATA1<>+0x028(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA LCDATA1<>+0x030(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA LCDATA1<>+0x038(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA LCDATA1<>+0x040(SB)/8, $0x0101010101010101
DATA LCDATA1<>+0x048(SB)/8, $0x0101010101010101
DATA LCDATA1<>+0x050(SB)/8, $0x0101010101010101
DATA LCDATA1<>+0x058(SB)/8, $0x0101010101010101
DATA LCDATA1<>+0x060(SB)/8, $0x000000000000010f
GLOBL LCDATA1<>(SB), 8, $104

TEXT ·_go_llama_q4_0_q8_0_8x8(SB), $880-32

    MOVQ q4+0(FP), DI
    MOVQ q8+8(FP), SI
    MOVQ blocks+16(FP), DX
    MOVQ out+24(FP), CX
    ADDQ $880, SP
    PUSHQ BP
    LEAQ LCDATA1<>(SB), BP

    LONG $0x68ec8148; WORD $0x0003; BYTE $0x00 // sub    rsp, 872
    WORD $0xd285                 // test    edx, edx
	JLE LBB0_1
    WORD $0xd289                 // mov    edx, edx
    WORD $0x8948; BYTE $0xd0     // mov    rax, rdx
    LONG $0x07e0c148             // shl    rax, 7
    LONG $0xd0048d48             // lea    rax, [rax + 8*rdx]
    LONG $0x04e2c148             // shl    rdx, 4
    LONG $0xd2148d48             // lea    rdx, [rdx + 8*rdx]
    LONG $0xc057f8c5             // vxorps    xmm0, xmm0, xmm0
    LONG $0x4411fcc5; WORD $0x6024 // vmovups    yword [rsp + 96], ymm0
    LONG $0x187de2c4; WORD $0x0045 // vbroadcastss    ymm0, dword 0[rbp] /* [rip + .LCPI0_0] */
    QUAD $0x000260248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 608], ymm0
    WORD $0x3145; BYTE $0xc0     // xor    r8d, r8d
    LONG $0x187de2c4; WORD $0x0445 // vbroadcastss    ymm0, dword 4[rbp] /* [rip + .LCPI0_1] */
    QUAD $0x000240248411fcc5; BYTE $0x00 // vmovups    yword [rsp + 576], ymm0
    LONG $0xc057f8c5             // vxorps    xmm0, xmm0, xmm0
    LONG $0x4411fcc5; WORD $0x4024 // vmovups    yword [rsp + 64], ymm0
    LONG $0x4411fcc5; WORD $0x2024 // vmovups    yword [rsp + 32], ymm0
    LONG $0x0411fcc5; BYTE $0x24 // vmovups    yword [rsp], ymm0
    LONG $0xf657c8c5             // vxorps    xmm6, xmm6, xmm6
    LONG $0xff57c0c5             // vxorps    xmm7, xmm7, xmm7
    LONG $0xe457d8c5             // vxorps    xmm4, xmm4, xmm4
    LONG $0xc957f0c5             // vxorps    xmm1, xmm1, xmm1
LBB0_4:
    QUAD $0x0000c0248c11fcc5; BYTE $0x00 // vmovups    yword [rsp + 192], ymm1
    QUAD $0x0000e024a411fcc5; BYTE $0x00 // vmovups    yword [rsp + 224], ymm4
    QUAD $0x00032024bc11fcc5; BYTE $0x00 // vmovups    yword [rsp + 800], ymm7
    QUAD $0x00034024b411fcc5; BYTE $0x00 // vmovups    yword [rsp + 832], ymm6
    QUAD $0x00026024846ffec5; BYTE $0x00 // vmovdqu    ymm0, yword [rsp + 608]
    LONG $0xef7da1c4; WORD $0x074c; BYTE $0x10 // vpxor    ymm1, ymm0, yword [rdi + r8 + 16]
    LONG $0xef7da1c4; WORD $0x0754; BYTE $0x30 // vpxor    ymm2, ymm0, yword [rdi + r8 + 48]
    LONG $0x387563c4; WORD $0x01c2 // vinserti128    ymm8, ymm1, xmm2, 1
    LONG $0x4675e3c4; WORD $0x31fa // vperm2i128    ymm7, ymm1, ymm2, 49
    LONG $0xef7da1c4; WORD $0x0754; BYTE $0x50 // vpxor    ymm2, ymm0, yword [rdi + r8 + 80]
    LONG $0xef7da1c4; WORD $0x075c; BYTE $0x70 // vpxor    ymm3, ymm0, yword [rdi + r8 + 112]
    LONG $0x386de3c4; WORD $0x01cb // vinserti128    ymm1, ymm2, xmm3, 1
    LONG $0x466de3c4; WORD $0x31c3 // vperm2i128    ymm0, ymm2, ymm3, 49
    LONG $0x7165c1c4; WORD $0x04d0 // vpsrlw    ymm3, ymm8, 4
    LONG $0xd771ddc5; BYTE $0x04 // vpsrlw    ymm4, ymm7, 4
    LONG $0xd171d5c5; BYTE $0x04 // vpsrlw    ymm5, ymm1, 4
    LONG $0xd071cdc5; BYTE $0x04 // vpsrlw    ymm6, ymm0, 4
    LONG $0x787de2c4; WORD $0x6055 // vpbroadcastb    ymm2, byte 96[rbp] /* [rip + .LCPI0_4] */
    LONG $0xf2db65c5             // vpand    ymm14, ymm3, ymm2
    LONG $0xfadb55c5             // vpand    ymm15, ymm5, ymm2
    LONG $0xdadb5dc5             // vpand    ymm11, ymm4, ymm2
    QUAD $0x000080249c7f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 128], ymm11
    LONG $0xdadbcdc5             // vpand    ymm3, ymm6, ymm2
    QUAD $0x0000a0249c7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 160], ymm3
    LONG $0x707dc1c4; WORD $0x88e7 // vpshufd    ymm4, ymm15, 136
    QUAD $0x0002e024a47ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 736], ymm4
    LONG $0x707dc1c4; WORD $0x88d6 // vpshufd    ymm2, ymm14, 136
    QUAD $0x0001e024947ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 480], ymm2
    LONG $0xf370fdc5; BYTE $0x88 // vpshufd    ymm6, ymm3, 136
    QUAD $0x0002c024b47ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 704], ymm6
    LONG $0x566f7ec5; BYTE $0x48 // vmovdqu    ymm10, yword [rsi + 72]
    LONG $0x6e6f7ec5; BYTE $0x68 // vmovdqu    ymm13, yword [rsi + 104]
    LONG $0x7079c1c4; WORD $0xa0dd // vpshufd    xmm3, xmm13, 160
    QUAD $0x00028024ac7f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 640], ymm13
    LONG $0x00fde3c4; WORD $0x44db // vpermq    ymm3, ymm3, 68
    LONG $0xedefd1c5             // vpxor    xmm5, xmm5, xmm5
    LONG $0x505de2c4; BYTE $0xeb                                 // {vex}    vpdpbusd    ymm5, ymm4, ymm3
    LONG $0xef3141c4; BYTE $0xc9 // vpxor    xmm9, xmm9, xmm9
    LONG $0x504d62c4; BYTE $0xcb                                 // {vex}    vpdpbusd    ymm9, ymm6, ymm3
    LONG $0x7079c1c4; WORD $0xa0da // vpshufd    xmm3, xmm10, 160
    LONG $0x00fde3c4; WORD $0x44db // vpermq    ymm3, ymm3, 68
    LONG $0x506de2c4; BYTE $0xeb                                 // {vex}    vpdpbusd    ymm5, ymm2, ymm3
    LONG $0x707dc1c4; WORD $0x88d3 // vpshufd    ymm2, ymm11, 136
    QUAD $0x00018024947ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 384], ymm2
    LONG $0x506d62c4; BYTE $0xcb                                 // {vex}    vpdpbusd    ymm9, ymm2, ymm3
    QUAD $0x000240249c6ffec5; BYTE $0x00 // vmovdqu    ymm3, yword [rsp + 576]
    LONG $0xd3dbf5c5             // vpand    ymm2, ymm1, ymm3
    LONG $0xe3db7dc5             // vpand    ymm12, ymm0, ymm3
    LONG $0x5e6f7ec5; BYTE $0x28 // vmovdqu    ymm11, yword [rsi + 40]
    LONG $0x7079c1c4; WORD $0xa0cb // vpshufd    xmm1, xmm11, 160
    LONG $0x00fde3c4; WORD $0x44c9 // vpermq    ymm1, ymm1, 68
    LONG $0xc270fdc5; BYTE $0x88 // vpshufd    ymm0, ymm2, 136
    QUAD $0x0001c024847ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 448], ymm0
    LONG $0x507de2c4; BYTE $0xe9                                 // {vex}    vpdpbusd    ymm5, ymm0, ymm1
    LONG $0x707dc1c4; WORD $0x88c4 // vpshufd    ymm0, ymm12, 136
    QUAD $0x00016024847ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 352], ymm0
    LONG $0x507d62c4; BYTE $0xc9                                 // {vex}    vpdpbusd    ymm9, ymm0, ymm1
    LONG $0xc3db3dc5             // vpand    ymm8, ymm8, ymm3
    LONG $0xe3dbc5c5             // vpand    ymm4, ymm7, ymm3
    LONG $0x766ffec5; BYTE $0x08 // vmovdqu    ymm6, yword [rsi + 8]
    LONG $0xfe70f9c5; BYTE $0xa0 // vpshufd    xmm7, xmm6, 160
    LONG $0x00fde3c4; WORD $0x44c7 // vpermq    ymm0, ymm7, 68
    LONG $0x707dc1c4; WORD $0x88c8 // vpshufd    ymm1, ymm8, 136
    QUAD $0x0001a0248c7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 416], ymm1
    LONG $0x5075e2c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm5, ymm1, ymm0
    LONG $0xcc70fdc5; BYTE $0x88 // vpshufd    ymm1, ymm4, 136
    QUAD $0x000140248c7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 320], ymm1
    LONG $0x507562c4; BYTE $0xc8                                 // {vex}    vpdpbusd    ymm9, ymm1, ymm0
    LONG $0x707d41c4; WORD $0xddff // vpshufd    ymm15, ymm15, 221
    QUAD $0x0002a024bc7f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 672], ymm15
    LONG $0x707dc1c4; WORD $0xddfe // vpshufd    ymm7, ymm14, 221
    QUAD $0x00020024bc7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 512], ymm7
    LONG $0xca70fdc5; BYTE $0xdd // vpshufd    ymm1, ymm2, 221
    QUAD $0x000220248c7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 544], ymm1
    LONG $0x707dc1c4; WORD $0xddd8 // vpshufd    ymm3, ymm8, 221
    QUAD $0x000120249c7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 288], ymm3
    LONG $0x7079c1c4; WORD $0xf5c5 // vpshufd    xmm0, xmm13, 245
    LONG $0x00fde3c4; WORD $0x44d0 // vpermq    ymm2, ymm0, 68
    LONG $0xc0eff9c5             // vpxor    xmm0, xmm0, xmm0
    LONG $0x5005e2c4; BYTE $0xc2                                 // {vex}    vpdpbusd    ymm0, ymm15, ymm2
    LONG $0x707941c4; WORD $0xf5c2 // vpshufd    xmm8, xmm10, 245
    LONG $0x00fd43c4; WORD $0x44c0 // vpermq    ymm8, ymm8, 68
    LONG $0x5045c2c4; BYTE $0xc0                                 // {vex}    vpdpbusd    ymm0, ymm7, ymm8
    LONG $0x707941c4; WORD $0xf5eb // vpshufd    xmm13, xmm11, 245
    LONG $0x00fd43c4; WORD $0x44fd // vpermq    ymm15, ymm13, 68
    LONG $0x5075c2c4; BYTE $0xc7                                 // {vex}    vpdpbusd    ymm0, ymm1, ymm15
    LONG $0xee7079c5; BYTE $0xf5 // vpshufd    xmm13, xmm6, 245
    LONG $0x00fdc3c4; WORD $0x44cd // vpermq    ymm1, ymm13, 68
    LONG $0x5065e2c4; BYTE $0xc1                                 // {vex}    vpdpbusd    ymm0, ymm3, ymm1
    LONG $0xc5fefdc5             // vpaddd    ymm0, ymm0, ymm5
    QUAD $0x0000a0249c70fdc5; WORD $0xdd00 // vpshufd    ymm3, yword [rsp + 160], 221
    QUAD $0x000100249c7ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 256], ymm3
    LONG $0xef0941c4; BYTE $0xf6 // vpxor    xmm14, xmm14, xmm14
    LONG $0x506562c4; BYTE $0xf2                                 // {vex}    vpdpbusd    ymm14, ymm3, ymm2
    QUAD $0x000080249470fdc5; WORD $0xdd00 // vpshufd    ymm2, yword [rsp + 128], 221
    QUAD $0x0000a024947ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 160], ymm2
    LONG $0x506d42c4; BYTE $0xf0                                 // {vex}    vpdpbusd    ymm14, ymm2, ymm8
    LONG $0x707dc1c4; WORD $0xddd4 // vpshufd    ymm2, ymm12, 221
    QUAD $0x00008024947ffec5; BYTE $0x00 // vmovdqu    yword [rsp + 128], ymm2
    LONG $0x506d42c4; BYTE $0xf7                                 // {vex}    vpdpbusd    ymm14, ymm2, ymm15
    LONG $0xec707dc5; BYTE $0xdd // vpshufd    ymm13, ymm4, 221
    LONG $0x501562c4; BYTE $0xf1                                 // {vex}    vpdpbusd    ymm14, ymm13, ymm1
    QUAD $0x00030024ac7f7ec5; BYTE $0x00 // vmovdqu    yword [rsp + 768], ymm13
    LONG $0xfe0dc1c4; BYTE $0xc9 // vpaddd    ymm1, ymm14, ymm9
    LONG $0xd2efe9c5             // vpxor    xmm2, xmm2, xmm2
    LONG $0x787de2c4; WORD $0x615d // vpbroadcastb    ymm3, byte 97[rbp] /* [rip + .LCPI0_5] */
    LONG $0x5065e2c4; BYTE $0xd6                                 // {vex}    vpdpbusd    ymm2, ymm3, ymm6
    LONG $0x5065c2c4; BYTE $0xd3                                 // {vex}    vpdpbusd    ymm2, ymm3, ymm11
    LONG $0x5065c2c4; BYTE $0xd2                                 // {vex}    vpdpbusd    ymm2, ymm3, ymm10
    QUAD $0x000280248c6f7ec5; BYTE $0x00 // vmovdqu    ymm9, yword [rsp + 640]
    LONG $0x5065c2c4; BYTE $0xd1                                 // {vex}    vpdpbusd    ymm2, ymm3, ymm9
    LONG $0x026de2c4; BYTE $0xd2 // vphaddd    ymm2, ymm2, ymm2
    LONG $0xf272ddc5; BYTE $0x03 // vpslld    ymm4, ymm2, 3
    LONG $0xd16cfdc5             // vpunpcklqdq    ymm2, ymm0, ymm1
    LONG $0x587de2c4; BYTE $0xdc // vpbroadcastd    ymm3, xmm4
    LONG $0xd3faedc5             // vpsubd    ymm2, ymm2, ymm3
    LONG $0x1ec4f9c5; BYTE $0x00 // vpinsrw    xmm3, xmm0, word [rsi], 0
    LONG $0x1379e2c4; BYTE $0xdb // vcvtph2ps    xmm3, xmm3
    LONG $0x187de2c4; BYTE $0xdb // vbroadcastss    ymm3, xmm3
    LONG $0x137d22c4; WORD $0x073c // vcvtph2ps    ymm15, oword [rdi + r8]
    LONG $0xdb5984c5             // vmulps    ymm3, ymm15, ymm3
    LONG $0xd25bfcc5             // vcvtdq2ps    ymm2, ymm2
    LONG $0x2c10fcc5; BYTE $0x24 // vmovups    ymm5, yword [rsp]
    LONG $0xb865e2c4; BYTE $0xea // vfmadd231ps    ymm5, ymm3, ymm2
    LONG $0x2c11fcc5; BYTE $0x24 // vmovups    yword [rsp], ymm5
    LONG $0xc16dfdc5             // vpunpckhqdq    ymm0, ymm0, ymm1
    LONG $0xcc70f9c5; BYTE $0x55 // vpshufd    xmm1, xmm4, 85
    LONG $0x587de2c4; BYTE $0xc9 // vpbroadcastd    ymm1, xmm1
    LONG $0x56c4f9c5; WORD $0x0002 // vpinsrw    xmm2, xmm0, word [rsi + 2], 0
    LONG $0xc1fafdc5             // vpsubd    ymm0, ymm0, ymm1
    LONG $0x1379e2c4; BYTE $0xca // vcvtph2ps    xmm1, xmm2
    LONG $0x187de2c4; BYTE $0xc9 // vbroadcastss    ymm1, xmm1
    LONG $0xc95984c5             // vmulps    ymm1, ymm15, ymm1
    LONG $0xc05bfcc5             // vcvtdq2ps    ymm0, ymm0
    LONG $0x5410fcc5; WORD $0x2024 // vmovups    ymm2, yword [rsp + 32]
    LONG $0xb875e2c4; BYTE $0xd0 // vfmadd231ps    ymm2, ymm1, ymm0
    LONG $0x5411fcc5; WORD $0x2024 // vmovups    yword [rsp + 32], ymm2
    LONG $0x707dc1c4; WORD $0xa0c1 // vpshufd    ymm0, ymm9, 160
    LONG $0x00fde3c4; WORD $0xeec8 // vpermq    ymm1, ymm0, 238
    LONG $0xd257e8c5             // vxorps    xmm2, xmm2, xmm2
    QUAD $0x0002e024bc6ffec5; BYTE $0x00 // vmovdqu    ymm7, yword [rsp + 736]
    LONG $0x5045e2c4; BYTE $0xd1                                 // {vex}    vpdpbusd    ymm2, ymm7, ymm1
    LONG $0xc0eff9c5             // vpxor    xmm0, xmm0, xmm0
    QUAD $0x0002c024b46f7ec5; BYTE $0x00 // vmovdqu    ymm14, yword [rsp + 704]
    LONG $0x500de2c4; BYTE $0xc1                                 // {vex}    vpdpbusd    ymm0, ymm14, ymm1
    LONG $0x707dc1c4; WORD $0xa0ca // vpshufd    ymm1, ymm10, 160
    LONG $0x00fde3c4; WORD $0xeec9 // vpermq    ymm1, ymm1, 238
    QUAD $0x0001e0249c6ffec5; BYTE $0x00 // vmovdqu    ymm3, yword [rsp + 480]
    LONG $0x5065e2c4; BYTE $0xd1                                 // {vex}    vpdpbusd    ymm2, ymm3, ymm1
    QUAD $0x000180249c6ffec5; BYTE $0x00 // vmovdqu    ymm3, yword [rsp + 384]
    LONG $0x5065e2c4; BYTE $0xc1                                 // {vex}    vpdpbusd    ymm0, ymm3, ymm1
    LONG $0x707dc1c4; WORD $0xa0cb // vpshufd    ymm1, ymm11, 160
    LONG $0x00fde3c4; WORD $0xeec9 // vpermq    ymm1, ymm1, 238
    QUAD $0x0001c0249c6ffec5; BYTE $0x00 // vmovdqu    ymm3, yword [rsp + 448]
    LONG $0x5065e2c4; BYTE $0xd1                                 // {vex}    vpdpbusd    ymm2, ymm3, ymm1
    QUAD $0x000160249c6ffec5; BYTE $0x00 // vmovdqu    ymm3, yword [rsp + 352]
    LONG $0x5065e2c4; BYTE $0xc1                                 // {vex}    vpdpbusd    ymm0, ymm3, ymm1
    LONG $0xce70fdc5; BYTE $0xa0 // vpshufd    ymm1, ymm6, 160
    LONG $0x00fde3c4; WORD $0xeec9 // vpermq    ymm1, ymm1, 238
    QUAD $0x0001a0249c6ffec5; BYTE $0x00 // vmovdqu    ymm3, yword [rsp + 416]
    LONG $0x5065e2c4; BYTE $0xd1                                 // {vex}    vpdpbusd    ymm2, ymm3, ymm1
    QUAD $0x000140249c6ffec5; BYTE $0x00 // vmovdqu    ymm3, yword [rsp + 320]
    LONG $0x5065e2c4; BYTE $0xc1                                 // {vex}    vpdpbusd    ymm0, ymm3, ymm1
    LONG $0x707dc1c4; WORD $0xf5c9 // vpshufd    ymm1, ymm9, 245
    LONG $0x00fde3c4; WORD $0xeec9 // vpermq    ymm1, ymm1, 238
    LONG $0xdbefe1c5             // vpxor    xmm3, xmm3, xmm3
    QUAD $0x0002a024a46f7ec5; BYTE $0x00 // vmovdqu    ymm12, yword [rsp + 672]
    LONG $0x501de2c4; BYTE $0xd9                                 // {vex}    vpdpbusd    ymm3, ymm12, ymm1
    LONG $0x707d41c4; WORD $0xf5c2 // vpshufd    ymm8, ymm10, 245
    LONG $0x00fd43c4; WORD $0xeec0 // vpermq    ymm8, ymm8, 238
    QUAD $0x00020024ac6ffec5; BYTE $0x00 // vmovdqu    ymm5, yword [rsp + 512]
    LONG $0x5055c2c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm3, ymm5, ymm8
    LONG $0x707d41c4; WORD $0xf5d3 // vpshufd    ymm10, ymm11, 245
    LONG $0x00fd43c4; WORD $0xeed2 // vpermq    ymm10, ymm10, 238
    QUAD $0x00022024ac6ffec5; BYTE $0x00 // vmovdqu    ymm5, yword [rsp + 544]
    LONG $0x5055c2c4; BYTE $0xda                                 // {vex}    vpdpbusd    ymm3, ymm5, ymm10
    LONG $0xf670fdc5; BYTE $0xf5 // vpshufd    ymm6, ymm6, 245
    LONG $0x00fde3c4; WORD $0xeef6 // vpermq    ymm6, ymm6, 238
    QUAD $0x00012024ac6ffec5; BYTE $0x00 // vmovdqu    ymm5, yword [rsp + 288]
    LONG $0x5055e2c4; BYTE $0xde                                 // {vex}    vpdpbusd    ymm3, ymm5, ymm6
    LONG $0xd2fee5c5             // vpaddd    ymm2, ymm3, ymm2
    LONG $0xdbefe1c5             // vpxor    xmm3, xmm3, xmm3
    QUAD $0x00010024ac6ffec5; BYTE $0x00 // vmovdqu    ymm5, yword [rsp + 256]
    LONG $0x5055e2c4; BYTE $0xd9                                 // {vex}    vpdpbusd    ymm3, ymm5, ymm1
    QUAD $0x0000a0248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 160]
    LONG $0x5075c2c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm3, ymm1, ymm8
    QUAD $0x000080248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 128]
    LONG $0x5075c2c4; BYTE $0xda                                 // {vex}    vpdpbusd    ymm3, ymm1, ymm10
    LONG $0x5015e2c4; BYTE $0xde                                 // {vex}    vpdpbusd    ymm3, ymm13, ymm6
    LONG $0xc0fee5c5             // vpaddd    ymm0, ymm3, ymm0
    LONG $0x397de3c4; WORD $0x01e1 // vextracti128    xmm1, ymm4, 1
    LONG $0x5ec4f9c5; WORD $0x0004 // vpinsrw    xmm3, xmm0, word [rsi + 4], 0
    LONG $0x587de2c4; BYTE $0xc9 // vpbroadcastd    ymm1, xmm1
    LONG $0x1379e2c4; BYTE $0xdb // vcvtph2ps    xmm3, xmm3
    LONG $0xf06cedc5             // vpunpcklqdq    ymm6, ymm2, ymm0
    LONG $0xc9facdc5             // vpsubd    ymm1, ymm6, ymm1
    LONG $0x187de2c4; BYTE $0xdb // vbroadcastss    ymm3, xmm3
    LONG $0xdb5984c5             // vmulps    ymm3, ymm15, ymm3
    LONG $0xc95bfcc5             // vcvtdq2ps    ymm1, ymm1
    LONG $0x6c10fcc5; WORD $0x4024 // vmovups    ymm5, yword [rsp + 64]
    LONG $0xb865e2c4; BYTE $0xe9 // vfmadd231ps    ymm5, ymm3, ymm1
    LONG $0x6c11fcc5; WORD $0x4024 // vmovups    yword [rsp + 64], ymm5
    LONG $0xc06dedc5             // vpunpckhqdq    ymm0, ymm2, ymm0
    LONG $0xcc70fdc5; BYTE $0x55 // vpshufd    ymm1, ymm4, 85
    LONG $0x00fde3c4; WORD $0xaac9 // vpermq    ymm1, ymm1, 170
    LONG $0x56c4f9c5; WORD $0x0006 // vpinsrw    xmm2, xmm0, word [rsi + 6], 0
    LONG $0x1379e2c4; BYTE $0xd2 // vcvtph2ps    xmm2, xmm2
    LONG $0xc1fafdc5             // vpsubd    ymm0, ymm0, ymm1
    LONG $0x187de2c4; BYTE $0xca // vbroadcastss    ymm1, xmm2
    LONG $0xc95984c5             // vmulps    ymm1, ymm15, ymm1
    LONG $0xc05bfcc5             // vcvtdq2ps    ymm0, ymm0
    LONG $0x5410fcc5; WORD $0x6024 // vmovups    ymm2, yword [rsp + 96]
    LONG $0xb875e2c4; BYTE $0xd0 // vfmadd231ps    ymm2, ymm1, ymm0
    LONG $0x5411fcc5; WORD $0x6024 // vmovups    yword [rsp + 96], ymm2
    LONG $0x4c6f7ec5; WORD $0x6806 // vmovdqu    ymm9, yword [rsi + rax + 104]
    LONG $0x7079c1c4; WORD $0xa0c1 // vpshufd    xmm0, xmm9, 160
    LONG $0x00fde3c4; WORD $0x44c0 // vpermq    ymm0, ymm0, 68
    LONG $0xe4efd9c5             // vpxor    xmm4, xmm4, xmm4
    LONG $0x5045e2c4; BYTE $0xe0                                 // {vex}    vpdpbusd    ymm4, ymm7, ymm0
    LONG $0xd257e8c5             // vxorps    xmm2, xmm2, xmm2
    LONG $0x500de2c4; BYTE $0xd0                                 // {vex}    vpdpbusd    ymm2, ymm14, ymm0
    LONG $0x707dc1c4; WORD $0xa0c1 // vpshufd    ymm0, ymm9, 160
    LONG $0x00fde3c4; WORD $0xeef0 // vpermq    ymm6, ymm0, 238
    LONG $0xed57d0c5             // vxorps    xmm5, xmm5, xmm5
    LONG $0x5045e2c4; BYTE $0xee                                 // {vex}    vpdpbusd    ymm5, ymm7, ymm6
    LONG $0xdb57e0c5             // vxorps    xmm3, xmm3, xmm3
    LONG $0x500de2c4; BYTE $0xde                                 // {vex}    vpdpbusd    ymm3, ymm14, ymm6
    LONG $0x746f7ec5; WORD $0x4806 // vmovdqu    ymm14, yword [rsi + rax + 72]
    LONG $0x7079c1c4; WORD $0xa0f6 // vpshufd    xmm6, xmm14, 160
    LONG $0x00fde3c4; WORD $0x44f6 // vpermq    ymm6, ymm6, 68
    QUAD $0x0001e024846ffec5; BYTE $0x00 // vmovdqu    ymm0, yword [rsp + 480]
    LONG $0x507de2c4; BYTE $0xe6                                 // {vex}    vpdpbusd    ymm4, ymm0, ymm6
    QUAD $0x000180248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 384]
    LONG $0x5075e2c4; BYTE $0xd6                                 // {vex}    vpdpbusd    ymm2, ymm1, ymm6
    LONG $0x707dc1c4; WORD $0xa0f6 // vpshufd    ymm6, ymm14, 160
    LONG $0x00fde3c4; WORD $0xeef6 // vpermq    ymm6, ymm6, 238
    LONG $0x507de2c4; BYTE $0xee                                 // {vex}    vpdpbusd    ymm5, ymm0, ymm6
    LONG $0x5075e2c4; BYTE $0xde                                 // {vex}    vpdpbusd    ymm3, ymm1, ymm6
    LONG $0x5c6f7ec5; WORD $0x2806 // vmovdqu    ymm11, yword [rsi + rax + 40]
    LONG $0x7079c1c4; WORD $0xa0f3 // vpshufd    xmm6, xmm11, 160
    LONG $0x00fde3c4; WORD $0x44f6 // vpermq    ymm6, ymm6, 68
    QUAD $0x0001c024846ffec5; BYTE $0x00 // vmovdqu    ymm0, yword [rsp + 448]
    LONG $0x507de2c4; BYTE $0xe6                                 // {vex}    vpdpbusd    ymm4, ymm0, ymm6
    QUAD $0x000160248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 352]
    LONG $0x5075e2c4; BYTE $0xd6                                 // {vex}    vpdpbusd    ymm2, ymm1, ymm6
    LONG $0x707dc1c4; WORD $0xa0f3 // vpshufd    ymm6, ymm11, 160
    LONG $0x00fde3c4; WORD $0xeef6 // vpermq    ymm6, ymm6, 238
    LONG $0x507de2c4; BYTE $0xee                                 // {vex}    vpdpbusd    ymm5, ymm0, ymm6
    LONG $0x5075e2c4; BYTE $0xde                                 // {vex}    vpdpbusd    ymm3, ymm1, ymm6
    LONG $0x746ffec5; WORD $0x0806 // vmovdqu    ymm6, yword [rsi + rax + 8]
    LONG $0xc67079c5; BYTE $0xa0 // vpshufd    xmm8, xmm6, 160
    LONG $0x00fd43c4; WORD $0x44c0 // vpermq    ymm8, ymm8, 68
    QUAD $0x0001a024846ffec5; BYTE $0x00 // vmovdqu    ymm0, yword [rsp + 416]
    LONG $0x507dc2c4; BYTE $0xe0                                 // {vex}    vpdpbusd    ymm4, ymm0, ymm8
    QUAD $0x000140248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 320]
    LONG $0x5075c2c4; BYTE $0xd0                                 // {vex}    vpdpbusd    ymm2, ymm1, ymm8
    LONG $0xc6707dc5; BYTE $0xa0 // vpshufd    ymm8, ymm6, 160
    LONG $0x00fd43c4; WORD $0xeec0 // vpermq    ymm8, ymm8, 238
    LONG $0x507dc2c4; BYTE $0xe8                                 // {vex}    vpdpbusd    ymm5, ymm0, ymm8
    LONG $0x5075c2c4; BYTE $0xd8                                 // {vex}    vpdpbusd    ymm3, ymm1, ymm8
    LONG $0x7079c1c4; WORD $0xf5f9 // vpshufd    xmm7, xmm9, 245
    LONG $0x00fd63c4; WORD $0x44d7 // vpermq    ymm10, ymm7, 68
    LONG $0xffefc1c5             // vpxor    xmm7, xmm7, xmm7
    LONG $0x6f7dc1c4; BYTE $0xc4 // vmovdqa    ymm0, ymm12
    LONG $0x501dc2c4; BYTE $0xfa                                 // {vex}    vpdpbusd    ymm7, ymm12, ymm10
    LONG $0xef3941c4; BYTE $0xc0 // vpxor    xmm8, xmm8, xmm8
    QUAD $0x000100248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 256]
    LONG $0x507542c4; BYTE $0xc2                                 // {vex}    vpdpbusd    ymm8, ymm1, ymm10
    LONG $0x707d41c4; WORD $0xf5d1 // vpshufd    ymm10, ymm9, 245
    LONG $0x00fd43c4; WORD $0xeee2 // vpermq    ymm12, ymm10, 238
    LONG $0xef2941c4; BYTE $0xd2 // vpxor    xmm10, xmm10, xmm10
    LONG $0x507d42c4; BYTE $0xd4                                 // {vex}    vpdpbusd    ymm10, ymm0, ymm12
    LONG $0xc0eff9c5             // vpxor    xmm0, xmm0, xmm0
    LONG $0x5075c2c4; BYTE $0xc4                                 // {vex}    vpdpbusd    ymm0, ymm1, ymm12
    LONG $0x707941c4; WORD $0xf5e6 // vpshufd    xmm12, xmm14, 245
    LONG $0x00fd43c4; WORD $0x44e4 // vpermq    ymm12, ymm12, 68
    QUAD $0x00020024ac6f7ec5; BYTE $0x00 // vmovdqu    ymm13, yword [rsp + 512]
    LONG $0x5015c2c4; BYTE $0xfc                                 // {vex}    vpdpbusd    ymm7, ymm13, ymm12
    QUAD $0x0000a0248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 160]
    LONG $0x507542c4; BYTE $0xc4                                 // {vex}    vpdpbusd    ymm8, ymm1, ymm12
    LONG $0x707d41c4; WORD $0xf5e6 // vpshufd    ymm12, ymm14, 245
    LONG $0x00fd43c4; WORD $0xeee4 // vpermq    ymm12, ymm12, 238
    LONG $0x501542c4; BYTE $0xd4                                 // {vex}    vpdpbusd    ymm10, ymm13, ymm12
    LONG $0x5075c2c4; BYTE $0xc4                                 // {vex}    vpdpbusd    ymm0, ymm1, ymm12
    LONG $0x707941c4; WORD $0xf5e3 // vpshufd    xmm12, xmm11, 245
    LONG $0x00fd43c4; WORD $0x44e4 // vpermq    ymm12, ymm12, 68
    QUAD $0x00022024ac6f7ec5; BYTE $0x00 // vmovdqu    ymm13, yword [rsp + 544]
    LONG $0x5015c2c4; BYTE $0xfc                                 // {vex}    vpdpbusd    ymm7, ymm13, ymm12
    QUAD $0x000080248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 128]
    LONG $0x507542c4; BYTE $0xc4                                 // {vex}    vpdpbusd    ymm8, ymm1, ymm12
    LONG $0x707d41c4; WORD $0xf5e3 // vpshufd    ymm12, ymm11, 245
    LONG $0x00fd43c4; WORD $0xeee4 // vpermq    ymm12, ymm12, 238
    LONG $0x501542c4; BYTE $0xd4                                 // {vex}    vpdpbusd    ymm10, ymm13, ymm12
    LONG $0x5075c2c4; BYTE $0xc4                                 // {vex}    vpdpbusd    ymm0, ymm1, ymm12
    LONG $0xe67079c5; BYTE $0xf5 // vpshufd    xmm12, xmm6, 245
    LONG $0x00fd43c4; WORD $0x44e4 // vpermq    ymm12, ymm12, 68
    QUAD $0x00012024ac6f7ec5; BYTE $0x00 // vmovdqu    ymm13, yword [rsp + 288]
    LONG $0x5015c2c4; BYTE $0xfc                                 // {vex}    vpdpbusd    ymm7, ymm13, ymm12
    QUAD $0x000300248c6ffec5; BYTE $0x00 // vmovdqu    ymm1, yword [rsp + 768]
    LONG $0x507542c4; BYTE $0xc4                                 // {vex}    vpdpbusd    ymm8, ymm1, ymm12
    LONG $0xe6707dc5; BYTE $0xf5 // vpshufd    ymm12, ymm6, 245
    LONG $0x00fd43c4; WORD $0xeee4 // vpermq    ymm12, ymm12, 238
    LONG $0x501542c4; BYTE $0xd4                                 // {vex}    vpdpbusd    ymm10, ymm13, ymm12
    LONG $0x5075c2c4; BYTE $0xc4                                 // {vex}    vpdpbusd    ymm0, ymm1, ymm12
    LONG $0xe4fec5c5             // vpaddd    ymm4, ymm7, ymm4
    QUAD $0x00032024bc10fcc5; BYTE $0x00 // vmovups    ymm7, yword [rsp + 800]
    LONG $0xd2febdc5             // vpaddd    ymm2, ymm8, ymm2
    LONG $0xc5fe2dc5             // vpaddd    ymm8, ymm10, ymm5
    LONG $0xc3fefdc5             // vpaddd    ymm0, ymm0, ymm3
    LONG $0xca6cddc5             // vpunpcklqdq    ymm1, ymm4, ymm2
    LONG $0xe26dddc5             // vpunpckhqdq    ymm4, ymm4, ymm2
    LONG $0xd2efe9c5             // vpxor    xmm2, xmm2, xmm2
    LONG $0x787de2c4; WORD $0x615d // vpbroadcastb    ymm3, byte 97[rbp] /* [rip + .LCPI0_5] */
    LONG $0x5065e2c4; BYTE $0xd6                                 // {vex}    vpdpbusd    ymm2, ymm3, ymm6
    QUAD $0x00034024b410fcc5; BYTE $0x00 // vmovups    ymm6, yword [rsp + 832]
    LONG $0x5065c2c4; BYTE $0xd3                                 // {vex}    vpdpbusd    ymm2, ymm3, ymm11
    LONG $0x5065c2c4; BYTE $0xd6                                 // {vex}    vpdpbusd    ymm2, ymm3, ymm14
    LONG $0x5065c2c4; BYTE $0xd1                                 // {vex}    vpdpbusd    ymm2, ymm3, ymm9
    LONG $0x026de2c4; BYTE $0xd2 // vphaddd    ymm2, ymm2, ymm2
    LONG $0xf272edc5; BYTE $0x03 // vpslld    ymm2, ymm2, 3
    LONG $0x587de2c4; BYTE $0xea // vpbroadcastd    ymm5, xmm2
    LONG $0xcdfaf5c5             // vpsubd    ymm1, ymm1, ymm5
    LONG $0x2cc4f9c5; WORD $0x0006 // vpinsrw    xmm5, xmm0, word [rsi + rax], 0
    LONG $0x1379e2c4; BYTE $0xed // vcvtph2ps    xmm5, xmm5
    LONG $0x187de2c4; BYTE $0xed // vbroadcastss    ymm5, xmm5
    LONG $0xed5984c5             // vmulps    ymm5, ymm15, ymm5
    LONG $0xc95bfcc5             // vcvtdq2ps    ymm1, ymm1
    LONG $0xb855e2c4; BYTE $0xf1 // vfmadd231ps    ymm6, ymm5, ymm1
    LONG $0xc86cbdc5             // vpunpcklqdq    ymm1, ymm8, ymm0
    LONG $0xea70f9c5; BYTE $0x55 // vpshufd    xmm5, xmm2, 85
    LONG $0x587de2c4; BYTE $0xed // vpbroadcastd    ymm5, xmm5
    LONG $0xe5faddc5             // vpsubd    ymm4, ymm4, ymm5
    LONG $0x6cc4f9c5; WORD $0x0206; BYTE $0x00 // vpinsrw    xmm5, xmm0, word [rsi + rax + 2], 0
    LONG $0x1379e2c4; BYTE $0xed // vcvtph2ps    xmm5, xmm5
    LONG $0x187de2c4; BYTE $0xed // vbroadcastss    ymm5, xmm5
    LONG $0xed5984c5             // vmulps    ymm5, ymm15, ymm5
    LONG $0xe45bfcc5             // vcvtdq2ps    ymm4, ymm4
    LONG $0xb855e2c4; BYTE $0xfc // vfmadd231ps    ymm7, ymm5, ymm4
    LONG $0x397de3c4; WORD $0x01d4 // vextracti128    xmm4, ymm2, 1
    LONG $0x587de2c4; BYTE $0xe4 // vpbroadcastd    ymm4, xmm4
    LONG $0x6cc4f9c5; WORD $0x0406; BYTE $0x00 // vpinsrw    xmm5, xmm0, word [rsi + rax + 4], 0
    LONG $0x1379e2c4; BYTE $0xed // vcvtph2ps    xmm5, xmm5
    LONG $0xccfaf5c5             // vpsubd    ymm1, ymm1, ymm4
    LONG $0x187de2c4; BYTE $0xe5 // vbroadcastss    ymm4, xmm5
    LONG $0xe45984c5             // vmulps    ymm4, ymm15, ymm4
    LONG $0xc95bfcc5             // vcvtdq2ps    ymm1, ymm1
    QUAD $0x0000e024ac10fcc5; BYTE $0x00 // vmovups    ymm5, yword [rsp + 224]
    LONG $0xb85de2c4; BYTE $0xe9 // vfmadd231ps    ymm5, ymm4, ymm1
    QUAD $0x0000e024ac11fcc5; BYTE $0x00 // vmovups    yword [rsp + 224], ymm5
    QUAD $0x0000e024a410fcc5; BYTE $0x00 // vmovups    ymm4, yword [rsp + 224]
    LONG $0xc06dbdc5             // vpunpckhqdq    ymm0, ymm8, ymm0
    LONG $0xca70fdc5; BYTE $0x55 // vpshufd    ymm1, ymm2, 85
    LONG $0x00fde3c4; WORD $0xaac9 // vpermq    ymm1, ymm1, 170
    LONG $0x54c4f9c5; WORD $0x0606; BYTE $0x00 // vpinsrw    xmm2, xmm0, word [rsi + rax + 6], 0
    LONG $0x1379e2c4; BYTE $0xd2 // vcvtph2ps    xmm2, xmm2
    LONG $0xc1fafdc5             // vpsubd    ymm0, ymm0, ymm1
    LONG $0x187de2c4; BYTE $0xca // vbroadcastss    ymm1, xmm2
    LONG $0xc95984c5             // vmulps    ymm1, ymm15, ymm1
    LONG $0xc05bfcc5             // vcvtdq2ps    ymm0, ymm0
    QUAD $0x0000c0249410fcc5; BYTE $0x00 // vmovups    ymm2, yword [rsp + 192]
    LONG $0xb875e2c4; BYTE $0xd0 // vfmadd231ps    ymm2, ymm1, ymm0
    QUAD $0x0000c0249411fcc5; BYTE $0x00 // vmovups    yword [rsp + 192], ymm2
    QUAD $0x0000c0248c10fcc5; BYTE $0x00 // vmovups    ymm1, yword [rsp + 192]
    LONG $0x88c68148; WORD $0x0000; BYTE $0x00 // add    rsi, 136
    LONG $0x90c08149; WORD $0x0000; BYTE $0x00 // add    r8, 144
    WORD $0x394c; BYTE $0xc2     // cmp    rdx, r8
	JNE LBB0_4
	JMP LBB0_2
LBB0_1:
    LONG $0xc057f8c5             // vxorps    xmm0, xmm0, xmm0
    LONG $0x4411fcc5; WORD $0x6024 // vmovups    yword [rsp + 96], ymm0
    LONG $0x4411fcc5; WORD $0x4024 // vmovups    yword [rsp + 64], ymm0
    LONG $0x4411fcc5; WORD $0x2024 // vmovups    yword [rsp + 32], ymm0
    LONG $0x0411fcc5; BYTE $0x24 // vmovups    yword [rsp], ymm0
    LONG $0xf657c8c5             // vxorps    xmm6, xmm6, xmm6
    LONG $0xff57c0c5             // vxorps    xmm7, xmm7, xmm7
    LONG $0xe457d8c5             // vxorps    xmm4, xmm4, xmm4
    LONG $0xc957f0c5             // vxorps    xmm1, xmm1, xmm1
LBB0_2:
    LONG $0x0410fcc5; BYTE $0x24 // vmovups    ymm0, yword [rsp]
    LONG $0x0111fcc5             // vmovups    yword [rcx], ymm0
    LONG $0x4410fcc5; WORD $0x2024 // vmovups    ymm0, yword [rsp + 32]
    LONG $0x4111fcc5; BYTE $0x20 // vmovups    yword [rcx + 32], ymm0
    LONG $0x4410fcc5; WORD $0x4024 // vmovups    ymm0, yword [rsp + 64]
    LONG $0x4111fcc5; BYTE $0x40 // vmovups    yword [rcx + 64], ymm0
    LONG $0x4410fcc5; WORD $0x6024 // vmovups    ymm0, yword [rsp + 96]
    LONG $0x4111fcc5; BYTE $0x60 // vmovups    yword [rcx + 96], ymm0
    QUAD $0x00000080b111fcc5     // vmovups    yword [rcx + 128], ymm6
    QUAD $0x000000a0b911fcc5     // vmovups    yword [rcx + 160], ymm7
    QUAD $0x000000c0a111fcc5     // vmovups    yword [rcx + 192], ymm4
    QUAD $0x000000e08911fcc5     // vmovups    yword [rcx + 224], ymm1
    LONG $0x68c48148; WORD $0x0003; BYTE $0x00 // add    rsp, 872
    VZEROUPPER
    POPQ BP
    SUBQ $880, SP
    RET
