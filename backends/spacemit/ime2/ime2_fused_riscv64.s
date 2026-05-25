#include "textflag.h"

// func fusedPackVmadot(wPacked *byte, actI8 *byte, M int, K int, out *int32)
// Fuses broadcast-pack + vmadot K-loop for M output rows.
// actI8 is pre-quantized int8[K]. wPacked is pre-packed weight tiles.
// out is int32[M] (caller extracts at stride 4 for the diagonal).
// M must be multiple of 4. K must be multiple of 8.
TEXT ·fusedPackVmadot(SB), NOSPLIT, $0-40
    MOV  wPacked+0(FP), X10   // a0 = wPacked
    MOV  actI8+8(FP), X11     // a1 = actI8 (pre-quantized)
    MOV  M+16(FP), X12        // a2 = M
    MOV  K+24(FP), X13        // a3 = K
    MOV  out+32(FP), X14      // a4 = out (int32 results)

    // For each group of 4 output rows:
row_loop:
    BEQ  X12, X0, done

    // Zero accumulator v28 (e32, m2 for 4×4 int32)
    WORD $0x011072d7            // vsetvli t0, zero, e32, m2
    WORD $0x5e003e57            // vmv.v.i v28, 0

    // K-loop: for each 8 elements
    MOV  X13, X15              // a5 = K_remaining
    MOV  X11, X16              // a6 = actI8 ptr (reset to start each row group)

k_inner:
    BEQ  X15, X0, k_end
    
    // Set vl=32 for loading 32 bytes (full tile)
    WORD $0x000072d7            // vsetvli t0, zero, e8, m1 (vl=32)
    
    // Load 8 bytes of actI8, broadcast to 32-byte tile in v0
    // Strategy: load 8 bytes into v2, then vrgather/slide to fill v0
    // Simpler: load 8 bytes with vl=8, then use 4× vse8 into temp, load back
    // Simplest for correctness: use the pre-tested approach of just loading 8
    // and putting into the right positions.
    
    // Actually: since vmadot C[r][0] = dot(vs1_row_0, vs2_row_r),
    // we only need row_0 of vs1 (bytes 0-7). Rows 1-3 can be garbage!
    // So: load 8 bytes of act into v0[0:7], leave v0[8:31] as whatever.
    // vmadot will compute C[r][0] using only v0[0:7]. Perfect!
    
    MOV  $8, X5
    WORD $0x0002f2d7            // vsetvli t0, t0, e8, m1 (vl=8)
    WORD $0x02080007            // vle8.v v0, (a6)  // load 8 act bytes into v0[0:7]
    
    // Reset vl=32 for weight load and vmadot
    WORD $0x000072d7            // vsetvli t0, zero, e8, m1 (vl=32)
    
    // Load weight tile (32 bytes) into v1
    WORD $0x02050087            // vle8.v v1, (a0)
    
    // vmadot v28, v0, v1 (accumulate: C += v0 × v1^T)
    // vs1=v0, vs2=v1, vd=v28, funct3=3(ss), funct7=111000
    // (0x38<<26)|(1<<25)|(1<<20)|(0<<15)|(3<<12)|(28<<7)|0x2b
    WORD $0xe2103e2b            // vmadot v28, v0, v1
    
    // Advance
    ADD  $8, X16, X16          // actI8_ptr += 8
    ADD  $32, X10, X10         // wPacked += 32 (next tile)
    ADD  $-8, X15, X15
    JMP  k_inner

k_end:
    // Store full accumulator to output (16 int32s = v28:v29)
    WORD $0x011072d7            // vsetvli t0, zero, e32, m2
    WORD $0x02076e27            // vse32.v v28, (a4)
    ADD  $64, X14, X14         // out += 64 bytes (16 int32s)
    ADD  $-4, X12, X12         // M -= 4
    JMP  row_loop

done:
    RET
