package k3

import (
	"math"
	"runtime"
	"testing"
	"unsafe"
)

// TestK3I8I4M4NativeVsFallback validates the native assembly M4 kernel
// produces the same results as the Go fallback (which calls M1 4 times).
func TestK3I8I4M4NativeVsFallback(t *testing.T) {
	runtime.LockOSThread()
	registerAIThread(8)

	const kBlks = 4 // 4 K32 blocks = 128 dimensions
	const nCols = 32
	const ldcBytes = nCols * 4
	const m4ABlock = 152 // 16+8+128 per k-block

	// Build A data in M4 layout: [scale[4]:16B][sum[4]:8B][q[4][32]:128B] per block
	aData := make([]byte, kBlks*m4ABlock)
	for k := 0; k < kBlks; k++ {
		off := k * m4ABlock
		// Scales: 1.0 for rows 0-3
		for r := 0; r < 4; r++ {
			*(*float32)(unsafe.Pointer(&aData[off+r*4])) = 1.0 + float32(r)*0.5
		}
		// Sums (int16): 32 for each row (sum of 32 ones)
		for r := 0; r < 4; r++ {
			*(*int16)(unsafe.Pointer(&aData[off+16+r*2])) = 32
		}
		// Quants: each row has 32 bytes, value=1
		for r := 0; r < 4; r++ {
			for i := 0; i < 32; i++ {
				aData[off+24+r*32+i] = 1
			}
		}
	}

	// Build B data: one N32 tile, kBlks subblocks of 608 bytes
	bData := make([]byte, kBlks*608)
	for k := 0; k < kBlks; k++ {
		off := k * 608
		// fp16 scales: 1.0 for all 32 columns
		for i := 0; i < 32; i++ {
			*(*uint16)(unsafe.Pointer(&bData[off+i*2])) = 0x3C00 // fp16(1.0)
		}
		// ZP: 0 (skip, at offset +64)
		// Nibbles at offset +96: 512 bytes, all nibbles = 0x11 (lo=1, hi=1)
		for i := 0; i < 512; i++ {
			bData[off+96+i] = 0x11
		}
	}

	// Run native M4
	outNative := make([]float32, 4*nCols)
	k3I8I4M4(&aData[0], &bData[0], &outNative[0], kBlks, ldcBytes)

	// Run fallback M4
	outFallback := make([]float32, 4*nCols)
	k3I8I4M4Fallback(&aData[0], &bData[0], &outFallback[0], kBlks, ldcBytes)

	// Compare
	maxDiff := float64(0)
	for r := 0; r < 4; r++ {
		for c := 0; c < nCols; c++ {
			idx := r*nCols + c
			diff := math.Abs(float64(outNative[idx] - outFallback[idx]))
			if diff > maxDiff {
				maxDiff = diff
			}
			if diff > 1e-2 {
				t.Errorf("row %d col %d: native=%.6f fallback=%.6f diff=%.6f",
					r, c, outNative[idx], outFallback[idx], diff)
				if r == 0 && c > 3 {
					t.Fatalf("too many errors, stopping")
				}
			}
		}
	}
	t.Logf("max diff: %e", maxDiff)
	t.Logf("native row0[0:4]:   %v", outNative[0:4])
	t.Logf("fallback row0[0:4]: %v", outFallback[0:4])
	t.Logf("native row3[0:4]:   %v", outNative[3*nCols:3*nCols+4])
	t.Logf("fallback row3[0:4]: %v", outFallback[3*nCols:3*nCols+4])
}
