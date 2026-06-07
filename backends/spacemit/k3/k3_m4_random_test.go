package k3

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"
)

func TestK3I8I4M4NativeRandomData(t *testing.T) {
	runtime.LockOSThread()
	registerAIThread(8)

	rng := rand.New(rand.NewSource(42))
	const kBlks = 32 // 1024 dimensions (typical for Qwen3 0.6B)
	const nCols = 32
	const ldcBytes = nCols * 4
	const m4ABlock = 152

	aData := make([]byte, kBlks*m4ABlock)
	for k := 0; k < kBlks; k++ {
		off := k * m4ABlock
		for r := 0; r < 4; r++ {
			scale := rng.Float32()*2.0 - 1.0
			*(*float32)(unsafe.Pointer(&aData[off+r*4])) = scale
		}
		for r := 0; r < 4; r++ {
			*(*int16)(unsafe.Pointer(&aData[off+16+r*2])) = int16(rng.Intn(65)-32)
		}
		for r := 0; r < 4; r++ {
			for i := 0; i < 32; i++ {
				aData[off+24+r*32+i] = byte(rng.Intn(256))
			}
		}
	}

	bData := make([]byte, kBlks*608)
	for k := 0; k < kBlks; k++ {
		off := k * 608
		for i := 0; i < 32; i++ {
			// Random fp16 scales (positive)
			val := float32(rng.Float64()*2.0 + 0.1)
			bits := math.Float32bits(val)
			fp16 := uint16((bits>>16)&0x8000) | uint16(((bits>>23)&0xFF-127+15)<<10) | uint16((bits>>13)&0x3FF)
			*(*uint16)(unsafe.Pointer(&bData[off+i*2])) = fp16
		}
		// ZP = 0 (offset +64, 32 bytes)
		for i := 0; i < 512; i++ {
			bData[off+96+i] = byte(rng.Intn(256))
		}
	}

	outNative := make([]float32, 4*nCols)
	k3I8I4M4(&aData[0], &bData[0], &outNative[0], kBlks, ldcBytes)

	outFallback := make([]float32, 4*nCols)
	k3I8I4M4Fallback(&aData[0], &bData[0], &outFallback[0], kBlks, ldcBytes)

	maxDiff := float64(0)
	errCount := 0
	for r := 0; r < 4; r++ {
		for c := 0; c < nCols; c++ {
			idx := r*nCols + c
			nv := float64(outNative[idx])
			fv := float64(outFallback[idx])
			diff := math.Abs(nv - fv)
			denom := math.Max(math.Abs(nv), math.Abs(fv))
			relDiff := float64(0)
			if denom > 1e-6 {
				relDiff = diff / denom
			}
			if diff > maxDiff {
				maxDiff = diff
			}
			if relDiff > 0.01 { // 1% relative tolerance
				errCount++
				if errCount <= 8 {
					t.Errorf("row %d col %d: native=%.4f fallback=%.4f diff=%.4f relDiff=%.4f",
						r, c, outNative[idx], outFallback[idx], diff, relDiff)
				}
			}
		}
	}
	if errCount > 0 {
		t.Errorf("total mismatches: %d / %d", errCount, 4*nCols)
	}
	t.Logf("max abs diff: %e", maxDiff)
	t.Logf("native row0[0:4]:   %v", outNative[0:4])
	t.Logf("fallback row0[0:4]: %v", outFallback[0:4])
}
