package aicpu

import (
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/backends/spacemit/aicpu/aipool"
)

func putQ6(raw []byte, blkOff, pos int, v int8) {
	u := uint8(int(v)+32) & 0x3f
	bi := pos / 32
	l := pos % 32
	q4Base := 64 * (bi / 4)
	qhBase := 128 + 32*(bi/4)
	switch bi % 4 {
	case 0:
		raw[blkOff+q4Base+l] = (raw[blkOff+q4Base+l] & 0xf0) | (u & 0x0f)
		raw[blkOff+qhBase+l] |= ((u >> 4) & 3) << 0
	case 1:
		raw[blkOff+q4Base+32+l] = (raw[blkOff+q4Base+32+l] & 0xf0) | (u & 0x0f)
		raw[blkOff+qhBase+l] |= ((u >> 4) & 3) << 2
	case 2:
		raw[blkOff+q4Base+l] = (raw[blkOff+q4Base+l] & 0x0f) | ((u & 0x0f) << 4)
		raw[blkOff+qhBase+l] |= ((u >> 4) & 3) << 4
	case 3:
		raw[blkOff+q4Base+32+l] = (raw[blkOff+q4Base+32+l] & 0x0f) | ((u & 0x0f) << 4)
		raw[blkOff+qhBase+l] |= ((u >> 4) & 3) << 6
	}
}

func TestRepackQ6KRawToQ80x32NativeRef(t *testing.T) {
	M, K := 32, 256
	d := float32(0.25)
	raw := make([]byte, M*q6KBlockBytes)
	f32 := make([]float32, M*K)
	for r := 0; r < M; r++ {
		blkOff := r * q6KBlockBytes
		for i := 0; i < 16; i++ {
			raw[blkOff+192+i] = 1 // Q6_K per-16 scale
		}
		bits := f32ToF16Bits(d)
		raw[blkOff+208] = byte(bits)
		raw[blkOff+209] = byte(bits >> 8)
		for k := 0; k < K; k++ {
			q := int8(((r*7 + k*5) % 41) - 20)
			putQ6(raw, blkOff, k, q)
			f32[r*K+k] = float32(q) * d
		}
	}
	w := repackQ6KRawToQ80x32(M, K, raw)
	if !w.Valid {
		t.Fatal("repack failed")
	}
	act := make([]float32, K)
	for k := range act {
		act[k] = float32((k%17)-8) / 5.0
	}
	pool := aipool.NewAIWorkerPool(6)
	defer pool.Close()
	got := make([]float32, M)
	if !q8Q80x32MatVecNative(w, act, got, pool) {
		t.Fatal("native matvec failed")
	}
	ref := make([]float32, M)
	matVecF32Direct(M, K, f32, act, ref)
	var maxDiff float64
	maxIdx := -1
	for i := range ref {
		if dd := math.Abs(float64(got[i] - ref[i])); dd > maxDiff {
			maxDiff = dd
			maxIdx = i
		}
	}
	t.Logf("maxDiff=%.6f idx=%d got=%.6f ref=%.6f", maxDiff, maxIdx, got[maxIdx], ref[maxIdx])
	if maxDiff > 0.50 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}
