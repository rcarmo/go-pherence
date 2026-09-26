package qwen

import (
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
	"testing"
)

func branchConvRowReference(dst, x, w []float32, parents []int, token int) {
	clear(dst)
	for tap := 0; tap < 4; tap++ {
		p := token
		for back := 0; back < 3-tap && p >= 0; back++ {
			p = parents[p]
		}
		if p >= 0 {
			for c := range dst {
				dst[c] += float32(x[p*6144+c] * w[tap*6144+c])
			}
		}
	}
}

func branchConvFixture() (x, w []float32, parents []int) {
	parents = []int{-1, 0, 1, 2, 2, 4, 2, 6, 7}
	x, w = make([]float32, len(parents)*6144), make([]float32, 4*6144)
	seed := uint32(91)
	for _, s := range [][]float32{x, w} {
		for i := range s {
			seed = 1664525*seed + 1013904223
			s[i] = float32(int32(seed>>8)-8388608) / 16777216
			if i%31 == 0 {
				s[i] = math.Float32frombits(0x80000000)
			}
			if i%37 == 0 {
				s[i] = math.Float32frombits(1)
			}
		}
	}
	return
}
func TestBranchConvRow(t *testing.T) {
	old := simd.HasVecAsm
	defer func() { simd.HasVecAsm = old }()
	for _, enabled := range []bool{false, old} {
		simd.HasVecAsm = enabled
		x, w, parents := branchConvFixture()
		savedX, savedW := append([]float32(nil), x...), append([]float32(nil), w...)
		dstStore, tmpStore := make([]float32, 6146), make([]float32, 6146)
		dstStore[0], dstStore[6145], tmpStore[0], tmpStore[6145] = 17, -23, 29, -31
		dst, tmp := dstStore[1:6145], tmpStore[1:6145]
		want := make([]float32, 6144)
		for token := range parents {
			branchConvRowReference(want, x, w, parents, token)
			qwen35BranchConvRow(dst, x, w, tmp, parents, token)
			for i, a := range want {
				if math.Float32bits(a) != math.Float32bits(dst[i]) {
					t.Fatalf("dispatch=%t token=%d channel=%d got=%x want=%x", enabled, token, i, math.Float32bits(dst[i]), math.Float32bits(a))
				}
			}
		}
		requireExactFloat32Slice(t, "x", x, savedX)
		requireExactFloat32Slice(t, "w", w, savedW)
		if dstStore[0] != 17 || dstStore[6145] != -23 || tmpStore[0] != 29 || tmpStore[6145] != -31 {
			t.Fatal("output/product guard overwritten")
		}
		// Token3 is a sibling of tokens4..8, never their ancestor.
		qwen35BranchConvRow(want, x, w, tmp, parents, 8)
		for i := 3 * 6144; i < 4*6144; i++ {
			x[i] += 1
		}
		qwen35BranchConvRow(dst, x, w, tmp, parents, 8)
		requireExactFloat32Slice(t, "unrelated sibling", dst, want)
		if n := testing.AllocsPerRun(20, func() { qwen35BranchConvRow(dst, x, w, tmp, parents, 8) }); n != 0 {
			t.Fatalf("allocations=%g", n)
		}
	}
}
func BenchmarkBranchConvRow(b *testing.B) {
	for _, name := range []string{"scalar", "simd"} {
		b.Run(name, func(b *testing.B) {
			x, w, parents := branchConvFixture()
			// Benchmark ordinary magnitudes; subnormals remain in correctness
			// tests rather than dominating the measured arithmetic cost.
			for _, values := range [][]float32{x, w} {
				for i, v := range values {
					if v != 0 && math.Abs(float64(v)) < 1e-30 {
						values[i] = .125
					}
				}
			}
			dst, tmp := make([]float32, 6144), make([]float32, 6144)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if name == "scalar" {
					branchConvRowReference(dst, x, w, parents, 8)
				} else {
					qwen35BranchConvRow(dst, x, w, tmp, parents, 8)
				}
			}
		})
	}
}
