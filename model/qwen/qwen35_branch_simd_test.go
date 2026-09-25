package qwen

import (
	"math"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	cfg "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/tensor"
)

func TestSIMDBranchProjection(t *testing.T) {
	for _, shape := range [][3]int{{3, 16, 13}, {8, 512, 32}, {13, 768, 65}, {1, 512, 128}, {17, 1024, 63}} {
		rows, n, k := shape[0], shape[1], shape[2]
		x, w := make([]float32, rows*k), make([]float32, n*k)
		for i := range x {
			x[i] = float32(i%19-9) * 0.013
		}
		for i := range w {
			w[i] = float32(i%13-6) * 0.027
		}
		weight := tensor.FromFloat32(w, []int{n, k})
		packed, e := simd.PackSgemmNTWeights(w, n, k, k)
		if e != nil {
			t.Fatal(e)
		}
		s := &Qwen35SIMDBranch{packed: map[*tensor.Tensor][]float32{weight: packed}}
		out := make([]float32, rows*n)
		for range 2 {
			if e = s.project(out, x, weight, rows, k, n); e != nil {
				t.Fatal(e)
			}
			for r := 0; r < rows; r++ {
				for c := 0; c < n; c++ {
					var want float32
					for j := 0; j < k; j++ {
						want += x[r*k+j] * w[c*k+j]
					}
					if math.Abs(float64(out[r*n+c]-want)) > 1e-5 {
						t.Fatal("SIMD projection mismatch", shape, r, c)
					}
				}
			}
		}
	}
}
func TestSIMDBranchConstructionRejects(t *testing.T) {
	for _, cap := range []int{-1, 0, 2, 513} {
		if s, e := NewQwen35SIMDBranch(nil, cfg.QwenNativeMTPMetadata{}, cap); e == nil || s != nil {
			t.Fatal("nil model")
		}
	}
	var s *Qwen35SIMDBranch
	if out, e := s.Forward(nil, 1, 1, nil, 1e-6); e == nil || out != nil {
		t.Fatal("nil executor")
	}
	// Validate before scratch access or matrix work.
	s = &Qwen35SIMDBranch{maxTokens: 8}
	for _, in := range [][][]float32{nil, {{0}}, {{0}, {0}, {0}}} {
		if out, e := s.Forward(in, 1, 1, nil, 1e-6); e == nil || out != nil {
			t.Fatal("invalid input")
		}
	}
}
func TestSIMDDeltaRowMatchesReference(t *testing.T) {
	a, b, q, k := make([]float32, 128), make([]float32, 128), make([]float32, 128), make([]float32, 128)
	for i := range q {
		q[i] = float32(i%11-5) * 0.02
		k[i] = float32(i%7-3) * 0.03
	}
	for i := 0; i < 64; i++ {
		v := float32(i%9-4) * 0.2
		want := qwen35LinearDeltaRowInPlace(a, q, k, v, 0.7, 0.95, 0.0883883476)
		simd.VecScale(b, b, 0.95)
		memory := simd.Sdot(b, k)
		simd.Saxpy((v-memory)*0.7, k, b)
		got := simd.Sdot(b, q) * 0.0883883476
		if math.Abs(float64(want-got)) > 1e-6 {
			t.Fatal("delta drift", i, want, got)
		}
	}
}
