package simd

import (
	"math"
	"runtime"
	"testing"
)

func TestSgemmNTPrepackedParallelRejectsMalformed(t *testing.T) {
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	const m, n, k = 64, 256, 256
	a := randFloats(m*k, 31)
	weights := randFloats(n*k, 37)
	packed, err := PackSgemmNTWeights(weights, n, k, k)
	if err != nil {
		t.Fatal(err)
	}
	out := randFloats(m*n, 41)
	before := append([]float32(nil), out...)
	for _, candidate := range []struct{ c, a, weights, packed []float32 }{
		{out[:len(out)-1], a, weights, packed},
		{out, a[:len(a)-1], weights, packed},
		{out, a, weights[:len(weights)-1], packed},
		{out, a, weights, packed[:len(packed)-1]},
		{out, a, weights, out},
	} {
		if SgemmNTPrepackedParallelTo(candidate.c, candidate.a, candidate.weights, candidate.packed, m, n, k, 1, k, k, n) {
			t.Fatal("accepted malformed or aliased operand")
		}
		for i, v := range out {
			if v != before[i] {
				t.Fatalf("rejected input changed output at %d", i)
			}
		}
	}
}

func TestSgemmNTPrepackedParallelParity(t *testing.T) {
	for _, workers := range []int{1, 2, 4} {
		old := runtime.GOMAXPROCS(workers)
		for _, shape := range []struct{ m, n, k int }{{3, 32, 64}, {75, 257, 128}, {320, 512, 512}, {541, 512, 512}, {320, 2048, 512}, {541, 512, 2048}} {
			a := randFloats(shape.m*shape.k, 13)
			weights := randFloats(shape.n*shape.k, 17)
			packed, err := PackSgemmNTWeights(weights, shape.n, shape.k, shape.k)
			if err != nil {
				t.Fatal(err)
			}
			initial := randFloats(shape.m*shape.n, 21)
			want := append([]float32(nil), initial...)
			got := append([]float32(nil), initial...)
			if !SgemmNTPrepackedTo(want, a, weights, packed, shape.m, shape.n, shape.k, .75, shape.k, shape.k, shape.n) || !SgemmNTPrepackedParallelTo(got, a, weights, packed, shape.m, shape.n, shape.k, .75, shape.k, shape.k, shape.n) {
				t.Fatalf("rejected shape=%+v workers=%d", shape, workers)
			}
			for i := range want {
				if diff := math.Abs(float64(want[i] - got[i])); diff > 1e-6 {
					t.Fatalf("shape=%+v workers=%d index=%d diff=%g", shape, workers, i, diff)
				}
			}
		}
		runtime.GOMAXPROCS(old)
	}
}
