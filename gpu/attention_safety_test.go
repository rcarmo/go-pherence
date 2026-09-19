package gpu

import (
	"math"
	"math/rand"
	"testing"
)

func TestAttentionOverwriteTailsAndScalarOracle(t *testing.T) {
	r := rand.New(rand.NewSource(9))
	for _, shape := range [][4]int{{1, 1, 1, 1}, {3, 7, 2, 9}, {2, 5, 3, 17}} {
		nq, nkv, heads, d := shape[0], shape[1], shape[2], shape[3]
		width := heads * d
		q, k, v := make([]float32, nq*width), make([]float32, nkv*width), make([]float32, nkv*width)
		for _, x := range [][]float32{q, k, v} {
			for i := range x {
				x[i] = r.Float32() - .5
			}
		}
		for _, scale := range []float32{0, .75} {
			out := make([]float32, nq*width+3)
			for i := range out {
				out[i] = 99
			}
			DevAttentionFull(out, q, k, v, nq, nkv, heads, d, scale)
			if scale == 0 {
				scale = float32(1 / math.Sqrt(float64(d)))
			}
			for row := 0; row < nq; row++ {
				for h := 0; h < heads; h++ {
					scores := make([]float64, nkv)
					max := -math.MaxFloat64
					for key := 0; key < nkv; key++ {
						for j := 0; j < d; j++ {
							scores[key] += float64(q[row*width+h*d+j]) * float64(k[key*width+h*d+j])
						}
						scores[key] *= float64(scale)
						if scores[key] > max {
							max = scores[key]
						}
					}
					total := 0.
					for i := range scores {
						scores[i] = math.Exp(scores[i] - max)
						total += scores[i]
					}
					for j := 0; j < d; j++ {
						want := 0.
						for key := 0; key < nkv; key++ {
							want += scores[key] / total * float64(v[key*width+h*d+j])
						}
						got := float64(out[row*width+h*d+j])
						if math.IsNaN(got) || math.Abs(got-want) > 2e-6 {
							t.Fatalf("shape=%v got=%g want=%g", shape, got, want)
						}
					}
				}
			}
			for _, x := range out[nq*width:] {
				if x != 99 {
					t.Fatal("tail clobbered")
				}
			}
			first := append([]float32(nil), out...)
			DevAttentionFull(out, q, k, v, nq, nkv, heads, d, scale)
			for i := range out {
				if out[i] != first[i] {
					t.Fatal("repeat accumulated old output")
				}
			}
		}
	}
}
func TestAttentionInvalidNoWrite(t *testing.T) {
	for _, dims := range [][4]int{{-1, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 1}, {1, 1, 2, int(^uint(0) >> 1)}, {1, 1, 1, 2}} {
		out := []float32{37}
		DevAttentionFull(out, []float32{1}, []float32{1}, []float32{1}, dims[0], dims[1], dims[2], dims[3], 1)
		if out[0] != 37 {
			t.Fatal(dims, out)
		}
	}
}
func TestAttentionScoreScratchAllocations(t *testing.T) {
	q, k, v, out := make([]float32, 4*2*9), make([]float32, 7*2*9), make([]float32, 7*2*9), make([]float32, 4*2*9)
	if n := testing.AllocsPerRun(50, func() { FullAttention(out, q, k, v, 4, 7, 2, 9) }); n > 1 {
		t.Fatalf("allocations=%g", n)
	}
}
func BenchmarkFullAttention(b *testing.B) {
	q, k, v, out := make([]float32, 32*4*32), make([]float32, 64*4*32), make([]float32, 64*4*32), make([]float32, 32*4*32)
	for i := range q {
		q[i] = .1
	}
	for i := range k {
		k[i] = .2
		v[i] = .3
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FullAttention(out, q, k, v, 32, 64, 4, 32)
	}
}
