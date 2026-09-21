package board

import (
	"math"
	"testing"
)

func TestSIMDBoardChecksBeforeParallelSlices(t *testing.T) {
	b := SIMDBackend{}
	out := []float32{99}
	for _, dims := range [][2]int{{512, 512}, {-1, 1}, {int(^uint(0) >> 1), 2}} {
		if err := b.GemvF32(out, []float32{1}, []float32{1}, dims[0], dims[1]); err == nil || out[0] != 99 {
			t.Fatal(err, out)
		}
	}
}
func TestSIMDBoardAttentionGroupsTailsAndOracle(t *testing.T) {
	b := SIMDBackend{}
	for _, dims := range [][4]int{{1, 3, 2, 1}, {1, 1, 2, 1}, {-1, 2, 1, 1}, {1, 2, 1, int(^uint(0) >> 1)}} {
		out := []float32{99}
		if err := b.AttentionScoresF32(out, nil, nil, dims[0], dims[1], dims[2], dims[3], 1); err == nil || out[0] != 99 {
			t.Fatal(err)
		}
	}
	const heads = 4
	const kv = 2
	const d = 9
	const seq = 3
	q, k, out := make([]float32, heads*d), make([]float32, seq*kv*d), make([]float32, heads*seq+1)
	for i := range q {
		q[i] = float32(i%7) * .1
	}
	for i := range k {
		k[i] = float32(i%5) * .2
	}
	out[len(out)-1] = 99
	if err := b.AttentionScoresF32(out, q, k, seq, heads, kv, d, .3); err != nil {
		t.Fatal(err)
	}
	for h := 0; h < heads; h++ {
		for pos := 0; pos < seq; pos++ {
			want := 0.
			for j := 0; j < d; j++ {
				want += float64(q[h*d+j]) * float64(k[(pos*kv+h/2)*d+j])
			}
			got := float64(out[h*seq+pos])
			if math.IsNaN(got) || math.Abs(got-want*.3) > 1e-6 {
				t.Fatal(got, want*.3)
			}
		}
	}
	if out[len(out)-1] != 99 {
		t.Fatal("tail written")
	}
}
