package nvidia

import (
	"math"
	"testing"
)

func TestCausalBatchAttentionMatchesCPU(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	const rows, pos0, heads, kvHeads, dim = 7, 9, 4, 2, 32
	kvLen := pos0 + rows
	q := make([]float32, rows*heads*dim)
	k := make([]float32, kvLen*kvHeads*dim)
	v := make([]float32, len(k))
	for i := range q {
		q[i] = float32((i*17)%37-18) * .013
	}
	for i := range k {
		k[i] = float32((i*11)%41-20) * .009
		v[i] = float32((i*7)%43-21) * .011
	}
	scale := float32(1 / math.Sqrt(dim))
	for _, window := range []int{0, 8} {
		t.Run(map[bool]string{true: "full", false: "windowed"}[window == 0], func(t *testing.T) {
			qBuf, _ := Malloc(len(q))
			kBuf, _ := Malloc(len(k))
			vBuf, _ := Malloc(len(v))
			outBuf, _ := Malloc(len(q))
			defer qBuf.Free()
			defer kBuf.Free()
			defer vBuf.Free()
			defer outBuf.Free()
			if err := qBuf.Upload(q); err != nil {
				t.Fatal(err)
			}
			if err := kBuf.Upload(k); err != nil {
				t.Fatal(err)
			}
			if err := vBuf.Upload(v); err != nil {
				t.Fatal(err)
			}
			if err := CausalBatchAttentionBuffer(outBuf, qBuf, kBuf, vBuf, rows, pos0, kvLen, window, heads, kvHeads, dim, scale); err != nil {
				t.Fatal(err)
			}
			if err := SyncErr(); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, len(q))
			if err := outBuf.Download(got); err != nil {
				t.Fatal(err)
			}
			want := causalAttentionCPU(q, k, v, rows, pos0, kvLen, window, heads, kvHeads, dim, scale)
			for i := range got {
				diff := math.Abs(float64(got[i] - want[i]))
				limit := 2e-5 * math.Max(1, math.Abs(float64(want[i])))
				if diff > limit {
					t.Fatalf("index=%d got=%g want=%g diff=%g limit=%g", i, got[i], want[i], diff, limit)
				}
			}
		})
	}
}

func causalAttentionCPU(q, k, v []float32, rows, pos0, kvLen, window, heads, kvHeads, dim int, scale float32) []float32 {
	out := make([]float32, len(q))
	group := heads / kvHeads
	for row := 0; row < rows; row++ {
		end := min(pos0+row+1, kvLen)
		start := 0
		if window > 0 && end > window {
			start = end - window
		}
		for head := 0; head < heads; head++ {
			kvHead := head / group
			scores := make([]float64, end-start)
			maxScore := math.Inf(-1)
			for key := start; key < end; key++ {
				sum := float32(0)
				for d := 0; d < dim; d++ {
					sum += q[(row*heads+head)*dim+d] * k[(key*kvHeads+kvHead)*dim+d]
				}
				score := float64(sum * scale)
				scores[key-start] = score
				maxScore = max(maxScore, score)
			}
			den := 0.
			for i := range scores {
				scores[i] = math.Exp(scores[i] - maxScore)
				den += scores[i]
			}
			for d := 0; d < dim; d++ {
				sum := 0.
				for key := start; key < end; key++ {
					sum += scores[key-start] / den * float64(v[(key*kvHeads+kvHead)*dim+d])
				}
				out[(row*heads+head)*dim+d] = float32(sum)
			}
		}
	}
	return out
}

// A short key range leaves some warps with no exponentials to calculate. They
// must not overwrite reduce[0] before other warps load the softmax maximum.
func TestCausalBatchAttentionShortSoftmaxRace(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	const rows, pos0, heads, kvHeads, dim = 24, 57, 16, 8, 256
	const kvLen = pos0 + rows
	q, k, v := make([]float32, rows*heads*dim), make([]float32, kvLen*kvHeads*dim), make([]float32, kvLen*kvHeads*dim)
	for i := range q {
		q[i] = 1 + float32(i%7)*.1
	}
	for i := range k {
		k[i] = 1 + float32(i%11)*.1
		v[i] = float32(i%37-18) * .07
	}
	qb, err := Malloc(len(q))
	if err != nil {
		t.Fatal(err)
	}
	defer qb.Free()
	kb, err := Malloc(len(k))
	if err != nil {
		t.Fatal(err)
	}
	defer kb.Free()
	vb, err := Malloc(len(v))
	if err != nil {
		t.Fatal(err)
	}
	defer vb.Free()
	ob, err := Malloc(len(q))
	if err != nil {
		t.Fatal(err)
	}
	defer ob.Free()
	for _, x := range []struct {
		b *Buffer
		x []float32
	}{{qb, q}, {kb, k}, {vb, v}} {
		if err := x.b.Upload(x.x); err != nil {
			t.Fatal(err)
		}
	}
	scale := float32(1 / math.Sqrt(dim))
	for _, window := range []int{0, 8} {
		want := causalAttentionCPU(q, k, v, rows, pos0, kvLen, window, heads, kvHeads, dim, scale)
		got := make([]float32, len(q))
		for repeat := 0; repeat < 32; repeat++ {
			if err := CausalBatchAttentionBuffer(ob, qb, kb, vb, rows, pos0, kvLen, window, heads, kvHeads, dim, scale); err != nil {
				t.Fatal(err)
			}
			if err := ob.Download(got); err != nil {
				t.Fatal(err)
			}
			for i, x := range got {
				d := math.Abs(float64(x - want[i]))
				if math.IsNaN(d) || d > 2e-5*math.Max(1, math.Abs(float64(want[i]))) {
					t.Fatalf("window=%d repeat=%d index=%d got=%g want=%g diff=%g", window, repeat, i, x, want[i], d)
				}
			}
		}
	}
}
