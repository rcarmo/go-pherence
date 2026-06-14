package simd

import (
	"math/rand"
	"testing"
)

// TestGQAAttentionHeadsParallelMatchesSerial verifies the heads-parallel decode
// attention is bit-identical to the serial reference, including a case large
// enough (numHeads*seqLen >= 4096) to exercise the real goroutine path.
func TestGQAAttentionHeadsParallelMatchesSerial(t *testing.T) {
	cases := []struct {
		name                             string
		seqLen, numHeads, numKVHeads, hd int
	}{
		{"small_serial_fallback", 8, 2, 1, 8},
		{"large_parallel", 512, 16, 4, 64},
		{"mha_parallel", 300, 16, 16, 32},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := rand.New(rand.NewSource(5))
			h := tc.numHeads * tc.hd
			kvDim := tc.numKVHeads * tc.hd
			q := make([]float32, h)
			kCache := make([]float32, tc.seqLen*kvDim)
			vCache := make([]float32, tc.seqLen*kvDim)
			for i := range q {
				q[i] = r.Float32()*2 - 1
			}
			for i := range kCache {
				kCache[i] = r.Float32()*2 - 1
				vCache[i] = r.Float32()*2 - 1
			}
			scale := float32(0.125)

			want := make([]float32, h)
			scoresW := make([]float32, tc.seqLen)
			if !GQAAttentionScaleTo(want, scoresW, q, kCache, vCache, tc.seqLen, tc.numHeads, tc.numKVHeads, tc.hd, scale) {
				t.Fatal("serial attention failed")
			}
			got := make([]float32, h)
			scoresG := make([]float32, tc.seqLen)
			if !GQAAttentionHeadsParallelTo(got, scoresG, q, kCache, vCache, tc.seqLen, tc.numHeads, tc.numKVHeads, tc.hd, scale) {
				t.Fatal("parallel attention failed")
			}
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("idx %d: serial=%v parallel=%v", i, want[i], got[i])
				}
			}
		})
	}
}
