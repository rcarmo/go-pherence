package gpu

import (
	"math"
	"testing"
)

func TestFullAttentionShape(t *testing.T) {
	// 2 heads, head_dim=4, seq=3
	numHeads := 2
	headDim := 4
	dModel := numHeads * headDim
	seqLen := 3

	q := make([]float32, seqLen*dModel)
	k := make([]float32, seqLen*dModel)
	v := make([]float32, seqLen*dModel)
	out := make([]float32, seqLen*dModel)

	// Set V to known values
	for i := range v {
		v[i] = float32(i) * 0.1
	}
	// Q and K as ones → uniform attention
	for i := range q {
		q[i] = 1
		k[i] = 1
	}

	FullAttention(out, q, k, v, seqLen, seqLen, numHeads, headDim)

	// With uniform Q/K, attention weights are uniform (1/seqLen each)
	// So output should be mean of V across time dimension for each head
	for tq := 0; tq < seqLen; tq++ {
		for h := 0; h < numHeads; h++ {
			for d := 0; d < headDim; d++ {
				var expected float32
				for tkv := 0; tkv < seqLen; tkv++ {
					expected += v[tkv*dModel+h*headDim+d]
				}
				expected /= float32(seqLen)
				got := out[tq*dModel+h*headDim+d]
				if math.Abs(float64(got-expected)) > 0.01 {
					t.Fatalf("out[%d,%d,%d]=%f want %f", tq, h, d, got, expected)
				}
			}
		}
	}
}

func TestCrossAttentionShape(t *testing.T) {
	numHeads := 2
	headDim := 4
	dModel := numHeads * headDim
	decLen := 2
	encLen := 5

	q := make([]float32, decLen*dModel)
	k := make([]float32, encLen*dModel)
	v := make([]float32, encLen*dModel)
	out := make([]float32, decLen*dModel)

	for i := range v {
		v[i] = 1
	}
	for i := range q {
		q[i] = 1
	}
	for i := range k {
		k[i] = 1
	}

	CrossAttention(out, q, k, v, decLen, encLen, numHeads, headDim)

	// With uniform attention over V=1, output should be 1 everywhere
	for i, val := range out {
		if math.Abs(float64(val)-1.0) > 0.01 {
			t.Fatalf("out[%d]=%f want 1.0", i, val)
		}
	}
}
