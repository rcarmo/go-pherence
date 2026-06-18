package model

import "testing"

func TestGGMLFlashAttnF16KVReferenceShape(t *testing.T) {
	q := make([]float32, 2*4)
	k := make([]float32, 3*1*4)
	v := make([]float32, 3*1*4)
	for i := range q {
		q[i] = float32(i+1) * 0.01
	}
	for i := range k {
		k[i] = float32(i-3) * 0.02
	}
	for i := range v {
		v[i] = float32(i+5) * -0.015
	}
	out := ggmlFlashAttnF16KVReference(q, k, v, 3, 2, 1, 4, 1)
	if len(out) != len(q) {
		t.Fatalf("out len=%d want %d", len(out), len(q))
	}
}
