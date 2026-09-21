package gliner2

import (
	"math"
	"testing"
)

func TestRelativeBuckets(t *testing.T) {
	for i := -128; i <= 128; i++ {
		v, err := RelativePositionBucket(i, 256, 512)
		if err != nil || v != i {
			t.Fatalf("%d %d %v", i, v, err)
		}
	}
	for _, i := range []int{129, 256, 511, 512, 1000} {
		a, _ := RelativePositionBucket(i, 256, 512)
		b, _ := RelativePositionBucket(-i, 256, 512)
		if a != -b || a < 128 {
			t.Fatal(i, a, b)
		}
	}
	a, _ := RelativePositionBucket(511, 256, 512)
	if a != 255 {
		t.Fatal(a)
	}
}
func TestDisentangledAttentionScalar(t *testing.T) {
	q := [][]float32{{1, 2}, {3, 4}}
	k := [][]float32{{.2, .3}, {.5, .7}}
	v := [][]float32{{2, 3}, {4, 5}}
	rq, rk := make([][]float32, 8), make([][]float32, 8)
	for i := range rq {
		rq[i] = []float32{float32(i) * .1, 1}
		rk[i] = []float32{2, float32(i) * .2}
	}
	got, err := DisentangledAttention(q, k, v, rq, rk, []bool{true, true}, 1, 4, 16)
	if err != nil {
		t.Fatal(err)
	}
	for i := range q {
		score := make([]float64, 2)
		for j := range k {
			idx := i - j + 4
			for d := 0; d < 2; d++ {
				score[j] += float64(q[i][d])*float64(k[j][d]) + float64(q[i][d])*float64(rk[idx][d]) + float64(k[j][d])*float64(rq[idx][d])
			}
			score[j] /= math.Sqrt(6)
		}
		p := 1 / (1 + math.Exp(score[1]-score[0]))
		for d := 0; d < 2; d++ {
			want := p*float64(v[0][d]) + (1-p)*float64(v[1][d])
			if math.Abs(float64(got[i][d])-want) > 1e-5 {
				t.Fatal(got, want)
			}
		}
	}
	got, err = DisentangledAttention(q, k, v, rq, rk, []bool{true, false}, 1, 4, 16)
	if err != nil {
		t.Fatal(err)
	}
	if got[0][0] != 2 || got[1][0] != 3 {
		t.Fatal("mask semantics", got)
	}
}
