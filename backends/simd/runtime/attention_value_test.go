package simd

import (
	"math"
	"reflect"
	"testing"
)

func TestAttentionValueExactOrderAndBounds(t *testing.T) {
	for _, rows := range []int{1, 3, 4, 56, 60} {
		for _, width := range []int{7, 8, 128, 135} {
			p, v := make([]float32, rows), make([]float32, rows*width)
			for i := range p {
				p[i] = float32(i%7-3) / 17
			}
			for i := range v {
				v[i] = float32(i%29-14) / 23
			}
			ps, vs := append([]float32(nil), p...), append([]float32(nil), v...)
			want := make([]float32, width)
			attentionValueRowScalar(want, p, v, rows, width)
			got := make([]float32, width+1)
			got[width] = 42
			if !AttentionValueRowTo(got, p, v, rows, width) {
				t.Fatal(rows, width)
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("r%d w%d i%d got%x want%x", rows, width, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
			if got[width] != 42 || !reflect.DeepEqual(p, ps) || !reflect.DeepEqual(v, vs) {
				t.Fatal("modified input/tail")
			}
		}
	}
	dst := []float32{5}
	if AttentionValueRowTo(dst, nil, nil, 0, 1) || dst[0] != 5 {
		t.Fatal("bad shape mutated")
	}
	p := make([]float32, 8)
	if AttentionValueRowTo(p, p, make([]float32, 64), 8, 8) {
		t.Fatal("overlap accepted")
	}
	v := make([]float32, 64)
	if AttentionValueRowTo(v[:8], p, v, 8, 8) {
		t.Fatal("value overlap accepted")
	}
	if AttentionValueRowTo(make([]float32, 8), p, v, int(^uint(0)>>1), 8) {
		t.Fatal("overflow accepted")
	}
}
func BenchmarkAttentionValueCached(b *testing.B) {
	p, v := make([]float32, 60), make([]float32, 60*128)
	for i := range p {
		p[i] = .01
	}
	for i := range v {
		v[i] = .2
	}
	dst := make([]float32, 128)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		AttentionValueRowTo(dst, p, v, 60, 128)
	}
}
