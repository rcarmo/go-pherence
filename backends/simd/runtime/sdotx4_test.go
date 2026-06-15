package simd

import (
	"fmt"
	"math"
	"testing"
)

func TestSdotx4MatchesSdot(t *testing.T) {
	for _, k := range []int{16, 64, 2816, 2817} {
		w := make([]float32, k)
		x := make([]float32, 4*k)
		for i := range w {
			w[i] = float32((i%17)-8) * 0.03125
		}
		for i := range x {
			x[i] = float32((i%23)-11) * 0.015625
		}
		got0, got1, got2, got3, ok := Sdotx4(w, x, k)
		if !ok {
			t.Fatalf("Sdotx4 rejected k=%d", k)
		}
		want := [4]float32{
			Sdot(w, x[:k]),
			Sdot(w, x[k:2*k]),
			Sdot(w, x[2*k:3*k]),
			Sdot(w, x[3*k:4*k]),
		}
		got := [4]float32{got0, got1, got2, got3}
		for i := range got {
			if math.Abs(float64(got[i]-want[i])) > 1e-3+1e-6*float64(k) {
				t.Fatalf("k=%d dot%d=%g want %g all got=%v want=%v", k, i, got[i], want[i], got, want)
			}
		}
	}
}

func TestSdotx4RejectsInvalidShape(t *testing.T) {
	w := make([]float32, 8)
	x := make([]float32, 24)
	if _, _, _, _, ok := Sdotx4(w, x, 7); ok {
		t.Fatal("Sdotx4 accepted stride smaller than len(w)")
	}
	if _, _, _, _, ok := Sdotx4(w, x, 8); ok {
		t.Fatal("Sdotx4 accepted missing fourth row")
	}
}

func BenchmarkSdotx4VsFourSdot(b *testing.B) {
	for _, k := range []int{704, 2816} {
		w := make([]float32, k)
		x := make([]float32, 4*k)
		for i := range w {
			w[i] = float32((i%17)-8) * 0.03125
		}
		for i := range x {
			x[i] = float32((i%23)-11) * 0.015625
		}
		b.Run(fmt.Sprintf("sdotx4_k_%d", k), func(b *testing.B) {
			var sink float32
			for i := 0; i < b.N; i++ {
				d0, d1, d2, d3, ok := Sdotx4(w, x, k)
				if !ok {
					b.Fatal("Sdotx4 rejected benchmark shape")
				}
				sink += d0 + d1 + d2 + d3
			}
			_ = sink
		})
		b.Run(fmt.Sprintf("four_sdot_k_%d", k), func(b *testing.B) {
			var sink float32
			for i := 0; i < b.N; i++ {
				sink += Sdot(w, x[:k])
				sink += Sdot(w, x[k:2*k])
				sink += Sdot(w, x[2*k:3*k])
				sink += Sdot(w, x[3*k:4*k])
			}
			_ = sink
		})
	}
}
