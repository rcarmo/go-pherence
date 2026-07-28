package simd

import (
	"fmt"
	"math"
	"testing"
)

func TestDotRowsx4MatchesSdot(t *testing.T) {
	for _, cols := range []int{1, 7, 8, 9, 16, 17, 64, 256, 1024, 4096} {
		w := make([]float32, 4*cols)
		x := make([]float32, cols)
		for i := range w {
			w[i] = float32((i%17)-8) * 0.03125
		}
		for i := range x {
			x[i] = float32((i%23)-11) * 0.015625
		}
		got0, got1, got2, got3, ok := DotRowsx4(w, x, cols)
		if !ok {
			t.Fatalf("DotRowsx4 rejected cols=%d", cols)
		}
		want := [4]float32{
			Sdot(x, w[:cols]),
			Sdot(x, w[cols:2*cols]),
			Sdot(x, w[2*cols:3*cols]),
			Sdot(x, w[3*cols:4*cols]),
		}
		got := [4]float32{got0, got1, got2, got3}
		for i := range got {
			if math.Abs(float64(got[i]-want[i])) > 1e-3+1e-6*float64(cols) {
				t.Fatalf("cols=%d dot%d=%g want %g all got=%v want=%v", cols, i, got[i], want[i], got, want)
			}
		}
	}
}

func TestDotRowsx4RejectsBadInputs(t *testing.T) {
	w := make([]float32, 32)
	x := make([]float32, 8)
	if _, _, _, _, ok := DotRowsx4(w[:31], x, 8); ok {
		t.Fatal("DotRowsx4 accepted short weights")
	}
	if _, _, _, _, ok := DotRowsx4(w, x[:7], 8); ok {
		t.Fatal("DotRowsx4 accepted short activation")
	}
}

func BenchmarkDotRowsx4VsFourSdot(b *testing.B) {
	for _, cols := range []int{256, 1024, 4096} {
		w := make([]float32, 4*cols)
		x := make([]float32, cols)
		for i := range w {
			w[i] = float32((i%17)-8) * 0.03125
		}
		for i := range x {
			x[i] = float32((i%23)-11) * 0.015625
		}
		b.Run(fmt.Sprintf("rowsx4_cols_%d", cols), func(b *testing.B) {
			b.SetBytes(int64((4*cols + cols) * 4))
			var sink float32
			for i := 0; i < b.N; i++ {
				d0, d1, d2, d3, ok := DotRowsx4(w, x, cols)
				if !ok {
					b.Fatal("DotRowsx4 rejected benchmark shape")
				}
				sink += d0 + d1 + d2 + d3
			}
			_ = sink
		})
		b.Run(fmt.Sprintf("four_sdot_cols_%d", cols), func(b *testing.B) {
			b.SetBytes(int64((4*cols + cols) * 4))
			var sink float32
			for i := 0; i < b.N; i++ {
				sink += Sdot(x, w[:cols])
				sink += Sdot(x, w[cols:2*cols])
				sink += Sdot(x, w[2*cols:3*cols])
				sink += Sdot(x, w[3*cols:4*cols])
			}
			_ = sink
		})
	}
}
