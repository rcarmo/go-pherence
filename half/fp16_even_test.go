package half

import (
	"math"
	"testing"
)

func TestF32ToF16EvenMidpoints(t *testing.T) {
	for h := uint16(0); h < 0x7bff; h++ {
		lo, hi := F16ToF32(h), F16ToF32(h+1)
		mid := (lo + hi) / 2
		want := h
		if h&1 != 0 {
			want++
		}
		if got := F32ToF16Even(mid); got != want {
			t.Fatalf("midpoint %x: got %x want %x", h, got, want)
		}
		if got := F32ToF16Even(-mid); got != want|0x8000 {
			t.Fatalf("negative midpoint %x: %x", h, got)
		}
		if F32ToF16Even(lo) != h {
			t.Fatalf("exact %x", h)
		}
	}
	for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		out := F16ToF32(F32ToF16Even(v))
		if math.IsNaN(float64(v)) {
			if !math.IsNaN(float64(out)) {
				t.Fatal("lost NaN")
			}
		} else if out != v {
			t.Fatal("lost infinity")
		}
	}
	if F32ToF16Even(65520) != 0x7c00 {
		t.Fatal("overflow tie")
	}
}
