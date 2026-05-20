package rope

import (
	"math"
	"testing"
)

func TestQwen35RotaryHalf(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.HeadDim = 256
	meta.PartialRotaryFactor = 0.25
	if got := Qwen35RotaryHalf(meta); got != 32 {
		t.Fatalf("rotHalf=%d want 32", got)
	}
	meta.MRoPEInterleaved = true
	meta.MRoPESection = []int{11, 11, 10}
	if !Qwen35UseMRoPE(meta) {
		t.Fatal("expected MRoPE flag")
	}
	freqs := NewQwen35RoPEFreqs(meta, 4)
	if len(freqs) != 4*32*2 {
		t.Fatalf("freq len=%d", len(freqs))
	}
}

func TestNewQwen35RoPEFreqsRejectsOverflowAndBadTheta(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.HeadDim = 256
	meta.PartialRotaryFactor = 1
	maxInt := int(^uint(0) >> 1)
	if got := NewQwen35RoPEFreqs(meta, maxInt); got != nil {
		t.Fatalf("overflow maxSeq returned len=%d, want nil", len(got))
	}
	meta.RopeTheta = -1
	freqs := NewQwen35RoPEFreqs(meta, 2)
	if len(freqs) == 0 {
		t.Fatal("negative theta fallback returned empty freqs")
	}
	for i, v := range freqs {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("freqs[%d]=%v, want finite fallback", i, v)
		}
	}
}
