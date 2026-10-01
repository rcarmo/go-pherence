package audio

import (
	"context"
	"math"
	"testing"
)

func TestWhisperLogMelClipFloor(t *testing.T) {
	ctx := context.Background()
	samples := make([]float32, 48000)
	for i := range samples {
		samples[i] = float32(0.1 * math.Sin(float64(i)*0.05) * math.Exp(-float64(i)/12000))
	}
	base, frames, err := WhisperLogMelContext(ctx, samples, 128)
	if err != nil {
		t.Fatal(err)
	}
	maxLog, err := WhisperLogMelMaxContext(ctx, samples, 128)
	if err != nil {
		t.Fatal(err)
	}
	var maxNorm float32 = -100
	for _, v := range base {
		maxNorm = float32(math.Max(float64(maxNorm), float64(v)))
	}
	if maxNorm != (maxLog+4)/4 {
		t.Fatal("max mismatch", maxNorm, maxLog)
	}
	// A clip maximum at or below the window maximum leaves output unchanged.
	for _, clip := range []float32{maxLog, maxLog - 3} {
		same, f2, err := WhisperLogMelClipFloorContext(ctx, samples, 128, clip)
		if err != nil || f2 != frames {
			t.Fatal(err)
		}
		for i := range base {
			if math.Float32bits(same[i]) != math.Float32bits(base[i]) {
				t.Fatal("changed", clip, i)
			}
		}
	}
	// A higher clip maximum lifts only clamped values to (clip-8+4)/4.
	clip := maxLog + 2
	lifted, _, err := WhisperLogMelClipFloorContext(ctx, samples, 128, clip)
	if err != nil {
		t.Fatal(err)
	}
	floor := (clip - 8 + 4) / 4
	changed := 0
	for i := range base {
		want := base[i]
		if base[i] < floor {
			want = floor
		}
		if math.Abs(float64(lifted[i]-want)) > 1e-6 {
			t.Fatal("lifted", i, lifted[i], want)
		}
		if lifted[i] != base[i] {
			changed++
		}
	}
	if changed == 0 {
		t.Fatal("no clamped values exercised")
	}
	// Digital silence: -10 raw maximum; default -1.5, lifted by a clip floor.
	zero := make([]float32, 32000)
	if m, err := WhisperLogMelMaxContext(ctx, zero, 80); err != nil || m != -10 {
		t.Fatal(m, err)
	}
	z, _, err := WhisperLogMelClipFloorContext(ctx, zero, 80, 2)
	if err != nil || z[0] != -0.5 || z[len(z)-1] != -0.5 {
		t.Fatal(z[0], err)
	}
	if d, _, _ := WhisperLogMelContext(ctx, zero, 80); d[0] != -1.5 {
		t.Fatal("silence default", d[0])
	}
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if _, _, err := WhisperLogMelClipFloorContext(ctx, samples, 128, bad); err == nil {
			t.Fatal("accepted", bad)
		}
	}
}
