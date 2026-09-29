package nvidia

import (
	"math"
	"testing"
)

func TestGELUErfF32BufferRejectMalformed(t *testing.T) {
	valid := &Buffer{Ptr: 0x10000, Size: 1024}
	for _, tc := range []struct {
		name string
		buf  *Buffer
		n    int
	}{
		{"nil", nil, 1},
		{"zero_pointer", &Buffer{Size: 1024}, 1},
		{"zero_count", valid, 0},
		{"negative_count", valid, -1},
		{"short_buffer", valid, 257},
		{"overflow_count", valid, int(^uint(0) >> 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := GELUErfF32Buffer(tc.buf, tc.n); err == nil {
				t.Fatal("accepted invalid erf-GELU buffer or dimension")
			}
		})
	}
}

func TestGELUErfF32BufferIndependentParity(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	// Covers tails, negative saturation, and the layer-1 MLP's typical
	// central interval. The reference is F64 math.Erf, not a Go GPU path.
	input := make([]float32, 20001)
	for i := range input {
		input[i] = -10 + float32(i)*20/float32(len(input)-1)
	}
	original := append([]float32(nil), input...)
	buf := NewDevBufFrom(input)
	defer buf.Free()
	if err := buf.EnsureGPU(); err != nil {
		t.Fatal(err)
	}
	if err := GELUErfF32Buffer(buf.GPUBuffer(), len(input)); err != nil {
		t.Fatal(err)
	}
	if err := SyncErr(); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(input))
	if err := buf.GPUBuffer().Download(got); err != nil {
		t.Fatal(err)
	}
	var maxAbs, sumAbs float64
	var outside int
	for i, value := range got {
		x := float64(original[i])
		want := float32(0.5 * x * (1 + math.Erf(x/math.Sqrt2)))
		delta := math.Abs(float64(value - want))
		maxAbs = math.Max(maxAbs, delta)
		sumAbs += delta
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-6 {
			outside++
		}
		if input[i] != original[i] {
			t.Fatalf("caller input mutated at %d", i)
		}
	}
	t.Logf("max_abs=%g mean_abs=%g outside=%d", maxAbs, sumAbs/float64(len(got)), outside)
	if outside != 0 {
		t.Fatal("resident erf-GELU differs from independent reference")
	}
}
