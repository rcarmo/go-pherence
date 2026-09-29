package nvidia

import (
	"fmt"
	"math"
	"testing"
)

func TestAffineLayerNormF32BufferRejectMalformed(t *testing.T) {
	out, x := &Buffer{Ptr: 0x10000, Size: 512 * 4}, &Buffer{Ptr: 0x20000, Size: 512 * 4}
	gamma, beta := &Buffer{Ptr: 0x30000, Size: 512 * 4}, &Buffer{Ptr: 0x40000, Size: 512 * 4}
	short := &Buffer{Ptr: 0x50000, Size: 511 * 4}
	for _, tc := range []struct {
		name                string
		out, x, gamma, beta *Buffer
		rows, cols          int
		eps                 float32
	}{
		{"nil_out", nil, x, gamma, beta, 1, 512, 1e-5},
		{"zero_pointer", &Buffer{Size: 2048}, x, gamma, beta, 1, 512, 1e-5},
		{"short_output", short, x, gamma, beta, 1, 512, 1e-5},
		{"short_input", out, short, gamma, beta, 1, 512, 1e-5},
		{"short_gamma", out, x, short, beta, 1, 512, 1e-5},
		{"short_beta", out, x, gamma, short, 1, 512, 1e-5},
		{"zero_rows", out, x, gamma, beta, 0, 512, 1e-5},
		{"too_many_rows", out, x, gamma, beta, 65536, 512, 1e-5},
		{"negative_cols", out, x, gamma, beta, 1, -1, 1e-5},
		{"overflow", out, x, gamma, beta, 65535, int(^uint(0) >> 1), 1e-5},
		{"negative_eps", out, x, gamma, beta, 1, 512, -1e-5},
		{"nan_eps", out, x, gamma, beta, 1, 512, float32(math.NaN())},
		{"infinite_eps", out, x, gamma, beta, 1, 512, float32(math.Inf(1))},
		{"aliased_gamma", gamma, x, gamma, beta, 1, 512, 1e-5},
		{"aliased_beta", beta, x, gamma, beta, 1, 512, 1e-5},
		{"overlapping_gamma", &Buffer{Ptr: gamma.Ptr + 4, Size: 2048}, x, gamma, beta, 1, 512, 1e-5},
		{"overlapping_input", &Buffer{Ptr: x.Ptr + 4, Size: 2048}, x, gamma, beta, 1, 512, 1e-5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := AffineLayerNormF32Buffer(tc.out, tc.x, tc.gamma, tc.beta, tc.rows, tc.cols, tc.eps); err == nil {
				t.Fatal("accepted malformed buffer or dimensions")
			}
		})
	}
}

func TestAffineLayerNormF32ShapesAndInPlace(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	for _, cols := range []int{1, 17, 257, 513} {
		t.Run(fmt.Sprintf("cols=%d", cols), func(t *testing.T) {
			const rows = 2
			x, gamma, beta := make([]float32, rows*cols), make([]float32, cols), make([]float32, cols)
			for c := range gamma {
				gamma[c], beta[c] = 0.75+float32(c%3)/10, float32(c%5-2)/10
				for r := 0; r < rows; r++ {
					x[r*cols+c] = float32((c%19)-9)/4 + float32(r)
				}
			}
			want := make([]float32, len(x))
			for r := 0; r < rows; r++ {
				var mean, variance float64
				for c := 0; c < cols; c++ {
					mean += float64(x[r*cols+c])
				}
				mean /= float64(cols)
				for c := 0; c < cols; c++ {
					d := float64(x[r*cols+c]) - mean
					variance += d * d
				}
				variance /= float64(cols)
				for c := 0; c < cols; c++ {
					want[r*cols+c] = float32((float64(x[r*cols+c])-mean)/math.Sqrt(variance+1e-5)*float64(gamma[c]) + float64(beta[c]))
				}
			}
			bX, bGamma, bBeta := NewDevBufFrom(x), NewDevBufFrom(gamma), NewDevBufFrom(beta)
			defer bX.Free()
			defer bGamma.Free()
			defer bBeta.Free()
			for _, b := range []*DevBuf{bX, bGamma, bBeta} {
				if err := b.EnsureGPU(); err != nil {
					t.Fatal(err)
				}
			}
			if err := AffineLayerNormF32Buffer(bX.GPUBuffer(), bX.GPUBuffer(), bGamma.GPUBuffer(), bBeta.GPUBuffer(), rows, cols, 1e-5); err != nil {
				t.Fatal(err)
			}
			if err := SyncErr(); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, len(x))
			if err := bX.GPUBuffer().Download(got); err != nil {
				t.Fatal(err)
			}
			var max float64
			for i := range got {
				delta := math.Abs(float64(got[i] - want[i]))
				max = math.Max(max, delta)
				if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || delta > 1e-5 {
					t.Fatalf("index %d: got %g want %g delta %g", i, got[i], want[i], delta)
				}
			}
			t.Logf("max_abs=%g", max)
		})
	}
}

func TestAffineLayerNormF32HighAmplitude(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	const rows, cols = 3, 512
	x, gamma, beta := make([]float32, rows*cols), make([]float32, cols), make([]float32, cols)
	for c := range gamma {
		gamma[c] = 0.5 + float32(c%7)/10
		beta[c] = float32(c%13-6) / 100
		for r := 0; r < rows; r++ {
			x[r*cols+c] = float32((r+1)*10000) + float32(c%17-8)*0.25
		}
	}
	want := make([]float32, len(x))
	for r := 0; r < rows; r++ {
		var mean, variance float64
		for c := 0; c < cols; c++ {
			mean += float64(x[r*cols+c])
		}
		mean /= cols
		for c := 0; c < cols; c++ {
			d := float64(x[r*cols+c]) - mean
			variance += d * d
		}
		variance /= cols
		for c := 0; c < cols; c++ {
			want[r*cols+c] = float32((float64(x[r*cols+c])-mean)/math.Sqrt(variance+1e-5)*float64(gamma[c]) + float64(beta[c]))
		}
	}
	bX, bGamma, bBeta := NewDevBufFrom(x), NewDevBufFrom(gamma), NewDevBufFrom(beta)
	bOut, err := NewDevBufGPU(len(x))
	if err != nil {
		t.Fatal(err)
	}
	defer bX.Free()
	defer bGamma.Free()
	defer bBeta.Free()
	defer bOut.Free()
	for _, b := range []*DevBuf{bX, bGamma, bBeta} {
		if err := b.EnsureGPU(); err != nil {
			t.Fatal(err)
		}
	}
	if err := AffineLayerNormF32Buffer(bOut.GPUBuffer(), bX.GPUBuffer(), bGamma.GPUBuffer(), bBeta.GPUBuffer(), rows, cols, 1e-5); err != nil {
		t.Fatal(err)
	}
	if err := SyncErr(); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(x))
	if err := bOut.GPUBuffer().Download(got); err != nil {
		t.Fatal(err)
	}
	var max float64
	for i := range got {
		delta := math.Abs(float64(got[i] - want[i]))
		max = math.Max(max, delta)
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || delta > 2e-3 {
			t.Fatalf("index %d: got %g want %g delta %g", i, got[i], want[i], delta)
		}
	}
	t.Logf("max_abs=%g", max)
}
