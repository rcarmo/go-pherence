package fft

import (
	"math"
	"testing"
	"unsafe"
)

func TestForwardRealDC(t *testing.T) {
	// All ones → DC bin has magnitude n, all others ~0
	n := 16
	input := make([]float32, n)
	for i := range input {
		input[i] = 1
	}
	out := ForwardReal(input)
	if out == nil {
		t.Fatal("ForwardReal returned nil")
	}
	// DC bin (index 0): real should be n, imag should be 0
	if math.Abs(float64(out[0])-float64(n)) > 0.01 {
		t.Fatalf("DC real=%f want %d", out[0], n)
	}
	if math.Abs(float64(out[1])) > 0.01 {
		t.Fatalf("DC imag=%f want 0", out[1])
	}
}

func TestForwardRealCosine(t *testing.T) {
	// Cosine at bin 3 of n=32
	n := 32
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(math.Cos(2 * math.Pi * 3 * float64(i) / float64(n)))
	}
	out := ForwardReal(input)
	if out == nil {
		t.Fatal("nil")
	}
	// Bin 3 should have magnitude n/2 = 16
	re := out[2*3]
	im := out[2*3+1]
	mag := float32(math.Sqrt(float64(re*re + im*im)))
	if mag < 15.5 || mag > 16.5 {
		t.Fatalf("bin 3 magnitude=%f want ~16", mag)
	}
	// DC should be ~0
	if math.Abs(float64(out[0])) > 0.1 {
		t.Fatalf("DC=%f want ~0", out[0])
	}
}

func TestPowerSpectrum(t *testing.T) {
	n := 16
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(math.Cos(2 * math.Pi * 2 * float64(i) / float64(n)))
	}
	power := PowerSpectrum(input)
	if len(power) != n/2+1 {
		t.Fatalf("power length=%d want %d", len(power), n/2+1)
	}
	// Bin 2 should have most energy
	maxBin := 0
	maxVal := power[0]
	for i, v := range power {
		if v > maxVal {
			maxVal = v
			maxBin = i
		}
	}
	if maxBin != 2 {
		t.Fatalf("max power at bin %d want 2", maxBin)
	}
}

func TestForwardRealIntoScratchAndBounds(t *testing.T) {
	for _, n := range []int{1, 16, 32, 512} {
		input := make([]float32, n)
		for i := range input {
			input[i] = float32(math.Sin(float64(i) * .13))
		}
		original := append([]float32(nil), input...)
		re, im := make([]float64, n+2), make([]float64, n+2)
		out := make([]float32, (n/2+1)*2+2)
		re[n], im[n], out[len(out)-1] = 11, 12, 13
		for pass := 0; pass < 2; pass++ {
			for i := 0; i < n; i++ {
				re[i], im[i] = 42, 99
			}
			if !ForwardRealInto(out, input, re, im) {
				t.Fatalf("n=%d valid buffers rejected", n)
			}
			want := ForwardReal(input)
			for i := range want {
				if math.Abs(float64(want[i]-out[i])) > 2e-5 {
					t.Fatalf("n=%d pass=%d bin=%d got=%g want=%g", n, pass, i, out[i], want[i])
				}
			}
			if re[n] != 11 || im[n] != 12 || out[len(out)-1] != 13 {
				t.Fatalf("n=%d scratch overrun", n)
			}
		}
		for i := range input {
			if input[i] != original[i] {
				t.Fatalf("n=%d input mutated", n)
			}
		}
		if ForwardRealInto(out[:len(out)-3], input, re, im) || ForwardRealInto(out, input, re[:n-1], im) || ForwardRealInto(out, input, re, im[:n-1]) {
			t.Fatalf("n=%d accepted undersized buffers", n)
		}
		if ForwardRealInto(out, input, re, re) {
			t.Fatalf("n=%d accepted overlapping FFT scratch", n)
		}
		if n >= 4 {
			alias := unsafe.Slice((*float32)(unsafe.Pointer(&re[0])), n*2)
			if ForwardRealInto(out, alias[:n], re, im) {
				t.Fatalf("n=%d accepted input/scratch overlap", n)
			}
		}
	}
	if ForwardRealInto(make([]float32, 10), make([]float32, 7), make([]float64, 7), make([]float64, 7)) {
		t.Fatal("accepted non-power-of-two FFT")
	}
}

func TestForwardRealNonPow2(t *testing.T) {
	// Non-power-of-2 should return nil
	input := make([]float32, 7)
	if out := ForwardReal(input); out != nil {
		t.Fatalf("expected nil for non-pow2, got %d elements", len(out))
	}
}
