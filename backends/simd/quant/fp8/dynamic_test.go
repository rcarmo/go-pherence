package fp8

import (
	"math"
	"testing"
)

func TestDecodeE4M3MatchesReferenceValues(t *testing.T) {
	cases := map[byte]float32{
		0x00: 0,
		0x01: 0.001953125,
		0x05: 0.009765625,
		0x1d: 0.1015625,
		0x38: 1,
		0x39: 1.125,
		0x76: 224,
		0x7e: 448,
		0x81: -0.001953125,
		0x85: -0.009765625,
		0x9d: -0.1015625,
		0xb8: -1,
		0xb9: -1.125,
		0xf6: -224,
		0xfe: -448,
	}
	for code, want := range cases {
		if got := DecodeE4M3(code); got != want {
			t.Fatalf("DecodeE4M3(0x%02x)=%g want %g", code, got, want)
		}
	}
	if !math.IsNaN(float64(DecodeE4M3(0x7f))) || !math.IsNaN(float64(DecodeE4M3(0xff))) {
		t.Fatalf("NaN codes not preserved")
	}
}

func TestQuantizeTokenE4M3DequantTo(t *testing.T) {
	x := []float32{-1, -0.5, 0, 0.25, 1}
	out := make([]float32, len(x))
	QuantizeTokenE4M3DequantTo(out, x)
	for i, v := range out {
		if math.IsNaN(float64(v)) {
			t.Fatalf("out[%d] is NaN", i)
		}
	}
	if out[2] != 0 {
		t.Fatalf("zero not preserved: %v", out)
	}
	if out[0] >= 0 || out[4] <= 0 {
		t.Fatalf("sign not preserved: %v", out)
	}
}

func TestQuantizeTokenE4M3DequantToUsesPerTokenScale(t *testing.T) {
	x := []float32{-500, -125, 0, 125, 500}
	out := make([]float32, len(x))
	QuantizeTokenE4M3DequantTo(out, x)
	// The maximum absolute value maps to finite E4M3 max 448 and then is
	// dequantized by maxAbs/448, so it round-trips exactly to ±maxAbs.
	if out[0] != -500 || out[4] != 500 {
		t.Fatalf("max token scale mismatch: %v", out)
	}
	if out[2] != 0 {
		t.Fatalf("zero not preserved with token scale: %v", out)
	}
}

func TestLinearGemvToDynamicToken(t *testing.T) {
	l := Linear{OutDim: 2, InDim: 3, Weight: []byte{EncodeE4M3Nearest(1), EncodeE4M3Nearest(0.5), EncodeE4M3Nearest(-1), EncodeE4M3Nearest(0.25), EncodeE4M3Nearest(1), EncodeE4M3Nearest(0.75)}, Scale: []float32{1, 1}}
	x := []float32{0.2, -0.4, 0.8}
	out := make([]float32, 2)
	if err := l.GemvToDynamicToken(x, out, make([]float32, 3)); err != nil {
		t.Fatal(err)
	}
	if out[0] == 0 && out[1] == 0 {
		t.Fatalf("dynamic gemv returned zero output")
	}
}

func TestLinearGemvToDynamicTokenRejectsShortBuffers(t *testing.T) {
	l := Linear{OutDim: 2, InDim: 3, Weight: make([]byte, 6), Scale: []float32{1, 1}}
	if err := l.GemvToDynamicToken([]float32{1, 2}, make([]float32, 2), make([]float32, 3)); err == nil {
		t.Fatalf("expected short input error")
	}
	if err := l.GemvToDynamicToken([]float32{1, 2, 3}, make([]float32, 1), make([]float32, 3)); err == nil {
		t.Fatalf("expected short output error")
	}
	if err := l.GemvToDynamicToken([]float32{1, 2, 3}, make([]float32, 2), make([]float32, 2)); err == nil {
		t.Fatalf("expected short scratch error")
	}
}

func TestLinearBatchGemvRejectsShortBuffers(t *testing.T) {
	l := Linear{OutDim: 2, InDim: 3, Weight: make([]byte, 6), Scale: []float32{1, 1}}
	if err := l.BatchGemvTo([]float32{1, 2, 3, 4, 5}, make([]float32, 4), 2); err == nil {
		t.Fatalf("expected short batched input error")
	}
	if err := l.BatchGemvTo(make([]float32, 6), make([]float32, 3), 2); err == nil {
		t.Fatalf("expected short batched output error")
	}
	if err := l.BatchGemvToBuf(make([]float32, 6), make([]float32, 4), 2, make([]float32, 2)); err == nil {
		t.Fatalf("expected short dequant scratch error")
	}
	if err := l.BatchGemvToBufDynamicToken(make([]float32, 6), make([]float32, 4), 2, make([]float32, 3), make([]float32, 5)); err == nil {
		t.Fatalf("expected short dynamic activation scratch error")
	}
}

func TestLinearValidateRejectsOverflowShape(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	l := Linear{OutDim: maxInt/2 + 1, InDim: 3, Weight: nil, Scale: []float32{1}}
	if err := l.Validate(); err == nil {
		t.Fatalf("expected overflow validation error")
	}
}
