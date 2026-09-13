package omnivoice

import (
	"context"
	"encoding/json"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"math"
	"os"
	"testing"
)

func TestCodecConvAndTranspose(t *testing.T) {
	w := &loader.CodecWeights{Tensors: map[string][]float32{"c.weight": {1, 2, 3}, "c.bias": {.5}, "t.weight": {1, 2, 3}, "t.bias": {.5}}, Shapes: map[string][]int{"c.weight": {1, 1, 3}, "t.weight": {1, 1, 3}}}
	d := &CodecDecoder{weights: w}
	x := signal{[]float32{1, 2, 3}, 1, 3}
	y, err := d.conv(x, "c", 1, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float32{8.5, 14.5, 8.5} {
		if y.data[i] != v {
			t.Fatalf("conv %v", y.data)
		}
	}
	z, err := d.transpose(x, "t", 2, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float32{2.5, 5.5, 4.5, 9.5, 6.5, 9.5} {
		if z.data[i] != v {
			t.Fatalf("transpose %v", z.data)
		}
	}
}
func TestRealCodecDecode(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_REAL_CODEC")
	if path == "" {
		t.Skip("set codec directory")
	}
	w, err := loader.LoadCodecDecoder(path)
	if err != nil {
		t.Fatal(err)
	}
	d, err := NewCodecDecoder(w)
	if err != nil {
		t.Fatal(err)
	}
	codes := make([]int, 8*2)
	for i := range codes {
		codes[i] = i * 13 % 1024
	}
	out, err := d.Decode(context.Background(), codes, 8, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1920 {
		t.Fatalf("samples=%d", len(out))
	}
	raw, err := os.ReadFile("../../testdata/omnivoice/codec-real.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Waveform []float32 `json:"waveform"`
	}
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	maxError := float64(0)
	for i, v := range out {
		diff := math.Abs(float64(v - f.Waveform[i]))
		maxError = math.Max(maxError, diff)
		if diff > 2e-5 || math.IsNaN(diff) {
			t.Fatalf("sample %d got %g want %g diff %g", i, v, f.Waveform[i], diff)
		}
	}
	t.Logf("all %d samples parity; max error %g", len(out), maxError)
	var decodeErr error
	allocations := testing.AllocsPerRun(3, func() { decodeErr = d.DecodeInto(context.Background(), out, codes, 8, 2) })
	if decodeErr != nil {
		t.Fatal(decodeErr)
	}
	t.Logf("DecodeInto allocations=%g", allocations)
	if allocations != 0 {
		t.Fatalf("decoder still allocates %g", allocations)
	}
}
