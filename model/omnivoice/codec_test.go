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

func TestCodecPrepareReuseAndBounds(t *testing.T) {
	d := &CodecDecoder{weights: &loader.CodecWeights{}}
	if err := d.Prepare(3); err != nil {
		t.Fatal(err)
	}
	first := d.scratch
	if first == nil || first.frames != 3 || first.maxFrames != 3 {
		t.Fatal("scratch not prepared")
	}
	if err := d.Prepare(1); err != nil {
		t.Fatal(err)
	}
	if d.scratch != first || d.scratch.frames != 1 || d.scratch.maxFrames != 3 {
		t.Fatal("smaller prepare reallocated scratch")
	}
	allocs := testing.AllocsPerRun(50, func() {
		if err := d.Prepare(2); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("prepare reuse allocs=%g", allocs)
	}
	if err := d.Prepare(4); err != nil {
		t.Fatal(err)
	}
	if d.scratch == first || d.scratch.frames != 4 || d.scratch.maxFrames != 4 {
		t.Fatal("larger prepare did not replace scratch")
	}
	for _, frames := range []int{0, 251} {
		if err := d.Prepare(frames); err == nil {
			t.Fatalf("accepted frames=%d", frames)
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
	exact := append([]float32(nil), out...)
	poison := func() {
		buffers := append([][]float32(nil), d.scratch.slots[:]...)
		buffers = append(buffers, d.scratch.packed, d.scratch.result, d.scratch.input, d.scratch.projected, d.scratch.gemmPanel)
		for _, buf := range buffers {
			for i := range buf {
				buf[i] = float32(math.NaN())
			}
		}
	}
	poison()
	if err := d.DecodeInto(context.Background(), out, codes, 8, 2); err != nil {
		t.Fatal(err)
	}
	assertFloat32Exact(t, out, exact)
	first := d.scratch
	if err := d.Prepare(1); err != nil {
		t.Fatal(err)
	}
	if d.scratch != first {
		t.Fatal("smaller real prepare reallocated scratch")
	}
	poison()
	if err := d.DecodeInto(context.Background(), make([]float32, 960), make([]int, 8), 8, 1); err != nil {
		t.Fatal(err)
	}
	if err := d.Prepare(2); err != nil {
		t.Fatal(err)
	}
	poison()
	ctx := &cancelAfterChecks{Context: context.Background(), remaining: 3}
	if err := d.DecodeInto(ctx, out, codes, 8, 2); err != context.Canceled {
		t.Fatalf("mid-decode cancellation: %v", err)
	}
	if err := d.DecodeInto(context.Background(), out, codes, 8, 2); err != nil {
		t.Fatal(err)
	}
	assertFloat32Exact(t, out, exact)
	for i, v := range out {
		if diff := math.Abs(float64(v - f.Waveform[i])); diff > 2e-5 || math.IsNaN(diff) {
			t.Fatalf("decode retry mismatch at %d", i)
		}
	}
}
