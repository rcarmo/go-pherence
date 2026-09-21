package jevlike

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestVisionEncoderPyTorchParity(t *testing.T) {
	var ref struct {
		Width   int
		Weights map[string]struct {
			Shape  []int
			Values []float32
		}
		Features  [][][]float32
		Positions [][]float32
	}
	raw, err := os.ReadFile("testdata/vision_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	enc, err := NewVisionEncoder(ref.Width)
	if err != nil {
		t.Fatal(err)
	}
	params := make(map[string][]float32)
	for name, p := range ref.Weights {
		params[name] = p.Values
	}
	if err = enc.LoadNamedParameters(params); err != nil {
		t.Fatal(err)
	}
	x := ObservationTensorResult{Shape: [4]int{1, 4, 120, 160}, Values: make([]float32, 4*120*160)}
	for i := range x.Values {
		x.Values[i] = float32(i%251) / 255
	}
	got, positions, err := enc.EncodeTensor(x)
	if err != nil {
		t.Fatal(err)
	}
	var delta float64
	for b := range got {
		for row := range got[b] {
			for d, v := range got[b][row] {
				diff := math.Abs(float64(v - ref.Features[b][row][d]))
				delta = math.Max(delta, diff)
				if diff > 2e-5 {
					t.Fatalf("feature %d,%d,%d got%g want%g", b, row, d, v, ref.Features[b][row][d])
				}
			}
		}
	}
	for row := range ref.Positions {
		for d, want := range ref.Positions[row] {
			if math.Abs(float64(positions[row*ref.Width+d]-want)) > 1e-6 {
				t.Fatal("position parity")
			}
		}
	}
	t.Logf("visual feature max absolute difference=%g", delta)
}
