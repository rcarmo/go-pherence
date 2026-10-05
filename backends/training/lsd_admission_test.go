package training

import (
	"math"
	"testing"
)

// Independent original host comparisons; no derived-cutoff arithmetic.
func legacyLSDMagnitudeAdmitted(channels int, lv float32, d float32) bool {
	exp := math.Exp(float64(lv))
	limit := math.Sqrt((float64(math.MaxFloat32) / 4) / float64(channels) / math.Max(1, exp))
	return !math.IsNaN(float64(d)) && !math.IsInf(float64(d), 0) && math.Abs(float64(d)) <= limit && 2*math.Abs(float64(d))*exp/float64(channels) <= float64(math.MaxFloat32)/4
}
func TestTrainingLSDPredictionCutoff(t *testing.T) {
	for _, channels := range []int{1, 2, 16, 1024, 16384} {
		for _, lv := range []float32{-80, -1, 0, .1, 1, 20, 80} {
			cutoff, err := LSDPredictionCutoff(channels, lv)
			if err != nil {
				t.Fatal(err)
			}
			if !legacyLSDMagnitudeAdmitted(channels, lv, cutoff) || legacyLSDMagnitudeAdmitted(channels, lv, math.Nextafter32(cutoff, float32(math.Inf(1)))) {
				t.Fatal("not maximal F32 cutoff", channels, lv, cutoff)
			}
			bits := math.Float32bits(cutoff)
			for _, d := range []float32{0, cutoff, -cutoff, math.Float32frombits(bits - 1), -math.Float32frombits(bits - 1), math.Float32frombits(bits + 1), -math.Float32frombits(bits + 1), math.MaxFloat32, float32(math.Inf(1)), float32(math.NaN())} {
				device := !math.IsNaN(float64(d)) && !math.IsInf(float64(d), 0) && float32(math.Abs(float64(d))) <= cutoff
				if device != legacyLSDMagnitudeAdmitted(channels, lv, d) {
					t.Fatal("host/device predicate", channels, lv, d)
				}
			}
		}
	}
	for _, kind := range []struct {
		n  int
		lv float32
	}{{0, 0}, {16385, 0}, {1, -81}, {1, 81}, {1, float32(math.NaN())}, {1, float32(math.Inf(1))}} {
		if _, err := LSDPredictionCutoff(kind.n, kind.lv); err == nil {
			t.Fatal("bad cutoff input")
		}
	}
}
func TestTrainingLSDPredictionAdmissionTape(t *testing.T) {
	cutoff, _ := LSDPredictionCutoff(2, .1)
	makeTape := func() Tape {
		return Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{2, 1}}, {Shape: []int{1}, Values: []float32{cutoff}}, {Shape: []int{1}}}, Commands: []Command{{Op: "lsd-loss", Inputs: []int{0, 1, 2}, Output: 3, Mode: 2, Scale: .1}}, Outputs: []int{3}}
	}
	if err := makeTape().Validate(24); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*Tape){func(x *Tape) { x.Tensors[2].Values = nil }, func(x *Tape) { x.Tensors[2].Values[0] = math.Nextafter32(cutoff, float32(math.Inf(1))) }, func(x *Tape) { x.Commands[0].Scale = 81 }, func(x *Tape) { x.Commands[0].Scale = float32(math.NaN()) }, func(x *Tape) { x.Tensors[3].Shape = []int{3} }, func(x *Tape) {
		x.Tensors = append(x.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
	}} {
		x := makeTape()
		change(&x)
		if x.Validate(100) == nil {
			t.Fatal("bad admission tape")
		}
	}
	if makeTape().Validate(23) == nil {
		t.Fatal("budget")
	}
}
