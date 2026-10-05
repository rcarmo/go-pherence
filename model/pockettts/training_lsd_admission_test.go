package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestTrainingLSDAdmissionDevicePredicate(t *testing.T) {
	for _, channels := range []int{1, 16, 1024, 16384} {
		for _, lv := range []float32{-80, 0, .1, 20, 80} {
			cutoff, err := train.LSDPredictionCutoff(channels, lv)
			if err != nil {
				t.Fatal(err)
			}
			for _, value := range []float32{0, cutoff, -cutoff, math.Nextafter32(cutoff, float32(math.Inf(1))), -math.Nextafter32(cutoff, float32(math.Inf(1)))} {
				prediction, desired := make([]float32, channels), make([]float32, channels)
				prediction[channels-1] = value
				tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{channels}, Values: prediction}, {Shape: []int{channels}, Values: desired}, {Shape: []int{1}, Values: []float32{cutoff}}, {Shape: []int{1}}}, Commands: []train.Command{{Op: "lsd-loss", Inputs: []int{0, 1, 2}, Output: 3, Mode: 2, Scale: lv}}, Outputs: []int{3}}
				if err := tape.Validate(1 << 20); err != nil {
					t.Fatal(err)
				}
				admitted := admitLSDLossTraining(prediction, desired, lv, 1<<20) == nil
				out := interpretFlowTape(t, tape)[0]
				if (out[0] == 0) != admitted {
					t.Fatal("host/device admission", channels, lv, value, out)
				}
				pruned, _, err := train.PruneTape(tape, tape.Outputs, 1<<20)
				if err != nil {
					t.Fatal(err)
				}
				if interpretFlowTape(t, pruned)[0][0] != out[0] {
					t.Fatal("pruned scalar admission")
				}
			}
		}
	}
}

// Finite admitted sources generate Inf/NaN at runtime, without bypassing
// Tape.Validate or the interpreter's complete-source admission.
func TestTrainingLSDAdmissionRuntimeInvalid(t *testing.T) {
	for _, kind := range []string{"prediction-inf", "desired-inf", "prediction-nan", "difference-overflow"} {
		cutoff, _ := train.LSDPredictionCutoff(1, 0)
		tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{math.MaxFloat32}}, {Shape: []int{1}, Values: []float32{-math.MaxFloat32}}, {Shape: []int{1}, Values: []float32{0}}, {Shape: []int{1}, Values: []float32{cutoff}}, {Shape: []int{1}}, {Shape: []int{1}}, {Shape: []int{1}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 0}, Output: 4, Scale: 1}, {Op: "vector", Inputs: []int{4, 4}, Output: 5, Scale: -1}}, Outputs: []int{6}}
		p, y := 4, 2
		switch kind {
		case "desired-inf":
			p, y = 2, 4
		case "prediction-nan":
			p = 5
		case "difference-overflow":
			p, y = 0, 1
		}
		tape.Commands = append(tape.Commands, train.Command{Op: "lsd-loss", Inputs: []int{p, y, 3}, Output: 6, Mode: 2, Scale: 0})
		if interpretFlowTape(t, tape)[0][0] != 1 {
			t.Fatal("invalid runtime accepted", kind)
		}
	}
}

func TestTrainingLSDAdmissionDifferenceRounding(t *testing.T) {
	for _, lv := range []float32{-80, 0, .1, 80} {
		cutoff, _ := train.LSDPredictionCutoff(1, lv)
		pairs := [][2]float32{{math.MaxFloat32, math.MaxFloat32}, {cutoff, 0}, {-cutoff, 0}, {cutoff / 2, -cutoff / 2}, {math.Nextafter32(cutoff, float32(math.Inf(1))), 0}, {math.Nextafter32(cutoff/2, float32(math.Inf(1))), -cutoff / 2}, {math.SmallestNonzeroFloat32, 0}, {0, -math.SmallestNonzeroFloat32}}
		for _, pair := range pairs {
			p, y := []float32{pair[0]}, []float32{pair[1]}
			tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: p}, {Shape: []int{1}, Values: y}, {Shape: []int{1}, Values: []float32{cutoff}}, {Shape: []int{1}}}, Commands: []train.Command{{Op: "lsd-loss", Inputs: []int{0, 1, 2}, Output: 3, Mode: 2, Scale: lv}}, Outputs: []int{3}}
			got := interpretFlowTape(t, tape)[0][0] == 0
			want := admitLSDLossTraining(p, y, lv, 64<<20) == nil
			if got != want {
				t.Fatal("F32 difference rounding", lv, pair, got, want)
			}
		}
	}
}
