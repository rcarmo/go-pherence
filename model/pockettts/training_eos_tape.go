package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
)

// TrainingEOSBackend evaluates unweighted mean BCE and weighted logit seeds.
// Nil FlowLM backend preserves the CPU F64 loss and legacy replay arithmetic.
type TrainingEOSBackend interface {
	EvaluateEOS(*FlowLMTrainingCPU, []float32, []bool, float32) (float64, []float32, error)
}

func BuildEOSTrainingTape(logits []float32, mask []bool, weight float32, maxBytes int64) (train.Tape, error) {
	var tape train.Tape
	n := len(logits)
	if n < 1 || n > 1024 || n != len(mask) || maxBytes <= 0 || !isFinite(weight) || weight < 0 {
		return tape, fmt.Errorf("invalid EOS tape inputs/ceiling")
	}
	labels, count, err := eosTrainingLabels(mask)
	if err != nil {
		return tape, err
	}
	for i, label := range labels {
		if label >= 0 && !isFinite(logits[i]) {
			return tape, fmt.Errorf("EOS tape active logit nonfinite")
		}
	}
	if _, err := train.EOSShape(n, count, 0, weight); err != nil {
		return tape, err
	}
	// Leave headroom for summation rounding and compensation. The CPU F64
	// loss contract stays wider; device modes reject before native mutation.
	safe := (float32(math.MaxFloat32) / 2) / float32(count)
	values := make([]float32, n)
	for i, label := range labels {
		if label >= 0 {
			if float32(math.Abs(float64(logits[i]))) > safe {
				return tape, fmt.Errorf("EOS tape reduction overflow ceiling")
			}
			values[i] = logits[i]
		}
	}
	if int64(4*(4*n+1)) > maxBytes {
		return tape, fmt.Errorf("EOS tape byte ceiling")
	}
	tape = train.Tape{Tensors: []train.TensorSpec{{Shape: []int{n}, Values: values}, {Shape: []int{n}, Values: labels}, {Shape: []int{n}}, {Shape: []int{n}}, {Shape: []int{1}}}, Commands: []train.Command{{Op: "eos", Inputs: []int{0, 1}, Output: 2, Rows: count, Mode: 0, Scale: weight}, {Op: "eos", Inputs: []int{0, 1}, Output: 3, Rows: count, Mode: 1, Scale: weight}, {Op: "eos", Inputs: []int{2, 1}, Output: 4, Rows: count, Mode: 2, Scale: weight}}, Outputs: []int{4, 3}}
	return tape, tape.Validate(maxBytes)
}
func (b *ResidentTrainingBackend) ownsEOS() bool {
	return b != nil && (b.Scope == "ptx-resident-eos-hybrid-v1" || b.Scope == "vulkan-resident-eos-hybrid-v1" || b.ownsLoss())
}
func (b *ResidentTrainingBackend) eosOwnerMatches(t *FullTrainer) bool {
	if b.ownsEOS() {
		return t.FlowLM.EOSBackend == b
	}
	return t.FlowLM.EOSBackend == nil
}
func (b *ResidentTrainingBackend) EvaluateEOS(m *FlowLMTrainingCPU, logits []float32, mask []bool, weight float32) (float64, []float32, error) {
	if err := b.admitGraph(); err != nil {
		return 0, nil, err
	}
	if !b.ownsEOS() || m != b.trainer.FlowLM {
		return 0, nil, b.fail(fmt.Errorf("resident EOS owner/scope changed"))
	}
	tape, err := BuildEOSTrainingTape(logits, mask, weight, b.MaxTapeBytes)
	if err != nil {
		return 0, nil, b.fail(err)
	}
	values, err := b.runUtilityTape(tape, tape.Outputs)
	if err != nil {
		return 0, nil, b.fail(err)
	}
	loss, seed, err := validateEOSBoundary(values, logits, mask)
	if err != nil {
		return 0, nil, b.fail(err)
	}
	return loss, seed, nil
}
func validateEOSBoundary(values [][]float32, logits []float32, mask []bool) (float64, []float32, error) {
	if len(logits) != len(mask) || len(values) != 2 || len(values[0]) != 1 || len(values[1]) != len(logits) || !finiteF32(values[0]) || !finiteF32(values[1]) || values[0][0] < 0 {
		return 0, nil, fmt.Errorf("EOS boundary shape/nonfinite/negative loss")
	}
	if f32SlicesOverlap(values[0], values[1]) || f32SlicesOverlap(values[0], logits) || f32SlicesOverlap(values[1], logits) {
		return 0, nil, fmt.Errorf("EOS borrowed/overlapping output")
	}
	for i := range mask {
		active := mask[i]
		if i > 0 {
			active = mask[i-1]
		}
		if !active && values[1][i] != 0 {
			return 0, nil, fmt.Errorf("EOS inactive gradient")
		}
	}
	return float64(values[0][0]), values[1], nil
}
func eosTrainingLabels(mask []bool) ([]float32, int, error) {
	labels := make([]float32, len(mask))
	count := 0
	seenPadding := false
	for i, valid := range mask {
		if !valid {
			seenPadding = true
		} else if seenPadding {
			return nil, 0, fmt.Errorf("EOS tape mask is not a contiguous prefix")
		}
		active := valid
		if i > 0 {
			active = mask[i-1]
		}
		labels[i] = -1
		if active {
			labels[i] = 0
			if i > 0 && !valid {
				labels[i] = 1
			}
			count++
		}
	}
	if count == 0 {
		return nil, 0, fmt.Errorf("EOS tape has no active positions")
	}
	return labels, count, nil
}
