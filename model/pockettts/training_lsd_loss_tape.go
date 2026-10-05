package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
)

// TrainingLSDLossBackend evaluates one normalized diagonal or minimal-LSD row.
// Mean over valid rows/objective weights remain in the step orchestration.
type TrainingLSDLossBackend interface {
	EvaluateLSDLoss(*FlowHeadCPU, []float32, []float32, float32) (LSDLossTrainingResult, error)
}
type LSDLossTrainingResult struct {
	RawSquare, Loss float64
	DLogVariance    float32
	DPrediction     []float32
}

func BuildLSDLossTrainingTape(prediction, desired []float32, logvar float32, maxBytes int64) (train.Tape, error) {
	var tape train.Tape
	if err := admitLSDLossTraining(prediction, desired, logvar, maxBytes); err != nil {
		return tape, err
	}
	c := len(prediction)
	tape = train.Tape{Tensors: []train.TensorSpec{{Shape: []int{c}, Values: append([]float32(nil), prediction...)}, {Shape: []int{c}, Values: append([]float32(nil), desired...)}, {Shape: []int{1}, Values: []float32{logvar}}, {Shape: []int{3}}, {Shape: []int{c}}}, Commands: []train.Command{{Op: "lsd-loss", Inputs: []int{0, 1, 2}, Output: 3}, {Op: "lsd-loss", Inputs: []int{0, 1, 2}, Output: 4, Mode: 1}}, Outputs: []int{3, 4}}
	return tape, tape.Validate(maxBytes)
}
func validateLSDLossBoundary(r LSDLossTrainingResult, prediction, desired []float32) error {
	if !finite64(r.RawSquare) || r.RawSquare < 0 || !finite64(r.Loss) || !isFinite(r.DLogVariance) || len(r.DPrediction) != len(prediction) || !finiteF32(r.DPrediction) {
		return fmt.Errorf("LSD loss boundary shape/nonfinite/square")
	}
	if f32SlicesOverlap(r.DPrediction, prediction) || f32SlicesOverlap(r.DPrediction, desired) {
		return fmt.Errorf("LSD loss borrowed gradient")
	}
	return nil
}
func (b *ResidentTrainingBackend) ownsLoss() bool {
	return b != nil && (b.Scope == "ptx-resident-loss-hybrid-v1" || b.Scope == "vulkan-resident-loss-hybrid-v1" || b.ownsBranch())
}
func (b *ResidentTrainingBackend) lossOwnerMatches(t *FullTrainer) bool {
	if b.ownsLoss() {
		return t.Flow.LossBackend == b
	}
	return t.Flow.LossBackend == nil
}
func (b *ResidentTrainingBackend) EvaluateLSDLoss(m *FlowHeadCPU, prediction, desired []float32, logvar float32) (LSDLossTrainingResult, error) {
	var result LSDLossTrainingResult
	if err := b.admitGraph(); err != nil {
		return result, err
	}
	if !b.ownsLoss() || m != b.trainer.Flow {
		return result, b.fail(fmt.Errorf("resident LSD loss owner/scope changed"))
	}
	tape, err := BuildLSDLossTrainingTape(prediction, desired, logvar, b.MaxTapeBytes)
	if err != nil {
		return result, b.fail(err)
	}
	values, err := b.runUtilityTape(tape, tape.Outputs)
	if err != nil {
		return result, b.fail(err)
	}
	if len(values) != 2 || len(values[0]) != 3 {
		return result, b.fail(fmt.Errorf("LSD loss result shape"))
	}
	result = LSDLossTrainingResult{RawSquare: float64(values[0][0]), Loss: float64(values[0][1]), DLogVariance: values[0][2], DPrediction: values[1]}
	if err := validateLSDLossBoundary(result, prediction, desired); err != nil {
		return LSDLossTrainingResult{}, b.fail(err)
	}
	return result, nil
}

func admitLSDLossTraining(prediction, desired []float32, logvar float32, maxBytes int64) error {
	c := len(prediction)
	if _, err := train.LSDLossShape(c, 0); err != nil {
		return err
	}
	if len(desired) != c || !finiteF32(prediction) || !finiteF32(desired) || !isFinite(logvar) || logvar < -80 || logvar > 80 || maxBytes <= 0 {
		return fmt.Errorf("invalid LSD loss inputs/logvar/ceiling")
	}
	if int64(4*(3*c+4)) > maxBytes {
		return fmt.Errorf("LSD loss tape byte ceiling")
	}
	// Conservative F64 admission, not an objective reference: headroom for raw
	// and exp-weighted F32 square reduction, plus doubled prediction seeds.
	exp := math.Exp(float64(logvar))
	limit := math.Sqrt((float64(math.MaxFloat32) / 4) / float64(c) / math.Max(1, exp))
	for i, p := range prediction {
		d := p - desired[i]
		if !isFinite(d) || math.Abs(float64(d)) > limit || 2*math.Abs(float64(d))*exp/float64(c) > float64(math.MaxFloat32)/4 {
			return fmt.Errorf("LSD loss reduction/seed overflow ceiling")
		}
	}
	return nil
}
