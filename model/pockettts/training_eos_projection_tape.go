package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
)

type EOSProjectionTrainingResult struct {
	Loss   float64
	DInput []float32
}
type TrainingEOSProjectionBackend interface {
	EOSProjectionEnabled() bool
	EvaluateEOSProjection(*FlowLMTrainingCPU, []float32, []bool, float32) (EOSProjectionTrainingResult, error)
}

func (b *ResidentTrainingBackend) EOSProjectionEnabled() bool {
	return b != nil && (b.AudioProjectionEnabled() || b.Scope == "ptx-resident-eosfused-hybrid-v1" || b.Scope == "vulkan-resident-eosfused-hybrid-v1")
}

// BuildEOSProjectionTrainingTape combines the affine, unweighted BCE mean and
// weighted seeded VJP without a logit/seed transfer. Objective weight is applied
// once by EOS gradient mode1; the returned mean stays unweighted.
func (m *FlowLMTrainingCPU) BuildEOSProjectionTrainingTape(hidden []float32, mask []bool, weight float32, maxBytes int64) (ProjectionTrainingTape, error) {
	return m.buildEOSProjectionTrainingTape(hidden, mask, weight, maxBytes, false)
}
func (m *FlowLMTrainingCPU) buildEOSProjectionTrainingTape(hidden []float32, mask []bool, weight float32, maxBytes int64, borrowParameters bool) (ProjectionTrainingTape, error) {
	labels, active, err := m.admitEOSProjection(hidden, mask, weight, maxBytes)
	if err != nil {
		return ProjectionTrainingTape{}, err
	}
	rows := len(mask)
	p, err := buildProjectionTrainingTape(m.EOS, "flow_lm.out_eos.weight", "flow_lm.out_eos.bias", hidden, make([]float32, rows), rows, maxBytes, borrowParameters)
	if err != nil {
		return p, err
	}
	// The projection builder's dy is the first backward command's seed. Replace
	// that initial value with EOS mode1 plus reshape; preserve every VJP command.
	var forward train.Command
	var backward []train.Command
	seed := -1
	projectionOutput := p.Tape.Outputs[0]
	logits := len(p.Tape.Tensors)
	p.Tape.Tensors = append(p.Tape.Tensors, train.TensorSpec{Shape: []int{rows}})
	flat := train.Command{Op: "reshape", Inputs: []int{projectionOutput}, Output: logits}
	for _, c := range p.Tape.Commands {
		if c.Op == "linear-forward" {
			forward = c
		} else {
			if c.Op == "linear-backward" {
				seed = c.Inputs[3]
			}
			backward = append(backward, c)
		}
	}
	if seed < 0 {
		return ProjectionTrainingTape{}, fmt.Errorf("fused EOS projection seed")
	}
	p.Tape.Tensors[seed].Values = nil
	labelID := len(p.Tape.Tensors)
	lossID, gradientID, meanID := labelID+1, labelID+2, labelID+3
	// Charge complete source, including the replaced seed tensor, before adding
	// constant copies. Public and private admission use the same logical ceiling.
	var bytes int64
	for _, s := range p.Tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	if bytes+int64(3*rows+1)*4 > maxBytes {
		return ProjectionTrainingTape{}, fmt.Errorf("fused EOS tape byte ceiling")
	}
	p.Tape.Tensors = append(p.Tape.Tensors, train.TensorSpec{Shape: []int{rows}, Values: labels}, train.TensorSpec{Shape: []int{rows}}, train.TensorSpec{Shape: []int{rows}}, train.TensorSpec{Shape: []int{1}})
	p.Tape.Commands = []train.Command{forward, flat, {Op: "eos", Inputs: []int{logits, labelID}, Output: lossID, Rows: active, Mode: 0, Scale: weight}, {Op: "eos", Inputs: []int{logits, labelID}, Output: gradientID, Rows: active, Mode: 1, Scale: weight}, {Op: "reshape", Inputs: []int{gradientID}, Output: seed}, {Op: "eos", Inputs: []int{lossID, labelID}, Output: meanID, Rows: active, Mode: 2, Scale: weight}}
	p.Tape.Commands = append(p.Tape.Commands, backward...)
	p.Tape.Outputs[0] = meanID
	if err = p.Tape.Validate(maxBytes); err != nil {
		return ProjectionTrainingTape{}, err
	}
	return p, nil
}
func validateEOSProjectionBoundary(r EOSProjectionTrainingResult, rows, hidden int, inputs ...[]float32) error {
	if !finite64(r.Loss) || r.Loss < 0 || len(r.DInput) != rows*hidden || !finiteF32(r.DInput) {
		return fmt.Errorf("fused EOS result shape/nonfinite")
	}
	for _, x := range inputs {
		if f32SlicesOverlap(r.DInput, x) {
			return fmt.Errorf("fused EOS borrowed input")
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) EvaluateEOSProjection(m *FlowLMTrainingCPU, hidden []float32, mask []bool, weight float32) (EOSProjectionTrainingResult, error) {
	var r EOSProjectionTrainingResult
	if err := b.admitGraph(); err != nil {
		return r, err
	}
	if !b.EOSProjectionEnabled() || b.trainer.FlowLM != m || m.EOSBackend != b || m.ProjectionBackend != b {
		return r, b.fail(fmt.Errorf("fused EOS owner/scope mismatch"))
	}
	p, err := m.buildEOSProjectionTrainingTape(hidden, mask, weight, b.MaxTapeBytes, true)
	if err != nil {
		return r, b.fail(err)
	}
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, p.Tape.Outputs[:2], true, 1)
	if err != nil {
		return r, b.fail(err)
	}
	r = EOSProjectionTrainingResult{Loss: float64(out[0][0]), DInput: out[1]}
	if err = validateEOSProjectionBoundary(r, len(mask), m.Hidden, hidden); err != nil {
		return EOSProjectionTrainingResult{}, b.fail(err)
	}
	return r, nil
}

func (m *FlowLMTrainingCPU) admitEOSProjection(hidden []float32, mask []bool, weight float32, maxBytes int64) ([]float32, int, error) {
	rows := len(mask)
	if m == nil || rows < 1 || rows > 1024 || m.Hidden < 1 || m.EOS.In != m.Hidden || m.EOS.Out != 1 || len(hidden) != rows*m.Hidden || !finiteF32(hidden) || !isFinite(weight) || weight < 0 || maxBytes <= 0 {
		return nil, 0, fmt.Errorf("fused EOS input/shape/ceiling")
	}
	if err := validateOwnedTrainingLinear(m.EOS, true); err != nil {
		return nil, 0, err
	}
	labels, active, err := eosTrainingLabels(mask)
	if err != nil {
		return nil, 0, err
	}
	// Independent absolute-dot bound admits all rows, including padding, before
	// dispatch. This is conservative F32 overflow headroom, not native accuracy.
	ceiling := float64(math.MaxFloat32) * .125 / float64(rows)
	for row := 0; row < rows; row++ {
		bound := math.Abs(float64(m.EOS.Bias[0]))
		for k, w := range m.EOS.Weight {
			bound += math.Abs(float64(w) * float64(hidden[row*m.Hidden+k]))
		}
		if bound > ceiling {
			return nil, 0, fmt.Errorf("fused EOS projection overflow ceiling")
		}
	}
	return labels, active, nil
}
