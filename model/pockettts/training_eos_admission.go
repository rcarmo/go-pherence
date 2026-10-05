package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

func (b *ResidentTrainingBackend) eosAdmissionEnabled() bool {
	return b != nil && (b.deviceHandoffEnabled() || b.predictionHandoffEnabled() || b.Scope == "ptx-resident-eosadmission-hybrid-v1" || b.Scope == "vulkan-resident-eosadmission-hybrid-v1")
}

// appendForwardEOSAdmission admits the complete selected-forward source and
// original EOS parameter inventory before appending a conservative scalar flag.
// Original output IDs, arithmetic and host overflow guards stay unchanged.
func appendForwardEOSAdmission(p BackboneTrainingTape, m *FlowLMTrainingCPU, batch FlowLMTrainingBatch, maxBytes int64) (BackboneTrainingTape, error) {
	if err := train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, maxBytes); err != nil {
		return BackboneTrainingTape{}, err
	}
	if m == nil || m.EOS.In != m.Hidden || m.EOS.Out != 1 || len(p.Tape.Outputs) < 5 {
		return BackboneTrainingTape{}, fmt.Errorf("EOS admission producer topology")
	}
	if _, _, err := train.EOSAdmissionShape(batch.Frames, m.Hidden); err != nil {
		return BackboneTrainingTape{}, err
	}
	outputs := forwardResultsOutputs(p)
	hidden := outputs[0]
	n, _ := p.Tape.Tensors[hidden].Elements()
	if n != batch.Frames*m.Hidden {
		return BackboneTrainingTape{}, fmt.Errorf("EOS admission hidden extent")
	}
	w, ok := p.ResidentBindings["flow_lm.out_eos.weight"]
	if !ok {
		return BackboneTrainingTape{}, fmt.Errorf("EOS admission weight inventory")
	}
	bias, ok := p.ResidentBindings["flow_lm.out_eos.bias"]
	if !ok {
		return BackboneTrainingTape{}, fmt.Errorf("EOS admission bias inventory")
	}
	wn, _ := p.Tape.Tensors[w.Parameter].Elements()
	bn, _ := p.Tape.Tensors[bias.Parameter].Elements()
	if wn != m.Hidden || bn != 1 || w.Parameter == bias.Parameter {
		return BackboneTrainingTape{}, fmt.Errorf("EOS admission parameter extent/alias")
	}
	// Require the bound weights to belong to the admitted EOS affine, not just
	// a similarly shaped parameter. The source builder supplies this exact path.
	producers := map[int]train.Command{}
	for _, c := range p.Tape.Commands {
		producers[c.Output] = c
	}
	selected, ok := producers[hidden]
	if !ok || selected.Op != "vector" || selected.Mode != 1 || len(selected.Inputs) != 2 {
		return BackboneTrainingTape{}, fmt.Errorf("EOS admission selected hidden provenance")
	}
	found := false
	for _, c := range p.Tape.Commands {
		if c.Op == "linear-forward" && c.Mode == 0 && len(c.Inputs) == 4 && c.Inputs[2] == w.Parameter && c.Inputs[3] == bias.Parameter {
			input := c.Inputs[0]
			for {
				alias, ok := producers[input]
				if !ok || alias.Op != "reshape" {
					break
				}
				input = alias.Inputs[0]
			}
			slice, ok := producers[input]
			if ok && slice.Op == "vector" && slice.Mode == 1 && len(slice.Inputs) == 2 {
				left, right := slice.Inputs[0], selected.Inputs[0]
				for {
					alias, ok := producers[left]
					if !ok || alias.Op != "reshape" {
						break
					}
					left = alias.Inputs[0]
				}
				for {
					alias, ok := producers[right]
					if !ok || alias.Op != "reshape" {
						break
					}
					right = alias.Inputs[0]
				}
				if left == right && slice.Offset == selected.Offset {
					found = true
				}
			}
		}
	}
	if !found {
		return BackboneTrainingTape{}, fmt.Errorf("EOS admission missing affine provenance")
	}
	t := p.Tape
	t.Tensors = append([]train.TensorSpec(nil), t.Tensors...)
	t.Commands = append([]train.Command(nil), t.Commands...)
	t.Outputs = append([]int(nil), t.Outputs...)
	flat, flag := len(t.Tensors), len(t.Tensors)+1
	t.Tensors = append(t.Tensors, train.TensorSpec{Shape: []int{m.Hidden}}, train.TensorSpec{Shape: []int{1}})
	t.Commands = append(t.Commands, train.Command{Op: "reshape", Inputs: []int{w.Parameter}, Output: flat}, train.Command{Op: "eos-admission", Inputs: []int{hidden, flat, bias.Parameter}, Output: flag, Rows: batch.Frames, Width: m.Hidden})
	t.Outputs = append(t.Outputs, flag)
	if err := train.ValidateResidentTapeBindings(t, p.ResidentBindings, maxBytes); err != nil {
		return BackboneTrainingTape{}, err
	}
	p.Tape = t
	return p, nil
}
func validateForwardEOSAdmission(values [][]float32, batch FlowLMTrainingBatch) error {
	if len(values) != 4 || len(values[3]) != 1 || !isFinite(values[3][0]) || values[3][0] != 0 {
		return fmt.Errorf("EOS projection admission failed")
	}
	for _, v := range values[:3] {
		if f32SlicesOverlap(values[3], v) {
			return fmt.Errorf("EOS admission aliased flag")
		}
	}
	for _, v := range [][]float32{batch.NormalizedLatents, batch.VoiceLatents} {
		if f32SlicesOverlap(values[3], v) {
			return fmt.Errorf("EOS admission borrowed flag")
		}
	}
	return nil
}
