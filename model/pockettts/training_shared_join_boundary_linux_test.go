//go:build linux

package pockettts

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// Checks composition without executing any AD operations. Exact comparisons
// here enforce immutable graph metadata/routing, never generated-signal parity.
func sharedCheckBackboneJoin(step StepWeightTrainingTape, backbone BackboneTrainingTape, joined StepBackboneTrainingTape, rows, hidden, prefix, frames, latent int) error {
	commands := joined.Tape.Commands
	if len(commands) != len(step.Tape.Commands)+len(backbone.Tape.Commands)+3 {
		return fmt.Errorf("join command inventory")
	}
	for i, c := range step.Tape.Commands {
		if !reflect.DeepEqual(c, commands[i]) {
			return fmt.Errorf("step prefix command %d", i)
		}
	}
	pad := commands[len(step.Tape.Commands)]
	if pad.Op != "vector" || pad.Mode != 2 || pad.Offset != prefix*hidden || len(pad.Inputs) != 2 || pad.Inputs[0] != step.Tape.Outputs[0] {
		return fmt.Errorf("condition seed pad")
	}
	if len(joined.Tape.Tensors[pad.Inputs[1]].Values) != 1 || joined.Tape.Tensors[pad.Inputs[1]].Values[0] != 0 || !equalShape(joined.Tape.Tensors[pad.Output].Shape, []int{rows * hidden}) {
		return fmt.Errorf("condition seed zero/extent")
	}
	reshape := commands[len(step.Tape.Commands)+1]
	if reshape.Op != "reshape" || !reflect.DeepEqual(reshape.Inputs, []int{pad.Output}) || !equalShape(joined.Tape.Tensors[reshape.Output].Shape, []int{rows, hidden}) {
		return fmt.Errorf("seed row reshape")
	}
	// Composition appends the complete admitted backbone, retaining tensor IDs
	// except the sole seed input. Infer offset from the unchanged output root.
	offset := joined.ResidentBindings["flow_lm.out_eos.weight"].Parameter - backbone.ResidentBindings["flow_lm.out_eos.weight"].Parameter
	if offset <= 0 {
		return fmt.Errorf("backbone source offset")
	}
	gradientViews := map[int]int{}
	for name, binding := range backbone.ResidentBindings {
		gradientViews[binding.Gradient] = joined.ResidentBindings[name].Gradient
	}
	mapping := func(id int) int {
		if id == backbone.seedID {
			return reshape.Output
		}
		if view, ok := gradientViews[id]; ok {
			return view
		}
		return id + offset
	}
	consumers := 0
	for i, c := range backbone.Tape.Commands {
		expected := c
		expected.Inputs = append([]int(nil), c.Inputs...)
		for j, id := range expected.Inputs {
			if id == backbone.seedID {
				consumers++
			}
			expected.Inputs[j] = mapping(id)
		}
		expected.Output = mapping(c.Output)
		if !reflect.DeepEqual(expected, commands[len(step.Tape.Commands)+2+i]) {
			// The standalone builder appends g.flat gradient views while ranging
			// over a map. Only this terminal reshape suffix may permute; no
			// arithmetic or seed consumer may move or change.
			if c.Op != "reshape" {
				return fmt.Errorf("backbone routing command %d", i)
			}
			for _, tail := range backbone.Tape.Commands[i:] {
				if tail.Op != "reshape" {
					return fmt.Errorf("arithmetic inside view suffix at %d", i)
				}
			}
			matches := 0
			for _, candidate := range commands[len(step.Tape.Commands)+2+i : len(commands)-1] {
				if reflect.DeepEqual(expected, candidate) {
					matches++
				}
			}
			// A matching view can precede this position within the same suffix.
			if matches == 0 {
				for _, candidate := range commands[len(step.Tape.Commands)+2 : len(commands)-1] {
					if reflect.DeepEqual(expected, candidate) {
						matches++
					}
				}
			}
			if matches != 1 {
				return fmt.Errorf("named gradient view %d", i)
			}
		}
	}
	if consumers != 1 {
		return fmt.Errorf("condition seed consumers=%d", consumers)
	}
	for name, b := range backbone.ResidentBindings {
		want := train.ResidentTapeBinding{Parameter: mapping(b.Parameter), Gradient: mapping(b.Gradient)}
		if joined.ResidentBindings[name] != want {
			return fmt.Errorf("backbone binding %s", name)
		}
	}
	for name, b := range step.ResidentBindings {
		if joined.ResidentBindings[name] != b {
			return fmt.Errorf("weight binding %s", name)
		}
	}
	if len(joined.ResidentBindings) != len(step.ResidentBindings)+len(backbone.ResidentBindings) {
		return fmt.Errorf("joined binding count")
	}
	last := commands[len(commands)-1]
	index := 2
	// Standalone roots precede exactly one output per parameter binding.
	boundaryCount := len(backbone.Tape.Outputs) - len(backbone.ResidentBindings)
	if boundaryCount == 4 {
		index++
	} else if boundaryCount != 3 {
		return fmt.Errorf("backbone boundary count")
	}
	if last.Op != "vector" || last.Mode != 0 || last.Scale != 1 || !reflect.DeepEqual(last.Inputs, []int{mapping(backbone.Tape.Outputs[index]), step.Tape.Outputs[1]}) || !equalShape(joined.Tape.Tensors[last.Output].Shape, []int{frames * latent}) {
		return fmt.Errorf("direct target final join")
	}
	return nil
}

func sharedJoinInputs(lm *FlowLMTrainingCPU, batch FlowLMTrainingBatch, valid int) (StepReductionInputs, WeightSampleInputs) {
	mask := make([]bool, batch.Frames)
	for i := 0; i < valid; i++ {
		mask[i] = true
	}
	h, n := lm.Hidden, lm.LatentDim
	in := StepReductionInputs{Frames: batch.Frames, Hidden: h, Latent: n, Mask: mask, DiagonalCondition: make([]float32, batch.Frames*h), DistillCondition: make([]float32, batch.Frames*h), DiagonalTarget: make([]float32, batch.Frames*n), DistillTarget: make([]float32, batch.Frames*n), DiagonalLogvar: make([]float32, batch.Frames), DistillLogvar: make([]float32, batch.Frames), RowMetrics: make([]float32, batch.Frames*4), EOS: .1}
	for row := 0; row < valid; row++ {
		for i := 0; i < h; i++ {
			in.DiagonalCondition[row*h+i] = float32((row+1)*3+i%5) * .0001
			in.DistillCondition[row*h+i] = float32((row+1)*7+i%3) * .0002
		}
		in.DiagonalLogvar[row] = .03
		in.DistillLogvar[row] = -.04
		in.RowMetrics[4*row] = .2
		in.RowMetrics[4*row+1] = .3
		in.RowMetrics[4*row+2] = .4
		in.RowMetrics[4*row+3] = .5
	}
	s := WeightSampleInputs{DiagonalTime: make([]float32, batch.Frames), DistillS: make([]float32, batch.Frames), DistillT: make([]float32, batch.Frames), DiagonalSeed: in.DiagonalLogvar, DistillSeed: in.DistillLogvar}
	for i := range s.DiagonalTime {
		s.DiagonalTime[i] = .2
		s.DistillS[i] = .3
		s.DistillT[i] = .8
	}
	return in, s
}

func sharedJoinFixture(t *testing.T, lm *FlowLMTrainingCPU, w *LSDWeightMLP, batch FlowLMTrainingBatch, valid int) (StepWeightTrainingTape, BackboneTrainingTape, StepBackboneTrainingTape, int, int) {
	t.Helper()
	const ceiling = int64(2 << 30)
	in, s := sharedJoinInputs(lm, batch, valid)
	rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
	prefix := rows - batch.Frames
	step, err := w.buildStepWeightTrainingTape(in, DefaultTrainingStepConfig(), s, ceiling, true)
	if err != nil {
		t.Fatal(err)
	}
	backbone, err := lm.buildEOSBackboneTrainingTape(batch, make([]float32, rows*lm.Hidden), make([]float32, batch.Frames*lm.Hidden), in.Mask, .1, ceiling, true)
	if err != nil {
		t.Fatal(err)
	}
	joined, err := lm.buildStepBackboneTrainingTape(w, batch, make([]float32, batch.Frames*lm.Hidden), in, DefaultTrainingStepConfig(), s, ceiling, true)
	if err != nil {
		t.Fatal(err)
	}
	return step, backbone, joined, rows, prefix
}

func TestPocketSharedBackboneJoinBoundary(t *testing.T) {
	r := requireRunner(t)
	batch := r.rows[0].Batch
	step, backbone, joined, rows, prefix := sharedJoinFixture(t, r.trainer.FlowLM, r.trainer.Weighting, batch, r.rows[0].ValidFrames)
	if err := sharedCheckBackboneJoin(step, backbone, joined, rows, r.trainer.FlowLM.Hidden, prefix, batch.Frames, r.trainer.FlowLM.LatentDim); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"seed-source", "seed-offset", "backbone-source", "binding", "target-scale"} {
		t.Run(kind, func(t *testing.T) {
			bad := joined
			bad.Tape.Commands = append([]train.Command(nil), joined.Tape.Commands...)
			bad.ResidentBindings = map[string]train.ResidentTapeBinding{}
			for k, v := range joined.ResidentBindings {
				bad.ResidentBindings[k] = v
			}
			switch kind {
			case "seed-source":
				c := bad.Tape.Commands[len(step.Tape.Commands)]
				c.Inputs = append([]int(nil), c.Inputs...)
				c.Inputs[0] = step.Tape.Outputs[1]
				bad.Tape.Commands[len(step.Tape.Commands)] = c
			case "seed-offset":
				bad.Tape.Commands[len(step.Tape.Commands)].Offset++
			case "backbone-source":
				bad.Tape.Commands[len(step.Tape.Commands)+2].Scale++
			case "binding":
				b := bad.ResidentBindings["flow_lm.out_eos.weight"]
				b.Gradient++
				bad.ResidentBindings["flow_lm.out_eos.weight"] = b
			case "target-scale":
				bad.Tape.Commands[len(bad.Tape.Commands)-1].Scale = 2
			}
			if err := sharedCheckBackboneJoin(step, backbone, bad, rows, r.trainer.FlowLM.Hidden, prefix, batch.Frames, r.trainer.FlowLM.LatentDim); err == nil {
				t.Fatal("misrouted join accepted")
			}
		})
	}
}

func TestPocketSharedReleasedBackboneJoinBoundary(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_JOIN_BOUNDARY") != "1" {
		t.Skip("set GO_PHERENCE_POCKET_SHARED_JOIN_BOUNDARY=1 with pinned model and corpus")
	}
	lm, _, w, _ := releasedTrainingModels(t)
	raw, err := os.ReadFile(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"))
	if err != nil || len(raw) > 1<<20 {
		t.Fatal("corpus", err)
	}
	digest := sha256.Sum256(raw)
	if hex.EncodeToString(digest[:]) != "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9" {
		t.Fatal("corpus pin")
	}
	var corpus struct {
		Rows []PreparedTrainingRow `json:"rows"`
	}
	if err := json.Unmarshal(raw, &corpus); err != nil || len(corpus.Rows) != 1 {
		t.Fatal(err)
	}
	batch := corpus.Rows[0].Batch
	step, backbone, joined, rows, prefix := sharedJoinFixture(t, lm, w, batch, corpus.Rows[0].ValidFrames)
	if err := sharedCheckBackboneJoin(step, backbone, joined, rows, lm.Hidden, prefix, batch.Frames, lm.LatentDim); err != nil {
		t.Fatal(err)
	}
	params := map[string][]float32{}
	addFlowLMParameters(params, lm)
	for i, l := range w.Layers {
		params[fmt.Sprintf("flow.w_s_t.%d.weight", 2*i)] = l.Weight
		params[fmt.Sprintf("flow.w_s_t.%d.bias", 2*i)] = l.Bias
	}
	if len(params) != len(joined.ResidentBindings) {
		t.Fatal("complete later inventory")
	}
	var values int64
	for name, b := range joined.ResidentBindings {
		source, ok := params[name]
		actual := joined.Tape.Tensors[b.Parameter].Values
		if !ok || len(source) != len(actual) || &source[0] != &actual[0] {
			t.Fatal("borrowed parameter association", name)
		}
		values += int64(len(source))
	}
	stats := map[string]int{}
	for _, c := range joined.Tape.Commands {
		stats[c.Op]++
	}
	t.Logf("released metadata join rows=%d prefix=%d commands=%d parameters=%d values=%d ops=%v; no execution, dummy seeds independent of saved outputs", rows, prefix, len(joined.Tape.Commands), len(params), values, stats)
}
