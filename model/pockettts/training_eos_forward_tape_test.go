package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestTrainingEOSForwardTapeReference(t *testing.T) {
	for _, frames := range []int{1, 2, 7} {
		for _, voice := range []int{0, 1} {
			for _, weight := range []float32{0, .1, 1, 2} {
				_, m := backboneOracle(t)
				batch := eosForwardBatch(frames, voice, []uint32{1, 1, 3})
				mask := make([]bool, frames)
				for i := range mask {
					mask[i] = i < frames-1 || frames == 1
				}
				rows := 1 + voice + len(batch.TextTokens) + frames
				seed := make([]float32, rows*m.Hidden)
				p, err := m.BuildBackboneTrainingTape(batch, seed, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				full := interpretFlowTape(t, p.Tape)
				hidden := full[1][(rows-frames)*m.Hidden:]
				e, err := m.BuildEOSProjectionTrainingTape(hidden, mask, weight, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				eos := interpretFlowTape(t, e.Tape)
				q, err := m.buildEOSForwardTrainingTape(batch, mask, weight, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				all := interpretFlowTape(t, q.Tape)
				assertSliceClose(t, "forward sequence", all[0], full[0], 2e-5)
				assertSliceClose(t, "forward hidden", all[1], full[1], 2e-5)
				assertSliceClose(t, "unweighted mean", all[len(all)-1], eos[0], 2e-6)
				// EOS roots stay declared and share backward bindings; no endpoint gradients.
				for name, b := range p.ResidentBindings {
					if q.ResidentBindings[name].Parameter != b.Parameter {
						t.Fatal("changed parameter identity")
					}
				}
				var bytes int64
				for _, s := range q.Tape.Tensors {
					n, _ := s.Elements()
					bytes += int64(n) * 4
				}
				if _, err := m.buildEOSForwardTrainingTape(batch, mask, weight, bytes-1); err == nil {
					t.Fatal("complete graph ceiling")
				}
				if _, err := m.buildEOSForwardTrainingTape(batch, mask, weight, bytes); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
}
func TestTrainingEOSForwardAdmission(t *testing.T) {
	for _, kind := range []string{"topology", "mask", "weight", "parameter", "input", "budget", "transformer"} {
		_, m := backboneOracle(t)
		batch := eosForwardBatch(3, 1, []uint32{1, 2})
		mask := []bool{true, true, false}
		weight := float32(.1)
		budget := int64(64 << 20)
		switch kind {
		case "topology":
			batch.Frames = 0
		case "mask":
			mask = nil
		case "weight":
			weight = -1
		case "parameter":
			m.EOS.Weight[0] = float32(math.NaN())
		case "input":
			batch.NormalizedLatents[0] = float32(math.NaN())
		case "budget":
			budget = 1
		case "transformer":
			m.Transformer = nil
		}
		if _, err := m.buildEOSForwardTrainingTape(batch, mask, weight, budget); err == nil {
			t.Fatal(kind)
		}
	}
}
func TestTrainingEOSForwardSourceBudget(t *testing.T) {
	_, m := backboneOracle(t)
	batch := eosForwardBatch(3, 1, []uint32{1, 2})
	mask := []bool{true, true, false}
	rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
	p, err := m.buildEOSBackboneTrainingTape(batch, make([]float32, rows*m.Hidden), make([]float32, batch.Frames*m.Hidden), mask, .1, 64<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	for _, s := range p.Tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	if _, err := m.buildEOSForwardTrainingTape(batch, mask, .1, bytes); err == nil {
		t.Fatal("forward roots not charged")
	}
	// Ordinary admission rejects invalid dead constants before pruning.
	p.Tape.Tensors = append(p.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
	if train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 64<<20) == nil {
		t.Fatal("dead source")
	}
}

func eosForwardBatch(frames, voice int, tokens []uint32) FlowLMTrainingBatch {
	batch := FlowLMTrainingBatch{Frames: frames, VoiceFrames: voice, TextTokens: tokens, NormalizedLatents: make([]float32, frames*2), VoiceLatents: make([]float32, voice*2)}
	for i := range batch.NormalizedLatents {
		batch.NormalizedLatents[i] = float32(i+1) * .03
	}
	for i := range batch.VoiceLatents {
		batch.VoiceLatents[i] = float32(i+1) * .02
	}
	return batch
}

func TestTrainingSharedForwardOnlyReference(t *testing.T) {
	for _, frames := range []int{2, 7} {
		for _, voice := range []int{0, 1} {
			_, lm := backboneOracle(t)
			batch := eosForwardBatch(frames, voice, []uint32{1, 1, 3})
			for i := range batch.VoiceLatents {
				batch.VoiceLatents[i] = float32(i+1) * .04
			}
			for i := range batch.NormalizedLatents {
				batch.NormalizedLatents[i] = float32(i+1) * .03
			}
			mask := make([]bool, frames)
			for i := range mask {
				mask[i] = i < frames-1
			}
			original, err := lm.buildEOSForwardTrainingTape(batch, mask, .1, 1<<24)
			if err != nil {
				t.Fatal(err)
			}
			lm.EOSBackend = &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1"}
			forward, err := lm.buildEOSForwardTrainingTape(batch, mask, .1, 1<<24)
			if err != nil {
				t.Fatal(err)
			}
			want := interpretFlowTape(t, original.Tape)
			got := interpretFlowTape(t, forward.Tape)
			assertSliceClose(t, "forward-only sequence", got[0], want[0], 6e-8)
			assertSliceClose(t, "forward-only transformed", got[1], want[1], 6e-8)
			assertSliceClose(t, "forward-only EOS", got[len(got)-1], want[len(want)-1], 6e-8)
			if len(forward.ResidentBindings) != len(original.ResidentBindings) {
				t.Fatal("complete forward inventory")
			}
			for name, b := range forward.ResidentBindings {
				if b.Parameter < 0 || b.Gradient < 0 || len(forward.Tape.Tensors[b.Gradient].Values) == 0 {
					t.Fatal("forward placeholder", name)
				}
			}
			if len(forward.Tape.Commands) >= len(original.Tape.Commands) {
				t.Fatal("reverse graph retained")
			}
			built, err := buildForwardResultsTape(forward, lm, batch, 1<<24)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = appendForwardEOSAdmission(built, lm, batch, 1<<24); err != nil {
				t.Fatal(err)
			}
		}
	}
}
