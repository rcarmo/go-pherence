package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/backends/mlx"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func TestMoeForwardMLXExpertsExactParity(t *testing.T) {
	const (
		hidden     = 128
		inter      = 256
		numExperts = 6
		active     = 2
	)
	layer := &LlamaLayer{
		RouterW:     benchMLXWeight(hidden, numExperts, 64),
		ExpertGateW: make([]*mlx.QuantWeight, numExperts),
		ExpertUpW:   make([]*mlx.QuantWeight, numExperts),
		ExpertDownW: make([]*mlx.QuantWeight, numExperts),
	}
	for i := 0; i < numExperts; i++ {
		layer.ExpertGateW[i] = benchMLXWeight(hidden, inter, 64)
		layer.ExpertUpW[i] = benchMLXWeight(hidden, inter, 64)
		layer.ExpertDownW[i] = benchMLXWeight(inter, hidden, 64)
	}
	cfg := LlamaConfig{NumExperts: numExperts, NumExpertsPerTok: active, MoEIntermediate: inter, NormTopKProb: true}
	x0 := benchSeq(hidden)
	x1 := benchSeq(hidden)
	for i := range x1 {
		if i&1 == 1 {
			x1[i] = -x1[i]
		}
	}
	for _, x := range [][]float32{x0, x1} {
		want := moeForwardMLXReference(x, layer, cfg)
		got := moeForward(x, layer, cfg)
		assertExactFloat32Slices(t, "moe", got, want)
	}
}

func moeForwardMLXReference(x []float32, layer *LlamaLayer, cfg LlamaConfig) []float32 {
	if layer == nil || len(x) == 0 || cfg.NumExperts <= 0 || cfg.MoEIntermediate <= 0 {
		return nil
	}
	h := len(x)
	numExperts := cfg.NumExperts
	numActive := cfg.NumExpertsPerTok
	if numActive <= 0 {
		numActive = 8
	}
	if numActive > numExperts {
		numActive = numExperts
	}

	routerLogits := make([]float32, numExperts)
	if layer.RouterW != nil {
		if !mlx.GemvTo(routerLogits, x, layer.RouterW) {
			return nil
		}
	}
	if !simd.SoftmaxInPlace(routerLogits) {
		return nil
	}

	type expertScore struct {
		id    int
		score float32
	}
	selected := make([]expertScore, 0, numActive)
	for i := 0; i < numActive; i++ {
		bestID := -1
		bestScore := float32(-1)
		for j, s := range routerLogits {
			if s > bestScore {
				alreadyPicked := false
				for _, sel := range selected {
					if sel.id == j {
						alreadyPicked = true
						break
					}
				}
				if !alreadyPicked {
					bestID = j
					bestScore = s
				}
			}
		}
		if bestID >= 0 {
			selected = append(selected, expertScore{id: bestID, score: bestScore})
		}
	}
	if len(selected) == 0 {
		return make([]float32, h)
	}
	if cfg.NormTopKProb {
		var sum float32
		for _, s := range selected {
			sum += s.score
		}
		if sum > 0 {
			for i := range selected {
				selected[i].score /= sum
			}
		}
	}

	out := make([]float32, h)
	gate := make([]float32, cfg.MoEIntermediate)
	up := make([]float32, cfg.MoEIntermediate)
	down := make([]float32, h)
	for _, exp := range selected {
		eid := exp.id
		if eid < 0 || eid >= len(layer.ExpertGateW) || eid >= len(layer.ExpertUpW) || eid >= len(layer.ExpertDownW) || layer.ExpertGateW[eid] == nil || layer.ExpertUpW[eid] == nil || layer.ExpertDownW[eid] == nil {
			continue
		}
		if !mlx.GemvTo(gate, x, layer.ExpertGateW[eid]) || !mlx.GemvTo(up, x, layer.ExpertUpW[eid]) {
			continue
		}
		simd.VecSiLUMul(gate, gate, up)
		if !mlx.GemvTo(down, gate, layer.ExpertDownW[eid]) {
			continue
		}
		for i := range out {
			out[i] += exp.score * down[i]
		}
	}
	return out
}

func assertExactFloat32Slices(tb testing.TB, name string, got, want []float32) {
	tb.Helper()
	if len(got) != len(want) {
		tb.Fatalf("%s len=%d want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			tb.Fatalf("%s[%d]=%v want %v", name, i, got[i], want[i])
		}
	}
}
