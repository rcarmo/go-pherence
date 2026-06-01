package model

import "math"

func (m *GGUFLlama) ggufMoEForward(out, gate, up, mid, x []float32, layer *GGUFLlamaLayer, layerIdx int) {
	for i := range out {
		out[i] = 0
	}
	if m == nil || layer == nil || layer.RouterM == nil || layer.ExpertGateM == nil || layer.ExpertUpM == nil || layer.ExpertDownM == nil {
		return
	}
	cfg := m.Config
	if cfg.NumExperts <= 0 || cfg.NumExpertsPerTok <= 0 || cfg.MoEHiddenSize <= 0 || len(gate) < cfg.MoEHiddenSize || len(up) < cfg.MoEHiddenSize || len(mid) < cfg.MoEHiddenSize {
		return
	}
	router := make([]float32, cfg.NumExperts)
	m.gemvMaybe(router, x, layer.RouterW, layer.RouterM, cfg.HiddenSize, cfg.NumExperts)
	softmaxInplace(router)
	selected := ggufTopKExperts(router, cfg.NumExpertsPerTok, m.REAP, layerIdx)
	if len(selected) == 0 {
		return
	}
	var sum float32
	for _, e := range selected {
		sum += e.score
	}
	if sum > 0 {
		for i := range selected {
			selected[i].score /= sum
		}
	}
	moe := cfg.MoEHiddenSize
	for _, e := range selected {
		if e.id < 0 || e.id >= cfg.NumExperts {
			continue
		}
		if err := layer.ExpertGateM.GemvExpertTo(gate[:moe], x, e.id); err != nil {
			continue
		}
		if err := layer.ExpertUpM.GemvExpertTo(up[:moe], x, e.id); err != nil {
			continue
		}
		m.siluMul(mid[:moe], gate[:moe], up[:moe])
		down := make([]float32, cfg.HiddenSize)
		if err := layer.ExpertDownM.GemvExpertTo(down, mid[:moe], e.id); err != nil {
			continue
		}
		for i := range out {
			out[i] += e.score * down[i]
		}
	}
	m.ggufSharedExpertAdd(out, gate, up, mid, x, layer)
}

func (m *GGUFLlama) ggufSharedExpertAdd(out, gate, up, mid, x []float32, layer *GGUFLlamaLayer) {
	cfg := m.Config
	shared := cfg.SharedMoEHiddenSize
	if shared <= 0 {
		shared = cfg.MoEHiddenSize
	}
	if layer == nil || layer.SharedGateM == nil || layer.SharedUpM == nil || layer.SharedDownM == nil || shared <= 0 || len(gate) < shared || len(up) < shared || len(mid) < shared {
		return
	}
	m.gemvMaybe(gate[:shared], x, nil, layer.SharedGateM, cfg.HiddenSize, shared)
	m.gemvMaybe(up[:shared], x, nil, layer.SharedUpM, cfg.HiddenSize, shared)
	m.siluMul(mid[:shared], gate[:shared], up[:shared])
	sharedDown := make([]float32, cfg.HiddenSize)
	m.gemvMaybe(sharedDown, mid[:shared], nil, layer.SharedDownM, shared, cfg.HiddenSize)
	scale := ggufSharedExpertGate(x, layer.SharedGateInp)
	for i := range out {
		out[i] += scale * sharedDown[i]
	}
}

func ggufSharedExpertGate(x, gate []float32) float32 {
	if len(gate) == 0 || len(gate) != len(x) {
		return 1
	}
	var dot float32
	for i, v := range x {
		dot += v * gate[i]
	}
	return ggufQwenNextSigmoid(dot)
}

type ggufExpertScore struct {
	id    int
	score float32
}

func ggufTopKExperts(scores []float32, k int, reap *REAPConfig, layerIdx int) []ggufExpertScore {
	if k <= 0 || len(scores) == 0 {
		return nil
	}
	if k > len(scores) {
		k = len(scores)
	}
	out := make([]ggufExpertScore, 0, k)
	for len(out) < k {
		bestID := -1
		bestScore := float32(-math.MaxFloat32)
		for id, score := range scores {
			if !reap.Allows(layerIdx, id) {
				continue
			}
			seen := false
			for _, e := range out {
				if e.id == id {
					seen = true
					break
				}
			}
			if !seen && score > bestScore {
				bestID = id
				bestScore = score
			}
		}
		if bestID < 0 {
			break
		}
		out = append(out, ggufExpertScore{id: bestID, score: bestScore})
	}
	return out
}
