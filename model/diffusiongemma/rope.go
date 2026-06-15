package diffusiongemma

import simd "github.com/rcarmo/go-pherence/backends/simd/runtime"

const diffusionGemmaSuppressedRoPEFactor = float32(1e30)

type RoPEPlan struct {
	SlidingFreqs []float32 `json:"-"`
	FullFreqs    []float32 `json:"-"`
	SlidingHalf  int       `json:"sliding_half"`
	FullHalf     int       `json:"full_half"`
	MaxSeq       int       `json:"max_seq"`
}

func synthesizedFullAttentionRoPEFactors(headDim int) []float32 {
	// DiffusionGemma/Gemma4 full-attention uses proportional RoPE. In GGUF this
	// appears as rope_freqs.weight: first partial_rotary_factor=0.25 of the head
	// dimensions have factor 1, and the rest are effectively disabled with 1e30.
	// Some safetensor FP8 checkpoints omit that tensor but still declare
	// rope_type=proportional in config.json, so synthesize llama.cpp's factors.
	half := headDim / 2
	if half <= 0 {
		return nil
	}
	activePairs := headDim / 8 // (partial_rotary_factor=0.25 * headDim) / 2
	if activePairs < 1 {
		activePairs = 1
	}
	if activePairs > half {
		activePairs = half
	}
	factors := make([]float32, half)
	for i := range factors {
		if i < activePairs {
			factors[i] = 1
		} else {
			factors[i] = diffusionGemmaSuppressedRoPEFactor
		}
	}
	return factors
}

func fullAttentionRoPEFactors(weights *TextWeights, fp TextForwardPlan, headDim int) ([]float32, error) {
	if fp.Globals.RopeFreqs != nil {
		return loadFloatVector(weights, fp.Globals.RopeFreqs)
	}
	return synthesizedFullAttentionRoPEFactors(headDim), nil
}

func BuildRoPEPlan(shape Shape) RoPEPlan {
	maxSeq := shape.CanvasLength
	if maxSeq <= 0 {
		maxSeq = 256
	}
	// Match llama.cpp/GGUF metadata: sliding_attention uses rope.dimension_count_swa
	// (= TextHeadDim) with theta=10000; full_attention uses rope.dimension_count
	// (= TextGlobalHeadDim) with theta=1e6 and optional rope_freqs factors at the
	// actual attention call sites.
	slidingHalf := shape.TextHeadDim / 2
	fullHeadDim := shape.TextGlobalHeadDim
	if fullHeadDim <= 0 {
		fullHeadDim = shape.TextHeadDim
	}
	fullHalf := fullHeadDim / 2
	return RoPEPlan{
		SlidingFreqs: simd.BuildRoPEFreqs(maxSeq, slidingHalf, shape.TextHeadDim, 10000),
		FullFreqs:    simd.BuildRoPEFreqsWithFactors(maxSeq, fullHalf, fullHeadDim, 1000000, synthesizedFullAttentionRoPEFactors(fullHeadDim)),
		SlidingHalf:  slidingHalf,
		FullHalf:     fullHalf,
		MaxSeq:       maxSeq,
	}
}

func (p RoPEPlan) Apply(layerType string, q []float32, qHeads, qHeadDim int, k []float32, kvHeads, kHeadDim int, pos int) bool {
	if layerType == "full_attention" {
		if len(p.FullFreqs) == 0 || p.FullHalf <= 0 {
			return false
		}
		simd.ApplyRoPEPartial(q, p.FullFreqs, pos, qHeads, qHeadDim, p.FullHalf)
		simd.ApplyRoPEPartial(k, p.FullFreqs, pos, kvHeads, kHeadDim, p.FullHalf)
		return true
	}
	if len(p.SlidingFreqs) == 0 || p.SlidingHalf <= 0 {
		return false
	}
	simd.ApplyRoPEPartial(q, p.SlidingFreqs, pos, qHeads, qHeadDim, p.SlidingHalf)
	simd.ApplyRoPEPartial(k, p.SlidingFreqs, pos, kvHeads, kHeadDim, p.SlidingHalf)
	return true
}
