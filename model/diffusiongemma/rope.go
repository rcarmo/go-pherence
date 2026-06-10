package diffusiongemma

import simd "github.com/rcarmo/go-pherence/backends/simd/runtime"

type RoPEPlan struct {
	SlidingFreqs []float32 `json:"-"`
	FullFreqs    []float32 `json:"-"`
	SlidingHalf  int       `json:"sliding_half"`
	FullHalf     int       `json:"full_half"`
	MaxSeq       int       `json:"max_seq"`
}

func BuildRoPEPlan(shape Shape) RoPEPlan {
	maxSeq := shape.CanvasLength
	if maxSeq <= 0 {
		maxSeq = 256
	}
	// Match published config defaults: sliding_attention uses default RoPE
	// theta=10000 over the full head dim; full_attention uses proportional RoPE
	// theta=1e6 with partial_rotary_factor=0.25 over global head dim.
	slidingHalf := shape.TextHeadDim / 2
	fullHeadDim := shape.TextGlobalHeadDim
	if fullHeadDim <= 0 {
		fullHeadDim = shape.TextHeadDim
	}
	fullHalf := fullHeadDim / 8 // 0.25 * head_dim / 2
	return RoPEPlan{SlidingFreqs: simd.BuildRoPEFreqs(maxSeq, slidingHalf, shape.TextHeadDim, 10000), FullFreqs: simd.BuildRoPEFreqs(maxSeq, fullHalf, fullHeadDim, 1000000), SlidingHalf: slidingHalf, FullHalf: fullHalf, MaxSeq: maxSeq}
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
