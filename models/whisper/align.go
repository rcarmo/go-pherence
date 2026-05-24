package whisper

import "math"

// WordAlignment represents a word with its precise audio timing.
type WordAlignment struct {
	Word  string
	Start float64 // seconds
	End   float64 // seconds
	Token int     // token ID
}

// ForceAlign performs forced alignment: given known text tokens and audio,
// determine the precise timing of each token using the Whisper decoder's
// cross-attention weights as an alignment signal.
//
// The approach: run the decoder with the known tokens (teacher forcing),
// then use the cross-attention scores to find which audio frames each token
// attends to most strongly.
func ForceAlign(dec *Decoder, state *DecoderState, tokens []int, cfg Config, audioDuration float64) []WordAlignment {
	if len(tokens) == 0 || dec == nil || state == nil {
		return nil
	}

	dModel := cfg.DecoderDModel
	numHeads := cfg.DecoderHeads
	headDim := cfg.HeadDim
	encLen := len(state.CrossK[0]) / dModel

	// Feed prompt
	prompt := []int{TokenSOT, TokenEnglish, TokenTranscribe, TokenNoTimestamps}
	for _, tok := range prompt {
		dec.ForwardToken(tok, state)
	}

	// Teacher-force each token and collect cross-attention alignments
	alignments := make([]WordAlignment, len(tokens))
	prevTok := prompt[len(prompt)-1]

	for i, tok := range tokens {
		// Forward one token (teacher forcing)
		dec.ForwardToken(prevTok, state)

		// Compute cross-attention scores for alignment
		// The last decoder layer's cross-attention is most informative
		lastLayer := len(dec.Layers) - 1
		layer := &dec.Layers[lastLayer]

		// Get decoder hidden state for this position (approximate from KV cache)
		// Since we just ran ForwardToken, the cross-attention Q was computed internally.
		// For alignment, we re-compute the cross-attention scores.
		normed := layerNorm(
			state.SelfKCache[lastLayer][state.Pos*dModel-dModel:state.Pos*dModel],
			layer.CrossAttnLNWeight, layer.CrossAttnLNBias, 1, dModel)
		crossQ := linearForwardOpt(normed, layer.CrossQWeight, layer.CrossQBias, 1, dModel, dModel)

		// Compute attention scores across encoder frames
		frameScores := make([]float32, encLen)
		scale := float32(1.0 / math.Sqrt(float64(headDim)))

		for h := 0; h < numHeads; h++ {
			hOff := h * headDim
			for f := 0; f < encLen; f++ {
				kOff := f*dModel + hOff
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += crossQ[hOff+d] * state.CrossK[lastLayer][kOff+d]
				}
				frameScores[f] += dot * scale
			}
		}

		// Average across heads
		for f := range frameScores {
			frameScores[f] /= float32(numHeads)
		}

		// Find peak frame (the frame this token aligns to most)
		peakFrame := argmaxF32(frameScores)
		frameTime := float64(peakFrame) * audioDuration / float64(encLen)

		alignments[i] = WordAlignment{
			Token: tok,
			Start: frameTime,
			End:   frameTime + audioDuration/float64(encLen), // approximate width
		}

		prevTok = tok
	}

	// Smooth alignments: enforce monotonicity and fill gaps
	smoothAlignments(alignments, audioDuration)

	return alignments
}

func argmaxF32(x []float32) int {
	if len(x) == 0 {
		return 0
	}
	best := 0
	for i, v := range x[1:] {
		if v > x[best] {
			best = i + 1
		}
	}
	return best
}

// smoothAlignments enforces monotonically increasing start times
// and fills gaps between consecutive tokens.
func smoothAlignments(aligns []WordAlignment, totalDuration float64) {
	if len(aligns) == 0 {
		return
	}

	// Enforce monotonicity
	for i := 1; i < len(aligns); i++ {
		if aligns[i].Start < aligns[i-1].Start {
			aligns[i].Start = aligns[i-1].Start
		}
	}

	// Set end times as start of next token (or total duration for last)
	for i := 0; i < len(aligns)-1; i++ {
		aligns[i].End = aligns[i+1].Start
	}
	aligns[len(aligns)-1].End = totalDuration
}
