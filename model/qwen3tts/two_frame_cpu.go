package qwen3tts

import (
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// TwoFrameCPUResult owns the output of a bounded, greedy CustomVoice continuation.
// The method is a reference boundary, not a streaming or general-length runtime.
type TwoFrameCPUResult struct {
	Semantic     []uint32
	Acoustic     []uint32 // 15 codes per frame, in frame order
	SecondHidden []float32
	SecondLogits []float32 // raw, before control-token suppression
	Waveform     []float32 // 3840 mono F32 samples at 24 kHz
}

// BoundedCPUResult owns a capped greedy reference output. Hidden and raw
// logits are retained for each continuation, never aliased to KV storage.
type BoundedCPUResult struct {
	Semantic           []uint32
	Acoustic           []uint32 // 15 codes per frame, in frame order
	ContinuationHidden [][]float32
	ContinuationLogits [][]float32
	Waveform           []float32
}

// GenerateTwoFramesCPU preserves the existing bounded two-frame API.
func GenerateTwoFramesCPU(plan RuntimeRequestPlan, talker *TalkerCPU, predictor *CodePredictorCPU, decoder *Decoder12HzCPU) (TwoFrameCPUResult, error) {
	if plan.MaxFrames != 2 {
		return TwoFrameCPUResult{}, fmt.Errorf("invalid two-frame Qwen3-TTS runtime")
	}
	result, err := generateBoundedFramesCPU(plan, talker, predictor, decoder)
	if err != nil {
		return TwoFrameCPUResult{}, err
	}
	if len(result.Semantic) != 2 {
		return TwoFrameCPUResult{}, fmt.Errorf("Qwen3-TTS two-frame reference stopped at EOS")
	}
	return TwoFrameCPUResult{Semantic: result.Semantic, Acoustic: result.Acoustic, SecondHidden: result.ContinuationHidden[0], SecondLogits: result.ContinuationLogits[0], Waveform: result.Waveform}, nil
}

// GenerateThreeFramesCPU runs one further greedy continuation with the same
// request-local Talker KV. It is a parity probe, not a general TTS sampler.
func GenerateThreeFramesCPU(plan RuntimeRequestPlan, talker *TalkerCPU, predictor *CodePredictorCPU, decoder *Decoder12HzCPU) (BoundedCPUResult, error) {
	if plan.MaxFrames != 3 {
		return BoundedCPUResult{}, fmt.Errorf("invalid three-frame Qwen3-TTS runtime")
	}
	result, err := generateBoundedFramesCPU(plan, talker, predictor, decoder)
	if err != nil {
		return BoundedCPUResult{}, err
	}
	if len(result.Semantic) != 3 {
		return BoundedCPUResult{}, fmt.Errorf("Qwen3-TTS three-frame reference stopped at EOS")
	}
	return result, nil
}

// GenerateFourFramesCPU checks a third greedy continuation and the PAD text
// boundary after TTS EOS. It remains a fixed-size parity probe.
func GenerateFourFramesCPU(plan RuntimeRequestPlan, talker *TalkerCPU, predictor *CodePredictorCPU, decoder *Decoder12HzCPU) (BoundedCPUResult, error) {
	if plan.MaxFrames != 4 {
		return BoundedCPUResult{}, fmt.Errorf("invalid four-frame Qwen3-TTS runtime")
	}
	result, err := generateBoundedFramesCPU(plan, talker, predictor, decoder)
	if err != nil {
		return BoundedCPUResult{}, err
	}
	if len(result.Semantic) != 4 {
		return BoundedCPUResult{}, fmt.Errorf("Qwen3-TTS four-frame reference stopped at EOS")
	}
	return result, nil
}

// GenerateCappedGreedyCPU runs at most 32 CustomVoice frames. Greedy EOS
// stops before an acoustic frame is generated for that token. This reference
// path does not implement stochastic sampling or streaming.
func GenerateCappedGreedyCPU(plan RuntimeRequestPlan, talker *TalkerCPU, predictor *CodePredictorCPU, decoder *Decoder12HzCPU) (BoundedCPUResult, error) {
	return generateCappedGreedyCPU(plan, talker, predictor, decoder, greedyTalkerToken)
}

// GenerateCappedGreedyMinTwoCPU follows the pinned Rust library's minimum
// two semantic tokens before EOS is eligible. The separate standalone Rust
// generate_audio CLI does not apply that minimum. Both paths remain greedy.
func GenerateCappedGreedyMinTwoCPU(plan RuntimeRequestPlan, talker *TalkerCPU, predictor *CodePredictorCPU, decoder *Decoder12HzCPU) (BoundedCPUResult, error) {
	selected := 0
	choose := func(logits []float32, eos uint32) (uint32, error) {
		token, err := greedyTalkerTokenWithEOS(logits, eos, selected >= 2)
		if err == nil {
			selected++
		}
		return token, err
	}
	return generateCappedGreedyCPU(plan, talker, predictor, decoder, choose)
}

// GenerateCappedSeededCPU is an opt-in, bounded CPU probe for the pinned Rust
// sampler defaults (temperature .7, top-k 50, top-p .9, penalty 1). It uses
// request-local PCG state and withholds EOS until two semantic tokens have
// been selected. It does not change the fixed-frame or greedy APIs.
func GenerateCappedSeededCPU(plan RuntimeRequestPlan, talker *TalkerCPU, predictor *CodePredictorCPU, decoder *Decoder12HzCPU, seed uint64) (BoundedCPUResult, error) {
	sampler := NewReferenceCPUSampler(seed)
	cfg := ReferenceSampleConfig{Temperature: 0.7, TopK: 50, TopP: 0.9, RepetitionPenalty: 1}
	selected := 0
	choose := func(logits []float32, eos uint32) (uint32, error) {
		token, err := sampler.Select(logits, cfg, nil, true, selected >= 2)
		if err == nil {
			selected++
		}
		return token, err
	}
	return generateCappedGreedyCPU(plan, talker, predictor, decoder, choose)
}

// The first prefix is recomputed once to retain owned per-layer KV. Both the
// prefix and continuation caches are local to this call.
func generateBoundedFramesCPU(plan RuntimeRequestPlan, talker *TalkerCPU, predictor *CodePredictorCPU, decoder *Decoder12HzCPU) (BoundedCPUResult, error) {
	if plan.MaxFrames < 2 || plan.MaxFrames > 4 {
		return BoundedCPUResult{}, fmt.Errorf("invalid fixed-frame Qwen3-TTS runtime")
	}
	return generateCappedGreedyCPU(plan, talker, predictor, decoder, greedyTalkerToken)
}

func generateCappedGreedyCPU(plan RuntimeRequestPlan, talker *TalkerCPU, predictor *CodePredictorCPU, decoder *Decoder12HzCPU, selectToken func([]float32, uint32) (uint32, error)) (BoundedCPUResult, error) {
	if talker == nil || predictor == nil || decoder == nil || selectToken == nil || plan.MaxFrames < 1 || plan.MaxFrames > 32 || talker.cfg.ModelType != CustomVoice || predictor.cfg.CPHiddenSize != talker.cfg.TalkerHiddenSize {
		return BoundedCPUResult{}, fmt.Errorf("invalid bounded Qwen3-TTS runtime")
	}
	first, err := talker.Prefill(plan) // validates the entire prefix and control tokens
	if err != nil {
		return BoundedCPUResult{}, err
	}
	firstToken, err := selectToken(first.Logits, CodecEOS)
	if err != nil {
		return BoundedCPUResult{}, err
	}
	if firstToken == CodecEOS {
		return BoundedCPUResult{}, fmt.Errorf("Qwen3-TTS first semantic is EOS; no frame")
	}
	acousticWork := newCodePredictorWorkspace(predictor.cfg, len(predictor.layers))
	codes0, _, err := predictor.firstAcousticFrameWithWorkspace(talker, first.Hidden, firstToken, acousticWork)
	if err != nil {
		return BoundedCPUResult{}, err
	}
	const prefixLen = CustomVoiceFirstTextIndex + 1
	h := talker.cfg.TalkerHiddenSize
	inputs := make([]float32, prefixLen*h)
	for pos := 0; pos < prefixLen; pos++ {
		row, err := talker.projectText(plan.Prompt.Text[pos])
		if err != nil {
			return BoundedCPUResult{}, err
		}
		copy(inputs[pos*h:(pos+1)*h], row)
	}
	for i, id := range plan.Prompt.Codec[:6] {
		if err := addTalkerEmbedding(inputs[(i+3)*h:(i+4)*h], talker.codecEmbedding, id, talker.cfg.TalkerVocabSize, h); err != nil {
			return BoundedCPUResult{}, err
		}
	}
	if err := addTalkerEmbedding(inputs[9*h:10*h], talker.codecEmbedding, plan.Prompt.Codec[6], talker.cfg.TalkerVocabSize, h); err != nil {
		return BoundedCPUResult{}, err
	}
	kvWidth := talker.cfg.TalkerNumKeyValueHeads * talker.cfg.TalkerHeadDim
	keys, values := make([][]float32, len(talker.layers)), make([][]float32, len(talker.layers))
	for i := range keys {
		keys[i], values[i] = make([]float32, 0, (prefixLen+plan.MaxFrames-1)*kvWidth), make([]float32, 0, (prefixLen+plan.MaxFrames-1)*kvWidth)
	}
	rope := simd.BuildRoPEFreqs(prefixLen+plan.MaxFrames-1, talker.cfg.TalkerHeadDim/2, talker.cfg.TalkerHeadDim, talker.cfg.TalkerRoPETheta)
	if len(rope) == 0 {
		return BoundedCPUResult{}, fmt.Errorf("Qwen3-TTS RoPE table unavailable")
	}
	work := make([]talkerLayerScratch, len(talker.layers))
	for i := range work {
		work[i] = newTalkerLayerScratch(talker.cfg, prefixLen+plan.MaxFrames-1)
	}
	var current []float32
	for pos := 0; pos < prefixLen; pos++ {
		current = append(current[:0], inputs[pos*h:(pos+1)*h]...)
		for layer := range talker.layers {
			current, keys[layer], values[layer], err = talker.layers[layer].forwardWithScratch(current, keys[layer], values[layer], pos, rope, talker.cfg, &work[layer])
			if err != nil {
				return BoundedCPUResult{}, fmt.Errorf("Qwen3-TTS prefill layer %d position %d: %w", layer, pos, err)
			}
		}
	}
	semantic := make([]uint32, 1, plan.MaxFrames)
	semantic[0] = firstToken
	acoustic := make([]uint32, 0, plan.MaxFrames*15)
	acoustic = append(acoustic, codes0...)
	hiddenRows := make([][]float32, 0, plan.MaxFrames-1)
	logitRows := make([][]float32, 0, plan.MaxFrames-1)
	for frame := 0; frame < plan.MaxFrames-1; frame++ {
		// The reference adds semantic + 15 acoustic embeddings + the next
		// projected text token. After text exhaustion, it uses TTS EOS then PAD.
		textID := TTSPad
		if prefixLen+frame < len(plan.Prompt.Text) {
			textID = plan.Prompt.Text[prefixLen+frame]
		} else if prefixLen+frame == len(plan.Prompt.Text) {
			textID = TTSEOS
		}
		step, err := talker.projectText(textID)
		if err != nil {
			return BoundedCPUResult{}, err
		}
		if err := addTalkerEmbedding(step, talker.codecEmbedding, semantic[frame], talker.cfg.TalkerVocabSize, h); err != nil {
			return BoundedCPUResult{}, err
		}
		for i, code := range acoustic[frame*15 : (frame+1)*15] {
			if err := addTalkerEmbedding(step, predictor.embeddings[i], code, predictor.cfg.CPVocabSize, h); err != nil {
				return BoundedCPUResult{}, err
			}
		}
		for layer := range talker.layers {
			step, keys[layer], values[layer], err = talker.layers[layer].forwardWithScratch(step, keys[layer], values[layer], prefixLen+frame, rope, talker.cfg, &work[layer])
			if err != nil {
				return BoundedCPUResult{}, fmt.Errorf("Qwen3-TTS continuation layer %d position %d: %w", layer, prefixLen+frame, err)
			}
		}
		normed := append([]float32(nil), step...)
		if !simd.RMSNormTo(normed, talker.norm, float32(talker.cfg.TalkerRMSNormEps)) {
			return BoundedCPUResult{}, fmt.Errorf("Qwen3-TTS continuation norm failed")
		}
		logits := make([]float32, talker.cfg.TalkerVocabSize)
		if err := talker.codecHead.forward(logits, normed); err != nil {
			return BoundedCPUResult{}, err
		}
		next, err := selectToken(logits, CodecEOS)
		if err != nil {
			return BoundedCPUResult{}, err
		}
		if next == CodecEOS {
			break // EOS has no acoustic frame; decode the complete preceding frames.
		}
		codes, _, err := predictor.firstAcousticFrameWithWorkspace(talker, normed, next, acousticWork)
		if err != nil {
			return BoundedCPUResult{}, err
		}
		semantic = append(semantic, next)
		acoustic = append(acoustic, codes...)
		hiddenRows = append(hiddenRows, normed)
		logitRows = append(logitRows, logits)
	}
	wave, err := decoder.DecodeWaveform(plan, semantic, acoustic)
	if err != nil {
		return BoundedCPUResult{}, err
	}
	return BoundedCPUResult{Semantic: semantic, Acoustic: acoustic, ContinuationHidden: hiddenRows, ContinuationLogits: logitRows, Waveform: wave}, nil
}
