package qwen3tts

import "fmt"

// RuntimePlan captures shape/cache sizes for the staged Qwen3-TTS pipeline. It
// is intentionally allocation-free: use it to validate model metadata and size
// buffers before implementing Talker/CodePredictor/Decoder execution.
type RuntimePlan struct {
	Talker        TransformerPlan `json:"talker"`
	CodePredictor TransformerPlan `json:"code_predictor"`
	Decoder12Hz   DecoderPlan     `json:"decoder12hz"`
}

type TransformerPlan struct {
	HiddenSize       int `json:"hidden_size"`
	IntermediateSize int `json:"intermediate_size"`
	Layers           int `json:"layers"`
	Heads            int `json:"heads"`
	KVHeads          int `json:"kv_heads"`
	HeadDim          int `json:"head_dim"`
	VocabSize        int `json:"vocab_size"`
	KVFloatsPerToken int `json:"kv_floats_per_token"`
}

type DecoderPlan struct {
	FrameRateHz   int `json:"frame_rate_hz"`
	CodeGroups    int `json:"code_groups"`
	CodesPerFrame int `json:"codes_per_frame"`
	CodecVocab    int `json:"codec_vocab"`
}

func NewRuntimePlan(cfg ParsedConfig) (RuntimePlan, error) {
	if err := cfg.Validate(); err != nil {
		return RuntimePlan{}, err
	}
	plan := RuntimePlan{
		Talker: TransformerPlan{
			HiddenSize:       cfg.TalkerHiddenSize,
			IntermediateSize: cfg.TalkerIntermediateSize,
			Layers:           cfg.TalkerNumHiddenLayers,
			Heads:            cfg.TalkerNumAttentionHeads,
			KVHeads:          cfg.TalkerNumKeyValueHeads,
			HeadDim:          cfg.TalkerHeadDim,
			VocabSize:        cfg.TalkerVocabSize,
			KVFloatsPerToken: 2 * cfg.TalkerNumHiddenLayers * cfg.TalkerNumKeyValueHeads * cfg.TalkerHeadDim,
		},
		CodePredictor: TransformerPlan{
			HiddenSize:       cfg.CPHiddenSize,
			IntermediateSize: cfg.CPIntermediateSize,
			Layers:           cfg.CPNumHiddenLayers,
			Heads:            cfg.CPNumAttentionHeads,
			KVHeads:          cfg.CPNumKeyValueHeads,
			HeadDim:          cfg.CPHeadDim,
			VocabSize:        cfg.CPVocabSize,
			KVFloatsPerToken: 2 * cfg.CPNumHiddenLayers * cfg.CPNumKeyValueHeads * cfg.CPHeadDim,
		},
		Decoder12Hz: DecoderPlan{FrameRateHz: 12, CodeGroups: cfg.CPNumCodeGroups - 1, CodesPerFrame: cfg.CPNumCodeGroups - 1, CodecVocab: cfg.CPVocabSize},
	}
	if err := plan.Validate(); err != nil {
		return RuntimePlan{}, err
	}
	return plan, nil
}

func (p RuntimePlan) Validate() error {
	if err := p.Talker.Validate("talker"); err != nil {
		return err
	}
	if err := p.CodePredictor.Validate("code_predictor"); err != nil {
		return err
	}
	if p.Decoder12Hz.FrameRateHz != 12 || p.Decoder12Hz.CodeGroups <= 0 || p.Decoder12Hz.CodesPerFrame != p.Decoder12Hz.CodeGroups || p.Decoder12Hz.CodecVocab <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS decoder plan: %+v", p.Decoder12Hz)
	}
	return nil
}

func (p TransformerPlan) Validate(label string) error {
	if p.HiddenSize <= 0 || p.Layers <= 0 || p.Heads <= 0 || p.KVHeads <= 0 || p.HeadDim <= 0 || p.VocabSize <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS %s plan: %+v", label, p)
	}
	if p.HiddenSize != p.Heads*p.HeadDim {
		return fmt.Errorf("invalid Qwen3-TTS %s hidden/head dims: hidden=%d heads=%d head_dim=%d", label, p.HiddenSize, p.Heads, p.HeadDim)
	}
	wantKV := 2 * p.Layers * p.KVHeads * p.HeadDim
	if p.KVFloatsPerToken != wantKV {
		return fmt.Errorf("invalid Qwen3-TTS %s KV floats/token=%d want=%d", label, p.KVFloatsPerToken, wantKV)
	}
	return nil
}

func (p TransformerPlan) KVBytes(maxSeq int, bytesPerFloat int) (int64, error) {
	if maxSeq < 0 || bytesPerFloat <= 0 {
		return 0, fmt.Errorf("invalid KV sizing arguments: max_seq=%d bytes_per_float=%d", maxSeq, bytesPerFloat)
	}
	return int64(maxSeq) * int64(p.KVFloatsPerToken) * int64(bytesPerFloat), nil
}
