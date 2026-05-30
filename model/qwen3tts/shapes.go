package qwen3tts

import "fmt"

// RuntimePlan captures shape/cache sizes for the staged Qwen3-TTS pipeline. It
// is intentionally allocation-free: use it to validate model metadata and size
// buffers before implementing Talker/CodePredictor/Decoder execution.
type RuntimePlan struct {
	Talker                  TransformerPlan         `json:"talker"`
	CodePredictor           TransformerPlan         `json:"code_predictor"`
	Decoder12Hz             DecoderPlan             `json:"decoder12hz"`
	SemanticTokenLayout     SemanticTokenLayout     `json:"semantic_token_layout"`
	AcousticFrameLayout     AcousticFrameLayout     `json:"acoustic_frame_layout"`
	CodePredictorHeadLayout CodePredictorHeadLayout `json:"code_predictor_head_layout"`
	WaveformLayout          WaveformLayout          `json:"waveform_layout"`
	DecoderInputLayout      DecoderInputLayout      `json:"decoder_input_layout"`
	SpeakerEncoderLayout    SpeakerEncoderLayout    `json:"speaker_encoder_layout"`
	TalkerAttentionLayout   AttentionLayout         `json:"talker_attention_layout"`
	CPAttentionLayout       AttentionLayout         `json:"code_predictor_attention_layout"`
	TalkerFFNLayout         FFNLayout               `json:"talker_ffn_layout"`
	CPFFNLayout             FFNLayout               `json:"code_predictor_ffn_layout"`
	EmbeddingLayout         EmbeddingLayout         `json:"embedding_layout"`
	Pipeline                PipelinePlan            `json:"pipeline"`
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
	pipeline, err := NewPipelinePlan(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	semanticLayout, err := NewSemanticTokenLayout(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	frameLayout, err := NewAcousticFrameLayout(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	headLayout, err := NewCodePredictorHeadLayout(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	decoderInputLayout, err := NewDecoderInputLayout(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	decoderPlan, err := decoderInputLayout.DecoderPlan()
	if err != nil {
		return RuntimePlan{}, err
	}
	waveformLayout, err := NewWaveformLayout(decoderPlan)
	if err != nil {
		return RuntimePlan{}, err
	}
	speakerEncoderLayout, err := NewSpeakerEncoderLayout(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	talkerAttentionLayout, err := NewTalkerAttentionLayout(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	cpAttentionLayout, err := NewCodePredictorAttentionLayout(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	talkerFFNLayout, err := NewTalkerFFNLayout(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	cpFFNLayout, err := NewCodePredictorFFNLayout(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	embeddingLayout, err := NewEmbeddingLayout(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	talkerKVFloats, err := talkerAttentionLayout.KVFloatsPerToken()
	if err != nil {
		return RuntimePlan{}, err
	}
	cpKVFloats, err := cpAttentionLayout.KVFloatsPerToken()
	if err != nil {
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
			KVFloatsPerToken: talkerKVFloats,
		},
		CodePredictor: TransformerPlan{
			HiddenSize:       cfg.CPHiddenSize,
			IntermediateSize: cfg.CPIntermediateSize,
			Layers:           cfg.CPNumHiddenLayers,
			Heads:            cfg.CPNumAttentionHeads,
			KVHeads:          cfg.CPNumKeyValueHeads,
			HeadDim:          cfg.CPHeadDim,
			VocabSize:        cfg.CPVocabSize,
			KVFloatsPerToken: cpKVFloats,
		},
		Decoder12Hz:             decoderPlan,
		SemanticTokenLayout:     semanticLayout,
		AcousticFrameLayout:     frameLayout,
		CodePredictorHeadLayout: headLayout,
		WaveformLayout:          waveformLayout,
		DecoderInputLayout:      decoderInputLayout,
		SpeakerEncoderLayout:    speakerEncoderLayout,
		TalkerAttentionLayout:   talkerAttentionLayout,
		CPAttentionLayout:       cpAttentionLayout,
		TalkerFFNLayout:         talkerFFNLayout,
		CPFFNLayout:             cpFFNLayout,
		EmbeddingLayout:         embeddingLayout,
		Pipeline:                pipeline,
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
	if p.DecoderInputLayout.CodecVocab > 0 {
		decoderPlan, err := p.DecoderInputLayout.DecoderPlan()
		if err != nil {
			return err
		}
		if decoderPlan != p.Decoder12Hz {
			return fmt.Errorf("Qwen3-TTS decoder input/plan mismatch: input=%+v plan=%+v", decoderPlan, p.Decoder12Hz)
		}
	}
	if p.SemanticTokenLayout.VocabSize > 0 {
		if err := p.SemanticTokenLayout.Validate(); err != nil {
			return err
		}
	}
	if p.AcousticFrameLayout.TotalCodeGroups > 0 {
		if err := p.AcousticFrameLayout.Validate(); err != nil {
			return err
		}
	}
	if p.CodePredictorHeadLayout.Heads > 0 {
		if err := p.CodePredictorHeadLayout.Validate(); err != nil {
			return err
		}
	}
	if p.WaveformLayout.SampleRateHz > 0 {
		if err := p.WaveformLayout.Validate(); err != nil {
			return err
		}
	}
	if p.DecoderInputLayout.CodecVocab > 0 {
		if err := p.DecoderInputLayout.Validate(); err != nil {
			return err
		}
	}
	if err := p.SpeakerEncoderLayout.Validate(); err != nil {
		return err
	}
	if p.TalkerAttentionLayout.HiddenSize > 0 {
		if err := p.TalkerAttentionLayout.Validate(); err != nil {
			return err
		}
		kvFloats, err := p.TalkerAttentionLayout.KVFloatsPerToken()
		if err != nil {
			return err
		}
		if p.Talker.KVFloatsPerToken != kvFloats {
			return fmt.Errorf("Qwen3-TTS talker KV plan/layout mismatch: plan=%d layout=%d", p.Talker.KVFloatsPerToken, kvFloats)
		}
	}
	if p.CPAttentionLayout.HiddenSize > 0 {
		if err := p.CPAttentionLayout.Validate(); err != nil {
			return err
		}
		kvFloats, err := p.CPAttentionLayout.KVFloatsPerToken()
		if err != nil {
			return err
		}
		if p.CodePredictor.KVFloatsPerToken != kvFloats {
			return fmt.Errorf("Qwen3-TTS code predictor KV plan/layout mismatch: plan=%d layout=%d", p.CodePredictor.KVFloatsPerToken, kvFloats)
		}
	}
	if p.TalkerFFNLayout.HiddenSize > 0 {
		if err := p.TalkerFFNLayout.Validate(); err != nil {
			return err
		}
		if err := p.Talker.MatchesFFNLayout(p.TalkerFFNLayout); err != nil {
			return err
		}
	}
	if p.CPFFNLayout.HiddenSize > 0 {
		if err := p.CPFFNLayout.Validate(); err != nil {
			return err
		}
		if err := p.CodePredictor.MatchesFFNLayout(p.CPFFNLayout); err != nil {
			return err
		}
	}
	if p.EmbeddingLayout.TalkerHiddenSize > 0 {
		if err := p.EmbeddingLayout.Validate(); err != nil {
			return err
		}
		if p.Talker.HiddenSize != p.EmbeddingLayout.TalkerHiddenSize || p.Talker.VocabSize != p.EmbeddingLayout.TalkerCodecVocabSize {
			return fmt.Errorf("Qwen3-TTS talker embedding layout mismatch: talker=%+v embedding=%+v", p.Talker, p.EmbeddingLayout)
		}
		if p.CodePredictor.HiddenSize != p.EmbeddingLayout.CodePredictorHiddenSize || p.CodePredictor.VocabSize != p.EmbeddingLayout.CodePredictorVocabSize {
			return fmt.Errorf("Qwen3-TTS code predictor embedding layout mismatch: code_predictor=%+v embedding=%+v", p.CodePredictor, p.EmbeddingLayout)
		}
	}
	if len(p.Pipeline.Steps) > 0 {
		if err := p.Pipeline.Validate(); err != nil {
			return err
		}
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

func (p TransformerPlan) MatchesFFNLayout(layout FFNLayout) error {
	if err := layout.Validate(); err != nil {
		return err
	}
	if p.HiddenSize != layout.HiddenSize || p.IntermediateSize != layout.IntermediateSize || p.Layers != layout.Layers {
		return fmt.Errorf("Qwen3-TTS %s transformer/FFN layout mismatch: transformer=%+v ffn=%+v", layout.Name, p, layout)
	}
	return nil
}

func (p TransformerPlan) KVBytes(maxSeq int, bytesPerFloat int) (int64, error) {
	if maxSeq < 0 || bytesPerFloat <= 0 {
		return 0, fmt.Errorf("invalid KV sizing arguments: max_seq=%d bytes_per_float=%d", maxSeq, bytesPerFloat)
	}
	return int64(maxSeq) * int64(p.KVFloatsPerToken) * int64(bytesPerFloat), nil
}
