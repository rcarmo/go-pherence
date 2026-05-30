package qwen3tts

import "fmt"

// AttentionLayout captures the shared Qwen3-TTS transformer attention contract
// for the Talker and CodePredictor stacks: GQA grouping, RoPE settings, and
// RMSNorm epsilon. It keeps these runtime assumptions explicit before attention
// kernels and KV-cache code are implemented.
type AttentionLayout struct {
	Name                  string  `json:"name"`
	HiddenSize            int     `json:"hidden_size"`
	Layers                int     `json:"layers"`
	Heads                 int     `json:"heads"`
	KVHeads               int     `json:"kv_heads"`
	HeadDim               int     `json:"head_dim"`
	QueriesPerKV          int     `json:"queries_per_kv"`
	RoPETheta             float64 `json:"rope_theta"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings,omitempty"`
	RMSNormEps            float64 `json:"rms_norm_eps"`
}

func NewTalkerAttentionLayout(cfg ParsedConfig) (AttentionLayout, error) {
	if err := cfg.Validate(); err != nil {
		return AttentionLayout{}, err
	}
	layout := AttentionLayout{
		Name:                  "talker",
		HiddenSize:            cfg.TalkerHiddenSize,
		Layers:                cfg.TalkerNumHiddenLayers,
		Heads:                 cfg.TalkerNumAttentionHeads,
		KVHeads:               cfg.TalkerNumKeyValueHeads,
		HeadDim:               cfg.TalkerHeadDim,
		QueriesPerKV:          cfg.TalkerNumAttentionHeads / cfg.TalkerNumKeyValueHeads,
		RoPETheta:             cfg.TalkerRoPETheta,
		MaxPositionEmbeddings: cfg.TalkerMaxPositionEmbedding,
		RMSNormEps:            cfg.TalkerRMSNormEps,
	}
	return layout, layout.Validate()
}

func NewCodePredictorAttentionLayout(cfg ParsedConfig) (AttentionLayout, error) {
	if err := cfg.Validate(); err != nil {
		return AttentionLayout{}, err
	}
	layout := AttentionLayout{
		Name:         "code_predictor",
		HiddenSize:   cfg.CPHiddenSize,
		Layers:       cfg.CPNumHiddenLayers,
		Heads:        cfg.CPNumAttentionHeads,
		KVHeads:      cfg.CPNumKeyValueHeads,
		HeadDim:      cfg.CPHeadDim,
		QueriesPerKV: cfg.CPNumAttentionHeads / cfg.CPNumKeyValueHeads,
		RoPETheta:    cfg.CPRoPETheta,
		RMSNormEps:   cfg.CPRMSNormEps,
	}
	return layout, layout.Validate()
}

func (l AttentionLayout) Validate() error {
	if l.Name == "" || l.HiddenSize <= 0 || l.Layers <= 0 || l.Heads <= 0 || l.KVHeads <= 0 || l.HeadDim <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS attention layout dims: %+v", l)
	}
	if l.HiddenSize != l.Heads*l.HeadDim {
		return fmt.Errorf("invalid Qwen3-TTS %s attention hidden/head dims: hidden=%d heads=%d head_dim=%d", l.Name, l.HiddenSize, l.Heads, l.HeadDim)
	}
	if l.Heads%l.KVHeads != 0 {
		return fmt.Errorf("invalid Qwen3-TTS %s GQA grouping: heads=%d kv_heads=%d", l.Name, l.Heads, l.KVHeads)
	}
	if l.QueriesPerKV != l.Heads/l.KVHeads {
		return fmt.Errorf("invalid Qwen3-TTS %s queries/kv=%d want=%d", l.Name, l.QueriesPerKV, l.Heads/l.KVHeads)
	}
	if l.RoPETheta <= 0 || l.RMSNormEps <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS %s rope/norm settings: theta=%g eps=%g", l.Name, l.RoPETheta, l.RMSNormEps)
	}
	if l.MaxPositionEmbeddings < 0 {
		return fmt.Errorf("invalid Qwen3-TTS %s max positions=%d", l.Name, l.MaxPositionEmbeddings)
	}
	return nil
}

func (l AttentionLayout) KVFloatsPerToken() (int, error) {
	if err := l.Validate(); err != nil {
		return 0, err
	}
	return 2 * l.Layers * l.KVHeads * l.HeadDim, nil
}

func (l AttentionLayout) KVBytes(maxSeq int, bytesPerFloat int) (int64, error) {
	floats, err := l.KVFloatsPerToken()
	if err != nil {
		return 0, err
	}
	if maxSeq < 0 || bytesPerFloat <= 0 {
		return 0, fmt.Errorf("invalid Qwen3-TTS %s KV sizing arguments: max_seq=%d bytes_per_float=%d", l.Name, maxSeq, bytesPerFloat)
	}
	return int64(maxSeq) * int64(floats) * int64(bytesPerFloat), nil
}

func (l AttentionLayout) ValidatePosition(pos int) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if pos < 0 {
		return fmt.Errorf("invalid Qwen3-TTS %s position=%d", l.Name, pos)
	}
	if l.MaxPositionEmbeddings > 0 && pos >= l.MaxPositionEmbeddings {
		return fmt.Errorf("Qwen3-TTS %s position %d exceeds max_position_embeddings=%d", l.Name, pos, l.MaxPositionEmbeddings)
	}
	return nil
}
