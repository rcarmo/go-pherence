package lfm2

import "fmt"

// RuntimePlan records state/cache sizing for LFM2 before runtime execution is
// implemented. It makes the conv/full-attention/MoE split explicit and keeps
// cache accounting testable without model weights.
type RuntimePlan struct {
	HiddenSize          int                       `json:"hidden_size"`
	HeadDim             int                       `json:"head_dim"`
	Layers              int                       `json:"layers"`
	ConvLayers          int                       `json:"conv_layers"`
	FullAttentionLayers int                       `json:"full_attention_layers"`
	KVHeads             int                       `json:"kv_heads"`
	ConvLCache          int                       `json:"conv_l_cache"`
	ConvStateFloats     int                       `json:"conv_state_floats"`
	KVFloatsPerToken    int                       `json:"kv_floats_per_token"`
	Experts             int                       `json:"experts"`
	ExpertsPerToken     int                       `json:"experts_per_token"`
	MoEIntermediate     int                       `json:"moe_intermediate"`
	Schedule            LayerSchedule             `json:"schedule"`
	Execution           ExecutionPlan             `json:"execution"`
	Routing             RoutingPlan               `json:"routing"`
	RouterLayout        RouterLayout              `json:"router_layout"`
	ConvStateLayout     ConvStateLayout           `json:"conv_state_layout"`
	ConvProjLayout      ConvProjectionLayout      `json:"conv_projection_layout"`
	AttentionKVLayout   AttentionKVLayout         `json:"attention_kv_layout"`
	AttentionProjLayout AttentionProjectionLayout `json:"attention_projection_layout"`
	ContextLayout       ContextLayout             `json:"context_layout"`
	RoPELayout          RoPELayout                `json:"rope_layout"`
	FFNLayout           FFNLayout                 `json:"ffn_layout"`
	NormLayout          NormLayout                `json:"norm_layout"`
	EmbeddingLayout     EmbeddingLayout           `json:"embedding_layout"`
}

func NewRuntimePlan(cfg Config) (RuntimePlan, error) {
	if err := cfg.Validate(); err != nil {
		return RuntimePlan{}, err
	}
	schedule, err := NewLayerSchedule(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	execution, err := NewExecutionPlan(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	routing, err := NewRoutingPlan(cfg, execution)
	if err != nil {
		return RuntimePlan{}, err
	}
	routerLayout, err := NewRouterLayout(cfg, execution)
	if err != nil {
		return RuntimePlan{}, err
	}
	convStateLayout, err := NewConvStateLayout(cfg, schedule)
	if err != nil {
		return RuntimePlan{}, err
	}
	convProjLayout, err := NewConvProjectionLayout(cfg, schedule)
	if err != nil {
		return RuntimePlan{}, err
	}
	attentionKVLayout, err := NewAttentionKVLayout(cfg, schedule)
	if err != nil {
		return RuntimePlan{}, err
	}
	attentionProjLayout, err := NewAttentionProjectionLayout(cfg, schedule)
	if err != nil {
		return RuntimePlan{}, err
	}
	contextLayout, err := NewContextLayout(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	ropeLayout, err := NewRoPELayout(cfg, schedule)
	if err != nil {
		return RuntimePlan{}, err
	}
	ffnLayout, err := NewFFNLayout(cfg, execution)
	if err != nil {
		return RuntimePlan{}, err
	}
	normLayout, err := NewNormLayout(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	embeddingLayout, err := NewEmbeddingLayout(cfg)
	if err != nil {
		return RuntimePlan{}, err
	}
	plan := RuntimePlan{
		HiddenSize:          cfg.HiddenSize,
		HeadDim:             cfg.HeadDim,
		Layers:              cfg.NumHiddenLayers,
		ConvLayers:          cfg.ConvLayerCount(),
		FullAttentionLayers: cfg.FullAttentionLayerCount(),
		KVHeads:             cfg.NumKeyValueHeads,
		ConvLCache:          cfg.ConvLCache,
		ConvStateFloats:     cfg.ConvLayerCount() * cfg.ConvLCache * cfg.HiddenSize,
		KVFloatsPerToken:    2 * cfg.FullAttentionLayerCount() * cfg.NumKeyValueHeads * cfg.HeadDim,
		Experts:             cfg.NumExperts,
		ExpertsPerToken:     cfg.NumExpertsPerTok,
		MoEIntermediate:     cfg.MoEIntermediateSize,
		Schedule:            schedule,
		Execution:           execution,
		Routing:             routing,
		RouterLayout:        routerLayout,
		ConvStateLayout:     convStateLayout,
		ConvProjLayout:      convProjLayout,
		AttentionKVLayout:   attentionKVLayout,
		AttentionProjLayout: attentionProjLayout,
		ContextLayout:       contextLayout,
		RoPELayout:          ropeLayout,
		FFNLayout:           ffnLayout,
		NormLayout:          normLayout,
		EmbeddingLayout:     embeddingLayout,
	}
	if err := plan.Validate(); err != nil {
		return RuntimePlan{}, err
	}
	return plan, nil
}

func (p RuntimePlan) Validate() error {
	if p.HiddenSize <= 0 || p.HeadDim <= 0 || p.Layers <= 0 || p.KVHeads <= 0 {
		return fmt.Errorf("invalid LFM2 runtime plan dims: %+v", p)
	}
	if p.ConvLayers+p.FullAttentionLayers != p.Layers {
		return fmt.Errorf("invalid LFM2 layer counts: conv=%d attention=%d layers=%d", p.ConvLayers, p.FullAttentionLayers, p.Layers)
	}
	if p.ConvStateFloats != p.ConvLayers*p.ConvLCache*p.HiddenSize {
		return fmt.Errorf("invalid LFM2 conv state floats=%d", p.ConvStateFloats)
	}
	wantKV := 2 * p.FullAttentionLayers * p.KVHeads * p.HeadDim
	if p.KVFloatsPerToken != wantKV {
		return fmt.Errorf("invalid LFM2 KV floats/token=%d want=%d", p.KVFloatsPerToken, wantKV)
	}
	if p.Experts <= 0 || p.ExpertsPerToken <= 0 || p.ExpertsPerToken > p.Experts || p.MoEIntermediate <= 0 {
		return fmt.Errorf("invalid LFM2 MoE plan: %+v", p)
	}
	if len(p.Schedule.Steps) > 0 {
		if err := p.Schedule.Validate(p.Layers); err != nil {
			return err
		}
	}
	if len(p.Execution.Steps) > 0 {
		if err := p.Execution.Validate(p.Layers); err != nil {
			return err
		}
	}
	if p.Routing.Experts > 0 {
		if err := p.Routing.Validate(); err != nil {
			return err
		}
	}
	if p.RouterLayout.Experts > 0 {
		if err := p.RouterLayout.Validate(); err != nil {
			return err
		}
	}
	if p.ConvStateLayout.Layers > 0 {
		if err := p.ConvStateLayout.Validate(); err != nil {
			return err
		}
	}
	if p.ConvProjLayout.HiddenSize > 0 {
		if err := p.ConvProjLayout.Validate(); err != nil {
			return err
		}
	}
	if p.AttentionKVLayout.Layers > 0 {
		if err := p.AttentionKVLayout.Validate(); err != nil {
			return err
		}
	}
	if p.AttentionProjLayout.HiddenSize > 0 {
		if err := p.AttentionProjLayout.Validate(); err != nil {
			return err
		}
	}
	if p.ContextLayout.VocabSize > 0 {
		if err := p.ContextLayout.Validate(); err != nil {
			return err
		}
	}
	if p.RoPELayout.HeadDim > 0 {
		if err := p.RoPELayout.Validate(); err != nil {
			return err
		}
	}
	if p.FFNLayout.HiddenSize > 0 {
		if err := p.FFNLayout.Validate(); err != nil {
			return err
		}
	}
	if p.NormLayout.HiddenSize > 0 {
		if err := p.NormLayout.Validate(); err != nil {
			return err
		}
	}
	if p.EmbeddingLayout.HiddenSize > 0 {
		if err := p.EmbeddingLayout.Validate(); err != nil {
			return err
		}
	}
	return nil
}

func (p RuntimePlan) KVBytes(maxSeq int, bytesPerFloat int) (int64, error) {
	if maxSeq < 0 || bytesPerFloat <= 0 {
		return 0, fmt.Errorf("invalid KV sizing arguments: max_seq=%d bytes_per_float=%d", maxSeq, bytesPerFloat)
	}
	return int64(maxSeq) * int64(p.KVFloatsPerToken) * int64(bytesPerFloat), nil
}

func (p RuntimePlan) ConvStateBytes(bytesPerFloat int) (int64, error) {
	if bytesPerFloat <= 0 {
		return 0, fmt.Errorf("invalid conv state bytes/float=%d", bytesPerFloat)
	}
	return int64(p.ConvStateFloats) * int64(bytesPerFloat), nil
}
