package common

// Config contains architecture-neutral decoder/model metadata plus optional
// architecture-specific fields that are interpreted by dedicated model loaders.
// It intentionally does not own loaded tensor weights or runtime state.
type Config struct {
	VocabSize             int      `json:"vocab_size"`
	HiddenSize            int      `json:"hidden_size"`
	Intermediate          int      `json:"intermediate_size"`
	NumLayers             int      `json:"num_hidden_layers"`
	NumHeads              int      `json:"num_attention_heads"`
	NumKVHeads            int      `json:"num_key_value_heads"`
	NumGlobalKVHeads      int      `json:"num_global_key_value_heads"`
	MaxSeqLen             int      `json:"max_position_embeddings"`
	RopeTheta             float64  `json:"rope_theta"`
	RMSNormEps            float64  `json:"rms_norm_eps"`
	ModelType             string   `json:"model_type"`
	Architectures         []string `json:"architectures"`
	TieEmbeddings         bool     `json:"tie_word_embeddings"`
	HeadDim               int      `json:"head_dim"`
	SlidingWindow         int      `json:"sliding_window"`
	SlidingWindowPattern  int      `json:"sliding_window_pattern"`
	RopeLocalBaseFreq     float64  `json:"rope_local_base_freq"`
	BOSTokenID            int      `json:"bos_token_id"`
	LayerTypes            []string `json:"layer_types"`
	NumKVSharedLayers     int      `json:"num_kv_shared_layers"`
	GlobalHeadDim         int      `json:"global_head_dim"`
	HiddenPerLayer        int      `json:"hidden_size_per_layer_input"`
	VocabPerLayer         int      `json:"vocab_size_per_layer_input"`
	TensorPrefix          string   `json:"-"` // "language_model.model." for Gemma4
	HiddenAct             string   `json:"hidden_activation"`
	FullAttentionInterval int      `json:"full_attention_interval"`
	LinearConvKernelDim   int      `json:"linear_conv_kernel_dim"`
	LinearKeyHeadDim      int      `json:"linear_key_head_dim"`
	LinearNumKeyHeads     int      `json:"linear_num_key_heads"`
	LinearNumValueHeads   int      `json:"linear_num_value_heads"`
	LinearValueHeadDim    int      `json:"linear_value_head_dim"`
	AttentionKEqV         bool     `json:"attention_k_eq_v"`

	MTPNumHiddenLayers        int  `json:"mtp_num_hidden_layers"`
	MTPUseDedicatedEmbeddings bool `json:"mtp_use_dedicated_embeddings"`

	OrthrusBlockSize   int `json:"block_size"`
	OrthrusMaskTokenID int `json:"mask_token_id"`

	QuantBits   int    `json:"-"`
	QuantGroup  int    `json:"-"`
	QuantSym    bool   `json:"-"`
	QuantFormat string `json:"-"` // "gptq" or "mlx"

	NumExperts        int  `json:"num_experts"`
	NumExpertsPerTok  int  `json:"num_experts_per_tok"`
	MoEIntermediate   int  `json:"moe_intermediate_size"`
	DecoderSparseStep int  `json:"decoder_sparse_step"`
	NormTopKProb      bool `json:"norm_topk_prob"`
}

func (c Config) HasNativeMTP() bool {
	return c.MTPNumHiddenLayers > 0
}

func (c Config) IsOrthrus() bool {
	if c.OrthrusBlockSize > 0 && c.OrthrusMaskTokenID >= 0 {
		for _, arch := range c.Architectures {
			if arch == "OrthrusLM" {
				return true
			}
		}
	}
	return false
}
