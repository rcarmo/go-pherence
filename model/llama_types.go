package model

import (
	"github.com/rcarmo/go-pherence/backends/mlx"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/runtime/kv"
	"github.com/rcarmo/go-pherence/tensor"
)

// QuantWeight holds GPTQ INT4 weight data for on-the-fly dequantization.
type QuantWeight struct {
	QWeight []int32   // [inDim/8, outDim] packed
	GIdx    []int32   // [inDim] group index
	Scales  []float32 // [numGroups, outDim]
	InDim   int
	OutDim  int
}

type LlamaConfig struct {
	VocabSize             int      `json:"vocab_size"`
	HiddenSize            int      `json:"hidden_size"`
	Intermediate          int      `json:"intermediate_size"`
	NumLayers             int      `json:"num_hidden_layers"`
	NumHeads              int      `json:"num_attention_heads"`
	NumKVHeads            int      `json:"num_key_value_heads"`
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

// LlamaModel holds loaded weights for a LLaMA-style decoder.
type LlamaModel struct {
	Config LlamaConfig
	Tok    *tokenizer.Tokenizer // optional, for chat templates

	EmbedTokens *tensor.Tensor // [vocab, hidden]

	EmbedPerLayer      []float32
	PerLayerModelProj  []float32
	PerLayerProjNorm   []float32
	PerLayerInputScale float32
	PerLayerProjScale  float32
	EmbedPerLayerScale float32
	Norm               *tensor.Tensor
	LMHead             *tensor.Tensor
	LMHeadMLX          *mlx.QuantWeight

	Layers []LlamaLayer

	RopeFreqs     []float32
	RopeFreqsSWA  []float32
	RopeFreqsFull []float32
	RopeHalfSWA   int
	RopeHalfFull  int
	Large         bool
	Quantized     bool
	OnTheFlyQuant bool

	EnableTurboQuant bool
	TurboQuantStates map[int]*kv.TurboQuantState
}

type LlamaLayer struct {
	InputNorm     *tensor.Tensor
	PostNorm      *tensor.Tensor
	PreFFNNorm    *tensor.Tensor
	VNorm         *tensor.Tensor
	LayerScalar   float32
	HeadDimLocal  int
	HasKV         bool
	KVSourceLayer int

	PLIGate     []float32
	PLIProj     []float32
	PLIPostNorm []float32
	PostFFNNorm *tensor.Tensor

	QW, KW, VW, OW *tensor.Tensor
	QB, KB, VB     *tensor.Tensor
	QNorm, KNorm   *tensor.Tensor

	QWq, KWq, VWq, OWq   *QuantWeight
	GateWq, UpWq, DownWq *QuantWeight

	QWm, KWm, VWm, OWm   *mlx.QuantWeight
	GateWm, UpWm, DownWm *mlx.QuantWeight

	GateW, UpW, DownW *tensor.Tensor

	IsMoE       bool
	RouterW     *mlx.QuantWeight
	ExpertGateW []*mlx.QuantWeight
	ExpertUpW   []*mlx.QuantWeight
	ExpertDownW []*mlx.QuantWeight
}

func (c LlamaConfig) HasNativeMTP() bool {
	return c.MTPNumHiddenLayers > 0
}

func (c LlamaConfig) IsOrthrus() bool {
	if c.OrthrusBlockSize > 0 && c.OrthrusMaskTokenID >= 0 {
		for _, arch := range c.Architectures {
			if arch == "OrthrusLM" {
				return true
			}
		}
	}
	return false
}
