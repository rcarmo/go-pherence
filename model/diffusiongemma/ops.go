package diffusiongemma

import "fmt"

// OpKind names a high-level text decoder operation in the future native
// DiffusionGemma denoiser. It is intentionally architecture-level, not a kernel
// name: concrete CPU/SIMD/GPU implementations can lower these later.
type OpKind string

const (
	OpCanvasEmbedding OpKind = "canvas_embedding"
	OpInputNorm       OpKind = "input_norm"
	OpSelfAttention   OpKind = "self_attention"
	OpPostAttention   OpKind = "post_attention_norm"
	OpDenseMLP        OpKind = "dense_mlp"
	OpPreMoE          OpKind = "pre_moe_norm"
	OpRouter          OpKind = "router"
	OpExperts         OpKind = "experts"
	OpPostMoE         OpKind = "post_moe_norm"
	OpLayerScalar     OpKind = "layer_scalar"
	OpSelfCondition   OpKind = "self_condition"
	OpFinalNorm       OpKind = "final_norm"
	OpLMHead          OpKind = "lm_head"

	OpImagePreprocess       OpKind = "image_preprocess"
	OpImageSoftTokenPrompt  OpKind = "image_soft_token_prompt"
	OpVisionPatchEmbedding  OpKind = "vision_patch_embedding"
	OpVisionEncoderTower    OpKind = "vision_encoder_tower"
	OpVisionInputNorm       OpKind = "vision_input_norm"
	OpVisionSelfAttention   OpKind = "vision_self_attention"
	OpVisionPostAttention   OpKind = "vision_post_attention_norm"
	OpVisionPreFFN          OpKind = "vision_pre_ffn_norm"
	OpVisionDenseMLP        OpKind = "vision_dense_mlp"
	OpVisionPostFFN         OpKind = "vision_post_ffn_norm"
	OpVisionEmbeddingInsert OpKind = "vision_embedding_insert"
)

type LayerOp struct {
	Layer int    `json:"layer"`
	Type  string `json:"type,omitempty"`
	Kind  OpKind `json:"kind"`
}

type ForwardOpPlan struct {
	Prefix []OpKind  `json:"prefix"`
	Layers []LayerOp `json:"layers"`
	Tail   []OpKind  `json:"tail"`
	Ready  bool      `json:"ready"`
	Reason string    `json:"reason,omitempty"`
}

func BuildForwardOpPlan(shape Shape, textPlan *TextForwardPlan) ForwardOpPlan {
	if shape.TextLayers <= 0 {
		return ForwardOpPlan{Ready: false, Reason: "missing text layers"}
	}
	if textPlan != nil && !textPlan.Ready {
		return ForwardOpPlan{Ready: false, Reason: "text forward bindings incomplete"}
	}
	plan := ForwardOpPlan{Ready: true, Prefix: []OpKind{OpCanvasEmbedding, OpSelfCondition}, Tail: []OpKind{OpFinalNorm, OpLMHead}}
	for i := 0; i < shape.TextLayers; i++ {
		lt := layerTypeAt(shape.LayerTypes, i)
		plan.Layers = append(plan.Layers,
			LayerOp{Layer: i, Type: lt, Kind: OpInputNorm},
			LayerOp{Layer: i, Type: lt, Kind: OpSelfAttention},
			LayerOp{Layer: i, Type: lt, Kind: OpPostAttention},
			LayerOp{Layer: i, Type: lt, Kind: OpPreMoE},
			LayerOp{Layer: i, Type: lt, Kind: OpDenseMLP},
			LayerOp{Layer: i, Type: lt, Kind: OpRouter},
			LayerOp{Layer: i, Type: lt, Kind: OpExperts},
			LayerOp{Layer: i, Type: lt, Kind: OpPostMoE},
			LayerOp{Layer: i, Type: lt, Kind: OpLayerScalar},
		)
	}
	return plan
}

// ForwardContext owns per-call text denoiser inputs. Hidden/logit buffers are
// intentionally absent here; the concrete dispatcher will allocate or borrow
// buffers according to ForwardBufferPlan.
type EncoderKVLayer struct {
	Keys    []float32 `json:"-"`
	Values  []float32 `json:"-"`
	SeqLen  int       `json:"seq_len"`
	KVHeads int       `json:"kv_heads"`
	HeadDim int       `json:"head_dim"`
}

type ForwardContext struct {
	PromptIDs              []int            `json:"prompt_ids,omitempty"`
	Canvas                 []int            `json:"canvas"`
	Step                   int              `json:"step"`
	SelfConditioning       []float32        `json:"-"`
	SelfConditioningLogits [][]float32      `json:"-"` // previous raw canvas logits [canvas][vocab], llama.cpp SC input
	SCTempInv              float32          `json:"-"` // 1/t for self-conditioning softmax
	SampleDraws            []float64        `json:"-"`
	EncoderKV              []EncoderKVLayer `json:"-"`
	EncoderSeqLen          int              `json:"encoder_seq_len,omitempty"`
}

// ForwardDispatcher is the explicit boundary for tensor-backed text forward
// execution (CPU, GPU, GGUF, FP8, or test/mock dispatchers).
type ForwardDispatcher interface {
	RunTextForward(ctx ForwardContext, weights *TextWeights, ops ForwardOpPlan, buffers ForwardBufferPlan) (ForwardOutput, error)
}

type NotImplementedDispatcher struct{}

func (NotImplementedDispatcher) RunTextForward(ctx ForwardContext, weights *TextWeights, ops ForwardOpPlan, buffers ForwardBufferPlan) (ForwardOutput, error) {
	return ForwardOutput{}, fmt.Errorf("DiffusionGemma text forward dispatcher is not configured")
}
