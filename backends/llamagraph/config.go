package llamagraph

// Config holds all hyper-parameters and per-layer quantisation types needed
// at init time so we can allocate one contiguous weight buffer up front.
type Config struct {
	NVocab, NEmbd, NHeads, NHeadsKV int
	NLayers, NFF, NCtx              int
	RopeBase, RmsEps                float32
	RopeDims, NThreads              int

	// GGML type ints (ggml_type enum) per weight tensor.
	// Use the GGMLType* constants below.
	TokEmbdType int
	OutputType  int
	// Per-layer types (length must be >= NLayers)
	WQType, WKType, WVType, WOType      []int
	FFNGateType, FFNUpType, FFNDownType []int
	// Output dimensions (0 = use default n_embd / n_embd_kv)
	WQOut, WKOut, WVOut, WOIn []int
	HasQKNorm                 bool // Qwen3+ QK norm
}

// GGML type enum constants (mirror ggml_type).
const (
	GGMLTypeF32  = 0
	GGMLTypeF16  = 1
	GGMLTypeQ4_0 = 2
	GGMLTypeQ4_1 = 3
	GGMLTypeQ4_K = 12
	GGMLTypeQ6_K = 14
	GGMLTypeQ2_K = 10
	GGMLTypeQ3_K = 11
	GGMLTypeQ8_K = 15
)
