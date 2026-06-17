package diffusiongemma

import (
	"fmt"
	"log"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// TextDenoiser is the future native tensor-backed DiffusionGemma denoiser. It
// currently validates and owns the metadata/weight binding needed by a forward
// pass, but does not yet implement layer math.
type TextDenoiser struct {
	Shape            Shape
	Weights          *TextWeights
	Plan             TextForwardPlan
	Ops              ForwardOpPlan
	Buffers          ForwardBufferPlan
	Dispatcher       ForwardDispatcher
	EncoderKV        []EncoderKVLayer
	EncoderPromptIDs []int
	EncoderPromptLen int
}

func NewTextDenoiser(shape Shape, weights *TextWeights) (*TextDenoiser, error) {
	return NewTextDenoiserWithDispatcher(shape, weights, NotImplementedDispatcher{})
}

func NewTextDenoiserWithDispatcher(shape Shape, weights *TextWeights, dispatcher ForwardDispatcher) (*TextDenoiser, error) {
	if weights == nil {
		return nil, fmt.Errorf("nil DiffusionGemma text weights")
	}
	plan := weights.ForwardPlan()
	if !plan.Ready {
		return nil, fmt.Errorf("DiffusionGemma text forward plan incomplete: %v", plan.Missing)
	}
	if shape.CanvasLength <= 0 || shape.VocabSize <= 0 || shape.TextHiddenSize <= 0 || shape.TextLayers <= 0 {
		return nil, fmt.Errorf("invalid DiffusionGemma shape for denoiser")
	}
	if len(plan.Layers) != shape.TextLayers {
		return nil, fmt.Errorf("DiffusionGemma text layer binding count=%d want %d", len(plan.Layers), shape.TextLayers)
	}
	if dispatcher == nil {
		dispatcher = NotImplementedDispatcher{}
	}
	ops := BuildForwardOpPlan(shape, &plan)
	if !ops.Ready {
		return nil, fmt.Errorf("DiffusionGemma text op plan not ready: %s", ops.Reason)
	}
	return &TextDenoiser{Shape: shape, Weights: weights, Plan: plan, Ops: ops, Buffers: BuildForwardBufferPlan(shape), Dispatcher: dispatcher}, nil
}

type PromptKVAppendOpportunity struct {
	PrefixTokens int `json:"prefix_tokens"`
	SuffixTokens int `json:"suffix_tokens"`
	NewTokens    int `json:"new_tokens"`
}

func promptAppendOpportunity(oldPrompt, newPrompt []int) (PromptKVAppendOpportunity, bool) {
	if len(oldPrompt) == 0 || len(newPrompt) <= len(oldPrompt) {
		return PromptKVAppendOpportunity{}, false
	}
	if !slices.Equal(oldPrompt, newPrompt[:len(oldPrompt)]) {
		return PromptKVAppendOpportunity{}, false
	}
	return PromptKVAppendOpportunity{PrefixTokens: len(oldPrompt), SuffixTokens: len(newPrompt) - len(oldPrompt), NewTokens: len(newPrompt)}, true
}

func promptAppendSuffix(oldPrompt, newPrompt []int) ([]int, bool) {
	op, ok := promptAppendOpportunity(oldPrompt, newPrompt)
	if !ok {
		return nil, false
	}
	return newPrompt[op.PrefixTokens:], true
}

func diffusionGemmaRequireIncrementalPromptKV() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_REQUIRE_INCREMENTAL_KV")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func diffusionGemmaIncrementalPromptKVEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_DISABLE_INCREMENTAL_KV")))
	if v == "1" || v == "true" || v == "yes" || v == "on" {
		return false
	}
	// GGUF prompt-KV append now matches full re-encode in local and CLI parity
	// checks, so keep it enabled by default for GGUF paths. Non-GGUF paths still
	// fall back to full re-encode because they do not have GGUFExpertIndex.
	return true
}

func (d *TextDenoiser) Denoise(in ForwardInput) (ForwardOutput, error) {
	if d == nil {
		return ForwardOutput{}, fmt.Errorf("nil DiffusionGemma text denoiser")
	}
	if len(in.Canvas) == 0 {
		return ForwardOutput{}, fmt.Errorf("empty DiffusionGemma canvas")
	}
	if d.Shape.VocabSize <= 0 {
		return ForwardOutput{}, fmt.Errorf("invalid DiffusionGemma vocab size %d", d.Shape.VocabSize)
	}
	if d.Dispatcher == nil {
		return ForwardOutput{}, fmt.Errorf("DiffusionGemma text forward dispatcher is not configured")
	}
	// Encode prompt on first call, and re-encode whenever the prompt token IDs
	// change. Block-autoregressive generation appends prior canvases to the
	// prompt context, and external callers may reuse a denoiser with a different
	// prompt of the same length; both cases must not reuse stale prefix KV.
	if len(in.PromptIDs) == 0 {
		d.EncoderKV = nil
		d.EncoderPromptIDs = nil
		d.EncoderPromptLen = 0
	} else if d.EncoderKV == nil || d.EncoderPromptLen != len(in.PromptIDs) || !slices.Equal(d.EncoderPromptIDs, in.PromptIDs) {
		t0 := time.Now()
		var kv []EncoderKVLayer
		var err error
		switch disp := d.Dispatcher.(type) {
		case CPUDispatcher:
			kv, err = disp.EncodePrompt(in.PromptIDs, d.Weights, d.Ops, d.Buffers)
		case GPUDispatcher:
			kv, err = disp.EncodePrompt(in.PromptIDs, d.Weights, d.Ops, d.Buffers)
		default:
			return ForwardOutput{}, fmt.Errorf("DiffusionGemma prompt prefill requires a GPU backend implementation")
		}
		if err != nil {
			return ForwardOutput{}, fmt.Errorf("DiffusionGemma encoder: %w", err)
		}
		d.EncoderKV = kv
		d.EncoderPromptIDs = append(d.EncoderPromptIDs[:0], in.PromptIDs...)
		d.EncoderPromptLen = len(in.PromptIDs)
		log.Printf("encoder: %d tokens → prompt KV in %.1fs", len(in.PromptIDs), time.Since(t0).Seconds())
	}
	encoderSeqLen := 0
	if len(d.EncoderKV) > 0 {
		encoderSeqLen = d.EncoderKV[0].SeqLen
	}
	return d.Dispatcher.RunTextForward(ForwardContext{PromptIDs: in.PromptIDs, Canvas: in.Canvas, Step: in.Step, SelfConditioning: in.SelfConditioning, SelfConditioningLogits: in.SelfConditioningLogits, DeviceSelfConditioning: in.DeviceSelfConditioning, SCTempInv: in.SCTempInv, SampleDraws: in.SampleDraws, EncoderKV: d.EncoderKV, EncoderSeqLen: encoderSeqLen}, d.Weights, d.Ops, d.Buffers)
}

// ForwardBufferPlan describes the major scratch buffers required by a future
// CPU/SIMD text denoiser forward pass. Sizes are element counts, not bytes.
type ForwardBufferPlan struct {
	CanvasLength  int `json:"canvas_length"`
	HiddenSize    int `json:"hidden_size"`
	VocabSize     int `json:"vocab_size"`
	SlidingWindow int `json:"sliding_window"`
	Hidden        int `json:"hidden"`
	Residual      int `json:"residual"`
	Logits        int `json:"logits"`
	Router        int `json:"router"`
	Experts       int `json:"experts"`
	TopKExperts   int `json:"top_k_experts"`
}

func BuildForwardBufferPlan(shape Shape) ForwardBufferPlan {
	canvas := shape.CanvasLength
	if canvas < 0 {
		canvas = 0
	}
	hidden := shape.TextHiddenSize
	if hidden < 0 {
		hidden = 0
	}
	vocab := shape.VocabSize
	if vocab < 0 {
		vocab = 0
	}
	hiddenElems := checkedMulOrZero(canvas, hidden)
	logitElems := checkedMulOrZero(canvas, vocab)
	routerElems := checkedMulOrZero(canvas, shape.NumExperts)
	expertElems := checkedMulOrZero(checkedMulOrZero(canvas, shape.TopKExperts), shape.MoEIntermediateSize)
	// Experts scratch is also used as the reusable GGUF CPU/SIMD pre-norm row
	// buffer, so size it for the larger of expert activations and full hidden rows.
	if hiddenElems > expertElems {
		expertElems = hiddenElems
	}
	return ForwardBufferPlan{
		CanvasLength:  canvas,
		HiddenSize:    hidden,
		VocabSize:     vocab,
		SlidingWindow: shape.SlidingWindow,
		Hidden:        hiddenElems,
		Residual:      hiddenElems,
		Logits:        logitElems,
		Router:        routerElems,
		Experts:       expertElems,
		TopKExperts:   shape.TopKExperts,
	}
}

func checkedMulOrZero(a, b int) int {
	v, ok := checked.MulInt(a, b)
	if !ok {
		return 0
	}
	return v
}
