package diffusiongemma

import "fmt"

// CPUDispatcher is the native CPU/SIMD forward scaffold for DiffusionGemma text
// denoising. It gives each semantic op an explicit hook so implementation can
// proceed operation-by-operation while keeping the Denoiser interface stable.
type CPUDispatcher struct{}

type ForwardScratch struct {
	Hidden   []float32
	Residual []float32
	Logits   [][]float32
	Router   []float32
	Experts  []float32
}

func NewForwardScratch(buffers ForwardBufferPlan) ForwardScratch {
	return ForwardScratch{
		Hidden:   make([]float32, maxNonNegative(buffers.Hidden)),
		Residual: make([]float32, maxNonNegative(buffers.Residual)),
		Router:   make([]float32, maxNonNegative(buffers.Router)),
		Experts:  make([]float32, maxNonNegative(buffers.Experts)),
		Logits:   makeLogitRows(buffers.CanvasLength, buffers.VocabSize),
	}
}

func makeLogitRows(rows, cols int) [][]float32 {
	if rows <= 0 || cols <= 0 {
		return nil
	}
	flat := make([]float32, rows*cols)
	out := make([][]float32, rows)
	for i := range out {
		out[i] = flat[i*cols : (i+1)*cols]
	}
	return out
}

func maxNonNegative(n int) int {
	if n < 0 {
		return 0
	}
	return n
}

func (CPUDispatcher) RunTextForward(ctx ForwardContext, weights *TextWeights, ops ForwardOpPlan, buffers ForwardBufferPlan) (ForwardOutput, error) {
	if weights == nil {
		return ForwardOutput{}, fmt.Errorf("DiffusionGemma CPU dispatcher missing text weights")
	}
	if !ops.Ready {
		return ForwardOutput{}, fmt.Errorf("DiffusionGemma CPU dispatcher op plan not ready: %s", ops.Reason)
	}
	if len(ctx.Canvas) == 0 {
		return ForwardOutput{}, fmt.Errorf("DiffusionGemma CPU dispatcher empty canvas")
	}
	scratch := NewForwardScratch(buffers)
	for _, op := range ops.Prefix {
		if err := dispatchPrefixOp(op, weights, scratch); err != nil {
			return ForwardOutput{}, err
		}
	}
	for _, op := range ops.Layers {
		if err := dispatchLayerOp(op, weights, scratch); err != nil {
			return ForwardOutput{}, err
		}
	}
	for _, op := range ops.Tail {
		if err := dispatchTailOp(op, weights, scratch); err != nil {
			return ForwardOutput{}, err
		}
	}
	return ForwardOutput{Logits: scratch.Logits}, nil
}

func dispatchPrefixOp(op OpKind, weights *TextWeights, scratch ForwardScratch) error {
	switch op {
	case OpCanvasEmbedding:
		return errOpNotImplemented(op)
	default:
		return fmt.Errorf("DiffusionGemma unknown prefix op %q", op)
	}
}

func dispatchLayerOp(op LayerOp, weights *TextWeights, scratch ForwardScratch) error {
	switch op.Kind {
	case OpInputNorm:
		return errOpNotImplemented(op.Kind)
	case OpSelfAttention:
		return errOpNotImplemented(op.Kind)
	case OpPostAttention:
		return errOpNotImplemented(op.Kind)
	case OpDenseMLP:
		return errOpNotImplemented(op.Kind)
	case OpPreMoE:
		return errOpNotImplemented(op.Kind)
	case OpRouter:
		return errOpNotImplemented(op.Kind)
	case OpExperts:
		return errOpNotImplemented(op.Kind)
	case OpPostMoE:
		return errOpNotImplemented(op.Kind)
	case OpLayerScalar:
		return errOpNotImplemented(op.Kind)
	default:
		return fmt.Errorf("DiffusionGemma unknown layer op %q", op.Kind)
	}
}

func dispatchTailOp(op OpKind, weights *TextWeights, scratch ForwardScratch) error {
	switch op {
	case OpSelfCondition:
		return errOpNotImplemented(op)
	case OpFinalNorm:
		return errOpNotImplemented(op)
	case OpLMHead:
		return errOpNotImplemented(op)
	default:
		return fmt.Errorf("DiffusionGemma unknown tail op %q", op)
	}
}

func errOpNotImplemented(op OpKind) error {
	return fmt.Errorf("DiffusionGemma CPU/SIMD op %s is not implemented", op)
}
