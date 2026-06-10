package diffusiongemma

import (
	"encoding/binary"
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

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
	return ForwardScratch{Hidden: make([]float32, maxNonNegative(buffers.Hidden)), Residual: make([]float32, maxNonNegative(buffers.Residual)), Router: make([]float32, maxNonNegative(buffers.Router)), Experts: make([]float32, maxNonNegative(buffers.Experts)), Logits: makeLogitRows(buffers.CanvasLength, buffers.VocabSize)}
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
		if err := dispatchPrefixOp(op, ctx, weights, scratch); err != nil {
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

func dispatchPrefixOp(op OpKind, ctx ForwardContext, weights *TextWeights, scratch ForwardScratch) error {
	switch op {
	case OpCanvasEmbedding:
		return runCanvasEmbedding(ctx, weights, scratch)
	default:
		return fmt.Errorf("DiffusionGemma unknown prefix op %q", op)
	}
}

func runCanvasEmbedding(ctx ForwardContext, weights *TextWeights, scratch ForwardScratch) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma canvas embedding missing weights")
	}
	plan := weights.ForwardPlan()
	if plan.Globals.EmbedTokens == nil {
		return fmt.Errorf("DiffusionGemma canvas embedding missing embed_tokens")
	}
	hiddenSize := 0
	if len(plan.Globals.EmbedTokens.Shape) == 2 {
		hiddenSize = plan.Globals.EmbedTokens.Shape[1]
	}
	if hiddenSize <= 0 {
		return fmt.Errorf("DiffusionGemma canvas embedding invalid shape %v", plan.Globals.EmbedTokens.Shape)
	}
	need := len(ctx.Canvas) * hiddenSize
	if len(scratch.Hidden) < need {
		return fmt.Errorf("DiffusionGemma canvas embedding hidden scratch=%d want %d", len(scratch.Hidden), need)
	}
	for i, token := range ctx.Canvas {
		row, dtype, shape, err := weights.RawTensorRow(plan.Globals.EmbedTokens.Name, token)
		if err != nil {
			return err
		}
		if len(shape) != 1 || shape[0] != hiddenSize {
			return fmt.Errorf("DiffusionGemma embed row shape %v want [%d]", shape, hiddenSize)
		}
		if err := decodeFloatRowTo(scratch.Hidden[i*hiddenSize:(i+1)*hiddenSize], row, dtype); err != nil {
			return err
		}
	}
	return nil
}

func decodeFloatRowTo(dst []float32, raw []byte, dtype string) error {
	switch dtype {
	case "F32":
		if len(raw) < len(dst)*4 {
			return fmt.Errorf("DiffusionGemma F32 row bytes=%d want %d", len(raw), len(dst)*4)
		}
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return nil
	case "BF16":
		if len(raw) < len(dst)*2 {
			return fmt.Errorf("DiffusionGemma BF16 row bytes=%d want %d", len(raw), len(dst)*2)
		}
		for i := range dst {
			dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
		return nil
	case "F16":
		if len(raw) < len(dst)*2 {
			return fmt.Errorf("DiffusionGemma F16 row bytes=%d want %d", len(raw), len(dst)*2)
		}
		for i := range dst {
			dst[i] = diffusionGemmaF16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return nil
	default:
		return fmt.Errorf("DiffusionGemma unsupported float row dtype %s", dtype)
	}
}

func diffusionGemmaF16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	mant := uint32(h & 0x03ff)
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for mant&0x0400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &^= 0x0400
	} else if exp == 31 {
		return math.Float32frombits(sign | 0x7f800000 | (mant << 13))
	}
	exp32 := uint32(int(exp) + (127 - 15))
	return math.Float32frombits(sign | (exp32 << 23) | (mant << 13))
}

func dispatchLayerOp(op LayerOp, weights *TextWeights, scratch ForwardScratch) error {
	switch op.Kind {
	case OpInputNorm:
		return runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.InputLayerNorm })
	case OpSelfAttention:
		return errOpNotImplemented(op.Kind)
	case OpPostAttention:
		return runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.PostAttentionLayerNorm })
	case OpDenseMLP:
		return errOpNotImplemented(op.Kind)
	case OpPreMoE:
		return runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.PreFFNLayerNorm })
	case OpRouter:
		return errOpNotImplemented(op.Kind)
	case OpExperts:
		return errOpNotImplemented(op.Kind)
	case OpPostMoE:
		return runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.PostFFNLayerNorm })
	case OpLayerScalar:
		return runLayerScalar(op, weights, scratch)
	default:
		return fmt.Errorf("DiffusionGemma unknown layer op %q", op.Kind)
	}
}

func runLayerRMSNorm(op LayerOp, weights *TextWeights, scratch ForwardScratch, pick func(TextLayerBindings) *TensorBinding) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma RMSNorm missing weights")
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("DiffusionGemma RMSNorm layer %d outside plan", op.Layer)
	}
	binding := pick(fp.Layers[op.Layer])
	if binding == nil {
		return fmt.Errorf("DiffusionGemma RMSNorm missing binding for layer %d", op.Layer)
	}
	weight, err := loadFloatVector(weights, binding)
	if err != nil {
		return err
	}
	if len(weight) == 0 {
		return fmt.Errorf("DiffusionGemma RMSNorm empty weight %s", binding.Name)
	}
	hiddenSize := len(weight)
	if len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("DiffusionGemma RMSNorm hidden len=%d not divisible by %d", len(scratch.Hidden), hiddenSize)
	}
	for off := 0; off < len(scratch.Hidden); off += hiddenSize {
		if !simd.RMSNormTo(scratch.Hidden[off:off+hiddenSize], weight, 1e-6) {
			return fmt.Errorf("DiffusionGemma RMSNorm rejected row at offset %d", off)
		}
	}
	return nil
}

func loadFloatVector(weights *TextWeights, binding *TensorBinding) ([]float32, error) {
	raw, dtype, shape, err := weights.RawTensor(binding.Name)
	if err != nil {
		return nil, err
	}
	if len(shape) != 1 {
		return nil, fmt.Errorf("DiffusionGemma tensor %q shape %v is not rank-1", binding.Name, shape)
	}
	out := make([]float32, shape[0])
	if err := decodeFloatRowTo(out, raw, dtype); err != nil {
		return nil, err
	}
	return out, nil
}

func runLayerScalar(op LayerOp, weights *TextWeights, scratch ForwardScratch) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma layer scalar missing weights")
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("DiffusionGemma layer scalar layer %d outside plan", op.Layer)
	}
	binding := fp.Layers[op.Layer].LayerScalar
	if binding == nil {
		return fmt.Errorf("DiffusionGemma layer scalar missing binding for layer %d", op.Layer)
	}
	scale, err := loadFloatScalar(weights, binding)
	if err != nil {
		return err
	}
	for i := range scratch.Hidden {
		scratch.Hidden[i] *= scale
	}
	return nil
}

func loadFloatScalar(weights *TextWeights, binding *TensorBinding) (float32, error) {
	raw, dtype, shape, err := weights.RawTensor(binding.Name)
	if err != nil {
		return 0, err
	}
	n := 1
	if len(shape) > 0 {
		n = 1
		for _, dim := range shape {
			if dim <= 0 {
				return 0, fmt.Errorf("DiffusionGemma tensor %q invalid scalar shape %v", binding.Name, shape)
			}
			n *= dim
		}
	}
	if n != 1 {
		return 0, fmt.Errorf("DiffusionGemma tensor %q shape %v is not scalar", binding.Name, shape)
	}
	var out [1]float32
	if err := decodeFloatRowTo(out[:], raw, dtype); err != nil {
		return 0, err
	}
	return out[0], nil
}

func dispatchTailOp(op OpKind, weights *TextWeights, scratch ForwardScratch) error {
	switch op {
	case OpSelfCondition:
		return errOpNotImplemented(op)
	case OpFinalNorm:
		return runFinalNorm(weights, scratch)
	case OpLMHead:
		return errOpNotImplemented(op)
	default:
		return fmt.Errorf("DiffusionGemma unknown tail op %q", op)
	}
}

func runFinalNorm(weights *TextWeights, scratch ForwardScratch) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma final norm missing weights")
	}
	fp := weights.ForwardPlan()
	if fp.Globals.FinalNorm == nil {
		return fmt.Errorf("DiffusionGemma final norm missing binding")
	}
	weight, err := loadFloatVector(weights, fp.Globals.FinalNorm)
	if err != nil {
		return err
	}
	if len(weight) == 0 {
		return fmt.Errorf("DiffusionGemma final norm empty weight")
	}
	hiddenSize := len(weight)
	if len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("DiffusionGemma final norm hidden len=%d not divisible by %d", len(scratch.Hidden), hiddenSize)
	}
	for off := 0; off < len(scratch.Hidden); off += hiddenSize {
		if !simd.RMSNormTo(scratch.Hidden[off:off+hiddenSize], weight, 1e-6) {
			return fmt.Errorf("DiffusionGemma final norm rejected row at offset %d", off)
		}
	}
	return nil
}

func errOpNotImplemented(op OpKind) error {
	return fmt.Errorf("DiffusionGemma CPU/SIMD op %s is not implemented", op)
}
