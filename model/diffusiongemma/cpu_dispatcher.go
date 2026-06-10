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
	TopKIDs  []int
	TopKVals []float32
}

func NewForwardScratch(buffers ForwardBufferPlan) ForwardScratch {
	topKSlots := maxNonNegative(buffers.CanvasLength * buffers.TopKExperts)
	return ForwardScratch{Hidden: make([]float32, maxNonNegative(buffers.Hidden)), Residual: make([]float32, maxNonNegative(buffers.Residual)), Router: make([]float32, maxNonNegative(buffers.Router)), Experts: make([]float32, maxNonNegative(buffers.Experts)), TopKIDs: make([]int, topKSlots), TopKVals: make([]float32, topKSlots), Logits: makeLogitRows(buffers.CanvasLength, buffers.VocabSize)}
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
		return runDenseMLP(op, weights, scratch)
	case OpPreMoE:
		return runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.PreFFNLayerNorm })
	case OpRouter:
		return runRouter(op, weights, scratch)
	case OpExperts:
		return runExperts(op, weights, scratch)
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

func runDenseMLP(op LayerOp, weights *TextWeights, scratch ForwardScratch) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma dense MLP missing weights")
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("DiffusionGemma dense MLP layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	gateW, gateRows, gateCols, err := loadFloatMatrix(weights, lb.MLPGateProj)
	if err != nil {
		return err
	}
	upW, upRows, upCols, err := loadFloatMatrix(weights, lb.MLPUpProj)
	if err != nil {
		return err
	}
	downW, downRows, downCols, err := loadFloatMatrix(weights, lb.MLPDownProj)
	if err != nil {
		return err
	}
	if gateRows != upRows || gateCols != upCols || downCols != gateRows || downRows != gateCols {
		return fmt.Errorf("DiffusionGemma dense MLP shape mismatch gate=[%d,%d] up=[%d,%d] down=[%d,%d]", gateRows, gateCols, upRows, upCols, downRows, downCols)
	}
	hiddenSize := gateCols
	intermediate := gateRows
	if hiddenSize <= 0 || intermediate <= 0 || len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("DiffusionGemma dense MLP hidden len=%d hidden_size=%d intermediate=%d", len(scratch.Hidden), hiddenSize, intermediate)
	}
	gate := make([]float32, intermediate)
	up := make([]float32, intermediate)
	act := make([]float32, intermediate)
	out := make([]float32, hiddenSize)
	for off := 0; off < len(scratch.Hidden); off += hiddenSize {
		row := scratch.Hidden[off : off+hiddenSize]
		if !simd.GemvRows(gate, row, gateW, intermediate, hiddenSize) {
			return fmt.Errorf("DiffusionGemma dense MLP gate GEMV rejected layer %d", op.Layer)
		}
		if !simd.GemvRows(up, row, upW, intermediate, hiddenSize) {
			return fmt.Errorf("DiffusionGemma dense MLP up GEMV rejected layer %d", op.Layer)
		}
		if !simd.GELUTanhMulTo(act, gate, up) {
			return fmt.Errorf("DiffusionGemma dense MLP activation rejected layer %d", op.Layer)
		}
		if !simd.GemvRows(out, act, downW, hiddenSize, intermediate) {
			return fmt.Errorf("DiffusionGemma dense MLP down GEMV rejected layer %d", op.Layer)
		}
		copy(row, out)
	}
	return nil
}

func runRouter(op LayerOp, weights *TextWeights, scratch ForwardScratch) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma router missing weights")
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("DiffusionGemma router layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	proj, rows, cols, err := loadFloatMatrix(weights, lb.RouterProj)
	if err != nil {
		return err
	}
	hiddenSize := cols
	experts := rows
	if hiddenSize <= 0 || experts <= 0 || len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("DiffusionGemma router hidden len=%d hidden_size=%d experts=%d", len(scratch.Hidden), hiddenSize, experts)
	}
	positions := len(scratch.Hidden) / hiddenSize
	if len(scratch.Router) < positions*experts {
		return fmt.Errorf("DiffusionGemma router scratch=%d want %d", len(scratch.Router), positions*experts)
	}
	routerScale, err := loadOptionalScalar(weights, lb.RouterScale, 1)
	if err != nil {
		return err
	}
	perExpertScale, err := loadOptionalVector(weights, lb.RouterPerExpertScale, experts)
	if err != nil {
		return err
	}
	for pos := 0; pos < positions; pos++ {
		hidden := scratch.Hidden[pos*hiddenSize : (pos+1)*hiddenSize]
		out := scratch.Router[pos*experts : (pos+1)*experts]
		if !simd.GemvRows(out, hidden, proj, experts, hiddenSize) {
			return fmt.Errorf("DiffusionGemma router GEMV rejected layer %d", op.Layer)
		}
		for i := range out {
			out[i] *= routerScale
			if len(perExpertScale) == experts {
				out[i] *= perExpertScale[i]
			}
		}
		topK := 0
		if positions > 0 {
			topK = len(scratch.TopKIDs) / positions
		}
		if topK > 0 && len(scratch.TopKVals) >= (pos+1)*topK {
			selectTopK(out, scratch.TopKIDs[pos*topK:(pos+1)*topK], scratch.TopKVals[pos*topK:(pos+1)*topK])
		}
	}
	return nil
}

func selectTopK(scores []float32, ids []int, vals []float32) {
	for i := range ids {
		ids[i] = -1
		vals[i] = float32(math.Inf(-1))
	}
	for id, score := range scores {
		for slot := range ids {
			if score > vals[slot] {
				copy(vals[slot+1:], vals[slot:len(vals)-1])
				copy(ids[slot+1:], ids[slot:len(ids)-1])
				vals[slot] = score
				ids[slot] = id
				break
			}
		}
	}
}

func runExperts(op LayerOp, weights *TextWeights, scratch ForwardScratch) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma experts missing weights")
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("DiffusionGemma experts layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	gateUp, experts, gateUpRows, hiddenSize, err := loadFloat3D(weights, lb.ExpertsGateUpProj)
	if err != nil {
		return err
	}
	down, downExperts, downRows, downCols, err := loadFloat3D(weights, lb.ExpertsDownProj)
	if err != nil {
		return err
	}
	if experts != downExperts || downRows != hiddenSize || gateUpRows%2 != 0 || downCols != gateUpRows/2 {
		return fmt.Errorf("DiffusionGemma expert shape mismatch gate_up=[%d,%d,%d] down=[%d,%d,%d]", experts, gateUpRows, hiddenSize, downExperts, downRows, downCols)
	}
	intermediate := gateUpRows / 2
	positions := 0
	if hiddenSize > 0 {
		positions = len(scratch.Hidden) / hiddenSize
	}
	if positions <= 0 || len(scratch.TopKIDs) < positions || len(scratch.TopKVals) < positions {
		return fmt.Errorf("DiffusionGemma expert top-k scratch invalid positions=%d ids=%d vals=%d", positions, len(scratch.TopKIDs), len(scratch.TopKVals))
	}
	topK := len(scratch.TopKIDs) / positions
	if topK <= 0 {
		return fmt.Errorf("DiffusionGemma expert top-k is zero")
	}
	gate := make([]float32, intermediate)
	up := make([]float32, intermediate)
	act := make([]float32, intermediate)
	tmp := make([]float32, hiddenSize)
	out := make([]float32, hiddenSize)
	for pos := 0; pos < positions; pos++ {
		hidden := scratch.Hidden[pos*hiddenSize : (pos+1)*hiddenSize]
		for i := range out {
			out[i] = 0
		}
		for slot := 0; slot < topK; slot++ {
			expertID := scratch.TopKIDs[pos*topK+slot]
			if expertID < 0 || expertID >= experts {
				continue
			}
			weight := scratch.TopKVals[pos*topK+slot]
			base := expertID * gateUpRows * hiddenSize
			gateW := gateUp[base : base+intermediate*hiddenSize]
			upW := gateUp[base+intermediate*hiddenSize : base+gateUpRows*hiddenSize]
			if !simd.GemvRows(gate, hidden, gateW, intermediate, hiddenSize) {
				return fmt.Errorf("DiffusionGemma expert gate GEMV rejected layer %d", op.Layer)
			}
			if !simd.GemvRows(up, hidden, upW, intermediate, hiddenSize) {
				return fmt.Errorf("DiffusionGemma expert up GEMV rejected layer %d", op.Layer)
			}
			if !simd.GELUTanhMulTo(act, gate, up) {
				return fmt.Errorf("DiffusionGemma expert activation rejected layer %d", op.Layer)
			}
			downBase := expertID * downRows * downCols
			if !simd.GemvRows(tmp, act, down[downBase:downBase+downRows*downCols], hiddenSize, intermediate) {
				return fmt.Errorf("DiffusionGemma expert down GEMV rejected layer %d", op.Layer)
			}
			for i := range out {
				out[i] += weight * tmp[i]
			}
		}
		copy(hidden, out)
	}
	return nil
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

func loadFloat3D(weights *TextWeights, binding *TensorBinding) ([]float32, int, int, int, error) {
	if binding == nil {
		return nil, 0, 0, 0, fmt.Errorf("DiffusionGemma missing 3D tensor binding")
	}
	raw, dtype, shape, err := weights.RawTensor(binding.Name)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	if len(shape) != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 {
		return nil, 0, 0, 0, fmt.Errorf("DiffusionGemma tensor %q shape %v is not rank-3", binding.Name, shape)
	}
	n := shape[0] * shape[1] * shape[2]
	out := make([]float32, n)
	if err := decodeFloatRowTo(out, raw, dtype); err != nil {
		return nil, 0, 0, 0, err
	}
	return out, shape[0], shape[1], shape[2], nil
}

func loadFloatMatrix(weights *TextWeights, binding *TensorBinding) ([]float32, int, int, error) {
	if binding == nil {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma missing matrix binding")
	}
	raw, dtype, shape, err := weights.RawTensor(binding.Name)
	if err != nil {
		return nil, 0, 0, err
	}
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma tensor %q shape %v is not rank-2", binding.Name, shape)
	}
	n := shape[0] * shape[1]
	out := make([]float32, n)
	if err := decodeFloatRowTo(out, raw, dtype); err != nil {
		return nil, 0, 0, err
	}
	return out, shape[0], shape[1], nil
}

func loadOptionalScalar(weights *TextWeights, binding *TensorBinding, fallback float32) (float32, error) {
	if binding == nil {
		return fallback, nil
	}
	return loadFloatScalar(weights, binding)
}

func loadOptionalVector(weights *TextWeights, binding *TensorBinding, want int) ([]float32, error) {
	if binding == nil {
		return nil, nil
	}
	v, err := loadFloatVector(weights, binding)
	if err != nil {
		return nil, err
	}
	if want > 0 && len(v) != want {
		return nil, fmt.Errorf("DiffusionGemma tensor %q vector len=%d want %d", binding.Name, len(v), want)
	}
	return v, nil
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
