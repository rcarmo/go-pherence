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
		if err := dispatchLayerOp(op, ctx, weights, scratch); err != nil {
			return ForwardOutput{}, err
		}
	}
	for _, op := range ops.Tail {
		if err := dispatchTailOp(op, weights, scratch); err != nil {
			return ForwardOutput{}, err
		}
	}
	selfConditioning, err := buildSelfConditioningFromLogits(weights, scratch)
	if err != nil {
		return ForwardOutput{}, err
	}
	return ForwardOutput{Logits: scratch.Logits, SelfConditioning: selfConditioning}, nil
}

func dispatchPrefixOp(op OpKind, ctx ForwardContext, weights *TextWeights, scratch ForwardScratch) error {
	switch op {
	case OpCanvasEmbedding:
		return runCanvasEmbedding(ctx, weights, scratch)
	case OpSelfCondition:
		return runSelfCondition(ctx, weights, scratch)
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

func runSelfCondition(ctx ForwardContext, weights *TextWeights, scratch ForwardScratch) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma self-conditioning missing weights")
	}
	fp := weights.ForwardPlan()
	hiddenSize := 0
	if fp.Globals.EmbedTokens != nil && len(fp.Globals.EmbedTokens.Shape) == 2 {
		hiddenSize = fp.Globals.EmbedTokens.Shape[1]
	}
	if hiddenSize <= 0 || len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("DiffusionGemma self-conditioning hidden len=%d hidden_size=%d", len(scratch.Hidden), hiddenSize)
	}
	if len(ctx.SelfConditioning) == 0 {
		for off := 0; off < len(scratch.Hidden); off += hiddenSize {
			if !simd.RMSNormNoScaleTo(scratch.Hidden[off:off+hiddenSize], 1e-6) {
				return fmt.Errorf("DiffusionGemma self-conditioning post norm rejected row at offset %d", off)
			}
		}
		return nil
	}
	if len(ctx.SelfConditioning) != len(scratch.Hidden) {
		return fmt.Errorf("DiffusionGemma self-conditioning len=%d want %d", len(ctx.SelfConditioning), len(scratch.Hidden))
	}
	preNorm, err := loadFloatVector(weights, fp.Globals.SelfCondPreNorm)
	if err != nil {
		return err
	}
	gateW, gateRows, gateCols, err := loadFloatMatrix(weights, fp.Globals.SelfCondGateProj)
	if err != nil {
		return err
	}
	upW, upRows, upCols, err := loadFloatMatrix(weights, fp.Globals.SelfCondUpProj)
	if err != nil {
		return err
	}
	downW, downRows, downCols, err := loadFloatMatrix(weights, fp.Globals.SelfCondDownProj)
	if err != nil {
		return err
	}
	if len(preNorm) != hiddenSize || gateCols != hiddenSize || upCols != hiddenSize || gateRows != upRows || downRows != hiddenSize || downCols != gateRows {
		return fmt.Errorf("DiffusionGemma self-conditioning shape mismatch")
	}
	intermediate := gateRows
	cond := make([]float32, hiddenSize)
	gate := make([]float32, intermediate)
	up := make([]float32, intermediate)
	act := make([]float32, intermediate)
	signal := make([]float32, hiddenSize)
	for off := 0; off < len(scratch.Hidden); off += hiddenSize {
		copy(cond, ctx.SelfConditioning[off:off+hiddenSize])
		if !simd.RMSNormTo(cond, preNorm, 1e-6) {
			return fmt.Errorf("DiffusionGemma self-conditioning pre norm rejected row at offset %d", off)
		}
		if !simd.GemvRows(gate, cond, gateW, intermediate, hiddenSize) || !simd.GemvRows(up, cond, upW, intermediate, hiddenSize) {
			return fmt.Errorf("DiffusionGemma self-conditioning gate/up GEMV rejected")
		}
		if !simd.GELUTanhMulTo(act, gate, up) {
			return fmt.Errorf("DiffusionGemma self-conditioning activation rejected")
		}
		if !simd.GemvRows(signal, act, downW, hiddenSize, intermediate) {
			return fmt.Errorf("DiffusionGemma self-conditioning down GEMV rejected")
		}
		row := scratch.Hidden[off : off+hiddenSize]
		for i := range row {
			row[i] += signal[i]
		}
		if !simd.RMSNormNoScaleTo(row, 1e-6) {
			return fmt.Errorf("DiffusionGemma self-conditioning post norm rejected row at offset %d", off)
		}
	}
	return nil
}

func dispatchLayerOp(op LayerOp, ctx ForwardContext, weights *TextWeights, scratch ForwardScratch) error {
	switch op.Kind {
	case OpInputNorm:
		return runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.InputLayerNorm })
	case OpSelfAttention:
		return runSelfAttention(op, ctx, weights, scratch)
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

func runSelfAttention(op LayerOp, ctx ForwardContext, weights *TextWeights, scratch ForwardScratch) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma self-attention missing weights")
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("DiffusionGemma self-attention layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	qW, qRows, hiddenSize, err := loadFloatMatrix(weights, lb.QProj)
	if err != nil {
		return err
	}
	kW, kRows, kCols, err := loadFloatMatrix(weights, lb.KProj)
	if err != nil {
		return err
	}
	if kCols != hiddenSize {
		return fmt.Errorf("DiffusionGemma attention K shape [%d,%d] hidden=%d", kRows, kCols, hiddenSize)
	}
	var vW []float32
	vRows, vCols := kRows, kCols
	if lb.VProj != nil {
		vW, vRows, vCols, err = loadFloatMatrix(weights, lb.VProj)
		if err != nil {
			return err
		}
		if vCols != hiddenSize {
			return fmt.Errorf("DiffusionGemma attention V shape [%d,%d] hidden=%d", vRows, vCols, hiddenSize)
		}
	}
	oW, oRows, oCols, err := loadFloatMatrix(weights, lb.OProj)
	if err != nil {
		return err
	}
	if oRows != hiddenSize || oCols != qRows {
		return fmt.Errorf("DiffusionGemma attention O shape [%d,%d] q=%d hidden=%d", oRows, oCols, qRows, hiddenSize)
	}
	qNorm, err := loadFloatVector(weights, lb.QNorm)
	if err != nil {
		return err
	}
	kNorm, err := loadFloatVector(weights, lb.KNorm)
	if err != nil {
		return err
	}
	headDim := len(qNorm)
	if headDim <= 0 || len(kNorm) != headDim || qRows%headDim != 0 || kRows%headDim != 0 || vRows%headDim != 0 {
		return fmt.Errorf("DiffusionGemma attention invalid head dims q=%d k=%d v=%d head=%d", qRows, kRows, vRows, headDim)
	}
	heads := qRows / headDim
	kvHeads := kRows / headDim
	if heads <= 0 || kvHeads <= 0 || heads%kvHeads != 0 || len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("DiffusionGemma attention invalid heads=%d kv_heads=%d hidden_len=%d hidden=%d", heads, kvHeads, len(scratch.Hidden), hiddenSize)
	}
	positions := len(scratch.Hidden) / hiddenSize
	qAll := make([]float32, positions*qRows)
	kAll := make([]float32, positions*kRows)
	vAll := make([]float32, positions*vRows)
	attnCtx := make([]float32, qRows)
	out := make([]float32, hiddenSize)
	ropeHalf := headDim / 2
	ropeTheta := 10000.0
	if op.Type == "full_attention" {
		ropeHalf = headDim / 8
		ropeTheta = 1000000.0
	}
	ropeFreqs := simd.BuildRoPEFreqs(positions, ropeHalf, headDim, ropeTheta)
	for pos := 0; pos < positions; pos++ {
		hidden := scratch.Hidden[pos*hiddenSize : (pos+1)*hiddenSize]
		q := qAll[pos*qRows : (pos+1)*qRows]
		k := kAll[pos*kRows : (pos+1)*kRows]
		v := vAll[pos*vRows : (pos+1)*vRows]
		if !simd.GemvRows(q, hidden, qW, qRows, hiddenSize) || !simd.GemvRows(k, hidden, kW, kRows, hiddenSize) {
			return fmt.Errorf("DiffusionGemma attention Q/K GEMV rejected layer %d", op.Layer)
		}
		if lb.VProj != nil {
			if !simd.GemvRows(v, hidden, vW, vRows, hiddenSize) {
				return fmt.Errorf("DiffusionGemma attention V GEMV rejected layer %d", op.Layer)
			}
		} else {
			copy(v, k)
		}
		for h := 0; h < heads; h++ {
			if !simd.RMSNormTo(q[h*headDim:(h+1)*headDim], qNorm, 1e-6) {
				return fmt.Errorf("DiffusionGemma attention q_norm rejected")
			}
		}
		for h := 0; h < kvHeads; h++ {
			if !simd.RMSNormTo(k[h*headDim:(h+1)*headDim], kNorm, 1e-6) {
				return fmt.Errorf("DiffusionGemma attention k_norm rejected")
			}
			if !simd.RMSNormNoScaleTo(v[h*headDim:(h+1)*headDim], 1e-6) {
				return fmt.Errorf("DiffusionGemma attention v_norm rejected")
			}
		}
		if len(ropeFreqs) > 0 && ropeHalf > 0 {
			simd.ApplyRoPEPartial(q, ropeFreqs, pos, heads, headDim, ropeHalf)
			simd.ApplyRoPEPartial(k, ropeFreqs, pos, kvHeads, headDim, ropeHalf)
		}
	}
	group := heads / kvHeads
	enc := EncoderKVLayer{}
	if op.Layer >= 0 && op.Layer < len(ctx.EncoderKV) {
		enc = ctx.EncoderKV[op.Layer]
	}
	encSeq := 0
	if enc.SeqLen > 0 {
		if enc.KVHeads != kvHeads || enc.HeadDim != headDim || len(enc.Keys) < enc.SeqLen*kRows || len(enc.Values) < enc.SeqLen*vRows {
			return fmt.Errorf("DiffusionGemma encoder KV layer %d shape mismatch seq=%d kv_heads=%d head_dim=%d", op.Layer, enc.SeqLen, enc.KVHeads, enc.HeadDim)
		}
		encSeq = enc.SeqLen
	}
	totalKV := encSeq + positions
	scores := make([]float32, totalKV)
	slidingWindow := 0
	if op.Type == "sliding_attention" {
		slidingWindow = 1024
	}
	for pos := 0; pos < positions; pos++ {
		for i := range attnCtx {
			attnCtx[i] = 0
		}
		for h := 0; h < heads; h++ {
			kvh := h / group
			q := qAll[pos*qRows+h*headDim : pos*qRows+(h+1)*headDim]
			for j := 0; j < totalKV; j++ {
				if j < encSeq {
					scores[j] = dot(q, enc.Keys[j*kRows+kvh*headDim:j*kRows+(kvh+1)*headDim])
					continue
				}
				canvasJ := j - encSeq
				if slidingWindow > 0 && absInt(pos-canvasJ) >= slidingWindow {
					scores[j] = float32(math.Inf(-1))
					continue
				}
				scores[j] = dot(q, kAll[canvasJ*kRows+kvh*headDim:canvasJ*kRows+(kvh+1)*headDim])
			}
			softmaxInPlace(scores)
			dst := attnCtx[h*headDim : (h+1)*headDim]
			for j, score := range scores {
				var vv []float32
				if j < encSeq {
					vv = enc.Values[j*vRows+kvh*headDim : j*vRows+(kvh+1)*headDim]
				} else {
					canvasJ := j - encSeq
					vv = vAll[canvasJ*vRows+kvh*headDim : canvasJ*vRows+(kvh+1)*headDim]
				}
				for d := range dst {
					dst[d] += score * vv[d]
				}
			}
		}
		if !simd.GemvRows(out, attnCtx, oW, hiddenSize, qRows) {
			return fmt.Errorf("DiffusionGemma attention O GEMV rejected layer %d", op.Layer)
		}
		copy(scratch.Hidden[pos*hiddenSize:(pos+1)*hiddenSize], out)
	}
	return nil
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
func softmaxInPlace(x []float32) {
	if len(x) == 0 {
		return
	}
	m := x[0]
	for _, v := range x[1:] {
		if v > m {
			m = v
		}
	}
	var sum float64
	for i, v := range x {
		e := math.Exp(float64(v - m))
		x[i] = float32(e)
		sum += e
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / sum)
	for i := range x {
		x[i] *= inv
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
	routerScale, err := loadOptionalVector(weights, lb.RouterScale, hiddenSize)
	if err != nil {
		return err
	}
	perExpertScale, err := loadOptionalVector(weights, lb.RouterPerExpertScale, experts)
	if err != nil {
		return err
	}
	routerInput := make([]float32, hiddenSize)
	for pos := 0; pos < positions; pos++ {
		hidden := scratch.Hidden[pos*hiddenSize : (pos+1)*hiddenSize]
		copy(routerInput, hidden)
		if len(routerScale) == hiddenSize {
			for i := range routerInput {
				routerInput[i] *= routerScale[i]
			}
		}
		out := scratch.Router[pos*experts : (pos+1)*experts]
		if !simd.GemvRows(out, routerInput, proj, experts, hiddenSize) {
			return fmt.Errorf("DiffusionGemma router GEMV rejected layer %d", op.Layer)
		}
		for i := range out {
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
		return runLMHead(weights, scratch)
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

func buildSelfConditioningFromLogits(weights *TextWeights, scratch ForwardScratch) ([]float32, error) {
	if weights == nil || len(scratch.Logits) == 0 {
		return nil, nil
	}
	fp := weights.ForwardPlan()
	if fp.Globals.EmbedTokens == nil || len(fp.Globals.EmbedTokens.Shape) != 2 {
		return nil, nil
	}
	vocab := fp.Globals.EmbedTokens.Shape[0]
	hiddenSize := fp.Globals.EmbedTokens.Shape[1]
	if vocab <= 0 || hiddenSize <= 0 {
		return nil, nil
	}
	positions := len(scratch.Logits)
	out := make([]float32, positions*hiddenSize)
	row := make([]float32, hiddenSize)
	for pos := 0; pos < positions; pos++ {
		if len(scratch.Logits[pos]) < vocab {
			return nil, fmt.Errorf("DiffusionGemma self-conditioning logits row=%d len=%d want %d", pos, len(scratch.Logits[pos]), vocab)
		}
		probs := append([]float32(nil), scratch.Logits[pos][:vocab]...)
		softmaxInPlace(probs)
		for vocabID, prob := range probs {
			if prob == 0 {
				continue
			}
			raw, dtype, shape, err := weights.RawTensorRow(fp.Globals.EmbedTokens.Name, vocabID)
			if err != nil {
				return nil, err
			}
			if len(shape) != 1 || shape[0] != hiddenSize {
				return nil, fmt.Errorf("DiffusionGemma self-conditioning row shape %v want [%d]", shape, hiddenSize)
			}
			if err := decodeFloatRowTo(row, raw, dtype); err != nil {
				return nil, err
			}
			dst := out[pos*hiddenSize : (pos+1)*hiddenSize]
			for i := range dst {
				dst[i] += prob * row[i]
			}
		}
	}
	return out, nil
}

func runLMHead(weights *TextWeights, scratch ForwardScratch) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma LM head missing weights")
	}
	fp := weights.ForwardPlan()
	if fp.Globals.EmbedTokens == nil {
		return fmt.Errorf("DiffusionGemma LM head missing tied embeddings")
	}
	if len(fp.Globals.EmbedTokens.Shape) != 2 {
		return fmt.Errorf("DiffusionGemma LM head tied embedding shape %v is not rank-2", fp.Globals.EmbedTokens.Shape)
	}
	vocab := fp.Globals.EmbedTokens.Shape[0]
	hiddenSize := fp.Globals.EmbedTokens.Shape[1]
	if hiddenSize <= 0 || vocab <= 0 || len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("DiffusionGemma LM head hidden len=%d hidden_size=%d vocab=%d", len(scratch.Hidden), hiddenSize, vocab)
	}
	positions := len(scratch.Hidden) / hiddenSize
	if len(scratch.Logits) < positions {
		return fmt.Errorf("DiffusionGemma LM head logits rows=%d want %d", len(scratch.Logits), positions)
	}
	row := make([]float32, hiddenSize)
	for vocabID := 0; vocabID < vocab; vocabID++ {
		raw, dtype, shape, err := weights.RawTensorRow(fp.Globals.EmbedTokens.Name, vocabID)
		if err != nil {
			return err
		}
		if len(shape) != 1 || shape[0] != hiddenSize {
			return fmt.Errorf("DiffusionGemma LM head row shape %v want [%d]", shape, hiddenSize)
		}
		if err := decodeFloatRowTo(row, raw, dtype); err != nil {
			return err
		}
		for pos := 0; pos < positions; pos++ {
			if len(scratch.Logits[pos]) < vocab {
				return fmt.Errorf("DiffusionGemma LM head logits row=%d len=%d want %d", pos, len(scratch.Logits[pos]), vocab)
			}
			scratch.Logits[pos][vocabID] = dot(scratch.Hidden[pos*hiddenSize:(pos+1)*hiddenSize], row)
		}
	}
	return nil
}

func errOpNotImplemented(op OpKind) error {
	return fmt.Errorf("DiffusionGemma CPU/SIMD op %s is not implemented", op)
}
