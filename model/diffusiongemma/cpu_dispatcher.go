package diffusiongemma

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sync"
	"time"
	"unsafe"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// CPUDispatcher is the native CPU/SIMD forward scaffold for DiffusionGemma text
// denoising. It gives each semantic op an explicit hook so implementation can
// proceed operation-by-operation while keeping the Denoiser interface stable.
type CPUDispatcher struct {
	ResidentLayerPrefix int
	MaxLayers           int
	TailAfterMaxLayers  bool
	LMHeadTopK          int
	Progress            bool
	SkipEviction        bool
}

type ForwardScratch struct {
	Hidden     []float32
	Residual   []float32
	MlpOut     []float32
	MoeOut     []float32
	Logits     [][]float32
	Router     []float32
	Experts    []float32
	TopKIDs    []int
	TopKVals   []float32
	LMHeadTopK int
}

func NewForwardScratch(buffers ForwardBufferPlan) ForwardScratch {
	topKSlots := maxNonNegative(buffers.CanvasLength * buffers.TopKExperts)
	hiddenSize := maxNonNegative(buffers.Hidden)
	return ForwardScratch{Hidden: make([]float32, hiddenSize), Residual: make([]float32, hiddenSize), MlpOut: make([]float32, hiddenSize), MoeOut: make([]float32, hiddenSize), Router: make([]float32, maxNonNegative(buffers.Router)), Experts: make([]float32, maxNonNegative(buffers.Experts)), TopKIDs: make([]int, topKSlots), TopKVals: make([]float32, topKSlots), Logits: makeLogitRows(buffers.CanvasLength, buffers.VocabSize)}
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

func (d CPUDispatcher) RunTextForward(ctx ForwardContext, weights *TextWeights, ops ForwardOpPlan, buffers ForwardBufferPlan) (ForwardOutput, error) {
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
	scratch.LMHeadTopK = d.LMHeadTopK
	// Resize working buffers to actual canvas length
	actualPositions := len(ctx.Canvas)
	actualHidden := actualPositions * buffers.HiddenSize
	if actualHidden > 0 && actualHidden < len(scratch.Hidden) {
		scratch.Hidden = scratch.Hidden[:actualHidden]
		scratch.Residual = scratch.Residual[:actualHidden]
		scratch.MlpOut = scratch.MlpOut[:actualHidden]
		scratch.MoeOut = scratch.MoeOut[:actualHidden]
		scratch.Logits = scratch.Logits[:actualPositions]
	}
	for _, op := range ops.Prefix {
		if err := dispatchPrefixOp(op, ctx, weights, scratch); err != nil {
			return ForwardOutput{}, err
		}
	}
	currentLayer := -1
	completedLayers := 0
	layerStarted := time.Now()
	for _, op := range ops.Layers {
		if currentLayer >= 0 && op.Layer != currentLayer {
			completedLayers++
			if d.Progress {
				fmt.Fprintf(os.Stderr, "DiffusionGemma CPU dispatcher: completed layer=%d cache_entries=%d cache_bytes=%d elapsed=%s\n", currentLayer, weights.FloatCacheEntries(), weights.FloatCacheBytes(), time.Since(layerStarted).Round(time.Millisecond))
			}
			if !d.SkipEviction && currentLayer >= d.ResidentLayerPrefix {
				weights.EvictLayer(currentLayer)
			}
			if d.MaxLayers > 0 && completedLayers >= d.MaxLayers {
				break
			}
			layerStarted = time.Now()
		}
		if op.Layer != currentLayer {
			currentLayer = op.Layer
			if d.Progress {
				fmt.Fprintf(os.Stderr, "DiffusionGemma CPU dispatcher: starting layer=%d\n", currentLayer)
			}
		}
		if err := dispatchLayerOp(op, ctx, weights, scratch); err != nil {
			return ForwardOutput{}, err
		}
	}
	if currentLayer >= 0 && currentLayer >= d.ResidentLayerPrefix {
		weights.EvictLayer(currentLayer)
	}
	if d.MaxLayers > 0 && !d.TailAfterMaxLayers {
		return ForwardOutput{Logits: scratch.Logits, SelfConditioning: ctx.SelfConditioning}, nil
	}
	for _, op := range ops.Tail {
		if d.Progress {
			fmt.Fprintf(os.Stderr, "DiffusionGemma CPU dispatcher: starting tail op=%s\n", op)
		}
		started := time.Now()
		if err := dispatchTailOp(op, weights, scratch); err != nil {
			return ForwardOutput{}, err
		}
		if d.Progress {
			fmt.Fprintf(os.Stderr, "DiffusionGemma CPU dispatcher: completed tail op=%s cache_entries=%d cache_bytes=%d elapsed=%s\n", op, weights.FloatCacheEntries(), weights.FloatCacheBytes(), time.Since(started).Round(time.Millisecond))
		}
	}
	if d.Progress && len(scratch.Logits) > 0 {
		for pos := 0; pos < len(scratch.Logits) && pos < 2; pos++ {
			row := scratch.Logits[pos]
			bestID, bestVal := 0, row[0]
			var sum float64
			for i, v := range row {
				if v > bestVal {
					bestID, bestVal = i, v
				}
				sum += float64(v)
			}
			fmt.Fprintf(os.Stderr, "DiffusionGemma CPU dispatcher: logits pos=%d top_id=%d top_val=%.6f mean=%.6f\n", pos, bestID, bestVal, sum/float64(len(row)))
		}
	}
	// Build self-conditioning from logits. With sparse top-k, softmax
	// naturally zeros -Inf positions, producing a weighted average of
	// only the top-k token embeddings.
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
	embedScale := float32(math.Sqrt(float64(hiddenSize)))
	for i := range scratch.Hidden[:need] {
		scratch.Hidden[i] *= embedScale
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
	case "F8_E4M3", "F8_E4M3FN":
		if len(raw) < len(dst) {
			return fmt.Errorf("DiffusionGemma F8_E4M3 row bytes=%d want %d", len(raw), len(dst))
		}
		for i := range dst {
			dst[i] = fp8DecodeE4M3(raw[i])
		}
		return nil
	default:
		return fmt.Errorf("DiffusionGemma unsupported float row dtype %s", dtype)
	}
}

// fp8DecodeE4M3 decodes a single FP8 E4M3 byte to float32.
func fp8DecodeE4M3(b byte) float32 {
	sign := float32(1)
	if b&0x80 != 0 {
		sign = -1
		b &= 0x7F
	}
	exp := int(b >> 3)
	mant := int(b & 0x07)
	if exp == 0 {
		// subnormal
		return sign * float32(mant) * (1.0 / 64.0) * (1.0 / 8.0)
	}
	if exp == 15 && mant == 7 {
		// NaN
		return 0
	}
	return sign * float32(8+mant) * float32(uint32(1)<<uint(exp-1)) / 64.0
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
		copy(scratch.Residual, scratch.Hidden)
		return runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.InputLayerNorm })
	case OpSelfAttention:
		return runSelfAttention(op, ctx, weights, scratch)
	case OpPostAttention:
		if err := runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.PostAttentionLayerNorm }); err != nil {
			return err
		}
		for i := range scratch.Hidden {
			scratch.Hidden[i] += scratch.Residual[i]
		}
		return nil
	case OpDenseMLP:
		return runDenseMLP(op, weights, scratch)
	case OpPreMoE:
		copy(scratch.Residual, scratch.Hidden)
		return runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.PreFFNLayerNorm })
	case OpRouter:
		return runRouterFromResidual(op, weights, scratch)
	case OpExperts:
		return runExpertsFromResidual(op, weights, scratch)
	case OpPostMoE:
		if err := runCombineMlpMoe(op, weights, scratch); err != nil {
			return err
		}
		for i := range scratch.Hidden {
			scratch.Hidden[i] += scratch.Residual[i]
		}
		return nil
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
	qM, err := loadMixedMatrix(weights, lb.QProj)
	if err != nil {
		return err
	}
	qRows, hiddenSize := qM.rows, qM.cols
	kM, err := loadMixedMatrix(weights, lb.KProj)
	if err != nil {
		return err
	}
	kRows := kM.rows
	if kM.cols != hiddenSize {
		return fmt.Errorf("DiffusionGemma attention K shape [%d,%d] hidden=%d", kRows, kM.cols, hiddenSize)
	}
	var vM *mixedMatrix
	vRows := kRows
	if lb.VProj != nil {
		vM, err = loadMixedMatrix(weights, lb.VProj)
		if err != nil {
			return err
		}
		vRows = vM.rows
		if vM.cols != hiddenSize {
			return fmt.Errorf("DiffusionGemma attention V shape [%d,%d] hidden=%d", vRows, vM.cols, hiddenSize)
		}
	}
	oM, err := loadMixedMatrix(weights, lb.OProj)
	if err != nil {
		return err
	}
	if oM.rows != hiddenSize || oM.cols != qRows {
		return fmt.Errorf("DiffusionGemma attention O shape [%d,%d] q=%d hidden=%d", oM.rows, oM.cols, qRows, hiddenSize)
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
	attnAll := make([]float32, positions*qRows)
	ropeHalf := headDim / 2
	ropeTheta := 10000.0
	if op.Type == "full_attention" {
		ropeHalf = headDim / 8
		ropeTheta = 1000000.0
	}
	ropeFreqs := simd.BuildRoPEFreqs(ctx.EncoderSeqLen+positions, ropeHalf, headDim, ropeTheta)
	qDone, kDone, vDone := false, false, false
	if lb.VProj != nil {
		if done, err := k3GemmManyRowsQ80([][]float32{qAll, kAll, vAll}, scratch.Hidden, positions, weights, []*TensorBinding{lb.QProj, lb.KProj, lb.VProj}); err != nil {
			return err
		} else if done {
			qDone, kDone, vDone = true, true, true
		}
	} else {
		if done, err := k3GemmManyRowsQ80([][]float32{qAll, kAll}, scratch.Hidden, positions, weights, []*TensorBinding{lb.QProj, lb.KProj}); err != nil {
			return err
		} else if done {
			qDone, kDone = true, true
		}
	}
	if !qDone {
		var err error
		qDone, err = k3GemmRowsQ80(qAll, scratch.Hidden, positions, weights, lb.QProj)
		if err != nil {
			return err
		}
	}
	if !kDone {
		var err error
		kDone, err = k3GemmRowsQ80(kAll, scratch.Hidden, positions, weights, lb.KProj)
		if err != nil {
			return err
		}
	}
	if lb.VProj != nil && !vDone {
		var err error
		vDone, err = k3GemmRowsQ80(vAll, scratch.Hidden, positions, weights, lb.VProj)
		if err != nil {
			return err
		}
	}
	for pos := 0; pos < positions; pos++ {
		hidden := scratch.Hidden[pos*hiddenSize : (pos+1)*hiddenSize]
		q := qAll[pos*qRows : (pos+1)*qRows]
		k := kAll[pos*kRows : (pos+1)*kRows]
		v := vAll[pos*vRows : (pos+1)*vRows]
		if !qDone && !qM.gemvRows(q, hidden) {
			return fmt.Errorf("DiffusionGemma attention Q GEMV rejected layer %d", op.Layer)
		}
		if !kDone && !kM.gemvRows(k, hidden) {
			return fmt.Errorf("DiffusionGemma attention K GEMV rejected layer %d", op.Layer)
		}
		if lb.VProj != nil {
			if !vDone && !vM.gemvRows(v, hidden) {
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
			simd.ApplyRoPEPartial(q, ropeFreqs, pos+ctx.EncoderSeqLen, heads, headDim, ropeHalf)
			simd.ApplyRoPEPartial(k, ropeFreqs, pos+ctx.EncoderSeqLen, kvHeads, headDim, ropeHalf)
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
	slidingWindow := 0
	if op.Type == "sliding_attention" {
		slidingWindow = 1024
	}
	runAttentionContextK3(attnAll, qAll, kAll, vAll, enc, positions, heads, kvHeads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow)
	if done, err := k3GemmRowsQ80(scratch.Hidden, attnAll, positions, weights, lb.OProj); err != nil {
		return err
	} else if !done {
		out := make([]float32, hiddenSize)
		for pos := 0; pos < positions; pos++ {
			attnCtx := attnAll[pos*qRows : (pos+1)*qRows]
			if !oM.gemvRows(out, attnCtx) {
				return fmt.Errorf("DiffusionGemma attention O GEMV rejected layer %d", op.Layer)
			}
			copy(scratch.Hidden[pos*hiddenSize:(pos+1)*hiddenSize], out)
		}
	}
	return nil
}

func runAttentionContextK3(attnAll, qAll, kAll, vAll []float32, enc EncoderKVLayer, positions, heads, kvHeads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow int) {
	if positions <= 0 || heads <= 0 || headDim <= 0 {
		return
	}
	totalKV := encSeq + positions
	work := func(start, end int) {
		scores := make([]float32, totalKV)
		for pos := start; pos < end; pos++ {
			attnCtx := attnAll[pos*qRows : (pos+1)*qRows]
			for i := range attnCtx {
				attnCtx[i] = 0
			}
			for h := 0; h < heads; h++ {
				kvh := h / group
				q := qAll[pos*qRows+h*headDim : pos*qRows+(h+1)*headDim]
				for j := 0; j < totalKV; j++ {
					if j < encSeq {
						scores[j] = k3Dot(q, enc.Keys[j*kRows+kvh*headDim:j*kRows+(kvh+1)*headDim])
						continue
					}
					canvasJ := j - encSeq
					if slidingWindow > 0 && absInt(pos-canvasJ) >= slidingWindow {
						scores[j] = float32(math.Inf(-1))
						continue
					}
					scores[j] = k3Dot(q, kAll[canvasJ*kRows+kvh*headDim:canvasJ*kRows+(kvh+1)*headDim])
				}
				k3SoftmaxInPlace(scores)
				dst := attnCtx[h*headDim : (h+1)*headDim]
				for j, score := range scores {
					var vv []float32
					if j < encSeq {
						vv = enc.Values[j*vRows+kvh*headDim : j*vRows+(kvh+1)*headDim]
					} else {
						canvasJ := j - encSeq
						vv = vAll[canvasJ*vRows+kvh*headDim : canvasJ*vRows+(kvh+1)*headDim]
					}
					k3SaxpyV(score, vv, dst)
				}
			}
		}
	}
	nw := 1
	if k3Enabled() && positions*heads >= 32 {
		nw = k3Threads()
		if nw > positions {
			nw = positions
		}
	}
	if nw <= 1 {
		work(0, positions)
		return
	}
	var wg sync.WaitGroup
	wg.Add(nw)
	for wid := 0; wid < nw; wid++ {
		start := wid * positions / nw
		end := (wid + 1) * positions / nw
		go func() {
			defer wg.Done()
			work(start, end)
		}()
	}
	wg.Wait()
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
	if binding == nil {
		return nil, fmt.Errorf("DiffusionGemma missing vector binding")
	}
	t, err := weights.CachedFloatTensor(binding.Name)
	if err != nil {
		return nil, err
	}
	if len(t.Shape) != 1 {
		return nil, fmt.Errorf("DiffusionGemma tensor %q shape %v is not rank-1", binding.Name, t.Shape)
	}
	return t.Data, nil
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
	gateM, err := loadMixedMatrix(weights, lb.MLPGateProj)
	if err != nil {
		return err
	}
	upM, err := loadMixedMatrix(weights, lb.MLPUpProj)
	if err != nil {
		return err
	}
	downM, err := loadMixedMatrix(weights, lb.MLPDownProj)
	if err != nil {
		return err
	}
	if gateM.rows != upM.rows || gateM.cols != upM.cols || downM.cols != gateM.rows || downM.rows != gateM.cols {
		return fmt.Errorf("DiffusionGemma dense MLP shape mismatch")
	}
	hiddenSize := gateM.cols
	intermediate := gateM.rows
	if hiddenSize <= 0 || intermediate <= 0 || len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("DiffusionGemma dense MLP hidden len=%d hidden_size=%d intermediate=%d", len(scratch.Hidden), hiddenSize, intermediate)
	}
	positions := len(scratch.Hidden) / hiddenSize
	gateAll := make([]float32, positions*intermediate)
	upAll := make([]float32, positions*intermediate)
	if done, err := k3Gemm2RowsQ80(gateAll, upAll, scratch.Hidden, positions, weights, lb.MLPGateProj, lb.MLPUpProj); err != nil {
		return err
	} else if !done {
		for off, pos := 0, 0; off < len(scratch.Hidden); off, pos = off+hiddenSize, pos+1 {
			row := scratch.Hidden[off : off+hiddenSize]
			gate := gateAll[pos*intermediate : (pos+1)*intermediate]
			up := upAll[pos*intermediate : (pos+1)*intermediate]
			if !gateM.gemvRows(gate, row) {
				return fmt.Errorf("DiffusionGemma dense MLP gate GEMV rejected layer %d", op.Layer)
			}
			if !upM.gemvRows(up, row) {
				return fmt.Errorf("DiffusionGemma dense MLP up GEMV rejected layer %d", op.Layer)
			}
		}
	}
	actAll := make([]float32, positions*intermediate)
	for pos := 0; pos < positions; pos++ {
		gate := gateAll[pos*intermediate : (pos+1)*intermediate]
		up := upAll[pos*intermediate : (pos+1)*intermediate]
		act := actAll[pos*intermediate : (pos+1)*intermediate]
		if !simd.GELUTanhMulTo(act, gate, up) {
			return fmt.Errorf("DiffusionGemma dense MLP activation rejected layer %d", op.Layer)
		}
	}
	if done, err := k3GemmRowsQ80(scratch.Hidden, actAll, positions, weights, lb.MLPDownProj); err != nil {
		return err
	} else if !done {
		out := make([]float32, hiddenSize)
		for pos := 0; pos < positions; pos++ {
			act := actAll[pos*intermediate : (pos+1)*intermediate]
			if !downM.gemvRows(out, act) {
				return fmt.Errorf("DiffusionGemma dense MLP down GEMV rejected layer %d", op.Layer)
			}
			copy(scratch.Hidden[pos*hiddenSize:(pos+1)*hiddenSize], out)
		}
	}
	postNorm1, err := loadFloatVector(weights, lb.PostFFNLayerNorm1)
	if err != nil {
		return err
	}
	for off := 0; off < len(scratch.Hidden); off += hiddenSize {
		if !simd.RMSNormTo(scratch.Hidden[off:off+hiddenSize], postNorm1, 1e-6) {
			return fmt.Errorf("DiffusionGemma dense MLP post_norm_1 rejected")
		}
	}
	copy(scratch.MlpOut, scratch.Hidden)
	copy(scratch.Hidden, scratch.Residual)
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

func runRouterFromResidual(op LayerOp, weights *TextWeights, scratch ForwardScratch) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma router missing weights")
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("DiffusionGemma router layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	scaleVec, err := loadFloatVector(weights, lb.RouterScale)
	if err != nil {
		return err
	}
	if lb.RouterProj == nil || len(lb.RouterProj.Shape) != 2 {
		return fmt.Errorf("DiffusionGemma router missing projection binding")
	}
	numExperts, projCols := lb.RouterProj.Shape[0], lb.RouterProj.Shape[1]
	hiddenSize := len(scaleVec)
	if projCols != hiddenSize || numExperts <= 0 || len(scratch.Residual)%hiddenSize != 0 {
		return fmt.Errorf("DiffusionGemma router shape mismatch scale=%d proj=[%d,%d] residual=%d", hiddenSize, numExperts, projCols, len(scratch.Residual))
	}
	positions := len(scratch.Residual) / hiddenSize
	if len(scratch.Router) < positions*numExperts {
		return fmt.Errorf("DiffusionGemma router scratch=%d want %d", len(scratch.Router), positions*numExperts)
	}
	routerInput := make([]float32, positions*hiddenSize)
	scalarRootSize := float32(1.0 / math.Sqrt(float64(hiddenSize)))
	for pos := 0; pos < positions; pos++ {
		in := routerInput[pos*hiddenSize : (pos+1)*hiddenSize]
		copy(in, scratch.Residual[pos*hiddenSize:(pos+1)*hiddenSize])
		if !simd.RMSNormNoScaleTo(in, 1e-6) {
			return fmt.Errorf("DiffusionGemma router norm rejected")
		}
		for i := range in {
			in[i] *= scaleVec[i] * scalarRootSize
		}
	}
	scoredAll := scratch.Router[:positions*numExperts]
	if done, err := k3GemmRowsQ80(scoredAll, routerInput, positions, weights, lb.RouterProj); err != nil {
		return err
	} else if !done {
		projW, projRows, projCols, err := loadFloatMatrix(weights, lb.RouterProj)
		if err != nil {
			return err
		}
		if projRows != numExperts || projCols != hiddenSize {
			return fmt.Errorf("DiffusionGemma router fallback shape mismatch proj=[%d,%d] expected=[%d,%d]", projRows, projCols, numExperts, hiddenSize)
		}
		for pos := 0; pos < positions; pos++ {
			if !simd.GemvRows(scoredAll[pos*numExperts:(pos+1)*numExperts], routerInput[pos*hiddenSize:(pos+1)*hiddenSize], projW, numExperts, hiddenSize) {
				return fmt.Errorf("DiffusionGemma router GEMV rejected")
			}
		}
	}
	for pos := 0; pos < positions; pos++ {
		scored := scoredAll[pos*numExperts : (pos+1)*numExperts]
		k3SoftmaxInPlace(scored)
		topK := len(scratch.TopKIDs) / positions
		if topK <= 0 {
			continue
		}
		ids := scratch.TopKIDs[pos*topK : (pos+1)*topK]
		vals := scratch.TopKVals[pos*topK : (pos+1)*topK]
		for i := range ids {
			ids[i] = -1
			vals[i] = float32(math.Inf(-1))
		}
		for expertID, score := range scored {
			insertTopK(ids, vals, expertID, score)
		}
		var topKSum float32
		for _, v := range vals {
			if v > float32(math.Inf(-1)) {
				topKSum += v
			}
		}
		if topKSum > 0 {
			for i := range vals {
				vals[i] /= topKSum
			}
		}
		if lb.RouterPerExpertScale != nil {
			perExpert, err2 := loadFloatVector(weights, lb.RouterPerExpertScale)
			if err2 != nil {
				return err2
			}
			for i, id := range ids {
				if id >= 0 && id < len(perExpert) {
					vals[i] *= perExpert[id]
				}
			}
		}
	}
	return nil
}

func runExpertsFromResidual(op LayerOp, weights *TextWeights, scratch ForwardScratch) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma experts missing weights")
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("DiffusionGemma experts layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	preNorm2, err := loadFloatVector(weights, lb.PreFFNLayerNorm2)
	if err != nil {
		return err
	}
	hiddenSize := len(preNorm2)
	if hiddenSize <= 0 || len(scratch.Residual)%hiddenSize != 0 {
		return fmt.Errorf("DiffusionGemma expert hidden mismatch")
	}
	layout, err := expertLayoutForLayer(weights, lb, hiddenSize)
	if err != nil {
		return err
	}
	intermediate := layout.intermediate

	positions := len(scratch.Residual) / hiddenSize
	for i := range scratch.MoeOut {
		scratch.MoeOut[i] = 0
	}
	normedRow := make([]float32, hiddenSize)
	gate := make([]float32, intermediate)
	up := make([]float32, intermediate)
	act := make([]float32, intermediate)
	expertOut := make([]float32, hiddenSize)
	topK := len(scratch.TopKIDs) / positions

	expertsDone, err := k3RunPerExpertA100(weights, lb, layout, scratch, preNorm2, hiddenSize, positions, topK)
	if err != nil {
		return err
	}
	if !expertsDone {
		// Collect unique expert IDs to decode only needed slices
		neededExperts := map[int]bool{}
		for pos := 0; pos < positions; pos++ {
			for k := 0; k < topK; k++ {
				id := scratch.TopKIDs[pos*topK+k]
				if id >= 0 && id < layout.nExperts {
					neededExperts[id] = true
				}
			}
		}
		decoded := make(map[int]decodedExpertWeights, len(neededExperts))
		for expertID := range neededExperts {
			ew, err := loadLayerExpertWeights(weights, lb, layout, expertID, hiddenSize)
			if err != nil {
				return err
			}
			decoded[expertID] = ew
		}

		for pos := 0; pos < positions; pos++ {
			resRow := scratch.Residual[pos*hiddenSize : (pos+1)*hiddenSize]
			copy(normedRow, resRow)
			if !simd.RMSNormTo(normedRow, preNorm2, 1e-6) {
				return fmt.Errorf("DiffusionGemma expert pre_norm_2 rejected")
			}
			dst := scratch.MoeOut[pos*hiddenSize : (pos+1)*hiddenSize]
			for k := 0; k < topK; k++ {
				expertID := scratch.TopKIDs[pos*topK+k]
				weight := scratch.TopKVals[pos*topK+k]
				ew, ok := decoded[expertID]
				if !ok {
					continue
				}
				if !simd.GemvRows(gate, normedRow, ew.gateW, intermediate, hiddenSize) || !simd.GemvRows(up, normedRow, ew.upW, intermediate, hiddenSize) {
					return fmt.Errorf("DiffusionGemma expert GEMV rejected")
				}
				if !simd.GELUTanhMulTo(act, gate, up) {
					return fmt.Errorf("DiffusionGemma expert activation rejected")
				}
				if !simd.GemvRows(expertOut, act, ew.downW, hiddenSize, intermediate) {
					return fmt.Errorf("DiffusionGemma expert down GEMV rejected")
				}
				k3SaxpyV(weight, expertOut, dst)
			}
		}
	}
	postNorm2, err := loadFloatVector(weights, lb.PostFFNLayerNorm2)
	if err != nil {
		return err
	}
	for off := 0; off < len(scratch.MoeOut); off += hiddenSize {
		if !simd.RMSNormTo(scratch.MoeOut[off:off+hiddenSize], postNorm2, 1e-6) {
			return fmt.Errorf("DiffusionGemma expert post_norm_2 rejected")
		}
	}
	return nil
}

func runCombineMlpMoe(op LayerOp, weights *TextWeights, scratch ForwardScratch) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma combine missing weights")
	}
	fp := weights.ForwardPlan()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("DiffusionGemma combine layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]
	postNorm, err := loadFloatVector(weights, lb.PostFFNLayerNorm)
	if err != nil {
		return err
	}
	hiddenSize := len(postNorm)
	for i := range scratch.Hidden {
		scratch.Hidden[i] = scratch.MlpOut[i] + scratch.MoeOut[i]
	}
	for off := 0; off < len(scratch.Hidden); off += hiddenSize {
		if !simd.RMSNormTo(scratch.Hidden[off:off+hiddenSize], postNorm, 1e-6) {
			return fmt.Errorf("DiffusionGemma combine post_norm rejected")
		}
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
	t, err := weights.CachedFloatTensor(binding.Name)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	if len(t.Shape) != 3 || t.Shape[0] <= 0 || t.Shape[1] <= 0 || t.Shape[2] <= 0 {
		return nil, 0, 0, 0, fmt.Errorf("DiffusionGemma tensor %q shape %v is not rank-3", binding.Name, t.Shape)
	}
	return t.Data, t.Shape[0], t.Shape[1], t.Shape[2], nil
}

// loadExpertSlice decodes a single expert's 2D slice from a 3D [experts, rows, cols]
// tensor stored in safetensors, without decoding all experts.
func loadExpertSlice(weights *TextWeights, binding *TensorBinding, expertID int) ([]float32, int, int, error) {
	if binding == nil {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma missing expert tensor binding")
	}
	if len(binding.Shape) != 3 {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma expert tensor %q shape %v is not rank-3", binding.Name, binding.Shape)
	}
	nExperts, rows, cols := binding.Shape[0], binding.Shape[1], binding.Shape[2]
	if expertID < 0 || expertID >= nExperts {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma expert %d outside [0,%d)", expertID, nExperts)
	}
	raw, dtype, _, err := weights.RawTensor(binding.Name)
	if err != nil {
		return nil, 0, 0, err
	}
	elemSize, ok := diffusionGemmaDTypeSize(dtype)
	if !ok {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma expert tensor %q unsupported dtype %s", binding.Name, dtype)
	}
	sliceElements := rows * cols
	sliceBytes := sliceElements * elemSize
	start := expertID * sliceBytes
	end := start + sliceBytes
	if end > len(raw) {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma expert %d byte range [%d,%d) exceeds %d", expertID, start, end, len(raw))
	}
	out := make([]float32, sliceElements)
	if err := decodeFloatRowTo(out, raw[start:end], dtype); err != nil {
		return nil, 0, 0, err
	}
	return out, rows, cols, nil
}

// loadExpertSliceBF16 returns a BF16 expert slice as []uint16 via zero-copy
// from the mmap'd tensor data. Returns nil if not BF16.
func loadExpertSliceBF16(weights *TextWeights, binding *TensorBinding, expertID int) ([]uint16, int, int, error) {
	if binding == nil || len(binding.Shape) != 3 {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma missing/invalid expert tensor binding")
	}
	nExperts, rows, cols := binding.Shape[0], binding.Shape[1], binding.Shape[2]
	if expertID < 0 || expertID >= nExperts {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma expert %d outside [0,%d)", expertID, nExperts)
	}
	raw, dtype, _, err := weights.RawTensor(binding.Name)
	if err != nil {
		return nil, 0, 0, err
	}
	if dtype != "BF16" {
		return nil, 0, 0, nil // caller should fall back to F32 path
	}
	sliceElements := rows * cols
	sliceBytes := sliceElements * 2
	start := expertID * sliceBytes
	end := start + sliceBytes
	if end > len(raw) {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma expert %d BF16 range [%d,%d) exceeds %d", expertID, start, end, len(raw))
	}
	out := unsafe.Slice((*uint16)(unsafe.Pointer(&raw[start])), sliceElements)
	return out, rows, cols, nil
}

func loadFloatMatrix(weights *TextWeights, binding *TensorBinding) ([]float32, int, int, error) {
	if binding == nil {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma missing matrix binding")
	}
	t, err := weights.CachedFloatTensor(binding.Name)
	if err != nil {
		return nil, 0, 0, err
	}
	if len(t.Shape) != 2 || t.Shape[0] <= 0 || t.Shape[1] <= 0 {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma tensor %q shape %v is not rank-2", binding.Name, t.Shape)
	}
	return t.Data, t.Shape[0], t.Shape[1], nil
}

// mixedMatrix holds either F32 or BF16 weight data for GEMV.
type mixedMatrix struct {
	f32  []float32
	bf16 []uint16
	rows int
	cols int
}

// gemvRows runs the appropriate GEMV: BF16 path if available, else F32.
func (m *mixedMatrix) gemvRows(out, x []float32) bool {
	if m.bf16 != nil {
		return simd.GemvRowsBF16(out, x, m.bf16, m.rows, m.cols)
	}
	return simd.GemvRows(out, x, m.f32, m.rows, m.cols)
}

// loadMixedMatrix tries cached F32 first (fast GEMV), avoiding BF16 path
// since BF16DotF32 is slower than F32 Sdot on most CPUs.
func loadMixedMatrix(weights *TextWeights, binding *TensorBinding) (*mixedMatrix, error) {
	if binding == nil {
		return nil, fmt.Errorf("DiffusionGemma missing matrix binding")
	}
	f32, rows, cols, err := loadFloatMatrix(weights, binding)
	if err != nil {
		return nil, err
	}
	return &mixedMatrix{f32: f32, rows: rows, cols: cols}, nil
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
	if binding == nil {
		return 0, fmt.Errorf("DiffusionGemma missing scalar binding")
	}
	t, err := weights.CachedFloatTensor(binding.Name)
	if err != nil {
		return 0, err
	}
	n := 1
	for _, dim := range t.Shape {
		if dim <= 0 {
			return 0, fmt.Errorf("DiffusionGemma tensor %q invalid scalar shape %v", binding.Name, t.Shape)
		}
		n *= dim
	}
	if n != 1 || len(t.Data) != 1 {
		return 0, fmt.Errorf("DiffusionGemma tensor %q shape %v is not scalar", binding.Name, t.Shape)
	}
	return t.Data[0], nil
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
		k3SoftmaxInPlace(probs)
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
	embedScale := float32(math.Sqrt(float64(hiddenSize)))
	for i := range out {
		out[i] *= embedScale
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
	topK := scratch.LMHeadTopK
	if topK > vocab {
		topK = vocab
	}
	if topK > 0 {
		for pos := 0; pos < positions; pos++ {
			if len(scratch.Logits[pos]) < vocab {
				return fmt.Errorf("DiffusionGemma LM head logits row=%d len=%d want %d", pos, len(scratch.Logits[pos]), vocab)
			}
			for i := 0; i < vocab; i++ {
				scratch.Logits[pos][i] = float32(math.Inf(-1))
			}
		}
	}
	topIDs := make([][]int, positions)
	topVals := make([][]float32, positions)
	if topK > 0 {
		for pos := 0; pos < positions; pos++ {
			topIDs[pos] = make([]int, topK)
			topVals[pos] = make([]float32, topK)
			for i := range topIDs[pos] {
				topIDs[pos][i] = -1
				topVals[pos][i] = float32(math.Inf(-1))
			}
		}
	}
	if topK > 0 {
		t, err := weights.CachedFloatTensor(fp.Globals.EmbedTokens.Name)
		if err != nil {
			return err
		}
		if len(t.Shape) != 2 || t.Shape[0] != vocab || t.Shape[1] != hiddenSize {
			return fmt.Errorf("DiffusionGemma LM head cached embedding shape %v want [%d %d]", t.Shape, vocab, hiddenSize)
		}
		scores := make([]float32, vocab)
		for pos := 0; pos < positions; pos++ {
			hidden := scratch.Hidden[pos*hiddenSize : (pos+1)*hiddenSize]
			if !simd.GemvRows(scores, hidden, t.Data, vocab, hiddenSize) {
				return fmt.Errorf("DiffusionGemma LM head SIMD GEMV failed vocab=%d hidden=%d", vocab, hiddenSize)
			}
			for vocabID, score := range scores {
				insertTopK(topIDs[pos], topVals[pos], vocabID, score)
			}
		}
	} else {
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
				scratch.Logits[pos][vocabID] = k3Dot(scratch.Hidden[pos*hiddenSize:(pos+1)*hiddenSize], row)
			}
		}
	}
	if topK > 0 {
		for pos := 0; pos < positions; pos++ {
			for i, id := range topIDs[pos] {
				if id >= 0 {
					scratch.Logits[pos][id] = topVals[pos][i]
				}
			}
		}
	}
	return nil
}

func insertTopK(ids []int, vals []float32, id int, val float32) {
	for i := range vals {
		if val <= vals[i] {
			continue
		}
		copy(vals[i+1:], vals[i:])
		copy(ids[i+1:], ids[i:])
		vals[i] = val
		ids[i] = id
		return
	}
}

func errOpNotImplemented(op OpKind) error {
	return fmt.Errorf("DiffusionGemma CPU/SIMD op %s is not implemented", op)
}
