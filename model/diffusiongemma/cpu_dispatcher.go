package diffusiongemma

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"time"
	"unsafe"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/internal/checked"
)

// CPUDispatcher is retired for DiffusionGemma generation.
//
// Do not use or develop this path further. It existed as an early correctness
// scaffold, but the project goal is GGUF Q4_K_M backend-graph execution on GPU.
// Keeping CPU generation available caused misleading progress and unacceptable
// performance; callers must use/implement GPU-resident paths instead.
type CPUDispatcher struct {
	ResidentLayerPrefix   int
	MaxLayers             int
	TailAfterMaxLayers    bool
	LMHeadTopK            int
	Progress              bool
	SkipEviction          bool
	ExpertIndex           *FP8ExpertIndex  // FP8 safetensor expert weights for optimized CPU MoE
	GGUFExpertIndex       *GGUFExpertIndex // GGUF expert weights for encoder/denoiser MoE
	FinalLogitSoftcapping float32
}

type ForwardScratch struct {
	Hidden                []float32
	Residual              []float32
	MlpOut                []float32
	MoeOut                []float32
	Logits                [][]float32
	Router                []float32
	Experts               []float32
	TopKIDs               []int
	TopKVals              []float32
	TopKExperts           int // per-position top-K count from model config
	LMHeadTopK            int
	FP8ExpertIndex        *FP8ExpertIndex
	GGUFExpertIndex       *GGUFExpertIndex
	FinalLogitSoftcapping float32 // tanh(x/c)*c after LM head; 0 = disabled
	SCTempInv             float32 // self-conditioning: 1/t from the current step (applied when building soft embeddings for the NEXT step)
	SlidingWindow         int     // attention.sliding_window (n_swa); 0 disables SWA clipping
}

func NewForwardScratch(buffers ForwardBufferPlan) ForwardScratch {
	topKSlots, ok := checked.MulInt(buffers.CanvasLength, buffers.TopKExperts)
	if !ok {
		topKSlots = 0
	}
	topKSlots = maxNonNegative(topKSlots)
	hiddenSize := maxNonNegative(buffers.Hidden)
	return ForwardScratch{Hidden: make([]float32, hiddenSize), Residual: make([]float32, hiddenSize), MlpOut: make([]float32, hiddenSize), MoeOut: make([]float32, hiddenSize), Router: make([]float32, maxNonNegative(buffers.Router)), Experts: make([]float32, maxNonNegative(buffers.Experts)), TopKIDs: make([]int, topKSlots), TopKVals: make([]float32, topKSlots), TopKExperts: buffers.TopKExperts, SlidingWindow: buffers.SlidingWindow, Logits: makeLogitRows(buffers.CanvasLength, buffers.VocabSize)}
}

func makeLogitRows(rows, cols int) [][]float32 {
	if rows <= 0 || cols <= 0 {
		return nil
	}
	total, ok := checked.MulInt(rows, cols)
	if !ok {
		return nil
	}
	flat := make([]float32, total)
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
	return ForwardOutput{}, fmt.Errorf("DiffusionGemma CPUDispatcher is disabled: CPU generation is not to be used or developed further; implement/use the GPU backend graph")
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
	scratch.FP8ExpertIndex = d.ExpertIndex
	scratch.GGUFExpertIndex = d.GGUFExpertIndex
	scratch.FinalLogitSoftcapping = d.FinalLogitSoftcapping
	scratch.SCTempInv = ctx.SCTempInv
	// Resize working buffers to actual canvas length.
	actualPositions := len(ctx.Canvas)
	actualHidden, ok := checked.MulInt(actualPositions, buffers.HiddenSize)
	if !ok || actualHidden > len(scratch.Hidden) || actualPositions > len(scratch.Logits) {
		return ForwardOutput{}, fmt.Errorf("DiffusionGemma CPU dispatcher canvas=%d exceeds buffer plan canvas=%d hidden=%d", actualPositions, buffers.CanvasLength, buffers.HiddenSize)
	}
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
	// Build self-conditioning from logits for the next denoising step. With
	// sparse top-k, softmax naturally zeros -Inf positions, producing a weighted
	// average of only the top-k token embeddings. The final step has no consumer.
	var selfConditioning []float32
	if ctx.Step > 1 {
		var err error
		selfConditioning, err = buildSelfConditioningFromLogits(weights, scratch)
		if err != nil {
			return ForwardOutput{}, err
		}
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

type embeddingRowCacheKey struct {
	weights uintptr
	token   int
	hidden  int
}

var embeddingRowCache sync.Map // map[embeddingRowCacheKey][]float32, stores embed row already scaled by sqrt(hidden)

func cachedScaledEmbeddingRow(weights *TextWeights, tensorName string, token, hiddenSize int) ([]float32, error) {
	if weights == nil || hiddenSize <= 0 {
		return nil, fmt.Errorf("DiffusionGemma embedding cache invalid weights/hidden")
	}
	key := embeddingRowCacheKey{weights: uintptr(unsafe.Pointer(weights)), token: token, hidden: hiddenSize}
	if v, ok := embeddingRowCache.Load(key); ok {
		return v.([]float32), nil
	}
	row, dtype, shape, err := weights.RawTensorRow(tensorName, token)
	if err != nil {
		return nil, err
	}
	if len(shape) != 1 || shape[0] != hiddenSize {
		return nil, fmt.Errorf("DiffusionGemma embed row shape %v want [%d]", shape, hiddenSize)
	}
	out := make([]float32, hiddenSize)
	if err := decodeFloatRowTo(out, row, dtype); err != nil {
		return nil, err
	}
	embedScale := float32(math.Sqrt(float64(hiddenSize)))
	for i := range out {
		out[i] *= embedScale
	}
	actual, _ := embeddingRowCache.LoadOrStore(key, out)
	return actual.([]float32), nil
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
		row, err := cachedScaledEmbeddingRow(weights, plan.Globals.EmbedTokens.Name, token, hiddenSize)
		if err != nil {
			return err
		}
		copy(scratch.Hidden[i*hiddenSize:(i+1)*hiddenSize], row)
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
		if len(dst) == 0 {
			return nil
		}
		// Use SIMD BF16→F32 widen when available (AVX2)
		src := unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), len(dst))
		simd.BF16WidenToF32(dst, src)
		return nil
	case "F16":
		if len(raw) < len(dst)*2 {
			return fmt.Errorf("DiffusionGemma F16 row bytes=%d want %d", len(raw), len(dst)*2)
		}
		for i := range dst {
			dst[i] = half.F16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return nil
	case "F8_E4M3":
		return fmt.Errorf("DiffusionGemma unsupported generic float row dtype %s (use FP8TextWeights/GPUFP8Model path for quantized projections)", dtype)
	default:
		return fmt.Errorf("DiffusionGemma unsupported float row dtype %s", dtype)
	}
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
		if !simd.GELUExactMulTo(act, gate, up) {
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
	qAllLen, okQ := checked.MulInt(positions, qRows)
	kAllLen, okK := checked.MulInt(positions, kRows)
	vAllLen, okV := checked.MulInt(positions, vRows)
	if !okQ || !okK || !okV {
		return fmt.Errorf("DiffusionGemma attention buffer overflow positions=%d q=%d k=%d v=%d", positions, qRows, kRows, vRows)
	}
	qAll := make([]float32, qAllLen)
	kAll := make([]float32, kAllLen)
	vAll := make([]float32, vAllLen)
	attnCtx := make([]float32, qRows)
	out := make([]float32, hiddenSize)
	ropeHalf := headDim / 2
	ropeTheta := 10000.0
	var ropeFactors []float32
	if op.Type == "full_attention" {
		// llama.cpp: full-attention layers use n_rot_full=headDim and
		// rope_freqs.weight factors for proportional RoPE. FP8 safetensors omit
		// rope_freqs, so synthesize the same factors from config defaults.
		ropeTheta = 1000000.0
		factors, err := fullAttentionRoPEFactors(weights, fp, headDim)
		if err != nil {
			return err
		}
		ropeFactors = factors
	}
	ropeFreqs := simd.BuildRoPEFreqsWithFactors(ctx.EncoderSeqLen+positions, ropeHalf, headDim, ropeTheta, ropeFactors)
	for pos := 0; pos < positions; pos++ {
		hidden := scratch.Hidden[pos*hiddenSize : (pos+1)*hiddenSize]
		q := qAll[pos*qRows : (pos+1)*qRows]
		k := kAll[pos*kRows : (pos+1)*kRows]
		v := vAll[pos*vRows : (pos+1)*vRows]
		if !qM.gemvRows(q, hidden) || !kM.gemvRows(k, hidden) {
			return fmt.Errorf("DiffusionGemma attention Q/K GEMV rejected layer %d", op.Layer)
		}
		if lb.VProj != nil {
			if !vM.gemvRows(v, hidden) {
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
		keyNeed, okK := checked.MulInt(enc.SeqLen, kRows)
		valueNeed, okV := checked.MulInt(enc.SeqLen, vRows)
		if enc.KVHeads != kvHeads || enc.HeadDim != headDim || !okK || !okV || len(enc.Keys) < keyNeed || len(enc.Values) < valueNeed {
			return fmt.Errorf("DiffusionGemma encoder KV layer %d shape mismatch seq=%d kv_heads=%d head_dim=%d", op.Layer, enc.SeqLen, enc.KVHeads, enc.HeadDim)
		}
		encSeq = enc.SeqLen
	}
	totalKV := encSeq + positions
	if totalKV < encSeq {
		return fmt.Errorf("DiffusionGemma attention total KV overflow enc=%d positions=%d", encSeq, positions)
	}
	scores := make([]float32, totalKV)
	slidingWindow := 0
	if op.Type == "sliding_attention" {
		slidingWindow = scratch.SlidingWindow
		if slidingWindow <= 0 {
			// DiffusionGemma GGUF key attention.sliding_window is required by llama.cpp;
			// keep the published default as a fallback for legacy callers that built
			// ForwardBufferPlan before this field existed.
			slidingWindow = 1024
		}
	}
	for pos := 0; pos < positions; pos++ {
		for i := range attnCtx {
			attnCtx[i] = 0
		}
		for h := 0; h < heads; h++ {
			kvh := h / group
			q := qAll[pos*qRows+h*headDim : pos*qRows+(h+1)*headDim]
			canvasPromptLo := encSeq - slidingWindow + 1
			for j := 0; j < totalKV; j++ {
				if j < encSeq {
					// llama.cpp decode mask: sliding layers let canvas queries see
					// only the last (n_swa-1) prompt keys; global layers see all prompt.
					if slidingWindow > 0 && j < canvasPromptLo {
						scores[j] = float32(math.Inf(-1))
						continue
					}
					scores[j] = dot(q, enc.Keys[j*kRows+kvh*headDim:j*kRows+(kvh+1)*headDim])
					continue
				}
				canvasJ := j - encSeq
				// llama.cpp decode mask is bidirectional over the full canvas even
				// for sliding layers.
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
		if !oM.gemvRows(out, attnCtx) {
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
	gate := make([]float32, intermediate)
	up := make([]float32, intermediate)
	act := make([]float32, intermediate)
	out := make([]float32, hiddenSize)
	for off := 0; off < len(scratch.Hidden); off += hiddenSize {
		row := scratch.Hidden[off : off+hiddenSize]
		if !gateM.gemvRows(gate, row) {
			return fmt.Errorf("DiffusionGemma dense MLP gate GEMV rejected layer %d", op.Layer)
		}
		if !upM.gemvRows(up, row) {
			return fmt.Errorf("DiffusionGemma dense MLP up GEMV rejected layer %d", op.Layer)
		}
		if !simd.GELUExactMulTo(act, gate, up) {
			return fmt.Errorf("DiffusionGemma dense MLP activation rejected layer %d", op.Layer)
		}
		if !downM.gemvRows(out, act) {
			return fmt.Errorf("DiffusionGemma dense MLP down GEMV rejected layer %d", op.Layer)
		}
		copy(row, out)
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
	projW, projRows, projCols, err := loadFloatMatrix(weights, lb.RouterProj)
	if err != nil {
		return err
	}
	hiddenSize := len(scaleVec)
	numExperts := projRows
	if projCols != hiddenSize || numExperts <= 0 || len(scratch.Residual)%hiddenSize != 0 {
		return fmt.Errorf("DiffusionGemma router shape mismatch scale=%d proj=[%d,%d] residual=%d", hiddenSize, projRows, projCols, len(scratch.Residual))
	}
	normBuf := make([]float32, hiddenSize)
	if len(scratch.Experts) >= hiddenSize {
		normBuf = scratch.Experts[:hiddenSize]
	}
	scored := make([]float32, numExperts)
	if len(scratch.Router) >= numExperts {
		scored = scratch.Router[:numExperts]
	}
	positions := len(scratch.Residual) / hiddenSize
	for pos := 0; pos < positions; pos++ {
		resRow := scratch.Residual[pos*hiddenSize : (pos+1)*hiddenSize]
		copy(normBuf, resRow)
		if !simd.RMSNormNoScaleTo(normBuf, 1e-6) {
			return fmt.Errorf("DiffusionGemma router norm rejected")
		}
		scalarRootSize := float32(1.0 / math.Sqrt(float64(hiddenSize)))
		for i := range normBuf {
			normBuf[i] *= scaleVec[i] * scalarRootSize
		}
		if !simd.GemvRows(scored, normBuf, projW, numExperts, hiddenSize) {
			return fmt.Errorf("DiffusionGemma router GEMV rejected")
		}
		softmaxInPlace(scored)
		topK := scratch.TopKExperts
		if topK <= 0 {
			topK = len(scratch.TopKIDs) / positions
		}
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
		// llama.cpp build_moe_ffn clamps the selected-weight sum to the
		// smallest positive F16 value before normalizing (avoids divide-by-zero
		// and keeps underflow behavior identical to ggml).
		if topKSum < 6.103515625e-5 {
			topKSum = 6.103515625e-5
		}
		for i := range vals {
			if vals[i] > float32(math.Inf(-1)) {
				vals[i] /= topKSum
			}
		}
		if lb.RouterPerExpertScale != nil && scratch.GGUFExpertIndex == nil {
			// In safetensors this tensor is router.per_expert_scale; applying it
			// to the selected weights is algebraically equivalent to llama.cpp's
			// down_exps_s multiplication on each expert output. In GGUF mode the
			// same tensor arrives as blk.*.ffn_down_exps.scale and is already
			// applied inside GGUFExpertIndex, so do not double-apply it here.
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
	if scratch.GGUFExpertIndex != nil {
		return runGGUFCPUExpertsIndexed(op, weights, scratch, scratch.GGUFExpertIndex)
	}
	if scratch.FP8ExpertIndex != nil {
		return runFP8CPUExpertsIndexed(op, weights, scratch, scratch.FP8ExpertIndex)
	}
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
	if lb.ExpertsGateUpProj == nil || lb.ExpertsDownProj == nil || len(lb.ExpertsGateUpProj.Shape) != 3 || len(lb.ExpertsDownProj.Shape) != 3 {
		return fmt.Errorf("DiffusionGemma expert tensor bindings missing")
	}
	nExperts := lb.ExpertsGateUpProj.Shape[0]
	gateUpDim := lb.ExpertsGateUpProj.Shape[1]
	if nExperts <= 0 || gateUpDim <= 0 || gateUpDim%2 != 0 || lb.ExpertsGateUpProj.Shape[2] != hiddenSize {
		return fmt.Errorf("DiffusionGemma expert gate_up shape %v incompatible with hidden=%d", lb.ExpertsGateUpProj.Shape, hiddenSize)
	}
	intermediate := gateUpDim / 2
	if lb.ExpertsDownProj.Shape[0] != nExperts || lb.ExpertsDownProj.Shape[1] != hiddenSize || lb.ExpertsDownProj.Shape[2] != intermediate {
		return fmt.Errorf("DiffusionGemma expert down shape %v want [%d,%d,%d]", lb.ExpertsDownProj.Shape, nExperts, hiddenSize, intermediate)
	}

	positions := len(scratch.Residual) / hiddenSize
	for i := range scratch.MoeOut {
		scratch.MoeOut[i] = 0
	}
	normedRow := make([]float32, hiddenSize)
	gate := make([]float32, intermediate)
	up := make([]float32, intermediate)
	act := make([]float32, intermediate)
	expertOut := make([]float32, hiddenSize)
	topK := scratch.TopKExperts
	if topK <= 0 {
		topK = len(scratch.TopKIDs) / positions
	}

	// Collect unique expert IDs to decode only needed slices
	neededExperts := map[int]bool{}
	for pos := 0; pos < positions; pos++ {
		for k := 0; k < topK; k++ {
			id := scratch.TopKIDs[pos*topK+k]
			if id >= 0 && id < nExperts {
				neededExperts[id] = true
			}
		}
	}
	type expertWeights struct {
		gateW, upW, downW []float32
	}
	decoded := make(map[int]expertWeights, len(neededExperts))
	for expertID := range neededExperts {
		guSlice, guRows, _, err := loadExpertSlice(weights, lb.ExpertsGateUpProj, expertID)
		if err != nil {
			return err
		}
		dSlice, _, _, err := loadExpertSlice(weights, lb.ExpertsDownProj, expertID)
		if err != nil {
			return err
		}
		decoded[expertID] = expertWeights{
			gateW: guSlice[:guRows/2*hiddenSize],
			upW:   guSlice[guRows/2*hiddenSize:],
			downW: dSlice,
		}
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
			if !simd.GELUExactMulTo(act, gate, up) {
				return fmt.Errorf("DiffusionGemma expert activation rejected")
			}
			if !simd.GemvRows(expertOut, act, ew.downW, hiddenSize, intermediate) {
				return fmt.Errorf("DiffusionGemma expert down GEMV rejected")
			}
			for i := range dst {
				dst[i] += weight * expertOut[i]
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
	sliceElements, okElems := checked.MulInt(rows, cols)
	sliceBytes, okBytes := checked.MulInt(sliceElements, elemSize)
	start, okStart := checked.MulInt(expertID, sliceBytes)
	end := start + sliceBytes
	if rows <= 0 || cols <= 0 || !okElems || !okBytes || !okStart || end < start || end > len(raw) {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma expert %d byte range [%d,%d) exceeds %d shape=[%d,%d] dtype=%s", expertID, start, end, len(raw), rows, cols, dtype)
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
	sliceElements, okElems := checked.MulInt(rows, cols)
	sliceBytes, okBytes := checked.MulInt(sliceElements, 2)
	start, okStart := checked.MulInt(expertID, sliceBytes)
	end := start + sliceBytes
	if rows <= 0 || cols <= 0 || !okElems || !okBytes || !okStart || end < start || end > len(raw) {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma expert %d BF16 range [%d,%d) exceeds %d shape=[%d,%d]", expertID, start, end, len(raw), rows, cols)
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
	if m.rows >= 256 {
		return simd.GemvRowsParallel(out, x, m.f32, m.rows, m.cols)
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
	for pos := 0; pos < positions; pos++ {
		if len(scratch.Logits[pos]) < vocab {
			return nil, fmt.Errorf("DiffusionGemma self-conditioning logits row=%d len=%d want %d", pos, len(scratch.Logits[pos]), vocab)
		}
	}
	tempInv := scratch.SCTempInv
	if tempInv <= 0 {
		tempInv = 1.0
	}

	// Fast path: cache/dequantize embed_tokens to F32 once and reuse it across
	// steps. This mirrors llama.cpp's dg_ensure_sc_embT strategy (dequant once,
	// matmul every step) more closely than re-widening BF16 rows every step.
	if t, err := weights.CachedFloatTensor(fp.Globals.EmbedTokens.Name); err == nil && len(t.Shape) == 2 && t.Shape[0] == vocab && t.Shape[1] == hiddenSize && len(t.Data) >= vocab*hiddenSize {
		accumulateSelfConditioningF32Batched(out, scratch.Logits[:positions], t.Data, vocab, hiddenSize, tempInv)
	} else if err != nil {
		return nil, err
	} else if bf16Embed, bf16Shape, err := weights.RawBF16Tensor(fp.Globals.EmbedTokens.Name); err != nil {
		return nil, err
	} else if bf16Embed != nil && len(bf16Shape) == 2 && bf16Shape[0] == vocab && bf16Shape[1] == hiddenSize && len(bf16Embed) >= vocab*hiddenSize {
		accumulateSelfConditioningBF16Batched(out, scratch.Logits[:positions], bf16Embed, vocab, hiddenSize, tempInv)
	} else {
		row := make([]float32, hiddenSize)
		for pos := 0; pos < positions; pos++ {
			if err := accumulateSelfConditioningRowRaw(out[pos*hiddenSize:(pos+1)*hiddenSize], scratch.Logits[pos][:vocab], weights, fp.Globals.EmbedTokens.Name, vocab, hiddenSize, tempInv, row); err != nil {
				return nil, err
			}
		}
	}
	embedScale := float32(math.Sqrt(float64(hiddenSize)))
	for i := range out {
		out[i] *= embedScale
	}
	return out, nil
}

func selfConditioningSoftmaxStats(logits []float32, tempInv float32) (float32, float64, bool) {
	maxVal := float32(math.Inf(-1))
	for _, v := range logits {
		if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
			continue
		}
		z := v * tempInv
		if z > maxVal {
			maxVal = z
		}
	}
	if math.IsInf(float64(maxVal), -1) {
		return maxVal, 0, false
	}
	var sumExp float64
	for _, v := range logits {
		if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
			continue
		}
		sumExp += math.Exp(float64(v*tempInv - maxVal))
	}
	return maxVal, sumExp, sumExp > 0 && !math.IsNaN(sumExp)
}

func accumulateSelfConditioningBF16Batched(out []float32, logits [][]float32, bf16Embed []uint16, vocab, hiddenSize int, tempInv float32) {
	positions := len(logits)
	maxVals := make([]float32, positions)
	invSums := make([]float64, positions)
	active := make([]bool, positions)
	for pos := 0; pos < positions; pos++ {
		maxVal, sumExp, ok := selfConditioningSoftmaxStats(logits[pos][:vocab], tempInv)
		maxVals[pos] = maxVal
		if ok {
			invSums[pos] = 1.0 / sumExp
			active[pos] = true
		}
	}
	nWorkers := runtime.GOMAXPROCS(0)
	if nWorkers > vocab {
		nWorkers = vocab
	}
	if nWorkers < 1 {
		nWorkers = 1
	}
	chunk := (vocab + nWorkers - 1) / nWorkers
	workerOuts := make([][]float32, nWorkers)
	var wg sync.WaitGroup
	for w := 0; w < nWorkers; w++ {
		start := w * chunk
		end := start + chunk
		if end > vocab {
			end = vocab
		}
		if start >= end {
			continue
		}
		workerOuts[w] = make([]float32, len(out))
		wg.Add(1)
		go func(worker, v0, v1 int) {
			defer wg.Done()
			row := make([]float32, hiddenSize)
			local := workerOuts[worker]
			for vocabID := v0; vocabID < v1; vocabID++ {
				start := vocabID * hiddenSize
				simd.BF16WidenToF32(row, bf16Embed[start:start+hiddenSize])
				for pos := 0; pos < positions; pos++ {
					if !active[pos] {
						continue
					}
					v := logits[pos][vocabID]
					if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
						continue
					}
					prob := float32(math.Exp(float64(v*tempInv-maxVals[pos])) * invSums[pos])
					if prob != 0 {
						simd.Saxpy(prob, row, local[pos*hiddenSize:(pos+1)*hiddenSize])
					}
				}
			}
		}(w, start, end)
	}
	wg.Wait()
	for _, local := range workerOuts {
		if len(local) == len(out) {
			simd.Saxpy(1, local, out)
		}
	}
}

func accumulateSelfConditioningF32Batched(out []float32, logits [][]float32, embed []float32, vocab, hiddenSize int, tempInv float32) {
	positions := len(logits)
	maxVals := make([]float32, positions)
	invSums := make([]float64, positions)
	active := make([]bool, positions)
	for pos := 0; pos < positions; pos++ {
		maxVal, sumExp, ok := selfConditioningSoftmaxStats(logits[pos][:vocab], tempInv)
		maxVals[pos] = maxVal
		if ok {
			invSums[pos] = 1.0 / sumExp
			active[pos] = true
		}
	}
	nWorkers := runtime.GOMAXPROCS(0)
	if nWorkers > vocab {
		nWorkers = vocab
	}
	if nWorkers < 1 {
		nWorkers = 1
	}
	chunk := (vocab + nWorkers - 1) / nWorkers
	workerOuts := make([][]float32, nWorkers)
	var wg sync.WaitGroup
	for w := 0; w < nWorkers; w++ {
		start := w * chunk
		end := start + chunk
		if end > vocab {
			end = vocab
		}
		if start >= end {
			continue
		}
		workerOuts[w] = make([]float32, len(out))
		wg.Add(1)
		go func(worker, v0, v1 int) {
			defer wg.Done()
			local := workerOuts[worker]
			for vocabID := v0; vocabID < v1; vocabID++ {
				row := embed[vocabID*hiddenSize : (vocabID+1)*hiddenSize]
				for pos := 0; pos < positions; pos++ {
					if !active[pos] {
						continue
					}
					v := logits[pos][vocabID]
					if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
						continue
					}
					prob := float32(math.Exp(float64(v*tempInv-maxVals[pos])) * invSums[pos])
					if prob != 0 {
						simd.Saxpy(prob, row, local[pos*hiddenSize:(pos+1)*hiddenSize])
					}
				}
			}
		}(w, start, end)
	}
	wg.Wait()
	for _, local := range workerOuts {
		if len(local) == len(out) {
			simd.Saxpy(1, local, out)
		}
	}
}

func accumulateSelfConditioningRowRaw(dst []float32, logits []float32, weights *TextWeights, embedName string, vocab, hiddenSize int, tempInv float32, row []float32) error {
	maxVal, sumExp, ok := selfConditioningSoftmaxStats(logits, tempInv)
	if !ok {
		return nil
	}
	invSum := 1.0 / sumExp
	for vocabID, v := range logits {
		if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
			continue
		}
		prob := float32(math.Exp(float64(v*tempInv-maxVal)) * invSum)
		if prob == 0 {
			continue
		}
		raw, dtype, shape, err := weights.RawTensorRow(embedName, vocabID)
		if err != nil {
			return err
		}
		if len(shape) != 1 || shape[0] != hiddenSize {
			return fmt.Errorf("DiffusionGemma self-conditioning row shape %v want [%d]", shape, hiddenSize)
		}
		if err := decodeFloatRowTo(row, raw, dtype); err != nil {
			return err
		}
		simd.Saxpy(prob, row, dst)
	}
	return nil
}

func applyFinalLogitSoftcapping(scratch ForwardScratch, positions, vocab int) {
	if c := scratch.FinalLogitSoftcapping; c > 0 {
		invC := float32(1.0) / c
		for pos := 0; pos < positions && pos < len(scratch.Logits); pos++ {
			row := scratch.Logits[pos]
			if vocab > 0 && vocab < len(row) {
				row = row[:vocab]
			}
			for i, v := range row {
				if v > float32(math.Inf(-1)) {
					row[i] = c * float32(math.Tanh(float64(v*invC)))
				}
			}
		}
	}
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
	if qm := weights.ggufTokenEmbd; qm != nil && qm.OutDim == vocab && qm.InDim == hiddenSize {
		row := make([]float32, hiddenSize)
		for vocabID := 0; vocabID < vocab; vocabID++ {
			if err := qm.DequantRowTo(row, vocabID); err != nil {
				return err
			}
			for pos := 0; pos < positions; pos++ {
				score := dot(scratch.Hidden[pos*hiddenSize:(pos+1)*hiddenSize], row)
				if topK > 0 {
					insertTopK(topIDs[pos], topVals[pos], vocabID, score)
				} else {
					if len(scratch.Logits[pos]) < vocab {
						return fmt.Errorf("DiffusionGemma LM head logits row=%d len=%d want %d", pos, len(scratch.Logits[pos]), vocab)
					}
					scratch.Logits[pos][vocabID] = score
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
		applyFinalLogitSoftcapping(scratch, positions, vocab)
		return nil
	}
	if topK > 0 {
		// Try BF16 native LM head (half the memory, direct mmap scan)
		bf16Embed, bf16Shape, err := weights.RawBF16Tensor(fp.Globals.EmbedTokens.Name)
		if err != nil {
			return err
		}
		if bf16Embed != nil && len(bf16Shape) == 2 && bf16Shape[0] == vocab && bf16Shape[1] == hiddenSize {
			for pos := 0; pos < positions; pos++ {
				hidden := scratch.Hidden[pos*hiddenSize : (pos+1)*hiddenSize]
				hiddenBF16 := simd.BF16FromF32Slice(hidden)
				for vocabID := 0; vocabID < vocab; vocabID++ {
					row := bf16Embed[vocabID*hiddenSize : (vocabID+1)*hiddenSize]
					score := simd.BF16DotAsm(row, hiddenBF16)
					insertTopK(topIDs[pos], topVals[pos], vocabID, score)
				}
			}
		} else {
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
				scratch.Logits[pos][vocabID] = dot(scratch.Hidden[pos*hiddenSize:(pos+1)*hiddenSize], row)
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
	// Final logit softcapping: tanh(x/c)*c (same as llama.cpp)
	applyFinalLogitSoftcapping(scratch, positions, vocab)
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
