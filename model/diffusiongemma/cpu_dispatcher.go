package diffusiongemma

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
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

type expertAssignment struct {
	pos    int
	weight float32
}

type q80PrefetchResult struct {
	count   int
	err     error
	elapsed time.Duration
}

type q80LayerPrefetcher struct {
	weights        *TextWeights
	includeExperts bool
	progress       bool
	mu             sync.Mutex
	done           map[int]chan q80PrefetchResult
}

func newQ80LayerPrefetcher(weights *TextWeights, progress bool) *q80LayerPrefetcher {
	if weights == nil || !k3Enabled() || !k3A100Q8Enabled() || !k3Q80PrefetchEnabled() {
		return nil
	}
	return &q80LayerPrefetcher{weights: weights, includeExperts: k3Q80PrefetchExperts(), progress: progress, done: map[int]chan q80PrefetchResult{}}
}

func (p *q80LayerPrefetcher) start(layer int) {
	if p == nil || p.weights == nil || layer < 0 || layer >= len(p.weights.Layers) {
		return
	}
	p.mu.Lock()
	if _, ok := p.done[layer]; ok {
		p.mu.Unlock()
		return
	}
	ch := make(chan q80PrefetchResult, 1)
	p.done[layer] = ch
	p.mu.Unlock()
	go func() {
		started := time.Now()
		count, err := p.weights.PreloadLayerQ80(layer, p.includeExperts)
		ch <- q80PrefetchResult{count: count, err: err, elapsed: time.Since(started)}
	}()
}

func startQ80TransposedBindingPrefetch(weights *TextWeights, binding *TensorBinding) chan q80PrefetchResult {
	if weights == nil || binding == nil || !k3Enabled() || !k3A100Q8Enabled() {
		return nil
	}
	ch := make(chan q80PrefetchResult, 1)
	go func() {
		started := time.Now()
		ok, err := k3PreloadQ80TransposedBinding(weights, binding)
		count := 0
		if ok {
			count = 1
		}
		ch <- q80PrefetchResult{count: count, err: err, elapsed: time.Since(started)}
	}()
	return ch
}

func startQ80BindingPrefetch(weights *TextWeights, binding *TensorBinding) chan q80PrefetchResult {
	if weights == nil || binding == nil || !k3Enabled() || !k3A100Q8Enabled() {
		return nil
	}
	ch := make(chan q80PrefetchResult, 1)
	go func() {
		started := time.Now()
		ok, err := k3PreloadQ80Binding(weights, binding)
		count := 0
		if ok {
			count = 1
		}
		ch <- q80PrefetchResult{count: count, err: err, elapsed: time.Since(started)}
	}()
	return ch
}

func (p *q80LayerPrefetcher) wait(layer int) error {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	ch := p.done[layer]
	p.mu.Unlock()
	if ch == nil {
		return nil
	}
	res := <-ch
	if p.progress {
		fmt.Fprintf(os.Stderr, "DiffusionGemma K3 Q80 prefetch: completed layer=%d tensors=%d include_experts=%v q80_entries=%d q80_bytes=%d elapsed=%s\n", layer, res.count, p.includeExperts, p.weights.Q80CacheEntries(), p.weights.Q80CacheBytes(), res.elapsed.Round(time.Millisecond))
	}
	return res.err
}

type ForwardScratch struct {
	Hidden         []float32
	Residual       []float32
	MlpOut         []float32
	MoeOut         []float32
	Logits         [][]float32
	LogitFlat      []float32
	VocabSize      int
	Router         []float32
	Experts        []float32
	TopKIDs        []int
	TopKVals       []float32
	TopKExperts    int
	LMHeadTopK     int
	ExpertPrefetch *k3SelectedExpertPrefetch
	ExpertAsync    chan error
}

func NewForwardScratch(buffers ForwardBufferPlan) ForwardScratch {
	topKSlots := maxNonNegative(buffers.CanvasLength * buffers.TopKExperts)
	hiddenSize := maxNonNegative(buffers.Hidden)
	logitFlat, logits := makeLogitRows(buffers.CanvasLength, buffers.VocabSize)
	return ForwardScratch{Hidden: make([]float32, hiddenSize), Residual: make([]float32, hiddenSize), MlpOut: make([]float32, hiddenSize), MoeOut: make([]float32, hiddenSize), Router: make([]float32, maxNonNegative(buffers.Router)), Experts: make([]float32, maxNonNegative(buffers.Experts)), TopKIDs: make([]int, topKSlots), TopKVals: make([]float32, topKSlots), TopKExperts: maxNonNegative(buffers.TopKExperts), LogitFlat: logitFlat, VocabSize: maxNonNegative(buffers.VocabSize), Logits: logits}
}

func makeLogitRows(rows, cols int) ([]float32, [][]float32) {
	if rows <= 0 || cols <= 0 {
		return nil, nil
	}
	flat := make([]float32, rows*cols)
	out := make([][]float32, rows)
	for i := range out {
		out[i] = flat[i*cols : (i+1)*cols]
	}
	return flat, out
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
		if buffers.VocabSize > 0 && len(scratch.LogitFlat) >= actualPositions*buffers.VocabSize {
			scratch.LogitFlat = scratch.LogitFlat[:actualPositions*buffers.VocabSize]
		}
	}
	actualTopK := actualPositions * buffers.TopKExperts
	if actualTopK > 0 && actualTopK < len(scratch.TopKIDs) {
		scratch.TopKIDs = scratch.TopKIDs[:actualTopK]
		scratch.TopKVals = scratch.TopKVals[:actualTopK]
	}
	var scEmbTPrefetch chan q80PrefetchResult
	if k3Enabled() && k3A100Q8Enabled() && ctx.Graph.Phase == ExecutionGraphDecode {
		fp := weights.ForwardPlan()
		scEmbTPrefetch = startQ80TransposedBindingPrefetch(weights, fp.Globals.EmbedTokens)
	}
	for _, op := range ops.Prefix {
		if diffusionGemmaTimingEnabled() {
			started := time.Now()
			err := dispatchPrefixOp(op, ctx, weights, scratch)
			fmt.Fprintf(os.Stderr, "timing diffusiongemma prefix op=%s elapsed=%s q80_entries=%d q80_bytes=%d\n", op, time.Since(started).Round(time.Millisecond), weights.Q80CacheEntries(), weights.Q80CacheBytes())
			if err != nil {
				return ForwardOutput{}, err
			}
			continue
		}
		if err := dispatchPrefixOp(op, ctx, weights, scratch); err != nil {
			return ForwardOutput{}, err
		}
	}
	var lmHeadPrefetch chan q80PrefetchResult
	if k3A100LMHeadEnabled() && k3A100LMHeadPrefetchEnabled() {
		fp := weights.ForwardPlan()
		lmHeadPrefetch = startQ80BindingPrefetch(weights, fp.Globals.EmbedTokens)
	}
	prefetcher := newQ80LayerPrefetcher(weights, d.Progress)
	if prefetcher != nil {
		prefetcher.start(0)
	}
	currentLayer := -1
	completedLayers := 0
	exitedByMaxLayers := false
	layerStarted := time.Now()
	for _, op := range ops.Layers {
		if currentLayer >= 0 && op.Layer != currentLayer {
			completedLayers++
			if d.Progress {
				fmt.Fprintf(os.Stderr, "DiffusionGemma CPU dispatcher: completed layer=%d cache_entries=%d cache_bytes=%d q80_entries=%d q80_bytes=%d elapsed=%s\n", currentLayer, weights.FloatCacheEntries(), weights.FloatCacheBytes(), weights.Q80CacheEntries(), weights.Q80CacheBytes(), time.Since(layerStarted).Round(time.Millisecond))
			}
			if !d.SkipEviction && currentLayer >= d.ResidentLayerPrefix {
				weights.EvictLayer(currentLayer)
			}
			if d.MaxLayers > 0 && completedLayers >= d.MaxLayers {
				exitedByMaxLayers = true
				break
			}
			layerStarted = time.Now()
		}
		if op.Layer != currentLayer {
			currentLayer = op.Layer
			if err := prefetcher.wait(currentLayer); err != nil {
				return ForwardOutput{}, err
			}
			if d.MaxLayers <= 0 || completedLayers+1 < d.MaxLayers {
				prefetcher.start(currentLayer + 1)
			}
			if d.Progress {
				fmt.Fprintf(os.Stderr, "DiffusionGemma CPU dispatcher: starting layer=%d\n", currentLayer)
			}
		}
		if err := dispatchLayerOp(op, ctx, weights, &scratch); err != nil {
			return ForwardOutput{}, err
		}
	}
	if currentLayer >= 0 && !exitedByMaxLayers {
		completedLayers++
		if d.Progress {
			fmt.Fprintf(os.Stderr, "DiffusionGemma CPU dispatcher: completed layer=%d cache_entries=%d cache_bytes=%d q80_entries=%d q80_bytes=%d elapsed=%s\n", currentLayer, weights.FloatCacheEntries(), weights.FloatCacheBytes(), weights.Q80CacheEntries(), weights.Q80CacheBytes(), time.Since(layerStarted).Round(time.Millisecond))
		}
	}
	if currentLayer >= 0 && currentLayer >= d.ResidentLayerPrefix {
		weights.EvictLayer(currentLayer)
	}
	if d.MaxLayers > 0 && !d.TailAfterMaxLayers {
		return ForwardOutput{Logits: scratch.Logits, SelfConditioning: ctx.SelfConditioning}, nil
	}
	timing := diffusionGemmaTimingEnabled()
	for _, op := range ops.Tail {
		if d.Progress {
			fmt.Fprintf(os.Stderr, "DiffusionGemma CPU dispatcher: starting tail op=%s\n", op)
		}
		if op == OpLMHead && lmHeadPrefetch != nil {
			res := <-lmHeadPrefetch
			if d.Progress || timing {
				fmt.Fprintf(os.Stderr, "timing diffusiongemma lm_head_prefetch tensors=%d q80_entries=%d q80_bytes=%d elapsed=%s\n", res.count, weights.Q80CacheEntries(), weights.Q80CacheBytes(), res.elapsed.Round(time.Millisecond))
			}
			if res.err != nil {
				return ForwardOutput{}, res.err
			}
			lmHeadPrefetch = nil
		}
		started := time.Now()
		if err := dispatchTailOp(op, weights, scratch); err != nil {
			return ForwardOutput{}, err
		}
		elapsed := time.Since(started)
		if timing {
			fmt.Fprintf(os.Stderr, "timing diffusiongemma tail op=%s elapsed=%s q80_entries=%d q80_bytes=%d\n", op, elapsed.Round(time.Millisecond), weights.Q80CacheEntries(), weights.Q80CacheBytes())
		}
		if d.Progress {
			fmt.Fprintf(os.Stderr, "DiffusionGemma CPU dispatcher: completed tail op=%s cache_entries=%d cache_bytes=%d q80_entries=%d q80_bytes=%d elapsed=%s\n", op, weights.FloatCacheEntries(), weights.FloatCacheBytes(), weights.Q80CacheEntries(), weights.Q80CacheBytes(), elapsed.Round(time.Millisecond))
		}
	}
	if scEmbTPrefetch != nil {
		res := <-scEmbTPrefetch
		if d.Progress || timing {
			fmt.Fprintf(os.Stderr, "timing diffusiongemma sc_embT_prefetch tensors=%d q80_entries=%d q80_bytes=%d elapsed=%s\n", res.count, weights.Q80CacheEntries(), weights.Q80CacheBytes(), res.elapsed.Round(time.Millisecond))
		}
		if res.err != nil {
			return ForwardOutput{}, res.err
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
	return ForwardOutput{Logits: scratch.Logits}, nil

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
			dst[i] = diffusionGemmaFP8E4M3Table[raw[i]]
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
	selfConditioning := ctx.SelfConditioning
	if len(selfConditioning) == 0 && len(ctx.SelfConditioningLogits) > 0 {
		tempInv := float32(1)
		if ctx.SelfConditioningTemperature > 0 {
			tempInv = float32(1.0 / ctx.SelfConditioningTemperature)
		}
		built, err := buildSelfConditioningFromLogitRows(weights, ctx.SelfConditioningLogits, tempInv)
		if err != nil {
			return err
		}
		selfConditioning = built
	}
	if len(selfConditioning) == 0 {
		for off := 0; off < len(scratch.Hidden); off += hiddenSize {
			if !simd.RMSNormNoScaleTo(scratch.Hidden[off:off+hiddenSize], 1e-6) {
				return fmt.Errorf("DiffusionGemma self-conditioning post norm rejected row at offset %d", off)
			}
		}
		return nil
	}
	if len(selfConditioning) != len(scratch.Hidden) {
		return fmt.Errorf("DiffusionGemma self-conditioning len=%d want %d", len(selfConditioning), len(scratch.Hidden))
	}
	preNorm, err := loadFloatVector(weights, fp.Globals.SelfCondPreNorm)
	if err != nil {
		return err
	}
	if fp.Globals.SelfCondGateProj == nil || fp.Globals.SelfCondUpProj == nil || fp.Globals.SelfCondDownProj == nil || len(fp.Globals.SelfCondGateProj.Shape) != 2 || len(fp.Globals.SelfCondUpProj.Shape) != 2 || len(fp.Globals.SelfCondDownProj.Shape) != 2 {
		return fmt.Errorf("DiffusionGemma self-conditioning missing projection bindings")
	}
	gateRows, gateCols := fp.Globals.SelfCondGateProj.Shape[0], fp.Globals.SelfCondGateProj.Shape[1]
	upRows, upCols := fp.Globals.SelfCondUpProj.Shape[0], fp.Globals.SelfCondUpProj.Shape[1]
	downRows, downCols := fp.Globals.SelfCondDownProj.Shape[0], fp.Globals.SelfCondDownProj.Shape[1]
	if len(preNorm) != hiddenSize || gateCols != hiddenSize || upCols != hiddenSize || gateRows != upRows || downRows != hiddenSize || downCols != gateRows {
		return fmt.Errorf("DiffusionGemma self-conditioning shape mismatch")
	}
	positions := len(scratch.Hidden) / hiddenSize
	intermediate := gateRows
	condAll := make([]float32, positions*hiddenSize)
	for pos := 0; pos < positions; pos++ {
		cond := condAll[pos*hiddenSize : (pos+1)*hiddenSize]
		copy(cond, selfConditioning[pos*hiddenSize:(pos+1)*hiddenSize])
		if !simd.RMSNormTo(cond, preNorm, 1e-6) {
			return fmt.Errorf("DiffusionGemma self-conditioning pre norm rejected row %d", pos)
		}
	}
	gateAll := make([]float32, positions*intermediate)
	upAll := make([]float32, positions*intermediate)
	if done, err := k3Gemm2RowsQ80(gateAll, upAll, condAll, positions, weights, fp.Globals.SelfCondGateProj, fp.Globals.SelfCondUpProj); err != nil {
		return err
	} else if !done {
		gateW, _, _, err := loadFloatMatrix(weights, fp.Globals.SelfCondGateProj)
		if err != nil {
			return err
		}
		upW, _, _, err := loadFloatMatrix(weights, fp.Globals.SelfCondUpProj)
		if err != nil {
			return err
		}
		if !simd.GemmRows(gateAll, condAll, gateW, positions, intermediate, hiddenSize) || !simd.GemmRows(upAll, condAll, upW, positions, intermediate, hiddenSize) {
			return fmt.Errorf("DiffusionGemma self-conditioning gate/up GEMM rejected")
		}
	}
	actAll := make([]float32, positions*intermediate)
	for pos := 0; pos < positions; pos++ {
		if !simd.GELUTanhMulTo(actAll[pos*intermediate:(pos+1)*intermediate], gateAll[pos*intermediate:(pos+1)*intermediate], upAll[pos*intermediate:(pos+1)*intermediate]) {
			return fmt.Errorf("DiffusionGemma self-conditioning activation rejected")
		}
	}
	signalAll := make([]float32, positions*hiddenSize)
	if done, err := k3GemmRowsQ80(signalAll, actAll, positions, weights, fp.Globals.SelfCondDownProj); err != nil {
		return err
	} else if !done {
		downW, _, _, err := loadFloatMatrix(weights, fp.Globals.SelfCondDownProj)
		if err != nil {
			return err
		}
		if !simd.GemmRows(signalAll, actAll, downW, positions, hiddenSize, intermediate) {
			return fmt.Errorf("DiffusionGemma self-conditioning down GEMM rejected")
		}
	}
	for pos := 0; pos < positions; pos++ {
		row := scratch.Hidden[pos*hiddenSize : (pos+1)*hiddenSize]
		signal := signalAll[pos*hiddenSize : (pos+1)*hiddenSize]
		k3SaxpyV(1, signal, row)
		if !simd.RMSNormNoScaleTo(row, 1e-6) {
			return fmt.Errorf("DiffusionGemma self-conditioning post norm rejected row %d", pos)
		}
	}
	return nil
}

func diffusionGemmaTimingEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_TIMING")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func dispatchLayerOp(op LayerOp, ctx ForwardContext, weights *TextWeights, scratch *ForwardScratch) error {
	if !diffusionGemmaTimingEnabled() {
		return dispatchLayerOpInner(op, ctx, weights, scratch)
	}
	started := time.Now()
	err := dispatchLayerOpInner(op, ctx, weights, scratch)
	fmt.Fprintf(os.Stderr, "timing diffusiongemma layer=%d op=%s type=%s elapsed=%s q80_entries=%d q80_bytes=%d\n", op.Layer, op.Kind, op.Type, time.Since(started).Round(time.Millisecond), weights.Q80CacheEntries(), weights.Q80CacheBytes())
	return err
}

func dispatchLayerOpInner(op LayerOp, ctx ForwardContext, weights *TextWeights, scratch *ForwardScratch) error {
	switch op.Kind {
	case OpInputNorm:
		copy(scratch.Residual, scratch.Hidden)
		return runLayerRMSNorm(op, weights, *scratch, func(lb TextLayerBindings) *TensorBinding { return lb.InputLayerNorm })
	case OpSelfAttention:
		return runSelfAttention(op, ctx, weights, *scratch)
	case OpPostAttention:
		if err := runLayerRMSNorm(op, weights, *scratch, func(lb TextLayerBindings) *TensorBinding { return lb.PostAttentionLayerNorm }); err != nil {
			return err
		}
		for i := range scratch.Hidden {
			scratch.Hidden[i] += scratch.Residual[i]
		}
		return nil
	case OpDenseMLP:
		return runDenseMLP(op, weights, *scratch)
	case OpPreMoE:
		copy(scratch.Residual, scratch.Hidden)
		return runLayerRMSNorm(op, weights, *scratch, func(lb TextLayerBindings) *TensorBinding { return lb.PreFFNLayerNorm })
	case OpRouter:
		if err := runRouterFromResidual(op, weights, *scratch); err != nil {
			return err
		}
		fp := weights.ForwardPlan()
		if op.Layer >= 0 && op.Layer < len(fp.Layers) {
			scratch.ExpertPrefetch = k3StartSelectedExpertQ80Prefetch(weights, fp.Layers[op.Layer], *scratch)
			expertOp := LayerOp{Layer: op.Layer, Type: op.Type, Kind: OpExperts}
			snapshot := *scratch
			scratch.ExpertAsync = make(chan error, 1)
			go func() {
				if snapshot.ExpertPrefetch != nil {
					if err := snapshot.ExpertPrefetch.Wait(weights, diffusionGemmaTimingEnabled()); err != nil {
						scratch.ExpertAsync <- err
						return
					}
				}
				scratch.ExpertAsync <- runExpertsFromResidual(expertOp, weights, snapshot)
			}()
		}
		return nil
	case OpExperts:
		if scratch.ExpertAsync != nil {
			err := <-scratch.ExpertAsync
			scratch.ExpertAsync = nil
			scratch.ExpertPrefetch = nil
			return err
		}
		if scratch.ExpertPrefetch != nil {
			if err := scratch.ExpertPrefetch.Wait(weights, diffusionGemmaTimingEnabled()); err != nil {
				return err
			}
			scratch.ExpertPrefetch = nil
		}
		return runExpertsFromResidual(op, weights, *scratch)
	case OpPostMoE:
		if err := runCombineMlpMoe(op, weights, *scratch); err != nil {
			return err
		}
		for i := range scratch.Hidden {
			scratch.Hidden[i] += scratch.Residual[i]
		}
		return nil
	case OpLayerScalar:
		return runLayerScalar(op, weights, *scratch)
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
	if lb.QProj == nil || lb.KProj == nil || lb.OProj == nil || len(lb.QProj.Shape) != 2 || len(lb.KProj.Shape) != 2 || len(lb.OProj.Shape) != 2 {
		return fmt.Errorf("DiffusionGemma attention missing projection bindings layer %d", op.Layer)
	}
	qRows, hiddenSize := lb.QProj.Shape[0], lb.QProj.Shape[1]
	kRows, kCols := lb.KProj.Shape[0], lb.KProj.Shape[1]
	if kCols != hiddenSize {
		return fmt.Errorf("DiffusionGemma attention K shape %v hidden=%d", lb.KProj.Shape, hiddenSize)
	}
	var qM, kM, vM, oM *mixedMatrix
	vRows := kRows
	if lb.VProj != nil {
		if len(lb.VProj.Shape) != 2 || lb.VProj.Shape[1] != hiddenSize {
			return fmt.Errorf("DiffusionGemma attention V shape %v hidden=%d", lb.VProj.Shape, hiddenSize)
		}
		vRows = lb.VProj.Shape[0]
	}
	if lb.OProj.Shape[0] != hiddenSize || lb.OProj.Shape[1] != qRows {
		return fmt.Errorf("DiffusionGemma attention O shape %v q=%d hidden=%d", lb.OProj.Shape, qRows, hiddenSize)
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
	timing := diffusionGemmaTimingEnabled()
	phaseStart := time.Now()
	var qkvElapsed, normRopeElapsed, contextElapsed, oElapsed time.Duration
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
	if !qDone {
		var err error
		qM, err = loadMixedMatrix(weights, lb.QProj)
		if err != nil {
			return err
		}
	}
	if !kDone {
		var err error
		kM, err = loadMixedMatrix(weights, lb.KProj)
		if err != nil {
			return err
		}
	}
	if lb.VProj != nil && !vDone {
		var err error
		vM, err = loadMixedMatrix(weights, lb.VProj)
		if err != nil {
			return err
		}
	}
	if timing {
		qkvElapsed = time.Since(phaseStart)
		phaseStart = time.Now()
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
	if timing {
		normRopeElapsed = time.Since(phaseStart)
		phaseStart = time.Now()
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
	if timing {
		contextElapsed = time.Since(phaseStart)
		phaseStart = time.Now()
	}
	if done, err := k3GemmRowsQ80(scratch.Hidden, attnAll, positions, weights, lb.OProj); err != nil {
		return err
	} else if !done {
		var err error
		oM, err = loadMixedMatrix(weights, lb.OProj)
		if err != nil {
			return err
		}
		out := make([]float32, hiddenSize)
		for pos := 0; pos < positions; pos++ {
			attnCtx := attnAll[pos*qRows : (pos+1)*qRows]
			if !oM.gemvRows(out, attnCtx) {
				return fmt.Errorf("DiffusionGemma attention O GEMV rejected layer %d", op.Layer)
			}
			copy(scratch.Hidden[pos*hiddenSize:(pos+1)*hiddenSize], out)
		}
	}
	if timing {
		oElapsed = time.Since(phaseStart)
		fmt.Fprintf(os.Stderr, "timing diffusiongemma attention layer=%d qkv=%s norm_rope=%s context=%s o_proj=%s\n", op.Layer, qkvElapsed.Round(time.Millisecond), normRopeElapsed.Round(time.Millisecond), contextElapsed.Round(time.Millisecond), oElapsed.Round(time.Millisecond))
	}
	return nil
}

func runAttentionContextK3(attnAll, qAll, kAll, vAll []float32, enc EncoderKVLayer, positions, heads, kvHeads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow int) {
	if positions <= 0 || heads <= 0 || headDim <= 0 {
		return
	}
	if k3FlashAttentionEnabled() {
		runFlashAttentionContextK3(attnAll, qAll, kAll, vAll, enc, positions, heads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow)
		return
	}
	runMaterializedAttentionContextK3(attnAll, qAll, kAll, vAll, enc, positions, heads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow)
}

func k3FlashAttentionEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_K3_FLASH_ATTENTION")))
	if v == "0" || v == "false" || v == "no" || v == "off" {
		return false
	}
	return true
}

func runFlashAttentionContextK3(attnAll, qAll, kAll, vAll []float32, enc EncoderKVLayer, positions, heads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow int) {
	for i := range attnAll {
		attnAll[i] = 0
	}
	work := func(start, end int) {
		acc := make([]float32, headDim)
		for task := start; task < end; task++ {
			pos := task / heads
			h := task - pos*heads
			kvh := h / group
			q := qAll[pos*qRows+h*headDim : pos*qRows+(h+1)*headDim]
			dst := attnAll[pos*qRows+h*headDim : pos*qRows+(h+1)*headDim]
			for i := range acc {
				acc[i] = 0
			}
			m := float32(math.Inf(-1))
			l := float32(0)
			update := func(score float32, vv []float32) {
				s := score
				if l == 0 {
					copy(acc, vv)
					m = s
					l = 1
					return
				}
				if s <= m {
					w := float32(math.Exp(float64(s - m)))
					for i, v := range vv {
						acc[i] += v * w
					}
					l += w
					return
				}
				scale := float32(math.Exp(float64(m - s)))
				for i, v := range vv {
					acc[i] = acc[i]*scale + v
				}
				l = l*scale + 1
				m = s
			}
			for j := 0; j < encSeq; j++ {
				if !promptAllowedForSlidingDecode(j, encSeq, slidingWindow) {
					continue
				}
				update(k3Dot(q, enc.Keys[j*kRows+kvh*headDim:j*kRows+(kvh+1)*headDim]), enc.Values[j*vRows+kvh*headDim:j*vRows+(kvh+1)*headDim])
			}
			for canvasJ := 0; canvasJ < positions; canvasJ++ {
				if slidingWindow > 0 && absInt(pos-canvasJ) >= slidingWindow {
					continue
				}
				update(k3Dot(q, kAll[canvasJ*kRows+kvh*headDim:canvasJ*kRows+(kvh+1)*headDim]), vAll[canvasJ*vRows+kvh*headDim:canvasJ*vRows+(kvh+1)*headDim])
			}
			if l == 0 {
				continue
			}
			inv := float32(1) / l
			for i := range dst {
				dst[i] = acc[i] * inv
			}
		}
	}
	parallelizeAttentionTasks(positions*heads, work)
}

func runMaterializedAttentionContextK3(attnAll, qAll, kAll, vAll []float32, enc EncoderKVLayer, positions, heads, headDim, qRows, kRows, vRows, encSeq, group, slidingWindow int) {
	for i := range attnAll {
		attnAll[i] = 0
	}
	totalKV := encSeq + positions
	work := func(start, end int) {
		scores := make([]float32, totalKV)
		for task := start; task < end; task++ {
			pos := task / heads
			h := task - pos*heads
			kvh := h / group
			q := qAll[pos*qRows+h*headDim : pos*qRows+(h+1)*headDim]
			for j := 0; j < totalKV; j++ {
				if j < encSeq {
					if !promptAllowedForSlidingDecode(j, encSeq, slidingWindow) {
						scores[j] = float32(math.Inf(-1))
						continue
					}
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
			dst := attnAll[pos*qRows+h*headDim : pos*qRows+(h+1)*headDim]
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
	parallelizeAttentionTasks(positions*heads, work)
}

func promptAllowedForSlidingDecode(promptIndex, promptSeq, slidingWindow int) bool {
	if slidingWindow <= 0 {
		return true
	}
	lo := promptSeq - slidingWindow + 1
	if lo < 0 {
		lo = 0
	}
	return promptIndex >= lo
}

func parallelizeAttentionTasks(tasks int, work func(start, end int)) {
	nw := 1
	if k3Enabled() && tasks >= 32 {
		nw = k3Threads()
		if nw > tasks {
			nw = tasks
		}
	}
	if nw <= 1 {
		work(0, tasks)
		return
	}
	var wg sync.WaitGroup
	wg.Add(nw)
	for wid := 0; wid < nw; wid++ {
		start := wid * tasks / nw
		end := (wid + 1) * tasks / nw
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

func k3ScaleV(scale float32, x []float32) {
	for i := range x {
		x[i] *= scale
	}
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
	if lb.MLPGateProj == nil || lb.MLPUpProj == nil || lb.MLPDownProj == nil || len(lb.MLPGateProj.Shape) != 2 || len(lb.MLPUpProj.Shape) != 2 || len(lb.MLPDownProj.Shape) != 2 {
		return fmt.Errorf("DiffusionGemma dense MLP missing projection bindings layer %d", op.Layer)
	}
	intermediate, hiddenSize := lb.MLPGateProj.Shape[0], lb.MLPGateProj.Shape[1]
	if lb.MLPUpProj.Shape[0] != intermediate || lb.MLPUpProj.Shape[1] != hiddenSize || lb.MLPDownProj.Shape[0] != hiddenSize || lb.MLPDownProj.Shape[1] != intermediate {
		return fmt.Errorf("DiffusionGemma dense MLP shape mismatch gate=%v up=%v down=%v", lb.MLPGateProj.Shape, lb.MLPUpProj.Shape, lb.MLPDownProj.Shape)
	}
	var gateM, upM, downM *mixedMatrix
	if hiddenSize <= 0 || intermediate <= 0 || len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("DiffusionGemma dense MLP hidden len=%d hidden_size=%d intermediate=%d", len(scratch.Hidden), hiddenSize, intermediate)
	}
	positions := len(scratch.Hidden) / hiddenSize
	gateAll := make([]float32, positions*intermediate)
	upAll := make([]float32, positions*intermediate)
	if done, err := k3Gemm2RowsQ80(gateAll, upAll, scratch.Hidden, positions, weights, lb.MLPGateProj, lb.MLPUpProj); err != nil {
		return err
	} else if !done {
		var err error
		gateM, err = loadMixedMatrix(weights, lb.MLPGateProj)
		if err != nil {
			return err
		}
		upM, err = loadMixedMatrix(weights, lb.MLPUpProj)
		if err != nil {
			return err
		}
		if !gateM.gemmRows(gateAll, scratch.Hidden, positions) {
			return fmt.Errorf("DiffusionGemma dense MLP gate GEMM rejected layer %d", op.Layer)
		}
		if !upM.gemmRows(upAll, scratch.Hidden, positions) {
			return fmt.Errorf("DiffusionGemma dense MLP up GEMM rejected layer %d", op.Layer)
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
		var err error
		downM, err = loadMixedMatrix(weights, lb.MLPDownProj)
		if err != nil {
			return err
		}
		if !downM.gemmRows(scratch.Hidden, actAll, positions) {
			return fmt.Errorf("DiffusionGemma dense MLP down GEMM rejected layer %d", op.Layer)
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
		topK := scratch.TopKExperts
		if topK > 0 && len(scratch.TopKIDs) >= (pos+1)*topK && len(scratch.TopKVals) >= (pos+1)*topK {
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
	topK := scratch.TopKExperts
	if topK <= 0 || len(scratch.TopKIDs) < positions*topK || len(scratch.TopKVals) < positions*topK {
		return fmt.Errorf("DiffusionGemma expert top-k scratch invalid positions=%d top_k=%d ids=%d vals=%d", positions, topK, len(scratch.TopKIDs), len(scratch.TopKVals))
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
		if !simd.GemmRows(scoredAll, routerInput, projW, positions, numExperts, hiddenSize) {
			return fmt.Errorf("DiffusionGemma router GEMM rejected")
		}
	}
	topK := scratch.TopKExperts
	if topK <= 0 || len(scratch.TopKIDs) < positions*topK || len(scratch.TopKVals) < positions*topK {
		return fmt.Errorf("DiffusionGemma router top-k scratch invalid positions=%d top_k=%d ids=%d vals=%d", positions, topK, len(scratch.TopKIDs), len(scratch.TopKVals))
	}
	for pos := 0; pos < positions; pos++ {
		scored := scoredAll[pos*numExperts : (pos+1)*numExperts]
		k3SoftmaxInPlace(scored)
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
	positions := len(scratch.Residual) / hiddenSize
	for i := range scratch.MoeOut {
		scratch.MoeOut[i] = 0
	}
	topK := scratch.TopKExperts
	if topK <= 0 || len(scratch.TopKIDs) < positions*topK || len(scratch.TopKVals) < positions*topK {
		return fmt.Errorf("DiffusionGemma expert top-k scratch invalid positions=%d top_k=%d ids=%d vals=%d", positions, topK, len(scratch.TopKIDs), len(scratch.TopKVals))
	}

	expertsDone, err := k3RunPerExpertA100(weights, lb, layout, scratch, preNorm2, hiddenSize, positions, topK)
	if err != nil {
		return err
	}
	if !expertsDone {
		if err := runBatchedExpertFallback(weights, lb, layout, scratch, preNorm2, hiddenSize, positions, topK); err != nil {
			return err
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

func reportExpertOccupancy(label string, assignments map[int][]expertAssignment, positions, topK int) {
	if !diffusionGemmaTimingEnabled() || len(assignments) == 0 {
		return
	}
	assignmentCount := 0
	maxBatch := 0
	hist := map[int]int{}
	for _, rows := range assignments {
		batch := len(rows)
		assignmentCount += batch
		if batch > maxBatch {
			maxBatch = batch
		}
		hist[batch]++
	}
	keys := make([]int, 0, len(hist))
	for k := range hist {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	var b strings.Builder
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%d:%d", k, hist[k])
	}
	avg := float64(assignmentCount) / float64(len(assignments))
	fmt.Fprintf(os.Stderr, "timing diffusiongemma experts_occupancy path=%s positions=%d top_k=%d unique=%d assignments=%d avg_batch=%.2f max_batch=%d batch_hist=%s\n", label, positions, topK, len(assignments), assignmentCount, avg, maxBatch, b.String())
}

func runBatchedExpertFallback(weights *TextWeights, lb TextLayerBindings, layout expertWeightLayout, scratch ForwardScratch, preNorm2 []float32, hiddenSize, positions, topK int) error {
	normed := make([]float32, positions*hiddenSize)
	for pos := 0; pos < positions; pos++ {
		row := normed[pos*hiddenSize : (pos+1)*hiddenSize]
		copy(row, scratch.Residual[pos*hiddenSize:(pos+1)*hiddenSize])
		if !simd.RMSNormTo(row, preNorm2, 1e-6) {
			return fmt.Errorf("DiffusionGemma expert pre_norm_2 rejected")
		}
	}

	assignments := map[int][]expertAssignment{}
	for pos := 0; pos < positions; pos++ {
		for k := 0; k < topK; k++ {
			expertID := scratch.TopKIDs[pos*topK+k]
			if expertID < 0 || expertID >= layout.nExperts {
				continue
			}
			assignments[expertID] = append(assignments[expertID], expertAssignment{pos: pos, weight: scratch.TopKVals[pos*topK+k]})
		}
	}
	expertIDs := make([]int, 0, len(assignments))
	for expertID := range assignments {
		expertIDs = append(expertIDs, expertID)
	}
	sort.Ints(expertIDs)
	reportExpertOccupancy("fallback_f32", assignments, positions, topK)

	ensure := func(buf []float32, n int) []float32 {
		if cap(buf) < n {
			return make([]float32, n)
		}
		return buf[:n]
	}
	var xBuf, gateBuf, upBuf, actBuf, downBuf []float32
	for _, expertID := range expertIDs {
		rows := assignments[expertID]
		batch := len(rows)
		if batch == 0 {
			continue
		}
		ew, err := loadLayerExpertWeights(weights, lb, layout, expertID, hiddenSize)
		if err != nil {
			return err
		}
		x := ensure(xBuf, batch*hiddenSize)
		xBuf = x
		for i, a := range rows {
			copy(x[i*hiddenSize:(i+1)*hiddenSize], normed[a.pos*hiddenSize:(a.pos+1)*hiddenSize])
		}
		gate := ensure(gateBuf, batch*layout.intermediate)
		gateBuf = gate
		up := ensure(upBuf, batch*layout.intermediate)
		upBuf = up
		act := ensure(actBuf, batch*layout.intermediate)
		actBuf = act
		down := ensure(downBuf, batch*hiddenSize)
		downBuf = down
		if err := runDecodedExpertBatch(down, gate, up, act, x, ew, batch, hiddenSize, layout.intermediate); err != nil {
			return err
		}
		for i, a := range rows {
			dst := scratch.MoeOut[a.pos*hiddenSize : (a.pos+1)*hiddenSize]
			src := down[i*hiddenSize : (i+1)*hiddenSize]
			k3SaxpyV(a.weight, src, dst)
		}
	}
	return nil
}

func runDecodedExpertBatch(out, gate, up, act, x []float32, ew decodedExpertWeights, batch, hiddenSize, intermediate int) error {
	if !simd.GemmRows(gate, x, ew.gateW, batch, intermediate, hiddenSize) || !simd.GemmRows(up, x, ew.upW, batch, intermediate, hiddenSize) {
		return fmt.Errorf("DiffusionGemma expert batched GEMM rejected")
	}
	for i := 0; i < batch; i++ {
		if !simd.GELUTanhMulTo(act[i*intermediate:(i+1)*intermediate], gate[i*intermediate:(i+1)*intermediate], up[i*intermediate:(i+1)*intermediate]) {
			return fmt.Errorf("DiffusionGemma expert activation rejected")
		}
	}
	if !simd.GemmRows(out, act, ew.downW, batch, hiddenSize, intermediate) {
		return fmt.Errorf("DiffusionGemma expert batched down GEMM rejected")
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

func (m *mixedMatrix) gemmRows(out, x []float32, batch int) bool {
	if batch <= 0 {
		return false
	}
	if m.bf16 == nil {
		return simd.GemmRows(out, x, m.f32, batch, m.rows, m.cols)
	}
	for b := 0; b < batch; b++ {
		if !simd.GemvRowsBF16(out[b*m.rows:(b+1)*m.rows], x[b*m.cols:(b+1)*m.cols], m.bf16, m.rows, m.cols) {
			return false
		}
	}
	return true
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

func buildSelfConditioningFromLogitRows(weights *TextWeights, logits [][]float32, tempInv float32) ([]float32, error) {
	if weights == nil || len(logits) == 0 {
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
	positions := len(logits)
	out := make([]float32, positions*hiddenSize)
	if done, err := k3SelfConditioningSoftEmbeddingQ80(out, logits, weights, fp.Globals.EmbedTokens, positions, vocab, hiddenSize, tempInv); err != nil {
		return nil, err
	} else if done {
		embedScale := float32(math.Sqrt(float64(hiddenSize)))
		for i := range out {
			out[i] *= embedScale
		}
		return out, nil
	}
	if raw, dtype, shape, err := weights.RawTensor(fp.Globals.EmbedTokens.Name); err == nil && len(shape) == 2 && shape[0] == vocab && shape[1] == hiddenSize {
		scales, err := loadOptionalFP8RowScales(weights, fp.Globals.EmbedTokens.Name, dtype, vocab)
		if err != nil {
			return nil, err
		}
		if err := buildSelfConditioningSoftEmbeddingRowsRaw(out, logits, raw, dtype, scales, positions, vocab, hiddenSize, tempInv); err != nil {
			return nil, err
		}
		embedScale := float32(math.Sqrt(float64(hiddenSize)))
		for i := range out {
			out[i] *= embedScale
		}
		return out, nil
	}
	row := make([]float32, hiddenSize)
	for pos := 0; pos < positions; pos++ {
		dst := out[pos*hiddenSize : (pos+1)*hiddenSize]
		if err := buildSelfConditioningSoftEmbeddingRow(dst, logits[pos], vocab, hiddenSize, tempInv, row, func(vocabID int, dst []float32) error {
			raw, dtype, shape, err := weights.RawTensorRow(fp.Globals.EmbedTokens.Name, vocabID)
			if err != nil {
				return err
			}
			if len(shape) != 1 || shape[0] != hiddenSize {
				return fmt.Errorf("DiffusionGemma self-conditioning row shape %v want [%d]", shape, hiddenSize)
			}
			return decodeFloatRowTo(dst, raw, dtype)
		}); err != nil {
			return nil, err
		}
	}
	embedScale := float32(math.Sqrt(float64(hiddenSize)))
	for i := range out {
		out[i] *= embedScale
	}
	return out, nil
}

func loadOptionalFP8RowScales(weights *TextWeights, name, dtype string, rows int) ([]float32, error) {
	if dtype != "F8_E4M3" && dtype != "F8_E4M3FN" {
		return nil, nil
	}
	scaleName := diffusionGemmaWeightScaleName(name)
	raw, scaleDType, shape, err := weights.RawTensor(scaleName)
	if err != nil {
		return nil, fmt.Errorf("DiffusionGemma FP8 tensor %q missing scale %q: %w", name, scaleName, err)
	}
	n := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("DiffusionGemma FP8 scale %q invalid shape %v", scaleName, shape)
		}
		n *= dim
	}
	if n != 1 && n != rows {
		return nil, fmt.Errorf("DiffusionGemma FP8 scale %q shape %v gives %d values, want 1 or %d", scaleName, shape, n, rows)
	}
	scales := make([]float32, n)
	if err := decodeFloatRowTo(scales, raw, scaleDType); err != nil {
		return nil, err
	}
	return scales, nil
}

func buildSelfConditioningSoftEmbeddingRowsRaw(out []float32, logits [][]float32, raw []byte, dtype string, scales []float32, positions, vocab, hiddenSize int, tempInv float32) error {
	if positions < 0 || vocab <= 0 || hiddenSize <= 0 || len(out) < positions*hiddenSize || len(logits) < positions {
		return fmt.Errorf("DiffusionGemma self-conditioning raw soft embedding shape mismatch out=%d logits=%d positions=%d vocab=%d hidden=%d", len(out), len(logits), positions, vocab, hiddenSize)
	}
	elemSize, ok := diffusionGemmaDTypeSize(dtype)
	if !ok {
		return fmt.Errorf("DiffusionGemma self-conditioning unsupported embedding dtype %s", dtype)
	}
	rowBytes := hiddenSize * elemSize
	if len(raw) < vocab*rowBytes {
		return fmt.Errorf("DiffusionGemma self-conditioning raw embedding bytes=%d want %d", len(raw), vocab*rowBytes)
	}
	if tempInv == 0 {
		tempInv = 1
	}
	maxLogits := make([]float32, positions)
	sums := make([]float64, positions)
	for pos := 0; pos < positions; pos++ {
		row := logits[pos]
		if len(row) < vocab {
			return fmt.Errorf("DiffusionGemma self-conditioning logits row=%d len=%d want %d", pos, len(row), vocab)
		}
		maxLogits[pos] = float32(math.Inf(-1))
		for vocabID := 0; vocabID < vocab; vocabID++ {
			v := row[vocabID]
			if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
				continue
			}
			v *= tempInv
			if v > maxLogits[pos] {
				maxLogits[pos] = v
			}
		}
		if math.IsInf(float64(maxLogits[pos]), -1) {
			continue
		}
		for vocabID := 0; vocabID < vocab; vocabID++ {
			v := row[vocabID]
			if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
				continue
			}
			sums[pos] += math.Exp(float64(v*tempInv - maxLogits[pos]))
		}
	}
	out = out[:positions*hiddenSize]
	for i := range out {
		out[i] = 0
	}
	workers := 1
	if k3Enabled() && vocab >= 8192 && positions > 0 {
		workers = k3Threads()
		if workers > vocab/4096 {
			workers = vocab / 4096
		}
		if workers < 1 {
			workers = 1
		}
	}
	runChunk := func(dst []float32, startVocab, endVocab int) error {
		embedRow := make([]float32, hiddenSize)
		for vocabID := startVocab; vocabID < endVocab; vocabID++ {
			start := vocabID * rowBytes
			if err := decodeFloatRowTo(embedRow, raw[start:start+rowBytes], dtype); err != nil {
				return err
			}
			if len(scales) == 1 {
				k3ScaleV(scales[0], embedRow)
			} else if len(scales) == vocab {
				k3ScaleV(scales[vocabID], embedRow)
			}
			for pos := 0; pos < positions; pos++ {
				if sums[pos] <= 0 || math.IsNaN(sums[pos]) || math.IsInf(float64(maxLogits[pos]), -1) {
					continue
				}
				v := logits[pos][vocabID]
				if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
					continue
				}
				prob := float32(math.Exp(float64(v*tempInv-maxLogits[pos])) / sums[pos])
				if prob == 0 {
					continue
				}
				k3SaxpyV(prob, embedRow, dst[pos*hiddenSize:(pos+1)*hiddenSize])
			}
		}
		return nil
	}
	if workers <= 1 {
		return runChunk(out, 0, vocab)
	}
	partials := make([][]float32, workers)
	for i := range partials {
		partials[i] = make([]float32, len(out))
	}
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	wg.Add(workers)
	for wid := 0; wid < workers; wid++ {
		startVocab := wid * vocab / workers
		endVocab := (wid + 1) * vocab / workers
		go func(wid, startVocab, endVocab int) {
			defer wg.Done()
			if err := runChunk(partials[wid], startVocab, endVocab); err != nil {
				errCh <- err
			}
		}(wid, startVocab, endVocab)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return err
		}
	}
	for _, part := range partials {
		for i, v := range part {
			out[i] += v
		}
	}
	return nil
}

func buildSelfConditioningSoftEmbeddingRowsF32(out []float32, logits [][]float32, embed []float32, positions, vocab, hiddenSize int, tempInv float32) error {
	if positions < 0 || vocab <= 0 || hiddenSize <= 0 || len(out) < positions*hiddenSize || len(logits) < positions || len(embed) < vocab*hiddenSize {
		return fmt.Errorf("DiffusionGemma self-conditioning soft embedding shape mismatch out=%d logits=%d embed=%d positions=%d vocab=%d hidden=%d", len(out), len(logits), len(embed), positions, vocab, hiddenSize)
	}
	if tempInv == 0 {
		tempInv = 1
	}
	for pos := 0; pos < positions; pos++ {
		rowLogits := logits[pos]
		if len(rowLogits) < vocab {
			return fmt.Errorf("DiffusionGemma self-conditioning logits row=%d len=%d want %d", pos, len(rowLogits), vocab)
		}
		dst := out[pos*hiddenSize : (pos+1)*hiddenSize]
		for i := range dst {
			dst[i] = 0
		}
		maxLogit := float32(math.Inf(-1))
		for vocabID := 0; vocabID < vocab; vocabID++ {
			v := rowLogits[vocabID]
			if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
				continue
			}
			v *= tempInv
			if v > maxLogit {
				maxLogit = v
			}
		}
		if math.IsInf(float64(maxLogit), -1) {
			continue
		}
		var sum float64
		for vocabID := 0; vocabID < vocab; vocabID++ {
			v := rowLogits[vocabID]
			if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
				continue
			}
			sum += math.Exp(float64(v*tempInv - maxLogit))
		}
		if sum <= 0 || math.IsNaN(sum) {
			continue
		}
		inv := 1.0 / sum
		for vocabID := 0; vocabID < vocab; vocabID++ {
			v := rowLogits[vocabID]
			if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
				continue
			}
			p := float32(math.Exp(float64(v*tempInv-maxLogit)) * inv)
			if p == 0 {
				continue
			}
			emb := embed[vocabID*hiddenSize : (vocabID+1)*hiddenSize]
			k3SaxpyV(p, emb, dst)
		}
	}
	return nil
}

func buildSelfConditioningSoftEmbeddingRow(dst []float32, logits []float32, vocab, hiddenSize int, tempInv float32, scratch []float32, loadEmbeddingRow func(vocabID int, dst []float32) error) error {
	if len(dst) != hiddenSize {
		return fmt.Errorf("DiffusionGemma self-conditioning dst len=%d want %d", len(dst), hiddenSize)
	}
	if len(scratch) != hiddenSize {
		return fmt.Errorf("DiffusionGemma self-conditioning scratch len=%d want %d", len(scratch), hiddenSize)
	}
	for i := range dst {
		dst[i] = 0
	}
	if len(logits) < vocab {
		return fmt.Errorf("DiffusionGemma self-conditioning logits len=%d want %d", len(logits), vocab)
	}
	logits = logits[:vocab]
	if tempInv == 0 {
		tempInv = 1
	}
	const sparseLimit = 4096
	finiteIDs := make([]int, 0, 16)
	finiteVals := make([]float32, 0, 16)
	dense := false
	for vocabID, v := range logits {
		if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
			continue
		}
		if len(finiteIDs) >= sparseLimit {
			dense = true
			break
		}
		finiteIDs = append(finiteIDs, vocabID)
		finiteVals = append(finiteVals, v*tempInv)
	}
	if !dense && len(finiteIDs) > 0 {
		maxLogit := finiteVals[0]
		for _, v := range finiteVals[1:] {
			if v > maxLogit {
				maxLogit = v
			}
		}
		var sum float64
		probs := make([]float32, len(finiteVals))
		for i, v := range finiteVals {
			e := math.Exp(float64(v - maxLogit))
			probs[i] = float32(e)
			sum += e
		}
		if sum == 0 {
			return nil
		}
		inv := float32(1 / sum)
		for i, vocabID := range finiteIDs {
			prob := probs[i] * inv
			if prob == 0 {
				continue
			}
			if err := loadEmbeddingRow(vocabID, scratch); err != nil {
				return err
			}
			k3SaxpyV(prob, scratch, dst)
		}
		return nil
	}
	probs := append([]float32(nil), logits...)
	for i := range probs {
		probs[i] *= tempInv
	}
	k3SoftmaxInPlace(probs)
	for vocabID, prob := range probs {
		if prob == 0 {
			continue
		}
		if err := loadEmbeddingRow(vocabID, scratch); err != nil {
			return err
		}
		k3SaxpyV(prob, scratch, dst)
	}
	return nil
}

func exactEmbeddingDotF32(raw []byte, dtype string, x []float32, rowScratch []float32) (float32, error) {
	if dtype == "BF16" {
		if len(raw) < len(x)*2 {
			return 0, fmt.Errorf("DiffusionGemma BF16 dot row bytes=%d want %d", len(raw), len(x)*2)
		}
		if len(x) == 0 {
			return 0, nil
		}
		bf16 := unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), len(x))
		if v, ok := simd.BF16DotF32To(bf16, x); ok {
			return v, nil
		}
		return 0, fmt.Errorf("DiffusionGemma BF16 dot rejected row len=%d", len(x))
	}
	if len(rowScratch) < len(x) {
		rowScratch = make([]float32, len(x))
	}
	if err := decodeFloatRowTo(rowScratch[:len(x)], raw, dtype); err != nil {
		return 0, err
	}
	return k3Dot(x, rowScratch[:len(x)]), nil
}

func runLMHead(weights *TextWeights, scratch ForwardScratch) error {
	started := time.Now()
	mode := "full_raw_rows"
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
	var lmQ80Elapsed, lmScanElapsed, lmRerankElapsed time.Duration
	if topK > vocab {
		topK = vocab
	}
	for pos := 0; pos < positions; pos++ {
		if len(scratch.Logits[pos]) < vocab {
			return fmt.Errorf("DiffusionGemma LM head logits row=%d len=%d want %d", pos, len(scratch.Logits[pos]), vocab)
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
		mode = "topk_f32"
		lmHeadDone := false
		if k3A100LMHeadEnabled() {
			scoresAll := scratch.LogitFlat
			if len(scoresAll) < positions*vocab {
				scoresAll = make([]float32, positions*vocab)
			}
			phaseStart := time.Now()
			if done, err := k3GemmRowsQ80(scoresAll[:positions*vocab], scratch.Hidden, positions, weights, fp.Globals.EmbedTokens); err != nil {
				return err
			} else if done {
				lmQ80Elapsed = time.Since(phaseStart)
				mode = "topk_k3_a100_rerank_full_logits"
				lmHeadDone = true
				candidateK := k3A100LMHeadCandidates(topK, vocab)
				row := make([]float32, hiddenSize)
				for pos := 0; pos < positions; pos++ {
					candIDs := make([]int, candidateK)
					candVals := make([]float32, candidateK)
					for i := range candIDs {
						candIDs[i] = -1
						candVals[i] = float32(math.Inf(-1))
					}
					scores := scoresAll[pos*vocab : (pos+1)*vocab]
					copy(scratch.Logits[pos][:vocab], scores)
					phaseStart = time.Now()
					selectTopKMin(candIDs, candVals, scores)
					lmScanElapsed += time.Since(phaseStart)
					hidden := scratch.Hidden[pos*hiddenSize : (pos+1)*hiddenSize]
					phaseStart = time.Now()
					rerankState := newTopKMinState(topIDs[pos], topVals[pos])
					for _, vocabID := range candIDs {
						if vocabID < 0 {
							continue
						}
						raw, dtype, shape, err := weights.RawTensorRow(fp.Globals.EmbedTokens.Name, vocabID)
						if err != nil {
							return err
						}
						if len(shape) != 1 || shape[0] != hiddenSize {
							return fmt.Errorf("DiffusionGemma LM head rerank row shape %v want [%d]", shape, hiddenSize)
						}
						score, err := exactEmbeddingDotF32(raw, dtype, hidden, row)
						if err != nil {
							return err
						}
						rerankState.Insert(vocabID, score)
					}
					lmRerankElapsed += time.Since(phaseStart)
				}
			}
		}
		if !lmHeadDone {
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
				selectTopKMin(topIDs[pos], topVals[pos], scores)
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
	if diffusionGemmaTimingEnabled() {
		fmt.Fprintf(os.Stderr, "timing diffusiongemma lm_head mode=%s positions=%d vocab=%d hidden=%d top_k=%d elapsed=%s q80=%s scan=%s rerank=%s\n", mode, positions, vocab, hiddenSize, topK, time.Since(started).Round(time.Millisecond), lmQ80Elapsed.Round(time.Millisecond), lmScanElapsed.Round(time.Millisecond), lmRerankElapsed.Round(time.Millisecond))
	}
	return nil
}

type topKMinState struct {
	ids    []int
	vals   []float32
	filled int
	minIdx int
	minVal float32
}

func newTopKMinState(ids []int, vals []float32) *topKMinState {
	for i := range ids {
		ids[i] = -1
	}
	for i := range vals {
		vals[i] = float32(math.Inf(-1))
	}
	return &topKMinState{ids: ids, vals: vals, minIdx: 0, minVal: float32(math.Inf(-1))}
}

func (s *topKMinState) recomputeMin() {
	if len(s.vals) == 0 {
		s.minIdx = 0
		s.minVal = float32(math.Inf(-1))
		return
	}
	idx := 0
	val := s.vals[0]
	for i, v := range s.vals[1:] {
		if v < val {
			idx = i + 1
			val = v
		}
	}
	s.minIdx = idx
	s.minVal = val
}

func (s *topKMinState) Insert(id int, val float32) {
	if len(s.ids) == 0 || math.IsNaN(float64(val)) {
		return
	}
	if s.filled < len(s.ids) {
		s.ids[s.filled] = id
		s.vals[s.filled] = val
		s.filled++
		if s.filled == len(s.ids) {
			s.recomputeMin()
		}
		return
	}
	if val <= s.minVal {
		return
	}
	s.ids[s.minIdx] = id
	s.vals[s.minIdx] = val
	s.recomputeMin()
}

func selectTopKMin(ids []int, vals []float32, scores []float32) {
	state := newTopKMinState(ids, vals)
	for id, score := range scores {
		state.Insert(id, score)
	}
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
