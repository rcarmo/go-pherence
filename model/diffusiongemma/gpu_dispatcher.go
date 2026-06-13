package diffusiongemma

import (
	"fmt"
	"os"
	"time"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"math"
	"runtime"
	"sync"
	"unsafe"
)

// GPUDispatcher offloads GEMV projections to GPU via DevBuf/DevGemv,
// keeping attention math, norms, and sampling on CPU.
type GPUDispatcher struct {
	ResidentLayerPrefix int
	MaxLayers           int
	TailAfterMaxLayers  bool
	LMHeadTopK          int
	Progress            bool
	SkipEviction        bool
	FP8Model            *GPUFP8Model
	FP8Weights          *FP8TextWeights
	ExpertPool          *GPUFP8ExpertPool
	ExpertCache         *ExpertLRUCache
}

func (d GPUDispatcher) cpuFallback() CPUDispatcher {
	return CPUDispatcher{
		ResidentLayerPrefix: d.ResidentLayerPrefix,
		MaxLayers:           d.MaxLayers,
		TailAfterMaxLayers:  d.TailAfterMaxLayers,
		LMHeadTopK:          d.LMHeadTopK,
		Progress:            d.Progress,
		SkipEviction:        d.SkipEviction,
	}
}

func (d GPUDispatcher) RunTextForward(ctx ForwardContext, weights *TextWeights, ops ForwardOpPlan, buffers ForwardBufferPlan) (ForwardOutput, error) {
	if !gpu.SgemmReady() {
		if d.Progress {
			fmt.Fprintf(os.Stderr, "DiffusionGemma GPU: SGEMM not ready, CPU fallback\n")
		}
		return d.cpuFallback().RunTextForward(ctx, weights, ops, buffers)
	}
	if weights == nil || !ops.Ready || len(ctx.Canvas) == 0 {
		return d.cpuFallback().RunTextForward(ctx, weights, ops, buffers)
	}

	if d.Progress {
		fmt.Fprintf(os.Stderr, "DiffusionGemma GPU: using %s\n", gpu.DeviceName())
	}

	fp := weights.ForwardPlan()
	hiddenSize := buffers.HiddenSize
	positions := len(ctx.Canvas)

	scratch := NewForwardScratch(buffers)
	scratch.LMHeadTopK = d.LMHeadTopK
	actualHidden := positions * hiddenSize
	if actualHidden > 0 && actualHidden < len(scratch.Hidden) {
		scratch.Hidden = scratch.Hidden[:actualHidden]
		scratch.Residual = scratch.Residual[:actualHidden]
		scratch.MlpOut = scratch.MlpOut[:actualHidden]
		scratch.MoeOut = scratch.MoeOut[:actualHidden]
		scratch.Logits = scratch.Logits[:positions]
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
				fmt.Fprintf(os.Stderr, "DiffusionGemma GPU: completed layer=%d elapsed=%s\n", currentLayer, time.Since(layerStarted).Round(time.Millisecond))
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
		}

		switch op.Kind {
		case OpInputNorm:
			copy(scratch.Residual, scratch.Hidden)
			if err := runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.InputLayerNorm }); err != nil {
				return ForwardOutput{}, err
			}
		case OpSelfAttention:
			if err := d.gpuAttention(op, ctx, weights, scratch, fp, hiddenSize, positions); err != nil {
				return ForwardOutput{}, err
			}
		case OpPostAttention:
			if err := runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.PostAttentionLayerNorm }); err != nil {
				return ForwardOutput{}, err
			}
			for i := range scratch.Hidden {
				scratch.Hidden[i] += scratch.Residual[i]
			}
		case OpPreMoE:
			copy(scratch.Residual, scratch.Hidden)
			if err := runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.PreFFNLayerNorm }); err != nil {
				return ForwardOutput{}, err
			}
		case OpDenseMLP:
			if err := d.gpuDenseMLP(op, weights, scratch, fp, hiddenSize); err != nil {
				return ForwardOutput{}, err
			}
		case OpRouter:
			if err := runRouterFromResidual(op, weights, scratch); err != nil {
				return ForwardOutput{}, err
			}
		case OpExperts:
			if d.ExpertPool != nil {
				if _, ok := d.ExpertPool.Layers[op.Layer]; ok {
					if err := runGPUResidentExperts(op, weights, scratch, d.ExpertPool); err != nil {
						return ForwardOutput{}, err
					}
					break
				}
			}
			if d.ExpertCache != nil && d.FP8Weights != nil {
				if err := runLRUCachedExperts(op, weights, scratch, d.FP8Weights, d.ExpertCache); err != nil {
					return ForwardOutput{}, err
				}
			} else {
				if err := runExpertsFromResidual(op, weights, scratch); err != nil {
					return ForwardOutput{}, err
				}
			}
		case OpPostMoE:
			if err := runCombineMlpMoe(op, weights, scratch); err != nil {
				return ForwardOutput{}, err
			}
			for i := range scratch.Hidden {
				scratch.Hidden[i] += scratch.Residual[i]
			}
		case OpLayerScalar:
			if err := runLayerScalar(op, weights, scratch); err != nil {
				return ForwardOutput{}, err
			}
		default:
			return ForwardOutput{}, fmt.Errorf("DiffusionGemma GPU unknown op %q", op.Kind)
		}
	}
	if currentLayer >= 0 && !d.SkipEviction && currentLayer >= d.ResidentLayerPrefix {
		weights.EvictLayer(currentLayer)
	}
	if d.MaxLayers > 0 && !d.TailAfterMaxLayers {
		return ForwardOutput{Logits: scratch.Logits, SelfConditioning: ctx.SelfConditioning}, nil
	}

	for _, op := range ops.Tail {
		if op == OpLMHead && d.FP8Weights != nil {
			if err := runDenseGPULMHead(d.FP8Weights, scratch, buffers.HiddenSize); err != nil {
				// Fall back to sparse BF16 scan
				if err2 := runLMHeadFromShards(d.FP8Weights.shards, scratch, "model.decoder.embed_tokens.weight"); err2 != nil {
					return ForwardOutput{}, err2
				}
			}
		} else {
			if err := dispatchTailOp(op, weights, scratch); err != nil {
				return ForwardOutput{}, err
			}
		}
	}
	selfConditioning, err := buildSelfConditioningFromLogits(weights, scratch)
	if err != nil {
		return ForwardOutput{}, err
	}
	return ForwardOutput{Logits: scratch.Logits, SelfConditioning: selfConditioning}, nil
}

// devGemv wraps DevGemv: uploads weight + input, computes on GPU, reads back.
func devGemv(out, x, w []float32, M, K int) {
	outBuf := gpu.NewDevBuf(M)
	xBuf := gpu.NewDevBufFrom(x[:K])
	wBuf := gpu.NewDevBufFrom(w[:M*K])
	gpu.DevGemv(outBuf, xBuf, wBuf, M, K)
	outBuf.ToCPU()
	copy(out[:M], outBuf.Data()[:M])
}

func (d GPUDispatcher) gpuAttention(op LayerOp, ctx ForwardContext, weights *TextWeights, scratch ForwardScratch, fp TextForwardPlan, hiddenSize, positions int) error {
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("GPU attention layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]

	// Load attention projection weights — skip BF16 decode when FP8 available
	var qRows, kRows, vRows int
	var oW, qW, kW, vW []float32
	if d.FP8Model != nil && op.Layer < len(d.FP8Model.Layers) {
		fl := &d.FP8Model.Layers[op.Layer]
		if fl.Q != nil {
			qRows = fl.Q.OutDim
		}
		if fl.K != nil {
			kRows = fl.K.OutDim
		}
		vRows = kRows
		if fl.V != nil {
			vRows = fl.V.OutDim
		}
		// O projection handled by batched FP8 GEMM below
	} else {
		var err error
		qW, qRows, _, err = loadFloatMatrix(weights, lb.QProj)
		if err != nil {
			return err
		}
		kW, kRows, _, err = loadFloatMatrix(weights, lb.KProj)
		if err != nil {
			return err
		}
		vRows = kRows
		if lb.VProj != nil {
			vW, vRows, _, err = loadFloatMatrix(weights, lb.VProj)
			if err != nil {
				return err
			}
		}
		oW, _, _, err = loadFloatMatrix(weights, lb.OProj)
		if err != nil {
			return err
		}
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
	heads := qRows / headDim
	kvHeads := kRows / headDim

	// Upload weight matrices to GPU once per layer

	qAll := make([]float32, positions*qRows)
	kAll := make([]float32, positions*kRows)
	vAll := make([]float32, positions*vRows)

	ropeHalf := headDim / 2
	ropeTheta := 10000.0
	if op.Type == "full_attention" {
		ropeHalf = headDim / 8
		ropeTheta = 1000000.0
	}
	ropeFreqs := simd.BuildRoPEFreqs(ctx.EncoderSeqLen+positions, ropeHalf, headDim, ropeTheta)

	// GPU projections: FP8 GEMV if available, else batched F32 SGEMM
	if d.FP8Model != nil && op.Layer < len(d.FP8Model.Layers) {
		fl := &d.FP8Model.Layers[op.Layer]
		// Batched FP8 GEMM: all positions in one GPU call per projection
		if err := gpu.GemmFP8E4M3(qAll, scratch.Hidden[:positions*hiddenSize], positions, fl.Q); err != nil {
			return fmt.Errorf("FP8 Q GEMM: %w", err)
		}
		if err := gpu.GemmFP8E4M3(kAll, scratch.Hidden[:positions*hiddenSize], positions, fl.K); err != nil {
			return fmt.Errorf("FP8 K GEMM: %w", err)
		}
		if fl.V != nil {
			if err := gpu.GemmFP8E4M3(vAll, scratch.Hidden[:positions*hiddenSize], positions, fl.V); err != nil {
				return fmt.Errorf("FP8 V GEMM: %w", err)
			}
		} else {
			copy(vAll, kAll)
		}
	} else {
		qResult, err := batchedGPUGemm(qW, scratch.Hidden[:positions*hiddenSize], qRows, hiddenSize, positions)
		if err != nil {
			return fmt.Errorf("GPU Q GEMM: %w", err)
		}
		scatterGemmResult(qAll, qResult, qRows, positions)
		kResult, err := batchedGPUGemm(kW, scratch.Hidden[:positions*hiddenSize], kRows, hiddenSize, positions)
		if err != nil {
			return fmt.Errorf("GPU K GEMM: %w", err)
		}
		scatterGemmResult(kAll, kResult, kRows, positions)
		if lb.VProj != nil {
			vResult, err := batchedGPUGemm(vW, scratch.Hidden[:positions*hiddenSize], vRows, hiddenSize, positions)
			if err != nil {
				return fmt.Errorf("GPU V GEMM: %w", err)
			}
			scatterGemmResult(vAll, vResult, vRows, positions)
		} else {
			copy(vAll, kAll)
		}
	}

	for pos := 0; pos < positions; pos++ {
		q := qAll[pos*qRows : (pos+1)*qRows]
		k := kAll[pos*kRows : (pos+1)*kRows]
		v := vAll[pos*vRows : (pos+1)*vRows]

		// Norms + RoPE on CPU
		for h := 0; h < heads; h++ {
			simd.RMSNormTo(q[h*headDim:(h+1)*headDim], qNorm, 1e-6)
		}
		for h := 0; h < kvHeads; h++ {
			simd.RMSNormTo(k[h*headDim:(h+1)*headDim], kNorm, 1e-6)
			simd.RMSNormNoScaleTo(v[h*headDim:(h+1)*headDim], 1e-6)
		}
		if len(ropeFreqs) > 0 && ropeHalf > 0 {
			simd.ApplyRoPEPartial(q, ropeFreqs, pos+ctx.EncoderSeqLen, heads, headDim, ropeHalf)
			simd.ApplyRoPEPartial(k, ropeFreqs, pos+ctx.EncoderSeqLen, kvHeads, headDim, ropeHalf)
		}
	}

	// Attention: GPU GQA when available, CPU fallback
	enc := EncoderKVLayer{}
	if op.Layer >= 0 && op.Layer < len(ctx.EncoderKV) {
		enc = ctx.EncoderKV[op.Layer]
	}
	encSeq := 0
	if enc.SeqLen > 0 {
		encSeq = enc.SeqLen
	}
	totalKV := encSeq + positions

	// Build concatenated KV cache: [totalKV, kvHeads * headDim]
	kvDim := kvHeads * headDim
	kConcat := make([]float32, totalKV*kvDim)
	vConcat := make([]float32, totalKV*kvDim)
	for j := 0; j < encSeq; j++ {
		copy(kConcat[j*kvDim:(j+1)*kvDim], enc.Keys[j*kRows:j*kRows+kvDim])
		copy(vConcat[j*kvDim:(j+1)*kvDim], enc.Values[j*vRows:j*vRows+kvDim])
	}
	for j := 0; j < positions; j++ {
		copy(kConcat[(encSeq+j)*kvDim:(encSeq+j+1)*kvDim], kAll[j*kRows:j*kRows+kvDim])
		copy(vConcat[(encSeq+j)*kvDim:(encSeq+j+1)*kvDim], vAll[j*vRows:j*vRows+kvDim])
	}

	attnAll := make([]float32, positions*qRows)
	for pos := 0; pos < positions; pos++ {
		attnCtx := attnAll[pos*qRows : (pos+1)*qRows]
		qPos := qAll[pos*qRows : (pos+1)*qRows]
		// Try GPU GQA attention
		if err := gpu.F32GQAAttention(attnCtx, qPos, kConcat, vConcat, totalKV, heads, kvHeads, headDim, 1.0); err != nil {
			// CPU fallback
			group := heads / kvHeads
			for i := range attnCtx {
				attnCtx[i] = 0
			}
			scores := make([]float32, totalKV)
			for h := 0; h < heads; h++ {
				kvh := h / group
				q := qPos[h*headDim : (h+1)*headDim]
				for j := 0; j < totalKV; j++ {
					scores[j] = dot(q, kConcat[j*kvDim+kvh*headDim:j*kvDim+(kvh+1)*headDim])
				}
				softmaxInPlace(scores)
				dst := attnCtx[h*headDim : (h+1)*headDim]
				for j, score := range scores {
					vv := vConcat[j*kvDim+kvh*headDim : j*kvDim+(kvh+1)*headDim]
					for dd := range dst {
						dst[dd] += score * vv[dd]
					}
				}
			}
		}
	}
	// Batched O projection — FP8 GEMM if available, else per-position CPU
	if d.FP8Model != nil && op.Layer < len(d.FP8Model.Layers) && d.FP8Model.Layers[op.Layer].O != nil {
		// Collect all attention contexts into contiguous batch
		attnBatch := make([]float32, positions*qRows)
		for pos := 0; pos < positions; pos++ {
			copy(attnBatch[pos*qRows:(pos+1)*qRows], attnAll[pos*qRows:(pos+1)*qRows])
		}
		oOut := make([]float32, positions*hiddenSize)
		if err := gpu.GemmFP8E4M3(oOut, attnBatch, positions, d.FP8Model.Layers[op.Layer].O); err != nil {
			return fmt.Errorf("FP8 O GEMM: %w", err)
		}
		copy(scratch.Hidden[:positions*hiddenSize], oOut)
	} else {
		for pos := 0; pos < positions; pos++ {
			oOut := make([]float32, hiddenSize)
			attnCtx := attnAll[pos*qRows : (pos+1)*qRows]
			simd.GemvRows(oOut, attnCtx, oW, hiddenSize, qRows)
			copy(scratch.Hidden[pos*hiddenSize:(pos+1)*hiddenSize], oOut)
		}
	}
	return nil
}

func (d GPUDispatcher) gpuDenseMLP(op LayerOp, weights *TextWeights, scratch ForwardScratch, fp TextForwardPlan, hiddenSize int) error {
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("GPU MLP layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]

	var gateW, upW, downW []float32
	var intermediate, gateCols int
	if d.FP8Model != nil && op.Layer < len(d.FP8Model.Layers) {
		fl := &d.FP8Model.Layers[op.Layer]
		if fl.Gate != nil {
			intermediate = fl.Gate.OutDim
			gateCols = fl.Gate.InDim
		}
	} else {
		var err error
		gateW, intermediate, gateCols, err = loadFloatMatrix(weights, lb.MLPGateProj)
		if err != nil {
			return err
		}
		upW, _, _, err = loadFloatMatrix(weights, lb.MLPUpProj)
		if err != nil {
			return err
		}
		downW, _, _, err = loadFloatMatrix(weights, lb.MLPDownProj)
		if err != nil {
			return err
		}
	}

	positions := len(scratch.Hidden) / hiddenSize
	if d.FP8Model != nil && op.Layer < len(d.FP8Model.Layers) {
		fl := &d.FP8Model.Layers[op.Layer]
		// Batched FP8 GEMM: all positions in 3 GPU calls
		gateBatch := make([]float32, positions*intermediate)
		upBatch := make([]float32, positions*intermediate)
		if err := gpu.GemmFP8E4M3(gateBatch, scratch.Hidden[:positions*hiddenSize], positions, fl.Gate); err != nil {
			return fmt.Errorf("FP8 gate GEMM: %w", err)
		}
		if err := gpu.GemmFP8E4M3(upBatch, scratch.Hidden[:positions*hiddenSize], positions, fl.Up); err != nil {
			return fmt.Errorf("FP8 up GEMM: %w", err)
		}
		// Activation per position
		actBatch := make([]float32, positions*intermediate)
		for i := 0; i < positions; i++ {
			simd.GELUTanhMulTo(actBatch[i*intermediate:(i+1)*intermediate], gateBatch[i*intermediate:(i+1)*intermediate], upBatch[i*intermediate:(i+1)*intermediate])
		}
		// Batched down projection
		downBatch := make([]float32, positions*hiddenSize)
		if err := gpu.GemmFP8E4M3(downBatch, actBatch, positions, fl.Down); err != nil {
			return fmt.Errorf("FP8 down GEMM: %w", err)
		}
		copy(scratch.Hidden[:positions*hiddenSize], downBatch)
	} else {
		gateBuf := gpu.NewDevBufFrom(gateW)
		defer gateBuf.Free()
		upBuf := gpu.NewDevBufFrom(upW)
		defer upBuf.Free()
		downBuf := gpu.NewDevBufFrom(downW)
		defer downBuf.Free()
		gate := make([]float32, intermediate)
		up := make([]float32, intermediate)
		act := make([]float32, intermediate)
		mlpOut := make([]float32, hiddenSize)
		for off := 0; off < len(scratch.Hidden); off += hiddenSize {
			row := scratch.Hidden[off : off+hiddenSize]
			xBuf := gpu.NewDevBufFrom(row)
			gOut := gpu.NewDevBuf(intermediate)
			gpu.DevGemv(gOut, xBuf, gateBuf, intermediate, gateCols)
			gOut.ToCPU()
			copy(gate, gOut.Data()[:intermediate])
			uOut := gpu.NewDevBuf(intermediate)
			gpu.DevGemv(uOut, xBuf, upBuf, intermediate, gateCols)
			uOut.ToCPU()
			copy(up, uOut.Data()[:intermediate])
			if !simd.GELUTanhMulTo(act, gate, up) {
				return fmt.Errorf("GPU MLP activation rejected")
			}
			aBuf := gpu.NewDevBufFrom(act)
			dOut := gpu.NewDevBuf(hiddenSize)
			gpu.DevGemv(dOut, aBuf, downBuf, hiddenSize, intermediate)
			dOut.ToCPU()
			copy(mlpOut, dOut.Data()[:hiddenSize])
			xBuf.Free()
			gOut.Free()
			uOut.Free()
			aBuf.Free()
			dOut.Free()
			copy(row, mlpOut)
		}
	}

	postNorm1, err := loadFloatVector(weights, lb.PostFFNLayerNorm1)
	if err != nil {
		return err
	}
	for off := 0; off < len(scratch.Hidden); off += hiddenSize {
		if !simd.RMSNormTo(scratch.Hidden[off:off+hiddenSize], postNorm1, 1e-6) {
			return fmt.Errorf("GPU MLP post_norm_1 rejected")
		}
	}
	copy(scratch.MlpOut, scratch.Hidden)
	copy(scratch.Hidden, scratch.Residual)
	return nil
}

// batchedGPUGemm computes Out[M,N] = W[M,K] × X_T[K,N] where
// hidden is [N,K] (N positions, K=hiddenSize) stored row-major.
// Returns Out as [M,N] row-major (M output features, N positions).
func batchedGPUGemm(W []float32, hidden []float32, M, K, N int) ([]float32, error) {
	// Transpose hidden [N,K] → X_T [K,N]
	xt := make([]float32, K*N)
	for pos := 0; pos < N; pos++ {
		for k := 0; k < K; k++ {
			xt[k*N+pos] = hidden[pos*K+k]
		}
	}
	return gpu.SgemmHost(M, N, K, 1.0, W, xt)
}

// scatterGemmResult copies GEMM output [M,N] back into per-position slices.
func scatterGemmResult(dst []float32, result []float32, M, N int) {
	for pos := 0; pos < N; pos++ {
		for m := 0; m < M; m++ {
			dst[pos*M+m] = result[m*N+pos]
		}
	}
}

// runLMHeadFromShards runs the sparse top-k LM head using BF16 embed_tokens
// from a specific safetensors file (e.g. the FP8 checkpoint).
func runLMHeadFromShards(shards *safetensors.ShardedFile, scratch ForwardScratch, embedName string) error {
	raw, dtype, shape, err := shards.GetRaw(embedName)
	if err != nil {
		return fmt.Errorf("LM head embed %s: %w", embedName, err)
	}
	if dtype != "BF16" || len(shape) != 2 {
		return fmt.Errorf("LM head embed %s: dtype=%s shape=%v (need BF16 rank-2)", embedName, dtype, shape)
	}
	vocab, hiddenSize := shape[0], shape[1]
	if vocab <= 0 || hiddenSize <= 0 {
		return fmt.Errorf("LM head embed invalid shape [%d,%d]", vocab, hiddenSize)
	}
	positions := len(scratch.Hidden) / hiddenSize
	if positions <= 0 {
		return nil
	}
	bf16Embed := unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), vocab*hiddenSize)
	topK := scratch.LMHeadTopK
	if topK > vocab {
		topK = vocab
	}
	if topK <= 0 {
		return nil
	}
	for pos := 0; pos < positions; pos++ {
		if len(scratch.Logits[pos]) < vocab {
			return fmt.Errorf("LM head logits row=%d len=%d want %d", pos, len(scratch.Logits[pos]), vocab)
		}
		for i := 0; i < vocab; i++ {
			scratch.Logits[pos][i] = float32(math.Inf(-1))
		}
	}
	topIDs := make([][]int, positions)
	topVals := make([][]float32, positions)
	for pos := 0; pos < positions; pos++ {
		topIDs[pos] = make([]int, topK)
		topVals[pos] = make([]float32, topK)
		for i := range topIDs[pos] {
			topIDs[pos][i] = -1
			topVals[pos][i] = float32(math.Inf(-1))
		}
	}
	for pos := 0; pos < positions; pos++ {
		hidden := scratch.Hidden[pos*hiddenSize : (pos+1)*hiddenSize]
		hiddenBF16 := simd.BF16FromF32Slice(hidden)
		// Parallel vocab scan across cores
		nWorkers := runtime.GOMAXPROCS(0)
		if nWorkers > 12 {
			nWorkers = 12
		}
		chunk := (vocab + nWorkers - 1) / nWorkers
		type topKResult struct {
			ids  []int
			vals []float32
		}
		results := make([]topKResult, nWorkers)
		var wg sync.WaitGroup
		for w := 0; w < nWorkers; w++ {
			start := w * chunk
			end := start + chunk
			if end > vocab {
				end = vocab
			}
			if start >= end {
				break
			}
			wg.Add(1)
			go func(s, e, wi int) {
				defer wg.Done()
				ids := make([]int, topK)
				vals := make([]float32, topK)
				for i := range ids {
					ids[i] = -1
					vals[i] = float32(math.Inf(-1))
				}
				for vocabID := s; vocabID < e; vocabID++ {
					row := bf16Embed[vocabID*hiddenSize : (vocabID+1)*hiddenSize]
					score := simd.BF16DotAsm(row, hiddenBF16)
					insertTopK(ids, vals, vocabID, score)
				}
				results[wi] = topKResult{ids, vals}
			}(start, end, w)
		}
		wg.Wait()
		// Merge worker top-k results
		for _, r := range results {
			for i, id := range r.ids {
				if id >= 0 {
					insertTopK(topIDs[pos], topVals[pos], id, r.vals[i])
				}
			}
		}
	}
	for pos := 0; pos < positions; pos++ {
		for i, id := range topIDs[pos] {
			if id >= 0 {
				scratch.Logits[pos][id] = topVals[pos][i]
			}
		}
	}
	return nil
}

// runDenseGPULMHead computes full dense logits on GPU using BF16 embeddings.
func runDenseGPULMHead(fp8w *FP8TextWeights, scratch ForwardScratch, hiddenSize int) error {
	if fp8w == nil || fp8w.shards == nil {
		return fmt.Errorf("no FP8 weights for dense LM head")
	}
	raw, dtype, shape, err := fp8w.shards.GetRaw("model.decoder.embed_tokens.weight")
	if err != nil {
		return err
	}
	if dtype != "BF16" || len(shape) != 2 {
		return fmt.Errorf("dense LM head: dtype=%s shape=%v", dtype, shape)
	}
	vocab, h := shape[0], shape[1]
	if h != hiddenSize {
		return fmt.Errorf("dense LM head: hidden=%d want %d", h, hiddenSize)
	}
	positions := len(scratch.Hidden) / hiddenSize
	wBuf, err := gpu.UploadBF16LMHead(raw, vocab, h)
	if err != nil || wBuf == nil {
		return fmt.Errorf("dense LM head GPU upload: %v", err)
	}
	defer wBuf.Free()
	for pos := 0; pos < positions; pos++ {
		hidden := scratch.Hidden[pos*hiddenSize : (pos+1)*hiddenSize]
		if pos >= len(scratch.Logits) || len(scratch.Logits[pos]) < vocab {
			break
		}
		if err := gpu.BF16LMHeadWithBuffer(scratch.Logits[pos][:vocab], wBuf, hidden, vocab, h); err != nil {
			return fmt.Errorf("dense LM head pos %d: %w", pos, err)
		}
	}
	return nil
}
