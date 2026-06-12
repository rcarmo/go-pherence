package diffusiongemma

import (
	"fmt"
	"math"
	"os"
	"time"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
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
			// Non-resident layers: use BF16 CPU experts (proven correct)
			if err := runExpertsFromResidual(op, weights, scratch); err != nil {
				return ForwardOutput{}, err
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

	qW, qRows, _, err := loadFloatMatrix(weights, lb.QProj)
	if err != nil {
		return err
	}
	kW, kRows, _, err := loadFloatMatrix(weights, lb.KProj)
	if err != nil {
		return err
	}
	var vW []float32
	vRows := kRows
	if lb.VProj != nil {
		vW, vRows, _, err = loadFloatMatrix(weights, lb.VProj)
		if err != nil {
			return err
		}
	}
	oW, _, _, err := loadFloatMatrix(weights, lb.OProj)
	if err != nil {
		return err
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

	// Attention math on CPU
	group := heads / kvHeads
	enc := EncoderKVLayer{}
	if op.Layer >= 0 && op.Layer < len(ctx.EncoderKV) {
		enc = ctx.EncoderKV[op.Layer]
	}
	encSeq := 0
	if enc.SeqLen > 0 {
		encSeq = enc.SeqLen
	}
	totalKV := encSeq + positions
	scores := make([]float32, totalKV)
	slidingWindow := 0
	if op.Type == "sliding_attention" {
		slidingWindow = 1024
	}
	attnCtx := make([]float32, qRows)
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
				} else {
					canvasJ := j - encSeq
					if slidingWindow > 0 && absInt(pos-canvasJ) >= slidingWindow {
						scores[j] = float32(math.Inf(-1))
					} else {
						scores[j] = dot(q, kAll[canvasJ*kRows+kvh*headDim:canvasJ*kRows+(kvh+1)*headDim])
					}
				}
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
				for dd := range dst {
					dst[dd] += score * vv[dd]
				}
			}
		}
		// O projection on CPU (attnCtx is per-position, small)
		oOut := make([]float32, hiddenSize)
		simd.GemvRows(oOut, attnCtx, oW, hiddenSize, qRows)
		copy(scratch.Hidden[pos*hiddenSize:(pos+1)*hiddenSize], oOut)
	}
	return nil
}

func (d GPUDispatcher) gpuDenseMLP(op LayerOp, weights *TextWeights, scratch ForwardScratch, fp TextForwardPlan, hiddenSize int) error {
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("GPU MLP layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]

	gateW, gateRows, gateCols, err := loadFloatMatrix(weights, lb.MLPGateProj)
	if err != nil {
		return err
	}
	upW, _, _, err := loadFloatMatrix(weights, lb.MLPUpProj)
	if err != nil {
		return err
	}
	downW, _, _, err := loadFloatMatrix(weights, lb.MLPDownProj)
	if err != nil {
		return err
	}
	intermediate := gateRows

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
		if d.FP8Model != nil && op.Layer < len(d.FP8Model.Layers) {
			fl := &d.FP8Model.Layers[op.Layer]
			if err := gpu.GemvFP8E4M3(gate, row, fl.Gate); err != nil {
				return fmt.Errorf("FP8 gate GEMV: %w", err)
			}
			if err := gpu.GemvFP8E4M3(up, row, fl.Up); err != nil {
				return fmt.Errorf("FP8 up GEMV: %w", err)
			}
			if !simd.GELUTanhMulTo(act, gate, up) {
				return fmt.Errorf("GPU MLP activation rejected")
			}
			if err := gpu.GemvFP8E4M3(mlpOut, act, fl.Down); err != nil {
				return fmt.Errorf("FP8 down GEMV: %w", err)
			}
		} else {
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
		}
		copy(row, mlpOut)
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
