package diffusiongemma

import (
	"fmt"
	"math"
	"os"
	"time"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// GPUDispatcher offloads GEMV-heavy operations to the GPU while keeping
// control flow, attention, and sampling on the CPU.
type GPUDispatcher struct {
	ResidentLayerPrefix int
	MaxLayers           int
	TailAfterMaxLayers  bool
	LMHeadTopK          int
	Progress            bool
	SkipEviction        bool
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

// gpuGemvF32 uploads x to GPU, runs SGEMM(M,1,K), downloads result.
// W must already be on GPU as a *gpu.Buffer of M*K float32s.
func gpuGemvF32(out, x []float32, wBuf *gpu.Buffer, M, K int) error {
	xBuf, err := gpu.Malloc(K)
	if err != nil {
		return err
	}
	defer xBuf.Free()
	if err := xBuf.Upload(x[:K]); err != nil {
		return err
	}
	outBuf, err := gpu.Malloc(M)
	if err != nil {
		return err
	}
	defer outBuf.Free()
	if err := gpu.Sgemm(M, 1, K, 1.0, wBuf, xBuf, outBuf); err != nil {
		return err
	}
	return outBuf.Download(out[:M])
}

// uploadF32 allocates GPU buffer and uploads float32 data.
func uploadF32(data []float32) (*gpu.Buffer, error) {
	buf, err := gpu.Malloc(len(data))
	if err != nil {
		return nil, err
	}
	if err := buf.Upload(data); err != nil {
		buf.Free()
		return nil, err
	}
	return buf, nil
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
		fmt.Fprintf(os.Stderr, "DiffusionGemma GPU: using %s for projections\n", gpu.DeviceName())
	}

	fp := weights.ForwardPlan()
	hiddenSize := buffers.HiddenSize
	positions := len(ctx.Canvas)

	// Allocate CPU scratch (same as CPU dispatcher)
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

	// Run prefix ops on CPU (canvas embedding, self-conditioning)
	for _, op := range ops.Prefix {
		if err := dispatchPrefixOp(op, ctx, weights, scratch); err != nil {
			return ForwardOutput{}, err
		}
	}

	// Layer loop: GPU for projections, CPU for attention/norms/experts
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
			if err := d.runGPUAttention(op, ctx, weights, scratch, fp, hiddenSize, positions); err != nil {
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
			if err := d.runGPUDenseMLP(op, weights, scratch, fp, hiddenSize); err != nil {
				return ForwardOutput{}, err
			}
		case OpRouter:
			if err := runRouterFromResidual(op, weights, scratch); err != nil {
				return ForwardOutput{}, err
			}
		case OpExperts:
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

	// Tail ops on CPU (final norm, LM head)
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

// runGPUAttention runs Q/K/V/O projections on GPU, attention math on CPU.
func (d GPUDispatcher) runGPUAttention(op LayerOp, ctx ForwardContext, weights *TextWeights, scratch ForwardScratch, fp TextForwardPlan, hiddenSize, positions int) error {
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("DiffusionGemma GPU attention layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]

	// Load and upload Q/K/V/O weight matrices
	qF32, qRows, _, err := loadFloatMatrix(weights, lb.QProj)
	if err != nil {
		return err
	}
	qGPU, err := uploadF32(qF32)
	if err != nil {
		return err
	}
	defer qGPU.Free()

	kF32, kRows, _, err := loadFloatMatrix(weights, lb.KProj)
	if err != nil {
		return err
	}
	kGPU, err := uploadF32(kF32)
	if err != nil {
		return err
	}
	defer kGPU.Free()

	var vGPU *gpu.Buffer
	vRows := kRows
	if lb.VProj != nil {
		vF32, vR, _, err := loadFloatMatrix(weights, lb.VProj)
		if err != nil {
			return err
		}
		vRows = vR
		vGPU, err = uploadF32(vF32)
		if err != nil {
			return err
		}
		defer vGPU.Free()
	}

	oF32, _, _, err := loadFloatMatrix(weights, lb.OProj)
	if err != nil {
		return err
	}
	oGPU, err := uploadF32(oF32)
	if err != nil {
		return err
	}
	defer oGPU.Free()

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

	// GPU projections per position
	for pos := 0; pos < positions; pos++ {
		hidden := scratch.Hidden[pos*hiddenSize : (pos+1)*hiddenSize]
		q := qAll[pos*qRows : (pos+1)*qRows]
		k := kAll[pos*kRows : (pos+1)*kRows]
		v := vAll[pos*vRows : (pos+1)*vRows]

		if err := gpuGemvF32(q, hidden, qGPU, qRows, hiddenSize); err != nil {
			return fmt.Errorf("GPU Q GEMV: %w", err)
		}
		if err := gpuGemvF32(k, hidden, kGPU, kRows, hiddenSize); err != nil {
			return fmt.Errorf("GPU K GEMV: %w", err)
		}
		if lb.VProj != nil && vGPU != nil {
			if err := gpuGemvF32(v, hidden, vGPU, vRows, hiddenSize); err != nil {
				return fmt.Errorf("GPU V GEMV: %w", err)
			}
		} else {
			copy(v, k)
		}

		// Norms and RoPE on CPU
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

	// Attention math on CPU (same as CPU dispatcher)
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
	out := make([]float32, hiddenSize)
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
		// O projection on GPU
		if err := gpuGemvF32(out, attnCtx, oGPU, hiddenSize, qRows); err != nil {
			return fmt.Errorf("GPU O GEMV: %w", err)
		}
		copy(scratch.Hidden[pos*hiddenSize:(pos+1)*hiddenSize], out)
	}
	return nil
}

// runGPUDenseMLP runs gate/up/down projections on GPU.
func (d GPUDispatcher) runGPUDenseMLP(op LayerOp, weights *TextWeights, scratch ForwardScratch, fp TextForwardPlan, hiddenSize int) error {
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("DiffusionGemma GPU MLP layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]

	gateF32, gateRows, gateCols, err := loadFloatMatrix(weights, lb.MLPGateProj)
	if err != nil {
		return err
	}
	gateGPU, err := uploadF32(gateF32)
	if err != nil {
		return err
	}
	defer gateGPU.Free()

	upF32, _, _, err := loadFloatMatrix(weights, lb.MLPUpProj)
	if err != nil {
		return err
	}
	upGPU, err := uploadF32(upF32)
	if err != nil {
		return err
	}
	defer upGPU.Free()

	downF32, _, _, err := loadFloatMatrix(weights, lb.MLPDownProj)
	if err != nil {
		return err
	}
	downGPU, err := uploadF32(downF32)
	if err != nil {
		return err
	}
	defer downGPU.Free()

	intermediate := gateRows
	gate := make([]float32, intermediate)
	up := make([]float32, intermediate)
	act := make([]float32, intermediate)
	mlpOut := make([]float32, hiddenSize)

	for off := 0; off < len(scratch.Hidden); off += hiddenSize {
		row := scratch.Hidden[off : off+hiddenSize]
		if err := gpuGemvF32(gate, row, gateGPU, intermediate, gateCols); err != nil {
			return fmt.Errorf("GPU gate GEMV: %w", err)
		}
		if err := gpuGemvF32(up, row, upGPU, intermediate, gateCols); err != nil {
			return fmt.Errorf("GPU up GEMV: %w", err)
		}
		if !simd.GELUTanhMulTo(act, gate, up) {
			return fmt.Errorf("DiffusionGemma GPU MLP activation rejected")
		}
		if err := gpuGemvF32(mlpOut, act, downGPU, hiddenSize, intermediate); err != nil {
			return fmt.Errorf("GPU down GEMV: %w", err)
		}
		copy(row, mlpOut)
	}

	// post_feedforward_layernorm_1 + save to MlpOut + restore Hidden
	postNorm1, err := loadFloatVector(weights, lb.PostFFNLayerNorm1)
	if err != nil {
		return err
	}
	for off := 0; off < len(scratch.Hidden); off += hiddenSize {
		if !simd.RMSNormTo(scratch.Hidden[off:off+hiddenSize], postNorm1, 1e-6) {
			return fmt.Errorf("DiffusionGemma GPU MLP post_norm_1 rejected")
		}
	}
	copy(scratch.MlpOut, scratch.Hidden)
	copy(scratch.Hidden, scratch.Residual)
	return nil
}
