package diffusiongemma

import (
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"math"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"unsafe"
)

// GPUDispatcher offloads GEMV projections to GPU via DevBuf/DevGemv,
// keeping attention math, norms, and sampling on CPU.
type GPUDispatcher struct {
	ResidentLayerPrefix   int
	MaxLayers             int
	TailAfterMaxLayers    bool
	LMHeadTopK            int
	Progress              bool
	SkipEviction          bool
	CPUExperts            bool // force CPU-only expert path (skip GPU LRU cache)
	FP8Model              *GPUFP8Model
	FP8Weights            *FP8TextWeights
	FP8LMHead             *gpu.Buffer // persistent BF16 embed_tokens buffer for dense full-vocab LM head
	FP8LMHeadVocab        int
	FP8LMHeadHidden       int
	F32LMHead             *gpu.Buffer // optional persistent F32 tied LM head [hidden,vocab] for GGUF/F32 paths
	F32LMHeadVocab        int
	F32LMHeadHidden       int
	F32LMHeadChunkSize    int         // optional streamed F32 tied LM-head chunk size for GGUF/F32 paths
	F32LMHeadUseCache     bool        // use already-dequantized F32 embed cache for GGUF chunked LM head instead of Q-row streaming
	SCEmbed               *gpu.Buffer // optional persistent F32 embed_tokens [vocab,hidden] for GPU self-conditioning
	SCEmbedVocab          int
	SCEmbedHidden         int
	ExpertCache           *ExpertLRUCache
	ExpertIndex           *FP8ExpertIndex  // pre-built FP8 index for fast CPU expert path
	GGUFExpertIndex       *GGUFExpertIndex // GGUF Q4_K expert index (same weights as llama.cpp)
	FinalLogitSoftcapping float32          // tanh(x/c)*c after LM head; 0 = disabled
}

func (d CPUDispatcher) ggufDenseLayerResident(layer int) bool {
	return false
}

func (d GPUDispatcher) ggufDenseLayerResident(layer int) bool {
	return d.ResidentLayerPrefix <= 0 || layer < d.ResidentLayerPrefix
}

func (d GPUDispatcher) cpuFallback() CPUDispatcher {
	return CPUDispatcher{
		ResidentLayerPrefix:   d.ResidentLayerPrefix,
		MaxLayers:             d.MaxLayers,
		TailAfterMaxLayers:    d.TailAfterMaxLayers,
		LMHeadTopK:            d.LMHeadTopK,
		Progress:              d.Progress,
		SkipEviction:          d.SkipEviction,
		ExpertIndex:           d.ExpertIndex,
		GGUFExpertIndex:       d.GGUFExpertIndex,
		FinalLogitSoftcapping: d.FinalLogitSoftcapping,
	}
}

// EncodePrompt builds prompt KV for GPUDispatcher inference.
//
// This is the GPU-dispatcher-owned prompt prefill entrypoint. It currently
// reuses the existing prefill implementation while preserving GPU dispatcher
// context (FP8 model/weights/expert cache and GGUF expert index). CPUDispatcher
// remains available as an explicit CPU/SIMD reference path, while this method
// requires CUDA so GPU prompt-prefill gaps stay visible.
func (d GPUDispatcher) EncodePrompt(promptIDs []int, weights *TextWeights, ops ForwardOpPlan, buffers ForwardBufferPlan) ([]EncoderKVLayer, error) {
	if !gpu.SgemmReady() {
		return nil, fmt.Errorf("DiffusionGemma GPU prompt prefill requires CUDA SGEMM")
	}
	enc := d.cpuFallback()
	var fp8 *GPUFP8Model
	if os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_ENCODER_NO_FP8") == "" {
		fp8 = d.FP8Model
	}
	return enc.EncodePromptWithFP8(promptIDs, weights, ops, buffers, fp8, d.FP8Weights, d.ExpertCache)
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
	ggufExpertStatsStart := ggufExpertDispatchStatsSnapshot()
	ggufCPUExpertStatsStart := ggufCPUExpertTimingSnapshot()
	ggufAttentionStatsStart := ggufAttentionTimingSnapshot()
	ggufTempDenseStatsStart := ggufTempDenseUploadSnapshot()
	exactGELUStatsStart := f32GELUExactMulSnapshot()

	fp := weights.ForwardPlan()
	hiddenSize := buffers.HiddenSize
	positions := len(ctx.Canvas)

	scratch := NewForwardScratch(buffers)
	setDiffusionGemmaTracePhase(ctx)
	scratch.LMHeadTopK = d.LMHeadTopK
	scratch.FP8ExpertIndex = d.ExpertIndex
	scratch.GGUFExpertIndex = d.GGUFExpertIndex
	scratch.FinalLogitSoftcapping = d.FinalLogitSoftcapping
	scratch.SCTempInv = ctx.SCTempInv
	actualHidden, ok := checked.MulInt(positions, hiddenSize)
	if !ok || actualHidden > len(scratch.Hidden) || positions > len(scratch.Logits) {
		return ForwardOutput{}, fmt.Errorf("DiffusionGemma GPU dispatcher canvas=%d exceeds buffer plan canvas=%d hidden=%d", positions, buffers.CanvasLength, hiddenSize)
	}
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

	// Per-component timing accumulators
	var tAttn, tExpert, tNorm, tRouter, tDenseMLP, tOther time.Duration
	var backendGraph *DiffusionGemmaBackendGraph
	if diffusionGemmaBackendGraphEnabled() && d.FP8Model != nil && len(d.FP8Model.Layers) > 0 && d.FP8Model.Layers[0].Gate != nil {
		g, err := NewDiffusionGemmaBackendGraph(positions, hiddenSize, d.FP8Model.Layers[0].Gate.OutDim)
		if err != nil {
			return ForwardOutput{}, err
		}
		backendGraph = g
		defer backendGraph.Free()
	}
	expertsUseGPUForLayer := func(layer int) bool {
		if d.GGUFExpertIndex != nil || d.CPUExperts {
			return false
		}
		return d.ExpertCache != nil && d.FP8Weights != nil
	}
	type asyncDenseResult struct {
		err error
		dur time.Duration
	}
	var denseAsync <-chan asyncDenseResult
	waitDense := func() error {
		if denseAsync == nil {
			return nil
		}
		res := <-denseAsync
		tDenseMLP += res.dur
		denseAsync = nil
		return res.err
	}
	finishWithErr := func(err error) (ForwardOutput, error) {
		if denseAsync != nil {
			_ = waitDense()
		}
		return ForwardOutput{}, err
	}

	traceRow := diffusionGemmaLayerTraceRow()
	traceOps := diffusionGemmaLayerTraceOpsEnabled()
	traceForwardRow("prefix", -1, traceRow, scratch, buffers.HiddenSize)
	currentLayer := -1
	completedLayers := 0
	layerStarted := time.Now()
	for _, op := range ops.Layers {
		if denseAsync != nil && op.Kind != OpRouter && op.Kind != OpExperts && op.Kind != OpPostMoE {
			if err := waitDense(); err != nil {
				return ForwardOutput{}, err
			}
		}
		if currentLayer >= 0 && op.Layer != currentLayer {
			traceForwardRow("layer", currentLayer, traceRow, scratch, buffers.HiddenSize)
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
			t0 := time.Now()
			copy(scratch.Residual, scratch.Hidden)
			if err := runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.InputLayerNorm }); err != nil {
				return ForwardOutput{}, err
			}
			tNorm += time.Since(t0)
		case OpSelfAttention:
			t0 := time.Now()
			if err := d.gpuAttention(op, ctx, weights, scratch, fp, hiddenSize, positions); err != nil {
				return ForwardOutput{}, err
			}
			tAttn += time.Since(t0)
		case OpPostAttention:
			t0 := time.Now()
			if err := runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.PostAttentionLayerNorm }); err != nil {
				return ForwardOutput{}, err
			}
			for i := range scratch.Hidden {
				scratch.Hidden[i] += scratch.Residual[i]
			}
			tNorm += time.Since(t0)
		case OpPreMoE:
			t0 := time.Now()
			copy(scratch.Residual, scratch.Hidden)
			if err := runLayerRMSNorm(op, weights, scratch, func(lb TextLayerBindings) *TensorBinding { return lb.PreFFNLayerNorm }); err != nil {
				return ForwardOutput{}, err
			}
			tNorm += time.Since(t0)
		case OpDenseMLP:
			if !expertsUseGPUForLayer(op.Layer) {
				if denseAsync != nil {
					if err := waitDense(); err != nil {
						return ForwardOutput{}, err
					}
				}
				done := make(chan asyncDenseResult, 1)
				opCopy := op
				go func() {
					t0 := time.Now()
					err := d.gpuDenseMLP(opCopy, weights, scratch, fp, hiddenSize, backendGraph)
					done <- asyncDenseResult{err: err, dur: time.Since(t0)}
				}()
				denseAsync = done
			} else {
				t0 := time.Now()
				if err := d.gpuDenseMLP(op, weights, scratch, fp, hiddenSize, backendGraph); err != nil {
					return ForwardOutput{}, err
				}
				tDenseMLP += time.Since(t0)
			}
		case OpRouter:
			t0 := time.Now()
			if err := runRouterFromResidual(op, weights, scratch); err != nil {
				return finishWithErr(err)
			}
			tRouter += time.Since(t0)
		case OpExperts:
			t0 := time.Now()
			if d.GGUFExpertIndex != nil {
				// llama.cpp lowers MoE to one graph boundary built from selected-expert
				// metadata and ggml_mul_mat_id-style grouped expert matmuls. Mirror that:
				// runGGUFGPUExpertsIndexed may choose GPU or CPU/SIMD grouped backend
				// implementations internally, but this model layer no longer falls back
				// to the legacy indexed per-expert executor.
				gpuAttemptStart := time.Now()
				usedGrouped, _, err := runGGUFGPUExpertsIndexed(op, weights, scratch, d.GGUFExpertIndex)
				ggufExpertDispatchCounters.gpuAttemptNS.Add(uint64(time.Since(gpuAttemptStart).Nanoseconds()))
				if err != nil {
					return finishWithErr(err)
				}
				if !usedGrouped {
					return finishWithErr(fmt.Errorf("GGUF grouped expert backend did not handle layer %d", op.Layer))
				}
				tExpert += time.Since(t0)
				break
			} else if d.CPUExperts && d.ExpertIndex != nil {
				if err := runFP8CPUExpertsIndexed(op, weights, scratch, d.ExpertIndex); err != nil {
					return finishWithErr(err)
				}
				tExpert += time.Since(t0)
				break
			} else if d.CPUExperts {
				if err := runExpertsFromResidual(op, weights, scratch); err != nil {
					return finishWithErr(err)
				}
				tExpert += time.Since(t0)
				break
			}
			if d.ExpertCache != nil && d.FP8Weights != nil {
				if err := runLRUCachedExperts(op, weights, scratch, d.FP8Weights, d.ExpertCache); err != nil {
					if errors.Is(err, ErrExpertCacheFull) {
						if err := runExpertsFromResidual(op, weights, scratch); err != nil {
							return finishWithErr(err)
						}
					} else {
						return finishWithErr(err)
					}
				}
			} else {
				if err := runExpertsFromResidual(op, weights, scratch); err != nil {
					return finishWithErr(err)
				}
			}
			tExpert += time.Since(t0)
		case OpPostMoE:
			if err := waitDense(); err != nil {
				return ForwardOutput{}, err
			}
			t0 := time.Now()
			if err := runCombineMlpMoe(op, weights, scratch); err != nil {
				return ForwardOutput{}, err
			}
			for i := range scratch.Hidden {
				scratch.Hidden[i] += scratch.Residual[i]
			}
			tOther += time.Since(t0)
		case OpLayerScalar:
			t0 := time.Now()
			if err := runLayerScalar(op, weights, scratch); err != nil {
				return ForwardOutput{}, err
			}
			tOther += time.Since(t0)
		default:
			return ForwardOutput{}, fmt.Errorf("DiffusionGemma GPU unknown op %q", op.Kind)
		}
		if traceOps {
			traceForwardRow("op/"+string(op.Kind), op.Layer, traceRow, scratch, buffers.HiddenSize)
		}
	}
	if err := waitDense(); err != nil {
		return ForwardOutput{}, err
	}
	if currentLayer >= 0 {
		traceForwardRow("layer", currentLayer, traceRow, scratch, buffers.HiddenSize)
	}
	if currentLayer >= 0 && !d.SkipEviction && currentLayer >= d.ResidentLayerPrefix {
		weights.EvictLayer(currentLayer)
	}
	if d.Progress {
		log.Printf("forward: attn=%.1fs expert=%.1fs norm=%.1fs router=%.1fs dense=%.1fs other=%.1fs",
			tAttn.Seconds(), tExpert.Seconds(), tNorm.Seconds(), tRouter.Seconds(), tDenseMLP.Seconds(), tOther.Seconds())
		if d.GGUFExpertIndex != nil {
			tmpDenseStats := ggufTempDenseUploadSnapshot().Sub(ggufTempDenseStatsStart)
			if tmpDenseStats.Calls > 0 {
				log.Printf("gguf_temp_dense_upload: calls=%d hits=%d misses=%d bytes=%.1fMiB transpose=%.1fs upload=%.1fs f_attn=%d/%d f_mlp=%d/%d e_attn=%d/%d e_mlp=%d/%d",
					tmpDenseStats.Calls, tmpDenseStats.CacheHits, tmpDenseStats.CacheMisses,
					float64(tmpDenseStats.Bytes)/(1024*1024),
					float64(tmpDenseStats.TransposeNS)/1e9,
					float64(tmpDenseStats.UploadNS)/1e9,
					tmpDenseStats.ForwardAttnHits, tmpDenseStats.ForwardAttnCalls,
					tmpDenseStats.ForwardMLPHits, tmpDenseStats.ForwardMLPCalls,
					tmpDenseStats.EncoderAttnHits, tmpDenseStats.EncoderAttnCalls,
					tmpDenseStats.EncoderMLPHits, tmpDenseStats.EncoderMLPCalls)
			}
			attnStats := ggufAttentionTimingSnapshot().Sub(ggufAttentionStatsStart)
			if attnStats.Calls > 0 {
				covered := attnStats.ProjNS + attnStats.NormRoPENS + attnStats.KVBuildNS + attnStats.AttnNS + attnStats.OProjNS
				other := uint64(0)
				if attnStats.TotalNS > covered {
					other = attnStats.TotalNS - covered
				}
				log.Printf("gguf_attention: calls=%d total=%.1fs setup_other=%.1fs proj=%.1fs norm_rope=%.1fs kv_build=%.1fs attn=%.1fs oproj=%.1fs",
					attnStats.Calls,
					float64(attnStats.TotalNS)/1e9,
					float64(other)/1e9,
					float64(attnStats.ProjNS)/1e9,
					float64(attnStats.NormRoPENS)/1e9,
					float64(attnStats.KVBuildNS)/1e9,
					float64(attnStats.AttnNS)/1e9,
					float64(attnStats.OProjNS)/1e9)
			}
			cpuStats := ggufCPUExpertTimingSnapshot().Sub(ggufCPUExpertStatsStart)
			if cpuStats.Calls > 0 {
				log.Printf("gguf_cpu_experts: calls=%d positions=%d work_items=%d active_experts=%d q4_direct_rows=%d q4_dequant_rows=%d q8_direct_rows=%d q8_dequant_rows=%d q5_direct_rows=%d q5_dequant_rows=%d q4_batches(d=%s dq=%s) q8_batches(d=%s dq=%s) q5_batches(d=%s dq=%s) norm=%.1fs collect=%.1fs schedule=%.1fs gate=%.1fs act=%.1fs down=%.1fs scatter=%.1fs post=%.1fs",
					cpuStats.Calls, cpuStats.Positions, cpuStats.WorkItems, cpuStats.ActiveExperts, cpuStats.Q4DirectRows, cpuStats.Q4DequantRows, cpuStats.Q8DirectRows, cpuStats.Q8DequantRows, cpuStats.Q5DirectRows, cpuStats.Q5DequantRows,
					ggufCPUExpertBatchBucketsString(cpuStats.Q4DirectBatches), ggufCPUExpertBatchBucketsString(cpuStats.Q4DequantBatches), ggufCPUExpertBatchBucketsString(cpuStats.Q8DirectBatches), ggufCPUExpertBatchBucketsString(cpuStats.Q8DequantBatches), ggufCPUExpertBatchBucketsString(cpuStats.Q5DirectBatches), ggufCPUExpertBatchBucketsString(cpuStats.Q5DequantBatches),
					float64(cpuStats.NormNS)/1e9, float64(cpuStats.CollectNS)/1e9, float64(cpuStats.ScheduleNS)/1e9, float64(cpuStats.GateNS)/1e9, float64(cpuStats.ActNS)/1e9, float64(cpuStats.DownNS)/1e9, float64(cpuStats.ScatterNS)/1e9, float64(cpuStats.PostNS)/1e9)
			}
			exactGELUStats := f32GELUExactMulSnapshot().Sub(exactGELUStatsStart)
			if exactGELUStats.Calls > 0 {
				log.Printf("gguf_exact_gelu: calls=%d elements=%d download=%.1fs gelu=%.1fs upload=%.1fs", exactGELUStats.Calls, exactGELUStats.Elements, float64(exactGELUStats.DownloadNS)/1e9, float64(exactGELUStats.GELUNS)/1e9, float64(exactGELUStats.UploadNS)/1e9)
			}
			stats := ggufExpertDispatchStatsSnapshot().Sub(ggufExpertStatsStart)
			if stats.Total() > 0 {
				cacheUsed, cacheLimit := activeExpertMatrixCacheUsageBytes()
				activeAvg, workAvg, missingAvg, missingMiB, missingMaxMiB := stats.ActiveSetSummary()
				log.Printf("gguf_experts: fused=%d legacy_grouped=%d cpu_fallback=%d gpu_attempt=%.1fs cpu_fallback_time=%.1fs cache=%.1f/%.1fMiB active_sets=%d active(avg/max)=%.1f/%d work(avg/max)=%.1f/%d partial(calls kept/dropped experts work)=%d %d/%d %d/%d q4_missing(avg/max)=%.1f/%d q4_missing_bytes=%.1fMiB max=%.1fMiB exceeds=%d q4(ptr/raw_ptr/cache/transient_ptr/transient_pack/budget)=%d/%d/%d/%d/%d/%d q4_budget=%.1fMiB/%dexperts q8(ptr/cache/transient_ptr/transient_pack/budget)=%d/%d/%d/%d/%d q8_budget=%.1fMiB/%dexperts q5(ptr/budget)=%d/%d q5_budget=%.1fMiB/%dexperts",
					stats.FusedUsed, stats.LegacyGroupedUsed, stats.CPUFallback, float64(stats.GPUAttemptNS)/1e9, float64(stats.CPUFallbackNS)/1e9, float64(cacheUsed)/(1024*1024), float64(cacheLimit)/(1024*1024),
					stats.ActiveSetCalls, activeAvg, stats.ActiveSetMaxExperts, workAvg, stats.ActiveSetMaxWorkItems, stats.PartialCalls, stats.PartialKeptExperts, stats.PartialDroppedExperts, stats.PartialKeptWork, stats.PartialDroppedWork, missingAvg, stats.Q4MissingMaxExperts, missingMiB, missingMaxMiB, stats.Q4MissingBudgetExceeds,
					stats.Q4PointerTable, stats.Q4RawPointerTable, stats.Q4PackedCache, stats.Q4TransientPointer, stats.Q4TransientPacked, stats.Q4BudgetFallback, float64(stats.Q4BudgetBytes)/(1024*1024), stats.Q4BudgetExperts,
					stats.Q8PointerTable, stats.Q8PackedCache, stats.Q8TransientPointer, stats.Q8TransientPacked, stats.Q8BudgetFallback, float64(stats.Q8BudgetBytes)/(1024*1024), stats.Q8BudgetExperts,
					stats.Q5PointerTable, stats.Q5BudgetFallback, float64(stats.Q5BudgetBytes)/(1024*1024), stats.Q5BudgetExperts)
			}
		}
	}
	if d.MaxLayers > 0 && !d.TailAfterMaxLayers {
		return ForwardOutput{Logits: scratch.Logits, SelfConditioning: ctx.SelfConditioning}, nil
	}

	var tLMHead, tTailOther, tSelfCondBuild time.Duration
	ggufLMHeadStatsStart := ggufChunkedLMHeadSnapshot()
	var sampledArgmax []int
	var sampledTokens []int
	var sampledEntropy []float64
	var deviceSelfConditioning []float32
	for _, op := range ops.Tail {
		t0 := time.Now()
		if op == OpLMHead && d.F32LMHeadChunkSize > 0 {
			if diffusionGemmaRequireGroupedExpertGraph() && d.GGUFExpertIndex != nil {
				return ForwardOutput{}, fmt.Errorf("GGUF backend graph requires device-resident LM-head/self-conditioning path; chunked host-visible LM head is disabled")
			}
			if err := runChunkedF32GPULMHead(weights, scratch, buffers.HiddenSize, d.F32LMHeadChunkSize, d.F32LMHeadUseCache); err != nil {
				return ForwardOutput{}, err
			}
		} else if op == OpLMHead && d.F32LMHead != nil {
			needDeviceSC := ctx.Step > 1 && d.SCEmbed != nil
			if diffusionGemmaRequireGroupedExpertGraph() && d.GGUFExpertIndex != nil {
				if len(ctx.SampleDraws) < positions || (ctx.Step > 1 && !needDeviceSC) {
					return ForwardOutput{}, fmt.Errorf("GGUF backend graph requires device-resident LM-head/self-conditioning path")
				}
				arg, ent, samp, sc, err := runDenseF32GPULMHeadDeviceGraph(scratch, buffers.HiddenSize, d.F32LMHead, d.F32LMHeadVocab, d.F32LMHeadHidden, ctx.SampleDraws[:positions], d.SCEmbed, d.SCEmbedVocab, d.SCEmbedHidden, needDeviceSC)
				if err != nil {
					return ForwardOutput{}, err
				}
				sampledArgmax, sampledEntropy, sampledTokens, deviceSelfConditioning = arg, ent, samp, sc
			} else if err := runDenseF32GPULMHead(scratch, buffers.HiddenSize, d.F32LMHead, d.F32LMHeadVocab, d.F32LMHeadHidden); err != nil {
				return ForwardOutput{}, err
			}
		} else if op == OpLMHead && d.FP8Weights != nil {
			needDeviceSC := ctx.Step > 1 && d.SCEmbed != nil
			if diffusionGemmaDeviceSamplerEnabled() && d.FP8LMHead != nil && len(ctx.SampleDraws) >= positions && (ctx.Step <= 1 || needDeviceSC) {
				arg, ent, samp, sc, err := runDenseGPULMHeadDeviceGraph(d.FP8Weights, scratch, buffers.HiddenSize, d.FP8LMHead, d.FP8LMHeadVocab, d.FP8LMHeadHidden, ctx.SampleDraws[:positions], d.SCEmbed, d.SCEmbedVocab, d.SCEmbedHidden, needDeviceSC)
				if err != nil {
					return ForwardOutput{}, err
				}
				sampledArgmax, sampledEntropy, sampledTokens, deviceSelfConditioning = arg, ent, samp, sc
			} else if err := runDenseGPULMHead(d.FP8Weights, scratch, buffers.HiddenSize, d.FP8LMHead, d.FP8LMHeadVocab, d.FP8LMHeadHidden); err != nil {
				if scratch.LMHeadTopK <= 0 {
					return ForwardOutput{}, fmt.Errorf("dense FP8 GPU LM head failed and no sparse top-k fallback was requested: %w", err)
				}
				// Explicit debug/approximate mode: fall back to sparse BF16 scan.
				if err2 := runLMHeadFromShards(d.FP8Weights.shards, scratch, "model.decoder.embed_tokens.weight"); err2 != nil {
					return ForwardOutput{}, fmt.Errorf("dense FP8 GPU LM head failed (%v), sparse fallback also failed: %w", err, err2)
				}
			}
		} else {
			if op == OpLMHead && diffusionGemmaRequireGroupedExpertGraph() && d.GGUFExpertIndex != nil {
				return ForwardOutput{}, fmt.Errorf("GGUF backend graph requires device-resident LM-head/self-conditioning path; CPU/host LM head is disabled")
			}
			if err := dispatchTailOp(op, weights, scratch); err != nil {
				return ForwardOutput{}, err
			}
		}
		if op == OpLMHead {
			tLMHead += time.Since(t0)
		} else {
			tTailOther += time.Since(t0)
		}
	}
	t0SC := time.Now()
	var selfConditioning []float32
	var err error
	if ctx.Step > 1 {
		if len(deviceSelfConditioning) > 0 {
			selfConditioning = deviceSelfConditioning
		} else {
			if len(scratch.Logits) == 0 || len(scratch.Logits[0]) == 0 {
				return ForwardOutput{}, fmt.Errorf("self-conditioning requires logits; device sampler path did not materialize them")
			}
			if d.SCEmbed != nil {
				selfConditioning, err = buildSelfConditioningFromLogitsGPU(weights, scratch, d.SCEmbed, d.SCEmbedVocab, d.SCEmbedHidden)
			} else {
				selfConditioning, err = buildSelfConditioningFromLogits(weights, scratch)
			}
		}
	}
	tSelfCondBuild = time.Since(t0SC)
	if err != nil {
		return ForwardOutput{}, err
	}
	if d.Progress {
		log.Printf("tail: lm_head=%.1fs tail_other=%.1fs selfcond=%.1fs", tLMHead.Seconds(), tTailOther.Seconds(), tSelfCondBuild.Seconds())
		if d.GGUFExpertIndex != nil {
			lmStats := ggufChunkedLMHeadSnapshot().Sub(ggufLMHeadStatsStart)
			if lmStats.Calls > 0 {
				log.Printf("gguf_lmhead: calls=%d chunks=%d bytes=%.1fMiB prepare=%.1fs upload=%.1fs sgemm=%.1fs download=%.1fs copy=%.1fs",
					lmStats.Calls, lmStats.Chunks, float64(lmStats.Bytes)/(1024*1024),
					float64(lmStats.PrepareNS)/1e9,
					float64(lmStats.UploadNS)/1e9,
					float64(lmStats.SgemmNS)/1e9,
					float64(lmStats.DownloadNS)/1e9,
					float64(lmStats.CopyNS)/1e9)
			}
		}
	}
	return ForwardOutput{Logits: scratch.Logits, SelfConditioning: selfConditioning, ArgmaxCanvas: sampledArgmax, SampledCanvas: sampledTokens, Entropy: sampledEntropy}, nil
}

func (d GPUDispatcher) gpuAttention(op LayerOp, ctx ForwardContext, weights *TextWeights, scratch ForwardScratch, fp TextForwardPlan, hiddenSize, positions int) error {
	attnStarted := time.Now()
	var tProj, tNormRoPE, tKVBuild, tAttnComp, tOProj time.Duration
	defer func() {
		if d.GGUFExpertIndex != nil {
			recordGGUFAttentionTiming(time.Since(attnStarted), tProj, tNormRoPE, tKVBuild, tAttnComp, tOProj)
		}
		if d.Progress && op.Layer == 0 {
			log.Printf("attn[L%d]: proj=%.1fms norm_rope=%.1fms kv_build=%.1fms attn=%.1fms oproj=%.1fms",
				op.Layer, tProj.Seconds()*1000, tNormRoPE.Seconds()*1000, tKVBuild.Seconds()*1000, tAttnComp.Seconds()*1000, tOProj.Seconds()*1000)
		}
	}()
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("GPU attention layer %d outside plan", op.Layer)
	}
	lb := fp.Layers[op.Layer]

	// Load attention projection weights — skip BF16 decode when FP8 available.
	var qRows, kRows, vRows int
	var oW, qW, kW, vW []float32
	var residentAttn *GGUFGPUAttentionWeights
	if d.FP8Model != nil && op.Layer < len(d.FP8Model.Layers) {
		fl := &d.FP8Model.Layers[op.Layer]
		if fl.Q == nil || fl.K == nil || fl.O == nil {
			return fmt.Errorf("GPU attention FP8 layer %d has nil Q/K/O projection", op.Layer)
		}
		if fl.Q.InDim != hiddenSize || fl.K.InDim != hiddenSize || (fl.V != nil && fl.V.InDim != hiddenSize) {
			vIn := hiddenSize
			if fl.V != nil {
				vIn = fl.V.InDim
			}
			return fmt.Errorf("GPU attention FP8 input dim mismatch layer=%d Q=%d K=%d V=%d hidden=%d", op.Layer, fl.Q.InDim, fl.K.InDim, vIn, hiddenSize)
		}
		qRows = fl.Q.OutDim
		kRows = fl.K.OutDim
		vRows = kRows
		if fl.V != nil {
			vRows = fl.V.OutDim
		}
		if fl.O.InDim != qRows || fl.O.OutDim != hiddenSize {
			return fmt.Errorf("GPU attention FP8 O shape mismatch layer=%d O=[%d,%d] qRows=%d hidden=%d", op.Layer, fl.O.OutDim, fl.O.InDim, qRows, hiddenSize)
		}
	} else {
		var qCols, kCols, vCols, oRows, oCols int
		var err error
		qW, qRows, qCols, err = loadFloatMatrix(weights, lb.QProj)
		if err != nil {
			return err
		}
		kW, kRows, kCols, err = loadFloatMatrix(weights, lb.KProj)
		if err != nil {
			return err
		}
		vRows, vCols = kRows, hiddenSize
		if lb.VProj != nil {
			vW, vRows, vCols, err = loadFloatMatrix(weights, lb.VProj)
			if err != nil {
				return err
			}
		}
		oW, oRows, oCols, err = loadFloatMatrix(weights, lb.OProj)
		if err != nil {
			return err
		}
		if qCols != hiddenSize || kCols != hiddenSize || vCols != hiddenSize || oRows != hiddenSize || oCols != qRows {
			return fmt.Errorf("GPU attention shape mismatch layer=%d Q=[%d,%d] K=[%d,%d] V=[%d,%d] O=[%d,%d] hidden=%d", op.Layer, qRows, qCols, kRows, kCols, vRows, vCols, oRows, oCols, hiddenSize)
		}
		if d.ggufDenseLayerResident(op.Layer) {
			residentAttn, err = residentGGUFGPUAttentionWeights(op.Layer, lb, qW, kW, vW, oW, qRows, kRows, vRows, hiddenSize)
			if err != nil {
				return err
			}
		} else {
			tmp := beginGGUFTempDenseUploadSession()
			defer tmp.Close()
			qBuf, err := tmp.Upload("forward_attn_q", qW, qRows, hiddenSize)
			if err != nil {
				return fmt.Errorf("upload temporary GGUF Q layer=%d: %w", op.Layer, err)
			}
			kBuf, err := tmp.Upload("forward_attn_k", kW, kRows, hiddenSize)
			if err != nil {
				return fmt.Errorf("upload temporary GGUF K layer=%d: %w", op.Layer, err)
			}
			var vBuf *gpu.Buffer
			if lb.VProj != nil {
				vBuf, err = tmp.Upload("forward_attn_v", vW, vRows, hiddenSize)
				if err != nil {
					return fmt.Errorf("upload temporary GGUF V layer=%d: %w", op.Layer, err)
				}
			}
			oBuf, err := tmp.Upload("forward_attn_o", oW, hiddenSize, qRows)
			if err != nil {
				return fmt.Errorf("upload temporary GGUF O layer=%d: %w", op.Layer, err)
			}
			residentAttn = &GGUFGPUAttentionWeights{Q: qBuf, K: kBuf, V: vBuf, O: oBuf, QRows: qRows, KRows: kRows, VRows: vRows, Hidden: hiddenSize}
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
	if headDim <= 0 || len(kNorm) != headDim || qRows <= 0 || kRows <= 0 || vRows <= 0 || qRows%headDim != 0 || kRows%headDim != 0 || vRows != kRows {
		return fmt.Errorf("GPU attention head shape mismatch layer=%d qRows=%d kRows=%d vRows=%d qNorm=%d kNorm=%d", op.Layer, qRows, kRows, vRows, len(qNorm), len(kNorm))
	}
	heads := qRows / headDim
	kvHeads := kRows / headDim
	if kvHeads <= 0 || heads <= 0 || heads%kvHeads != 0 {
		return fmt.Errorf("GPU attention GQA shape mismatch layer=%d heads=%d kvHeads=%d headDim=%d", op.Layer, heads, kvHeads, headDim)
	}

	// Upload weight matrices to GPU once per layer

	qAllLen, okQ := checked.MulInt(positions, qRows)
	kAllLen, okK := checked.MulInt(positions, kRows)
	vAllLen, okV := checked.MulInt(positions, vRows)
	if !okQ || !okK || !okV {
		return fmt.Errorf("GPU attention buffer overflow positions=%d q=%d k=%d v=%d", positions, qRows, kRows, vRows)
	}
	qAll := make([]float32, qAllLen)
	kAll := make([]float32, kAllLen)
	vAll := make([]float32, vAllLen)

	ropeHalf := headDim / 2
	ropeTheta := 10000.0
	var ropeFactors []float32
	if op.Type == "full_attention" {
		// llama.cpp: full-attention layers use n_rot_full=headDim plus
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

	t0Proj := time.Now()
	// GPU projections: FP8 GEMV if available, else batched F32 SGEMM
	if d.FP8Model != nil && op.Layer < len(d.FP8Model.Layers) {
		fl := &d.FP8Model.Layers[op.Layer]
		// Batched FP8 QKV: upload hidden once, run 3 GEMM kernels, download Q/K/V.
		hiddenIn := scratch.Hidden[:positions*hiddenSize]
		var qHidden []float32
		if diffusionGemmaFP8DynamicActivationEnabled() {
			qHidden = quantizeDynamicTokenBatch(qHidden, hiddenIn, positions, hiddenSize)
			hiddenIn = qHidden
		}
		if err := gpu.BatchedFP8QKV(qAll, kAll, vAll, hiddenIn, positions, fl.Q, fl.K, fl.V); err != nil {
			return fmt.Errorf("FP8 QKV: %w", err)
		}
	} else {
		if residentAttn == nil || residentAttn.Q == nil || residentAttn.K == nil || residentAttn.O == nil {
			return fmt.Errorf("GPU attention missing resident GGUF Q/K/O weights layer=%d", op.Layer)
		}
		if err := batchedGPUGemmTransposed(qAll, scratch.Hidden[:positions*hiddenSize], positions, qRows, hiddenSize, residentAttn.Q); err != nil {
			return fmt.Errorf("resident GPU Q GEMM: %w", err)
		}
		if err := batchedGPUGemmTransposed(kAll, scratch.Hidden[:positions*hiddenSize], positions, kRows, hiddenSize, residentAttn.K); err != nil {
			return fmt.Errorf("resident GPU K GEMM: %w", err)
		}
		if lb.VProj != nil {
			if residentAttn.V == nil {
				return fmt.Errorf("GPU attention missing resident GGUF V weights layer=%d", op.Layer)
			}
			if err := batchedGPUGemmTransposed(vAll, scratch.Hidden[:positions*hiddenSize], positions, vRows, hiddenSize, residentAttn.V); err != nil {
				return fmt.Errorf("resident GPU V GEMM: %w", err)
			}
		} else {
			copy(vAll, kAll)
		}
	}

	tProj = time.Since(t0Proj)
	t0Norm := time.Now()
	for pos := 0; pos < positions; pos++ {
		q := qAll[pos*qRows : (pos+1)*qRows]
		k := kAll[pos*kRows : (pos+1)*kRows]
		v := vAll[pos*vRows : (pos+1)*vRows]

		// Norms + RoPE on CPU. llama.cpp applies q/k RMSNorm with scale and
		// V no-scale RMSNorm after copying raw K when V projection is absent.
		for h := 0; h < heads; h++ {
			if !simd.RMSNormTo(q[h*headDim:(h+1)*headDim], qNorm, 1e-6) {
				return fmt.Errorf("GPU attention Q RMSNorm rejected layer=%d pos=%d head=%d", op.Layer, pos, h)
			}
		}
		for h := 0; h < kvHeads; h++ {
			if !simd.RMSNormTo(k[h*headDim:(h+1)*headDim], kNorm, 1e-6) {
				return fmt.Errorf("GPU attention K RMSNorm rejected layer=%d pos=%d head=%d", op.Layer, pos, h)
			}
			if !simd.RMSNormNoScaleTo(v[h*headDim:(h+1)*headDim], 1e-6) {
				return fmt.Errorf("GPU attention V RMSNorm rejected layer=%d pos=%d head=%d", op.Layer, pos, h)
			}
		}
		if len(ropeFreqs) > 0 && ropeHalf > 0 {
			simd.ApplyRoPEPartial(q, ropeFreqs, pos+ctx.EncoderSeqLen, heads, headDim, ropeHalf)
			simd.ApplyRoPEPartial(k, ropeFreqs, pos+ctx.EncoderSeqLen, kvHeads, headDim, ropeHalf)
		}
	}

	tNormRoPE = time.Since(t0Norm)
	t0KV := time.Now()
	// Attention: GPU GQA when available, CPU fallback
	enc := EncoderKVLayer{}
	if op.Layer >= 0 && op.Layer < len(ctx.EncoderKV) {
		enc = ctx.EncoderKV[op.Layer]
	}
	encSeq := 0
	if enc.SeqLen > 0 {
		if enc.KVHeads != kvHeads || enc.HeadDim != headDim || len(enc.Keys) < enc.SeqLen*kRows || len(enc.Values) < enc.SeqLen*vRows {
			return fmt.Errorf("GPU attention encoder KV layer %d shape mismatch seq=%d kv_heads=%d head_dim=%d", op.Layer, enc.SeqLen, enc.KVHeads, enc.HeadDim)
		}
		encSeq = enc.SeqLen
	}
	totalKV := encSeq + positions
	if totalKV < encSeq {
		return fmt.Errorf("GPU attention total KV overflow enc=%d positions=%d", encSeq, positions)
	}

	// Build concatenated KV cache: [totalKV, kvHeads * headDim]
	kvDim, ok := checked.MulInt(kvHeads, headDim)
	kvLen, okLen := checked.MulInt(totalKV, kvDim)
	if !ok || !okLen {
		return fmt.Errorf("GPU attention KV concat size overflow total=%d kv_dim=%d", totalKV, kvDim)
	}
	kConcat := make([]float32, kvLen)
	vConcat := make([]float32, kvLen)
	for j := 0; j < encSeq; j++ {
		copy(kConcat[j*kvDim:(j+1)*kvDim], enc.Keys[j*kRows:j*kRows+kvDim])
		copy(vConcat[j*kvDim:(j+1)*kvDim], enc.Values[j*vRows:j*vRows+kvDim])
	}
	for j := 0; j < positions; j++ {
		copy(kConcat[(encSeq+j)*kvDim:(encSeq+j+1)*kvDim], kAll[j*kRows:j*kRows+kvDim])
		copy(vConcat[(encSeq+j)*kvDim:(encSeq+j+1)*kvDim], vAll[j*vRows:j*vRows+kvDim])
	}

	tKVBuild = time.Since(t0KV)
	t0Attn := time.Now()
	attnAllLen, okAttn := checked.MulInt(positions, qRows)
	if !okAttn {
		return fmt.Errorf("GPU attention output buffer overflow positions=%d q=%d", positions, qRows)
	}
	attnAll := make([]float32, attnAllLen)
	// Parallel CPU GQA attention — faster than 256 individual GPU kernel launches
	// at canvas=256 because the total FLOPs are small (390M) but GPU launch
	// overhead is ~18ms × 256 = 4.6s.
	group := heads / kvHeads
	slidingWindow := scratch.SlidingWindow
	if slidingWindow <= 0 {
		// llama.cpp reads attention.sliding_window from GGUF/config metadata; this
		// fallback preserves legacy callers that do not populate ForwardBufferPlan.
		slidingWindow = 1024
	}
	forceCPUAttention := op.Type == "sliding_attention" && encSeq >= slidingWindow
	if positions >= 16 || forceCPUAttention {
		// Parallel: each goroutine handles a chunk of positions
		nWorkers := 12
		if nWorkers > positions {
			nWorkers = positions
		}
		chunkSize := (positions + nWorkers - 1) / nWorkers
		var wg sync.WaitGroup
		for w := 0; w < nWorkers; w++ {
			w := w
			wg.Add(1)
			go func() {
				defer wg.Done()
				start := w * chunkSize
				end := start + chunkSize
				if end > positions {
					end = positions
				}
				scores := make([]float32, totalKV)
				canvasPromptLo := encSeq - slidingWindow + 1
				for pos := start; pos < end; pos++ {
					attnCtx := attnAll[pos*qRows : (pos+1)*qRows]
					qPos := qAll[pos*qRows : (pos+1)*qRows]
					for i := range attnCtx {
						attnCtx[i] = 0
					}
					for h := 0; h < heads; h++ {
						kvh := h / group
						q := qPos[h*headDim : (h+1)*headDim]
						for j := 0; j < totalKV; j++ {
							if op.Type == "sliding_attention" && j < encSeq && j < canvasPromptLo {
								scores[j] = float32(math.Inf(-1))
								continue
							}
							scores[j] = simd.Sdot(q, kConcat[j*kvDim+kvh*headDim:j*kvDim+(kvh+1)*headDim])
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
			}()
		}
		wg.Wait()
	} else {
		// Small canvas: use GPU batched attention (low launch overhead). This path
		// is only used when no additive mask is needed beyond bidirectional decode.
		if err := gpu.F32BatchedGQAAttention(attnAll, qAll, kConcat, vConcat, positions, totalKV, heads, kvHeads, headDim, 1.0); err != nil {
			// CPU fallback
			for pos := 0; pos < positions; pos++ {
				attnCtx := attnAll[pos*qRows : (pos+1)*qRows]
				qPos := qAll[pos*qRows : (pos+1)*qRows]
				for i := range attnCtx {
					attnCtx[i] = 0
				}
				scores := make([]float32, totalKV)
				for h := 0; h < heads; h++ {
					kvh := h / group
					q := qPos[h*headDim : (h+1)*headDim]
					for j := 0; j < totalKV; j++ {
						scores[j] = simd.Sdot(q, kConcat[j*kvDim+kvh*headDim:j*kvDim+(kvh+1)*headDim])
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
	}
	tAttnComp = time.Since(t0Attn)
	t0O := time.Now()
	// Batched O projection — FP8 GEMM if available, else per-position CPU
	if d.FP8Model != nil && op.Layer < len(d.FP8Model.Layers) && d.FP8Model.Layers[op.Layer].O != nil {
		// Batched O projection: attnAll is already contiguous [positions, qRows]
		oOutLen, okO := checked.MulInt(positions, hiddenSize)
		if !okO {
			return fmt.Errorf("GPU attention O output overflow positions=%d hidden=%d", positions, hiddenSize)
		}
		oOut := make([]float32, oOutLen)
		attnIn := attnAll
		var qAttn []float32
		if diffusionGemmaFP8DynamicActivationEnabled() {
			qAttn = quantizeDynamicTokenBatch(qAttn, attnAll, positions, qRows)
			attnIn = qAttn
		}
		if err := gpu.BatchedFP8OProj(oOut, attnIn, positions, d.FP8Model.Layers[op.Layer].O); err != nil {
			return fmt.Errorf("FP8 O GEMM: %w", err)
		}
		copy(scratch.Hidden[:positions*hiddenSize], oOut)
	} else {
		if residentAttn == nil || residentAttn.O == nil {
			return fmt.Errorf("GPU attention missing resident GGUF O weights layer=%d", op.Layer)
		}
		if err := batchedGPUGemmTransposed(scratch.Hidden[:positions*hiddenSize], attnAll, positions, hiddenSize, qRows, residentAttn.O); err != nil {
			return fmt.Errorf("resident GPU O GEMM: %w", err)
		}
	}
	tOProj = time.Since(t0O)
	return nil
}

func (d GPUDispatcher) gpuDenseMLP(op LayerOp, weights *TextWeights, scratch ForwardScratch, fp TextForwardPlan, hiddenSize int, backendGraph *DiffusionGemmaBackendGraph) error {
	if op.Layer < 0 || op.Layer >= len(fp.Layers) {
		return fmt.Errorf("GPU MLP layer %d outside plan", op.Layer)
	}
	if hiddenSize <= 0 || len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("GPU MLP hidden len=%d hidden_size=%d", len(scratch.Hidden), hiddenSize)
	}
	positions := len(scratch.Hidden) / hiddenSize
	if positions <= 0 {
		return nil
	}
	lb := fp.Layers[op.Layer]

	var gateW, upW, downW []float32
	var intermediate int
	if d.FP8Model != nil && op.Layer < len(d.FP8Model.Layers) {
		fl := &d.FP8Model.Layers[op.Layer]
		if fl.Gate == nil || fl.Up == nil || fl.Down == nil {
			return fmt.Errorf("GPU MLP FP8 layer %d has nil gate/up/down projection", op.Layer)
		}
		intermediate = fl.Gate.OutDim
		if intermediate <= 0 || fl.Gate.InDim != hiddenSize || fl.Up.InDim != hiddenSize || fl.Up.OutDim != intermediate || fl.Down.InDim != intermediate || fl.Down.OutDim != hiddenSize {
			return fmt.Errorf("GPU MLP FP8 shape mismatch layer=%d gate=[%d,%d] up=[%d,%d] down=[%d,%d] hidden=%d", op.Layer, fl.Gate.OutDim, fl.Gate.InDim, fl.Up.OutDim, fl.Up.InDim, fl.Down.OutDim, fl.Down.InDim, hiddenSize)
		}
	} else {
		var gateCols, upRows, upCols, downRows, downCols int
		var err error
		gateW, intermediate, gateCols, err = loadFloatMatrix(weights, lb.MLPGateProj)
		if err != nil {
			return err
		}
		upW, upRows, upCols, err = loadFloatMatrix(weights, lb.MLPUpProj)
		if err != nil {
			return err
		}
		downW, downRows, downCols, err = loadFloatMatrix(weights, lb.MLPDownProj)
		if err != nil {
			return err
		}
		if intermediate <= 0 || gateCols != hiddenSize || upRows != intermediate || upCols != hiddenSize || downRows != hiddenSize || downCols != intermediate {
			return fmt.Errorf("GPU MLP shape mismatch layer=%d gate=[%d,%d] up=[%d,%d] down=[%d,%d] hidden=%d", op.Layer, intermediate, gateCols, upRows, upCols, downRows, downCols, hiddenSize)
		}
	}

	midLen, okMid := checked.MulInt(positions, intermediate)
	hidLen, okHid := checked.MulInt(positions, hiddenSize)
	if !okMid || !okHid {
		return fmt.Errorf("GPU MLP batch size overflow positions=%d intermediate=%d hidden=%d", positions, intermediate, hiddenSize)
	}
	if d.FP8Model != nil && op.Layer < len(d.FP8Model.Layers) {
		fl := &d.FP8Model.Layers[op.Layer]
		if backendGraph != nil && backendGraph.Compatible(positions, hiddenSize, intermediate) {
			if err := backendGraph.DenseMLP(scratch.Hidden[:hidLen], fl); err != nil {
				return err
			}
		} else if diffusionGemmaDenseDeviceMLPEnabled() {
			if err := runFP8DenseMLPDevice(fl, scratch, positions, hiddenSize, intermediate); err != nil {
				return fmt.Errorf("FP8 dense MLP device path: %w", err)
			}
		} else {
			// Batched FP8 GEMM: prefer resident dequantized/transposed F32 weights to
			// avoid per-step FP8 dequant+transpose on the shared dense MLP.
			gateBatch := make([]float32, midLen)
			upBatch := make([]float32, midLen)
			denseIn := scratch.Hidden[:hidLen]
			var qDense []float32
			if diffusionGemmaFP8DynamicActivationEnabled() {
				qDense = quantizeDynamicTokenBatch(qDense, denseIn, positions, hiddenSize)
				denseIn = qDense
			}
			if fl.GateT != nil && fl.UpT != nil {
				if err := batchedGPUGemm2Transposed(gateBatch, upBatch, denseIn, positions, fl.Gate.OutDim, fl.Gate.InDim, fl.GateT, fl.Up.OutDim, fl.Up.InDim, fl.UpT); err != nil {
					return fmt.Errorf("FP8 gate/up resident SGEMM: %w", err)
				}
			} else {
				if err := gpu.GemmFP8E4M3(gateBatch, denseIn, positions, fl.Gate); err != nil {
					return fmt.Errorf("FP8 gate GEMM: %w", err)
				}
				if err := gpu.GemmFP8E4M3(upBatch, denseIn, positions, fl.Up); err != nil {
					return fmt.Errorf("FP8 up GEMM: %w", err)
				}
			}
			actBatch := make([]float32, midLen)
			for i := 0; i < positions; i++ {
				if !diffusionGemmaGELUMulTo(actBatch[i*intermediate:(i+1)*intermediate], gateBatch[i*intermediate:(i+1)*intermediate], upBatch[i*intermediate:(i+1)*intermediate]) {
					return fmt.Errorf("FP8 MLP activation rejected")
				}
			}
			downBatch := make([]float32, hidLen)
			downIn := actBatch
			var qAct []float32
			if diffusionGemmaFP8DynamicActivationEnabled() {
				qAct = quantizeDynamicTokenBatch(nil, actBatch, positions, intermediate)
				downIn = qAct
			}
			if fl.DownT != nil {
				if err := batchedGPUGemmTransposed(downBatch, downIn, positions, fl.Down.OutDim, fl.Down.InDim, fl.DownT); err != nil {
					return fmt.Errorf("FP8 down resident SGEMM: %w", err)
				}
			} else if err := gpu.GemmFP8E4M3(downBatch, downIn, positions, fl.Down); err != nil {
				return fmt.Errorf("FP8 down GEMM: %w", err)
			}
			copy(scratch.Hidden[:hidLen], downBatch)
		}
	} else {
		// GGUF/F32 SGEMM path. By default, upload gate/up/down once per layer and
		// reuse across prompt prefill and denoise steps. When ResidentLayerPrefix is
		// positive, layers outside the prefix use temporary device weights to avoid
		// unbounded resident-cache growth under dense + LM-head + expert residency.
		var resident *GGUFGPUMLPWeights
		var err error
		if d.ggufDenseLayerResident(op.Layer) {
			resident, err = residentGGUFGPUMLPWeights(op.Layer, lb, gateW, upW, downW, hiddenSize, intermediate)
			if err != nil {
				return err
			}
		} else {
			tmp := beginGGUFTempDenseUploadSession()
			defer tmp.Close()
			gateBuf, err := tmp.Upload("forward_mlp_gate", gateW, intermediate, hiddenSize)
			if err != nil {
				return fmt.Errorf("upload temporary GGUF MLP gate layer=%d: %w", op.Layer, err)
			}
			upBuf, err := tmp.Upload("forward_mlp_up", upW, intermediate, hiddenSize)
			if err != nil {
				return fmt.Errorf("upload temporary GGUF MLP up layer=%d: %w", op.Layer, err)
			}
			downBuf, err := tmp.Upload("forward_mlp_down", downW, hiddenSize, intermediate)
			if err != nil {
				return fmt.Errorf("upload temporary GGUF MLP down layer=%d: %w", op.Layer, err)
			}
			resident = &GGUFGPUMLPWeights{Gate: gateBuf, Up: upBuf, Down: downBuf, Hidden: hiddenSize, Intermediate: intermediate}
		}
		gateBatch := make([]float32, midLen)
		upBatch := make([]float32, midLen)
		if err := batchedGPUGemmTransposed(gateBatch, scratch.Hidden[:hidLen], positions, intermediate, hiddenSize, resident.Gate); err != nil {
			return fmt.Errorf("resident GPU MLP gate SGEMM: %w", err)
		}
		if err := batchedGPUGemmTransposed(upBatch, scratch.Hidden[:hidLen], positions, intermediate, hiddenSize, resident.Up); err != nil {
			return fmt.Errorf("resident GPU MLP up SGEMM: %w", err)
		}
		actBatch := make([]float32, midLen)
		for pos := 0; pos < positions; pos++ {
			if !diffusionGemmaGELUMulTo(actBatch[pos*intermediate:(pos+1)*intermediate], gateBatch[pos*intermediate:(pos+1)*intermediate], upBatch[pos*intermediate:(pos+1)*intermediate]) {
				return fmt.Errorf("GPU MLP activation rejected")
			}
		}
		if err := batchedGPUGemmTransposed(scratch.Hidden[:hidLen], actBatch, positions, hiddenSize, intermediate, resident.Down); err != nil {
			return fmt.Errorf("resident GPU MLP down SGEMM: %w", err)
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

func diffusionGemmaDeviceSamplerEnabled() bool {
	v := os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_DEVICE_SAMPLER")
	return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on"
}

func diffusionGemmaBackendGraphEnabled() bool {
	v := os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_BACKEND_GRAPH")
	return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on"
}

func diffusionGemmaRequireGroupedExpertGraph() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_REQUIRE_GROUPED_EXPERT_GRAPH")))
	return v == "1" || v == "true" || v == "yes" || v == "on" || diffusionGemmaBackendGraphEnabled()
}

func diffusionGemmaDenseDeviceMLPEnabled() bool {
	v := os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_DENSE_DEVICE")
	return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on"
}

func runFP8DenseMLPDevice(fl *GPUFP8Layer, scratch ForwardScratch, positions, hiddenSize, intermediate int) error {
	if fl == nil || fl.Gate == nil || fl.Up == nil || fl.Down == nil {
		return fmt.Errorf("missing FP8 dense MLP linears")
	}
	hidLen, okHid := checked.MulInt(positions, hiddenSize)
	midLen, okMid := checked.MulInt(positions, intermediate)
	if positions <= 0 || hiddenSize <= 0 || intermediate <= 0 || !okHid || !okMid || len(scratch.Hidden) < hidLen {
		return fmt.Errorf("invalid FP8 dense MLP device dims positions=%d hidden=%d intermediate=%d", positions, hiddenSize, intermediate)
	}
	xBuf, err := gpu.Malloc(hidLen)
	if err != nil {
		return fmt.Errorf("alloc dense MLP input: %w", err)
	}
	defer xBuf.Free()
	gateBuf, err := gpu.Malloc(midLen)
	if err != nil {
		return fmt.Errorf("alloc dense MLP gate: %w", err)
	}
	defer gateBuf.Free()
	upBuf, err := gpu.Malloc(midLen)
	if err != nil {
		return fmt.Errorf("alloc dense MLP up: %w", err)
	}
	defer upBuf.Free()
	downBuf, err := gpu.Malloc(hidLen)
	if err != nil {
		return fmt.Errorf("alloc dense MLP down: %w", err)
	}
	defer downBuf.Free()
	denseIn := scratch.Hidden[:hidLen]
	var qDense []float32
	if diffusionGemmaFP8DynamicActivationEnabled() {
		qDense = quantizeDynamicTokenBatch(qDense, denseIn, positions, hiddenSize)
		denseIn = qDense
	}
	if err := xBuf.Upload(denseIn); err != nil {
		return fmt.Errorf("upload dense MLP input: %w", err)
	}
	if err := gpu.GemmFP8E4M3ViaSgemmBuffer(gateBuf, xBuf, positions, fl.Gate); err != nil {
		return fmt.Errorf("gate SGEMM: %w", err)
	}
	if err := gpu.GemmFP8E4M3ViaSgemmBuffer(upBuf, xBuf, positions, fl.Up); err != nil {
		return fmt.Errorf("up SGEMM: %w", err)
	}
	if err := f32GELUExactMulBuffer(gateBuf, upBuf, midLen); err != nil {
		return fmt.Errorf("exact GELU activation: %w", err)
	}
	if diffusionGemmaFP8DynamicActivationEnabled() {
		act := make([]float32, midLen)
		if err := gateBuf.Download(act); err != nil {
			return fmt.Errorf("download dense MLP activation for dynamic FP8 down: %w", err)
		}
		qAct := quantizeDynamicTokenBatch(nil, act, positions, intermediate)
		if err := gateBuf.Upload(qAct); err != nil {
			return fmt.Errorf("upload dense MLP dynamic FP8 down activation: %w", err)
		}
	}
	if err := gpu.GemmFP8E4M3ViaSgemmBuffer(downBuf, gateBuf, positions, fl.Down); err != nil {
		return fmt.Errorf("down SGEMM: %w", err)
	}
	if err := downBuf.Download(scratch.Hidden[:hidLen]); err != nil {
		return fmt.Errorf("download dense MLP output: %w", err)
	}
	return nil
}

type residentF32MatrixKey struct {
	ptr uintptr
	m   int
	k   int
}

var residentF32MatrixCache sync.Map // map[residentF32MatrixKey]*gpu.Buffer

func ResidentF32MatrixCacheStats() (entries int, bytes int64) {
	residentF32MatrixCache.Range(func(_, v any) bool {
		if b, ok := v.(*gpu.Buffer); ok && b != nil {
			entries++
			bytes += int64(b.Size)
		}
		return true
	})
	return entries, bytes
}

func FreeResidentF32MatrixCache() {
	residentF32MatrixCache.Range(func(_, v any) bool {
		if b, ok := v.(*gpu.Buffer); ok && b != nil {
			b.Free()
		}
		return true
	})
	residentF32MatrixCache = sync.Map{}
}

type GGUFGPUAttentionWeights struct {
	Q      *gpu.Buffer
	K      *gpu.Buffer
	V      *gpu.Buffer
	O      *gpu.Buffer
	QRows  int
	KRows  int
	VRows  int
	Hidden int
}

type ggufGPUAttentionKey struct {
	layer int
	qName string
	kName string
	vName string
	oName string
}

var ggufGPUAttentionCache sync.Map // map[ggufGPUAttentionKey]*GGUFGPUAttentionWeights

type GGUFGPUMLPWeights struct {
	Gate         *gpu.Buffer
	Up           *gpu.Buffer
	Down         *gpu.Buffer
	Hidden       int
	Intermediate int
}

type ggufGPUMLPKey struct {
	layer int
	gate  string
	up    string
	down  string
}

var ggufGPUMLPCache sync.Map // map[ggufGPUMLPKey]*GGUFGPUMLPWeights

func GGUFGPUWeightCacheStats() (attentionEntries, mlpEntries int, bytes int64) {
	ggufGPUAttentionCache.Range(func(_, v any) bool {
		if w, ok := v.(*GGUFGPUAttentionWeights); ok && w != nil {
			attentionEntries++
			for _, b := range []*gpu.Buffer{w.Q, w.K, w.V, w.O} {
				if b != nil {
					bytes += int64(b.Size)
				}
			}
		}
		return true
	})
	ggufGPUMLPCache.Range(func(_, v any) bool {
		if w, ok := v.(*GGUFGPUMLPWeights); ok && w != nil {
			mlpEntries++
			for _, b := range []*gpu.Buffer{w.Gate, w.Up, w.Down} {
				if b != nil {
					bytes += int64(b.Size)
				}
			}
		}
		return true
	})
	return attentionEntries, mlpEntries, bytes
}

func FreeGGUFGPUWeightCaches() {
	ggufGPUAttentionCache.Range(func(_, v any) bool {
		if w, ok := v.(*GGUFGPUAttentionWeights); ok && w != nil {
			for _, b := range []*gpu.Buffer{w.Q, w.K, w.V, w.O} {
				if b != nil {
					b.Free()
				}
			}
		}
		return true
	})
	ggufGPUMLPCache.Range(func(_, v any) bool {
		if w, ok := v.(*GGUFGPUMLPWeights); ok && w != nil {
			for _, b := range []*gpu.Buffer{w.Gate, w.Up, w.Down} {
				if b != nil {
					b.Free()
				}
			}
		}
		return true
	})
	ggufGPUAttentionCache = sync.Map{}
	ggufGPUMLPCache = sync.Map{}
}

func residentGGUFGPUMLPWeights(layer int, lb TextLayerBindings, gateW, upW, downW []float32, hidden, intermediate int) (*GGUFGPUMLPWeights, error) {
	key := ggufGPUMLPKey{layer: layer, gate: lb.MLPGateProj.Name, up: lb.MLPUpProj.Name, down: lb.MLPDownProj.Name}
	if cached, ok := ggufGPUMLPCache.Load(key); ok {
		return cached.(*GGUFGPUMLPWeights), nil
	}
	gateBuf, err := uploadTransposedF32Matrix(gateW, intermediate, hidden)
	if err != nil {
		return nil, fmt.Errorf("upload resident GGUF MLP gate layer=%d: %w", layer, err)
	}
	upBuf, err := uploadTransposedF32Matrix(upW, intermediate, hidden)
	if err != nil {
		gateBuf.Free()
		return nil, fmt.Errorf("upload resident GGUF MLP up layer=%d: %w", layer, err)
	}
	downBuf, err := uploadTransposedF32Matrix(downW, hidden, intermediate)
	if err != nil {
		gateBuf.Free()
		upBuf.Free()
		return nil, fmt.Errorf("upload resident GGUF MLP down layer=%d: %w", layer, err)
	}
	weights := &GGUFGPUMLPWeights{Gate: gateBuf, Up: upBuf, Down: downBuf, Hidden: hidden, Intermediate: intermediate}
	actual, loaded := ggufGPUMLPCache.LoadOrStore(key, weights)
	if loaded {
		gateBuf.Free()
		upBuf.Free()
		downBuf.Free()
		return actual.(*GGUFGPUMLPWeights), nil
	}
	return weights, nil
}

var errActiveExpertMatrixCacheBudget = errors.New("active expert matrix cache budget exceeded")

type ggufExpertDispatchStats struct {
	FusedUsed              uint64
	LegacyGroupedUsed      uint64
	CPUFallback            uint64
	Q4PointerTable         uint64
	Q4RawPointerTable      uint64
	Q4PackedCache          uint64
	Q4TransientPacked      uint64
	Q4TransientPointer     uint64
	Q4BudgetFallback       uint64
	Q4BudgetBytes          uint64
	Q4BudgetExperts        uint64
	Q8PointerTable         uint64
	Q8PackedCache          uint64
	Q8TransientPacked      uint64
	Q8TransientPointer     uint64
	Q8BudgetFallback       uint64
	Q8BudgetBytes          uint64
	Q8BudgetExperts        uint64
	Q5PointerTable         uint64
	Q5BudgetFallback       uint64
	Q5BudgetBytes          uint64
	Q5BudgetExperts        uint64
	ActiveSetCalls         uint64
	ActiveSetExperts       uint64
	ActiveSetMaxExperts    uint64
	ActiveSetWorkItems     uint64
	ActiveSetMaxWorkItems  uint64
	Q4MissingExperts       uint64
	Q4MissingMaxExperts    uint64
	Q4MissingBytes         uint64
	Q4MissingMaxBytes      uint64
	Q4MissingBudgetExceeds uint64
	PartialCalls           uint64
	PartialKeptExperts     uint64
	PartialDroppedExperts  uint64
	PartialKeptWork        uint64
	PartialDroppedWork     uint64
	GPUAttemptNS           uint64
	CPUFallbackNS          uint64
}

var ggufExpertDispatchCounters struct {
	fusedUsed              atomic.Uint64
	legacyGroupedUsed      atomic.Uint64
	cpuFallback            atomic.Uint64
	q4PointerTable         atomic.Uint64
	q4RawPointerTable      atomic.Uint64
	q4PackedCache          atomic.Uint64
	q4TransientPacked      atomic.Uint64
	q4TransientPointer     atomic.Uint64
	q4BudgetFallback       atomic.Uint64
	q4BudgetBytes          atomic.Uint64
	q4BudgetExperts        atomic.Uint64
	q8PointerTable         atomic.Uint64
	q8PackedCache          atomic.Uint64
	q8TransientPacked      atomic.Uint64
	q8TransientPointer     atomic.Uint64
	q8BudgetFallback       atomic.Uint64
	q8BudgetBytes          atomic.Uint64
	q8BudgetExperts        atomic.Uint64
	q5PointerTable         atomic.Uint64
	q5BudgetFallback       atomic.Uint64
	q5BudgetBytes          atomic.Uint64
	q5BudgetExperts        atomic.Uint64
	activeSetCalls         atomic.Uint64
	activeSetExperts       atomic.Uint64
	activeSetMaxExperts    atomic.Uint64
	activeSetWorkItems     atomic.Uint64
	activeSetMaxWorkItems  atomic.Uint64
	q4MissingExperts       atomic.Uint64
	q4MissingMaxExperts    atomic.Uint64
	q4MissingBytes         atomic.Uint64
	q4MissingMaxBytes      atomic.Uint64
	q4MissingBudgetExceeds atomic.Uint64
	partialCalls           atomic.Uint64
	partialKeptExperts     atomic.Uint64
	partialDroppedExperts  atomic.Uint64
	partialKeptWork        atomic.Uint64
	partialDroppedWork     atomic.Uint64
	gpuAttemptNS           atomic.Uint64
	cpuFallbackNS          atomic.Uint64
}

func ggufExpertDispatchStatsSnapshot() ggufExpertDispatchStats {
	return ggufExpertDispatchStats{
		FusedUsed:              ggufExpertDispatchCounters.fusedUsed.Load(),
		LegacyGroupedUsed:      ggufExpertDispatchCounters.legacyGroupedUsed.Load(),
		CPUFallback:            ggufExpertDispatchCounters.cpuFallback.Load(),
		Q4PointerTable:         ggufExpertDispatchCounters.q4PointerTable.Load(),
		Q4RawPointerTable:      ggufExpertDispatchCounters.q4RawPointerTable.Load(),
		Q4PackedCache:          ggufExpertDispatchCounters.q4PackedCache.Load(),
		Q4TransientPacked:      ggufExpertDispatchCounters.q4TransientPacked.Load(),
		Q4TransientPointer:     ggufExpertDispatchCounters.q4TransientPointer.Load(),
		Q4BudgetFallback:       ggufExpertDispatchCounters.q4BudgetFallback.Load(),
		Q4BudgetBytes:          ggufExpertDispatchCounters.q4BudgetBytes.Load(),
		Q4BudgetExperts:        ggufExpertDispatchCounters.q4BudgetExperts.Load(),
		Q8PointerTable:         ggufExpertDispatchCounters.q8PointerTable.Load(),
		Q8PackedCache:          ggufExpertDispatchCounters.q8PackedCache.Load(),
		Q8TransientPacked:      ggufExpertDispatchCounters.q8TransientPacked.Load(),
		Q8TransientPointer:     ggufExpertDispatchCounters.q8TransientPointer.Load(),
		Q8BudgetFallback:       ggufExpertDispatchCounters.q8BudgetFallback.Load(),
		Q8BudgetBytes:          ggufExpertDispatchCounters.q8BudgetBytes.Load(),
		Q8BudgetExperts:        ggufExpertDispatchCounters.q8BudgetExperts.Load(),
		Q5PointerTable:         ggufExpertDispatchCounters.q5PointerTable.Load(),
		Q5BudgetFallback:       ggufExpertDispatchCounters.q5BudgetFallback.Load(),
		Q5BudgetBytes:          ggufExpertDispatchCounters.q5BudgetBytes.Load(),
		Q5BudgetExperts:        ggufExpertDispatchCounters.q5BudgetExperts.Load(),
		ActiveSetCalls:         ggufExpertDispatchCounters.activeSetCalls.Load(),
		ActiveSetExperts:       ggufExpertDispatchCounters.activeSetExperts.Load(),
		ActiveSetMaxExperts:    ggufExpertDispatchCounters.activeSetMaxExperts.Load(),
		ActiveSetWorkItems:     ggufExpertDispatchCounters.activeSetWorkItems.Load(),
		ActiveSetMaxWorkItems:  ggufExpertDispatchCounters.activeSetMaxWorkItems.Load(),
		Q4MissingExperts:       ggufExpertDispatchCounters.q4MissingExperts.Load(),
		Q4MissingMaxExperts:    ggufExpertDispatchCounters.q4MissingMaxExperts.Load(),
		Q4MissingBytes:         ggufExpertDispatchCounters.q4MissingBytes.Load(),
		Q4MissingMaxBytes:      ggufExpertDispatchCounters.q4MissingMaxBytes.Load(),
		Q4MissingBudgetExceeds: ggufExpertDispatchCounters.q4MissingBudgetExceeds.Load(),
		PartialCalls:           ggufExpertDispatchCounters.partialCalls.Load(),
		PartialKeptExperts:     ggufExpertDispatchCounters.partialKeptExperts.Load(),
		PartialDroppedExperts:  ggufExpertDispatchCounters.partialDroppedExperts.Load(),
		PartialKeptWork:        ggufExpertDispatchCounters.partialKeptWork.Load(),
		PartialDroppedWork:     ggufExpertDispatchCounters.partialDroppedWork.Load(),
		GPUAttemptNS:           ggufExpertDispatchCounters.gpuAttemptNS.Load(),
		CPUFallbackNS:          ggufExpertDispatchCounters.cpuFallbackNS.Load(),
	}
}

func ggufMaxSince(current, base uint64) uint64 {
	if current > base {
		return current
	}
	return 0
}

func (s ggufExpertDispatchStats) Sub(base ggufExpertDispatchStats) ggufExpertDispatchStats {
	return ggufExpertDispatchStats{
		FusedUsed:              s.FusedUsed - base.FusedUsed,
		LegacyGroupedUsed:      s.LegacyGroupedUsed - base.LegacyGroupedUsed,
		CPUFallback:            s.CPUFallback - base.CPUFallback,
		Q4PointerTable:         s.Q4PointerTable - base.Q4PointerTable,
		Q4RawPointerTable:      s.Q4RawPointerTable - base.Q4RawPointerTable,
		Q4PackedCache:          s.Q4PackedCache - base.Q4PackedCache,
		Q4TransientPacked:      s.Q4TransientPacked - base.Q4TransientPacked,
		Q4TransientPointer:     s.Q4TransientPointer - base.Q4TransientPointer,
		Q4BudgetFallback:       s.Q4BudgetFallback - base.Q4BudgetFallback,
		Q4BudgetBytes:          s.Q4BudgetBytes - base.Q4BudgetBytes,
		Q4BudgetExperts:        s.Q4BudgetExperts - base.Q4BudgetExperts,
		Q8PointerTable:         s.Q8PointerTable - base.Q8PointerTable,
		Q8PackedCache:          s.Q8PackedCache - base.Q8PackedCache,
		Q8TransientPacked:      s.Q8TransientPacked - base.Q8TransientPacked,
		Q8TransientPointer:     s.Q8TransientPointer - base.Q8TransientPointer,
		Q8BudgetFallback:       s.Q8BudgetFallback - base.Q8BudgetFallback,
		Q8BudgetBytes:          s.Q8BudgetBytes - base.Q8BudgetBytes,
		Q8BudgetExperts:        s.Q8BudgetExperts - base.Q8BudgetExperts,
		Q5PointerTable:         s.Q5PointerTable - base.Q5PointerTable,
		Q5BudgetFallback:       s.Q5BudgetFallback - base.Q5BudgetFallback,
		Q5BudgetBytes:          s.Q5BudgetBytes - base.Q5BudgetBytes,
		Q5BudgetExperts:        s.Q5BudgetExperts - base.Q5BudgetExperts,
		ActiveSetCalls:         s.ActiveSetCalls - base.ActiveSetCalls,
		ActiveSetExperts:       s.ActiveSetExperts - base.ActiveSetExperts,
		ActiveSetMaxExperts:    ggufMaxSince(s.ActiveSetMaxExperts, base.ActiveSetMaxExperts),
		ActiveSetWorkItems:     s.ActiveSetWorkItems - base.ActiveSetWorkItems,
		ActiveSetMaxWorkItems:  ggufMaxSince(s.ActiveSetMaxWorkItems, base.ActiveSetMaxWorkItems),
		Q4MissingExperts:       s.Q4MissingExperts - base.Q4MissingExperts,
		Q4MissingMaxExperts:    ggufMaxSince(s.Q4MissingMaxExperts, base.Q4MissingMaxExperts),
		Q4MissingBytes:         s.Q4MissingBytes - base.Q4MissingBytes,
		Q4MissingMaxBytes:      ggufMaxSince(s.Q4MissingMaxBytes, base.Q4MissingMaxBytes),
		Q4MissingBudgetExceeds: s.Q4MissingBudgetExceeds - base.Q4MissingBudgetExceeds,
		PartialCalls:           s.PartialCalls - base.PartialCalls,
		PartialKeptExperts:     s.PartialKeptExperts - base.PartialKeptExperts,
		PartialDroppedExperts:  s.PartialDroppedExperts - base.PartialDroppedExperts,
		PartialKeptWork:        s.PartialKeptWork - base.PartialKeptWork,
		PartialDroppedWork:     s.PartialDroppedWork - base.PartialDroppedWork,
		GPUAttemptNS:           s.GPUAttemptNS - base.GPUAttemptNS,
		CPUFallbackNS:          s.CPUFallbackNS - base.CPUFallbackNS,
	}
}

func (s ggufExpertDispatchStats) Total() uint64 {
	return s.FusedUsed + s.LegacyGroupedUsed + s.CPUFallback
}

func (s ggufExpertDispatchStats) ActiveSetSummary() (activeAvg float64, workAvg float64, missingAvg float64, missingMiB float64, missingMaxMiB float64) {
	if s.ActiveSetCalls > 0 {
		activeAvg = float64(s.ActiveSetExperts) / float64(s.ActiveSetCalls)
		workAvg = float64(s.ActiveSetWorkItems) / float64(s.ActiveSetCalls)
		missingAvg = float64(s.Q4MissingExperts) / float64(s.ActiveSetCalls)
	}
	missingMiB = float64(s.Q4MissingBytes) / (1024 * 1024)
	missingMaxMiB = float64(s.Q4MissingMaxBytes) / (1024 * 1024)
	return activeAvg, workAvg, missingAvg, missingMiB, missingMaxMiB
}

func ResetGGUFGPUDiagnosticStats() {
	ggufExpertDispatchCounters.fusedUsed.Store(0)
	ggufExpertDispatchCounters.legacyGroupedUsed.Store(0)
	ggufExpertDispatchCounters.cpuFallback.Store(0)
	ggufExpertDispatchCounters.q4PointerTable.Store(0)
	ggufExpertDispatchCounters.q4RawPointerTable.Store(0)
	ggufExpertDispatchCounters.q4PackedCache.Store(0)
	ggufExpertDispatchCounters.q4TransientPacked.Store(0)
	ggufExpertDispatchCounters.q4TransientPointer.Store(0)
	ggufExpertDispatchCounters.q4BudgetFallback.Store(0)
	ggufExpertDispatchCounters.q4BudgetBytes.Store(0)
	ggufExpertDispatchCounters.q4BudgetExperts.Store(0)
	ggufExpertDispatchCounters.q8PointerTable.Store(0)
	ggufExpertDispatchCounters.q8PackedCache.Store(0)
	ggufExpertDispatchCounters.q8TransientPacked.Store(0)
	ggufExpertDispatchCounters.q8TransientPointer.Store(0)
	ggufExpertDispatchCounters.q8BudgetFallback.Store(0)
	ggufExpertDispatchCounters.q8BudgetBytes.Store(0)
	ggufExpertDispatchCounters.q8BudgetExperts.Store(0)
	ggufExpertDispatchCounters.q5PointerTable.Store(0)
	ggufExpertDispatchCounters.q5BudgetFallback.Store(0)
	ggufExpertDispatchCounters.q5BudgetBytes.Store(0)
	ggufExpertDispatchCounters.q5BudgetExperts.Store(0)
	ggufExpertDispatchCounters.activeSetCalls.Store(0)
	ggufExpertDispatchCounters.activeSetExperts.Store(0)
	ggufExpertDispatchCounters.activeSetMaxExperts.Store(0)
	ggufExpertDispatchCounters.activeSetWorkItems.Store(0)
	ggufExpertDispatchCounters.activeSetMaxWorkItems.Store(0)
	ggufExpertDispatchCounters.q4MissingExperts.Store(0)
	ggufExpertDispatchCounters.q4MissingMaxExperts.Store(0)
	ggufExpertDispatchCounters.q4MissingBytes.Store(0)
	ggufExpertDispatchCounters.q4MissingMaxBytes.Store(0)
	ggufExpertDispatchCounters.q4MissingBudgetExceeds.Store(0)
	ggufExpertDispatchCounters.partialCalls.Store(0)
	ggufExpertDispatchCounters.partialKeptExperts.Store(0)
	ggufExpertDispatchCounters.partialDroppedExperts.Store(0)
	ggufExpertDispatchCounters.partialKeptWork.Store(0)
	ggufExpertDispatchCounters.partialDroppedWork.Store(0)
	ggufExpertDispatchCounters.gpuAttemptNS.Store(0)
	ggufExpertDispatchCounters.cpuFallbackNS.Store(0)
	resetF32GELUExactMulStats()

	ggufChunkedLMHeadCounters.calls.Store(0)
	ggufChunkedLMHeadCounters.chunks.Store(0)
	ggufChunkedLMHeadCounters.bytes.Store(0)
	ggufChunkedLMHeadCounters.prepareNS.Store(0)
	ggufChunkedLMHeadCounters.uploadNS.Store(0)
	ggufChunkedLMHeadCounters.sgemmNS.Store(0)
	ggufChunkedLMHeadCounters.downloadNS.Store(0)
	ggufChunkedLMHeadCounters.copyNS.Store(0)

	ggufTempDenseUploadCounters.calls.Store(0)
	ggufTempDenseUploadCounters.bytes.Store(0)
	ggufTempDenseUploadCounters.transposeNS.Store(0)
	ggufTempDenseUploadCounters.uploadNS.Store(0)
	ggufTempDenseUploadCounters.cacheHits.Store(0)
	ggufTempDenseUploadCounters.cacheMisses.Store(0)
	ggufTempDenseUploadCounters.forwardAttnCalls.Store(0)
	ggufTempDenseUploadCounters.forwardAttnHits.Store(0)
	ggufTempDenseUploadCounters.forwardMLPCalls.Store(0)
	ggufTempDenseUploadCounters.forwardMLPHits.Store(0)
	ggufTempDenseUploadCounters.encoderAttnCalls.Store(0)
	ggufTempDenseUploadCounters.encoderAttnHits.Store(0)
	ggufTempDenseUploadCounters.encoderMLPCalls.Store(0)
	ggufTempDenseUploadCounters.encoderMLPHits.Store(0)

	ggufAttentionTimingCounters.calls.Store(0)
	ggufAttentionTimingCounters.totalNS.Store(0)
	ggufAttentionTimingCounters.projNS.Store(0)
	ggufAttentionTimingCounters.normRoPENS.Store(0)
	ggufAttentionTimingCounters.kvBuildNS.Store(0)
	ggufAttentionTimingCounters.attnNS.Store(0)
	ggufAttentionTimingCounters.oProjNS.Store(0)
}

type ggufChunkedLMHeadStats struct {
	Calls      uint64
	Chunks     uint64
	Bytes      uint64
	PrepareNS  uint64
	UploadNS   uint64
	SgemmNS    uint64
	DownloadNS uint64
	CopyNS     uint64
}

var ggufChunkedLMHeadCounters struct {
	calls      atomic.Uint64
	chunks     atomic.Uint64
	bytes      atomic.Uint64
	prepareNS  atomic.Uint64
	uploadNS   atomic.Uint64
	sgemmNS    atomic.Uint64
	downloadNS atomic.Uint64
	copyNS     atomic.Uint64
}

func ggufChunkedLMHeadSnapshot() ggufChunkedLMHeadStats {
	return ggufChunkedLMHeadStats{
		Calls:      ggufChunkedLMHeadCounters.calls.Load(),
		Chunks:     ggufChunkedLMHeadCounters.chunks.Load(),
		Bytes:      ggufChunkedLMHeadCounters.bytes.Load(),
		PrepareNS:  ggufChunkedLMHeadCounters.prepareNS.Load(),
		UploadNS:   ggufChunkedLMHeadCounters.uploadNS.Load(),
		SgemmNS:    ggufChunkedLMHeadCounters.sgemmNS.Load(),
		DownloadNS: ggufChunkedLMHeadCounters.downloadNS.Load(),
		CopyNS:     ggufChunkedLMHeadCounters.copyNS.Load(),
	}
}

func (s ggufChunkedLMHeadStats) Sub(base ggufChunkedLMHeadStats) ggufChunkedLMHeadStats {
	return ggufChunkedLMHeadStats{
		Calls:      s.Calls - base.Calls,
		Chunks:     s.Chunks - base.Chunks,
		Bytes:      s.Bytes - base.Bytes,
		PrepareNS:  s.PrepareNS - base.PrepareNS,
		UploadNS:   s.UploadNS - base.UploadNS,
		SgemmNS:    s.SgemmNS - base.SgemmNS,
		DownloadNS: s.DownloadNS - base.DownloadNS,
		CopyNS:     s.CopyNS - base.CopyNS,
	}
}

type ggufTempDenseUploadStats struct {
	Calls            uint64
	Bytes            uint64
	TransposeNS      uint64
	UploadNS         uint64
	CacheHits        uint64
	CacheMisses      uint64
	ForwardAttnCalls uint64
	ForwardAttnHits  uint64
	ForwardMLPCalls  uint64
	ForwardMLPHits   uint64
	EncoderAttnCalls uint64
	EncoderAttnHits  uint64
	EncoderMLPCalls  uint64
	EncoderMLPHits   uint64
}

var ggufTempDenseUploadCounters struct {
	calls            atomic.Uint64
	bytes            atomic.Uint64
	transposeNS      atomic.Uint64
	uploadNS         atomic.Uint64
	cacheHits        atomic.Uint64
	cacheMisses      atomic.Uint64
	forwardAttnCalls atomic.Uint64
	forwardAttnHits  atomic.Uint64
	forwardMLPCalls  atomic.Uint64
	forwardMLPHits   atomic.Uint64
	encoderAttnCalls atomic.Uint64
	encoderAttnHits  atomic.Uint64
	encoderMLPCalls  atomic.Uint64
	encoderMLPHits   atomic.Uint64
}

func ggufTempDenseUploadSnapshot() ggufTempDenseUploadStats {
	return ggufTempDenseUploadStats{
		Calls:            ggufTempDenseUploadCounters.calls.Load(),
		Bytes:            ggufTempDenseUploadCounters.bytes.Load(),
		TransposeNS:      ggufTempDenseUploadCounters.transposeNS.Load(),
		UploadNS:         ggufTempDenseUploadCounters.uploadNS.Load(),
		CacheHits:        ggufTempDenseUploadCounters.cacheHits.Load(),
		CacheMisses:      ggufTempDenseUploadCounters.cacheMisses.Load(),
		ForwardAttnCalls: ggufTempDenseUploadCounters.forwardAttnCalls.Load(),
		ForwardAttnHits:  ggufTempDenseUploadCounters.forwardAttnHits.Load(),
		ForwardMLPCalls:  ggufTempDenseUploadCounters.forwardMLPCalls.Load(),
		ForwardMLPHits:   ggufTempDenseUploadCounters.forwardMLPHits.Load(),
		EncoderAttnCalls: ggufTempDenseUploadCounters.encoderAttnCalls.Load(),
		EncoderAttnHits:  ggufTempDenseUploadCounters.encoderAttnHits.Load(),
		EncoderMLPCalls:  ggufTempDenseUploadCounters.encoderMLPCalls.Load(),
		EncoderMLPHits:   ggufTempDenseUploadCounters.encoderMLPHits.Load(),
	}
}

func (s ggufTempDenseUploadStats) Sub(base ggufTempDenseUploadStats) ggufTempDenseUploadStats {
	return ggufTempDenseUploadStats{
		Calls:            s.Calls - base.Calls,
		Bytes:            s.Bytes - base.Bytes,
		TransposeNS:      s.TransposeNS - base.TransposeNS,
		UploadNS:         s.UploadNS - base.UploadNS,
		CacheHits:        s.CacheHits - base.CacheHits,
		CacheMisses:      s.CacheMisses - base.CacheMisses,
		ForwardAttnCalls: s.ForwardAttnCalls - base.ForwardAttnCalls,
		ForwardAttnHits:  s.ForwardAttnHits - base.ForwardAttnHits,
		ForwardMLPCalls:  s.ForwardMLPCalls - base.ForwardMLPCalls,
		ForwardMLPHits:   s.ForwardMLPHits - base.ForwardMLPHits,
		EncoderAttnCalls: s.EncoderAttnCalls - base.EncoderAttnCalls,
		EncoderAttnHits:  s.EncoderAttnHits - base.EncoderAttnHits,
		EncoderMLPCalls:  s.EncoderMLPCalls - base.EncoderMLPCalls,
		EncoderMLPHits:   s.EncoderMLPHits - base.EncoderMLPHits,
	}
}

type ggufAttentionTimingStats struct {
	Calls      uint64
	TotalNS    uint64
	ProjNS     uint64
	NormRoPENS uint64
	KVBuildNS  uint64
	AttnNS     uint64
	OProjNS    uint64
}

var ggufAttentionTimingCounters struct {
	calls      atomic.Uint64
	totalNS    atomic.Uint64
	projNS     atomic.Uint64
	normRoPENS atomic.Uint64
	kvBuildNS  atomic.Uint64
	attnNS     atomic.Uint64
	oProjNS    atomic.Uint64
}

func ggufAttentionTimingSnapshot() ggufAttentionTimingStats {
	return ggufAttentionTimingStats{
		Calls:      ggufAttentionTimingCounters.calls.Load(),
		TotalNS:    ggufAttentionTimingCounters.totalNS.Load(),
		ProjNS:     ggufAttentionTimingCounters.projNS.Load(),
		NormRoPENS: ggufAttentionTimingCounters.normRoPENS.Load(),
		KVBuildNS:  ggufAttentionTimingCounters.kvBuildNS.Load(),
		AttnNS:     ggufAttentionTimingCounters.attnNS.Load(),
		OProjNS:    ggufAttentionTimingCounters.oProjNS.Load(),
	}
}

func (s ggufAttentionTimingStats) Sub(base ggufAttentionTimingStats) ggufAttentionTimingStats {
	return ggufAttentionTimingStats{
		Calls:      s.Calls - base.Calls,
		TotalNS:    s.TotalNS - base.TotalNS,
		ProjNS:     s.ProjNS - base.ProjNS,
		NormRoPENS: s.NormRoPENS - base.NormRoPENS,
		KVBuildNS:  s.KVBuildNS - base.KVBuildNS,
		AttnNS:     s.AttnNS - base.AttnNS,
		OProjNS:    s.OProjNS - base.OProjNS,
	}
}

func recordGGUFAttentionTiming(total, proj, normRoPE, kvBuild, attn, oProj time.Duration) {
	ggufAttentionTimingCounters.calls.Add(1)
	ggufAttentionTimingCounters.totalNS.Add(uint64(total.Nanoseconds()))
	ggufAttentionTimingCounters.projNS.Add(uint64(proj.Nanoseconds()))
	ggufAttentionTimingCounters.normRoPENS.Add(uint64(normRoPE.Nanoseconds()))
	ggufAttentionTimingCounters.kvBuildNS.Add(uint64(kvBuild.Nanoseconds()))
	ggufAttentionTimingCounters.attnNS.Add(uint64(attn.Nanoseconds()))
	ggufAttentionTimingCounters.oProjNS.Add(uint64(oProj.Nanoseconds()))
}

func ggufAtomicMaxUint64(dst *atomic.Uint64, v uint64) {
	for {
		cur := dst.Load()
		if v <= cur || dst.CompareAndSwap(cur, v) {
			return
		}
	}
}

func recordGGUFActiveExpertSetTelemetry(idx *GGUFExpertIndex, layer int, active []int, workItems int, missingExperts int, missingBytes int64, budgetExceeds bool) {
	if len(active) == 0 {
		return
	}
	ggufExpertDispatchCounters.activeSetCalls.Add(1)
	ggufExpertDispatchCounters.activeSetExperts.Add(uint64(len(active)))
	ggufAtomicMaxUint64(&ggufExpertDispatchCounters.activeSetMaxExperts, uint64(len(active)))
	if workItems > 0 {
		ggufExpertDispatchCounters.activeSetWorkItems.Add(uint64(workItems))
		ggufAtomicMaxUint64(&ggufExpertDispatchCounters.activeSetMaxWorkItems, uint64(workItems))
	}
	if missingExperts > 0 {
		ggufExpertDispatchCounters.q4MissingExperts.Add(uint64(missingExperts))
		ggufAtomicMaxUint64(&ggufExpertDispatchCounters.q4MissingMaxExperts, uint64(missingExperts))
	}
	if missingBytes > 0 {
		ggufExpertDispatchCounters.q4MissingBytes.Add(uint64(missingBytes))
		ggufAtomicMaxUint64(&ggufExpertDispatchCounters.q4MissingMaxBytes, uint64(missingBytes))
	}
	if budgetExceeds {
		ggufExpertDispatchCounters.q4MissingBudgetExceeds.Add(1)
	}
}

func recordQ4BudgetFallback(idx *GGUFExpertIndex, layer int, active []int) {
	ggufExpertDispatchCounters.q4BudgetFallback.Add(1)
	if b, err := q4KGateUpExpertResidentBytes(idx, layer); err == nil && b > 0 {
		ggufExpertDispatchCounters.q4BudgetBytes.Add(uint64(b) * uint64(len(active)))
		ggufExpertDispatchCounters.q4BudgetExperts.Add(uint64(len(active)))
	}
}

func recordQ8BudgetFallback(idx *GGUFExpertIndex, layer int, active []int) {
	ggufExpertDispatchCounters.q8BudgetFallback.Add(1)
	if b, err := q8DownExpertDeviceBytes(idx, layer); err == nil && b > 0 {
		ggufExpertDispatchCounters.q8BudgetBytes.Add(uint64(b) * uint64(len(active)))
		ggufExpertDispatchCounters.q8BudgetExperts.Add(uint64(len(active)))
	}
}

var activeExpertMatrixCacheBudget = struct {
	sync.Mutex
	bytes int64
}{}

func reserveActiveExpertMatrixCacheBytes(bytes int64) bool {
	budget := diffusionGemmaGGUFGPUExpertCacheBytes()
	if budget <= 0 || bytes <= 0 {
		return false
	}
	activeExpertMatrixCacheBudget.Lock()
	defer activeExpertMatrixCacheBudget.Unlock()
	if activeExpertMatrixCacheBudget.bytes+bytes > budget {
		return false
	}
	activeExpertMatrixCacheBudget.bytes += bytes
	return true
}

func releaseActiveExpertMatrixCacheBytes(bytes int64) {
	if bytes <= 0 {
		return
	}
	activeExpertMatrixCacheBudget.Lock()
	defer activeExpertMatrixCacheBudget.Unlock()
	activeExpertMatrixCacheBudget.bytes -= bytes
	if activeExpertMatrixCacheBudget.bytes < 0 {
		activeExpertMatrixCacheBudget.bytes = 0
	}
}

func activeExpertMatrixCacheUsageBytes() (used, limit int64) {
	activeExpertMatrixCacheBudget.Lock()
	used = activeExpertMatrixCacheBudget.bytes
	activeExpertMatrixCacheBudget.Unlock()
	return used, diffusionGemmaGGUFGPUExpertCacheBytes()
}

func ggufExpertIndexCacheID(idx *GGUFExpertIndex) uintptr {
	return uintptr(unsafe.Pointer(idx))
}

type activeQ8DownKey struct {
	index uintptr
	layer int
	hash  uint64
	count int
}

var activeQ8DownCache sync.Map // map[activeQ8DownKey]*gpu.GPUQ8_0Matrix

type q8DownExpertKey struct {
	index         uintptr
	layer, expert int
}

var q8DownExpertCache sync.Map             // map[q8DownExpertKey]*gpu.GPUQ8_0Matrix
var activeQ8DownPointerTableCache sync.Map // map[activeQ8DownKey]*gpu.GPUQ8_0PointerTable

type activeQ5DownKey struct {
	index uintptr
	layer int
	hash  uint64
	count int
}

type q5DownExpertKey struct {
	index         uintptr
	layer, expert int
}

var q5DownExpertCache sync.Map             // map[q5DownExpertKey]*gpu.GPUQ5_0Matrix
var activeQ5DownPointerTableCache sync.Map // map[activeQ5DownKey]*gpu.GPUQ5_0PointerTable

var ggufTransientPointerExpertScratch = struct {
	q4 []*gpu.GPUQ4KMatrix
	q8 []*gpu.GPUQ8_0Matrix
	q5 []*gpu.GPUQ5_0Matrix
}{}

func residentQ8DownExpertMatrix(idx *GGUFExpertIndex, layer, expert int) (*gpu.GPUQ8_0Matrix, error) {
	return residentQ8DownExpertMatrixWithReservation(idx, layer, expert, true)
}

func residentQ8DownExpertMatrixWithReservation(idx *GGUFExpertIndex, layer, expert int, reserve bool) (*gpu.GPUQ8_0Matrix, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || expert < 0 || expert >= idx.NumExperts {
		return nil, fmt.Errorf("invalid resident Q8 down expert layer=%d expert=%d", layer, expert)
	}
	key := q8DownExpertKey{index: ggufExpertIndexCacheID(idx), layer: layer, expert: expert}
	if cached, ok := q8DownExpertCache.Load(key); ok {
		return cached.(*gpu.GPUQ8_0Matrix), nil
	}
	le := idx.entries[layer]
	if le.down.QType != gguf.QuantQ8_0 {
		return nil, fmt.Errorf("resident Q8 down requires Q8_0, got %s", le.down.QType)
	}
	rowBytes, err := le.down.RowBytes()
	if err != nil {
		return nil, err
	}
	rows := le.down.OutDim
	start := expert * rows * rowBytes
	end := start + rows*rowBytes
	if start < 0 || end < start || end > len(le.down.Raw) {
		return nil, fmt.Errorf("resident Q8 down expert %d raw outside", expert)
	}
	blocks := le.down.InDim / 32
	cacheBytes := int64(rows * (le.down.InDim + blocks*4))
	if reserve && !reserveActiveExpertMatrixCacheBytes(cacheBytes) {
		return nil, errActiveExpertMatrixCacheBudget
	}
	m, err := gpu.UploadQ8_0MatrixRows(le.down.Raw[start:end], le.down.InDim, rows)
	if err != nil {
		if reserve {
			releaseActiveExpertMatrixCacheBytes(cacheBytes)
		}
		return nil, err
	}
	actual, loaded := q8DownExpertCache.LoadOrStore(key, m)
	if loaded {
		m.Free()
		if reserve {
			releaseActiveExpertMatrixCacheBytes(cacheBytes)
		}
		return actual.(*gpu.GPUQ8_0Matrix), nil
	}
	return m, nil
}

func activeQ8DownPointerTable(idx *GGUFExpertIndex, layer int, active []int) (*gpu.GPUQ8_0PointerTable, bool, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || len(active) == 0 {
		return nil, false, fmt.Errorf("invalid active Q8 pointer table request")
	}
	var h uint64 = 1469598103934665603
	for _, expert := range active {
		h ^= uint64(uint32(expert) + 0x9e3779b9)
		h *= 1099511628211
	}
	key := activeQ8DownKey{index: ggufExpertIndexCacheID(idx), layer: layer, hash: h, count: len(active)}
	if cached, ok := activeQ8DownPointerTableCache.Load(key); ok {
		return cached.(*gpu.GPUQ8_0PointerTable), true, nil
	}
	missingBytes := int64(0)
	for _, expert := range active {
		if expert < 0 || expert >= idx.NumExperts {
			return nil, false, fmt.Errorf("active Q8 expert %d outside %d", expert, idx.NumExperts)
		}
		if _, ok := q8DownExpertCache.Load(q8DownExpertKey{index: ggufExpertIndexCacheID(idx), layer: layer, expert: expert}); !ok {
			b, err := q8DownExpertDeviceBytes(idx, layer)
			if err != nil {
				return nil, false, err
			}
			missingBytes += b
		}
	}
	if missingBytes > 0 && !reserveActiveExpertMatrixCacheBytes(missingBytes) {
		return nil, false, nil
	}
	mats := make([]*gpu.GPUQ8_0Matrix, len(active))
	for i, expert := range active {
		m, err := residentQ8DownExpertMatrixWithReservation(idx, layer, expert, false)
		if err != nil {
			return nil, false, err
		}
		mats[i] = m
	}
	table, err := gpu.UploadQ8_0PointerTable(mats)
	if err != nil {
		return nil, false, err
	}
	actual, loaded := activeQ8DownPointerTableCache.LoadOrStore(key, table)
	if loaded {
		table.Free()
		return actual.(*gpu.GPUQ8_0PointerTable), true, nil
	}
	return table, true, nil
}

func q5DownExpertDeviceBytes(idx *GGUFExpertIndex, layer int) (int64, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers {
		return 0, fmt.Errorf("invalid Q5 down byte request layer=%d", layer)
	}
	le := idx.entries[layer]
	if le.down.QType != gguf.QuantQ5_0 {
		return 0, fmt.Errorf("Q5 down expert requires Q5_0, got %s", le.down.QType)
	}
	blocks := le.down.InDim / 32
	return int64(le.down.OutDim*(le.down.InDim/2) + blocks*le.down.OutDim*(4+4)), nil
}

func residentQ5DownExpertMatrix(idx *GGUFExpertIndex, layer, expert int) (*gpu.GPUQ5_0Matrix, error) {
	return residentQ5DownExpertMatrixWithReservation(idx, layer, expert, true)
}

func residentQ5DownExpertMatrixWithReservation(idx *GGUFExpertIndex, layer, expert int, reserve bool) (*gpu.GPUQ5_0Matrix, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || expert < 0 || expert >= idx.NumExperts {
		return nil, fmt.Errorf("invalid resident Q5 down expert layer=%d expert=%d", layer, expert)
	}
	key := q5DownExpertKey{index: ggufExpertIndexCacheID(idx), layer: layer, expert: expert}
	if cached, ok := q5DownExpertCache.Load(key); ok {
		return cached.(*gpu.GPUQ5_0Matrix), nil
	}
	le := idx.entries[layer]
	if le.down.QType != gguf.QuantQ5_0 {
		return nil, fmt.Errorf("resident Q5 down requires Q5_0, got %s", le.down.QType)
	}
	rowBytes, err := le.down.RowBytes()
	if err != nil {
		return nil, err
	}
	rows := le.down.OutDim
	start := expert * rows * rowBytes
	end := start + rows*rowBytes
	if start < 0 || end < start || end > len(le.down.Raw) {
		return nil, fmt.Errorf("resident Q5 down expert %d raw outside", expert)
	}
	cacheBytes, err := q5DownExpertDeviceBytes(idx, layer)
	if err != nil {
		return nil, err
	}
	if reserve && !reserveActiveExpertMatrixCacheBytes(cacheBytes) {
		return nil, errActiveExpertMatrixCacheBudget
	}
	m, err := gpu.UploadQ5_0MatrixRows(le.down.Raw[start:end], le.down.InDim, rows)
	if err != nil {
		if reserve {
			releaseActiveExpertMatrixCacheBytes(cacheBytes)
		}
		return nil, err
	}
	actual, loaded := q5DownExpertCache.LoadOrStore(key, m)
	if loaded {
		m.Free()
		if reserve {
			releaseActiveExpertMatrixCacheBytes(cacheBytes)
		}
		return actual.(*gpu.GPUQ5_0Matrix), nil
	}
	return m, nil
}

func activeQ5DownPointerTable(idx *GGUFExpertIndex, layer int, active []int) (*gpu.GPUQ5_0PointerTable, bool, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || len(active) == 0 {
		return nil, false, fmt.Errorf("invalid active Q5 pointer table request")
	}
	var h uint64 = 1469598103934665603
	for _, expert := range active {
		h ^= uint64(uint32(expert) + 0x9e3779b9)
		h *= 1099511628211
	}
	key := activeQ5DownKey{index: ggufExpertIndexCacheID(idx), layer: layer, hash: h, count: len(active)}
	if cached, ok := activeQ5DownPointerTableCache.Load(key); ok {
		return cached.(*gpu.GPUQ5_0PointerTable), true, nil
	}
	mats := make([]*gpu.GPUQ5_0Matrix, len(active))
	for i, expert := range active {
		m, err := residentQ5DownExpertMatrix(idx, layer, expert)
		if err != nil {
			if errors.Is(err, errActiveExpertMatrixCacheBudget) {
				return nil, false, nil
			}
			return nil, false, err
		}
		mats[i] = m
	}
	table, err := gpu.UploadQ5_0PointerTable(mats)
	if err != nil {
		return nil, false, err
	}
	actual, loaded := activeQ5DownPointerTableCache.LoadOrStore(key, table)
	if loaded {
		table.Free()
		return actual.(*gpu.GPUQ5_0PointerTable), true, nil
	}
	return table, true, nil
}

func transientActiveQ5DownPointerTable(idx *GGUFExpertIndex, layer int, active []int) (*gpu.GPUQ5_0PointerTable, func(), error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || len(active) == 0 {
		return nil, nil, fmt.Errorf("invalid transient active Q5 pointer table request")
	}
	le := idx.entries[layer]
	if le.down.QType != gguf.QuantQ5_0 {
		return nil, nil, fmt.Errorf("transient active Q5 down requires Q5_0, got %s", le.down.QType)
	}
	rowBytes, err := le.down.RowBytes()
	if err != nil {
		return nil, nil, err
	}
	rows := le.down.OutDim
	for len(ggufTransientPointerExpertScratch.q5) < len(active) {
		ggufTransientPointerExpertScratch.q5 = append(ggufTransientPointerExpertScratch.q5, nil)
	}
	mats := make([]*gpu.GPUQ5_0Matrix, len(active))
	for i, expert := range active {
		start := expert * rows * rowBytes
		end := start + rows*rowBytes
		if start < 0 || end < start || end > len(le.down.Raw) {
			return nil, nil, fmt.Errorf("transient active Q5 expert %d raw outside", expert)
		}
		m := ggufTransientPointerExpertScratch.q5[i]
		if m != nil && (m.InDim != le.down.InDim || m.OutDim != rows) {
			m.Free()
			m = nil
			ggufTransientPointerExpertScratch.q5[i] = nil
		}
		if m == nil {
			m, err = gpu.UploadQ5_0MatrixRows(le.down.Raw[start:end], le.down.InDim, rows)
			if err != nil {
				return nil, nil, err
			}
			ggufTransientPointerExpertScratch.q5[i] = m
		} else if err := gpu.UploadQ5_0MatrixRowsInto(m, le.down.Raw[start:end], le.down.InDim, rows); err != nil {
			return nil, nil, err
		}
		mats[i] = m
	}
	table, err := gpu.UploadQ5_0PointerTable(mats)
	if err != nil {
		return nil, nil, err
	}
	cleanup := func() { table.Free() }
	return table, cleanup, nil
}

func transientQ8DownExpertMatrix(idx *GGUFExpertIndex, layer, expert int) (*gpu.GPUQ8_0Matrix, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || expert < 0 || expert >= idx.NumExperts {
		return nil, fmt.Errorf("invalid transient Q8 down expert layer=%d expert=%d", layer, expert)
	}
	le := idx.entries[layer]
	if le.down.QType != gguf.QuantQ8_0 {
		return nil, fmt.Errorf("transient Q8 down requires Q8_0, got %s", le.down.QType)
	}
	rowBytes, err := le.down.RowBytes()
	if err != nil {
		return nil, err
	}
	rows := le.down.OutDim
	start := expert * rows * rowBytes
	end := start + rows*rowBytes
	if start < 0 || end < start || end > len(le.down.Raw) {
		return nil, fmt.Errorf("transient Q8 down expert %d raw outside", expert)
	}
	return gpu.UploadQ8_0MatrixRows(le.down.Raw[start:end], le.down.InDim, rows)
}

func transientActiveQ8DownPointerTable(idx *GGUFExpertIndex, layer int, active []int) (*gpu.GPUQ8_0PointerTable, func(), error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || len(active) == 0 {
		return nil, nil, fmt.Errorf("invalid transient active Q8 pointer table request")
	}
	le := idx.entries[layer]
	rowBytes, err := le.down.RowBytes()
	if err != nil {
		return nil, nil, err
	}
	rows := le.down.OutDim
	for len(ggufTransientPointerExpertScratch.q8) < len(active) {
		ggufTransientPointerExpertScratch.q8 = append(ggufTransientPointerExpertScratch.q8, nil)
	}
	mats := make([]*gpu.GPUQ8_0Matrix, len(active))
	for i, expert := range active {
		start := expert * rows * rowBytes
		end := start + rows*rowBytes
		if start < 0 || end < start || end > len(le.down.Raw) {
			return nil, nil, fmt.Errorf("transient active Q8 expert %d raw outside", expert)
		}
		m := ggufTransientPointerExpertScratch.q8[i]
		if m != nil && (m.InDim != le.down.InDim || m.OutDim != rows) {
			m.Free()
			m = nil
			ggufTransientPointerExpertScratch.q8[i] = nil
		}
		if m == nil {
			m, err = gpu.UploadQ8_0MatrixRows(le.down.Raw[start:end], le.down.InDim, rows)
			if err != nil {
				return nil, nil, err
			}
			ggufTransientPointerExpertScratch.q8[i] = m
		} else if err := gpu.UploadQ8_0MatrixRowsInto(m, le.down.Raw[start:end], le.down.InDim, rows); err != nil {
			return nil, nil, err
		}
		mats[i] = m
	}
	table, err := gpu.UploadQ8_0PointerTable(mats)
	if err != nil {
		return nil, nil, err
	}
	cleanup := func() { table.Free() }
	return table, cleanup, nil
}

func activeQ8DownMatrix(idx *GGUFExpertIndex, layer int, active []int) (*gpu.GPUQ8_0Matrix, error) {
	return activeQ8DownMatrixWithCache(idx, layer, active, true)
}

func transientActiveQ8DownMatrix(idx *GGUFExpertIndex, layer int, active []int) (*gpu.GPUQ8_0Matrix, error) {
	return activeQ8DownMatrixWithCache(idx, layer, active, false)
}

func activeQ8DownMatrixWithCache(idx *GGUFExpertIndex, layer int, active []int, cache bool) (*gpu.GPUQ8_0Matrix, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || len(active) == 0 {
		return nil, fmt.Errorf("invalid active Q8 down request")
	}
	var h uint64 = 1469598103934665603
	for _, expert := range active {
		h ^= uint64(uint32(expert) + 0x9e3779b9)
		h *= 1099511628211
	}
	key := activeQ8DownKey{index: ggufExpertIndexCacheID(idx), layer: layer, hash: h, count: len(active)}
	if cache {
		if cached, ok := activeQ8DownCache.Load(key); ok {
			return cached.(*gpu.GPUQ8_0Matrix), nil
		}
	}
	le := idx.entries[layer]
	if le.down.QType != gguf.QuantQ8_0 {
		return nil, fmt.Errorf("active Q8 down requires Q8_0, got %s", le.down.QType)
	}
	rowBytes, err := le.down.RowBytes()
	if err != nil {
		return nil, err
	}
	rowsPerExpert := le.down.OutDim
	for _, expert := range active {
		if expert < 0 || expert >= idx.NumExperts {
			return nil, fmt.Errorf("active expert %d outside %d", expert, idx.NumExperts)
		}
		start := expert * rowsPerExpert * rowBytes
		end := start + rowsPerExpert*rowBytes
		if start < 0 || end < start || end > len(le.down.Raw) {
			return nil, fmt.Errorf("active expert %d down raw outside", expert)
		}
	}
	blocks := le.down.InDim / 32
	cacheBytes := int64(len(active) * rowsPerExpert * (le.down.InDim + blocks*4))
	if cache && !reserveActiveExpertMatrixCacheBytes(cacheBytes) {
		return nil, errActiveExpertMatrixCacheBudget
	}
	raw := make([]byte, len(active)*rowsPerExpert*rowBytes)
	for ai, expert := range active {
		start := expert * rowsPerExpert * rowBytes
		end := start + rowsPerExpert*rowBytes
		copy(raw[ai*rowsPerExpert*rowBytes:(ai+1)*rowsPerExpert*rowBytes], le.down.Raw[start:end])
	}
	m, err := gpu.UploadQ8_0MatrixRows(raw, le.down.InDim, rowsPerExpert*len(active))
	if err != nil {
		if cache {
			releaseActiveExpertMatrixCacheBytes(cacheBytes)
		}
		return nil, err
	}
	if !cache {
		return m, nil
	}
	actual, loaded := activeQ8DownCache.LoadOrStore(key, m)
	if loaded {
		m.Free()
		releaseActiveExpertMatrixCacheBytes(cacheBytes)
		return actual.(*gpu.GPUQ8_0Matrix), nil
	}
	return m, nil
}

type activeQ4KGateUpKey struct {
	index uintptr
	layer int
	hash  uint64
	count int
}

var activeQ4KGateUpCache sync.Map // map[activeQ4KGateUpKey]*gpu.GPUQ4KMatrix

type q4KGateUpExpertKey struct {
	index         uintptr
	layer, expert int
}

var q4KGateUpExpertCache sync.Map                // map[q4KGateUpExpertKey]*gpu.GPUQ4KMatrix
var q4KGateUpExpertRawCache sync.Map             // map[q4KGateUpExpertKey]*gpu.GPUQ4KMatrixRaw
var activeQ4KGateUpPointerTableCache sync.Map    // map[activeQ4KGateUpKey]*gpu.GPUQ4KPointerTable
var activeQ4KGateUpRawPointerTableCache sync.Map // map[activeQ4KGateUpKey]*gpu.GPUQ4KPointerTableRaw

func ggufPointerExpertLayerFullyResident(idx *GGUFExpertIndex, layer int) bool {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || idx.NumExperts <= 0 {
		return false
	}
	index := ggufExpertIndexCacheID(idx)
	for expert := 0; expert < idx.NumExperts; expert++ {
		if diffusionGemmaGGUFGPUExpertRawQ4Enabled() {
			if _, ok := q4KGateUpExpertRawCache.Load(q4KGateUpExpertKey{index: index, layer: layer, expert: expert}); !ok {
				return false
			}
		} else if _, ok := q4KGateUpExpertCache.Load(q4KGateUpExpertKey{index: index, layer: layer, expert: expert}); !ok {
			return false
		}
		if _, ok := q8DownExpertCache.Load(q8DownExpertKey{index: index, layer: layer, expert: expert}); !ok {
			return false
		}
	}
	return true
}

func shouldSkipDoomedGGUFGPUExpertAttempt(idx *GGUFExpertIndex, layer int) bool {
	if !diffusionGemmaGGUFGPUExpertSkipDoomedAttemptsEnabled() || diffusionGemmaGGUFGPUExpertTransientActiveEnabled() || diffusionGemmaGGUFGPUExpertTransientPointerEnabled() {
		return false
	}
	if ggufPointerExpertLayerFullyResident(idx, layer) {
		return false
	}
	used, limit := activeExpertMatrixCacheUsageBytes()
	return limit <= 0 || used >= limit
}

func traceGGUFActiveExpertSet(idx *GGUFExpertIndex, layer int, groupedArrays SelectedExpertGroupedArrays) {
	topN := diffusionGemmaGGUFGPUExpertActiveTraceTop()
	if topN <= 0 || idx == nil || len(groupedArrays.ActiveExperts) == 0 || len(groupedArrays.Offsets) != len(groupedArrays.ActiveExperts)+1 {
		return
	}
	perExpert, err := q4KGateUpExpertResidentBytes(idx, layer)
	if err != nil || perExpert <= 0 {
		return
	}
	index := ggufExpertIndexCacheID(idx)
	type expertTrace struct {
		id      int
		work    int
		missing bool
	}
	items := make([]expertTrace, 0, len(groupedArrays.ActiveExperts))
	missingExperts := 0
	missingBytes := int64(0)
	for i, expert := range groupedArrays.ActiveExperts {
		if expert < 0 || expert >= idx.NumExperts {
			continue
		}
		work := groupedArrays.Offsets[i+1] - groupedArrays.Offsets[i]
		missing := false
		if diffusionGemmaGGUFGPUExpertRawQ4Enabled() {
			if _, ok := q4KGateUpExpertRawCache.Load(q4KGateUpExpertKey{index: index, layer: layer, expert: expert}); !ok {
				missing = true
				missingExperts++
				missingBytes += perExpert
			}
		} else if _, ok := q4KGateUpExpertCache.Load(q4KGateUpExpertKey{index: index, layer: layer, expert: expert}); !ok {
			missing = true
			missingExperts++
			missingBytes += perExpert
		}
		items = append(items, expertTrace{id: expert, work: work, missing: missing})
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].work == items[j].work {
			return items[i].id < items[j].id
		}
		return items[i].work > items[j].work
	})
	if topN > len(items) {
		topN = len(items)
	}
	parts := make([]string, 0, topN)
	for _, item := range items[:topN] {
		flag := ""
		if item.missing {
			flag = "!"
		}
		parts = append(parts, fmt.Sprintf("%d:%d%s", item.id, item.work, flag))
	}
	log.Printf("gguf_expert_active_trace: layer=%d active=%d work=%d missing_q4=%d missing_q4_bytes=%.1fMiB top=%s", layer, len(groupedArrays.ActiveExperts), len(groupedArrays.WorkPositions), missingExperts, float64(missingBytes)/(1024*1024), strings.Join(parts, ","))
}

func shouldSkipDoomedGGUFActiveExpertSet(idx *GGUFExpertIndex, layer int, active []int, workItems int) bool {
	missingExperts, missingBytes, exceeds := ggufActiveExpertSetQ4MissingStats(idx, layer, active)
	recordGGUFActiveExpertSetTelemetry(idx, layer, active, workItems, missingExperts, missingBytes, exceeds)
	return !diffusionGemmaGGUFGPUExpertTransientActiveEnabled() && !diffusionGemmaGGUFGPUExpertTransientPointerEnabled() && exceeds
}

func ggufActiveExpertSetQ4MissingBytesExceedsBudget(idx *GGUFExpertIndex, layer int, active []int) bool {
	_, _, exceeds := ggufActiveExpertSetQ4MissingStats(idx, layer, active)
	return exceeds
}

func ggufActiveExpertSetQ4MissingStats(idx *GGUFExpertIndex, layer int, active []int) (missingExperts int, missingBytes int64, exceeds bool) {
	if idx == nil || len(active) == 0 {
		return 0, 0, false
	}
	if ggufPointerExpertLayerFullyResident(idx, layer) {
		return 0, 0, false
	}
	perExpert, err := q4KGateUpExpertResidentBytes(idx, layer)
	if err != nil || perExpert <= 0 {
		return 0, 0, false
	}
	index := ggufExpertIndexCacheID(idx)
	for _, expert := range active {
		if expert < 0 || expert >= idx.NumExperts {
			continue
		}
		if diffusionGemmaGGUFGPUExpertRawQ4Enabled() {
			if _, ok := q4KGateUpExpertRawCache.Load(q4KGateUpExpertKey{index: index, layer: layer, expert: expert}); !ok {
				missingExperts++
				missingBytes += perExpert
			}
		} else if _, ok := q4KGateUpExpertCache.Load(q4KGateUpExpertKey{index: index, layer: layer, expert: expert}); !ok {
			missingExperts++
			missingBytes += perExpert
		}
	}
	if missingBytes <= 0 {
		return missingExperts, missingBytes, false
	}
	used, limit := activeExpertMatrixCacheUsageBytes()
	return missingExperts, missingBytes, limit <= 0 || used+missingBytes > limit
}

func shouldEarlySkipDoomedGGUFExpertAttempt(idx *GGUFExpertIndex, layer int, topKIDs []int, positions, topK int) ([]int, bool) {
	if !diffusionGemmaGGUFGPUExpertSkipDoomedAttemptsEnabled() || diffusionGemmaGGUFGPUExpertTransientActiveEnabled() || diffusionGemmaGGUFGPUExpertTransientPointerEnabled() || idx == nil || positions <= 0 || topK <= 0 || len(topKIDs) < positions*topK {
		return nil, false
	}
	seen := make([]bool, idx.NumExperts)
	active := make([]int, 0, idx.NumExperts)
	for pos := 0; pos < positions; pos++ {
		for k := 0; k < topK; k++ {
			expert := topKIDs[pos*topK+k]
			if expert >= 0 && expert < idx.NumExperts && !seen[expert] {
				seen[expert] = true
				active = append(active, expert)
			}
		}
	}
	return active, ggufActiveExpertSetQ4MissingBytesExceedsBudget(idx, layer, active)
}

func residentQ4KGateUpExpertMatrix(idx *GGUFExpertIndex, layer, expert int) (*gpu.GPUQ4KMatrix, error) {
	return residentQ4KGateUpExpertMatrixWithReservation(idx, layer, expert, true)
}

func residentQ4KGateUpExpertRawMatrixWithReservation(idx *GGUFExpertIndex, layer, expert int, reserve bool) (*gpu.GPUQ4KMatrixRaw, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || expert < 0 || expert >= idx.NumExperts {
		return nil, fmt.Errorf("invalid resident raw Q4_K gate/up expert layer=%d expert=%d", layer, expert)
	}
	key := q4KGateUpExpertKey{index: ggufExpertIndexCacheID(idx), layer: layer, expert: expert}
	if cached, ok := q4KGateUpExpertRawCache.Load(key); ok {
		return cached.(*gpu.GPUQ4KMatrixRaw), nil
	}
	le := idx.entries[layer]
	if le.gateUp.QType != gguf.QuantQ4_K {
		return nil, fmt.Errorf("resident raw Q4_K gate/up requires Q4_K, got %s", le.gateUp.QType)
	}
	rowBytes, err := le.gateUp.RowBytes()
	if err != nil {
		return nil, err
	}
	rows := le.gateUp.OutDim
	start := expert * rows * rowBytes
	end := start + rows*rowBytes
	if start < 0 || end < start || end > len(le.gateUp.Raw) {
		return nil, fmt.Errorf("resident raw Q4_K gate/up expert %d raw outside", expert)
	}
	blocks := le.gateUp.InDim / 256
	cacheBytes := int64(rows * blocks * 144)
	if reserve && !reserveActiveExpertMatrixCacheBytes(cacheBytes) {
		return nil, errActiveExpertMatrixCacheBudget
	}
	m, err := gpu.UploadQ4KMatrixRowsRaw(le.gateUp.Raw[start:end], le.gateUp.InDim, rows)
	if err != nil {
		if reserve {
			releaseActiveExpertMatrixCacheBytes(cacheBytes)
		}
		return nil, err
	}
	actual, loaded := q4KGateUpExpertRawCache.LoadOrStore(key, m)
	if loaded {
		m.Free()
		if reserve {
			releaseActiveExpertMatrixCacheBytes(cacheBytes)
		}
		return actual.(*gpu.GPUQ4KMatrixRaw), nil
	}
	return m, nil
}

func activeQ4KGateUpRawPointerTable(idx *GGUFExpertIndex, layer int, active []int) (*gpu.GPUQ4KPointerTableRaw, bool, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || len(active) == 0 {
		return nil, false, fmt.Errorf("invalid active raw Q4_K pointer table request")
	}
	var h uint64 = 1469598103934665603
	for _, expert := range active {
		h ^= uint64(uint32(expert) + 0x9e3779b9)
		h *= 1099511628211
	}
	key := activeQ4KGateUpKey{index: ggufExpertIndexCacheID(idx), layer: layer, hash: h, count: len(active)}
	if cached, ok := activeQ4KGateUpRawPointerTableCache.Load(key); ok {
		return cached.(*gpu.GPUQ4KPointerTableRaw), true, nil
	}
	missingBytes := int64(0)
	blocks := idx.entries[layer].gateUp.InDim / 256
	rows := idx.entries[layer].gateUp.OutDim
	perExpert := int64(rows * blocks * 144)
	for _, expert := range active {
		if expert < 0 || expert >= idx.NumExperts {
			return nil, false, fmt.Errorf("active raw Q4_K expert %d outside %d", expert, idx.NumExperts)
		}
		if _, ok := q4KGateUpExpertRawCache.Load(q4KGateUpExpertKey{index: ggufExpertIndexCacheID(idx), layer: layer, expert: expert}); !ok {
			missingBytes += perExpert
		}
	}
	if missingBytes > 0 && !reserveActiveExpertMatrixCacheBytes(missingBytes) {
		return nil, false, nil
	}
	mats := make([]*gpu.GPUQ4KMatrixRaw, len(active))
	for i, expert := range active {
		m, err := residentQ4KGateUpExpertRawMatrixWithReservation(idx, layer, expert, false)
		if err != nil {
			return nil, false, err
		}
		mats[i] = m
	}
	table, err := gpu.UploadQ4KPointerTableRaw(mats)
	if err != nil {
		return nil, false, err
	}
	actual, loaded := activeQ4KGateUpRawPointerTableCache.LoadOrStore(key, table)
	if loaded {
		table.Free()
		return actual.(*gpu.GPUQ4KPointerTableRaw), true, nil
	}
	return table, true, nil
}

func residentQ4KGateUpExpertMatrixWithReservation(idx *GGUFExpertIndex, layer, expert int, reserve bool) (*gpu.GPUQ4KMatrix, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || expert < 0 || expert >= idx.NumExperts {
		return nil, fmt.Errorf("invalid resident Q4_K gate/up expert layer=%d expert=%d", layer, expert)
	}
	key := q4KGateUpExpertKey{index: ggufExpertIndexCacheID(idx), layer: layer, expert: expert}
	if cached, ok := q4KGateUpExpertCache.Load(key); ok {
		return cached.(*gpu.GPUQ4KMatrix), nil
	}
	le := idx.entries[layer]
	if le.gateUp.QType != gguf.QuantQ4_K {
		return nil, fmt.Errorf("resident Q4_K gate/up requires Q4_K, got %s", le.gateUp.QType)
	}
	rowBytes, err := le.gateUp.RowBytes()
	if err != nil {
		return nil, err
	}
	rows := le.gateUp.OutDim
	start := expert * rows * rowBytes
	end := start + rows*rowBytes
	if start < 0 || end < start || end > len(le.gateUp.Raw) {
		return nil, fmt.Errorf("resident Q4_K gate/up expert %d raw outside", expert)
	}
	blocks := le.gateUp.InDim / 256
	cacheBytes := int64(rows * blocks * (128 + 8*4 + 8*4))
	if reserve && !reserveActiveExpertMatrixCacheBytes(cacheBytes) {
		return nil, errActiveExpertMatrixCacheBudget
	}
	m, err := gpu.UploadQ4KMatrixRows(le.gateUp.Raw[start:end], le.gateUp.InDim, rows)
	if err != nil {
		if reserve {
			releaseActiveExpertMatrixCacheBytes(cacheBytes)
		}
		return nil, err
	}
	actual, loaded := q4KGateUpExpertCache.LoadOrStore(key, m)
	if loaded {
		m.Free()
		if reserve {
			releaseActiveExpertMatrixCacheBytes(cacheBytes)
		}
		return actual.(*gpu.GPUQ4KMatrix), nil
	}
	return m, nil
}

func activeQ4KGateUpPointerTable(idx *GGUFExpertIndex, layer int, active []int) (*gpu.GPUQ4KPointerTable, bool, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || len(active) == 0 {
		return nil, false, fmt.Errorf("invalid active Q4_K pointer table request")
	}
	var h uint64 = 1469598103934665603
	for _, expert := range active {
		h ^= uint64(uint32(expert) + 0x9e3779b9)
		h *= 1099511628211
	}
	key := activeQ4KGateUpKey{index: ggufExpertIndexCacheID(idx), layer: layer, hash: h, count: len(active)}
	if cached, ok := activeQ4KGateUpPointerTableCache.Load(key); ok {
		return cached.(*gpu.GPUQ4KPointerTable), true, nil
	}
	missingBytes := int64(0)
	for _, expert := range active {
		if expert < 0 || expert >= idx.NumExperts {
			return nil, false, fmt.Errorf("active Q4_K expert %d outside %d", expert, idx.NumExperts)
		}
		if _, ok := q4KGateUpExpertCache.Load(q4KGateUpExpertKey{index: ggufExpertIndexCacheID(idx), layer: layer, expert: expert}); !ok {
			b, err := q4KGateUpExpertResidentBytes(idx, layer)
			if err != nil {
				return nil, false, err
			}
			missingBytes += b
		}
	}
	if missingBytes > 0 && !reserveActiveExpertMatrixCacheBytes(missingBytes) {
		return nil, false, nil
	}
	mats := make([]*gpu.GPUQ4KMatrix, len(active))
	for i, expert := range active {
		m, err := residentQ4KGateUpExpertMatrixWithReservation(idx, layer, expert, false)
		if err != nil {
			return nil, false, err
		}
		mats[i] = m
	}
	table, err := gpu.UploadQ4KPointerTable(mats)
	if err != nil {
		return nil, false, err
	}
	actual, loaded := activeQ4KGateUpPointerTableCache.LoadOrStore(key, table)
	if loaded {
		table.Free()
		return actual.(*gpu.GPUQ4KPointerTable), true, nil
	}
	return table, true, nil
}

func transientQ4KGateUpExpertMatrix(idx *GGUFExpertIndex, layer, expert int) (*gpu.GPUQ4KMatrix, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || expert < 0 || expert >= idx.NumExperts {
		return nil, fmt.Errorf("invalid transient Q4_K gate/up expert layer=%d expert=%d", layer, expert)
	}
	le := idx.entries[layer]
	if le.gateUp.QType != gguf.QuantQ4_K {
		return nil, fmt.Errorf("transient Q4_K gate/up requires Q4_K, got %s", le.gateUp.QType)
	}
	rowBytes, err := le.gateUp.RowBytes()
	if err != nil {
		return nil, err
	}
	rows := le.gateUp.OutDim
	start := expert * rows * rowBytes
	end := start + rows*rowBytes
	if start < 0 || end < start || end > len(le.gateUp.Raw) {
		return nil, fmt.Errorf("transient Q4_K gate/up expert %d raw outside", expert)
	}
	return gpu.UploadQ4KMatrixRows(le.gateUp.Raw[start:end], le.gateUp.InDim, rows)
}

func transientActiveQ4KGateUpPointerTable(idx *GGUFExpertIndex, layer int, active []int) (*gpu.GPUQ4KPointerTable, func(), error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || len(active) == 0 {
		return nil, nil, fmt.Errorf("invalid transient active Q4_K pointer table request")
	}
	le := idx.entries[layer]
	rowBytes, err := le.gateUp.RowBytes()
	if err != nil {
		return nil, nil, err
	}
	rows := le.gateUp.OutDim
	for len(ggufTransientPointerExpertScratch.q4) < len(active) {
		ggufTransientPointerExpertScratch.q4 = append(ggufTransientPointerExpertScratch.q4, nil)
	}
	mats := make([]*gpu.GPUQ4KMatrix, len(active))
	for i, expert := range active {
		start := expert * rows * rowBytes
		end := start + rows*rowBytes
		if start < 0 || end < start || end > len(le.gateUp.Raw) {
			return nil, nil, fmt.Errorf("transient active Q4_K expert %d raw outside", expert)
		}
		m := ggufTransientPointerExpertScratch.q4[i]
		if m != nil && (m.InDim != le.gateUp.InDim || m.OutDim != rows) {
			m.Free()
			m = nil
			ggufTransientPointerExpertScratch.q4[i] = nil
		}
		if m == nil {
			m, err = gpu.UploadQ4KMatrixRows(le.gateUp.Raw[start:end], le.gateUp.InDim, rows)
			if err != nil {
				return nil, nil, err
			}
			ggufTransientPointerExpertScratch.q4[i] = m
		} else if err := gpu.UploadQ4KMatrixRowsInto(m, le.gateUp.Raw[start:end], le.gateUp.InDim, rows); err != nil {
			return nil, nil, err
		}
		mats[i] = m
	}
	table, err := gpu.UploadQ4KPointerTable(mats)
	if err != nil {
		return nil, nil, err
	}
	cleanup := func() { table.Free() }
	return table, cleanup, nil
}

func FreeGGUFGPURuntimeCaches() {
	FreeGGUFGPUExpertCaches()
	FreeGGUFGPUWeightCaches()
	FreeResidentF32MatrixCache()
	FreeGGUFGPUScratchBuffers()
	FreeGGUFTempDenseUploadScratch()
	FreeGGUFDenseTransposeCache()
	FreeGGUFChunkedLMHeadScratch()
	ResetGGUFGPUDiagnosticStats()
	ResetGGUFCPUExpertTimingStats()
}

func freeGPUBufferPtr(p **gpu.Buffer) {
	if p != nil && *p != nil {
		(*p).Free()
		*p = nil
	}
}

func FreeGGUFGPUScratchBuffers() {
	freeF32GELUExactMulScratch()
	selectedExpertWorkGPUBuffers.Lock()
	selectedExpertWorkGPUBuffers.bufs.Free()
	selectedExpertWorkGPUBuffers.groupedBufs.Free()
	selectedExpertWorkGPUBuffers.groupedArrays.Free()
	selectedExpertWorkGPUBuffers.items = nil
	selectedExpertWorkGPUBuffers.arrays = SelectedExpertWorkArrays{}
	selectedExpertWorkGPUBuffers.Unlock()

	ggufGPUExpertScratch.Lock()
	freeGPUBufferPtr(&ggufGPUExpertScratch.x)
	freeGPUBufferPtr(&ggufGPUExpertScratch.gu)
	freeGPUBufferPtr(&ggufGPUExpertScratch.act)
	freeGPUBufferPtr(&ggufGPUExpertScratch.up)
	freeGPUBufferPtr(&ggufGPUExpertScratch.down)
	freeGPUBufferPtr(&ggufGPUExpertScratch.pos)
	freeGPUBufferPtr(&ggufGPUExpertScratch.weights)
	ggufGPUExpertScratch.xN, ggufGPUExpertScratch.guN, ggufGPUExpertScratch.actN = 0, 0, 0
	ggufGPUExpertScratch.upN, ggufGPUExpertScratch.downN, ggufGPUExpertScratch.posN, ggufGPUExpertScratch.weightN = 0, 0, 0, 0
	ggufGPUExpertScratch.hostBatchIn = nil
	ggufGPUExpertScratch.hostPosIDs = nil
	ggufGPUExpertScratch.hostWeights = nil
	ggufGPUExpertScratch.Unlock()

	ggufGPUFusedExpertScratch.Lock()
	freeGPUBufferPtr(&ggufGPUFusedExpertScratch.residual)
	freeGPUBufferPtr(&ggufGPUFusedExpertScratch.workInput)
	freeGPUBufferPtr(&ggufGPUFusedExpertScratch.act)
	freeGPUBufferPtr(&ggufGPUFusedExpertScratch.up)
	freeGPUBufferPtr(&ggufGPUFusedExpertScratch.moeOut)
	ggufGPUFusedExpertScratch.residualN, ggufGPUFusedExpertScratch.workN = 0, 0
	ggufGPUFusedExpertScratch.actN, ggufGPUFusedExpertScratch.upN, ggufGPUFusedExpertScratch.moeOutN = 0, 0, 0
	ggufGPUFusedExpertScratch.Unlock()

	residentGemmScratch.Lock()
	freeGPUBufferPtr(&residentGemmScratch.x)
	freeGPUBufferPtr(&residentGemmScratch.y)
	residentGemmScratch.xLen, residentGemmScratch.yLen = 0, 0
	residentGemmScratch.Unlock()
}

func GGUFGPUScratchStats() (buffers int, bytes int64) {
	count := func(b *gpu.Buffer) {
		if b != nil {
			buffers++
			bytes += int64(b.Size)
		}
	}
	selectedExpertWorkGPUBuffers.Lock()
	count(selectedExpertWorkGPUBuffers.bufs.Positions)
	count(selectedExpertWorkGPUBuffers.bufs.Experts)
	count(selectedExpertWorkGPUBuffers.bufs.Slots)
	count(selectedExpertWorkGPUBuffers.bufs.Weights)
	count(selectedExpertWorkGPUBuffers.groupedBufs.WorkOrder)
	count(selectedExpertWorkGPUBuffers.groupedBufs.ActiveExperts)
	count(selectedExpertWorkGPUBuffers.groupedBufs.Offsets)
	count(selectedExpertWorkGPUBuffers.groupedArrays.WorkPositions)
	count(selectedExpertWorkGPUBuffers.groupedArrays.WorkWeights)
	count(selectedExpertWorkGPUBuffers.groupedArrays.WorkDownScales)
	count(selectedExpertWorkGPUBuffers.groupedArrays.WorkSlots)
	count(selectedExpertWorkGPUBuffers.groupedArrays.WorkActive)
	count(selectedExpertWorkGPUBuffers.groupedArrays.ActiveExperts)
	count(selectedExpertWorkGPUBuffers.groupedArrays.Offsets)
	count(selectedExpertWorkGPUBuffers.groupedArrays.EffectiveWeights)
	selectedExpertWorkGPUBuffers.Unlock()
	ggufGPUExpertScratch.Lock()
	count(ggufGPUExpertScratch.x)
	count(ggufGPUExpertScratch.gu)
	count(ggufGPUExpertScratch.act)
	count(ggufGPUExpertScratch.up)
	count(ggufGPUExpertScratch.down)
	count(ggufGPUExpertScratch.pos)
	count(ggufGPUExpertScratch.weights)
	ggufGPUExpertScratch.Unlock()
	ggufGPUFusedExpertScratch.Lock()
	count(ggufGPUFusedExpertScratch.residual)
	count(ggufGPUFusedExpertScratch.workInput)
	count(ggufGPUFusedExpertScratch.act)
	count(ggufGPUFusedExpertScratch.up)
	count(ggufGPUFusedExpertScratch.moeOut)
	ggufGPUFusedExpertScratch.Unlock()
	residentGemmScratch.Lock()
	count(residentGemmScratch.x)
	count(residentGemmScratch.y)
	residentGemmScratch.Unlock()
	return buffers, bytes
}

func FreeGGUFGPUExpertCaches() {
	activeQ5DownPointerTableCache.Range(func(_, v any) bool {
		if t, ok := v.(*gpu.GPUQ5_0PointerTable); ok && t != nil {
			t.Free()
		}
		return true
	})
	activeQ8DownPointerTableCache.Range(func(_, v any) bool {
		if t, ok := v.(*gpu.GPUQ8_0PointerTable); ok && t != nil {
			t.Free()
		}
		return true
	})
	activeQ4KGateUpRawPointerTableCache.Range(func(_, v any) bool {
		if t, ok := v.(*gpu.GPUQ4KPointerTableRaw); ok && t != nil {
			t.Free()
		}
		return true
	})
	activeQ4KGateUpPointerTableCache.Range(func(_, v any) bool {
		if t, ok := v.(*gpu.GPUQ4KPointerTable); ok && t != nil {
			t.Free()
		}
		return true
	})
	activeQ8DownCache.Range(func(_, v any) bool {
		if m, ok := v.(*gpu.GPUQ8_0Matrix); ok && m != nil {
			m.Free()
		}
		return true
	})
	activeQ4KGateUpCache.Range(func(_, v any) bool {
		if m, ok := v.(*gpu.GPUQ4KMatrix); ok && m != nil {
			m.Free()
		}
		return true
	})
	q8DownExpertCache.Range(func(_, v any) bool {
		if m, ok := v.(*gpu.GPUQ8_0Matrix); ok && m != nil {
			m.Free()
		}
		return true
	})
	q5DownExpertCache.Range(func(_, v any) bool {
		if m, ok := v.(*gpu.GPUQ5_0Matrix); ok && m != nil {
			m.Free()
		}
		return true
	})
	q4KGateUpExpertRawCache.Range(func(_, v any) bool {
		if m, ok := v.(*gpu.GPUQ4KMatrixRaw); ok && m != nil {
			m.Free()
		}
		return true
	})
	q4KGateUpExpertCache.Range(func(_, v any) bool {
		if m, ok := v.(*gpu.GPUQ4KMatrix); ok && m != nil {
			m.Free()
		}
		return true
	})
	activeQ5DownPointerTableCache = sync.Map{}
	activeQ8DownPointerTableCache = sync.Map{}
	activeQ4KGateUpPointerTableCache = sync.Map{}
	activeQ4KGateUpRawPointerTableCache = sync.Map{}
	activeQ8DownCache = sync.Map{}
	activeQ4KGateUpCache = sync.Map{}
	q8DownExpertCache = sync.Map{}
	q5DownExpertCache = sync.Map{}
	q4KGateUpExpertCache = sync.Map{}
	q4KGateUpExpertRawCache = sync.Map{}
	for _, m := range ggufTransientPointerExpertScratch.q4 {
		if m != nil {
			m.Free()
		}
	}
	for _, m := range ggufTransientPointerExpertScratch.q8 {
		if m != nil {
			m.Free()
		}
	}
	ggufTransientPointerExpertScratch.q4 = nil
	ggufTransientPointerExpertScratch.q8 = nil
	ggufGPUExpertCache.Lock()
	for _, w := range ggufGPUExpertCache.items {
		if w == nil {
			continue
		}
		if w.GateUp != nil {
			w.GateUp.Free()
		}
		if w.GateUpQ4K != nil {
			w.GateUpQ4K.Free()
		}
		if w.Down != nil {
			w.Down.Free()
		}
		if w.DownQ8 != nil {
			w.DownQ8.Free()
		}
		if w.DownQ5 != nil {
			w.DownQ5.Free()
		}
	}
	ggufGPUExpertCache.items = map[ggufGPUExpertKey]*ggufGPUExpertWeights{}
	ggufGPUExpertCache.bytes = 0
	ggufGPUExpertCache.Unlock()
	activeExpertMatrixCacheBudget.Lock()
	activeExpertMatrixCacheBudget.bytes = 0
	activeExpertMatrixCacheBudget.Unlock()
}

func activeQ4KGateUpMatrix(idx *GGUFExpertIndex, layer int, active []int) (*gpu.GPUQ4KMatrix, error) {
	return activeQ4KGateUpMatrixWithCache(idx, layer, active, true)
}

func transientActiveQ4KGateUpMatrix(idx *GGUFExpertIndex, layer int, active []int) (*gpu.GPUQ4KMatrix, error) {
	return activeQ4KGateUpMatrixWithCache(idx, layer, active, false)
}

func activeQ4KGateUpMatrixWithCache(idx *GGUFExpertIndex, layer int, active []int, cache bool) (*gpu.GPUQ4KMatrix, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || len(active) == 0 {
		return nil, fmt.Errorf("invalid active Q4_K gate/up request layer=%d active=%d", layer, len(active))
	}
	var h uint64 = 1469598103934665603
	for _, expert := range active {
		h ^= uint64(uint32(expert) + 0x9e3779b9)
		h *= 1099511628211
	}
	key := activeQ4KGateUpKey{index: ggufExpertIndexCacheID(idx), layer: layer, hash: h, count: len(active)}
	if cache {
		if cached, ok := activeQ4KGateUpCache.Load(key); ok {
			return cached.(*gpu.GPUQ4KMatrix), nil
		}
	}
	le := idx.entries[layer]
	rowBytes, err := le.gateUp.RowBytes()
	if err != nil {
		return nil, err
	}
	rowsPerExpert := le.gateUp.OutDim
	for _, expert := range active {
		if expert < 0 || expert >= idx.NumExperts {
			return nil, fmt.Errorf("active expert %d outside %d", expert, idx.NumExperts)
		}
		start := expert * rowsPerExpert * rowBytes
		end := start + rowsPerExpert*rowBytes
		if start < 0 || end < start || end > len(le.gateUp.Raw) {
			return nil, fmt.Errorf("active expert %d gate_up raw outside", expert)
		}
	}
	blocks := le.gateUp.InDim / 256
	cacheBytes := int64(len(active) * rowsPerExpert * blocks * (128 + 8*4 + 8*4))
	if cache && !reserveActiveExpertMatrixCacheBytes(cacheBytes) {
		return nil, errActiveExpertMatrixCacheBudget
	}
	raw := make([]byte, len(active)*rowsPerExpert*rowBytes)
	for ai, expert := range active {
		start := expert * rowsPerExpert * rowBytes
		end := start + rowsPerExpert*rowBytes
		copy(raw[ai*rowsPerExpert*rowBytes:(ai+1)*rowsPerExpert*rowBytes], le.gateUp.Raw[start:end])
	}
	m, err := gpu.UploadQ4KMatrixRows(raw, le.gateUp.InDim, rowsPerExpert*len(active))
	if err != nil {
		if cache {
			releaseActiveExpertMatrixCacheBytes(cacheBytes)
		}
		return nil, err
	}
	if !cache {
		return m, nil
	}
	actual, loaded := activeQ4KGateUpCache.LoadOrStore(key, m)
	if loaded {
		m.Free()
		releaseActiveExpertMatrixCacheBytes(cacheBytes)
		return actual.(*gpu.GPUQ4KMatrix), nil
	}
	return m, nil
}

type ggufGPUExpertKey struct{ layer, expert int }
type ggufGPUExpertWeights struct {
	GateUp    *gpu.Buffer
	GateUpQ4K *gpu.GPUQ4KMatrix
	Down      *gpu.Buffer
	DownQ8    *gpu.GPUQ8_0Matrix
	DownQ5    *gpu.GPUQ5_0Matrix
	DownScale float32
	Bytes     int64
}

var ggufGPUExpertCache = struct {
	sync.Mutex
	items map[ggufGPUExpertKey]*ggufGPUExpertWeights
	bytes int64
}{items: map[ggufGPUExpertKey]*ggufGPUExpertWeights{}}

var selectedExpertWorkGPUBuffers = struct {
	sync.Mutex
	bufs          SelectedExpertWorkGPUBuffers
	groupedBufs   SelectedExpertGroupedWorkGPUBuffers
	groupedArrays SelectedExpertGroupedArraysGPUBuffers
	items         []SelectedExpertWorkItem
	arrays        SelectedExpertWorkArrays
}{}

func diffusionGemmaGGUFGPUExpertCacheBytes() int64 {
	v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_CACHE_MB"))
	if v == "" {
		// Experimental selected-expert F32 GPU cache is opt-in. On RTX 3060 it is
		// slower than the batched GGUF CPU expert path due first-use dequant/upload
		// and per-expert launch overhead, so do not enable it by default.
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 0
	}
	return int64(n) * 1024 * 1024
}

func diffusionGemmaGGUFGPUExpertTransientActiveEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_TRANSIENT_ACTIVE")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func diffusionGemmaGGUFGPUExpertTransientPointerEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_TRANSIENT_POINTER")))
	return v == "1" || v == "true" || v == "yes" || v == "on" || diffusionGemmaRequireGroupedExpertGraph()
}

func diffusionGemmaGGUFGPUExpertRawQ4Enabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_RAW_Q4")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func diffusionGemmaGGUFGPUExpertAllowTanhGELUEnabled() bool {
	// llama.cpp's LLM_FFN_GELU lowers to ggml_gelu, the tanh-approximation
	// GELU. Do not allow the old dot-only + host activation boundary in the
	// production GGUF GPU expert path: it is both slower and no longer the
	// llama.cpp graph we are matching.
	return true
}

func diffusionGemmaGGUFGPUExpertPartialResidentEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PARTIAL_RESIDENT")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func diffusionGemmaGGUFGPUExpertPartialResidentLayers() int {
	v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PARTIAL_RESIDENT_LAYERS"))
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 0
	}
	return n
}

func diffusionGemmaGGUFGPUExpertSkipDoomedAttemptsEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_SKIP_DOOMED")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func diffusionGemmaGGUFGPUExpertActiveTraceTop() int {
	v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_ACTIVE_TRACE_TOP"))
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 0
	}
	if n > 128 {
		return 128
	}
	return n
}

func diffusionGemmaGGUFGPUExpertPrewarmQ4OnlyEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_Q4_ONLY")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func diffusionGemmaGGUFGPUExpertPrewarmPlanOnlyEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_PLAN_ONLY")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func diffusionGemmaGGUFGPUExpertPrewarmReserveBytes() uint64 {
	v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_RESERVE_MB"))
	if v == "" {
		return 512 * 1024 * 1024
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return 512 * 1024 * 1024
	}
	return uint64(n) * 1024 * 1024
}

func diffusionGemmaGGUFGPUExpertPrewarmCacheReserveBytes() int64 {
	v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_CACHE_RESERVE_MB"))
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 0
	}
	return int64(n) * 1024 * 1024
}

type ggufExpertPrewarmTarget struct {
	Layer  int
	Expert int
}

func diffusionGemmaGGUFGPUExpertPrewarmPlan(maxLayers, numExperts int) []ggufExpertPrewarmTarget {
	v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_PLAN"))
	if v == "" || maxLayers <= 0 || numExperts <= 0 {
		return nil
	}
	seen := make(map[ggufExpertPrewarmTarget]bool)
	out := make([]ggufExpertPrewarmTarget, 0)
	for _, group := range strings.Split(v, ";") {
		group = strings.TrimSpace(group)
		if group == "" {
			continue
		}
		parts := strings.SplitN(group, ":", 2)
		if len(parts) != 2 {
			continue
		}
		layer, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil || layer < 0 || layer >= maxLayers {
			continue
		}
		for _, expertText := range strings.Split(parts[1], ",") {
			expert, err := strconv.Atoi(strings.TrimSpace(expertText))
			if err != nil || expert < 0 || expert >= numExperts {
				continue
			}
			t := ggufExpertPrewarmTarget{Layer: layer, Expert: expert}
			if !seen[t] {
				seen[t] = true
				out = append(out, t)
			}
		}
	}
	return out
}

func residentGGUFGPUExpertWeights(idx *GGUFExpertIndex, layer, expert int) (*ggufGPUExpertWeights, bool, error) {
	budget := diffusionGemmaGGUFGPUExpertCacheBytes()
	if budget <= 0 || idx == nil || layer < 0 || layer >= idx.NumLayers || expert < 0 || expert >= idx.NumExperts {
		return nil, false, nil
	}
	key := ggufGPUExpertKey{layer: layer, expert: expert}
	ggufGPUExpertCache.Lock()
	if w := ggufGPUExpertCache.items[key]; w != nil {
		ggufGPUExpertCache.Unlock()
		return w, true, nil
	}
	ggufGPUExpertCache.Unlock()

	le := idx.entries[layer]
	hidden, intermediate := idx.HiddenSize, idx.Intermediate
	gateRows := intermediate * 2
	var gateF32 []float32
	var gateQ4K *gpu.GPUQ4KMatrix
	if le.gateUp.QType == gguf.QuantQ4_K {
		rowBytes, err := le.gateUp.RowBytes()
		if err != nil {
			return nil, false, err
		}
		start := expert * le.gateUp.OutDim * rowBytes
		end := start + le.gateUp.OutDim*rowBytes
		if start < 0 || end < start || end > len(le.gateUp.Raw) {
			return nil, false, fmt.Errorf("GGUF GPU expert Q4 gate_up raw range invalid expert=%d", expert)
		}
		gateQ4K, err = gpu.UploadQ4KMatrixRows(le.gateUp.Raw[start:end], le.gateUp.InDim, le.gateUp.OutDim)
		if err != nil {
			return nil, false, err
		}
	} else {
		gateF32 = make([]float32, gateRows*hidden)
		row := make([]float32, hidden)
		for r := 0; r < gateRows; r++ {
			if err := le.gateUp.DequantExpertRowTo(row, expert, r); err != nil {
				return nil, false, err
			}
			copy(gateF32[r*hidden:(r+1)*hidden], row)
		}
	}

	var downF32 []float32
	var downQ8 *gpu.GPUQ8_0Matrix
	var downQ5 *gpu.GPUQ5_0Matrix
	downScale := float32(1)
	if le.downScale != nil {
		downScale = le.downScale[expert]
	}
	if le.down.QType == gguf.QuantQ8_0 {
		rowBytes, err := le.down.RowBytes()
		if err != nil {
			if gateQ4K != nil {
				gateQ4K.Free()
			}
			return nil, false, err
		}
		start := expert * le.down.OutDim * rowBytes
		end := start + le.down.OutDim*rowBytes
		if start < 0 || end < start || end > len(le.down.Raw) {
			if gateQ4K != nil {
				gateQ4K.Free()
			}
			return nil, false, fmt.Errorf("GGUF GPU expert Q8 down raw range invalid expert=%d", expert)
		}
		downQ8, err = gpu.UploadQ8_0MatrixRows(le.down.Raw[start:end], le.down.InDim, le.down.OutDim)
		if err != nil {
			if gateQ4K != nil {
				gateQ4K.Free()
			}
			return nil, false, err
		}
	} else if le.down.QType == gguf.QuantQ5_0 {
		rowBytes, err := le.down.RowBytes()
		if err != nil {
			if gateQ4K != nil {
				gateQ4K.Free()
			}
			return nil, false, err
		}
		start := expert * le.down.OutDim * rowBytes
		end := start + le.down.OutDim*rowBytes
		if start < 0 || end < start || end > len(le.down.Raw) {
			if gateQ4K != nil {
				gateQ4K.Free()
			}
			return nil, false, fmt.Errorf("GGUF GPU expert Q5 down raw range invalid expert=%d", expert)
		}
		downQ5, err = gpu.UploadQ5_0MatrixRows(le.down.Raw[start:end], le.down.InDim, le.down.OutDim)
		if err != nil {
			if gateQ4K != nil {
				gateQ4K.Free()
			}
			return nil, false, err
		}
	} else {
		downF32 = make([]float32, hidden*intermediate)
		downRow := make([]float32, intermediate)
		for r := 0; r < hidden; r++ {
			if err := le.down.DequantExpertRowTo(downRow, expert, r); err != nil {
				if gateQ4K != nil {
					gateQ4K.Free()
				}
				return nil, false, err
			}
			if le.downScale != nil {
				for i := range downRow {
					downRow[i] *= downScale
				}
			}
			copy(downF32[r*intermediate:(r+1)*intermediate], downRow)
		}
	}
	var bytes int64
	if gateQ4K != nil {
		bytes += int64(le.gateUp.OutDim*le.gateUp.InDim/2 + le.gateUp.OutDim*(le.gateUp.InDim/256)*8*4*2)
	} else {
		bytes += int64(len(gateF32) * 4)
	}
	if downQ8 != nil {
		bytes += int64(le.down.OutDim*le.down.InDim + (le.down.InDim/32)*le.down.OutDim*4)
	} else if downQ5 != nil {
		bytes += int64(le.down.OutDim*(le.down.InDim/2) + (le.down.InDim/32)*le.down.OutDim*(4+4))
	} else {
		bytes += int64(len(downF32) * 4)
	}
	ggufGPUExpertCache.Lock()
	if existing := ggufGPUExpertCache.items[key]; existing != nil {
		ggufGPUExpertCache.Unlock()
		if gateQ4K != nil {
			gateQ4K.Free()
		}
		if downQ8 != nil {
			downQ8.Free()
		}
		if downQ5 != nil {
			downQ5.Free()
		}
		return existing, true, nil
	}
	if ggufGPUExpertCache.bytes+bytes > budget {
		ggufGPUExpertCache.Unlock()
		if gateQ4K != nil {
			gateQ4K.Free()
		}
		if downQ8 != nil {
			downQ8.Free()
		}
		if downQ5 != nil {
			downQ5.Free()
		}
		return nil, false, nil
	}
	ggufGPUExpertCache.bytes += bytes
	ggufGPUExpertCache.Unlock()

	var gateBuf *gpu.Buffer
	if gateQ4K == nil {
		var err error
		gateBuf, err = uploadTransposedF32Matrix(gateF32, gateRows, hidden)
		if err != nil {
			if downQ8 != nil {
				downQ8.Free()
			}
			if downQ5 != nil {
				downQ5.Free()
			}
			return nil, false, err
		}
	}
	var downBuf *gpu.Buffer
	if downQ8 == nil && downQ5 == nil {
		var err error
		downBuf, err = uploadTransposedF32Matrix(downF32, hidden, intermediate)
		if err != nil {
			if gateBuf != nil {
				gateBuf.Free()
			}
			if gateQ4K != nil {
				gateQ4K.Free()
			}
			return nil, false, err
		}
	}
	w := &ggufGPUExpertWeights{GateUp: gateBuf, GateUpQ4K: gateQ4K, Down: downBuf, DownQ8: downQ8, DownQ5: downQ5, DownScale: downScale, Bytes: bytes}
	ggufGPUExpertCache.Lock()
	ggufGPUExpertCache.items[key] = w
	ggufGPUExpertCache.Unlock()
	return w, true, nil
}

func transposeF32MatrixInto(dst, w []float32, rows, cols int) error {
	n, ok := checked.MulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(w) < n || len(dst) < n {
		return fmt.Errorf("invalid F32 matrix for transpose rows=%d cols=%d len=%d dst=%d", rows, cols, len(w), len(dst))
	}
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			dst[c*rows+r] = w[r*cols+c]
		}
	}
	return nil
}

type ggufDenseTransposeCacheKey struct {
	ptr        uintptr
	rows, cols int
}

type ggufDenseTransposeCacheEntry struct {
	data  []float32
	bytes int64
}

var ggufDenseTransposeCache = struct {
	sync.Mutex
	entries map[ggufDenseTransposeCacheKey]ggufDenseTransposeCacheEntry
	order   []ggufDenseTransposeCacheKey
	bytes   int64
}{entries: map[ggufDenseTransposeCacheKey]ggufDenseTransposeCacheEntry{}}

func diffusionGemmaGGUFDenseTransposeCacheBytes() int64 {
	v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_DENSE_TRANSPOSE_CACHE_MB"))
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 0
	}
	return int64(n) * 1024 * 1024
}

func GGUFDenseTransposeCacheStats() (entries int, bytes int64) {
	ggufDenseTransposeCache.Lock()
	defer ggufDenseTransposeCache.Unlock()
	return len(ggufDenseTransposeCache.entries), ggufDenseTransposeCache.bytes
}

func FreeGGUFDenseTransposeCache() {
	ggufDenseTransposeCache.Lock()
	defer ggufDenseTransposeCache.Unlock()
	ggufDenseTransposeCache.entries = map[ggufDenseTransposeCacheKey]ggufDenseTransposeCacheEntry{}
	ggufDenseTransposeCache.order = nil
	ggufDenseTransposeCache.bytes = 0
}

func cachedTransposedF32Matrix(w []float32, rows, cols int) ([]float32, bool, error) {
	return cachedTransposedF32MatrixWithPolicy(w, rows, cols, false)
}

func cachedTransposedF32MatrixNoEvict(w []float32, rows, cols int) ([]float32, bool, error) {
	return cachedTransposedF32MatrixWithPolicy(w, rows, cols, false)
}

func cachedTransposedF32MatrixWithPolicy(w []float32, rows, cols int, allowEvict bool) ([]float32, bool, error) {
	n, ok := checked.MulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(w) < n {
		return nil, false, fmt.Errorf("invalid F32 matrix for cached transpose rows=%d cols=%d len=%d", rows, cols, len(w))
	}
	budget := diffusionGemmaGGUFDenseTransposeCacheBytes()
	if budget <= 0 {
		return nil, false, nil
	}
	key := ggufDenseTransposeCacheKey{ptr: uintptr(unsafe.Pointer(&w[0])), rows: rows, cols: cols}
	ggufDenseTransposeCache.Lock()
	defer ggufDenseTransposeCache.Unlock()
	if ent, ok := ggufDenseTransposeCache.entries[key]; ok && len(ent.data) >= n {
		for i, k := range ggufDenseTransposeCache.order {
			if k == key {
				copy(ggufDenseTransposeCache.order[i:], ggufDenseTransposeCache.order[i+1:])
				ggufDenseTransposeCache.order[len(ggufDenseTransposeCache.order)-1] = key
				break
			}
		}
		return ent.data[:n], true, nil
	}
	bytes := int64(n * 4)
	if bytes > budget {
		return nil, false, nil
	}
	if !allowEvict && ggufDenseTransposeCache.bytes+bytes > budget {
		return nil, false, nil
	}
	for allowEvict && ggufDenseTransposeCache.bytes+bytes > budget && len(ggufDenseTransposeCache.order) > 0 {
		old := ggufDenseTransposeCache.order[0]
		ggufDenseTransposeCache.order = ggufDenseTransposeCache.order[1:]
		if ent, ok := ggufDenseTransposeCache.entries[old]; ok {
			delete(ggufDenseTransposeCache.entries, old)
			ggufDenseTransposeCache.bytes -= ent.bytes
		}
	}
	if ggufDenseTransposeCache.bytes+bytes > budget {
		return nil, false, nil
	}
	data := make([]float32, n)
	if err := transposeF32MatrixInto(data, w, rows, cols); err != nil {
		return nil, false, err
	}
	ggufDenseTransposeCache.entries[key] = ggufDenseTransposeCacheEntry{data: data, bytes: bytes}
	ggufDenseTransposeCache.order = append(ggufDenseTransposeCache.order, key)
	ggufDenseTransposeCache.bytes += bytes
	return data, false, nil
}

func uploadTransposedF32Matrix(w []float32, rows, cols int) (*gpu.Buffer, error) {
	n, ok := checked.MulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(w) < n {
		return nil, fmt.Errorf("invalid F32 matrix for transpose upload rows=%d cols=%d len=%d", rows, cols, len(w))
	}
	transposed := make([]float32, n)
	if err := transposeF32MatrixInto(transposed, w, rows, cols); err != nil {
		return nil, err
	}
	buf, err := gpu.Malloc(n)
	if err != nil {
		return nil, err
	}
	if err := buf.Upload(transposed); err != nil {
		buf.Free()
		return nil, err
	}
	return buf, nil
}

type ggufTempDenseUploadSlot struct {
	buf   *gpu.Buffer
	elems int
	host  []float32
}

var ggufTempDenseUploadScratch = struct {
	sync.Mutex
	slots map[string]*ggufTempDenseUploadSlot
}{slots: map[string]*ggufTempDenseUploadSlot{}}

type ggufTempDenseUploadSession struct {
	closed bool
}

func beginGGUFTempDenseUploadSession() *ggufTempDenseUploadSession {
	ggufTempDenseUploadScratch.Lock()
	return &ggufTempDenseUploadSession{}
}

func (s *ggufTempDenseUploadSession) Close() {
	if s != nil && !s.closed {
		s.closed = true
		ggufTempDenseUploadScratch.Unlock()
	}
}

func FreeGGUFTempDenseUploadScratch() {
	ggufTempDenseUploadScratch.Lock()
	defer ggufTempDenseUploadScratch.Unlock()
	for _, slot := range ggufTempDenseUploadScratch.slots {
		if slot != nil && slot.buf != nil {
			slot.buf.Free()
		}
	}
	ggufTempDenseUploadScratch.slots = map[string]*ggufTempDenseUploadSlot{}
}

func GGUFTempDenseUploadScratchStats() (slots int, deviceBytes int64, hostElems int) {
	ggufTempDenseUploadScratch.Lock()
	defer ggufTempDenseUploadScratch.Unlock()
	for _, slot := range ggufTempDenseUploadScratch.slots {
		if slot == nil {
			continue
		}
		slots++
		if slot.buf != nil {
			deviceBytes += int64(slot.buf.Size)
		}
		hostElems += cap(slot.host)
	}
	return slots, deviceBytes, hostElems
}

func (s *ggufTempDenseUploadSession) Upload(slotName string, w []float32, rows, cols int) (*gpu.Buffer, error) {
	if s == nil {
		return nil, fmt.Errorf("nil GGUF temp dense upload session")
	}
	n, ok := checked.MulInt(rows, cols)
	if slotName == "" || rows <= 0 || cols <= 0 || !ok || len(w) < n {
		return nil, fmt.Errorf("invalid GGUF temp dense upload slot=%q rows=%d cols=%d len=%d", slotName, rows, cols, len(w))
	}
	slot := ggufTempDenseUploadScratch.slots[slotName]
	if slot == nil {
		slot = &ggufTempDenseUploadSlot{}
		ggufTempDenseUploadScratch.slots[slotName] = slot
	}
	t0 := time.Now()
	host, cacheHit, err := cachedTransposedF32Matrix(w, rows, cols)
	if err != nil {
		return nil, err
	}
	if strings.HasPrefix(slotName, "forward_attn_") {
		ggufTempDenseUploadCounters.forwardAttnCalls.Add(1)
		if cacheHit {
			ggufTempDenseUploadCounters.forwardAttnHits.Add(1)
		}
	} else if strings.HasPrefix(slotName, "forward_mlp_") {
		ggufTempDenseUploadCounters.forwardMLPCalls.Add(1)
		if cacheHit {
			ggufTempDenseUploadCounters.forwardMLPHits.Add(1)
		}
	} else if strings.HasPrefix(slotName, "encoder_attn_") {
		ggufTempDenseUploadCounters.encoderAttnCalls.Add(1)
		if cacheHit {
			ggufTempDenseUploadCounters.encoderAttnHits.Add(1)
		}
	} else if strings.HasPrefix(slotName, "encoder_mlp_") {
		ggufTempDenseUploadCounters.encoderMLPCalls.Add(1)
		if cacheHit {
			ggufTempDenseUploadCounters.encoderMLPHits.Add(1)
		}
	}
	if cacheHit {
		ggufTempDenseUploadCounters.cacheHits.Add(1)
	} else {
		ggufTempDenseUploadCounters.cacheMisses.Add(1)
	}
	if host == nil {
		if cap(slot.host) < n {
			slot.host = make([]float32, n)
		}
		host = slot.host[:n]
		if err := transposeF32MatrixInto(host, w, rows, cols); err != nil {
			return nil, err
		}
		ggufTempDenseUploadCounters.transposeNS.Add(uint64(time.Since(t0).Nanoseconds()))
	} else if !cacheHit {
		ggufTempDenseUploadCounters.transposeNS.Add(uint64(time.Since(t0).Nanoseconds()))
	}
	if slot.buf == nil || slot.elems < n {
		if slot.buf != nil {
			slot.buf.Free()
			slot.buf = nil
			slot.elems = 0
		}
		buf, err := gpu.Malloc(n)
		if err != nil {
			return nil, err
		}
		slot.buf = buf
		slot.elems = n
	}
	t0 = time.Now()
	if err := slot.buf.Upload(host); err != nil {
		return nil, err
	}
	ggufTempDenseUploadCounters.uploadNS.Add(uint64(time.Since(t0).Nanoseconds()))
	ggufTempDenseUploadCounters.calls.Add(1)
	ggufTempDenseUploadCounters.bytes.Add(uint64(n * 4))
	return slot.buf, nil
}

func residentGGUFGPUAttentionWeights(layer int, lb TextLayerBindings, qW, kW, vW, oW []float32, qRows, kRows, vRows, hiddenSize int) (*GGUFGPUAttentionWeights, error) {
	vName := ""
	if lb.VProj != nil {
		vName = lb.VProj.Name
	}
	key := ggufGPUAttentionKey{layer: layer, qName: lb.QProj.Name, kName: lb.KProj.Name, vName: vName, oName: lb.OProj.Name}
	if cached, ok := ggufGPUAttentionCache.Load(key); ok {
		return cached.(*GGUFGPUAttentionWeights), nil
	}
	qBuf, err := uploadTransposedF32Matrix(qW, qRows, hiddenSize)
	if err != nil {
		return nil, fmt.Errorf("upload resident GGUF Q layer=%d: %w", layer, err)
	}
	kBuf, err := uploadTransposedF32Matrix(kW, kRows, hiddenSize)
	if err != nil {
		qBuf.Free()
		return nil, fmt.Errorf("upload resident GGUF K layer=%d: %w", layer, err)
	}
	var vBuf *gpu.Buffer
	if lb.VProj != nil {
		vBuf, err = uploadTransposedF32Matrix(vW, vRows, hiddenSize)
		if err != nil {
			qBuf.Free()
			kBuf.Free()
			return nil, fmt.Errorf("upload resident GGUF V layer=%d: %w", layer, err)
		}
	}
	oBuf, err := uploadTransposedF32Matrix(oW, hiddenSize, qRows)
	if err != nil {
		qBuf.Free()
		kBuf.Free()
		if vBuf != nil {
			vBuf.Free()
		}
		return nil, fmt.Errorf("upload resident GGUF O layer=%d: %w", layer, err)
	}
	weights := &GGUFGPUAttentionWeights{Q: qBuf, K: kBuf, V: vBuf, O: oBuf, QRows: qRows, KRows: kRows, VRows: vRows, Hidden: hiddenSize}
	actual, loaded := ggufGPUAttentionCache.LoadOrStore(key, weights)
	if loaded {
		qBuf.Free()
		kBuf.Free()
		if vBuf != nil {
			vBuf.Free()
		}
		oBuf.Free()
		return actual.(*GGUFGPUAttentionWeights), nil
	}
	return weights, nil
}

// residentF32MatrixBuffer uploads a stable F32 weight matrix once and reuses it.
// This is the first step away from SgemmHost's per-GEMM weight uploads; the
// remaining activation/output buffers are still per-call until the graph is made
// fully device-resident.
func residentF32MatrixBuffer(W []float32, M, K int) (*gpu.Buffer, error) {
	mk, ok := checked.MulInt(M, K)
	if M <= 0 || K <= 0 || !ok || len(W) < mk {
		return nil, fmt.Errorf("invalid resident F32 matrix M=%d K=%d len=%d want=%d", M, K, len(W), mk)
	}
	key := residentF32MatrixKey{ptr: uintptr(unsafe.Pointer(&W[0])), m: M, k: K}
	if v, ok := residentF32MatrixCache.Load(key); ok {
		buf := v.(*gpu.Buffer)
		if buf != nil && buf.Ptr != 0 && buf.Size >= mk*4 {
			return buf, nil
		}
	}
	buf, err := gpu.Malloc(mk)
	if err != nil {
		return nil, err
	}
	if err := buf.Upload(W[:mk]); err != nil {
		buf.Free()
		return nil, err
	}
	actual, loaded := residentF32MatrixCache.LoadOrStore(key, buf)
	if loaded {
		buf.Free()
		return actual.(*gpu.Buffer), nil
	}
	return buf, nil
}

var ggufGPUExpertScratch = struct {
	sync.Mutex
	x           *gpu.Buffer
	gu          *gpu.Buffer
	act         *gpu.Buffer
	up          *gpu.Buffer
	down        *gpu.Buffer
	pos         *gpu.Buffer
	weights     *gpu.Buffer
	xN          int
	guN         int
	actN        int
	upN         int
	downN       int
	posN        int
	weightN     int
	hostBatchIn []float32
	hostPosIDs  []uint32
	hostWeights []float32
}{}

var ggufGPUFusedExpertScratch = struct {
	sync.Mutex
	residual  *gpu.Buffer
	workInput *gpu.Buffer
	act       *gpu.Buffer
	up        *gpu.Buffer
	moeOut    *gpu.Buffer
	residualN int
	workN     int
	actN      int
	upN       int
	moeOutN   int
}{}

func ggufGPUFusedExpertScratchBuffers(positions, workLen, hidden, intermediate int) (residual, workInput, act, up, moeOut *gpu.Buffer, unlock func(), err error) {
	ggufGPUFusedExpertScratch.Lock()
	unlock = func() { ggufGPUFusedExpertScratch.Unlock() }
	ensureFloat := func(cur **gpu.Buffer, curN *int, need int, label string) error {
		if need <= 0 {
			return fmt.Errorf("invalid GGUF fused GPU expert %s scratch size %d", label, need)
		}
		if *cur != nil && *curN >= need {
			return nil
		}
		if *cur != nil {
			(*cur).Free()
			*cur = nil
			*curN = 0
		}
		buf, err := gpu.Malloc(need)
		if err != nil {
			return fmt.Errorf("alloc GGUF fused GPU expert %s scratch: %w", label, err)
		}
		*cur = buf
		*curN = need
		return nil
	}
	if err := ensureFloat(&ggufGPUFusedExpertScratch.residual, &ggufGPUFusedExpertScratch.residualN, positions*hidden, "residual"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, err
	}
	if err := ensureFloat(&ggufGPUFusedExpertScratch.workInput, &ggufGPUFusedExpertScratch.workN, workLen*hidden, "work_input"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, err
	}
	if err := ensureFloat(&ggufGPUFusedExpertScratch.act, &ggufGPUFusedExpertScratch.actN, workLen*intermediate, "activation"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, err
	}
	if err := ensureFloat(&ggufGPUFusedExpertScratch.up, &ggufGPUFusedExpertScratch.upN, workLen*intermediate, "up"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, err
	}
	if err := ensureFloat(&ggufGPUFusedExpertScratch.moeOut, &ggufGPUFusedExpertScratch.moeOutN, positions*hidden, "moe_out"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, err
	}
	return ggufGPUFusedExpertScratch.residual, ggufGPUFusedExpertScratch.workInput, ggufGPUFusedExpertScratch.act, ggufGPUFusedExpertScratch.up, ggufGPUFusedExpertScratch.moeOut, unlock, nil
}

func ggufGPUExpertScratchBuffers(nPos, hidden, intermediate int) (x, gu, act, up, down, pos, weights *gpu.Buffer, hostBatchIn []float32, hostPosIDs []uint32, hostWeights []float32, unlock func(), err error) {
	ggufGPUExpertScratch.Lock()
	unlock = func() { ggufGPUExpertScratch.Unlock() }
	ensureFloat := func(cur **gpu.Buffer, curN *int, need int, label string) error {
		if need <= 0 {
			return fmt.Errorf("invalid GGUF GPU expert %s scratch size %d", label, need)
		}
		if *cur != nil && *curN >= need {
			return nil
		}
		if *cur != nil {
			(*cur).Free()
			*cur = nil
			*curN = 0
		}
		buf, err := gpu.Malloc(need)
		if err != nil {
			return fmt.Errorf("alloc GGUF GPU expert %s scratch: %w", label, err)
		}
		*cur = buf
		*curN = need
		return nil
	}
	ensureBytes := func(cur **gpu.Buffer, curN *int, need int, label string) error {
		if need <= 0 {
			return fmt.Errorf("invalid GGUF GPU expert %s scratch bytes %d", label, need)
		}
		if *cur != nil && *curN >= need {
			return nil
		}
		if *cur != nil {
			(*cur).Free()
			*cur = nil
			*curN = 0
		}
		buf, err := gpu.MallocBytes(need)
		if err != nil {
			return fmt.Errorf("alloc GGUF GPU expert %s scratch: %w", label, err)
		}
		*cur = buf
		*curN = need
		return nil
	}
	if err := ensureFloat(&ggufGPUExpertScratch.x, &ggufGPUExpertScratch.xN, nPos*hidden, "input"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, err
	}
	if err := ensureFloat(&ggufGPUExpertScratch.gu, &ggufGPUExpertScratch.guN, nPos*intermediate*2, "gate_up"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, err
	}
	if err := ensureFloat(&ggufGPUExpertScratch.act, &ggufGPUExpertScratch.actN, nPos*intermediate, "activation"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, err
	}
	if err := ensureFloat(&ggufGPUExpertScratch.up, &ggufGPUExpertScratch.upN, nPos*intermediate, "up"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, err
	}
	if err := ensureFloat(&ggufGPUExpertScratch.down, &ggufGPUExpertScratch.downN, nPos*hidden, "down"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, err
	}
	if err := ensureBytes(&ggufGPUExpertScratch.pos, &ggufGPUExpertScratch.posN, nPos*4, "positions"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, err
	}
	if err := ensureFloat(&ggufGPUExpertScratch.weights, &ggufGPUExpertScratch.weightN, nPos, "weights"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, err
	}
	if cap(ggufGPUExpertScratch.hostBatchIn) < nPos*hidden {
		ggufGPUExpertScratch.hostBatchIn = make([]float32, nPos*hidden)
	}
	if cap(ggufGPUExpertScratch.hostPosIDs) < nPos {
		ggufGPUExpertScratch.hostPosIDs = make([]uint32, nPos)
	}
	if cap(ggufGPUExpertScratch.hostWeights) < nPos {
		ggufGPUExpertScratch.hostWeights = make([]float32, nPos)
	}
	return ggufGPUExpertScratch.x, ggufGPUExpertScratch.gu, ggufGPUExpertScratch.act, ggufGPUExpertScratch.up, ggufGPUExpertScratch.down, ggufGPUExpertScratch.pos, ggufGPUExpertScratch.weights, ggufGPUExpertScratch.hostBatchIn[:nPos*hidden], ggufGPUExpertScratch.hostPosIDs[:nPos], ggufGPUExpertScratch.hostWeights[:nPos], unlock, nil

}

func runGGUFGPUExpertsGroupedFused(op LayerOp, scratch ForwardScratch, idx *GGUFExpertIndex, normedRows []float32, groupedArrays SelectedExpertGroupedArrays, metadata *SelectedExpertGroupedArraysGPUBuffers) (bool, error) {
	if idx == nil || len(groupedArrays.ActiveExperts) == 0 || metadata == nil || metadata.WorkActive == nil || metadata.WorkPositions == nil || metadata.EffectiveWeights == nil {
		return false, nil
	}
	hidden, intermediate := idx.HiddenSize, idx.Intermediate
	le := idx.entries[op.Layer]
	if le.gateUp.QType != gguf.QuantQ4_K || (le.down.QType != gguf.QuantQ8_0 && le.down.QType != gguf.QuantQ5_0) {
		return false, nil
	}
	var err error
	activeQ4Ptrs, activeQ4PtrsOK, err := activeQ4KGateUpPointerTable(idx, op.Layer, groupedArrays.ActiveExperts)
	if err != nil {
		return false, err
	}
	activeQ4PtrsTransient := false
	if !activeQ4PtrsOK {
		var cleanup func()
		activeQ4Ptrs, cleanup, err = transientActiveQ4KGateUpPointerTable(idx, op.Layer, groupedArrays.ActiveExperts)
		if err != nil {
			return false, err
		}
		defer cleanup()
		activeQ4PtrsOK = true
		activeQ4PtrsTransient = true
	}
	if !activeQ4PtrsOK {
		return false, fmt.Errorf("Q4_K gate/up pointer-table expert op required")
	}
	workLen := len(groupedArrays.WorkPositions)
	positions := len(scratch.MoeOut) / hidden
	if positions <= 0 || len(normedRows) != positions*hidden || len(scratch.MoeOut) != positions*hidden {
		return false, fmt.Errorf("invalid fused GGUF expert dimensions: normed=%d moe=%d hidden=%d", len(normedRows), len(scratch.MoeOut), hidden)
	}
	residualBuf, workInput, actBuf, _, moeOutBuf, unlockFusedScratch, err := ggufGPUFusedExpertScratchBuffers(positions, workLen, hidden, intermediate)
	if err != nil {
		return false, err
	}
	defer unlockFusedScratch()
	if err := residualBuf.Upload(normedRows); err != nil {
		return false, err
	}
	if err := gpu.GatherRows(workInput, residualBuf, metadata.WorkPositions, workLen, hidden); err != nil {
		return false, err
	}
	if activeQ4PtrsTransient {
		ggufExpertDispatchCounters.q4TransientPointer.Add(1)
	} else {
		ggufExpertDispatchCounters.q4PointerTable.Add(1)
	}
	if err := gpu.GateUpGELUQ4KByWorkPtrsToBuffer(actBuf, workInput, metadata.WorkActive, workLen, intermediate, activeQ4Ptrs); err != nil {
		return false, err
	}
	if err := gpu.ZeroFloat32Buffer(moeOutBuf, len(scratch.MoeOut)); err != nil {
		return false, err
	}
	if le.down.QType == gguf.QuantQ5_0 {
		activeQ5Ptrs, ok, err := activeQ5DownPointerTable(idx, op.Layer, groupedArrays.ActiveExperts)
		if err != nil {
			return false, err
		}
		var cleanup func()
		if !ok && diffusionGemmaGGUFGPUExpertTransientPointerEnabled() {
			activeQ5Ptrs, cleanup, err = transientActiveQ5DownPointerTable(idx, op.Layer, groupedArrays.ActiveExperts)
			if err != nil {
				return false, err
			}
			defer cleanup()
			ok = true
			ggufExpertDispatchCounters.q5PointerTable.Add(1)
		}
		if !ok {
			ggufExpertDispatchCounters.q5BudgetFallback.Add(1)
			if b, err := q5DownExpertDeviceBytes(idx, op.Layer); err == nil && b > 0 {
				ggufExpertDispatchCounters.q5BudgetBytes.Add(uint64(b) * uint64(len(groupedArrays.ActiveExperts)))
				ggufExpertDispatchCounters.q5BudgetExperts.Add(uint64(len(groupedArrays.ActiveExperts)))
			}
			return false, nil
		}
		if cleanup == nil {
			ggufExpertDispatchCounters.q5PointerTable.Add(1)
		}
		if err := gpu.GemvQ5_0ScatterByWorkPtrs(moeOutBuf, actBuf, metadata.WorkActive, metadata.WorkPositions, metadata.EffectiveWeights, workLen, activeQ5Ptrs); err != nil {
			return false, err
		}
	} else if activeQ8Ptrs, ok, err := activeQ8DownPointerTable(idx, op.Layer, groupedArrays.ActiveExperts); err != nil {
		return false, err
	} else if ok {
		ggufExpertDispatchCounters.q8PointerTable.Add(1)
		if err := gpu.GemvQ8_0ScatterByWorkPtrs(moeOutBuf, actBuf, metadata.WorkActive, metadata.WorkPositions, metadata.EffectiveWeights, workLen, activeQ8Ptrs); err != nil {
			return false, err
		}
	} else if diffusionGemmaGGUFGPUExpertTransientPointerEnabled() {
		activeQ8Ptrs, cleanup, err := transientActiveQ8DownPointerTable(idx, op.Layer, groupedArrays.ActiveExperts)
		if err != nil {
			return false, err
		}
		defer cleanup()
		ggufExpertDispatchCounters.q8TransientPointer.Add(1)
		if err := gpu.GemvQ8_0ScatterByWorkPtrs(moeOutBuf, actBuf, metadata.WorkActive, metadata.WorkPositions, metadata.EffectiveWeights, workLen, activeQ8Ptrs); err != nil {
			return false, err
		}
	} else {
		recordQ8BudgetFallback(idx, op.Layer, groupedArrays.ActiveExperts)
		return false, fmt.Errorf("Q8_0 down pointer-table expert op required")
	}
	ggufExpertDispatchCounters.fusedUsed.Add(1)
	return true, moeOutBuf.Download(scratch.MoeOut)
}

func ggufPointerExpertResident(idx *GGUFExpertIndex, layer, expert int) bool {
	if idx == nil || layer < 0 || layer >= idx.NumLayers || expert < 0 || expert >= idx.NumExperts {
		return false
	}
	if idx.entries[layer].gateUp.QType != gguf.QuantQ4_K || (idx.entries[layer].down.QType != gguf.QuantQ8_0 && idx.entries[layer].down.QType != gguf.QuantQ5_0) {
		return false
	}
	cacheID := ggufExpertIndexCacheID(idx)
	if diffusionGemmaGGUFGPUExpertRawQ4Enabled() {
		if _, ok := q4KGateUpExpertRawCache.Load(q4KGateUpExpertKey{index: cacheID, layer: layer, expert: expert}); !ok {
			return false
		}
	} else if _, ok := q4KGateUpExpertCache.Load(q4KGateUpExpertKey{index: cacheID, layer: layer, expert: expert}); !ok {
		return false
	}
	if idx.entries[layer].down.QType == gguf.QuantQ8_0 {
		if _, ok := q8DownExpertCache.Load(q8DownExpertKey{index: cacheID, layer: layer, expert: expert}); !ok {
			return false
		}
	} else if _, ok := q5DownExpertCache.Load(q5DownExpertKey{index: cacheID, layer: layer, expert: expert}); !ok {
		return false
	}
	return true
}

func runGGUFGPUExpertsGroupedPartialResident(op LayerOp, scratch ForwardScratch, idx *GGUFExpertIndex, normedRows []float32, groupedArrays SelectedExpertGroupedArrays, metadata *SelectedExpertGroupedArraysGPUBuffers) (bool, error) {
	if !diffusionGemmaGGUFGPUExpertPartialResidentEnabled() || idx == nil || metadata == nil || len(groupedArrays.ActiveExperts) == 0 {
		return false, nil
	}
	if n := diffusionGemmaGGUFGPUExpertPartialResidentLayers(); n > 0 && op.Layer >= n {
		return false, nil
	}
	kept, dropped, err := SplitSelectedExpertGroupedArrays(groupedArrays, func(expert int) bool {
		return ggufPointerExpertResident(idx, op.Layer, expert)
	})
	if err != nil {
		return false, err
	}
	if len(kept.ActiveExperts) == 0 || len(dropped.ActiveExperts) == 0 {
		return false, nil
	}
	ggufExpertDispatchCounters.partialCalls.Add(1)
	ggufExpertDispatchCounters.partialKeptExperts.Add(uint64(len(kept.ActiveExperts)))
	ggufExpertDispatchCounters.partialDroppedExperts.Add(uint64(len(dropped.ActiveExperts)))
	ggufExpertDispatchCounters.partialKeptWork.Add(uint64(len(kept.WorkPositions)))
	ggufExpertDispatchCounters.partialDroppedWork.Add(uint64(len(dropped.WorkPositions)))
	if err := metadata.Upload(kept); err != nil {
		return false, err
	}
	usedGPU, err := runGGUFGPUExpertsGroupedFused(op, scratch, idx, normedRows, kept, metadata)
	if err != nil || !usedGPU {
		return usedGPU, err
	}
	if err := runGGUFCPUExpertsGroupedNoPostNorm(op, scratch, idx, normedRows, dropped); err != nil {
		return false, err
	}
	return true, nil
}

func runGGUFGPUExpertsIndexed(op LayerOp, weights *TextWeights, scratch ForwardScratch, idx *GGUFExpertIndex) (bool, []float32, error) {
	if idx == nil || weights == nil {
		return false, nil, nil
	}
	if op.Layer < 0 || op.Layer >= idx.NumLayers {
		return false, nil, fmt.Errorf("GGUF GPU experts layer %d outside index", op.Layer)
	}
	hiddenSize := idx.HiddenSize
	positions := len(scratch.Residual) / hiddenSize
	topK := scratch.TopKExperts
	if topK <= 0 && positions > 0 {
		topK = len(scratch.TopKIDs) / positions
	}
	if positions <= 0 || topK <= 0 || len(scratch.TopKIDs) < positions*topK || len(scratch.TopKVals) < positions*topK || len(scratch.MoeOut) < len(scratch.Residual) {
		return false, nil, fmt.Errorf("GGUF GPU experts invalid scratch positions=%d topK=%d", positions, topK)
	}
	fp := weights.ForwardPlan()
	if op.Layer >= len(fp.Layers) {
		return false, nil, fmt.Errorf("GGUF GPU experts layer %d outside plan", op.Layer)
	}
	preNorm2, err := loadFloatVector(weights, fp.Layers[op.Layer].PreFFNLayerNorm2)
	if err != nil {
		return false, nil, err
	}
	if len(preNorm2) != hiddenSize {
		return false, nil, fmt.Errorf("GGUF GPU experts preNorm hidden=%d want %d", len(preNorm2), hiddenSize)
	}
	for i := range scratch.MoeOut {
		scratch.MoeOut[i] = 0
	}
	normedLen := positions * hiddenSize
	normedRows := scratch.Experts
	if len(normedRows) < normedLen {
		normedRows = make([]float32, normedLen)
	} else {
		normedRows = normedRows[:normedLen]
	}
	for pos := 0; pos < positions; pos++ {
		dst := normedRows[pos*hiddenSize : (pos+1)*hiddenSize]
		copy(dst, scratch.Residual[pos*hiddenSize:(pos+1)*hiddenSize])
		if !simd.RMSNormTo(dst, preNorm2, 1e-6) {
			return false, nil, fmt.Errorf("GGUF GPU experts pre_norm_2 rejected")
		}
	}
	selectedExpertWorkGPUBuffers.Lock()
	defer selectedExpertWorkGPUBuffers.Unlock()
	workItems, err := FlattenSelectedExpertsInto(selectedExpertWorkGPUBuffers.items, scratch.TopKIDs, scratch.TopKVals, positions, topK, idx.NumExperts)
	if err != nil {
		return false, nil, err
	}
	selectedExpertWorkGPUBuffers.items = workItems
	BuildSelectedExpertWorkArraysInto(&selectedExpertWorkGPUBuffers.arrays, workItems)
	workArrays := selectedExpertWorkGPUBuffers.arrays
	if err := workArrays.Validate(); err != nil {
		return false, nil, err
	}
	groupedWork, err := BuildSelectedExpertGroupedWork(workArrays, idx.NumExperts)
	if err != nil {
		return false, nil, err
	}
	if err := groupedWork.Validate(workArrays.Len()); err != nil {
		return false, nil, err
	}
	if _, err := SummarizeSelectedExpertGroupedWork(groupedWork, workArrays.Len()); err != nil {
		return false, nil, err
	}
	groupedArrays, err := BuildSelectedExpertGroupedArrays(workArrays, groupedWork)
	if err != nil {
		return false, nil, err
	}
	if idx.entries[op.Layer].downScale != nil {
		if err := groupedArrays.ApplyDownScalesByExpert(idx.entries[op.Layer].downScale); err != nil {
			return false, nil, err
		}
	}
	if err := groupedArrays.Validate(); err != nil {
		return false, nil, err
	}
	traceGGUFActiveExpertSet(idx, op.Layer, groupedArrays)
	if shouldSkipDoomedGGUFActiveExpertSet(idx, op.Layer, groupedArrays.ActiveExperts, len(groupedArrays.WorkPositions)) {
		usedPartial, err := runGGUFGPUExpertsGroupedPartialResident(op, scratch, idx, normedRows, groupedArrays, &selectedExpertWorkGPUBuffers.groupedArrays)
		if err != nil {
			return false, nil, err
		}
		if usedPartial {
			postNorm2, err := loadFloatVector(weights, fp.Layers[op.Layer].PostFFNLayerNorm2)
			if err != nil {
				return false, nil, err
			}
			for off := 0; off < len(scratch.MoeOut); off += hiddenSize {
				if !simd.RMSNormTo(scratch.MoeOut[off:off+hiddenSize], postNorm2, 1e-6) {
					return false, nil, fmt.Errorf("GGUF GPU experts post_norm_2 rejected")
				}
			}
			return true, nil, nil
		}
		recordQ4BudgetFallback(idx, op.Layer, groupedArrays.ActiveExperts)
		return false, nil, fmt.Errorf("GGUF GPU expert fused pointer-table op required: layer=%d active=%d work=%d", op.Layer, len(groupedArrays.ActiveExperts), len(groupedArrays.WorkPositions))
	}
	if workArrays.Len() > 0 {
		if err := selectedExpertWorkGPUBuffers.bufs.Upload(workArrays); err != nil {
			return false, nil, err
		}
		if err := selectedExpertWorkGPUBuffers.groupedBufs.Upload(groupedWork, workArrays.Len()); err != nil {
			return false, nil, err
		}
		if err := selectedExpertWorkGPUBuffers.groupedArrays.Upload(groupedArrays); err != nil {
			return false, nil, err
		}
		if len(groupedArrays.ActiveExperts) > 0 {
			counts, err := gpu.MallocBytes(len(groupedArrays.ActiveExperts) * 4)
			if err != nil {
				return false, nil, err
			}
			sums, err := gpu.Malloc(len(groupedArrays.ActiveExperts))
			if err != nil {
				counts.Free()
				return false, nil, err
			}
			if err := gpu.ExpertMetaReduce(selectedExpertWorkGPUBuffers.groupedArrays.Offsets, selectedExpertWorkGPUBuffers.groupedArrays.EffectiveWeights, counts, sums, len(groupedArrays.ActiveExperts)); err != nil {
				counts.Free()
				sums.Free()
				return false, nil, err
			}
			counts.Free()
			sums.Free()
		}
	}
	used, err := runGGUFGPUExpertsGroupedFused(op, scratch, idx, normedRows, groupedArrays, &selectedExpertWorkGPUBuffers.groupedArrays)
	if err != nil {
		return false, nil, err
	}
	if used {
		postNorm2, err := loadFloatVector(weights, fp.Layers[op.Layer].PostFFNLayerNorm2)
		if err != nil {
			return false, nil, err
		}
		for off := 0; off < len(scratch.MoeOut); off += hiddenSize {
			if !simd.RMSNormTo(scratch.MoeOut[off:off+hiddenSize], postNorm2, 1e-6) {
				return false, nil, fmt.Errorf("GGUF GPU experts post_norm_2 rejected")
			}
		}
		return true, nil, nil
	}
	return false, nil, fmt.Errorf("GGUF GPU expert fused pointer-table op did not handle layer=%d active=%d work=%d", op.Layer, len(groupedArrays.ActiveExperts), len(groupedArrays.WorkPositions))
}

func q4KGateUpExpertDeviceBytes(idx *GGUFExpertIndex, layer int) (int64, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers {
		return 0, fmt.Errorf("invalid Q4_K gate/up byte request layer=%d", layer)
	}
	le := idx.entries[layer]
	if le.gateUp.QType != gguf.QuantQ4_K {
		return 0, fmt.Errorf("Q4_K gate/up expert prewarm requires Q4_K, got %s", le.gateUp.QType)
	}
	blocks := le.gateUp.InDim / 256
	return int64(le.gateUp.OutDim * blocks * (128 + 8*4 + 8*4)), nil
}

func q4KGateUpExpertResidentBytes(idx *GGUFExpertIndex, layer int) (int64, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers {
		return 0, fmt.Errorf("invalid Q4_K resident byte request layer=%d", layer)
	}
	le := idx.entries[layer]
	if le.gateUp.QType != gguf.QuantQ4_K {
		return 0, fmt.Errorf("Q4_K gate/up expert residency requires Q4_K, got %s", le.gateUp.QType)
	}
	blocks := le.gateUp.InDim / 256
	if diffusionGemmaGGUFGPUExpertRawQ4Enabled() {
		return int64(le.gateUp.OutDim * blocks * 144), nil
	}
	return int64(le.gateUp.OutDim * blocks * (128 + 8*4 + 8*4)), nil
}

func q8DownExpertDeviceBytes(idx *GGUFExpertIndex, layer int) (int64, error) {
	if idx == nil || layer < 0 || layer >= idx.NumLayers {
		return 0, fmt.Errorf("invalid Q8 down byte request layer=%d", layer)
	}
	le := idx.entries[layer]
	if le.down.QType != gguf.QuantQ8_0 {
		return 0, fmt.Errorf("Q8 down expert prewarm requires Q8_0, got %s", le.down.QType)
	}
	blocks := le.down.InDim / 32
	return int64(le.down.OutDim * (le.down.InDim + blocks*4)), nil
}

// PrewarmGGUFGPUPointerExpertCache uploads resident per-expert Q4_K gate/up and
// Q8_0 down buffers used by the fused selected-expert pointer-table kernels. It
// is opt-in via GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_CACHE_MB and stops
// when that budget is exhausted or when free VRAM would fall below
// GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_RESERVE_MB (default 512).
// Pointer tables themselves remain active-set specific and are built lazily
// because selected expert combinations vary.
func PrewarmGGUFGPUPointerExpertCache(idx *GGUFExpertIndex, maxLayers int) (layers int, experts int, bytes int64, err error) {
	if !gpu.SgemmReady() {
		return 0, 0, 0, fmt.Errorf("DiffusionGemma GGUF GPU expert prewarm requires CUDA SGEMM")
	}
	if idx == nil {
		return 0, 0, 0, fmt.Errorf("DiffusionGemma GGUF GPU expert prewarm missing GGUF expert index")
	}
	if diffusionGemmaGGUFGPUExpertCacheBytes() <= 0 || maxLayers == 0 {
		return 0, 0, 0, nil
	}
	if maxLayers < 0 || maxLayers > idx.NumLayers {
		maxLayers = idx.NumLayers
	}
	freeReserve := diffusionGemmaGGUFGPUExpertPrewarmReserveBytes()
	cacheReserve := diffusionGemmaGGUFGPUExpertPrewarmCacheReserveBytes()
	cacheBudget := diffusionGemmaGGUFGPUExpertCacheBytes()
	q4Only := diffusionGemmaGGUFGPUExpertPrewarmQ4OnlyEnabled()
	planOnly := diffusionGemmaGGUFGPUExpertPrewarmPlanOnlyEnabled()
	cacheID := ggufExpertIndexCacheID(idx)
	prewarmed := make(map[ggufExpertPrewarmTarget]bool)
	prewarmOne := func(layer, expert int) (bool, error) {
		need := int64(0)
		if diffusionGemmaGGUFGPUExpertRawQ4Enabled() {
			if _, ok := q4KGateUpExpertRawCache.Load(q4KGateUpExpertKey{index: cacheID, layer: layer, expert: expert}); !ok {
				b, err := q4KGateUpExpertResidentBytes(idx, layer)
				if err != nil {
					return false, err
				}
				need += b
			}
		} else if _, ok := q4KGateUpExpertCache.Load(q4KGateUpExpertKey{index: cacheID, layer: layer, expert: expert}); !ok {
			b, err := q4KGateUpExpertResidentBytes(idx, layer)
			if err != nil {
				return false, err
			}
			need += b
		}
		if !q4Only {
			switch idx.entries[layer].down.QType {
			case gguf.QuantQ8_0:
				if _, ok := q8DownExpertCache.Load(q8DownExpertKey{index: cacheID, layer: layer, expert: expert}); !ok {
					b, err := q8DownExpertDeviceBytes(idx, layer)
					if err != nil {
						return false, err
					}
					need += b
				}
			case gguf.QuantQ5_0:
				if _, ok := q5DownExpertCache.Load(q5DownExpertKey{index: cacheID, layer: layer, expert: expert}); !ok {
					b, err := q5DownExpertDeviceBytes(idx, layer)
					if err != nil {
						return false, err
					}
					need += b
				}
			default:
				return false, fmt.Errorf("unsupported GGUF pointer expert down qtype layer=%d type=%s", layer, idx.entries[layer].down.QType)
			}
		}
		if need > 0 {
			if cacheReserve > 0 {
				used, _ := activeExpertMatrixCacheUsageBytes()
				prewarmLimit := cacheBudget - cacheReserve
				if prewarmLimit <= 0 || used+need > prewarmLimit {
					return false, nil
				}
			}
			if freeReserve > 0 {
				free, _ := gpu.MemInfo()
				if free > 0 && (free <= freeReserve || uint64(need) > free-freeReserve) {
					return false, nil
				}
			}
			if !reserveActiveExpertMatrixCacheBytes(need) {
				return false, nil
			}
		}
		if diffusionGemmaGGUFGPUExpertRawQ4Enabled() {
			if _, err := residentQ4KGateUpExpertRawMatrixWithReservation(idx, layer, expert, false); err != nil {
				if need > 0 {
					releaseActiveExpertMatrixCacheBytes(need)
				}
				return false, err
			}
		} else if _, err := residentQ4KGateUpExpertMatrixWithReservation(idx, layer, expert, false); err != nil {
			if need > 0 {
				releaseActiveExpertMatrixCacheBytes(need)
			}
			return false, err
		}
		if !q4Only {
			switch idx.entries[layer].down.QType {
			case gguf.QuantQ8_0:
				if _, err := residentQ8DownExpertMatrixWithReservation(idx, layer, expert, false); err != nil {
					if need > 0 {
						releaseActiveExpertMatrixCacheBytes(need)
					}
					return false, err
				}
			case gguf.QuantQ5_0:
				if _, err := residentQ5DownExpertMatrixWithReservation(idx, layer, expert, false); err != nil {
					if need > 0 {
						releaseActiveExpertMatrixCacheBytes(need)
					}
					return false, err
				}
			}
		}
		bytes += need
		experts++
		prewarmed[ggufExpertPrewarmTarget{Layer: layer, Expert: expert}] = true
		return true, nil
	}
	plan := diffusionGemmaGGUFGPUExpertPrewarmPlan(maxLayers, idx.NumExperts)
	for _, target := range plan {
		ok, err := prewarmOne(target.Layer, target.Expert)
		if err != nil {
			return layers, experts, bytes, err
		}
		if !ok {
			return layers, experts, bytes, nil
		}
	}
	if planOnly && len(plan) > 0 {
		return layers, experts, bytes, nil
	}
	for layer := 0; layer < maxLayers; layer++ {
		layerComplete := true
		for expert := 0; expert < idx.NumExperts; expert++ {
			if prewarmed[ggufExpertPrewarmTarget{Layer: layer, Expert: expert}] {
				continue
			}
			ok, err := prewarmOne(layer, expert)
			if err != nil {
				return layers, experts, bytes, err
			}
			if !ok {
				layerComplete = false
				break
			}
		}
		if !layerComplete {
			break
		}
		layers++
	}
	return layers, experts, bytes, nil
}

// PrewarmGGUFDenseTransposeCache builds the bounded host-side transposed dense
// cache for non-resident GGUF layers. It does not allocate persistent VRAM; the
// temporary dense upload path still streams these cached transposes through a
// reusable device buffer. residentPrefix follows GPUDispatcher semantics:
// prefix<=0 means all dense layers are resident, so there is nothing to prewarm.
func PrewarmGGUFDenseTransposeCache(weights *TextWeights, residentPrefix int) (matrices int, bytes int64, err error) {
	if weights == nil {
		return 0, 0, fmt.Errorf("DiffusionGemma GGUF dense transpose prewarm missing weights")
	}
	if diffusionGemmaGGUFDenseTransposeCacheBytes() <= 0 {
		return 0, 0, nil
	}
	fp := weights.ForwardPlan()
	if !fp.Ready {
		return 0, 0, fmt.Errorf("DiffusionGemma GGUF dense transpose prewarm text plan not ready: %v", fp.Missing)
	}
	if residentPrefix <= 0 {
		return 0, 0, nil
	}
	if residentPrefix > len(fp.Layers) {
		residentPrefix = len(fp.Layers)
	}
	cacheMatrix := func(w []float32, rows, cols int) error {
		data, hit, err := cachedTransposedF32MatrixNoEvict(w, rows, cols)
		if err != nil {
			return err
		}
		if data != nil && !hit {
			matrices++
			bytes += int64(len(data)) * 4
		}
		return nil
	}
	for i := residentPrefix; i < len(fp.Layers); i++ {
		lb := fp.Layers[i]
		if lb.QProj == nil || lb.KProj == nil || lb.OProj == nil {
			return matrices, bytes, fmt.Errorf("DiffusionGemma GGUF dense transpose prewarm layer %d missing attention bindings", i)
		}
		qW, qRows, qCols, err := loadFloatMatrix(weights, lb.QProj)
		if err != nil {
			return matrices, bytes, err
		}
		kW, kRows, kCols, err := loadFloatMatrix(weights, lb.KProj)
		if err != nil {
			return matrices, bytes, err
		}
		vRows, vCols := kRows, qCols
		var vW []float32
		if lb.VProj != nil {
			vW, vRows, vCols, err = loadFloatMatrix(weights, lb.VProj)
			if err != nil {
				return matrices, bytes, err
			}
		}
		oW, oRows, oCols, err := loadFloatMatrix(weights, lb.OProj)
		if err != nil {
			return matrices, bytes, err
		}
		hidden := qCols
		if hidden <= 0 || kCols != hidden || vCols != hidden || oRows != hidden || oCols != qRows {
			return matrices, bytes, fmt.Errorf("DiffusionGemma GGUF dense transpose prewarm attention shape mismatch layer=%d", i)
		}
		if err := cacheMatrix(qW, qRows, hidden); err != nil {
			return matrices, bytes, err
		}
		if err := cacheMatrix(kW, kRows, hidden); err != nil {
			return matrices, bytes, err
		}
		if lb.VProj != nil {
			if err := cacheMatrix(vW, vRows, hidden); err != nil {
				return matrices, bytes, err
			}
		}
		if err := cacheMatrix(oW, hidden, qRows); err != nil {
			return matrices, bytes, err
		}
	}
	for i := residentPrefix; i < len(fp.Layers); i++ {
		lb := fp.Layers[i]
		if lb.MLPGateProj == nil || lb.MLPUpProj == nil || lb.MLPDownProj == nil || lb.QProj == nil {
			return matrices, bytes, fmt.Errorf("DiffusionGemma GGUF dense transpose prewarm layer %d missing MLP bindings", i)
		}
		_, _, hidden, err := loadFloatMatrix(weights, lb.QProj)
		if err != nil {
			return matrices, bytes, err
		}
		gateW, intermediate, gateCols, err := loadFloatMatrix(weights, lb.MLPGateProj)
		if err != nil {
			return matrices, bytes, err
		}
		upW, upRows, upCols, err := loadFloatMatrix(weights, lb.MLPUpProj)
		if err != nil {
			return matrices, bytes, err
		}
		downW, downRows, downCols, err := loadFloatMatrix(weights, lb.MLPDownProj)
		if err != nil {
			return matrices, bytes, err
		}
		if intermediate <= 0 || gateCols != hidden || upRows != intermediate || upCols != hidden || downRows != hidden || downCols != intermediate {
			return matrices, bytes, fmt.Errorf("DiffusionGemma GGUF dense transpose prewarm MLP shape mismatch layer=%d", i)
		}
		if err := cacheMatrix(gateW, intermediate, hidden); err != nil {
			return matrices, bytes, err
		}
		if err := cacheMatrix(upW, intermediate, hidden); err != nil {
			return matrices, bytes, err
		}
		if err := cacheMatrix(downW, hidden, intermediate); err != nil {
			return matrices, bytes, err
		}
	}
	return matrices, bytes, nil
}

// PrewarmGGUFGPUWeightCache uploads dense GGUF/F32 projection weights once for
// all text layers: attention Q/K/V/O and dense MLP gate/up/down. It is intended
// to run before inference so generation/prefill reuse resident device buffers
// instead of uploading weights per GEMM.
func PrewarmGGUFGPUWeightCache(weights *TextWeights) (layers int, bytes int64, err error) {
	return PrewarmGGUFGPUWeightCacheLayers(weights, 0)
}

// PrewarmGGUFGPUWeightCacheLayers is the bounded-prefix form of
// PrewarmGGUFGPUWeightCache. maxLayers<=0 preserves legacy all-layer behavior.
func PrewarmGGUFGPUWeightCacheLayers(weights *TextWeights, maxLayers int) (layers int, bytes int64, err error) {
	if !gpu.SgemmReady() {
		return 0, 0, fmt.Errorf("DiffusionGemma GGUF GPU prewarm requires CUDA SGEMM")
	}
	if weights == nil {
		return 0, 0, fmt.Errorf("DiffusionGemma GGUF GPU prewarm missing weights")
	}
	fp := weights.ForwardPlan()
	if !fp.Ready {
		return 0, 0, fmt.Errorf("DiffusionGemma GGUF GPU prewarm text plan not ready: %v", fp.Missing)
	}
	layerLimit := len(fp.Layers)
	if maxLayers > 0 && maxLayers < layerLimit {
		layerLimit = maxLayers
	}
	for i, lb := range fp.Layers[:layerLimit] {
		if lb.QProj == nil || lb.KProj == nil || lb.OProj == nil || lb.MLPGateProj == nil || lb.MLPUpProj == nil || lb.MLPDownProj == nil {
			return layers, bytes, fmt.Errorf("DiffusionGemma GGUF GPU prewarm layer %d missing dense bindings", i)
		}
		qW, qRows, qCols, err := loadFloatMatrix(weights, lb.QProj)
		if err != nil {
			return layers, bytes, err
		}
		kW, kRows, kCols, err := loadFloatMatrix(weights, lb.KProj)
		if err != nil {
			return layers, bytes, err
		}
		vRows, vCols := kRows, qCols
		var vW []float32
		if lb.VProj != nil {
			vW, vRows, vCols, err = loadFloatMatrix(weights, lb.VProj)
			if err != nil {
				return layers, bytes, err
			}
		}
		oW, oRows, oCols, err := loadFloatMatrix(weights, lb.OProj)
		if err != nil {
			return layers, bytes, err
		}
		hidden := qCols
		if hidden <= 0 || kCols != hidden || vCols != hidden || oRows != hidden || oCols != qRows {
			return layers, bytes, fmt.Errorf("DiffusionGemma GGUF GPU prewarm attention shape mismatch layer=%d", i)
		}
		if _, err := residentGGUFGPUAttentionWeights(i, lb, qW, kW, vW, oW, qRows, kRows, vRows, hidden); err != nil {
			return layers, bytes, err
		}
		gateW, intermediate, gateCols, err := loadFloatMatrix(weights, lb.MLPGateProj)
		if err != nil {
			return layers, bytes, err
		}
		upW, upRows, upCols, err := loadFloatMatrix(weights, lb.MLPUpProj)
		if err != nil {
			return layers, bytes, err
		}
		downW, downRows, downCols, err := loadFloatMatrix(weights, lb.MLPDownProj)
		if err != nil {
			return layers, bytes, err
		}
		if intermediate <= 0 || gateCols != hidden || upRows != intermediate || upCols != hidden || downRows != hidden || downCols != intermediate {
			return layers, bytes, fmt.Errorf("DiffusionGemma GGUF GPU prewarm MLP shape mismatch layer=%d", i)
		}
		if _, err := residentGGUFGPUMLPWeights(i, lb, gateW, upW, downW, hidden, intermediate); err != nil {
			return layers, bytes, err
		}
		attnElems := qRows*hidden + kRows*hidden + hidden*qRows
		if lb.VProj != nil {
			attnElems += vRows * hidden
		}
		mlpElems := intermediate*hidden*2 + hidden*intermediate
		bytes += int64(attnElems+mlpElems) * 4
		layers++
	}
	return layers, bytes, nil
}

// batchedGPUGemm computes Out[M,N] = W[M,K] × X_T[K,N] where
// hidden is [N,K] (N positions, K=hiddenSize) stored row-major.
// Returns Out as [M,N] row-major (M output features, N positions).
func batchedGPUGemm(W []float32, hidden []float32, M, K, N int) ([]float32, error) {
	kn, okKN := checked.MulInt(K, N)
	nk, okNK := checked.MulInt(N, K)
	mn, okMN := checked.MulInt(M, N)
	if M <= 0 || K <= 0 || N <= 0 || !okKN || !okNK || !okMN || len(hidden) < nk {
		return nil, fmt.Errorf("invalid batched GPU GEMM buffers M=%d K=%d N=%d hidden=%d/%d", M, K, N, len(hidden), nk)
	}
	// Transpose hidden [N,K] → X_T [K,N]
	xt := make([]float32, kn)
	for pos := 0; pos < N; pos++ {
		for k := 0; k < K; k++ {
			xt[k*N+pos] = hidden[pos*K+k]
		}
	}
	wBuf, err := residentF32MatrixBuffer(W, M, K)
	if err != nil {
		return nil, err
	}
	xBuf, err := gpu.Malloc(kn)
	if err != nil {
		return nil, err
	}
	defer xBuf.Free()
	outBuf, err := gpu.Malloc(mn)
	if err != nil {
		return nil, err
	}
	defer outBuf.Free()
	if err := xBuf.Upload(xt); err != nil {
		return nil, err
	}
	if err := gpu.Sgemm(M, N, K, 1.0, wBuf, xBuf, outBuf); err != nil {
		return nil, err
	}
	out := make([]float32, mn)
	if err := outBuf.Download(out); err != nil {
		return nil, err
	}
	return out, nil
}

// scatterGemmResult copies GEMM output [M,N] back into per-position slices.
func batchedGPUGemm2Transposed(outA, outB, hidden []float32, batch, outADim, inADim int, wtA *gpu.Buffer, outBDim, inBDim int, wtB *gpu.Buffer) error {
	if inADim != inBDim {
		return fmt.Errorf("resident SGEMM pair input mismatch A=%d B=%d", inADim, inBDim)
	}
	inLen, okIn := checked.MulInt(batch, inADim)
	outALen, okOutA := checked.MulInt(batch, outADim)
	outBLen, okOutB := checked.MulInt(batch, outBDim)
	wtALen, okWtA := checked.MulInt(inADim, outADim)
	wtBLen, okWtB := checked.MulInt(inBDim, outBDim)
	if batch <= 0 || outADim <= 0 || outBDim <= 0 || inADim <= 0 || !okIn || !okOutA || !okOutB || !okWtA || !okWtB || len(hidden) < inLen || len(outA) < outALen || len(outB) < outBLen || wtA == nil || wtB == nil || wtA.Ptr == 0 || wtB.Ptr == 0 || wtA.Size < wtALen*4 || wtB.Size < wtBLen*4 {
		return fmt.Errorf("invalid resident SGEMM pair buffers batch=%d A=[%d,%d] B=[%d,%d]", batch, outADim, inADim, outBDim, inBDim)
	}
	xBuf, err := gpu.Malloc(inLen)
	if err != nil {
		return err
	}
	defer xBuf.Free()
	outABuf, err := gpu.Malloc(outALen)
	if err != nil {
		return err
	}
	defer outABuf.Free()
	outBBuf, err := gpu.Malloc(outBLen)
	if err != nil {
		return err
	}
	defer outBBuf.Free()
	if err := xBuf.Upload(hidden[:inLen]); err != nil {
		return err
	}
	if err := gpu.Sgemm(batch, outADim, inADim, 1, xBuf, wtA, outABuf); err != nil {
		return err
	}
	if err := gpu.Sgemm(batch, outBDim, inBDim, 1, xBuf, wtB, outBBuf); err != nil {
		return err
	}
	if err := outABuf.Download(outA[:outALen]); err != nil {
		return err
	}
	return outBBuf.Download(outB[:outBLen])
}

var residentGemmScratch = struct {
	sync.Mutex
	x    *gpu.Buffer
	y    *gpu.Buffer
	xLen int
	yLen int
}{}

func residentGemmScratchBuffers(inLen, outLen int) (*gpu.Buffer, *gpu.Buffer, func(), error) {
	residentGemmScratch.Lock()
	unlock := func() { residentGemmScratch.Unlock() }
	ensure := func(cur **gpu.Buffer, curLen *int, need int, label string) error {
		if need <= 0 {
			return fmt.Errorf("invalid resident GEMM %s size %d", label, need)
		}
		if *cur != nil && *curLen >= need {
			return nil
		}
		if *cur != nil {
			(*cur).Free()
			*cur = nil
			*curLen = 0
		}
		buf, err := gpu.Malloc(need)
		if err != nil {
			return fmt.Errorf("alloc resident GEMM %s scratch: %w", label, err)
		}
		*cur = buf
		*curLen = need
		return nil
	}
	if err := ensure(&residentGemmScratch.x, &residentGemmScratch.xLen, inLen, "input"); err != nil {
		unlock()
		return nil, nil, nil, err
	}
	if err := ensure(&residentGemmScratch.y, &residentGemmScratch.yLen, outLen, "output"); err != nil {
		unlock()
		return nil, nil, nil, err
	}
	return residentGemmScratch.x, residentGemmScratch.y, unlock, nil
}

func batchedGPUGemmTransposed(out, hidden []float32, batch, outDim, inDim int, wt *gpu.Buffer) error {
	inLen, okIn := checked.MulInt(batch, inDim)
	outLen, okOut := checked.MulInt(batch, outDim)
	wtLen, okWt := checked.MulInt(inDim, outDim)
	if batch <= 0 || outDim <= 0 || inDim <= 0 || !okIn || !okOut || !okWt || len(hidden) < inLen || len(out) < outLen || wt == nil || wt.Ptr == 0 || wt.Size < wtLen*4 {
		return fmt.Errorf("invalid resident SGEMM buffers batch=%d out=%d in=%d hidden=%d/%d outbuf=%d/%d", batch, outDim, inDim, len(hidden), inLen, len(out), outLen)
	}
	xBuf, outBuf, unlock, err := residentGemmScratchBuffers(inLen, outLen)
	if err != nil {
		return err
	}
	defer unlock()
	if err := xBuf.Upload(hidden[:inLen]); err != nil {
		return err
	}
	if err := gpu.Sgemm(batch, outDim, inDim, 1, xBuf, wt, outBuf); err != nil {
		return err
	}
	return outBuf.Download(out[:outLen])
}

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
	if vocab <= 0 || hiddenSize <= 0 || len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("LM head embed invalid shape [%d,%d] hidden_len=%d", vocab, hiddenSize, len(scratch.Hidden))
	}
	needElems, okElems := checked.MulInt(vocab, hiddenSize)
	needBytes, okBytes := checked.MulInt(needElems, 2)
	if !okElems || !okBytes || len(raw) < needBytes {
		return fmt.Errorf("LM head embed raw BF16 bytes=%d want at least %d for shape [%d,%d]", len(raw), needBytes, vocab, hiddenSize)
	}
	positions := len(scratch.Hidden) / hiddenSize
	if positions <= 0 {
		return nil
	}
	bf16Embed := unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), needElems)
	topK := scratch.LMHeadTopK
	if topK > vocab {
		topK = vocab
	}
	if topK <= 0 {
		return fmt.Errorf("sparse LM head fallback requires LMHeadTopK > 0")
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
	applyFinalLogitSoftcapping(scratch, positions, vocab)
	return nil
}

func UploadTiedLMHeadTransposeBuffer(weights *TextWeights) (*gpu.Buffer, int, int, error) {
	if weights == nil {
		return nil, 0, 0, fmt.Errorf("nil weights for tied LM head")
	}
	if !gpu.SgemmReady() {
		return nil, 0, 0, fmt.Errorf("GPU SGEMM not available for tied LM head upload")
	}
	fp := weights.ForwardPlan()
	if fp.Globals.EmbedTokens == nil || len(fp.Globals.EmbedTokens.Shape) != 2 {
		return nil, 0, 0, fmt.Errorf("missing embed_tokens for tied LM head")
	}
	vocab, hidden := fp.Globals.EmbedTokens.Shape[0], fp.Globals.EmbedTokens.Shape[1]
	if vocab <= 0 || hidden <= 0 {
		return nil, 0, 0, fmt.Errorf("invalid tied LM head embedding shape [%d,%d]", vocab, hidden)
	}
	t, err := weights.CachedFloatTensor(fp.Globals.EmbedTokens.Name)
	if err != nil {
		return nil, 0, 0, err
	}
	elems, ok := checked.MulInt(vocab, hidden)
	if !ok || len(t.Shape) != 2 || t.Shape[0] != vocab || t.Shape[1] != hidden || len(t.Data) < elems {
		return nil, 0, 0, fmt.Errorf("tied LM head cache shape %v len=%d want [%d,%d]", t.Shape, len(t.Data), vocab, hidden)
	}
	bytes, okBytes := checked.MulInt(elems, 4)
	// Keep this conservative until the GGUF LM-head island is chunked/streamed;
	// a full DiffusionGemma tied embedding transpose is ~2.95GB and can exceed
	// practical per-allocation/runtime limits on the current CUDA wrapper.
	const maxTiedLMHeadUploadBytes = 2_500_000_000
	if !okBytes || bytes > maxTiedLMHeadUploadBytes {
		return nil, 0, 0, fmt.Errorf("tied LM head upload too large: %d bytes (limit %d)", bytes, maxTiedLMHeadUploadBytes)
	}
	transposed := make([]float32, elems)
	for v := 0; v < vocab; v++ {
		row := t.Data[v*hidden : (v+1)*hidden]
		for h := 0; h < hidden; h++ {
			transposed[h*vocab+v] = row[h]
		}
	}
	buf, err := gpu.Malloc(elems)
	if err != nil {
		return nil, 0, 0, err
	}
	if err := buf.Upload(transposed); err != nil {
		buf.Free()
		return nil, 0, 0, err
	}
	return buf, vocab, hidden, nil
}

func UploadSelfConditioningEmbeddingBuffer(weights *TextWeights) (*gpu.Buffer, int, int, error) {
	if weights == nil {
		return nil, 0, 0, fmt.Errorf("nil weights for self-conditioning embedding")
	}
	fp := weights.ForwardPlan()
	if fp.Globals.EmbedTokens == nil || len(fp.Globals.EmbedTokens.Shape) != 2 {
		return nil, 0, 0, fmt.Errorf("missing embed_tokens for self-conditioning embedding")
	}
	vocab, hidden := fp.Globals.EmbedTokens.Shape[0], fp.Globals.EmbedTokens.Shape[1]
	if vocab <= 0 || hidden <= 0 {
		return nil, 0, 0, fmt.Errorf("invalid self-conditioning embedding shape [%d,%d]", vocab, hidden)
	}
	t, err := weights.CachedFloatTensor(fp.Globals.EmbedTokens.Name)
	if err != nil {
		return nil, 0, 0, err
	}
	embedElems, ok := checked.MulInt(vocab, hidden)
	if !ok || len(t.Shape) != 2 || t.Shape[0] != vocab || t.Shape[1] != hidden || len(t.Data) < embedElems {
		return nil, 0, 0, fmt.Errorf("self-conditioning embedding cache shape %v len=%d want [%d,%d]", t.Shape, len(t.Data), vocab, hidden)
	}
	buf, err := gpu.Malloc(embedElems)
	if err != nil {
		return nil, 0, 0, err
	}
	if err := buf.Upload(t.Data[:embedElems]); err != nil {
		buf.Free()
		return nil, 0, 0, err
	}
	return buf, vocab, hidden, nil
}

func buildSelfConditioningFromLogitsGPU(weights *TextWeights, scratch ForwardScratch, embed *gpu.Buffer, embedVocab, embedHidden int) ([]float32, error) {
	if weights == nil || embed == nil || len(scratch.Logits) == 0 {
		return nil, nil
	}
	fp := weights.ForwardPlan()
	if fp.Globals.EmbedTokens == nil || len(fp.Globals.EmbedTokens.Shape) != 2 {
		return nil, nil
	}
	vocab, hidden := fp.Globals.EmbedTokens.Shape[0], fp.Globals.EmbedTokens.Shape[1]
	positions := len(scratch.Logits)
	if vocab <= 0 || hidden <= 0 || embedVocab != vocab || embedHidden != hidden {
		return nil, fmt.Errorf("GPU self-conditioning embedding shape [%d,%d] want [%d,%d]", embedVocab, embedHidden, vocab, hidden)
	}
	probLen, okProb := checked.MulInt(positions, vocab)
	outLen, okOut := checked.MulInt(positions, hidden)
	if !okProb || !okOut {
		return nil, fmt.Errorf("GPU self-conditioning size overflow positions=%d vocab=%d hidden=%d", positions, vocab, hidden)
	}
	probs := make([]float32, probLen)
	tempInv := scratch.SCTempInv
	if tempInv <= 0 {
		tempInv = 1
	}
	for pos := 0; pos < positions; pos++ {
		if len(scratch.Logits[pos]) < vocab {
			return nil, fmt.Errorf("GPU self-conditioning logits row=%d len=%d want %d", pos, len(scratch.Logits[pos]), vocab)
		}
		row := scratch.Logits[pos][:vocab]
		maxVal, sumExp, ok := selfConditioningSoftmaxStats(row, tempInv)
		if !ok {
			continue
		}
		invSum := 1.0 / sumExp
		dst := probs[pos*vocab : (pos+1)*vocab]
		for i, v := range row {
			if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
				continue
			}
			dst[i] = float32(math.Exp(float64(v*tempInv-maxVal)) * invSum)
		}
	}
	probBuf, err := gpu.Malloc(probLen)
	if err != nil {
		return nil, err
	}
	defer probBuf.Free()
	outBuf, err := gpu.Malloc(outLen)
	if err != nil {
		return nil, err
	}
	defer outBuf.Free()
	if err := probBuf.Upload(probs); err != nil {
		return nil, err
	}
	if err := gpu.Sgemm(positions, hidden, vocab, 1, probBuf, embed, outBuf); err != nil {
		return nil, err
	}
	out := make([]float32, outLen)
	if err := outBuf.Download(out); err != nil {
		return nil, err
	}
	scale := float32(math.Sqrt(float64(hidden)))
	for i := range out {
		out[i] *= scale
	}
	return out, nil
}

var ggufChunkedLMHeadScratch = struct {
	sync.Mutex
	x             *gpu.Buffer
	wt            *gpu.Buffer
	out           *gpu.Buffer
	xN            int
	wtN           int
	outN          int
	wtHost        []float32
	outHost       []float32
	cachePtr      uintptr
	cacheVocab    int
	cacheHidden   int
	cacheChunk    int
	cacheF32Chunk [][]float32
}{}

func ggufChunkedLMHeadScratchBuffers(xLen, wtLen, outLen int) (*gpu.Buffer, *gpu.Buffer, *gpu.Buffer, []float32, []float32, func(), error) {
	ggufChunkedLMHeadScratch.Lock()
	unlock := func() { ggufChunkedLMHeadScratch.Unlock() }
	ensure := func(cur **gpu.Buffer, curN *int, need int, label string) error {
		if need <= 0 {
			return fmt.Errorf("GGUF chunked LM head invalid %s size %d", label, need)
		}
		if *cur != nil && *curN >= need {
			return nil
		}
		if *cur != nil {
			(*cur).Free()
			*cur = nil
			*curN = 0
		}
		buf, err := gpu.Malloc(need)
		if err != nil {
			return fmt.Errorf("alloc GGUF chunked LM head %s scratch: %w", label, err)
		}
		*cur = buf
		*curN = need
		return nil
	}
	if err := ensure(&ggufChunkedLMHeadScratch.x, &ggufChunkedLMHeadScratch.xN, xLen, "hidden"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, err
	}
	if err := ensure(&ggufChunkedLMHeadScratch.wt, &ggufChunkedLMHeadScratch.wtN, wtLen, "weight"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, err
	}
	if err := ensure(&ggufChunkedLMHeadScratch.out, &ggufChunkedLMHeadScratch.outN, outLen, "output"); err != nil {
		unlock()
		return nil, nil, nil, nil, nil, nil, err
	}
	if len(ggufChunkedLMHeadScratch.wtHost) < wtLen {
		ggufChunkedLMHeadScratch.wtHost = make([]float32, wtLen)
	}
	if len(ggufChunkedLMHeadScratch.outHost) < outLen {
		ggufChunkedLMHeadScratch.outHost = make([]float32, outLen)
	}
	return ggufChunkedLMHeadScratch.x, ggufChunkedLMHeadScratch.wt, ggufChunkedLMHeadScratch.out, ggufChunkedLMHeadScratch.wtHost[:wtLen], ggufChunkedLMHeadScratch.outHost[:outLen], unlock, nil
}

func GGUFChunkedLMHeadScratchStats() (cachedChunks int, cachedBytes int64) {
	ggufChunkedLMHeadScratch.Lock()
	defer ggufChunkedLMHeadScratch.Unlock()
	for _, chunk := range ggufChunkedLMHeadScratch.cacheF32Chunk {
		if len(chunk) > 0 {
			cachedChunks++
			cachedBytes += int64(len(chunk)) * 4
		}
	}
	return cachedChunks, cachedBytes
}

func FreeGGUFChunkedLMHeadScratch() {
	ggufChunkedLMHeadScratch.Lock()
	defer ggufChunkedLMHeadScratch.Unlock()
	if ggufChunkedLMHeadScratch.x != nil {
		ggufChunkedLMHeadScratch.x.Free()
		ggufChunkedLMHeadScratch.x = nil
	}
	if ggufChunkedLMHeadScratch.wt != nil {
		ggufChunkedLMHeadScratch.wt.Free()
		ggufChunkedLMHeadScratch.wt = nil
	}
	if ggufChunkedLMHeadScratch.out != nil {
		ggufChunkedLMHeadScratch.out.Free()
		ggufChunkedLMHeadScratch.out = nil
	}
	ggufChunkedLMHeadScratch.xN = 0
	ggufChunkedLMHeadScratch.wtN = 0
	ggufChunkedLMHeadScratch.outN = 0
	ggufChunkedLMHeadScratch.wtHost = nil
	ggufChunkedLMHeadScratch.outHost = nil
	ggufChunkedLMHeadScratch.cachePtr = 0
	ggufChunkedLMHeadScratch.cacheVocab = 0
	ggufChunkedLMHeadScratch.cacheHidden = 0
	ggufChunkedLMHeadScratch.cacheChunk = 0
	ggufChunkedLMHeadScratch.cacheF32Chunk = nil
}

func cachedF32LMHeadChunk(embed []float32, vocab, hidden, chunkSize, chunkIndex, start, end int) []float32 {
	if len(embed) == 0 {
		return nil
	}
	ptr := uintptr(unsafe.Pointer(&embed[0]))
	nChunks := (vocab + chunkSize - 1) / chunkSize
	if ggufChunkedLMHeadScratch.cachePtr != ptr || ggufChunkedLMHeadScratch.cacheVocab != vocab || ggufChunkedLMHeadScratch.cacheHidden != hidden || ggufChunkedLMHeadScratch.cacheChunk != chunkSize || len(ggufChunkedLMHeadScratch.cacheF32Chunk) != nChunks {
		ggufChunkedLMHeadScratch.cachePtr = ptr
		ggufChunkedLMHeadScratch.cacheVocab = vocab
		ggufChunkedLMHeadScratch.cacheHidden = hidden
		ggufChunkedLMHeadScratch.cacheChunk = chunkSize
		ggufChunkedLMHeadScratch.cacheF32Chunk = make([][]float32, nChunks)
	}
	if ggufChunkedLMHeadScratch.cacheF32Chunk[chunkIndex] != nil {
		return ggufChunkedLMHeadScratch.cacheF32Chunk[chunkIndex]
	}
	chunk := end - start
	wtLen := hidden * chunk
	wt := make([]float32, wtLen)
	for v := start; v < end; v++ {
		row := embed[v*hidden : (v+1)*hidden]
		col := v - start
		for h := 0; h < hidden; h++ {
			wt[h*chunk+col] = row[h]
		}
	}
	ggufChunkedLMHeadScratch.cacheF32Chunk[chunkIndex] = wt
	return wt
}

// PrewarmGGUFF32LMHeadChunks builds the host-side transposed F32 LM-head chunk
// cache before inference. The GPU path still streams chunks through a reusable
// device weight buffer, but this removes first-step host transpose/cache cost
// from the inference timeline.
func PrewarmGGUFF32LMHeadChunks(weights *TextWeights, chunkSize int) (chunks int, bytes int64, err error) {
	if weights == nil {
		return 0, 0, fmt.Errorf("DiffusionGemma GGUF LM-head chunk prewarm missing weights")
	}
	if chunkSize <= 0 {
		return 0, 0, fmt.Errorf("DiffusionGemma GGUF LM-head chunk prewarm invalid chunk size %d", chunkSize)
	}
	fp := weights.ForwardPlan()
	if fp.Globals.EmbedTokens == nil || len(fp.Globals.EmbedTokens.Shape) != 2 {
		return 0, 0, fmt.Errorf("DiffusionGemma GGUF LM-head chunk prewarm missing embed_tokens")
	}
	vocab, hidden := fp.Globals.EmbedTokens.Shape[0], fp.Globals.EmbedTokens.Shape[1]
	embed, err := weights.CachedFloatTensor(fp.Globals.EmbedTokens.Name)
	if err != nil {
		return 0, 0, err
	}
	elems, ok := checked.MulInt(vocab, hidden)
	if vocab <= 0 || hidden <= 0 || !ok || len(embed.Data) < elems {
		return 0, 0, fmt.Errorf("DiffusionGemma GGUF LM-head chunk prewarm shape %v len=%d", embed.Shape, len(embed.Data))
	}
	if chunkSize > vocab {
		chunkSize = vocab
	}
	ggufChunkedLMHeadScratch.Lock()
	defer ggufChunkedLMHeadScratch.Unlock()
	for chunkIndex, start := 0, 0; start < vocab; chunkIndex, start = chunkIndex+1, start+chunkSize {
		end := start + chunkSize
		if end > vocab {
			end = vocab
		}
		chunk := cachedF32LMHeadChunk(embed.Data, vocab, hidden, chunkSize, chunkIndex, start, end)
		bytes += int64(len(chunk)) * 4
		chunks++
	}
	return chunks, bytes, nil
}

func runChunkedF32GPULMHead(weights *TextWeights, scratch ForwardScratch, hiddenSize, chunkSize int, useF32Cache bool) error {
	if weights == nil {
		return fmt.Errorf("chunked F32 GPU LM head missing weights")
	}
	if chunkSize <= 0 {
		return fmt.Errorf("chunked F32 GPU LM head invalid chunk size %d", chunkSize)
	}
	fp := weights.ForwardPlan()
	if fp.Globals.EmbedTokens == nil || len(fp.Globals.EmbedTokens.Shape) != 2 {
		return fmt.Errorf("chunked F32 GPU LM head missing embed_tokens")
	}
	vocab, hidden := fp.Globals.EmbedTokens.Shape[0], fp.Globals.EmbedTokens.Shape[1]
	if vocab <= 0 || hidden <= 0 || hidden != hiddenSize || len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("chunked F32 GPU LM head shape mismatch vocab=%d hidden=%d want_hidden=%d hidden_len=%d", vocab, hidden, hiddenSize, len(scratch.Hidden))
	}
	positions := len(scratch.Hidden) / hiddenSize
	if len(scratch.Logits) < positions {
		return fmt.Errorf("chunked F32 GPU LM head logits rows=%d want %d", len(scratch.Logits), positions)
	}
	hidLen, okHid := checked.MulInt(positions, hiddenSize)
	if positions <= 0 || !okHid {
		return fmt.Errorf("chunked F32 GPU LM head invalid hidden size positions=%d hidden=%d", positions, hiddenSize)
	}
	var embed FloatTensor
	var useQuantRows bool
	if qm := weights.ggufTokenEmbd; !useF32Cache && qm != nil && qm.OutDim == vocab && qm.InDim == hidden {
		useQuantRows = true
	} else {
		var err error
		embed, err = weights.CachedFloatTensor(fp.Globals.EmbedTokens.Name)
		if err != nil {
			return err
		}
		elems, okElems := checked.MulInt(vocab, hidden)
		if !okElems || len(embed.Shape) != 2 || embed.Shape[0] != vocab || embed.Shape[1] != hidden || len(embed.Data) < elems {
			return fmt.Errorf("chunked F32 GPU LM head embed shape %v len=%d want [%d,%d]", embed.Shape, len(embed.Data), vocab, hidden)
		}
	}
	if chunkSize > vocab {
		chunkSize = vocab
	}
	maxWtLen, okWtMax := checked.MulInt(hidden, chunkSize)
	maxOutLen, okOutMax := checked.MulInt(positions, chunkSize)
	if !okWtMax || !okOutMax {
		return fmt.Errorf("chunked F32 GPU LM head buffer overflow hidden=%d positions=%d chunk=%d", hidden, positions, chunkSize)
	}
	xBuf, wtBuf, outBuf, wt, chunkOut, unlock, err := ggufChunkedLMHeadScratchBuffers(hidLen, maxWtLen, maxOutLen)
	if err != nil {
		return err
	}
	defer unlock()
	if err := xBuf.Upload(scratch.Hidden[:hidLen]); err != nil {
		return err
	}
	ggufChunkedLMHeadCounters.calls.Add(1)
	row := make([]float32, hidden)
	for chunkIndex, start := 0, 0; start < vocab; chunkIndex, start = chunkIndex+1, start+chunkSize {
		end := start + chunkSize
		if end > vocab {
			end = vocab
		}
		chunk := end - start
		wtLen, okWt := checked.MulInt(hidden, chunk)
		outLen, okOut := checked.MulInt(positions, chunk)
		if !okWt || !okOut {
			return fmt.Errorf("chunked F32 GPU LM head chunk overflow hidden=%d chunk=%d positions=%d", hidden, chunk, positions)
		}
		prepareStart := time.Now()
		wtChunk := wt[:wtLen]
		if useF32Cache {
			cached := cachedF32LMHeadChunk(embed.Data, vocab, hidden, chunkSize, chunkIndex, start, end)
			if len(cached) != wtLen {
				return fmt.Errorf("chunked F32 GPU LM head cached chunk len=%d want %d", len(cached), wtLen)
			}
			wtChunk = cached
		} else {
			for v := start; v < end; v++ {
				var rowData []float32
				if useQuantRows {
					if err := weights.ggufTokenEmbd.DequantRowTo(row, v); err != nil {
						return err
					}
					rowData = row
				} else {
					rowData = embed.Data[v*hidden : (v+1)*hidden]
				}
				col := v - start
				for h := 0; h < hidden; h++ {
					wtChunk[h*chunk+col] = rowData[h]
				}
			}
		}
		ggufChunkedLMHeadCounters.prepareNS.Add(uint64(time.Since(prepareStart).Nanoseconds()))
		uploadStart := time.Now()
		if err := wtBuf.Upload(wtChunk); err != nil {
			return err
		}
		ggufChunkedLMHeadCounters.uploadNS.Add(uint64(time.Since(uploadStart).Nanoseconds()))
		ggufChunkedLMHeadCounters.bytes.Add(uint64(wtLen * 4))
		ggufChunkedLMHeadCounters.chunks.Add(1)
		sgemmStart := time.Now()
		if err := gpu.Sgemm(positions, chunk, hidden, 1, xBuf, wtBuf, outBuf); err != nil {
			return fmt.Errorf("chunked F32 GPU LM head SGEMM chunk [%d,%d): %w", start, end, err)
		}
		ggufChunkedLMHeadCounters.sgemmNS.Add(uint64(time.Since(sgemmStart).Nanoseconds()))
		chunkOutSlice := chunkOut[:outLen]
		downloadStart := time.Now()
		if err := outBuf.Download(chunkOutSlice); err != nil {
			return err
		}
		ggufChunkedLMHeadCounters.downloadNS.Add(uint64(time.Since(downloadStart).Nanoseconds()))
		copyStart := time.Now()
		for pos := 0; pos < positions; pos++ {
			if len(scratch.Logits[pos]) < vocab {
				return fmt.Errorf("chunked F32 GPU LM head logits row=%d len=%d want %d", pos, len(scratch.Logits[pos]), vocab)
			}
			copy(scratch.Logits[pos][start:end], chunkOutSlice[pos*chunk:(pos+1)*chunk])
		}
		ggufChunkedLMHeadCounters.copyNS.Add(uint64(time.Since(copyStart).Nanoseconds()))
	}
	applyFinalLogitSoftcapping(scratch, positions, vocab)
	return nil
}

func UploadGGUFF32LMHeadBuffer(weights *TextWeights) (*gpu.Buffer, int, int, error) {
	if weights == nil {
		return nil, 0, 0, fmt.Errorf("nil weights for GGUF F32 LM head")
	}
	fp := weights.ForwardPlan()
	if fp.Globals.EmbedTokens.Name == "" {
		return nil, 0, 0, fmt.Errorf("GGUF F32 LM head missing embed_tokens binding")
	}
	t, err := weights.CachedFloatTensor(fp.Globals.EmbedTokens.Name)
	if err != nil {
		return nil, 0, 0, err
	}
	if len(t.Shape) != 2 {
		return nil, 0, 0, fmt.Errorf("GGUF F32 LM head embed shape=%v", t.Shape)
	}
	vocab, hidden := t.Shape[0], t.Shape[1]
	if vocab <= 0 || hidden <= 0 || len(t.Data) < vocab*hidden {
		return nil, 0, 0, fmt.Errorf("GGUF F32 LM head bad shape/data shape=%v data=%d", t.Shape, len(t.Data))
	}
	transposed := make([]float32, hidden*vocab)
	for v := 0; v < vocab; v++ {
		for h := 0; h < hidden; h++ {
			transposed[h*vocab+v] = t.Data[v*hidden+h]
		}
	}
	buf, err := gpu.Malloc(len(transposed))
	if err != nil {
		return nil, 0, 0, err
	}
	if err := buf.Upload(transposed); err != nil {
		buf.Free()
		return nil, 0, 0, err
	}
	return buf, vocab, hidden, nil
}

func runDenseF32GPULMHead(scratch ForwardScratch, hiddenSize int, cached *gpu.Buffer, cachedVocab, cachedHidden int) error {
	if cached == nil || cachedVocab <= 0 || cachedHidden <= 0 {
		return fmt.Errorf("dense F32 GPU LM head missing cached weight")
	}
	if cachedHidden != hiddenSize || hiddenSize <= 0 || len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("dense F32 GPU LM head shape mismatch cached=[%d,%d] hidden=%d hidden_len=%d", cachedVocab, cachedHidden, hiddenSize, len(scratch.Hidden))
	}
	positions := len(scratch.Hidden) / hiddenSize
	if len(scratch.Logits) < positions {
		return fmt.Errorf("dense F32 GPU LM head logits rows=%d want %d", len(scratch.Logits), positions)
	}
	hidLen, okHid := checked.MulInt(positions, hiddenSize)
	outLen, okOut := checked.MulInt(positions, cachedVocab)
	if positions <= 0 || !okHid || !okOut {
		return fmt.Errorf("dense F32 GPU LM head invalid sizes positions=%d vocab=%d hidden=%d", positions, cachedVocab, hiddenSize)
	}
	xBuf, err := gpu.Malloc(hidLen)
	if err != nil {
		return err
	}
	defer xBuf.Free()
	outBuf, err := gpu.Malloc(outLen)
	if err != nil {
		return err
	}
	defer outBuf.Free()
	if err := xBuf.Upload(scratch.Hidden[:hidLen]); err != nil {
		return err
	}
	if err := gpu.Sgemm(positions, cachedVocab, hiddenSize, 1, xBuf, cached, outBuf); err != nil {
		return fmt.Errorf("dense F32 GPU LM head SGEMM: %w", err)
	}
	flat := make([]float32, outLen)
	if err := outBuf.Download(flat); err != nil {
		return err
	}
	for pos := 0; pos < positions; pos++ {
		if len(scratch.Logits[pos]) < cachedVocab {
			return fmt.Errorf("dense F32 GPU LM head logits row=%d len=%d want %d", pos, len(scratch.Logits[pos]), cachedVocab)
		}
		copy(scratch.Logits[pos][:cachedVocab], flat[pos*cachedVocab:(pos+1)*cachedVocab])
	}
	applyFinalLogitSoftcapping(scratch, positions, cachedVocab)
	return nil
}

func runDenseF32GPULMHeadDeviceGraph(scratch ForwardScratch, hiddenSize int, cached *gpu.Buffer, cachedVocab, cachedHidden int, sampleDraws []float64, scEmbed *gpu.Buffer, scVocab, scHidden int, needSC bool) ([]int, []float64, []int, []float32, error) {
	if cached == nil || cachedVocab <= 0 || cachedHidden <= 0 {
		return nil, nil, nil, nil, fmt.Errorf("dense F32 LM head device graph missing cached weight")
	}
	if cachedHidden != hiddenSize || hiddenSize <= 0 || len(scratch.Hidden)%hiddenSize != 0 {
		return nil, nil, nil, nil, fmt.Errorf("dense F32 LM head device graph shape mismatch cached=[%d,%d] hidden=%d hidden_len=%d", cachedVocab, cachedHidden, hiddenSize, len(scratch.Hidden))
	}
	positions := len(scratch.Hidden) / hiddenSize
	if len(sampleDraws) < positions {
		return nil, nil, nil, nil, fmt.Errorf("GGUF device sampler uniforms=%d want %d", len(sampleDraws), positions)
	}
	hidLen, okHid := checked.MulInt(positions, hiddenSize)
	outLen, okOut := checked.MulInt(positions, cachedVocab)
	if positions <= 0 || !okHid || !okOut {
		return nil, nil, nil, nil, fmt.Errorf("dense F32 LM head device graph invalid sizes positions=%d vocab=%d hidden=%d", positions, cachedVocab, hiddenSize)
	}
	xBuf, err := gpu.Malloc(hidLen)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	defer xBuf.Free()
	logitsBuf, err := gpu.Malloc(outLen)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	defer logitsBuf.Free()
	if err := xBuf.Upload(scratch.Hidden[:hidLen]); err != nil {
		return nil, nil, nil, nil, err
	}
	if err := gpu.Sgemm(positions, cachedVocab, hiddenSize, 1, xBuf, cached, logitsBuf); err != nil {
		return nil, nil, nil, nil, fmt.Errorf("dense F32 LM head device graph SGEMM: %w", err)
	}
	if scratch.FinalLogitSoftcapping > 0 {
		if err := gpu.LogitSoftcapF32(logitsBuf, outLen, scratch.FinalLogitSoftcapping); err != nil {
			return nil, nil, nil, nil, err
		}
	}
	uniforms := make([]float32, positions)
	for i := 0; i < positions; i++ {
		uniforms[i] = float32(sampleDraws[i])
	}
	arg, ent, samp, err := gpu.DiffusionDenseSample(logitsBuf, uniforms, positions, cachedVocab, scratch.SCTempInv)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	var sc []float32
	if needSC {
		if scEmbed == nil || scVocab != cachedVocab || scHidden != hiddenSize {
			return nil, nil, nil, nil, fmt.Errorf("GGUF device self-conditioning embedding [%d,%d] want [%d,%d]", scVocab, scHidden, cachedVocab, hiddenSize)
		}
		probBuf, err := gpu.Malloc(outLen)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		defer probBuf.Free()
		if err := gpu.DiffusionSoftmaxRows(logitsBuf, probBuf, positions, cachedVocab, scratch.SCTempInv); err != nil {
			return nil, nil, nil, nil, err
		}
		scLen, okSC := checked.MulInt(positions, hiddenSize)
		if !okSC {
			return nil, nil, nil, nil, fmt.Errorf("GGUF device self-conditioning output overflow")
		}
		scBuf, err := gpu.Malloc(scLen)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		defer scBuf.Free()
		if err := gpu.Sgemm(positions, hiddenSize, cachedVocab, 1, probBuf, scEmbed, scBuf); err != nil {
			return nil, nil, nil, nil, fmt.Errorf("GGUF device self-conditioning SGEMM: %w", err)
		}
		sc = make([]float32, scLen)
		if err := scBuf.Download(sc); err != nil {
			return nil, nil, nil, nil, err
		}
		scale := float32(math.Sqrt(float64(hiddenSize)))
		for i := range sc {
			sc[i] *= scale
		}
	}
	return arg, ent, samp, sc, nil
}

func UploadFP8LMHeadBuffer(fp8w *FP8TextWeights) (*gpu.Buffer, int, int, error) {
	if fp8w == nil || fp8w.shards == nil {
		return nil, 0, 0, fmt.Errorf("no FP8 weights for dense LM head")
	}
	raw, dtype, shape, err := fp8w.shards.GetRaw("model.decoder.embed_tokens.weight")
	if err != nil {
		return nil, 0, 0, err
	}
	if dtype != "BF16" || len(shape) != 2 {
		return nil, 0, 0, fmt.Errorf("dense LM head: dtype=%s shape=%v", dtype, shape)
	}
	vocab, h := shape[0], shape[1]
	if vocab <= 0 || h <= 0 {
		return nil, 0, 0, fmt.Errorf("dense LM head invalid shape [%d,%d]", vocab, h)
	}
	needElems, ok := checked.MulInt(vocab, h)
	needBytes, okBytes := checked.MulInt(needElems, 2)
	if !ok || !okBytes || len(raw) < needBytes {
		return nil, 0, 0, fmt.Errorf("dense LM head raw BF16 bytes=%d want at least %d for shape [%d,%d]", len(raw), needBytes, vocab, h)
	}
	wBuf, err := gpu.UploadBF16LMHead(raw[:needBytes], vocab, h)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("dense LM head GPU upload: %w", err)
	}
	return wBuf, vocab, h, nil
}

func denseLMHeadRows(out []float32, wBuf *gpu.Buffer, hidden []float32, positions, vocab, hiddenSize int) error {
	if positions <= 0 || vocab <= 0 || hiddenSize <= 0 {
		return fmt.Errorf("dense LM head rows invalid dims positions=%d vocab=%d hidden=%d", positions, vocab, hiddenSize)
	}
	needHidden, okHidden := checked.MulInt(positions, hiddenSize)
	needOut, okOut := checked.MulInt(positions, vocab)
	if !okHidden || !okOut || len(hidden) < needHidden || len(out) < needOut {
		return fmt.Errorf("dense LM head rows buffers hidden=%d/%d out=%d/%d", len(hidden), needHidden, len(out), needOut)
	}
	for pos := 0; pos < positions; pos++ {
		if err := gpu.BF16LMHeadWithBuffer(out[pos*vocab:(pos+1)*vocab], wBuf, hidden[pos*hiddenSize:(pos+1)*hiddenSize], vocab, hiddenSize); err != nil {
			return fmt.Errorf("row %d: %w", pos, err)
		}
	}
	return nil
}

func denseLMHeadRowsToBuffer(outBuf *gpu.Buffer, wBuf *gpu.Buffer, hidden []float32, positions, vocab, hiddenSize int) error {
	needOut, okOut := checked.MulInt(positions, vocab)
	if outBuf == nil || !okOut {
		return fmt.Errorf("dense LM head output buffer invalid positions=%d vocab=%d", positions, vocab)
	}
	needBytes, okBytes := checked.MulInt(needOut, 4)
	if !okBytes || outBuf.Size < needBytes {
		return fmt.Errorf("dense LM head output buffer bytes=%d want %d", outBuf.Size, needBytes)
	}
	out := make([]float32, needOut)
	if err := denseLMHeadRows(out, wBuf, hidden, positions, vocab, hiddenSize); err != nil {
		return err
	}
	return outBuf.Upload(out)
}

func runDenseGPULMHeadDeviceGraph(fp8w *FP8TextWeights, scratch ForwardScratch, hiddenSize int, cached *gpu.Buffer, cachedVocab, cachedHidden int, sampleDraws []float64, scEmbed *gpu.Buffer, scVocab, scHidden int, needSC bool) ([]int, []float64, []int, []float32, error) {
	if fp8w == nil || fp8w.shards == nil {
		return nil, nil, nil, nil, fmt.Errorf("no FP8 weights for dense LM head")
	}
	_, dtype, shape, err := fp8w.shards.GetRaw("model.decoder.embed_tokens.weight")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	if dtype != "BF16" || len(shape) != 2 {
		return nil, nil, nil, nil, fmt.Errorf("dense LM head: dtype=%s shape=%v", dtype, shape)
	}
	vocab, h := shape[0], shape[1]
	if cached == nil || cachedVocab != vocab || cachedHidden != h || h != hiddenSize || len(scratch.Hidden)%hiddenSize != 0 {
		return nil, nil, nil, nil, fmt.Errorf("dense LM head device graph cached shape [%d,%d] want [%d,%d]", cachedVocab, cachedHidden, vocab, h)
	}
	positions := len(scratch.Hidden) / hiddenSize
	if len(sampleDraws) < positions {
		return nil, nil, nil, nil, fmt.Errorf("device sampler uniforms=%d want %d", len(sampleDraws), positions)
	}
	hidLen, okHid := checked.MulInt(positions, hiddenSize)
	outLen, okOut := checked.MulInt(positions, vocab)
	if !okHid || !okOut {
		return nil, nil, nil, nil, fmt.Errorf("dense LM head device graph size overflow")
	}
	logitsBuf, err := gpu.Malloc(outLen)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	defer logitsBuf.Free()
	if err := denseLMHeadRowsToBuffer(logitsBuf, cached, scratch.Hidden[:hidLen], positions, vocab, h); err != nil {
		return nil, nil, nil, nil, err
	}
	uniforms := make([]float32, positions)
	for i := 0; i < positions; i++ {
		uniforms[i] = float32(sampleDraws[i])
	}
	arg, ent, samp, err := gpu.DiffusionDenseSample(logitsBuf, uniforms, positions, vocab, scratch.SCTempInv)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	var sc []float32
	if needSC {
		if scEmbed == nil || scVocab != vocab || scHidden != h {
			return nil, nil, nil, nil, fmt.Errorf("device self-conditioning embedding [%d,%d] want [%d,%d]", scVocab, scHidden, vocab, h)
		}
		probBuf, err := gpu.Malloc(outLen)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		defer probBuf.Free()
		if err := gpu.DiffusionSoftmaxRows(logitsBuf, probBuf, positions, vocab, scratch.SCTempInv); err != nil {
			return nil, nil, nil, nil, err
		}
		scLen, okSC := checked.MulInt(positions, h)
		if !okSC {
			return nil, nil, nil, nil, fmt.Errorf("device self-conditioning output overflow")
		}
		scBuf, err := gpu.Malloc(scLen)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		defer scBuf.Free()
		if err := gpu.Sgemm(positions, h, vocab, 1, probBuf, scEmbed, scBuf); err != nil {
			return nil, nil, nil, nil, err
		}
		sc = make([]float32, scLen)
		if err := scBuf.Download(sc); err != nil {
			return nil, nil, nil, nil, err
		}
		scale := float32(math.Sqrt(float64(h)))
		for i := range sc {
			sc[i] *= scale
		}
	}
	return arg, ent, samp, sc, nil
}

// runDenseGPULMHead computes full dense logits on GPU using BF16 embeddings.
func runDenseGPULMHead(fp8w *FP8TextWeights, scratch ForwardScratch, hiddenSize int, cached *gpu.Buffer, cachedVocab, cachedHidden int) error {
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
	if vocab <= 0 || h <= 0 || hiddenSize <= 0 || h != hiddenSize || len(scratch.Hidden)%hiddenSize != 0 {
		return fmt.Errorf("dense LM head shape mismatch: vocab=%d hidden=%d want_hidden=%d hidden_len=%d", vocab, h, hiddenSize, len(scratch.Hidden))
	}
	needElems, ok := checked.MulInt(vocab, h)
	needBytes, okBytes := checked.MulInt(needElems, 2)
	if !ok || !okBytes || len(raw) < needBytes {
		return fmt.Errorf("dense LM head raw BF16 bytes=%d want at least %d for shape [%d,%d]", len(raw), needBytes, vocab, h)
	}
	positions := len(scratch.Hidden) / hiddenSize
	if len(scratch.Logits) < positions {
		return fmt.Errorf("dense LM head logits rows=%d want %d", len(scratch.Logits), positions)
	}
	wBuf := cached
	freeAfter := false
	if wBuf != nil && (cachedVocab != vocab || cachedHidden != h) {
		return fmt.Errorf("dense LM head cached shape [%d,%d] want [%d,%d]", cachedVocab, cachedHidden, vocab, h)
	}
	if wBuf == nil {
		wBuf, err = gpu.UploadBF16LMHead(raw[:needBytes], vocab, h)
		if err != nil {
			return fmt.Errorf("dense LM head GPU upload: %w", err)
		}
		freeAfter = true
	}
	if freeAfter {
		defer wBuf.Free()
	}
	for pos := 0; pos < positions; pos++ {
		if len(scratch.Logits[pos]) < vocab {
			return fmt.Errorf("dense LM head logits row=%d len=%d want %d", pos, len(scratch.Logits[pos]), vocab)
		}
	}
	flatLogits := make([]float32, positions*vocab)
	if err := denseLMHeadRows(flatLogits, wBuf, scratch.Hidden[:positions*hiddenSize], positions, vocab, h); err != nil {
		return fmt.Errorf("dense LM head rows: %w", err)
	}
	for pos := 0; pos < positions; pos++ {
		copy(scratch.Logits[pos][:vocab], flatLogits[pos*vocab:(pos+1)*vocab])
	}
	applyFinalLogitSoftcapping(scratch, positions, vocab)
	return nil
}
