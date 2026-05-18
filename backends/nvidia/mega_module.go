package nvidia

// Unified kernel loader: merges ALL PTX entries into one module.
// Solves the cuModuleLoadData error 201 (can't load multiple modules).

import (
	"strings"
	"sync"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/nvidia/kernels"
)

var (
	megaModuleOnce sync.Once
	megaModule     CUmodule
	megaModuleOK   bool

	// Function handles extracted from the mega module
	sgemmReady bool

	// Set in sgemm.go, runtime helper files
)

// stripPTXHeader removes .version/.target/.address_size lines from PTX
func stripPTXHeader(ptx string) string {
	var lines []string
	for _, l := range strings.Split(ptx, "\n") {
		trimmed := strings.TrimSpace(l)
		if strings.HasPrefix(trimmed, ".version") ||
			strings.HasPrefix(trimmed, ".target") ||
			strings.HasPrefix(trimmed, ".address_size") {
			continue
		}
		lines = append(lines, l)
	}
	return strings.Join(lines, "\n")
}

// loadMegaModule compiles ALL PTX kernels into one CUDA module.
func loadMegaModule() {
	megaModuleOnce.Do(func() {
		if !Init() {
			return
		}

		// Pre-warm allocator
		var warmPtr CUdeviceptr
		if r := cuMemAlloc(&warmPtr, 64*1024*1024); r == CUDA_SUCCESS {
			cuMemFree(warmPtr)
		}

		// Combine all PTX entries into one module
		var combined strings.Builder
		combined.WriteString(".version 7.0\n.target sm_80\n.address_size 64\n\n")

		entries := []struct {
			name string
			ptx  string
		}{
			{"sgemm_nn", kernels.SgemmPTX},
			{"vec_add", kernels.VecAddPTX},
			{"vec_mul", kernels.VecMulPTX},
			{"vec_scale", kernels.VecScalePTX},
			{"vec_add_scaled", kernels.VecAddScaledPTX},
			{"to_bf16_f32", kernels.ToBF16F32PTX},
			{"vec_silu", kernels.VecSiLUPTX},
			{"rms_norm", kernels.RmsNormPTX},
			{"rope_apply", kernels.RoPEPTX},
			{"rope_partial", kernels.RoPEPartialPTX},
			{"gqa_attention_scores", kernels.AttentionScoresPTX},
			{"row_softmax_debug", kernels.SoftmaxRowsPTX},
			{"gqa_attention", kernels.AttentionPTX},
			{"gelu_tanh_mul", kernels.GELUTanhMulPTX},
			{"gemv_q4sym", kernels.GemvQ4OptPTX},
			{"fused_silu_mul", kernels.FusedSiLUMulPTX},
			{"prefetch_l2", kernels.PrefetchPTX},
			{"gemm_q4sym", kernels.GemmQ4PTX},
			{"lm_head_gemv", kernels.LMHeadPTX},
			{"mlx_gemv", kernels.MLXGemvPTX},
			{"mlx_gemm", kernels.MLXGemmPTX},
			{"mlx_correct", kernels.MLXCorrectPTX},
			{"bf16_rms_norm", kernels.BF16RMSNormPTX},
			{"bf16_rms_norm_no_scale", kernels.BF16RMSNormNoScalePTX},
			{"bf16_vec_add", kernels.BF16VecAddPTX},
			{"bf16_silu_mul", kernels.BF16SiLUMulPTX},
			{"bf16_gelu_tanh_mul", kernels.BF16GELUTanhMulPTX},
			{"bf16_lm_head_gemv", kernels.BF16LMHeadPTX},
			{"rms_norm_no_scale", kernels.RmsNormNoScalePTX},
			{"nvfp4_dequant_f32", kernels.NVFP4DequantF32PTX},
			{"nvfp4_gemv_f32", kernels.NVFP4GemvF32PTX},
		}

		for _, e := range entries {
			combined.WriteString(stripPTXHeader(e.ptx))
			combined.WriteString("\n")
		}

		ptxStr := combined.String()
		ptxBytes := append([]byte(ptxStr), 0)

		EnsureContext()
		if r := cuModuleLoadData(&megaModule, unsafe.Pointer(&ptxBytes[0])); r != CUDA_SUCCESS {
			debugf("[gpu] mega module load failed: error %d\n", r)
			return
		}

		// Extract all function handles
		allOK := true
		extractFn := func(name string) CUfunction {
			nameBytes := append([]byte(name), 0)
			var fn CUfunction
			if r := cuModuleGetFunction(&fn, megaModule, unsafe.Pointer(&nameBytes[0])); r != CUDA_SUCCESS {
				debugf("[gpu] get %s: error %d\n", name, r)
				allOK = false
				return 0
			}
			return fn
		}

		sgemmFn = extractFn("sgemm_nn")
		fnVecAdd = extractFn("vec_add")
		fnVecMul = extractFn("vec_mul")
		fnVecScale = extractFn("vec_scale")
		fnVecAddScaled = extractFn("vec_add_scaled")
		fnToBF16F32 = extractFn("to_bf16_f32")
		fnVecSilu = extractFn("vec_silu")
		fnRmsNorm = extractFn("rms_norm")
		ropeFn = extractFn("rope_apply")
		ropePartialFn = extractFn("rope_partial")
		attnScoreFn = extractFn("gqa_attention_scores")
		softmaxRowsFn = extractFn("row_softmax_debug")
		attnFn = extractFn("gqa_attention")
		q4Fn = extractFn("gemv_q4sym")
		fnFusedSiLUMul = extractFn("fused_silu_mul")
		fnPrefetch = extractFn("prefetch_l2")
		fnGemmQ4 = extractFn("gemm_q4sym")
		fnLMHead = extractFn("lm_head_gemv")
		fnMLXGemv = extractFn("mlx_gemv")
		fnMLXGemm = extractFn("mlx_gemm")
		fnMLXCorrect = extractFn("mlx_correct")
		fnBF16RMSNorm = extractFn("bf16_rms_norm")
		fnBF16RMSNormNoScale = extractFn("bf16_rms_norm_no_scale")
		fnBF16VecAdd = extractFn("bf16_vec_add")
		fnBF16SiLUMul = extractFn("bf16_silu_mul")
		fnBF16GELUTanhMul = extractFn("bf16_gelu_tanh_mul")
		fnBF16LMHead = extractFn("bf16_lm_head_gemv")
		fnRmsNormNoScale = extractFn("rms_norm_no_scale")
		fnGELUTanhMul = extractFn("gelu_tanh_mul")
		fnNVFP4DequantF32 = extractFn("nvfp4_dequant_f32")
		fnNVFP4GemvF32 = extractFn("nvfp4_gemv_f32")

		if allOK {
			megaModuleOK = true
			sgemmReady = true
			sgemmOK = true
			kernelsLoaded = true
			ropeReady = true
			ropePartialReady = true
			attnScoreReady = true
			softmaxRowsReady = true
			attnReady = true
			q4Ready = true
			fusedSiLUMulOK = true
			debugf("[gpu] All %d kernels loaded in 1 module\n", len(entries))
			// Initialize streams for prefetch overlap
			if err := initStreams(); err != nil {
				debugf("[gpu] streams: %v\n", err)
			}
			// Try native BF16 kernels (Ampere+)
			InitNativeBF16()
		}
	})
}

// InitAllKernels loads all GPU kernels. Call from the CUDA-owning thread.
func InitAllKernels() {
	loadMegaModule()
}

// SgemmReady returns true if GPU SGEMM is available.
func SgemmReady() bool {
	loadMegaModule()
	return sgemmReady
}

// Q4Ready returns true if the INT4 GPU kernel is available.
func Q4Ready() bool {
	loadMegaModule()
	return q4Ready
}

func freeNVFP4Scratch() {
	nvfp4Scratch.Lock()
	defer nvfp4Scratch.Unlock()
	if nvfp4Scratch.x != nil {
		nvfp4Scratch.x.Free()
		nvfp4Scratch.x = nil
	}
	if nvfp4Scratch.out != nil {
		nvfp4Scratch.out.Free()
		nvfp4Scratch.out = nil
	}
	nvfp4Scratch.xN = 0
	nvfp4Scratch.outN = 0
	nvfp4Scratch.xPtr = 0
	nvfp4Scratch.xLen = 0
}

func shutdownMegaModule() {
	freeNVFP4Scratch()
	FreeBF16LMHeadScratch()
	if megaModule != 0 && cuModuleUnload != nil {
		EnsureContext()
		cuModuleUnload(megaModule)
	}
	megaModule = 0
	megaModuleOK = false
	megaModuleOnce = sync.Once{}
	sgemmReady = false
	sgemmOK = false
	kernelsLoaded = false
	ropeReady = false
	ropePartialReady = false
	attnScoreReady = false
	softmaxRowsReady = false
	attnReady = false
	q4Ready = false
	fusedSiLUMulOK = false
	fnPrefetch = 0
}
