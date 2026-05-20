package modules

import (
	"github.com/rcarmo/go-pherence/backends/nvidia/ptx"
	ptxbf16 "github.com/rcarmo/go-pherence/backends/nvidia/ptx/bf16"
	ptxmlx "github.com/rcarmo/go-pherence/backends/nvidia/ptx/mlx"
	ptxnvfp4 "github.com/rcarmo/go-pherence/backends/nvidia/ptx/nvfp4"
	ptxq4 "github.com/rcarmo/go-pherence/backends/nvidia/ptx/q4"
)

type moduleEntry struct {
	name string
	ptx  string
}

func megaModuleEntries() []moduleEntry {
	return []moduleEntry{
		{"sgemm_nn", ptx.SgemmPTX},
		{"vec_add", ptx.VecAddPTX},
		{"vec_mul", ptx.VecMulPTX},
		{"vec_scale", ptx.VecScalePTX},
		{"vec_add_scaled", ptx.VecAddScaledPTX},
		{"to_bf16_f32", ptx.ToBF16F32PTX},
		{"vec_silu", ptx.VecSiLUPTX},
		{"rms_norm", ptx.RmsNormPTX},
		{"rope_apply", ptx.RoPEPTX},
		{"rope_partial", ptx.RoPEPartialPTX},
		{"gqa_attention_scores", ptx.AttentionScoresPTX},
		{"row_softmax_debug", ptx.SoftmaxRowsPTX},
		{"gqa_attention", ptx.AttentionPTX},
		{"gelu_tanh_mul", ptx.GELUTanhMulPTX},
		{"gemv_q4sym", ptxq4.GemvQ4OptPTX},
		{"fused_silu_mul", ptx.FusedSiLUMulPTX},
		{"prefetch_l2", ptx.PrefetchPTX},
		{"gemm_q4sym", ptxq4.GemmQ4PTX},
		{"lm_head_gemv", ptx.LMHeadPTX},
		{"mlx_gemv", ptxmlx.MLXGemvPTX},
		{"mlx_gemm", ptxmlx.MLXGemmPTX},
		{"mlx_correct", ptxmlx.MLXCorrectPTX},
		{"bf16_rms_norm", ptxbf16.BF16RMSNormPTX},
		{"bf16_rms_norm_no_scale", ptxbf16.BF16RMSNormNoScalePTX},
		{"bf16_vec_add", ptxbf16.BF16VecAddPTX},
		{"bf16_silu_mul", ptxbf16.BF16SiLUMulPTX},
		{"bf16_gelu_tanh_mul", ptxbf16.BF16GELUTanhMulPTX},
		{"bf16_lm_head_gemv", ptxbf16.BF16LMHeadPTX},
		{"rms_norm_no_scale", ptx.RmsNormNoScalePTX},
		{"nvfp4_dequant_f32", ptxnvfp4.NVFP4DequantF32PTX},
		{"nvfp4_gemv_f32", ptxnvfp4.NVFP4GemvF32PTX},
	}
}
