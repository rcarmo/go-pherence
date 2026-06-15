package nvidia

import (
	"fmt"
	"strings"

	ptxwhisper "github.com/rcarmo/go-pherence/backends/cuda/ptx"
	"github.com/rcarmo/go-pherence/backends/nvidia/ptx"
	ptxbf16 "github.com/rcarmo/go-pherence/backends/nvidia/ptx/bf16"
	ptxfp8 "github.com/rcarmo/go-pherence/backends/nvidia/ptx/fp8"
	ptxideogram "github.com/rcarmo/go-pherence/backends/nvidia/ptx/ideogram"
	ptxmlx "github.com/rcarmo/go-pherence/backends/nvidia/ptx/mlx"
	ptxnvfp4 "github.com/rcarmo/go-pherence/backends/nvidia/ptx/nvfp4"
	ptxq4 "github.com/rcarmo/go-pherence/backends/nvidia/ptx/q4"
	ptxq5 "github.com/rcarmo/go-pherence/backends/nvidia/ptx/q5"
	ptxq8 "github.com/rcarmo/go-pherence/backends/nvidia/ptx/q8"
)

type moduleEntry struct {
	name string
	ptx  string
}

func validateModuleEntries(entries []moduleEntry) error {
	if len(entries) == 0 {
		return fmt.Errorf("empty NVIDIA mega module entry table")
	}
	seen := make(map[string]struct{}, len(entries))
	for i, e := range entries {
		name := strings.TrimSpace(e.name)
		if name == "" {
			return fmt.Errorf("module entry %d has empty kernel name", i)
		}
		if strings.TrimSpace(e.ptx) == "" {
			return fmt.Errorf("module entry %q has empty PTX", name)
		}
		if _, ok := seen[name]; ok {
			return fmt.Errorf("duplicate module entry %q", name)
		}
		seen[name] = struct{}{}
	}
	return nil
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
		{"split_gate_up", ptx.SplitGateUpPTX},
		{"gate_up_gelu", ptx.GateUpGELUPTX},
		{"scatter_weighted_rows", ptx.ScatterWeightedRowsPTX},
		{"scatter_weighted_rows_batch", ptx.ScatterWeightedRowsBatchPTX},
		{"gather_rows", ptx.GatherRowsPTX},
		{"mul_weights", ptx.MulWeightsPTX},
		{"expert_meta_reduce", ptx.ExpertMetaReducePTX},
		{"gemv_q4sym", ptxq4.GemvQ4OptPTX},
		{"gemv_q4_k", ptxq4.GemvQ4KPTX},
		{"gemv_q4_k_batch", ptxq4.GemvQ4KBatchPTX},
		{"gate_up_gelu_q4_k", ptxq4.GateUpGELUQ4KPTX},
		{"gate_up_gelu_q4_k_by_work", ptxq4.GateUpGELUQ4KByWorkPTX},
		{"gate_up_gelu_q4_k_by_work_ptrs", ptxq4.GateUpGELUQ4KByWorkPtrsPTX},
		{"gate_up_q4_k_by_work_ptrs", ptxq4.GateUpQ4KByWorkPtrsPTX},
		{"gate_up_q4_k_by_work_ptrs_raw", ptxq4.GateUpQ4KByWorkPtrsRawPTX},
		{"gemv_q5_0_batch", ptxq5.GemvQ5_0BatchPTX},
		{"gemv_q5_0_scatter_by_work", ptxq5.GemvQ5_0ScatterByWorkPTX},
		{"gemv_q5_0_scatter_by_work_ptrs", ptxq5.GemvQ5_0ScatterByWorkPtrsPTX},
		{"gemv_q8_0", ptxq8.GemvQ8_0PTX},
		{"gemv_q8_0_batch", ptxq8.GemvQ8_0BatchPTX},
		{"gemv_q8_0_scatter", ptxq8.GemvQ8_0ScatterPTX},
		{"gemv_q8_0_scatter_by_work", ptxq8.GemvQ8_0ScatterByWorkPTX},
		{"gemv_q8_0_scatter_by_work_ptrs", ptxq8.GemvQ8_0ScatterByWorkPtrsPTX},
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
		{"whisper_mel_spectrogram", ptxwhisper.FFTPTX},
		{"whisper_conv1d_k3_s1", ptxwhisper.Conv1DK3S1PTX},
		{"whisper_conv1d_k3_s2", ptxwhisper.Conv1DK3S2PTX},
		{"whisper_attention_full", ptxwhisper.AttentionFullPTX},
		{"whisper_cross_attention", ptxwhisper.CrossAttentionPTX},
		{"whisper_attentive_stat_pool", ptxwhisper.AttentivePoolPTX},
		{"rms_norm_no_scale", ptx.RmsNormNoScalePTX},
		{"nvfp4_dequant_f32", ptxnvfp4.NVFP4DequantF32PTX},
		{"nvfp4_gemv_f32", ptxnvfp4.NVFP4GemvF32PTX},
		{"fp8_e4m3_gemv_f32", ptxfp8.FP8E4M3GemvF32PTX},
		{"fp8_e4m3_gemm_f32", ptxfp8.FP8E4M3GemmF32PTX},
		{"fp8_e4m3_dequant_transpose_f32", ptxfp8.FP8E4M3DequantTransposeF32PTX},
		{"ideogram_cfg_step_f32", ptxideogram.IdeogramCFGStepPTX},
		{"ideogram_layer_norm_no_affine_f32", ptxideogram.IdeogramLayerNormNoAffinePTX},
		{"ideogram_rms_norm_rows_f32", ptxideogram.IdeogramRMSNormRowsPTX},
		{"ideogram_adaln_transform_f32", ptxideogram.IdeogramAdaLNTransformPTX},
		{"ideogram_gated_residual_f32", ptxideogram.IdeogramGatedResidualPTX},
		{"ideogram_gated_residual_rows_f32", ptxideogram.IdeogramGatedResidualRowsPTX},
		{"ideogram_mrope_f32", ptxideogram.IdeogramMRoPEPTX},
		{"ideogram_attention_scores_f32", ptxideogram.IdeogramAttentionScoresPTX},
		{"ideogram_attention_values_f32", ptxideogram.IdeogramAttentionValuesPTX},
		{"ideogram_split_qkv_f32", ptxideogram.IdeogramSplitQKVPTX},
		{"ideogram_latent_denorm_f32", ptxideogram.IdeogramLatentDenormPTX},
		{"ideogram_rgb_clamp_f32", ptxideogram.IdeogramRGBClampF32PTX},
		{"ideogram_upsample_nearest_f32", ptxideogram.IdeogramUpsampleNearestPTX},
		{"ideogram_unpatchify_f32", ptxideogram.IdeogramUnpatchifyPTX},
		{"ideogram_group_norm_f32", ptxideogram.IdeogramGroupNormPTX},
		{"ideogram_conv2d_f32", ptxideogram.IdeogramConv2DPTX},
	}
}
