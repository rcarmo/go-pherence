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
	ptxq6 "github.com/rcarmo/go-pherence/backends/nvidia/ptx/q6"
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
		{"q6_staged_j24_o64", ptx.QKStagedPTX},
		{"q5_staged_j24_o64", "// included in QKStagedPTX\n"},
		{"q5_staged_j32_o64", "// included in QKStagedPTX\n"},
		{"sgemm_nn", ptx.SgemmPTX},
		{"sgemm_nn_compensated", ptx.SgemmCompensatedPTX},
		{"sgemm_nn_reg2", ptx.SgemmReg2PTX},
		{"sgemm_nn_skinny", ptx.SgemmSkinnyPTX},
		{"vec_add", ptx.VecAddPTX},
		{"vec_mul", ptx.VecMulPTX},
		{"vec_scale", ptx.VecScalePTX},
		{"vec_add_scaled", ptx.VecAddScaledPTX},
		{"to_bf16_f32", ptx.ToBF16F32PTX},
		{"widen_bf16_transpose", ptx.WidenBF16TransposePTX},
		{"vec_silu", ptx.VecSiLUPTX},
		{"gelu_erf", ptx.GELUErfPTX},
		{"rms_norm", ptx.RmsNormPTX},
		{"rope_apply", ptx.RoPEPTX},
		{"rope_partial", ptx.RoPEPartialPTX},
		{"rope_partial_sequence", ptx.RoPEPartialSequencePTX},
		{"gqa_attention_scores", ptx.AttentionScoresPTX},
		{"row_softmax_debug", ptx.SoftmaxRowsPTX},
		{"gqa_attention", ptx.AttentionPTX},
		{"gqa_attention_independent", ptx.IndependentBranchAttentionPTX},
		{"gqa_attention_causal_batch", ptx.CausalBatchAttentionWarpPTX},
		{"gqa_attention_long", ptx.LongAttentionPTX},
		{"gqa_attention_segmented", ptx.PackedDecisionPTX},
		{"rope_partial_segmented", "// included in PackedDecisionPTX\n"},
		{"gemv_q5_packed_selected_batch_f32", "// included in PackedDecisionPTX\n"},
		{"gqa_attention_splitkv_partial", ptx.AttentionSplitKVPartialPTX},
		{"gqa_attention_splitkv_merge", ptx.AttentionSplitKVMergePTX},
		{"gelu_tanh_mul", ptx.GELUTanhMulPTX},
		{"split_gate_up", ptx.SplitGateUpPTX},
		{"gate_up_gelu", ptx.GateUpGELUPTX},
		{"scatter_weighted_rows", ptx.ScatterWeightedRowsPTX},
		{"scatter_weighted_rows_batch", ptx.ScatterWeightedRowsBatchPTX},
		{"gather_rows", ptx.GatherRowsPTX},
		{"mul_weights", ptx.MulWeightsPTX},
		{"expert_meta_reduce", ptx.ExpertMetaReducePTX},
		{"logit_softcap_f32", ptx.LogitSoftcapPTX},
		{"gemv_q4sym", ptxq4.GemvQ4OptPTX},
		{"gemv_q4_k", ptxq4.GemvQ4KPTX},
		{"gemv_q4_k_batch", ptxq4.GemvQ4KBatchPTX},
		{"gemm_q4_k_batch8", ptxq4.GemmQ4KBatch8PTX},
		{"quantize_q8_rows_sum", ptxq4.Q8SumPTX},
		{"gemm_q4_k_q8_batch4", ptxq4.GemmQ4Q8PTX},
		{"gemm_q4_k_q8_batch8", ptxq4.GemmQ4Q8Batch8PTX},
		{"gemm_q4_k_q8_batch16", ptxq4.GemmQ4Q8Batch16PTX},
		{"gemm_q4_coalesced_q8_batch8", ptxq4.GemmQ4CoalescedQ8Batch8PTX},
		{"gemm_q4_coalesced_q8_batch16", ptxq4.GemmQ4CoalescedQ8PTX},
		{"gemm_q4_coalesced_f32", ptxq4.GemmQ4CoalescedF32PTX},
		{"gemm_q4_coalesced_mmq8", ptxq4.GemmQ4CoalescedMMQ8PTX},
		{"gemm_q4_coalesced_mmq_j16", ptxq4.GemmQ4CoalescedMMQJ16PTX},
		{"gemm_q4_pair_mmq_j16", ptxq4.GemmQ4PairMMQJ16PTX},
		{"go_system_one_mmq_q4_k_j8", ptxq4.UpstreamQuantMMQPTX},
		{"go_system_one_mmq_q4_k_j16", "// entry included in pinned Q4 MMQ PTX\n"},
		{"go_system_one_mmq_q4_k_j24", "// entry included in pinned Q4 MMQ PTX\n"},
		{"go_system_one_mmq_q4_k_j32", "// entry included in pinned Q4 MMQ PTX\n"},
		{"go_system_one_mmq_q4_k_j64", "// entry included in pinned Q4 MMQ PTX\n"},
		{"quantize_q8_1_mmq_ds4", ptxq4.QuantizeQ81MMQ2DPTX},
		{"gemm_q4_raw_f32", ptxq4.GemmQ4RawF32PTX},
		{"gate_up_gelu_q4_k", ptxq4.GateUpGELUQ4KPTX},
		{"gate_up_gelu_q4_k_by_work", ptxq4.GateUpGELUQ4KByWorkPTX},
		{"gate_up_gelu_q4_k_by_work_ptrs", ptxq4.GateUpGELUQ4KByWorkPtrsPTX},
		{"gate_up_q4_k_by_work_ptrs", ptxq4.GateUpQ4KByWorkPtrsPTX},
		{"gemv_q5_0_batch", ptxq5.GemvQ5_0BatchPTX},
		{"gemv_q5_k_batch", ptxq5.GemvQ5KBatchPTX},
		{"gemm_q5_k_warp", ptxq5.GemmQ5KWarpPTX},
		{"gemm_q5_packed_f32", ptxq5.GemmQ5PackedF32PTX},
		{"gemm_q5_packed_q8_batch4", ptxq5.GemmQ5PackedQ8PTX},
		{"gemm_q5_packed_mmq64_j8", ptxq5.GemmQ5PackedMMQ64TilesPTX},
		{"gemm_q5_packed_mmq64_j16", "// entry included in Q5_K MMQ64 tile PTX\n"},
		{"gemm_q5_packed_mmq64_j24", "// entry included in Q5_K MMQ64 tile PTX\n"},
		{"gemv_q5_packed_selected_f32", ptxq5.GemvQ5PackedSelectedF32PTX},
		{"gemv_q6_k_batch", ptxq6.GemvQ6KBatchPTX},
		{"gemm_q6_k_warp", ptxq6.GemmQ6KWarpPTX},
		{"gemm_q6_k_batch8", ptxq6.GemmQ6KBatch8PTX},
		{"quantize_q8_rows", ptxq6.QuantizeQ8PTX},
		{"gemm_q6_k_q8_batch4", ptxq6.GemmQ6Q8PTX},
		{"quantize_q8_rows16", ptxq6.QuantizeQ8Rows16GroupedPTX},
		{"gemm_q6_packed_q8_batch4", ptxq6.GemmQ6PackedQ8PTX},
		{"gemm_q6_packed_mmq8", ptxq6.GemmQ6PackedMMQ8PTX},
		{"gemm_q6_packed_mmq64", ptxq6.GemmQ6PackedMMQ64PTX},
		{"gemm_q6_packed_mmq64_j12", "// entry included in Q6_K MMQ64 tile PTX\n"},
		{"gemm_q6_packed_mmq64_j16", "// entry included in Q6_K MMQ64 tile PTX\n"},
		{"gemm_q6_packed_f32", ptxq6.GemmQ6PackedF32PTX},
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
		{"mlx_selected_expert_gemv_persistent", ptxmlx.MLXSelectedExpertPersistentPTX},
		{"bf16_rms_norm", ptxbf16.BF16RMSNormPTX},
		{"bf16_rms_norm_no_scale", ptxbf16.BF16RMSNormNoScalePTX},
		{"bf16_vec_add", ptxbf16.BF16VecAddPTX},
		{"bf16_silu_mul", ptxbf16.BF16SiLUMulPTX},
		{"bf16_gelu_tanh_mul", ptxbf16.BF16GELUTanhMulPTX},
		{"bf16_lm_head_gemv", ptxbf16.BF16LMHeadPTX},
		{"mel_spectrogram", ptxwhisper.FFTPTX},
		{"conv1d_k3_s1", ptxwhisper.Conv1DK3S1PTX},
		{"conv1d_k3_s2", ptxwhisper.Conv1DK3S2PTX},
		{"attention_full", ptxwhisper.AttentionFullPTX},
		{"attention_full_online", ptxwhisper.AttentionFullOnlinePTX},
		{"cross_attention", ptxwhisper.CrossAttentionPTX},
		{"attentive_stat_pool", ptxwhisper.AttentivePoolPTX},
		{"whisper_row_affine_f32", ptxwhisper.EncoderRowAffinePTX},
		{"whisper_row_bias_f32", ptxwhisper.EncoderRowBiasPTX},
		{"whisper_transpose_f32", ptxwhisper.EncoderTransposePTX},
		{"whisper_gelu_tanh_f32", ptxwhisper.EncoderGELUTanhPTX},
		{"rms_norm_no_scale", ptx.RmsNormNoScalePTX},
		{"rms_norm_rows_no_weight", ptx.RMSNormRowsNoWeightPTX},
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
