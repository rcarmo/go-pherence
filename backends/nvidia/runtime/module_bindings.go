package nvidia

type moduleFunctions map[string]CUfunction

func (f moduleFunctions) get(name string) CUfunction { return f[name] }

func bindMegaModuleFunctions(f moduleFunctions) {
	sgemmFn = f.get("sgemm_nn")
	fnVecAdd = f.get("vec_add")
	fnVecMul = f.get("vec_mul")
	fnVecScale = f.get("vec_scale")
	fnVecAddScaled = f.get("vec_add_scaled")
	fnToBF16F32 = f.get("to_bf16_f32")
	fnVecSilu = f.get("vec_silu")
	fnRmsNorm = f.get("rms_norm")
	ropeFn = f.get("rope_apply")
	ropePartialFn = f.get("rope_partial")
	attnScoreFn = f.get("gqa_attention_scores")
	softmaxRowsFn = f.get("row_softmax_debug")
	attnFn = f.get("gqa_attention")
	q4Fn = f.get("gemv_q4sym")
	fnFusedSiLUMul = f.get("fused_silu_mul")
	fnPrefetch = f.get("prefetch_l2")
	fnGemmQ4 = f.get("gemm_q4sym")
	fnLMHead = f.get("lm_head_gemv")
	fnMLXGemv = f.get("mlx_gemv")
	fnMLXGemm = f.get("mlx_gemm")
	fnMLXCorrect = f.get("mlx_correct")
	fnBF16RMSNorm = f.get("bf16_rms_norm")
	fnBF16RMSNormNoScale = f.get("bf16_rms_norm_no_scale")
	fnBF16VecAdd = f.get("bf16_vec_add")
	fnBF16SiLUMul = f.get("bf16_silu_mul")
	fnBF16GELUTanhMul = f.get("bf16_gelu_tanh_mul")
	fnBF16LMHead = f.get("bf16_lm_head_gemv")
	fnRmsNormNoScale = f.get("rms_norm_no_scale")
	fnGELUTanhMul = f.get("gelu_tanh_mul")
	fnNVFP4DequantF32 = f.get("nvfp4_dequant_f32")
	fnNVFP4GemvF32 = f.get("nvfp4_gemv_f32")
	fnFP8E4M3GemvF32 = f.get("fp8_e4m3_gemv_f32")
}
