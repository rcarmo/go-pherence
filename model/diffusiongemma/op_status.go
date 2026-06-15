package diffusiongemma

// OpStatus records implementation/readiness at the semantic operation level.
type OpStatus struct {
	Kind              OpKind `json:"kind"`
	Implemented       bool   `json:"implemented"`
	ReferenceComplete bool   `json:"reference_complete"`
	Domain            string `json:"domain,omitempty"`
	Note              string `json:"note,omitempty"`
}

type OpDomainSummary struct {
	Implemented       int `json:"implemented"`
	ReferenceComplete int `json:"reference_complete"`
	Total             int `json:"total"`
}

func OperationDomainSummaries(ops []OpStatus) map[string]OpDomainSummary {
	out := make(map[string]OpDomainSummary)
	for _, op := range ops {
		domain := op.Domain
		if domain == "" {
			domain = "unknown"
		}
		s := out[domain]
		s.Total++
		if op.Implemented {
			s.Implemented++
		}
		if op.ReferenceComplete {
			s.ReferenceComplete++
		}
		out[domain] = s
	}
	return out
}

func OperationStatuses() []OpStatus {
	return []OpStatus{
		{Kind: OpCanvasEmbedding, Domain: "text", Implemented: true, ReferenceComplete: true, Note: "prompt embed*sqrt(n_embd); canvas RMSNormNoScale after optional self-conditioning"},
		{Kind: OpSelfCondition, Domain: "text", Implemented: true, ReferenceComplete: true, Note: "raw-logit softmax feedback with current temp_inv and exact GELU SC MLP"},
		{Kind: OpInputNorm, Domain: "text", Implemented: true, ReferenceComplete: true, Note: "RMSNorm eps=1e-6"},
		{Kind: OpSelfAttention, Domain: "text", Implemented: true, ReferenceComplete: true, Note: "q/k norm, V no-scale norm, RoPE/proportional factors, causal prompt and bidirectional canvas masks"},
		{Kind: OpPostAttention, Domain: "text", Implemented: true, ReferenceComplete: true, Note: "RMSNorm eps=1e-6 + residual"},
		{Kind: OpDenseMLP, Domain: "text", Implemented: true, ReferenceComplete: true, Note: "Gemma4 shared dense MLP with exact ggml_gelu/GEGLU"},
		{Kind: OpPreMoE, Domain: "text", Implemented: true, ReferenceComplete: true, Note: "MoE pre-norms and router no-scale norm"},
		{Kind: OpRouter, Domain: "text", Implemented: true, ReferenceComplete: true, Note: "softmax top-k, selected-weight sum clamp, GGUF down-scale handling"},
		{Kind: OpExperts, Domain: "text", Implemented: true, ReferenceComplete: true, Note: "GGUFExpertIndex keeps Q4_K/Q8_0 expert tensors quantized and applies per-expert down scales"},
		{Kind: OpPostMoE, Domain: "text", Implemented: true, ReferenceComplete: true, Note: "post expert/shared FFN RMSNorm and residual"},
		{Kind: OpLayerScalar, Domain: "text", Implemented: true, ReferenceComplete: true, Note: "region-aware encoder/decoder layer scalar"},
		{Kind: OpFinalNorm, Domain: "text", Implemented: true, ReferenceComplete: true, Note: "output RMSNorm eps=1e-6"},
		{Kind: OpLMHead, Domain: "text", Implemented: true, ReferenceComplete: true, Note: "tied embedding LM head plus final-logit softcap"},

		{Kind: OpImagePreprocess, Domain: "vision", Implemented: true, ReferenceComplete: true, Note: "Gemma4 RGB resize/rescale/normalize to BCHW from processor_config.json"},
		{Kind: OpImageSoftTokenPrompt, Domain: "vision", Implemented: true, ReferenceComplete: true, Note: "<image> prompt expansion to BOI + 280 image-token slots + EOI"},
		{Kind: OpVisionPatchEmbedding, Domain: "vision", Implemented: true, ReferenceComplete: false, Note: "safetensors-side patch flattening/input projection, positional table, std scale/bias, optional preloaded/streaming tower-prefix image-embedding entrypoints, and embed_vision projection boundary; GGUF Q4_K_M has no vision tensors"},
		{Kind: OpVisionEmbeddingInsert, Domain: "vision", Implemented: true, ReferenceComplete: false, Note: "image embedding insertion boundary is implemented; full tower output is not reference-backed yet"},
		{Kind: OpVisionEncoderTower, Domain: "vision", Implemented: true, ReferenceComplete: false, Note: "27-layer vision transformer graph/bindings and full streaming entrypoint exist; full-depth seq=1 and 1-patch image embedding smokes pass; full image-sequence reference fixtures remain pending; current GGUF Q4_K_M is text-only"},
	}
}

func OperationStatusSummary() (implemented, referenceComplete, total int) {
	return OperationStatusSummaryFromStatuses(OperationStatuses())
}

func OperationStatusSummaryFromStatuses(ops []OpStatus) (implemented, referenceComplete, total int) {
	for _, op := range ops {
		total++
		if op.Implemented {
			implemented++
		}
		if op.ReferenceComplete {
			referenceComplete++
		}
	}
	return implemented, referenceComplete, total
}

func OperationStatusSummaryForDomain(domain string) (implemented, referenceComplete, total int) {
	ops := OperationStatuses()
	for _, op := range ops {
		if domain != "" && op.Domain != domain {
			continue
		}
		total++
		if op.Implemented {
			implemented++
		}
		if op.ReferenceComplete {
			referenceComplete++
		}
	}
	return implemented, referenceComplete, total
}
