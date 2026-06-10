package diffusiongemma

// OpStatus records implementation/readiness at the semantic operation level.
type OpStatus struct {
	Kind              OpKind `json:"kind"`
	Implemented       bool   `json:"implemented"`
	ReferenceComplete bool   `json:"reference_complete"`
	Note              string `json:"note,omitempty"`
}

func OperationStatuses() []OpStatus {
	return []OpStatus{
		{Kind: OpCanvasEmbedding, Implemented: true, ReferenceComplete: false, Note: "row-wise BF16/F16/F32 embedding load scaffold"},
		{Kind: OpSelfCondition, Implemented: true, ReferenceComplete: false, Note: "soft-embedding feedback scaffold; needs parity"},
		{Kind: OpInputNorm, Implemented: true, ReferenceComplete: false, Note: "SIMD RMSNorm"},
		{Kind: OpSelfAttention, Implemented: true, ReferenceComplete: false, Note: "canvas/encoder attention scaffold with RoPE and sliding mask; needs parity"},
		{Kind: OpPostAttention, Implemented: true, ReferenceComplete: false, Note: "SIMD RMSNorm"},
		{Kind: OpDenseMLP, Implemented: true, ReferenceComplete: false, Note: "correctness-first full-matrix GEMV scaffold"},
		{Kind: OpPreMoE, Implemented: true, ReferenceComplete: false, Note: "SIMD RMSNorm"},
		{Kind: OpRouter, Implemented: true, ReferenceComplete: false, Note: "router scores and top-k scaffold"},
		{Kind: OpExperts, Implemented: true, ReferenceComplete: false, Note: "selected expert MLP scaffold; needs weighting parity"},
		{Kind: OpPostMoE, Implemented: true, ReferenceComplete: false, Note: "SIMD RMSNorm"},
		{Kind: OpLayerScalar, Implemented: true, ReferenceComplete: false, Note: "scalar hidden scaling"},
		{Kind: OpFinalNorm, Implemented: true, ReferenceComplete: false, Note: "SIMD RMSNorm"},
		{Kind: OpLMHead, Implemented: true, ReferenceComplete: false, Note: "row-wise tied embedding projection scaffold"},
	}
}

func OperationStatusSummary() (implemented, referenceComplete, total int) {
	ops := OperationStatuses()
	total = len(ops)
	for _, op := range ops {
		if op.Implemented {
			implemented++
		}
		if op.ReferenceComplete {
			referenceComplete++
		}
	}
	return implemented, referenceComplete, total
}
