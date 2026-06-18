//go:build ggml && cgo && linux

package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/backends/ggmlcompute"
	"github.com/rcarmo/go-pherence/half"
)

func TestGGMLFlashAttnF16KVReferenceAgainstGGML(t *testing.T) {
	const headDim, seqLen, nHead, nKV = 256, 5, 8, 2
	q := syntheticOracleVec(headDim*nHead, 0.011)
	k := syntheticOracleVec(headDim*seqLen*nKV, -0.007)
	v := syntheticOracleVec(headDim*seqLen*nKV, 0.009)
	kF16, vF16, kRounded, vRounded := ggmlF16KVFromGoCache(k, v, seqLen, nKV, headDim)
	gotGGML := make([]float32, len(q))
	if err := ggmlcompute.FlashAttnF32F16(gotGGML, q, kF16, vF16, nil, headDim, seqLen, nHead, nKV, 1.0); err != nil {
		t.Fatal(err)
	}
	gotRef := ggmlFlashAttnF16KVReference(q, kRounded, vRounded, seqLen, nHead, nKV, headDim, 1.0)
	maxDiff, meanDiff := maxMeanAbsDiff(gotGGML, gotRef)
	t.Logf("pure Go F16 flash ref vs ggml max=%g mean=%g", maxDiff, meanDiff)
	if maxDiff > 1e-3 {
		t.Fatalf("pure Go F16 flash ref vs ggml max=%g mean=%g", maxDiff, meanDiff)
	}
}

func TestGGMLFlashAttnF16KVReferenceUsesF16Storage(t *testing.T) {
	const headDim, seqLen, nHead, nKV = 8, 3, 2, 1
	q := syntheticOracleVec(headDim*nHead, 0.13)
	k := syntheticOracleVec(headDim*seqLen*nKV, 0.17)
	v := syntheticOracleVec(headDim*seqLen*nKV, -0.19)
	_, _, kRounded, vRounded := ggmlF16KVFromGoCache(k, v, seqLen, nKV, headDim)
	got := ggmlFlashAttnF16KVReference(q, kRounded, vRounded, seqLen, nHead, nKV, headDim, 1.0)
	gotF32 := gqaAttentionScale(q, kRounded, vRounded, seqLen, nHead, nKV, headDim, 1.0)
	maxDiff, _ := maxMeanAbsDiff(got, gotF32)
	if maxDiff == 0 {
		t.Fatal("F16 storage reference unexpectedly equals F32 attention")
	}
	_ = half.F32ToF16 // keep half import tied to the storage contract documentation
}
