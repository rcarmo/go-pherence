package nvidia

import (
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-pherence/internal/checked"
)

var attnIndependentFn CUfunction

// IndependentBranchAttentionBuffer computes one query per branch over one
// shared trunk and a branch-local suffix arena. No branch can address another
// branch's suffix rows.
func IndependentBranchAttentionBuffer(out, q, trunkK, trunkV, suffixK, suffixV, active *Buffer, batch, branchSlots, trunkLen, suffixStride, seqStart, seqLen, nHeads, nKVHeads, headDim int, scale float32) error {
	qDim, okQ := checked.MulInt(nHeads, headDim)
	kvDim, okKV := checked.MulInt(nKVHeads, headDim)
	qN, okQN := checked.MulInt(batch, qDim)
	trunkN, okTrunk := checked.MulInt(trunkLen, kvDim)
	suffixRows, okRows := checked.MulInt(branchSlots, suffixStride)
	suffixN, okSuffix := checked.MulInt(suffixRows, kvDim)
	if !okQ || !okKV || !okQN || !okTrunk || !okRows || !okSuffix || batch <= 0 || branchSlots < batch || trunkLen <= 0 || suffixStride < 0 || seqStart < 0 || seqLen <= 0 || seqLen > 2048 || seqStart+seqLen > trunkLen+suffixStride || nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 || nHeads%nKVHeads != 0 || out == nil || q == nil || trunkK == nil || trunkV == nil || active == nil || out.Ptr == 0 || q.Ptr == 0 || trunkK.Ptr == 0 || trunkV.Ptr == 0 || active.Ptr == 0 || out.Size < qN*4 || q.Size < qN*4 || trunkK.Size < trunkN*4 || trunkV.Size < trunkN*4 || active.Size < batch*4 || suffixStride > 0 && (suffixK == nil || suffixV == nil || suffixK.Ptr == 0 || suffixV.Ptr == 0 || suffixK.Size < suffixN*4 || suffixV.Size < suffixN*4) || attnIndependentFn == 0 || !fitsUint32(batch) || !fitsUint32(trunkLen) || !fitsUint32(suffixStride) || !fitsUint32(seqStart) || !fitsUint32(seqLen) || !fitsUint32(nHeads) || !fitsUint32(nKVHeads) || !fitsUint32(headDim) {
		return fmt.Errorf("invalid independent branch attention batch=%d seq=%d heads=%d kv=%d dim=%d", batch, seqLen, nHeads, nKVHeads, headDim)
	}
	if suffixK == nil {
		suffixK = &Buffer{}
	}
	if suffixV == nil {
		suffixV = &Buffer{}
	}
	bb, trunk, stride, start, ss, hh, kk, dd := uint32(batch), uint32(trunkLen), uint32(suffixStride), uint32(seqStart), uint32(seqLen), uint32(nHeads), uint32(nKVHeads), uint32(headDim)
	return LaunchKernel(attnIndependentFn, hh, bb, 1, 256, 1, 1, 2048*4,
		unsafe.Pointer(&q.Ptr), unsafe.Pointer(&trunkK.Ptr), unsafe.Pointer(&trunkV.Ptr), unsafe.Pointer(&suffixK.Ptr), unsafe.Pointer(&suffixV.Ptr), unsafe.Pointer(&active.Ptr), unsafe.Pointer(&out.Ptr),
		unsafe.Pointer(&bb), unsafe.Pointer(&trunk), unsafe.Pointer(&stride), unsafe.Pointer(&start), unsafe.Pointer(&ss), unsafe.Pointer(&hh), unsafe.Pointer(&kk), unsafe.Pointer(&dd), unsafe.Pointer(&scale))
}
