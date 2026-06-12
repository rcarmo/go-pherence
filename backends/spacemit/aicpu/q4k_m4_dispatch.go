package aicpu

import (
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

// q4kQ41x32MatVec4GoAsm computes four Q4_K matvecs against the same packed
// B matrix using the native-style ime2.K3I8I4 dispatcher. The dispatcher selects
// ime2.K3I8I4M4 for countM>=4, matching llama.cpp gemm_kernel_i8i4().
//
// This is not used by single-token decode (countM=1), but gives the M4 port a
// production-facing entrypoint for batched/prefill work and regression tests.
func q4kQ41x32MatVec4GoAsm(w q4kQ41x32, acts [4][]float32, outs [4][]float32) bool {
	if !w.Valid || w.K%32 != 0 || w.M%32 != 0 {
		return false
	}
	for i := 0; i < 4; i++ {
		if len(acts[i]) < w.K || len(outs[i]) < w.M {
			return false
		}
	}
	subs := w.K / 32
	groups := w.M / 32
	packedA := quantizeQ8RowsM4Bytes(acts, subs)
	aPtr := (*byte)(unsafe.Pointer(&packedA[0]))
	for rg := 0; rg < groups; rg++ {
		var tmp [4 * 32]float32
		bPtr := (*byte)(unsafe.Pointer(&w.BData[rg*subs*608]))
		handled := ime2.K3I8I4(aPtr, bPtr, &tmp[0], 4, 32, subs, 32)
		if handled != 4 {
			return false
		}
		for r := 0; r < 4; r++ {
			copy(outs[r][rg*32:(rg+1)*32], tmp[r*32:(r+1)*32])
		}
	}
	return true
}
