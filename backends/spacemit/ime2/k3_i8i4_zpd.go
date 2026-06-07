package ime2

import (
	"unsafe"
)

// k3I8I4M1ZPDFused is the full asm-fused kernel + ZPD correction.
// Processes one N32 group: dot product + ZPD accumulation in one pass.
// zpd: pointer to zpd[kBlks*32] float32 for this group.
// sumCorr: pointer to sumCorr[kBlks] float32 (shared across groups).
func k3I8I4M1ZPDFused(a *byte, b *byte, c *float32, kBlks int, zpd *float32, sumCorr *float32)

// k3I8I4M1ZPD is K3I8I4M1 followed by ZPD correction in one call.
// Eliminates the separate ScaleAccF32RVV loop (32 function calls per group).
// zpd layout: [subs][32]float32 = subs*32 contiguous float32s for this group.
// sumCorr: [subs]float32 = per-subblock correction scalars.
func k3I8I4M1ZPD(a *byte, b *byte, c *float32, kBlks int, nBlks int, zpd *float32, sumCorr *float32) {
	// Core kernel
	K3I8I4M1(a, b, c, kBlks, nBlks)
	// ZPD correction using RVV SAXPY (same as ScaleAccF32RVV but inlined pointer math)
	outSlice := unsafe.Slice(c, 32)
	for sb := 0; sb < kBlks; sb++ {
		sc := *(*float32)(unsafe.Add(unsafe.Pointer(sumCorr), uintptr(sb)*4))
		if sc < -1e-6 || sc > 1e-6 {
			zpdStart := unsafe.Slice((*float32)(unsafe.Add(unsafe.Pointer(zpd), uintptr(sb*32)*4)), 32)
			ScaleAccF32RVV(outSlice, zpdStart, sc)
		}
	}
}

// K3I8I4M1GroupsZPD processes nGroups with fused ZPD correction.
// Replaces K3I8I4M1Groups + separate ZPD loop.
func K3I8I4M1GroupsZPD(a *byte, b *byte, c *float32, kBlks int, nGroups int, zpd *float32, sumCorr *float32) {
	strideB := uintptr(kBlks) * 608
	strideC := uintptr(32 * 4)
	strideZPD := uintptr(kBlks * 32 * 4) // kBlks*32 float32 per group
	for g := 0; g < nGroups; g++ {
		k3I8I4M1ZPDFused(a, b, c, kBlks, zpd, sumCorr)
		b = (*byte)(unsafe.Add(unsafe.Pointer(b), strideB))
		c = (*float32)(unsafe.Add(unsafe.Pointer(c), strideC))
		zpd = (*float32)(unsafe.Add(unsafe.Pointer(zpd), strideZPD))
	}
}
