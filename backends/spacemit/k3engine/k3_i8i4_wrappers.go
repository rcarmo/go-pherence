package k3engine

import "unsafe"

// k3I8I4M1GroupsGo processes nGroups contiguous N32 output groups against
// the same A activation, striding B by kBlks*608 bytes per group.
// This is a Go-level loop wrapper around k3I8I4M1.
// NOTE: Must only be called from an AI-registered worker goroutine.
//
// b must point to BData[gStart*kBlks*608]; c to Out[gStart*32].
func k3I8I4M1GroupsImpl(a *byte, b *byte, c *float32, kBlks int, nGroups int) {
	stride := uintptr(kBlks) * 608
	for g := 0; g < nGroups; g++ {
		k3I8I4M1(a, b, c, kBlks, 32)
		b = (*byte)(unsafe.Add(unsafe.Pointer(b), stride))
		c = (*float32)(unsafe.Add(unsafe.Pointer(c), 128)) // 32 float32 = 128 bytes
	}
}

// k3I8I4M1ResidualGroupsImpl is the residual-correction variant of the groups loop.
func k3I8I4M1ResidualGroupsImpl(a *byte, b *byte, residual *float32, c *float32, kBlks int, nGroups int) {
	strideB := uintptr(kBlks) * 608
	strideC := uintptr(32 * 4)           // 32 float32 = 128 bytes
	strideRes := uintptr(kBlks * 32 * 4) // kBlks*32 float32 per group
	for g := 0; g < nGroups; g++ {
		k3I8I4M1Residual(a, b, residual, c, kBlks, 32)
		b = (*byte)(unsafe.Add(unsafe.Pointer(b), strideB))
		c = (*float32)(unsafe.Add(unsafe.Pointer(c), strideC))
		residual = (*float32)(unsafe.Add(unsafe.Pointer(residual), strideRes))
	}
}
