package k3

import "unsafe"

// k3I8I4M1 is a direct Go/asm port of llama.cpp's IME2 i8×i4 M1/N32 kernel.
// Must be called from an AI-registered worker goroutine (cores 8-15).
func k3I8I4M1(a *byte, b *byte, c *float32, kBlks int, nBlks int)

// k3I8I4M1C is the C-correction-order variant of k3I8I4M1.
func k3I8I4M1C(a *byte, b *byte, c *float32, kBlks int, nBlks int)

// k3I8I4M1CResidual fuses exact residual correction into k3I8I4M1C.
func k3I8I4M1CResidual(a *byte, b *byte, residual *float32, c *float32, kBlks int, nBlks int)

// k3I8I4M4 processes 4 A rows (M4 tile) against one N32 B tile.
// Native assembly using smt.vmadotsu/u.hp with VLEN=1024.
// a layout: [fp32 scale[4]][int16 sum[4]][int8 q[4][32]] per K32 block (152B)
// b: N32 tile B pointer (608B subblocks)
// c: C base pointer
// ldcBytes: C row stride in bytes (ldc * 4)
func k3I8I4M4(a *byte, b *byte, c *float32, kBlks int, ldcBytes int)

// k3I8I4M4Fallback is the pure-Go fallback that unpacks M4 into 4×M1.
// Kept for debugging/validation against the native M4 kernel.
func k3I8I4M4Fallback(a *byte, b *byte, c *float32, kBlks int, ldcBytes int) {
	const m1AStride = 38
	const m4AStride = 4*4 + 4*2 + 4*32

	rows := [4][]byte{}
	for r := 0; r < 4; r++ {
		rows[r] = make([]byte, kBlks*m1AStride)
	}
	aPtr := a
	for k := 0; k < kBlks; k++ {
		scales := (*[4]float32)(unsafe.Pointer(aPtr))
		aPtr = (*byte)(unsafe.Add(unsafe.Pointer(aPtr), 16))
		sums := (*[4]int16)(unsafe.Pointer(aPtr))
		aPtr = (*byte)(unsafe.Add(unsafe.Pointer(aPtr), 8))
		for r := 0; r < 4; r++ {
			off := k * m1AStride
			*(*float32)(unsafe.Pointer(&rows[r][off])) = scales[r]
			*(*int16)(unsafe.Pointer(&rows[r][off+4])) = sums[r]
			copy(rows[r][off+6:off+6+32], unsafe.Slice(aPtr, 32))
			aPtr = (*byte)(unsafe.Add(unsafe.Pointer(aPtr), 32))
		}
	}
	cRow := c
	for r := 0; r < 4; r++ {
		k3I8I4M1(&rows[r][0], b, cRow, kBlks, 32)
		cRow = (*float32)(unsafe.Add(unsafe.Pointer(cRow), ldcBytes))
	}
}

// k3I8I4M1Residual applies exact Q4_K min correction after k3I8I4M1.
func k3I8I4M1Residual(a *byte, b *byte, residual *float32, c *float32, kBlks int, nBlks int) {
	k3I8I4M1(a, b, c, kBlks, nBlks)
}

// k3I8I4M1ResidualGroups processes nGroups contiguous N32 groups with residual.
func k3I8I4M1ResidualGroups(a *byte, b *byte, residual *float32, c *float32, kBlks int, nGroups int) {
	k3I8I4M1ResidualGroupsImpl(a, b, residual, c, kBlks, nGroups)
}

// k3I8I4M1Groups processes nGroups contiguous N32 groups without residual.
func k3I8I4M1Groups(a *byte, b *byte, c *float32, kBlks int, nGroups int) {
	k3I8I4M1GroupsImpl(a, b, c, kBlks, nGroups)
}
