package main

import "unsafe"

// k3I8I4M1 is a direct Go/asm port of llama.cpp's IME2 i8×i4 M1/N32 kernel.
// Must be called from an AI-registered worker goroutine (cores 8-15).
func k3I8I4M1(a *byte, b *byte, c *float32, kBlks int, nBlks int)

// k3I8I4M1C is the C-correction-order variant of k3I8I4M1.
func k3I8I4M1C(a *byte, b *byte, c *float32, kBlks int, nBlks int)

// k3I8I4M1CResidual fuses exact residual correction into k3I8I4M1C.
func k3I8I4M1CResidual(a *byte, b *byte, residual *float32, c *float32, kBlks int, nBlks int)

// k3I8I4M1Residual applies exact Q4_K min correction after k3I8I4M1.
// Falls back to k3I8I4M1 (residual correction applied separately if needed).
func k3I8I4M1Residual(a *byte, b *byte, residual *float32, c *float32, kBlks int, nBlks int) {
	// Residual-fused assembly not available in this build; call M1 baseline.
	// The caller must apply residual correction separately if q4kExactOn.
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

// k3I8I4M4 processes 4 A rows (M4 tile) against one N32 B tile.
// Falls back to 4 sequential k3I8I4M1 calls.
// a layout: [fp32 scale[4]][int16 sumNeg[4]][int8 q[4][32]] per K32 block (152B)
// b/c/ldcBytes: N32 tile B pointer, C base, C row stride in bytes.
func k3I8I4M4(a *byte, b *byte, c *float32, kBlks int, ldcBytes int) {
	const m1AStride = 38 // fp32 + i16 + 32 i8 per K32 block
	const m4AStride = 4*4 + 4*2 + 4*32 // 152B per K32 block: 4 scales + 4 sums + 4*32 quants

	// Unpack M4 A layout: [fp32 scale[4]][int16 sum[4]][int8[4][32]] per block
	// into 4 separate M1 A buffers: each [fp32 scale][int16 sum][int8[32]] per block.
	// Then call k3I8I4M1 for each row.
	type blk struct {
		scale float32
		sum   int16
		_pad  [2]byte
		q     [32]int8
	}
	rows := [4][]byte{}
	for r := 0; r < 4; r++ {
		rows[r] = make([]byte, kBlks*m1AStride)
	}
	// Unpack from M4 format
	aPtr := a
	for k := 0; k < kBlks; k++ {
		scales := (*[4]float32)(unsafe.Pointer(aPtr))
		aPtr = (*byte)(unsafe.Add(unsafe.Pointer(aPtr), 16)) // skip 4 float32
		sums := (*[4]int16)(unsafe.Pointer(aPtr))
		aPtr = (*byte)(unsafe.Add(unsafe.Pointer(aPtr), 8)) // skip 4 int16
		for r := 0; r < 4; r++ {
			off := k * m1AStride
			// store scale
			*(*float32)(unsafe.Pointer(&rows[r][off])) = scales[r]
			// store sum
			*(*int16)(unsafe.Pointer(&rows[r][off+4])) = sums[r]
			// copy 32 quant bytes
			copy(rows[r][off+6:off+6+32], unsafe.Slice(aPtr, 32))
			aPtr = (*byte)(unsafe.Add(unsafe.Pointer(aPtr), 32))
		}
	}
	// Run 4 M1 kernels
	cRow := c
	for r := 0; r < 4; r++ {
		k3I8I4M1(&rows[r][0], b, cRow, kBlks, 32)
		cRow = (*float32)(unsafe.Add(unsafe.Pointer(cRow), ldcBytes))
	}
}
