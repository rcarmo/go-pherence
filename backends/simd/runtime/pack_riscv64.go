//go:build riscv64

package simd

import "unsafe"

const hasNeonPack = false
const hasAvxPack = false

func packBNTAsm(
	b0, b1, b2, b3, b4, b5, b6, b7,
	b8, b9, b10, b11, b12, b13, b14, b15 uintptr,
	k int, bp uintptr) {
	if k <= 0 || bp == 0 {
		return
	}
	rows := [gebpNR]uintptr{b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15}
	out := unsafe.Pointer(bp)
	for p := 0; p < k; p++ {
		for d := 0; d < gebpNR; d++ {
			if rows[d] == 0 {
				continue
			}
			v := loadF32(unsafe.Pointer(rows[d]), p)
			storeF32(out, p*gebpNR+d, v)
		}
	}
}
