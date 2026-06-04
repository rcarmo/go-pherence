// +build !riscv64

package main

// Stub for non-RISC-V platforms.
func vmadotQ4KIntLoop1024(wTiles, actBcast *byte, scratch, intBuf *int32, numSubs int) {
	panic("vmadotQ4KIntLoop1024: not implemented on this platform")
}
