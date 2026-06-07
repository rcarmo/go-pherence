//go:build !riscv64
// +build !riscv64

package ime2

// Stub for non-RISC-V platforms.
func VmadotQ4KIntLoop1024(wTiles, actBcast *byte, scratch, intBuf *int32, numSubs int) {
	panic("VmadotQ4KIntLoop1024: not implemented on this platform")
}
