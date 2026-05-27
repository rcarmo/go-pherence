// +build !riscv64

package main

func vmadotQ4KIntLoop1024x4(wBase *byte, wRGStride int, actBcast *byte, scratch *int32, intBuf *int32, intRGStride int, numSubs int) {
	panic("vmadotQ4KIntLoop1024x4: not implemented on this platform")
}
