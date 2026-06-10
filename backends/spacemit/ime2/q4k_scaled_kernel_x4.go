//go:build riscv64

package ime2

// vmadotQ4KIntLoop1024x4 processes 4 consecutive 8-row groups per call,
// loading actBcast ONCE per tile (vs 4× for separate calls).
// Reduces actBcast DRAM pressure 4×, targeting the down-projection bottleneck.
//
// wBase:       pointer to first row group's packed weights
// wRGStride:   byte stride between row groups (tilesPerRow * 128)
// actBcast:    shared broadcast activation (K/16 × 128 bytes)
// scratch:     [64]int32 scratch buffer (overwritten each subblock)
// intBuf:      output [subs*8]int32 for rg0; rg1-3 follow at intRGStride offsets
// intRGStride: byte stride between row groups in intBuf (subs*8*4 bytes)
// numSubs:     K/32 (number of subblocks)
//
//go:noescape
func vmadotQ4KIntLoop1024x4(wBase *byte, wRGStride int, actBcast *byte, scratch *int32, intBuf *int32, intRGStride int, numSubs int)
