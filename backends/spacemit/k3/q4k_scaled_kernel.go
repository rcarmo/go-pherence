package k3

// vmadotQ4KIntLoop1024 performs Q4_K vmadot for one 8-row tile group across
// all K/32 subblocks in a single assembly call.
//
// For each subblock sb = 0..numSubs-1:
//   - Zeroes v28 via vmv.v.i (no memory access; uses e32,m2 vtype set at entry).
//   - Runs 2 vmadot passes (one 32-column subblock, = 2×16-col tiles).
//   - Stores v28 to scratch, extracts 8 diagonal int32 values.
//   - Stores to intBuf[sb*8+r] for r=0..7.
//
// scratch must be a [64]int32 (overwritten each subblock).
// intBuf must have capacity numSubs*8.
//
//go:noescape
func vmadotQ4KIntLoop1024(wTiles, actBcast *byte, scratch, intBuf *int32, numSubs int)

// vmadotQ4KScaledLoop1024 performs the full per-subblock Q4_K matvec update
// for one 8-row group in RISC-V/SpacemiT assembly.
//
//go:noescape
func vmadotQ4KScaledLoop1024(wTiles, actBcast *byte, scales, mins, actScale, actSumScaled, out *float32, scratch *int32, numSubs int)
