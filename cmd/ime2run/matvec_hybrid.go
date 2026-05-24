package main

import (
	"unsafe"


	_ "github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

// matVecQ4KVmadot performs out[M] = W_q4k[M×K] · act[K] using vmadot
// with per-sub-block scale correction. Combines vmadot speed with correct scales.
// wRaw: row-major INT8 nibbles (0-15), wScales/wMins: per-sub-block scales.
func matVecQ4KVmadot(M, K int, wRaw []int8, wScales, wMins []float32, act []float32, out []float32) {
	// Per-sub-block activation quantization (32 elements per scale)
	subsPerRow := K / 32
	actI8 := make([]int8, K)
	actInvScales := make([]float32, subsPerRow) // per-sub-block inverse scale
	actSums := make([]int32, subsPerRow)
	for sb := 0; sb < subsPerRow; sb++ {
		// Find max abs in this 32-element sub-block
		var maxAbs float32
		for i := 0; i < 32; i++ {
			a := act[sb*32+i]
			if a < 0 { a = -a }
			if a > maxAbs { maxAbs = a }
		}
		if maxAbs == 0 {
			actInvScales[sb] = 0
			continue
		}
		actScale := 127.0 / maxAbs
		actInvScales[sb] = maxAbs / 127.0
		var s int32
		for i := 0; i < 32; i++ {
			v := act[sb*32+i] * actScale
			if v > 127 { v = 127 } else if v < -128 { v = -128 }
			actI8[sb*32+i] = int8(v)
			s += int32(actI8[sb*32+i])
		}
		actSums[sb] = s
	}

	// Process 4 rows at a time using vmadot
	var tile [32]byte   // 4×8 tile for vmadot
	var actTile [32]byte // 4×8 activation tile (broadcast)
	var acc [16]int32    // 4×4 accumulator

	for row := 0; row < M; row += 4 {
		rowCount := 4
		if row+4 > M { rowCount = M - row }
		
		// Zero F32 output for these rows
		for r := 0; r < rowCount; r++ { out[row+r] = 0 }

		// Process each sub-block (32 elements)
		for sb := 0; sb < subsPerRow; sb++ {
			elemOff := sb * 32

			// Zero INT32 accumulator
			for i := range acc { acc[i] = 0 }

			// 4 vmadot passes per sub-block (32 elements / 8 per vmadot)
			for pass := 0; pass < 4; pass++ {
				passOff := elemOff + pass*8

				// Pack weight tile: 4 rows × 8 elements
				for r := 0; r < rowCount; r++ {
					wOff := (row+r)*K + passOff
					copy(tile[r*8:(r+1)*8], (*[8]byte)(unsafe.Pointer(&wRaw[wOff]))[:])
				}
				// Zero remaining rows if rowCount < 4
				for r := rowCount; r < 4; r++ {
					for i := 0; i < 8; i++ { tile[r*8+i] = 0 }
				}

				// Pack activation tile: broadcast same 8 elements to 4 rows
				for r := 0; r < 4; r++ {
					copy(actTile[r*8:(r+1)*8], (*[8]byte)(unsafe.Pointer(&actI8[passOff]))[:])
				}

				// Scalar accumulate (vmadot replacement for correctness test)
				for r := 0; r < 4; r++ {
					for c2 := 0; c2 < 4; c2++ {
						var d int32
						for i := 0; i < 8; i++ { d += int32(int8(actTile[c2*8+i])) * int32(int8(tile[r*8+i])) }
						acc[r*4+c2] += d
					}
				}
			}

			// Apply per-sub-block scale to accumulated result
			// C[i][j] contains sum(act[k] * wt[i][k]) for k in sub-block
			// The correct result for row i = acc[i][i] * wScale * actDeScale - actSum * wMin * actDeScale
			// Wait: acc is 4×4 where row i of acc has dot(actTile_row[i], wtTile_row[r])
			// Since actTile rows are all identical (broadcast), acc[r][c] = dot(act_8, wt_row_r_8)
			// for each pass. After 4 passes: acc[r][0] = full sub-block dot for row (row+r).
			// Actually acc[r][c] = dot(act_broadcast_row_c, wt_row_r). 
			// With broadcast: all act rows are the same, so acc[r][0]==acc[r][1]==acc[r][2]==acc[r][3]
			// We just need acc[r][0] for each output row r.

			for r := 0; r < rowCount; r++ {
				dotResult := acc[r*4] // first column (all columns are same due to broadcast)
				wScale := wScales[(row+r)*subsPerRow+sb]
				wMin := wMins[(row+r)*subsPerRow+sb]
				out[row+r] += float32(dotResult) * wScale * actInvScales[sb]
				out[row+r] -= float32(actSums[sb]) * wMin * actInvScales[sb]
			}
		}
	}
}

// VmadotAccSS4x8 is the public accessor for the assembly function
