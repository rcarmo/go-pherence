package aicpu

import (
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
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
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
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
			if v > 127 {
				v = 127
			} else if v < -128 {
				v = -128
			}
			actI8[sb*32+i] = int8(v)
			s += int32(actI8[sb*32+i])
		}
		actSums[sb] = s
	}

	// Process 4 rows at a time using vmadot
	var tile [32]byte    // 4×8 tile for vmadot
	var actTile [32]byte // 4×8 activation tile (broadcast)
	var acc [16]int32    // 4×4 accumulator

	for row := 0; row < M; row += 4 {
		rowCount := 4
		if row+4 > M {
			rowCount = M - row
		}

		// Zero F32 output for these rows
		for r := 0; r < rowCount; r++ {
			out[row+r] = 0
		}

		// Process each sub-block (32 elements)
		for sb := 0; sb < subsPerRow; sb++ {
			elemOff := sb * 32

			// Zero INT32 accumulator
			for i := range acc {
				acc[i] = 0
			}

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
					for i := 0; i < 8; i++ {
						tile[r*8+i] = 0
					}
				}

				// Pack activation tile: broadcast same 8 elements to 4 rows
				for r := 0; r < 4; r++ {
					copy(actTile[r*8:(r+1)*8], (*[8]byte)(unsafe.Pointer(&actI8[passOff]))[:])
				}

				// Scalar accumulate (vmadot replacement for correctness test)
				for r := 0; r < 4; r++ {
					for c2 := 0; c2 < 4; c2++ {
						var d int32
						for i := 0; i < 8; i++ {
							d += int32(int8(actTile[c2*8+i])) * int32(int8(tile[r*8+i]))
						}
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

// matVecQ4KF32 performs out[M] = W_q4k[M×K] · act[K] using direct F32 computation.
// No activation quantization — maximum precision, slower.
func matVecQ4KF32(M, K int, wRaw []int8, wScales, wMins []float32, act []float32, out []float32) {
	subsPerRow := K / 32
	for row := 0; row < M; row++ {
		var sum float32
		for sb := 0; sb < subsPerRow; sb++ {
			wScale := wScales[row*subsPerRow+sb]
			wMin := wMins[row*subsPerRow+sb]
			var dotF, actSum float32
			for i := 0; i < 32; i++ {
				nib := float32(wRaw[row*K+sb*32+i])
				dotF += nib * act[sb*32+i]
				actSum += act[sb*32+i]
			}
			sum += dotF*wScale - actSum*wMin
		}
		out[row] = sum
	}
}

// matVecF32Direct performs out[M] = W_f32[M×K] · act[K] (pure scalar F32).
func matVecF32Direct(M, K int, wF32 []float32, act []float32, out []float32) {
	for row := 0; row < M; row++ {
		var sum float32
		for k := 0; k < K; k++ {
			sum += wF32[row*K+k] * act[k]
		}
		out[row] = sum
	}
}

var _actScaleG float32

func packActG(x []float32, K int) []int8 {
	xI8 := make([]int8, K)
	var maxAbs float32
	for _, v := range x[:K] {
		a := v
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs == 0 {
		_actScaleG = 0
		return ime2.PackTiles(make([]int8, 4*K), 4, K)
	}
	_actScaleG = maxAbs / 127.0
	s := float32(127.0) / maxAbs
	for i := 0; i < K; i++ {
		v := x[i] * s
		if v > 127 {
			v = 127
		} else if v < -128 {
			v = -128
		}
		xI8[i] = int8(v)
	}
	bc := make([]int8, 4*K)
	copy(bc[0:K], xI8)
	copy(bc[K:2*K], xI8)
	copy(bc[2*K:3*K], xI8)
	copy(bc[3*K:4*K], xI8)
	return ime2.PackTiles(bc, 4, K)
}

// MatVecBufs holds pre-allocated buffers for matVecFast (zero alloc in hot path).
type MatVecBufs struct {
	actPad []float32
	xI8    []int8
	bc     []int8
	packed []int8
	res    []int32
}

func NewMatVecBufs(maxK, maxM int) *MatVecBufs {
	Kp := ((maxK + 7) / 8) * 8
	Mp := ((maxM + 3) / 4) * 4
	return &MatVecBufs{
		actPad: make([]float32, Kp),
		xI8:    make([]int8, Kp),
		bc:     make([]int8, 4*Kp),
		packed: make([]int8, 4*Kp),
		res:    make([]int32, Mp*4),
	}
}

func matVecFast(M, K int, f32 []float32, packed []int8, scale float32, act []float32, out []float32) {
	matVecFastBuf(M, K, f32, packed, scale, act, out, globalBufs)
}

var globalBufs *MatVecBufs
var globalPool *ime2.WorkerPool

func matVecFastBuf(M, K int, f32 []float32, packed []int8, scale float32, act []float32, out []float32, bufs *MatVecBufs) {
	if packed != nil && len(packed) > 0 && bufs != nil {
		Kp := ((K + 7) / 8) * 8
		Mp := ((M + 3) / 4) * 4
		// Zero-alloc activation quantize + pack
		actPad := bufs.actPad[:Kp]
		copy(actPad, act[:K])
		for i := K; i < Kp; i++ {
			actPad[i] = 0
		}
		// Quantize
		var maxAbs float32
		for _, v := range actPad {
			a := v
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		if maxAbs == 0 {
			for i := range out[:M] {
				out[i] = 0
			}
			return
		}
		_actScaleG = maxAbs / 127.0
		s := float32(127.0) / maxAbs
		xI8 := bufs.xI8[:Kp]
		for i := 0; i < Kp; i++ {
			v := actPad[i] * s
			if v > 127 {
				v = 127
			} else if v < -128 {
				v = -128
			}
			xI8[i] = int8(v)
		}
		// Broadcast-pack using RVV (vectorized 4× copy)
		pk := bufs.packed[:4*Kp]
		ime2.BroadcastPackRVV(xI8[:Kp], Kp, pk)
		// GEMM
		res := bufs.res[:Mp*4]
		for i := range res {
			res[i] = 0
		}
		if globalPool != nil && Mp >= 512 {
			ime2.GemmINT8PackedPool(Mp, 4, Kp, packed, pk, res, globalPool)
		} else {
			ime2.GemmINT8Packed(Mp, 4, Kp, packed, pk, res)
		}
		for i := 0; i < M; i++ {
			out[i] = float32(res[i*4]) * scale * _actScaleG
		}
	} else {
		for row := 0; row < M; row++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += f32[row*K+k] * act[k]
			}
			out[row] = sum
		}
	}
}
