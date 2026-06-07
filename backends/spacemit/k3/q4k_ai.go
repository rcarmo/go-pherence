package k3

import (
	"fmt"
	"math"
	"os"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

type q4kQ41Packed struct {
	M, K  int
	Q     []int8
	D     []float32
	ZP    []int8
	Valid bool
}

type q4kQ41x32 struct {
	M, K     int
	D        []float32 // [rowGroup][subblock][32 rows]
	ZP       []uint8   // [rowGroup][subblock][32 rows]
	ZPD      []float32 // [rowGroup][subblock][32 rows]: float32(ZP)*D — precomputed for fast correction
	Residual []float32 // [rowGroup][subblock][32 rows]: dmin - ZP*D (exact min residual)
	QS       []byte    // [rowGroup][subblock][32 rows][16 packed q bytes]
	BData    []byte    // kernel layout: fp16 d[32]=64B, int8 zp[32]=32B, qs[512]=512B per subblock (608B)
	Valid    bool
}

func q41x32MetaIndex(rowGroup, subblock, rowInGroup, subs int) int {
	return (rowGroup*subs+subblock)*32 + rowInGroup
}

func q41x32QSOffset(rowGroup, subblock, rowInGroup, subs int) int {
	return ((rowGroup*subs+subblock)*32 + rowInGroup) * 16
}

func f32ToF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		rounded := (mant + (1 << (shift - 1))) >> shift
		return sign | uint16(rounded)
	}
	if exp >= 31 {
		return sign | 0x7c00
	}
	rounded := mant + 0x1000
	if rounded&0x800000 != 0 {
		rounded = 0
		exp++
		if exp >= 31 {
			return sign | 0x7c00
		}
	}
	return sign | uint16(exp<<10) | uint16(rounded>>13)
}

func f32ToF16ToF32(f float32) float32 { return fp16ToFloat(f32ToF16Bits(f)) }

// repackQ4KToQ41x32 ports llama.cpp PR #22863's Q4_K -> Q4_1x32 repack.
func repackQ4KToQ41x32(M, K int, raw []int8, scales, mins []float32) q4kQ41x32 {
	if M%32 != 0 || K%32 != 0 || scales == nil || mins == nil {
		return q4kQ41x32{}
	}
	subs := K / 32
	groups := M / 32
	out := q4kQ41x32{
		M:     M,
		K:     K,
		D:        make([]float32, groups*subs*32),
		ZP:       make([]uint8, groups*subs*32),
		ZPD:      make([]float32, groups*subs*32),
		Residual: make([]float32, groups*subs*32), // dmin - ZP*D per sub-block/row
		QS:       make([]byte, groups*subs*32*16),
		BData: make([]byte, groups*subs*(64+32+512)), // 608B per subblock
		Valid: true,
	}
	for rg := 0; rg < groups; rg++ {
		for sb := 0; sb < subs; sb++ {
			for r := 0; r < 32; r++ {
				row := rg*32 + r
				srcIdx := row*subs + sb
				metaIdx := q41x32MetaIndex(rg, sb, r, subs)
				d := f32ToF16ToF32(scales[srcIdx])
				m := f32ToF16ToF32(mins[srcIdx])
				out.D[metaIdx] = d
				if d != 0 {
					zp := math.Round(float64(m / d))
					if zp < 0 {
						zp = 0
					} else if zp > 15 {
						zp = 15
					}
					out.ZP[metaIdx] = uint8(zp)
					out.ZPD[metaIdx] = float32(uint8(zp)) * d // precomputed ZP×D
					out.Residual[metaIdx] = mins[srcIdx] - float32(uint8(zp))*d // exact residual = dmin - ZP*D
				}
				qsOff := q41x32QSOffset(rg, sb, r, subs)
				base := row*K + sb*32
				for i := 0; i < 16; i++ {
					lo := byte(raw[base+2*i]) & 0x0f
					hi := (byte(raw[base+2*i+1]) & 0x0f) << 4
					out.QS[qsOff+i] = lo | hi
				}
			}
		}
	}
	for rg := 0; rg < groups; rg++ {
		for sb := 0; sb < subs; sb++ {
			blkOff := (rg*subs + sb) * (64 + 32 + 512) // 608B per subblock
			for r := 0; r < 32; r++ {
				metaIdx := q41x32MetaIndex(rg, sb, r, subs)
				dh := f32ToF16Bits(out.D[metaIdx])
				out.BData[blkOff+r*2] = byte(dh)
				out.BData[blkOff+r*2+1] = byte(dh >> 8)
				out.BData[blkOff+64+r] = 0 // zp=0; min correction applied externally
			}
			// QS: column-major layout — column r at offset 96+r*16, 16 bytes per column
			// vmadotsu/vmadotu.hp process 8 columns at a time from v4/v5/v6/v7
			for r := 0; r < 32; r++ {
				qsBase := q41x32QSOffset(rg, sb, r, subs)
				copy(out.BData[blkOff+96+r*16:blkOff+96+(r+1)*16], out.QS[qsBase:qsBase+16])
			}
		}
	}
	return out
}

// repackQ4KToQ41A100 mirrors llama.cpp PR #22863's Q4_K -> Q4_1-style
// repack at the semantic level: each 32-value Q4_K subblock becomes q values
// plus d and a rounded zero point zp = nearbyint(min/d), clamped to [0,15].
// Q is additionally packed into native A100 8×16 tiles for vmadot.
func repackQ4KToQ41A100(M, K int, raw []int8, scales, mins []float32) q4kQ41Packed {
	if M%8 != 0 || K%32 != 0 || scales == nil || mins == nil {
		return q4kQ41Packed{}
	}
	subsPerRow := K / 32
	zp := make([]int8, M*subsPerRow)
	for row := 0; row < M; row++ {
		for sb := 0; sb < subsPerRow; sb++ {
			idx := row*subsPerRow + sb
			d := scales[idx]
			if d == 0 {
				continue
			}
			z := float32(math.Round(float64(mins[idx] / d)))
			if z < 0 {
				z = 0
			} else if z > 15 {
				z = 15
			}
			zp[idx] = int8(z)
		}
	}
	return q4kQ41Packed{
		M:     M,
		K:     K,
		Q:     ime2.PackTiles1024(raw, M, K),
		D:     scales,
		ZP:    zp,
		Valid: true,
	}
}

// q4kBlockMatVecAI computes out[M] = W_q4k[M×K] · act[K] using Q4_K
// per-subblock scales/mins and native A100 VLEN=1024 vmadot tiles.
//
// wRaw contains unpacked Q4_K values in [0,15] in logical row-major order.
// scales/mins contain one entry per 32-column subblock per row.
func q4kBlockMatVecAI(M, K int, wRaw []int8, scales, mins []float32, act []float32, out []float32, pool *AIWorkerPool) {
	if K%32 != 0 || M%8 != 0 {
		// Fallback for odd shapes; model shapes should not hit this.
		matVecQ4KF32(M, K, wRaw, scales, mins, act, out)
		return
	}
	wPacked := ime2.PackTiles1024(wRaw, M, K)
	q4kBlockMatVecAIPacked(M, K, wPacked, scales, mins, act, out, pool)
}

// q4kBlockMatVecAIPacked is the same kernel using persistent native 8×16
// raw-Q4 tiles (the output of PackTiles1024 on unpacked Q4 nibbles).
func q4kBlockMatVecAIPacked(M, K int, wPacked []int8, scales, mins []float32, act []float32, out []float32, pool *AIWorkerPool) {
	if K%32 != 0 || M%8 != 0 {
		panic("q4kBlockMatVecAIPacked: unsupported shape")
	}
	if q4kScaledLoopOn && scales != nil && mins != nil {
		// Fast path: quantize activations per-subblock then call the single-dispatch kernel.
		subs := K / 32
		actI8 := make([]int8, K)
		actScale := make([]float32, subs)
		// actSumScaled[sb] = float32(sum(actI8[sb])) * actScale[sb]
		// This matches Q4K_A100's min-correction: float32(actSum)*minTerm*as.
		actSumScaled := make([]float32, subs)
		for sb := 0; sb < subs; sb++ {
			base := sb * 32
			var maxAbs float32
			for i := 0; i < 32; i++ {
				a := act[base+i]
				if a < 0 { a = -a }
				if a > maxAbs { maxAbs = a }
			}
			if maxAbs == 0 { continue }
			actScale[sb] = maxAbs / 127.0
			s := float32(127.0) / maxAbs
			var isum int32
			for i := 0; i < 32; i++ {
				v := act[base+i] * s
				if v > 127 { v = 127 } else if v < -128 { v = -128 }
				q := int8(v)
				actI8[base+i] = q
				isum += int32(q)
			}
			actSumScaled[sb] = float32(isum) * actScale[sb]
		}
		q4kBlockMatVecScaledLoop(M, K, wPacked, scales, mins, actI8, actScale, actSumScaled, out, pool)
		return
	}
	q41 := q4kQ41Packed{M: M, K: K, Q: wPacked, D: scales, Valid: true}
	returnQ4KBlockMatVecQ41(q41, mins, act, out, pool)
}

func q4kBlockMatVecQ41(q41 q4kQ41Packed, act []float32, out []float32, pool *AIWorkerPool) {
	returnQ4KBlockMatVecQ41(q41, nil, act, out, pool)
}

func returnQ4KBlockMatVecQ41(q41 q4kQ41Packed, exactMins []float32, act []float32, out []float32, pool *AIWorkerPool) {
	M, K := q41.M, q41.K
	if !q41.Valid || K%32 != 0 || M%8 != 0 {
		panic("q4kBlockMatVecQ41: unsupported shape")
	}
	wPacked := q41.Q
	scales := q41.D
	subsPerRow := K / 32
	actI8 := make([]int8, K)
	actScale := make([]float32, subsPerRow)
	actSum := make([]int32, subsPerRow)

	for sb := 0; sb < subsPerRow; sb++ {
		base := sb * 32
		var maxAbs float32
		for i := 0; i < 32; i++ {
			a := act[base+i]
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		if maxAbs == 0 {
			continue
		}
		actScale[sb] = maxAbs / 127.0
		s := float32(127.0) / maxAbs
		var sum int32
		for i := 0; i < 32; i++ {
			v := act[base+i] * s
			if v > 127 {
				v = 127
			} else if v < -128 {
				v = -128
			}
			q := int8(v)
			actI8[base+i] = q
			sum += int32(q)
		}
		actSum[sb] = sum
	}

	// Pre-broadcast activation tiles once (outside row loop) to avoid O(M/8 * K/16) copies.
	tilesPerRow := K / 16
	actBcast := make([]int8, tilesPerRow*128) // [K/16 × 128]: each 16 act values broadcast to 8 rows
	for tileIdx := 0; tileIdx < tilesPerRow; tileIdx++ {
		kBase := tileIdx * 16
		for r := 0; r < 8; r++ {
			copy(actBcast[tileIdx*128+r*16:tileIdx*128+(r+1)*16], actI8[kBase:kBase+16])
		}
	}

	pool.Run(func(workerID, nWorkers int) {
		rowStart := (workerID * M / nWorkers / 8) * 8
		rowEnd := ((workerID + 1) * M / nWorkers / 8) * 8
		if workerID == nWorkers-1 {
			rowEnd = M
		}
		for row := rowStart; row < rowEnd; row += 8 {
			for r := 0; r < 8 && row+r < M; r++ {
				out[row+r] = 0
			}
			for sb := 0; sb < subsPerRow; sb++ {
				as := actScale[sb]
				if as == 0 {
					continue
				}
				// Two native 8×16 vmadot passes cover one 32-element Q4_K subblock.
				var dots [8]int32
				for half := 0; half < 2; half++ {
					kBase := sb*32 + half*16
					tileIdx := kBase / 16
					var acc [64]int32
					wOff := ((row/8)*tilesPerRow + tileIdx) * 128
					ime2.VmadotKLoop1024(
						(*byte)(unsafe.Pointer(&wPacked[wOff])),
						(*byte)(unsafe.Pointer(&actBcast[tileIdx*128])),
						&acc[0], 16,
					)
					for r := 0; r < 8; r++ {
						dots[r] += acc[r*8]
					}
				}
				for r := 0; r < 8 && row+r < M; r++ {
					idx := (row+r)*subsPerRow + sb
					var minTerm float32
					if exactMins != nil && !q4kLlamaZPOn {
						minTerm = exactMins[idx]
					} else {
						minTerm = scales[idx] * float32(q41.ZP[idx])
					}
					out[row+r] += float32(dots[r])*scales[idx]*as - float32(actSum[sb])*minTerm*as
				}
			}
		}
	})
}

// applyQ4KMinCorr adds the Q4_K minimum correction to a matmul output:
//   out[row] -= sum_sb { mins[row*subs+sb] * sumAct[sb] }
// where sumAct[sb] = sum(act[sb*32 : (sb+1)*32]).
// This corrects the missing "-min" term when raw nibbles are used without dequant.
func applyQ4KMinCorr(out []float32, mins []float32, act []float32, M, K int) {
	if mins == nil {
		return
	}
	subs := K / 32
	// Pre-compute per-subblock activation sums once.
	sumAct := make([]float32, subs)
	for sb := 0; sb < subs; sb++ {
		var s float32
		base := sb * 32
		for i := 0; i < 32; i++ {
			s += act[base+i]
		}
		sumAct[sb] = s
	}
	for row := 0; row < M; row++ {
		mOff := row * subs
		var corr float32
		for sb := 0; sb < subs; sb++ {
			corr += mins[mOff+sb] * sumAct[sb]
		}
		out[row] -= corr
	}
}

// q4kBlockMatVecScaledLoop runs a Q4_K-correct M×K matrix-vector multiply
// using vmadotQ4KIntLoop1024: one Go function call per 8-row group for all
// subblocks in assembly, then float scaling in scalar Go.
//
// Requires M%8==0, K%32==0, scales and mins non-nil.
// actI8 must be pre-quantized per-subblock with actScale[sb] = maxAbs[sb]/127.
// q4kBlockMatVecScaledLoop runs all K/32 subblocks in one assembly dispatch per row group.
// actSumScaled[sb] must be float32(sum(actI8[sb])) * actScale[sb] to match Q4K_A100 correction.
func q4kBlockMatVecScaledLoop(M, K int, wPacked []int8, scales, mins []float32,
	actI8 []int8, actScale []float32, actSumScaled []float32, out []float32, pool *AIWorkerPool) {
	if M%8 != 0 || K%32 != 0 {
		panic("q4kBlockMatVecScaledLoop: unsupported shape")
	}
	subs := K / 32
	tilesPerRow := K / 16 // = subs * 2

	// Pre-broadcast activation tiles once (K/16 × 128 bytes).
	actBcast := make([]int8, tilesPerRow*128)
	for t := 0; t < tilesPerRow; t++ {
		kBase := t * 16
		for r := 0; r < 8; r++ {
			copy(actBcast[t*128+r*16:t*128+(r+1)*16], actI8[kBase:kBase+16])
		}
	}

	slCompareOn := os.Getenv("IME2_Q4K_SL_COMPARE") != ""
	pool.Run(func(workerID, nWorkers int) {
		rowStart := (workerID * M / nWorkers / 8) * 8
		rowEnd := ((workerID + 1) * M / nWorkers / 8) * 8
		if workerID == nWorkers-1 {
			rowEnd = M
		}
		// Per-worker buffers: int32 results, scratch (v28 dump, 64 int32).
		intBuf := make([]int32, subs*8)
		scratch := make([]int32, 64) // overwritten each subblock

		for row := rowStart; row < rowEnd; row += 8 {
			// One assembly call processes all subs; eliminates per-tile Go call overhead.
			vmadotQ4KIntLoop1024(
				(*byte)(unsafe.Pointer(&wPacked[(row/8)*tilesPerRow*128])),
				(*byte)(unsafe.Pointer(&actBcast[0])),
				&scratch[0],
				&intBuf[0],
				subs,
			)
			// Apply per-subblock scale and min corrections in scalar Go.
			for r := 0; r < 8; r++ {
				out[row+r] = 0
			}
			rowBase := row * subs
			for sb := 0; sb < subs; sb++ {
				as := actScale[sb]
				sf := actSumScaled[sb]
				sb8 := sb * 8
				sbOff := rowBase + sb
				for r := 0; r < 8; r++ {
					out[row+r] += scales[sbOff+r*subs] * as * float32(intBuf[sb8+r])
					out[row+r] -= mins[sbOff+r*subs] * sf
				}
			}
			// Diagnostic: compare intBuf against tile-by-tile reference (IME2_Q4K_SL_COMPARE=1).
			if slCompareOn && row == rowStart {
				var atile [128]int8
				wOff := (row / 8) * tilesPerRow * 128
				for sb := 0; sb < subs; sb++ {
					var acc [64]int32
					for half := 0; half < 2; half++ {
						kBase := sb*32 + half*16
						for rr := 0; rr < 8; rr++ {
							copy(atile[rr*16:rr*16+16], actI8[kBase:kBase+16])
						}
						ime2.VmadotKLoop1024(
							(*byte)(unsafe.Pointer(&wPacked[wOff+(sb*2+half)*128])),
							(*byte)(unsafe.Pointer(&atile[0])),
							&acc[0], 16)
					}
					for r := 0; r < 8; r++ {
						got := intBuf[sb*8+r]
						ref := acc[r*8]
						if got != ref {
							fmt.Fprintf(os.Stderr, "SL_cmp M=%d K=%d row=%d sb=%d r=%d got=%d ref=%d\n",
								M, K, row, sb, r, got, ref)
						}
					}
				}
			}
		}
	})
}
//
// Requires M%8==0, K%32==0, scales and mins non-nil.
// actI8 must be pre-quantized per-subblock with actScale[sb] = maxAbs[sb]/127.
