package main

import (
	"fmt"
	"math"
	"os"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

type q8Block32 struct {
	Q      []int8
	Scale  []float32
	SumNeg []int32
}

func quantizeQ8Blocks32(act []float32) q8Block32 {
	subs := len(act) / 32
	q := q8Block32{Q: make([]int8, len(act)), Scale: make([]float32, subs), SumNeg: make([]int32, subs)}
	for sb := 0; sb < subs; sb++ {
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
		q.Scale[sb] = maxAbs / 127.0
		rep := float32(127.0) / maxAbs
		var sum int32
		for i := 0; i < 32; i++ {
			v := float32(math.RoundToEven(float64(act[base+i] * rep)))
			if v > 127 {
				v = 127
			} else if v < -128 {
				v = -128
			}
			qi := int8(v)
			q.Q[base+i] = qi
			sum += int32(qi)
		}
		q.SumNeg[sb] = -sum
	}
	return q
}

func q4kQ41x32MatVecRef(w q4kQ41x32, act []float32, out []float32) {
	if !w.Valid || w.K%32 != 0 || w.M%32 != 0 {
		panic("q4kQ41x32MatVecRef: invalid shape")
	}
	M, K := w.M, w.K
	subs := K / 32
	q8 := quantizeQ8Blocks32(act)
	for row := 0; row < M; row++ {
		out[row] = 0
		rg := row / 32
		ri := row % 32
		var part float32
		for sb := 0; sb < subs; sb++ {
			metaIdx := q41x32MetaIndex(rg, sb, ri, subs)
			qsOff := q41x32QSOffset(rg, sb, ri, subs)
			acc := q8.SumNeg[sb] * int32(w.ZP[metaIdx])
			for i := 0; i < 16; i++ {
				b := w.QS[qsOff+i]
				acc += int32(q8.Q[sb*32+2*i]) * int32(int8(b&0x0f))
				acc += int32(q8.Q[sb*32+2*i+1]) * int32(int8((b>>4)&0x0f))
			}
			part = f32ToF16ToF32(part + float32(acc)*q8.Scale[sb]*w.D[metaIdx])
			if sb%16 == 15 {
				out[row] += part
				part = 0
			}
		}
		out[row] += part
	}
}

func q8Block32ToBytes(q8 q8Block32) []byte {
	subs := len(q8.Scale)
	buf := make([]byte, subs*(4+2+32))
	for sb := 0; sb < subs; sb++ {
		off := sb * 38
		bits := math.Float32bits(q8.Scale[sb])
		buf[off+0] = byte(bits)
		buf[off+1] = byte(bits >> 8)
		buf[off+2] = byte(bits >> 16)
		buf[off+3] = byte(bits >> 24)
		sum := int16(q8.SumNeg[sb])
		buf[off+4] = byte(sum)
		buf[off+5] = byte(uint16(sum) >> 8)
		for i := 0; i < 32; i++ {
			buf[off+6+i] = byte(q8.Q[sb*32+i])
		}
	}
	return buf
}

func q8Blocks32Bytes(act []float32) []byte { return q8Block32ToBytes(quantizeQ8Blocks32(act)) }

func q4kQ41x32MatVecExactAI(w q4kQ41x32, exactMins []float32, act []float32, out []float32, pool *AIWorkerPool) {
	q4kQ41x32MatVecGoAsmWithCorrection(w, exactMins, act, out, pool)
	if q4kCompareOn {
		ref := make([]float32, len(out))
		q4kQ41x32MatVecRef(w, act, ref)
		var maxDiff float32
		maxIdx := 0
		for i := range out {
			d := out[i] - ref[i]
			if d < 0 {
				d = -d
			}
			if d > maxDiff {
				maxDiff = d
				maxIdx = i
			}
		}
		fmt.Fprintf(os.Stderr, "q4k_exact_compare M=%d K=%d maxDiff=%.6f idx=%d go=%.6f ref=%.6f\n", w.M, w.K, maxDiff, maxIdx, out[maxIdx], ref[maxIdx])
	}
}

func q4kQ41x32MatVecGoAsm(w q4kQ41x32, act []float32, out []float32, pool *AIWorkerPool) {
	q4kQ41x32MatVecGoAsmWithCorrection(w, nil, act, out, pool)
}

func q4kQ41x32MatVecGoAsmWithCorrection(w q4kQ41x32, exactMins []float32, act []float32, out []float32, pool *AIWorkerPool) {
	if !w.Valid || w.K%32 != 0 || w.M%32 != 0 {
		panic("q4kQ41x32MatVecGoAsm: invalid shape")
	}
	q8 := quantizeQ8Blocks32(act)
	quantA := q8Block32ToBytes(q8)
	subs := w.K / 32
	groups := w.M / 32
	dbg := os.Getenv("IME2_K3_DBG") != ""
	runGroup := func(rg int) {
		k3I8I4M1((*byte)(unsafe.Pointer(&quantA[0])), (*byte)(unsafe.Pointer(&w.BData[rg*subs*608])), &out[rg*32], subs, 32)
		if dbg && rg == 0 {
			fmt.Fprintf(os.Stderr, "k3raw[0..3]: %.5f %.5f %.5f %.5f\n", out[0], out[1], out[2], out[3])
		}
		for r := 0; r < 32; r++ {
			corr := float32(0)
			for sb := 0; sb < subs; sb++ {
				metaIdx := q41x32MetaIndex(rg, sb, r, subs)
				minTerm := float32(w.ZP[metaIdx]) * w.D[metaIdx]
				if exactMins != nil {
					minTerm = exactMins[rg*32*subs+r*subs+sb]
				}
				corr += float32(q8.SumNeg[sb]) * q8.Scale[sb] * minTerm
			}
			out[rg*32+r] += corr
		}
	}
	if q4kGoAsmSerialOn {
		registerAIThread(8)
		for rg := 0; rg < groups; rg++ {
			runGroup(rg)
		}
		return
	}
	pool.Run(func(workerID, nWorkers int) {
		gStart := workerID * groups / nWorkers
		gEnd := (workerID + 1) * groups / nWorkers
		for rg := gStart; rg < gEnd; rg++ {
			runGroup(rg)
		}
	})
}

func q4kQ41x32MatVecCShim(w q4kQ41x32, act []float32, out []float32, pool *AIWorkerPool) {
	if !w.Valid || w.K%32 != 0 || w.M%32 != 0 {
		panic("q4kQ41x32MatVecCShim: invalid shape")
	}
	quantA := q8Blocks32Bytes(act)
	subs := w.K / 32
	groups := w.M / 32
	pool.Run(func(workerID, nWorkers int) {
		gStart := workerID * groups / nWorkers
		gEnd := (workerID + 1) * groups / nWorkers
		for rg := gStart; rg < gEnd; rg++ {
			callLocalK3I8I4M1(quantA, w.BData[rg*subs*608:(rg+1)*subs*608], out[rg*32:(rg+1)*32], 32, subs)
		}
	})
}

// q4kQ41x32MatVecAI ports llama.cpp's q4_k_32x32_q8_0 data contract:
// q8 activations are quantized in 32-wide blocks with scale and negative sum;
// weights are Q4_K->Q4_1x32 with fp scale and uint4 zero-point per output row.
func q4kQ41x32MatVecAI(w q4kQ41x32, act []float32, out []float32, pool *AIWorkerPool) {
	if q4kGoAsmOn {
		q4kQ41x32MatVecGoAsm(w, act, out, pool)
		if q4kCompareOn {
			ref := make([]float32, len(out))
			q4kQ41x32MatVecRef(w, act, ref)
			var maxDiff float32
			maxIdx := 0
			for i := range out {
				d := out[i] - ref[i]
				if d < 0 {
					d = -d
				}
				if d > maxDiff {
					maxDiff = d
					maxIdx = i
				}
			}
			fmt.Fprintf(os.Stderr, "q4k_goasm_compare M=%d K=%d maxDiff=%.6f idx=%d go=%.6f ref=%.6f\n", w.M, w.K, maxDiff, maxIdx, out[maxIdx], ref[maxIdx])
		}
		return
	}
	if q4kCShimOn {
		q4kQ41x32MatVecCShim(w, act, out, pool)
		return
	}
	if q4kLlamaX32RefOn {
		q4kQ41x32MatVecRef(w, act, out)
		return
	}
	if !w.Valid || w.K%32 != 0 || w.M%32 != 0 {
		panic("q4kQ41x32MatVecAI: invalid shape")
	}
	M, K := w.M, w.K
	subs := K / 32
	q8 := quantizeQ8Blocks32(act)
	pool.Run(func(workerID, nWorkers int) {
		rowStart := (workerID * M / nWorkers / 8) * 8
		rowEnd := ((workerID + 1) * M / nWorkers / 8) * 8
		if workerID == nWorkers-1 {
			rowEnd = M
		}
		var wTile [128]int8
		var aTile [128]int8
		for row := rowStart; row < rowEnd; row += 8 {
			for r := 0; r < 8; r++ {
				out[row+r] = 0
			}
			var part [8]float32
			rg := row / 32
			for sb := 0; sb < subs; sb++ {
				as := q8.Scale[sb]
				if as == 0 {
					continue
				}
				var dots [8]int32
				for half := 0; half < 2; half++ {
					for r := 0; r < 8; r++ {
						qsOff := q41x32QSOffset(rg, sb, (row%32)+r, subs) + half*8
						for i := 0; i < 8; i++ {
							b := w.QS[qsOff+i]
							wTile[r*16+2*i] = int8(b & 0x0f)
							wTile[r*16+2*i+1] = int8((b >> 4) & 0x0f)
						}
						copy(aTile[r*16:(r+1)*16], q8.Q[sb*32+half*16:sb*32+half*16+16])
					}
					var acc [64]int32
					ime2.VmadotKLoop1024(
						(*byte)(unsafe.Pointer(&wTile[0])),
						(*byte)(unsafe.Pointer(&aTile[0])),
						&acc[0], 16,
					)
					for r := 0; r < 8; r++ {
						dots[r] += acc[r*8]
					}
				}
				for r := 0; r < 8; r++ {
					metaIdx := q41x32MetaIndex(rg, sb, (row%32)+r, subs)
					corr := q8.SumNeg[sb] * int32(w.ZP[metaIdx])
					part[r] = f32ToF16ToF32(part[r] + float32(dots[r]+corr)*as*w.D[metaIdx])
				}
				if sb%16 == 15 {
					for r := 0; r < 8; r++ {
						out[row+r] += part[r]
						part[r] = 0
					}
				}
			}
			for r := 0; r < 8; r++ {
				out[row+r] += part[r]
			}
		}
	})
}
