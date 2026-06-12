package aicpu

import (
	"fmt"
	"math"
	"os"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/aicpu/aipool"
	"github.com/rcarmo/go-pherence/backends/spacemit/aicpu/config"
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
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

func q8Blocks32Bytes(act []float32) []byte         { return q8Block32ToBytes(quantizeQ8Blocks32(act)) }
func quantizeQ8Blocks32Bytes(act []float32) []byte { return q8Blocks32Bytes(act) }

func q4kQ41x32MatVecExactAI(w q4kQ41x32, exactMins []float32, act []float32, out []float32, pool *aipool.AIWorkerPool) {
	q4kQ41x32MatVecGoAsmWithCorrection(w, exactMins, act, out, pool)
	if config.Q4kCompareOn {
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

func q4kQ41x32MatVecGoAsm(w q4kQ41x32, act []float32, out []float32, pool *aipool.AIWorkerPool) {
	// Try B-wave TCM single matvec first
	if q4kQ41x32BWaveMatVecGoAsm(w, act, out, pool) {
		return
	}
	q4kQ41x32MatVecGoAsmWithCorrection(w, nil, act, out, pool)
}

func q4kQ41x32MatVecGoAsmWithCorrection(w q4kQ41x32, exactMins []float32, act []float32, out []float32, pool *aipool.AIWorkerPool) {
	if !w.Valid || w.K%32 != 0 || w.M%32 != 0 {
		panic("q4kQ41x32MatVecGoAsm: invalid shape")
	}
	q8 := quantizeQ8Blocks32(act)
	quantA := q8Block32ToBytes(q8)
	subs := w.K / 32
	groups := w.M / 32
	dbg := os.Getenv("IME2_K3_DBG") != ""
	// Precompute per-subblock activation correction factors
	sumActCorr := make([]float32, subs)
	for sb := 0; sb < subs; sb++ {
		sumActCorr[sb] = float32(q8.SumNeg[sb]) * q8.Scale[sb]
	}
	runGroup := func(rg int) {
		ime2.K3I8I4M1((*byte)(unsafe.Pointer(&quantA[0])), (*byte)(unsafe.Pointer(&w.BData[rg*subs*608])), &out[rg*32], subs, 32)
		if dbg && rg == 0 {
			fmt.Fprintf(os.Stderr, "k3raw[0..3]: %.5f %.5f %.5f %.5f\n", out[0], out[1], out[2], out[3])
		}
		if exactMins != nil {
			// Exact path: use provided dmin values (diagnostic only)
			for r := 0; r < 32; r++ {
				var corr float32
				for sb := 0; sb < subs; sb++ {
					corr += sumActCorr[sb] * exactMins[rg*32*subs+r*subs+sb]
				}
				out[rg*32+r] += corr
			}
		} else if len(w.ZPD) > 0 {
			// Fast path: RVV SAXPY, sequential ZPD access, skip near-zero blocks
			base0 := rg * subs * 32
			outSlice := out[rg*32 : rg*32+32]
			for sb := 0; sb < subs; sb++ {
				sc := sumActCorr[sb]
				if sc < -1e-6 || sc > 1e-6 {
					ime2.ScaleAccF32RVV(outSlice, w.ZPD[base0+sb*32:base0+sb*32+32], sc)
				}
			}
		} else {
			// Fallback: compute ZP*D on the fly
			for r := 0; r < 32; r++ {
				var corr float32
				for sb := 0; sb < subs; sb++ {
					metaIdx := q41x32MetaIndex(rg, sb, r, subs)
					corr += sumActCorr[sb] * float32(w.ZP[metaIdx]) * w.D[metaIdx]
				}
				out[rg*32+r] += corr
			}
		}
	}
	if config.Q4kGoAsmSerialOn {
		aipool.RegisterAIThread(8)
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

func q4kQ41x32MatVecCShim(w q4kQ41x32, act []float32, out []float32, pool *aipool.AIWorkerPool) {
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
func q4kQ41x32MatVecAI(w q4kQ41x32, act []float32, out []float32, pool *aipool.AIWorkerPool) {
	if config.Q4kGoAsmOn {
		q4kQ41x32MatVecGoAsm(w, act, out, pool)
		if config.Q4kCompareOn {
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
	if config.Q4kCShimOn {
		q4kQ41x32MatVecCShim(w, act, out, pool)
		return
	}
	if config.Q4kLlamaX32RefOn {
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

// q4kBatchMatVecSpec pairs a Q4_K weight matrix with its output slice for
// batched same-activation dispatch.
type q4kBatchMatVecSpec struct {
	W   q4kQ41x32
	Out []float32
}

// q4kQ41x32MatVecBatchSameAct computes multiple Q4_K matvecs that share the
// same activation in one pool.Run call. It quantizes the activation once and
// dispatches all specs across workers in a single barrier. Returns false if
// any spec is invalid or if conditions for pooled dispatch are not met.
// q4kQ41x32MatVecBatchSameAct runs each spec through q4kQ41x32MatVecGoAsmWithCorrection.
// Separate pool.Run calls per spec; correct ZP correction applied for each.
// q4kQ41x32MatVecBatchSameAct runs multiple Q4K matvecs sharing one activation
// in a single pool.Run call. ime2.K3I8I4M1 handles ZP correction internally via
// the B ZP bytes at BData+64; no Go-level correction loop is needed.
// q4kQ41x32MatVecBatchSameAct runs multiple Q4K matvecs sharing one activation
// in a single pool.Run. ime2.K3I8I4M1 does the main dot product (partial ZP);
// the Go correction loop adds the full ZP×SumNeg correction per group/row.
// q4kQ41x32MatVecBatchSameAct runs multiple Q4K matvecs sharing one activation.
// ime2.K3I8I4M1 handles the dot product; ZP correction zeroed in kernel and not applied here.
// q4kQ41x32MatVecBatchSameAct calls q4kQ41x32MatVecGoAsmWithCorrection for each spec.
// This gives correct output via the existing ZP correction loop.
// Sequential pool.Run calls per spec (not batched) but correct.
// q4kQ41x32MatVecBatchSameAct uses matVecRef (F32) for all specs — correct baseline.
// q4kQ41x32MatVecBatchSameAct calls q4kQ41x32MatVecGoAsmWithCorrection sequentially.
// q4kQ41x32MatVecBatchSameAct computes multiple Q4K matvecs in one pool.Run.
// ime2.K3I8I4M1 computes the dot + partial ZP; the Go correction loop adds full fp32 ZP.
// q4kQ41x32MatVecBatchSameAct computes multiple Q4K matvecs in one pool.Run.
// ime2.K3I8I4M1 uses constant ZP=8 (matching native PR#22863 active path).
// No Go correction loop needed; kernel handles full ZP approximation.
// q4kQ41x32MatVecBatchSameAct runs multiple Q4K matvecs in one pool.Run with ZP correction.
// q4kQ41x32MatVecBatchSameAct runs multiple Q4K matvecs in one pool.Run.
// ime2.K3I8I4M1 computes dot+partial-ZP; the correction loop applies exact fp32 ZP.
// Uses precomputed ZPD = float32(ZP)*D for cache-efficient sequential access.
// q4kQ41x32MatVecBatchSameAct runs multiple Q4K matvecs in one pool.Run.
// ime2.K3I8I4M1 correctly applies ZP correction for all 32 output rows on VLEN=1024 AI cores.
// No Go correction loop needed (would double-count the kernel's ZP correction).
// q4kQ41x32MatVecBatchSameAct runs multiple Q4K matvecs in one pool.Run.
// Kernel ZP=0; Go ZPD correction applies exact fp32 ZP correction.
// q4kQ41x32MatVecBatchSameAct: constant ZP=8 kernel, no Go correction.
// q4kQ41x32MatVecBatchSameAct: kernel ZP=0, Go ZPD correction for exact results.
// q4kQ41x32MatVecBatchSameAct: kernel handles ZP via fixed vwcvtu (vl=32, VLEN=1024).
// q4kQ41x32MatVecBatchSameAct: kernel ZP=0, Go ZPD correction for exact results.
// q4kQ41x32MatVecBatchSameAct: kernel ZP=0, RVV-accelerated ZPD correction.
// Uses ime2.ScaleAccF32RVV (e32,m4, vl=128 on VLEN=1024) for the SAXPY correction.
// q4kQ41x32MatVecBatchSameAct: precompute correction vector in main goroutine,
// pool.Run() does kernel + simple vector add (no ZPD reads inside pool.Run).
// q4kQ41x32MatVecBatchSameAct: kernel ZP=0, parallel ZPD correction inside pool.Run.
func q4kQ41x32MatVecBatchSameAct(act []float32, pool *aipool.AIWorkerPool, specs ...q4kBatchMatVecSpec) bool {
	if len(specs) == 0 || pool == nil {
		return false
	}
	if config.Q4kExactOn || config.Q4kNativeCGOOn {
		return false
	}
	K := specs[0].W.K
	if K%32 != 0 {
		return false
	}
	for _, sp := range specs {
		if !sp.W.Valid || sp.W.K != K || sp.W.M%32 != 0 || len(sp.Out) < sp.W.M {
			return false
		}
	}
	// B-wave batch for QKV (TCM double-buffered weights)
	if config.Q4kTCMBWaveBatchOn && q4kQ41x32BWaveMatVecBatchSameAct(act, pool, specs...) {
		return true
	}
	subs := K / 32
	q8 := quantizeQ8Blocks32(act)
	quantBytes := q8Block32ToBytes(q8)
	sumActCorr := make([]float32, subs)
	for sb := 0; sb < subs; sb++ {
		sumActCorr[sb] = float32(q8.SumNeg[sb]) * q8.Scale[sb]
	}
	pool.Run(func(workerID, nWorkers int) {
		tcmSlice := getTCMSlice(workerID)
		var quantPtr *byte
		if tcmSlice != nil && len(tcmSlice) >= len(quantBytes) {
			rvv.CopyTCMBytes(tcmSlice[:len(quantBytes)], quantBytes)
			quantPtr = (*byte)(unsafe.Pointer(&tcmSlice[0]))
		} else {
			quantPtr = (*byte)(unsafe.Pointer(&quantBytes[0]))
		}
		for _, sp := range specs {
			groups := sp.W.M / 32
			gStart := workerID * groups / nWorkers
			gEnd := (workerID + 1) * groups / nWorkers
			if gStart >= gEnd {
				continue
			}
			ime2.K3I8I4M1GroupsZPD(quantPtr, (*byte)(unsafe.Pointer(&sp.W.BData[gStart*subs*608])), &sp.Out[gStart*32], subs, gEnd-gStart, &sp.W.ZPD[gStart*subs*32], &sumActCorr[0])
		}
	})
	return true
}

// q4kQ41x32MatVecCM1 is a stub for the C-style correction-order matmul.
// Falls back to q4kQ41x32MatVecGoAsmWithCorrection.
func q4kQ41x32MatVecCM1(w q4kQ41x32, exactMins []float32, act []float32, out []float32, pool *aipool.AIWorkerPool) {
	q4kQ41x32MatVecGoAsmWithCorrection(w, exactMins, act, out, pool)
}
