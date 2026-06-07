package k3

import (
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/k3/aipool"
	"github.com/rcarmo/go-pherence/backends/spacemit/k3/config"
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
)

// q4kQ41x32BWaveMatVecBatchSameAct is the B/N32 TCM wave scheduler for
// batched Q4_K matmuls. Delegates to per-spec B-wave single calls.
// Each spec is processed in its own pool.Run with a fresh pair barrier,
// ensuring correct barrier state between specs.
func q4kQ41x32BWaveMatVecBatchSameAct(act []float32, pool *aipool.AIWorkerPool, specs ...q4kBatchMatVecSpec) bool {
	if len(specs) == 0 || pool == nil {
		return false
	}
	if config.Q4kExactOn || config.Q4kNativeCGOOn {
		return false
	}
	if !config.Q4kTCMBWaveBatchOn || pool.N%2 != 0 || pool.TcmSlices == nil {
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
	// Process all specs in ONE pool.Run to minimize dispatch overhead.
	// Each worker processes all specs sequentially using the B-wave pattern.
	subs := K / 32
	bBytes := subs * 608
	q8 := quantizeQ8Blocks32(act)
	quantBytes := q8Block32ToBytes(q8)
	sumActCorr := make([]float32, subs)
	for sb := 0; sb < subs; sb++ {
		sumActCorr[sb] = float32(q8.SumNeg[sb]) * q8.Scale[sb]
	}
	aOff := (len(quantBytes) + 63) &^ 63
	needTCM := aOff + bBytes
	if len(pool.TcmSlices) > 0 && len(pool.TcmSlices[0]) < needTCM {
		return false
	}
	// Check all specs satisfy group conditions
	for _, sp := range specs {
		groups := sp.W.M / 32
		if groups < pool.N || (groups%pool.N)%2 != 0 {
			return false
		}
	}
	pairBarrier := &aipool.Q4KPairBarrier{}
	pool.Run(func(workerID, nWorkers int) {
		tcmSlice := getTCMSlice(workerID)
		if tcmSlice == nil || len(tcmSlice) < needTCM {
			return
		}
		copy(tcmSlice[:len(quantBytes)], quantBytes)
		quantPtr := (*byte)(unsafe.Pointer(&tcmSlice[0]))
		bSlice := tcmSlice[aOff : aOff+bBytes]
		pair := workerID / 2
		for _, sp := range specs {
			spGroups := sp.W.M / 32
			rg := workerID
			// Initial prefetch
			if workerID%2 == 0 && rg < spGroups {
				rvv.CopyTCMBytes(bSlice, sp.W.BData[rg*subs*608:(rg+1)*subs*608])
			}
			pairBarrier.Wait(pair)
			if workerID%2 != 0 && rg < spGroups {
				rvv.CopyTCMBytes(bSlice, sp.W.BData[rg*subs*608:(rg+1)*subs*608])
			}
			bPtr := (*byte)(unsafe.Pointer(&bSlice[0]))
			for ; rg < spGroups; rg += nWorkers {
				if workerID%2 != 0 {
					pairBarrier.Wait(pair)
				}
				k3I8I4M1(quantPtr, bPtr, &sp.Out[rg*32], subs, 32)
				// ZPD correction
				base0 := rg * subs * 32
				outSlice := sp.Out[rg*32 : rg*32+32]
				for sb := 0; sb < subs; sb++ {
					sc := sumActCorr[sb]
					if sc < -1e-6 || sc > 1e-6 {
						ime2.ScaleAccF32RVV(outSlice, sp.W.ZPD[base0+sb*32:base0+sb*32+32], sc)
					}
				}
				if workerID%2 == 0 {
					pairBarrier.Wait(pair)
				}
				nextRg := rg + nWorkers
				if nextRg < spGroups {
					rvv.CopyTCMBytes(bSlice, sp.W.BData[nextRg*subs*608:(nextRg+1)*subs*608])
				}
			}
		}
	})
	return true
}

// q4kQ41x32BWaveMatVecGoAsm is the B-wave TCM variant of q4kQ41x32MatVecGoAsm.
// Uses worker-pair barriers for double-buffered B access from TCM.
func q4kQ41x32BWaveMatVecGoAsm(w q4kQ41x32, act []float32, out []float32, pool *aipool.AIWorkerPool) bool {
	if !w.Valid || w.K%32 != 0 || w.M%32 != 0 || pool == nil {
		return false
	}
	if !config.Q4kTCMBWaveSingleOn || pool.N%2 != 0 || pool.TcmSlices == nil {
		return false
	}
	subs := w.K / 32
	groups := w.M / 32
	if groups < pool.N || (groups%pool.N)%2 != 0 {
		return false
	}
	bBytes := subs * 608
	q8 := quantizeQ8Blocks32(act)
	quantBytes := q8Block32ToBytes(q8)
	sumActCorr := make([]float32, subs)
	for sb := 0; sb < subs; sb++ {
		sumActCorr[sb] = float32(q8.SumNeg[sb]) * q8.Scale[sb]
	}
	aOff := (len(quantBytes) + 63) &^ 63
	needTCM := aOff + bBytes
	if len(pool.TcmSlices) > 0 && len(pool.TcmSlices[0]) < needTCM {
		return false
	}
	pairBarrier := &aipool.Q4KPairBarrier{}
	pool.Run(func(workerID, nWorkers int) {
		tcmSlice := getTCMSlice(workerID)
		if tcmSlice == nil || len(tcmSlice) < needTCM {
			return
		}
		copy(tcmSlice[:len(quantBytes)], quantBytes)
		quantPtr := (*byte)(unsafe.Pointer(&tcmSlice[0]))
		bSlice := tcmSlice[aOff : aOff+bBytes]
		pair := workerID / 2
		rg := workerID
		if workerID%2 == 0 && rg < groups {
			rvv.CopyTCMBytes(bSlice, w.BData[rg*subs*608:(rg+1)*subs*608])
		}
		pairBarrier.Wait(pair)
		if workerID%2 != 0 && rg < groups {
			rvv.CopyTCMBytes(bSlice, w.BData[rg*subs*608:(rg+1)*subs*608])
		}
		bPtr := (*byte)(unsafe.Pointer(&bSlice[0]))
		for ; rg < groups; rg += nWorkers {
			if workerID%2 != 0 {
				pairBarrier.Wait(pair)
			}
			k3I8I4M1(quantPtr, bPtr, &out[rg*32], subs, 32)
			base0 := rg * subs * 32
			outSlice := out[rg*32 : rg*32+32]
			for sb := 0; sb < subs; sb++ {
				sc := sumActCorr[sb]
				if sc < -1e-6 || sc > 1e-6 {
					ime2.ScaleAccF32RVV(outSlice, w.ZPD[base0+sb*32:base0+sb*32+32], sc)
				}
			}
			if workerID%2 == 0 {
				pairBarrier.Wait(pair)
			}
			nextRg := rg + nWorkers
			if nextRg < groups {
				rvv.CopyTCMBytes(bSlice, w.BData[nextRg*subs*608:(nextRg+1)*subs*608])
			}
		}
	})
	return true
}
