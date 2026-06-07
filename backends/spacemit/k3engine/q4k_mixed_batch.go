package k3engine

import (
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/k3engine/aipool"
	"github.com/rcarmo/go-pherence/backends/spacemit/k3engine/config"
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
)

// q4kQ41x32MatVecBatchSameActWithI8 runs Q4_K matvecs and INT8 matvecs
// sharing the same activation in one AI-worker dispatch. This is useful for
// QKV on Q4_K_M models where Q/K are Q4_K and V is Q6_K-derived INT8.
func q4kQ41x32MatVecBatchSameActWithI8(act []float32, pool *aipool.AIWorkerPool, q4Specs []q4kBatchMatVecSpec, i8Specs ...aipool.AIGemmSpec) bool {
	if len(q4Specs) == 0 {
		return false
	}
	// Keep diagnostic/special modes on their established code paths.
	if config.Q4kExactOn || config.Q4kNativeCGOOn {
		return false
	}
	// k3I8I4M1Groups kernel is imprecise without Go ZP correction loop.
	// Fall through to matVecRef which uses GoAsmWithCorrection.
	return false
	K := q4Specs[0].W.K
	if K%32 != 0 {
		return false
	}
	for _, sp := range q4Specs {
		if !sp.W.Valid || sp.W.K != K || sp.W.M%32 != 0 || len(sp.Out) < sp.W.M {
			return false
		}
	}
	for _, sp := range i8Specs {
		if sp.K <= 0 || sp.M <= 0 || len(sp.Out) < sp.M || len(sp.ActPacked) == 0 || len(sp.WPacked) == 0 {
			return false
		}
	}
	quantA := quantizeQ8Blocks32Bytes(act)
	subs := K / 32
	int8PairBarrier := &aipool.Q4KPairBarrier{}
	q4PairBarrier := &aipool.Q4KPairBarrier{}
	pool.Run(func(workerID, nWorkers int) {
		quantPtr := (*byte)(unsafe.Pointer(&quantA[0]))
		tcmSlice := getTCMSlice(workerID)
		if len(tcmSlice) >= len(quantA) {
			copy(tcmSlice[:len(quantA)], quantA)
			quantPtr = (*byte)(unsafe.Pointer(&tcmSlice[0]))
		}
		for _, sp := range q4Specs {
			groups := sp.W.M / 32
			gStart := workerID * groups / nWorkers
			gEnd := (workerID + 1) * groups / nWorkers
			if gEnd <= gStart {
				continue
			}
			bBytes := subs * 608
			bOff := (len(quantA) + 63) &^ 63
			if config.Q4kTCMBWaveBatchOn && subs%2 == 0 && nWorkers%2 == 0 && groups >= nWorkers && (groups%nWorkers)%2 == 0 && len(tcmSlice) >= bOff+bBytes {
				pair := workerID / 2
				rg := workerID
				if workerID%2 == 0 {
					rvv.CopyTCMBytes(tcmSlice[bOff:bOff+bBytes], sp.W.BData[rg*subs*608:(rg+1)*subs*608])
				}
				q4PairBarrier.Wait(pair)
				if workerID%2 != 0 {
					rvv.CopyTCMBytes(tcmSlice[bOff:bOff+bBytes], sp.W.BData[rg*subs*608:(rg+1)*subs*608])
				}
				bPtr := (*byte)(unsafe.Pointer(&tcmSlice[bOff]))
				for ; rg < groups; rg += nWorkers {
					if workerID%2 != 0 {
						q4PairBarrier.Wait(pair)
					}
					k3I8I4M1Groups(quantPtr, bPtr, &sp.Out[rg*32], subs, 1)
					if workerID%2 == 0 {
						q4PairBarrier.Wait(pair)
					}
					nextRg := rg + nWorkers
					if nextRg < groups {
						rvv.CopyTCMBytes(tcmSlice[bOff:bOff+bBytes], sp.W.BData[nextRg*subs*608:(nextRg+1)*subs*608])
					}
				}
				q4PairBarrier.Wait(pair)
				continue
			}
			if subs%2 == 0 {
				k3I8I4M1Groups(quantPtr, (*byte)(unsafe.Pointer(&sp.W.BData[gStart*subs*608])), &sp.Out[gStart*32], subs, gEnd-gStart)
			} else {
				for rg := gStart; rg < gEnd; rg++ {
					k3I8I4M1(quantPtr, (*byte)(unsafe.Pointer(&sp.W.BData[rg*subs*608])), &sp.Out[rg*32], subs, 32)
				}
			}
		}
		for _, sp := range i8Specs {
			actPacked := sp.ActPacked
			if len(tcmSlice) != 0 && aipool.RunAIGemmWorkerTCMBWave(sp, workerID, nWorkers, actPacked, tcmSlice, int8PairBarrier) {
				continue
			}
			if len(tcmSlice) >= len(sp.ActPacked) {
				buf := tcmSlice[:len(sp.ActPacked)]
				copy(buf, unsafe.Slice((*byte)(unsafe.Pointer(&sp.ActPacked[0])), len(sp.ActPacked)))
				actPacked = unsafe.Slice((*int8)(unsafe.Pointer(&buf[0])), len(buf))
			}
			aipool.RunAIGemmWorkerWithAct(sp, workerID, nWorkers, actPacked)
		}
	})
	return true
}
