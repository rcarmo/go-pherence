package k3

import (
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
)

// q8Q80x32MatVec4Pooled computes four activations against one native q8_0_32x32
// matrix using the i8i8 dispatcher (M4 for countM>=4), distributed over N32 groups.
func q8Q80x32MatVec4Pooled(w q8Q80x32, acts [4][]float32, outs [4][]float32, pool *AIWorkerPool) bool {
	if !w.Valid || w.K%32 != 0 || w.M%32 != 0 || pool == nil {
		return false
	}
	for i := 0; i < 4; i++ {
		if len(acts[i]) < w.K || len(outs[i]) < w.M {
			return false
		}
	}
	subs := w.K / 32
	groups := w.M / 32
	packedA := quantizeQ8RowsM4Bytes(acts, subs)
	pairBarrier := &q4kPairBarrier{}
	pool.Run(func(workerID, nWorkers int) {
		aPtr := (*byte)(unsafe.Pointer(&packedA[0]))
		tcmSlice := getTCMSlice(workerID)
		if len(tcmSlice) >= len(packedA) {
			copy(tcmSlice[:len(packedA)], packedA)
			aPtr = (*byte)(unsafe.Pointer(&tcmSlice[0]))
		}
		bBytes := subs * 1088
		bOff := (len(packedA) + 63) &^ 63
		if int8TCMBWaveOn && nWorkers%2 == 0 && groups >= nWorkers && (groups%nWorkers)%2 == 0 && len(tcmSlice) >= bOff+bBytes {
			pair := workerID / 2
			rg := workerID
			if workerID%2 == 0 {
				rvv.CopyTCMBytes(tcmSlice[bOff:bOff+bBytes], w.BData[rg*subs*1088:(rg+1)*subs*1088])
			}
			pairBarrier.wait(pair)
			if workerID%2 != 0 {
				rvv.CopyTCMBytes(tcmSlice[bOff:bOff+bBytes], w.BData[rg*subs*1088:(rg+1)*subs*1088])
			}
			bPtr := (*byte)(unsafe.Pointer(&tcmSlice[bOff]))
			for ; rg < groups; rg += nWorkers {
				if workerID%2 != 0 {
					pairBarrier.wait(pair)
				}
				var tmp [4 * 32]float32
				handled := q8I8Dispatcher(aPtr, bPtr, &tmp[0], 4, 32, subs, 32)
				if handled == 4 {
					for r := 0; r < 4; r++ {
						copy(outs[r][rg*32:(rg+1)*32], tmp[r*32:(r+1)*32])
					}
				}
				if workerID%2 == 0 {
					pairBarrier.wait(pair)
				}
				nextRg := rg + nWorkers
				if nextRg < groups {
					rvv.CopyTCMBytes(tcmSlice[bOff:bOff+bBytes], w.BData[nextRg*subs*1088:(nextRg+1)*subs*1088])
				}
			}
			return
		}
		gStart := workerID * groups / nWorkers
		gEnd := (workerID + 1) * groups / nWorkers
		for rg := gStart; rg < gEnd; rg++ {
			var tmp [4 * 32]float32
			bPtr := (*byte)(unsafe.Pointer(&w.BData[rg*subs*1088]))
			handled := q8I8Dispatcher(aPtr, bPtr, &tmp[0], 4, 32, subs, 32)
			if handled != 4 {
				return
			}
			for r := 0; r < 4; r++ {
				copy(outs[r][rg*32:(rg+1)*32], tmp[r*32:(r+1)*32])
			}
		}
	})
	return true
}
