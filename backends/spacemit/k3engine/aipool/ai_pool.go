package aipool

import (
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
)

// AIGemmSpec describes one M×K matmul using already-packed weight and
// activation layouts.
type AIGemmSpec struct {
	M, K      int
	WPacked   []int8
	ActPacked []int8
	WScale    float32
	ActScale  float32
	Out       []float32
}

// runAIGemmWorker executes one GEMM shard for a single AI worker.
func runAIGemmWorker(spec AIGemmSpec, workerID, nWorkers int) {
	RunAIGemmWorkerWithAct(spec, workerID, nWorkers, spec.ActPacked)
}

func RunAIGemmWorkerWithAct(spec AIGemmSpec, workerID, nWorkers int, actPacked []int8) {
	runAIGemmWorkerWithActMode(spec, workerID, nWorkers, actPacked, false)
}

func runAIGemmWorkerAddWithAct(spec AIGemmSpec, workerID, nWorkers int, actPacked []int8) {
	runAIGemmWorkerWithActMode(spec, workerID, nWorkers, actPacked, true)
}

func runAIGemmWorkerWithActMode(spec AIGemmSpec, workerID, nWorkers int, actPacked []int8, add bool) {
	tilesPerRow := spec.K / 16
	combined := spec.WScale * spec.ActScale
	rowStart := (workerID * spec.M / nWorkers / 8) * 8
	rowEnd := ((workerID + 1) * spec.M / nWorkers / 8) * 8
	if workerID == nWorkers-1 {
		rowEnd = spec.M
	}
	if rowEnd > rowStart && rowStart%8 == 0 && rowEnd%8 == 0 {
		var scratch [64]int32
		if add {
			ime2.VmadotI8GroupsAdd1024(
				(*byte)(unsafe.Pointer(&spec.WPacked[(rowStart/8)*tilesPerRow*128])),
				(*byte)(unsafe.Pointer(&actPacked[0])),
				&scratch[0], &spec.Out[rowStart], &combined,
				(rowEnd-rowStart)/8, spec.K,
			)
		} else {
			ime2.VmadotI8Groups1024(
				(*byte)(unsafe.Pointer(&spec.WPacked[(rowStart/8)*tilesPerRow*128])),
				(*byte)(unsafe.Pointer(&actPacked[0])),
				&scratch[0], &spec.Out[rowStart], &combined,
				(rowEnd-rowStart)/8, spec.K,
			)
		}
		return
	}
	for i := rowStart; i < rowEnd; i += 8 {
		var acc [64]int32
		ime2.VmadotKLoop1024(
			(*byte)(unsafe.Pointer(&spec.WPacked[(i/8)*tilesPerRow*128])),
			(*byte)(unsafe.Pointer(&actPacked[0])),
			&acc[0], spec.K,
		)
		for r := 0; r < 8 && i+r < spec.M; r++ {
			v := float32(acc[r*8]) * combined
			if add {
				spec.Out[i+r] += v
			} else {
				spec.Out[i+r] = v
			}
		}
	}
}

func RunAIGemmWorkerTCMBWave(spec AIGemmSpec, workerID, nWorkers int, actPacked []int8, tcmSlice []byte, pairBarrier *Q4KPairBarrier) bool {
	return runAIGemmWorkerTCMBWaveMode(spec, workerID, nWorkers, actPacked, tcmSlice, pairBarrier, false)
}

func runAIGemmWorkerTCMBWaveAdd(spec AIGemmSpec, workerID, nWorkers int, actPacked []int8, tcmSlice []byte, pairBarrier *Q4KPairBarrier) bool {
	return runAIGemmWorkerTCMBWaveMode(spec, workerID, nWorkers, actPacked, tcmSlice, pairBarrier, true)
}

func runAIGemmWorkerTCMBWaveMode(spec AIGemmSpec, workerID, nWorkers int, actPacked []int8, tcmSlice []byte, pairBarrier *Q4KPairBarrier, add bool) bool {
	if !Int8TCMBWaveOn || spec.K%16 != 0 || spec.M%8 != 0 || nWorkers%2 != 0 {
		return false
	}
	groups := spec.M / 8
	if groups < nWorkers || (groups%nWorkers)%2 != 0 {
		return false
	}
	actBytes := len(actPacked)
	groupBytes := (spec.K / 16) * 128
	bOff := (actBytes + 63) &^ 63
	if len(tcmSlice) < bOff+groupBytes {
		return false
	}
	if actBytes > 0 {
		actBytesSrc := unsafe.Slice((*byte)(unsafe.Pointer(&actPacked[0])), actBytes)
		copy(tcmSlice[:actBytes], actBytesSrc)
		actPacked = unsafe.Slice((*int8)(unsafe.Pointer(&tcmSlice[0])), actBytes)
	}
	combined := spec.WScale * spec.ActScale
	var scratch [64]int32
	pair := workerID / 2
	rg := workerID
	wBytes := unsafe.Slice((*byte)(unsafe.Pointer(&spec.WPacked[0])), len(spec.WPacked))
	if workerID%2 == 0 {
		rvv.CopyTCMBytes(tcmSlice[bOff:bOff+groupBytes], wBytes[rg*groupBytes:(rg+1)*groupBytes])
	}
	pairBarrier.Wait(pair)
	if workerID%2 != 0 {
		rvv.CopyTCMBytes(tcmSlice[bOff:bOff+groupBytes], wBytes[rg*groupBytes:(rg+1)*groupBytes])
	}
	bPtr := (*byte)(unsafe.Pointer(&tcmSlice[bOff]))
	actPtr := (*byte)(unsafe.Pointer(&actPacked[0]))
	for ; rg < groups; rg += nWorkers {
		if workerID%2 != 0 {
			pairBarrier.Wait(pair)
		}
		if add {
			ime2.VmadotI8GroupsAdd1024(bPtr, actPtr, &scratch[0], &spec.Out[rg*8], &combined, 1, spec.K)
		} else {
			ime2.VmadotI8Groups1024(bPtr, actPtr, &scratch[0], &spec.Out[rg*8], &combined, 1, spec.K)
		}
		if workerID%2 == 0 {
			pairBarrier.Wait(pair)
		}
		nextRg := rg + nWorkers
		if nextRg < groups {
			rvv.CopyTCMBytes(tcmSlice[bOff:bOff+groupBytes], wBytes[nextRg*groupBytes:(nextRg+1)*groupBytes])
		}
	}
	return true
}

// GemmAIPooled performs M×K matmul distributed across the AI worker pool.
func GemmAIPooled(M, K int, wPacked, actPacked []int8, wScale, actScale float32, out []float32, pool *AIWorkerPool) {
	GemmAIPooledBatch(pool, AIGemmSpec{M: M, K: K, WPacked: wPacked, ActPacked: actPacked, WScale: wScale, ActScale: actScale, Out: out})
}

// GemmAIPooledAdd performs M×K matmul and adds the result into out.
func GemmAIPooledAdd(M, K int, wPacked, actPacked []int8, wScale, actScale float32, out []float32, pool *AIWorkerPool) {
	spec := AIGemmSpec{M: M, K: K, WPacked: wPacked, ActPacked: actPacked, WScale: wScale, ActScale: actScale, Out: out}
	pairBarrier := &Q4KPairBarrier{}
	pool.Run(func(workerID, nWorkers int) {
		actPackedLocal := spec.ActPacked
		var tcmSlice []byte
		if pool.TcmSlices != nil && workerID < len(pool.TcmSlices) {
			tcmSlice = pool.TcmSlices[workerID]
			need := len(spec.ActPacked)
			if need > 0 && need <= len(tcmSlice) {
				buf := tcmSlice[:need]
				copy(buf, *(*[]byte)(unsafe.Pointer(&spec.ActPacked)))
				actPackedLocal = *(*[]int8)(unsafe.Pointer(&buf))
			}
		}
		if len(tcmSlice) != 0 && runAIGemmWorkerTCMBWaveAdd(spec, workerID, nWorkers, actPackedLocal, tcmSlice, pairBarrier) {
			return
		}
		runAIGemmWorkerAddWithAct(spec, workerID, nWorkers, actPackedLocal)
	})
}

// runAIGemmWorkerVL32 executes one GEMM shard with the known-good forced-vl=32
// AI-core kernel and legacy 4×8 tile layout.
func runAIGemmWorkerVL32(spec AIGemmSpec, workerID, nWorkers int) {
	tilesPerRow := spec.K / 8
	combined := spec.WScale * spec.ActScale
	rowStart := (workerID * spec.M / nWorkers / 4) * 4
	rowEnd := ((workerID + 1) * spec.M / nWorkers / 4) * 4
	if workerID == nWorkers-1 {
		rowEnd = spec.M
	}
	for i := rowStart; i < rowEnd; i += 4 {
		var acc [16]int32
		ime2.VmadotKLoopAI(
			(*byte)(unsafe.Pointer(&spec.WPacked[(i/4)*tilesPerRow*32])),
			(*byte)(unsafe.Pointer(&spec.ActPacked[0])),
			&acc[0], spec.K,
		)
		for r := 0; r < 4 && i+r < spec.M; r++ {
			spec.Out[i+r] = float32(acc[r*4]) * combined
		}
	}
}

// GemmAIPooledVL32 performs M×K matmul on A100 workers with forced vl=32.
func GemmAIPooledVL32(M, K int, wPacked, actPacked []int8, wScale, actScale float32, out []float32, pool *AIWorkerPool) {
	GemmAIPooledBatchVL32(pool, AIGemmSpec{M: M, K: K, WPacked: wPacked, ActPacked: actPacked, WScale: wScale, ActScale: actScale, Out: out})
}

// GemmAIPooledBatch performs multiple independent matmuls in one worker-pool
// dispatch. This reduces channel/barrier overhead for Q/K/V and Gate/Up pairs,
// which otherwise launch hundreds of tiny AI-core jobs per decoded token.
func GemmAIPooledBatch(pool *AIWorkerPool, specs ...AIGemmSpec) {
	pairBarrier := &Q4KPairBarrier{}
	pool.Run(func(workerID, nWorkers int) {
		for _, spec := range specs {
			actPacked := spec.ActPacked
			var tcmSlice []byte
			if pool.TcmSlices != nil && workerID < len(pool.TcmSlices) {
				tcmSlice = pool.TcmSlices[workerID]
				need := len(spec.ActPacked)
				if need > 0 && need <= len(tcmSlice) {
					buf := tcmSlice[:need]
					copy(buf, *(*[]byte)(unsafe.Pointer(&spec.ActPacked)))
					actPacked = *(*[]int8)(unsafe.Pointer(&buf))
				}
			}
			if len(tcmSlice) != 0 && RunAIGemmWorkerTCMBWave(spec, workerID, nWorkers, actPacked, tcmSlice, pairBarrier) {
				continue
			}
			RunAIGemmWorkerWithAct(spec, workerID, nWorkers, actPacked)
		}
	})
}

// GemmAIPooledBatchVL32 is the forced-vl=32 equivalent of GemmAIPooledBatch.
func GemmAIPooledBatchVL32(pool *AIWorkerPool, specs ...AIGemmSpec) {
	pool.Run(func(workerID, nWorkers int) {
		for _, spec := range specs {
			runAIGemmWorkerVL32(spec, workerID, nWorkers)
		}
	})
}
