package k3

import (
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

// aiGemmSpec describes one M×K matmul using already-packed weight and
// activation layouts.
type aiGemmSpec struct {
	M, K      int
	wPacked   []int8
	actPacked []int8
	wScale    float32
	actScale  float32
	out       []float32
}

// runAIGemmWorker executes one GEMM shard for a single AI worker.
func runAIGemmWorker(spec aiGemmSpec, workerID, nWorkers int) {
	runAIGemmWorkerWithAct(spec, workerID, nWorkers, spec.actPacked)
}

func runAIGemmWorkerWithAct(spec aiGemmSpec, workerID, nWorkers int, actPacked []int8) {
	runAIGemmWorkerWithActMode(spec, workerID, nWorkers, actPacked, false)
}

func runAIGemmWorkerAddWithAct(spec aiGemmSpec, workerID, nWorkers int, actPacked []int8) {
	runAIGemmWorkerWithActMode(spec, workerID, nWorkers, actPacked, true)
}

func runAIGemmWorkerWithActMode(spec aiGemmSpec, workerID, nWorkers int, actPacked []int8, add bool) {
	tilesPerRow := spec.K / 16
	combined := spec.wScale * spec.actScale
	rowStart := (workerID * spec.M / nWorkers / 8) * 8
	rowEnd := ((workerID + 1) * spec.M / nWorkers / 8) * 8
	if workerID == nWorkers-1 {
		rowEnd = spec.M
	}
	if rowEnd > rowStart && rowStart%8 == 0 && rowEnd%8 == 0 {
		var scratch [64]int32
		if add {
			vmadotI8GroupsAdd1024(
				(*byte)(unsafe.Pointer(&spec.wPacked[(rowStart/8)*tilesPerRow*128])),
				(*byte)(unsafe.Pointer(&actPacked[0])),
				&scratch[0], &spec.out[rowStart], &combined,
				(rowEnd-rowStart)/8, spec.K,
			)
		} else {
			vmadotI8Groups1024(
				(*byte)(unsafe.Pointer(&spec.wPacked[(rowStart/8)*tilesPerRow*128])),
				(*byte)(unsafe.Pointer(&actPacked[0])),
				&scratch[0], &spec.out[rowStart], &combined,
				(rowEnd-rowStart)/8, spec.K,
			)
		}
		return
	}
	for i := rowStart; i < rowEnd; i += 8 {
		var acc [64]int32
		ime2.VmadotKLoop1024(
			(*byte)(unsafe.Pointer(&spec.wPacked[(i/8)*tilesPerRow*128])),
			(*byte)(unsafe.Pointer(&actPacked[0])),
			&acc[0], spec.K,
		)
		for r := 0; r < 8 && i+r < spec.M; r++ {
			v := float32(acc[r*8]) * combined
			if add {
				spec.out[i+r] += v
			} else {
				spec.out[i+r] = v
			}
		}
	}
}

func runAIGemmWorkerTCMBWave(spec aiGemmSpec, workerID, nWorkers int, actPacked []int8, tcmSlice []byte, pairBarrier *q4kPairBarrier) bool {
	return runAIGemmWorkerTCMBWaveMode(spec, workerID, nWorkers, actPacked, tcmSlice, pairBarrier, false)
}

func runAIGemmWorkerTCMBWaveAdd(spec aiGemmSpec, workerID, nWorkers int, actPacked []int8, tcmSlice []byte, pairBarrier *q4kPairBarrier) bool {
	return runAIGemmWorkerTCMBWaveMode(spec, workerID, nWorkers, actPacked, tcmSlice, pairBarrier, true)
}

func runAIGemmWorkerTCMBWaveMode(spec aiGemmSpec, workerID, nWorkers int, actPacked []int8, tcmSlice []byte, pairBarrier *q4kPairBarrier, add bool) bool {
	if !int8TCMBWaveOn || spec.K%16 != 0 || spec.M%8 != 0 || nWorkers%2 != 0 {
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
	combined := spec.wScale * spec.actScale
	var scratch [64]int32
	pair := workerID / 2
	rg := workerID
	wBytes := unsafe.Slice((*byte)(unsafe.Pointer(&spec.wPacked[0])), len(spec.wPacked))
	if workerID%2 == 0 {
		copyTCMBytes(tcmSlice[bOff:bOff+groupBytes], wBytes[rg*groupBytes:(rg+1)*groupBytes])
	}
	pairBarrier.wait(pair)
	if workerID%2 != 0 {
		copyTCMBytes(tcmSlice[bOff:bOff+groupBytes], wBytes[rg*groupBytes:(rg+1)*groupBytes])
	}
	bPtr := (*byte)(unsafe.Pointer(&tcmSlice[bOff]))
	actPtr := (*byte)(unsafe.Pointer(&actPacked[0]))
	for ; rg < groups; rg += nWorkers {
		if workerID%2 != 0 {
			pairBarrier.wait(pair)
		}
		if add {
			vmadotI8GroupsAdd1024(bPtr, actPtr, &scratch[0], &spec.out[rg*8], &combined, 1, spec.K)
		} else {
			vmadotI8Groups1024(bPtr, actPtr, &scratch[0], &spec.out[rg*8], &combined, 1, spec.K)
		}
		if workerID%2 == 0 {
			pairBarrier.wait(pair)
		}
		nextRg := rg + nWorkers
		if nextRg < groups {
			copyTCMBytes(tcmSlice[bOff:bOff+groupBytes], wBytes[nextRg*groupBytes:(nextRg+1)*groupBytes])
		}
	}
	return true
}

// GemmAIPooled performs M×K matmul distributed across the AI worker pool.
func GemmAIPooled(M, K int, wPacked, actPacked []int8, wScale, actScale float32, out []float32, pool *AIWorkerPool) {
	GemmAIPooledBatch(pool, aiGemmSpec{M: M, K: K, wPacked: wPacked, actPacked: actPacked, wScale: wScale, actScale: actScale, out: out})
}

// GemmAIPooledAdd performs M×K matmul and adds the result into out.
func GemmAIPooledAdd(M, K int, wPacked, actPacked []int8, wScale, actScale float32, out []float32, pool *AIWorkerPool) {
	spec := aiGemmSpec{M: M, K: K, wPacked: wPacked, actPacked: actPacked, wScale: wScale, actScale: actScale, out: out}
	pairBarrier := &q4kPairBarrier{}
	pool.Run(func(workerID, nWorkers int) {
		actPackedLocal := spec.actPacked
		var tcmSlice []byte
		if pool.tcmSlices != nil && workerID < len(pool.tcmSlices) {
			tcmSlice = pool.tcmSlices[workerID]
			need := len(spec.actPacked)
			if need > 0 && need <= len(tcmSlice) {
				buf := tcmSlice[:need]
				copy(buf, *(*[]byte)(unsafe.Pointer(&spec.actPacked)))
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
func runAIGemmWorkerVL32(spec aiGemmSpec, workerID, nWorkers int) {
	tilesPerRow := spec.K / 8
	combined := spec.wScale * spec.actScale
	rowStart := (workerID * spec.M / nWorkers / 4) * 4
	rowEnd := ((workerID + 1) * spec.M / nWorkers / 4) * 4
	if workerID == nWorkers-1 {
		rowEnd = spec.M
	}
	for i := rowStart; i < rowEnd; i += 4 {
		var acc [16]int32
		ime2.VmadotKLoopAI(
			(*byte)(unsafe.Pointer(&spec.wPacked[(i/4)*tilesPerRow*32])),
			(*byte)(unsafe.Pointer(&spec.actPacked[0])),
			&acc[0], spec.K,
		)
		for r := 0; r < 4 && i+r < spec.M; r++ {
			spec.out[i+r] = float32(acc[r*4]) * combined
		}
	}
}

// GemmAIPooledVL32 performs M×K matmul on A100 workers with forced vl=32.
func GemmAIPooledVL32(M, K int, wPacked, actPacked []int8, wScale, actScale float32, out []float32, pool *AIWorkerPool) {
	GemmAIPooledBatchVL32(pool, aiGemmSpec{M: M, K: K, wPacked: wPacked, actPacked: actPacked, wScale: wScale, actScale: actScale, out: out})
}

// GemmAIPooledBatch performs multiple independent matmuls in one worker-pool
// dispatch. This reduces channel/barrier overhead for Q/K/V and Gate/Up pairs,
// which otherwise launch hundreds of tiny AI-core jobs per decoded token.
func GemmAIPooledBatch(pool *AIWorkerPool, specs ...aiGemmSpec) {
	pairBarrier := &q4kPairBarrier{}
	pool.Run(func(workerID, nWorkers int) {
		for _, spec := range specs {
			actPacked := spec.actPacked
			var tcmSlice []byte
			if pool.tcmSlices != nil && workerID < len(pool.tcmSlices) {
				tcmSlice = pool.tcmSlices[workerID]
				need := len(spec.actPacked)
				if need > 0 && need <= len(tcmSlice) {
					buf := tcmSlice[:need]
					copy(buf, *(*[]byte)(unsafe.Pointer(&spec.actPacked)))
					actPacked = *(*[]int8)(unsafe.Pointer(&buf))
				}
			}
			if len(tcmSlice) != 0 && runAIGemmWorkerTCMBWave(spec, workerID, nWorkers, actPacked, tcmSlice, pairBarrier) {
				continue
			}
			runAIGemmWorkerWithAct(spec, workerID, nWorkers, actPacked)
		}
	})
}

// GemmAIPooledBatchVL32 is the forced-vl=32 equivalent of GemmAIPooledBatch.
func GemmAIPooledBatchVL32(pool *AIWorkerPool, specs ...aiGemmSpec) {
	pool.Run(func(workerID, nWorkers int) {
		for _, spec := range specs {
			runAIGemmWorkerVL32(spec, workerID, nWorkers)
		}
	})
}
