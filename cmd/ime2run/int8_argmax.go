package main

import "unsafe"

func GemmAIArgmaxI32(M, K int, wPacked, actPacked []int8, tmp []int32, pool *AIWorkerPool) (int, int32) {
	if M == 0 {
		return 0, 0
	}
	bestIDs := make([]int, pool.n)
	bestVals := make([]int32, pool.n)
	pool.Run(func(workerID, nWorkers int) {
		actPtr := (*byte)(unsafe.Pointer(&actPacked[0]))
		if pool.tcmSlices != nil && workerID < len(pool.tcmSlices) {
			need := len(actPacked)
			if need > 0 && need <= len(pool.tcmSlices[workerID]) {
				buf := pool.tcmSlices[workerID][:need]
				copy(buf, *(*[]byte)(unsafe.Pointer(&actPacked)))
				actPtr = (*byte)(unsafe.Pointer(&buf[0]))
			}
		}
		rowStart := (workerID * M / nWorkers / 8) * 8
		rowEnd := ((workerID + 1) * M / nWorkers / 8) * 8
		if workerID == nWorkers-1 {
			rowEnd = M
		}
		if rowEnd <= rowStart {
			bestIDs[workerID] = -1
			bestVals[workerID] = -1 << 31
			return
		}
		var scratch [64]int32
		tilesPerRow := K / 16
		var localBest int32
		var localID int64
		vmadotI8ArgmaxGroups1024(
			(*byte)(unsafe.Pointer(&wPacked[(rowStart/8)*tilesPerRow*128])),
			actPtr,
			&scratch[0], &localBest, &localID,
			(rowEnd-rowStart)/8, K, rowStart,
		)
		bestIDs[workerID] = int(localID)
		bestVals[workerID] = localBest
	})
	bestID := 0
	bestVal := bestVals[0]
	for i := range bestVals {
		if bestIDs[i] >= 0 && (bestIDs[bestID] < 0 || bestVals[i] > bestVal) {
			bestID = i
			bestVal = bestVals[i]
		}
	}
	return bestIDs[bestID], bestVal
}
