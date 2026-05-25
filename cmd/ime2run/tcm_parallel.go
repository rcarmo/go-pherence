package main

import (
	"sync"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	tcmpkg "github.com/rcarmo/go-pherence/backends/spacemit/tcm"
)

const tcmBlockSize = 393216 // 384KB per worker

// tcmStageAndVmadot copies weight tiles to TCM, then runs vmadot from TCM.
// Eliminates L2 cache contention between workers on weight reads.
func tcmStageAndVmadot(
	wPacked []int8,
	actPacked []int8,
	startGroup, endGroup int,
	tilesPerRow int,
	K int,
	scale float32,
	out []float32,
	tcmSlice []byte,
) {
	tileBytes := tilesPerRow * 32
	groupsPerChunk := tcmBlockSize / tileBytes
	if groupsPerChunk < 1 {
		groupsPerChunk = 1
	}

	for g := startGroup; g < endGroup; g += groupsPerChunk {
		gEnd := g + groupsPerChunk
		if gEnd > endGroup {
			gEnd = endGroup
		}
		nG := gEnd - g

		// Stage weight chunk to TCM
		srcOff := g * tileBytes
		copySize := nG * tileBytes
		if copySize > tcmBlockSize {
			copySize = tcmBlockSize
		}
		// Copy from DRAM to TCM
		src := wPacked[srcOff : srcOff+copySize]
		copy(tcmSlice[:copySize], *(*[]byte)(unsafe.Pointer(&src)))

		// Run vmadot reading from TCM (no cache contention)
		for gi := 0; gi < nG; gi++ {
			var acc [16]int32
			tcmOff := gi * tileBytes
			ime2.VmadotKLoop(
				(*byte)(unsafe.Pointer(&tcmSlice[tcmOff])),
				(*byte)(unsafe.Pointer(&actPacked[0])),
				&acc[0], K,
			)
			outIdx := (g + gi) * 4
			for r := 0; r < 4 && outIdx+r < len(out); r++ {
				out[outIdx+r] = float32(acc[r*4]) * scale
			}
		}
	}
}

var tcmDev *tcmpkg.TCM

func initTCMDevice() {
	if tcmpkg.IsAvailable() {
		tcmDev, _ = tcmpkg.Open()
	}
}

func getTCMSlice(workerID int) []byte { return nil //
	if tcmDev == nil {
		return nil
	}
	return tcmDev.Slice(workerID)
}

var _ = sync.WaitGroup{}
