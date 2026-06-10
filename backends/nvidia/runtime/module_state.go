package nvidia

import (
	"github.com/rcarmo/go-pherence/backends/nvidia/internal/debuglog"
	"sync"
)

func markMegaModuleReady(entryCount int) {
	megaModuleOK = true
	sgemmReady = true
	kernelsLoaded = true
	ropeReady = true
	ropePartialReady = true
	attnScoreReady = true
	softmaxRowsReady = true
	attnReady = true
	q4Ready = true
	fusedSiLUMulOK = true
	debuglog.Printf("[gpu] All %d kernels loaded in 1 module\n", entryCount)
}

func freeNVFP4Scratch() {
	nvfp4Scratch.Lock()
	defer nvfp4Scratch.Unlock()
	if nvfp4Scratch.x != nil {
		nvfp4Scratch.x.Free()
		nvfp4Scratch.x = nil
	}
	if nvfp4Scratch.out != nil {
		nvfp4Scratch.out.Free()
		nvfp4Scratch.out = nil
	}
	nvfp4Scratch.xN = 0
	nvfp4Scratch.outN = 0
	nvfp4Scratch.xPtr = 0
	nvfp4Scratch.xLen = 0
}

func resetMegaModuleState() {
	megaModule = 0
	megaModuleOK = false
	megaModuleOnce = sync.Once{}
	sgemmReady = false
	kernelsLoaded = false
	ropeReady = false
	ropePartialReady = false
	attnScoreReady = false
	softmaxRowsReady = false
	attnReady = false
	q4Ready = false
	fusedSiLUMulOK = false
	fnPrefetch = 0
}
