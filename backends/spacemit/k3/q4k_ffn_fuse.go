package k3

import (
	"math"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/k3/aipool"
)

// q4kQ41x32FFNFusedSameAct runs Gate/Up Q4_K, SiLU, global INT8 quant/pack,
// and the INT8 down projection inside one AI-worker dispatch. It preserves the
// existing single global activation scale for down by reducing per-worker maxAbs
// before quantization.
func q4kQ41x32FFNFusedSameAct(act []float32, pool *aipool.AIWorkerPool, gate, up q4kQ41x32, down aipool.AIGemmSpec, gateOut, upOut, hidden, downOut []float32, downActPad, downActPacked []int8) (float32, bool) {
	if !q4kFFNFuseOn || q4kTCMBWaveOn || q4kExactOn || q4kNativeCGOOn || aipool.Int8TCMBWaveOn {
		return 0, false
	}
	if !gate.Valid || !up.Valid || gate.K != up.K || gate.M != up.M || gate.K%32 != 0 || gate.M%32 != 0 {
		return 0, false
	}
	if down.K != len(downActPad) || len(downActPacked) < 8*down.K || down.M <= 0 || len(downOut) < down.M {
		return 0, false
	}
	if len(gateOut) < gate.M || len(upOut) < up.M || len(hidden) < gate.M || gate.M > down.K {
		return 0, false
	}
	quantA := quantizeQ8Blocks32Bytes(act)
	subs := gate.K / 32
	localMax := make([]float32, pool.N)
	var scaleBox [1]float32
	barrier := aipool.NewAIBarrier(pool.N)
	pool.Run(func(workerID, nWorkers int) {
		quantPtr := (*byte)(unsafe.Pointer(&quantA[0]))
		tcmSlice := getTCMSlice(workerID)
		if len(tcmSlice) >= len(quantA) {
			copy(tcmSlice[:len(quantA)], quantA)
			quantPtr = (*byte)(unsafe.Pointer(&tcmSlice[0]))
		}
		groups := gate.M / 32
		gStart := workerID * groups / nWorkers
		gEnd := (workerID + 1) * groups / nWorkers
		if gEnd > gStart {
			if subs%2 == 0 {
				k3I8I4M1Groups(quantPtr, (*byte)(unsafe.Pointer(&gate.BData[gStart*subs*608])), &gateOut[gStart*32], subs, gEnd-gStart)
				k3I8I4M1Groups(quantPtr, (*byte)(unsafe.Pointer(&up.BData[gStart*subs*608])), &upOut[gStart*32], subs, gEnd-gStart)
			} else {
				for rg := gStart; rg < gEnd; rg++ {
					k3I8I4M1(quantPtr, (*byte)(unsafe.Pointer(&gate.BData[rg*subs*608])), &gateOut[rg*32], subs, 32)
					k3I8I4M1(quantPtr, (*byte)(unsafe.Pointer(&up.BData[rg*subs*608])), &upOut[rg*32], subs, 32)
				}
			}
		}
		start := gStart * 32
		end := gEnd * 32
		var mx float32
		for i := start; i < end; i++ {
			v := silu(gateOut[i]) * upOut[i]
			hidden[i] = v
			a := v
			if a < 0 {
				a = -a
			}
			if a > mx {
				mx = a
			}
		}
		localMax[workerID] = mx
		barrier.Wait()
		if workerID == 0 {
			var globalMax float32
			for _, v := range localMax {
				if v > globalMax {
					globalMax = v
				}
			}
			if globalMax == 0 || math.IsNaN(float64(globalMax)) {
				scaleBox[0] = 0
			} else {
				scaleBox[0] = globalMax / 127.0
			}
		}
		barrier.Wait()
		actScale := scaleBox[0]
		// Quantize hidden and broadcast-pack down activation in parallel by K16 tiles.
		if actScale == 0 {
			for i := workerID * down.K / nWorkers; i < (workerID+1)*down.K/nWorkers; i++ {
				downActPad[i] = 0
			}
		} else {
			s := float32(127.0) / (actScale * 127.0) // == 127/globalMax while keeping actScale source-of-truth
			qStart := workerID * down.K / nWorkers
			qEnd := (workerID + 1) * down.K / nWorkers
			if qEnd > gate.M {
				qEnd = gate.M
			}
			for i := qStart; i < qEnd; i++ {
				q := hidden[i] * s
				if q > 127 {
					q = 127
				} else if q < -128 {
					q = -128
				}
				downActPad[i] = int8(q)
			}
			for i := qEnd; i < (workerID+1)*down.K/nWorkers; i++ {
				downActPad[i] = 0
			}
		}
		barrier.Wait()
		K := down.K
		tiles := K / 16
		tStart := workerID * tiles / nWorkers
		tEnd := (workerID + 1) * tiles / nWorkers
		for t := tStart; t < tEnd; t++ {
			src := downActPad[t*16 : t*16+16]
			dstBase := t * 128
			for r := 0; r < 8; r++ {
				copy(downActPacked[dstBase+r*16:dstBase+(r+1)*16], src)
			}
		}
		barrier.Wait()
		sp := down
		sp.ActScale = actScale
		sp.ActPacked = downActPacked
		actPacked := sp.ActPacked
		if len(tcmSlice) >= len(actPacked) {
			buf := tcmSlice[:len(actPacked)]
			copy(buf, unsafe.Slice((*byte)(unsafe.Pointer(&actPacked[0])), len(actPacked)))
			actPacked = unsafe.Slice((*int8)(unsafe.Pointer(&buf[0])), len(buf))
		}
		aipool.RunAIGemmWorkerWithAct(sp, workerID, nWorkers, actPacked)
	})
	return scaleBox[0], true
}
