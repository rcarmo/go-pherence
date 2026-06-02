package main

import (
	"unsafe"
	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

// q4kQ41x32GateUpSiluSameAct runs Gate and Up Q4_K matvecs for the same
// activation in a single AI-worker dispatch, then computes the local SiLU*Up
// shard before leaving the worker. It supports both the direct grouped path and
// the native-style B/N32 TCM pair-wave path.
func q4kQ41x32GateUpSiluSameAct(act []float32, pool *AIWorkerPool, gate, up q4kQ41x32, gateOut, upOut, hidden []float32) bool {
	if !q4kGateFuseOn || q4kExactOn || q4kNativeCGOOn {
		return false
	}
	if !gate.Valid || !up.Valid || gate.K != up.K || gate.M != up.M || gate.K%32 != 0 || gate.M%32 != 0 {
		return false
	}
	if len(gateOut) < gate.M || len(upOut) < up.M || len(hidden) < gate.M {
		return false
	}
	q8 := quantizeQ8Blocks32(act)
	quantA := q8Block32ToBytes(q8)
	subs := gate.K / 32
	groups := gate.M / 32
	bBytes := subs * 608
	bOff := (len(quantA) + 63) &^ 63
	useWave := q4kTCMBWaveGateOn && subs%2 == 0 && pool != nil && pool.n%2 == 0 && groups >= pool.n && (groups%pool.n)%2 == 0 && pool.tcmSlices != nil && len(pool.tcmSlices) >= pool.n
	if useWave {
		for i := 0; i < pool.n; i++ {
			if len(pool.tcmSlices[i]) < bOff+bBytes {
				useWave = false
				break
			}
		}
	}
	gateBarrier := &q4kPairBarrier{}
	upBarrier := &q4kPairBarrier{}
	pool.Run(func(workerID, nWorkers int) {
		quantPtr := (*byte)(unsafe.Pointer(&quantA[0]))
		tcmSlice := getTCMSlice(workerID)
		if len(tcmSlice) >= len(quantA) {
			copy(tcmSlice[:len(quantA)], quantA)
			quantPtr = (*byte)(unsafe.Pointer(&tcmSlice[0]))
		}
		gStart := workerID * groups / nWorkers
		gEnd := (workerID + 1) * groups / nWorkers
		if gEnd <= gStart {
			return
		}
		runWave := func(w q4kQ41x32, out []float32, barrier *q4kPairBarrier) {
			pair := workerID / 2
			rg := workerID
			if workerID%2 == 0 {
				copyTCMBytes(tcmSlice[bOff:bOff+bBytes], w.BData[rg*subs*608:(rg+1)*subs*608])
			}
			barrier.wait(pair)
			if workerID%2 != 0 {
				copyTCMBytes(tcmSlice[bOff:bOff+bBytes], w.BData[rg*subs*608:(rg+1)*subs*608])
			}
			bPtr := (*byte)(unsafe.Pointer(&tcmSlice[bOff]))
			for ; rg < groups; rg += nWorkers {
				if workerID%2 != 0 {
					barrier.wait(pair)
				}
				k3I8I4M1Groups(quantPtr, bPtr, &out[rg*32], subs, 1)
				if workerID%2 == 0 {
					barrier.wait(pair)
				}
				nextRg := rg + nWorkers
				if nextRg < groups {
					copyTCMBytes(tcmSlice[bOff:bOff+bBytes], w.BData[nextRg*subs*608:(nextRg+1)*subs*608])
				}
			}
			// Batched Go schedulers chain another matrix after this wave; native
			// exits the kernel here. Add a final pair sync so even workers do not
			// enter the next wave while odd workers are still computing the final tile.
			if workerID%2 != 0 {
				barrier.wait(pair)
			} else {
				barrier.wait(pair)
			}
		}
		if useWave {
			runWave(gate, gateOut, gateBarrier)
			runWave(up, upOut, upBarrier)
		} else if subs%2 == 0 {
			k3I8I4M1Groups(quantPtr, (*byte)(unsafe.Pointer(&gate.BData[gStart*subs*608])), &gateOut[gStart*32], subs, gEnd-gStart)
			k3I8I4M1Groups(quantPtr, (*byte)(unsafe.Pointer(&up.BData[gStart*subs*608])), &upOut[gStart*32], subs, gEnd-gStart)
			for rg := gStart; rg < gEnd; rg++ {
				base0 := rg * subs * 32
				go_ := gateOut[rg*32 : rg*32+32]
				uo_ := upOut[rg*32 : rg*32+32]
				for sb := 0; sb < subs; sb++ {
					sc := float32(q8.SumNeg[sb]) * q8.Scale[sb]
					if sc < -1e-6 || sc > 1e-6 {
						b := base0 + sb*32
						ime2.ScaleAccF32RVV(go_, gate.ZPD[b:b+32], sc)
						ime2.ScaleAccF32RVV(uo_, up.ZPD[b:b+32], sc)
					}
				}
			}
		} else {
			for rg := gStart; rg < gEnd; rg++ {
				k3I8I4M1(quantPtr, (*byte)(unsafe.Pointer(&gate.BData[rg*subs*608])), &gateOut[rg*32], subs, 32)
				k3I8I4M1(quantPtr, (*byte)(unsafe.Pointer(&up.BData[rg*subs*608])), &upOut[rg*32], subs, 32)
			}
			for rg := gStart; rg < gEnd; rg++ {
				base0 := rg * subs * 32
				go_ := gateOut[rg*32 : rg*32+32]
				uo_ := upOut[rg*32 : rg*32+32]
				for sb := 0; sb < subs; sb++ {
					sc := float32(q8.SumNeg[sb]) * q8.Scale[sb]
					if sc < -1e-6 || sc > 1e-6 {
						b := base0 + sb*32
						ime2.ScaleAccF32RVV(go_, gate.ZPD[b:b+32], sc)
						ime2.ScaleAccF32RVV(uo_, up.ZPD[b:b+32], sc)
					}
				}
			}
		}
		if useWave {
			// B-wave work is assigned by strided N32 group ownership
			// (rg=workerID; rg+=nWorkers), not by contiguous gStart:gEnd.
			// Compute hidden over exactly the groups this worker produced.
			for rg := workerID; rg < groups; rg += nWorkers {
				start := rg * 32
				end := start + 32
				for i := start; i < end; i++ {
					hidden[i] = silu(gateOut[i]) * upOut[i]
				}
			}
		} else {
			start := gStart * 32
			end := gEnd * 32
			for i := start; i < end; i++ {
				hidden[i] = silu(gateOut[i]) * upOut[i]
			}
		}
	})
	return true
}
