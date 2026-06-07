package k3

import (
	"sync/atomic"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/k3/aipool"
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
)

// layerBarrier is a reusable N-party barrier for intra-layer sync.
type layerBarrier struct {
	count atomic.Int64
	phase atomic.Int64
	n     int64
}

func (b *layerBarrier) Wait() {
	p := b.phase.Load()
	if b.count.Add(1) == b.n {
		b.count.Store(0)
		b.phase.Add(1)
		return
	}
	for b.phase.Load() == p {
	}
}

// q4kLayerCoalesced coalesces all Q4K matmuls that share the same activation
// into a single pool.Run dispatch. Within the layer:
//   - QKV share xn → 1 dispatch (already done by batch path)
//   - Gate+Up share xn2 → 1 dispatch (already done by gate fuse)
//   - Wo and Down each have unique activations → separate dispatches
//
// The REAL win comes from coalescing the QKV batch + ZPD into one call
// with barrier-separated Gate+Up batch + ZPD in the SAME pool.Run.
// But this requires xn2 (FFN norm output) to be ready, which depends on
// attention + Wo + residual + norm completing first.
//
// The only feasible coalescing without restructuring the layer:
//   - Combine QKV-batch into the Wo dispatch (if attention is done inline)
//   - Combine Gate+Up+SiLU+Down into one dispatch
//
// q4kFFNFused combines Gate + Up + SiLU + Down quantize + Down matmul
// into a single dispatch, cutting 2 dispatches per layer to 1.
func q4kFFNFused(
	xn2 []float32,
	hidden []float32,
	gateF, upF []float32,
	downF []float32,
	gate, up, down q4kQ41x32,
	nEmbd, nFF int,
	pool *aipool.AIWorkerPool,
) bool {
	if !gate.Valid || !up.Valid || !down.Valid {
		return false
	}
	if gate.K%32 != 0 || gate.M%32 != 0 || down.K%32 != 0 || down.M%32 != 0 {
		return false
	}
	if pool == nil || pool.N < 2 {
		return false
	}

	subsGate := gate.K / 32
	groupsGate := gate.M / 32
	subsDown := down.K / 32
	groupsDown := down.M / 32

	// Pre-quantize Gate/Up activation (xn2)
	q8Gate := quantizeQ8Blocks32(xn2)
	quantGate := q8Block32ToBytes(q8Gate)
	sumCorrGate := make([]float32, subsGate)
	for sb := 0; sb < subsGate; sb++ {
		sumCorrGate[sb] = float32(q8Gate.SumNeg[sb]) * q8Gate.Scale[sb]
	}

	barrier := &layerBarrier{n: int64(pool.N)}

	// Shared buffers for Down quantization
	sharedQuantDown := make([]byte, subsDown*38)
	sharedSumCorrDown := make([]float32, subsDown)

	pool.Run(func(workerID, nWorkers int) {
		// Stage Gate/Up quantized activation
		tcmSlice := getTCMSlice(workerID)
		var quantPtrGate *byte
		if tcmSlice != nil && len(tcmSlice) >= len(quantGate) {
			rvv.CopyTCMBytes(tcmSlice[:len(quantGate)], quantGate)
			quantPtrGate = (*byte)(unsafe.Pointer(&tcmSlice[0]))
		} else {
			quantPtrGate = (*byte)(unsafe.Pointer(&quantGate[0]))
		}

		// === Gate matmul ===
		gStart := workerID * groupsGate / nWorkers
		gEnd := (workerID + 1) * groupsGate / nWorkers
		if gStart < gEnd {
			k3I8I4M1Groups(quantPtrGate, (*byte)(unsafe.Pointer(&gate.BData[gStart*subsGate*608])), &gateF[gStart*32], subsGate, gEnd-gStart)
			for rg := gStart; rg < gEnd; rg++ {
				base0 := rg * subsGate * 32
				outSlice := gateF[rg*32 : rg*32+32]
				for sb := 0; sb < subsGate; sb++ {
					sc := sumCorrGate[sb]
					if sc < -1e-6 || sc > 1e-6 {
						ime2.ScaleAccF32RVV(outSlice, gate.ZPD[base0+sb*32:base0+sb*32+32], sc)
					}
				}
			}
		}

		// === Up matmul ===
		if gStart < gEnd {
			k3I8I4M1Groups(quantPtrGate, (*byte)(unsafe.Pointer(&up.BData[gStart*subsGate*608])), &upF[gStart*32], subsGate, gEnd-gStart)
			for rg := gStart; rg < gEnd; rg++ {
				base0 := rg * subsGate * 32
				outSlice := upF[rg*32 : rg*32+32]
				for sb := 0; sb < subsGate; sb++ {
					sc := sumCorrGate[sb]
					if sc < -1e-6 || sc > 1e-6 {
						ime2.ScaleAccF32RVV(outSlice, up.ZPD[base0+sb*32:base0+sb*32+32], sc)
					}
				}
			}
		}

		// === SiLU(gate) * up → hidden ===
		start := gStart * 32
		end := gEnd * 32
		for i := start; i < end; i++ {
			hidden[i] = silu(gateF[i]) * upF[i]
		}

		// === BARRIER: all workers must finish SiLU before Down can read hidden ===
		barrier.Wait()

		// Worker 0 quantizes hidden for Down; others wait at second barrier
		if workerID == 0 {
			q8d := quantizeQ8Blocks32(hidden[:nFF])
			qb := q8Block32ToBytes(q8d)
			copy(sharedQuantDown, qb)
			for sb := 0; sb < subsDown; sb++ {
				sharedSumCorrDown[sb] = float32(q8d.SumNeg[sb]) * q8d.Scale[sb]
			}
		}
		barrier.Wait()

		var quantPtrDown *byte
		if tcmSlice != nil && len(tcmSlice) >= len(sharedQuantDown) {
			rvv.CopyTCMBytes(tcmSlice[:len(sharedQuantDown)], sharedQuantDown)
			quantPtrDown = (*byte)(unsafe.Pointer(&tcmSlice[0]))
		} else {
			quantPtrDown = (*byte)(unsafe.Pointer(&sharedQuantDown[0]))
		}

		// === Down matmul ===
		dStart := workerID * groupsDown / nWorkers
		dEnd := (workerID + 1) * groupsDown / nWorkers
		if dStart < dEnd {
			k3I8I4M1Groups(quantPtrDown, (*byte)(unsafe.Pointer(&down.BData[dStart*subsDown*608])), &downF[dStart*32], subsDown, dEnd-dStart)
			for rg := dStart; rg < dEnd; rg++ {
				base0 := rg * subsDown * 32
				outSlice := downF[rg*32 : rg*32+32]
				for sb := 0; sb < subsDown; sb++ {
					sc := sharedSumCorrDown[sb]
					if sc < -1e-6 || sc > 1e-6 {
						ime2.ScaleAccF32RVV(outSlice, down.ZPD[base0+sb*32:base0+sb*32+32], sc)
					}
				}
			}
		}
	})

	return true
}
