package qwen

import (
	"fmt"

	"github.com/rcarmo/go-pherence/backends/mlx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func qwen35GemvMLXGPU(out, x []float32, w *nvidia.GPUMLXWeight, inDim, outDim int) bool {
	if w == nil || len(x) < inDim || len(out) < outDim || inDim <= 0 || outDim <= 0 {
		return false
	}
	qwen35MLXGPUScratch.Lock()
	defer qwen35MLXGPUScratch.Unlock()
	if qwen35MLXGPUScratch.x == nil || qwen35MLXGPUScratch.xN < inDim {
		if qwen35MLXGPUScratch.x != nil {
			qwen35MLXGPUScratch.x.Free()
		}
		qwen35MLXGPUScratch.x = nvidia.NewDevBuf(inDim)
		qwen35MLXGPUScratch.xN = inDim
	}
	if qwen35MLXGPUScratch.out == nil || qwen35MLXGPUScratch.outN < outDim {
		if qwen35MLXGPUScratch.out != nil {
			qwen35MLXGPUScratch.out.Free()
		}
		qwen35MLXGPUScratch.out = nvidia.NewDevBuf(outDim)
		qwen35MLXGPUScratch.outN = outDim
	}
	xb := qwen35MLXGPUScratch.x
	ob := qwen35MLXGPUScratch.out
	copy(xb.Data()[:inDim], x[:inDim])
	xb.MarkDirty()
	if err := xb.ToGPU(); err != nil {
		return false
	}
	if err := ob.EnsureGPU(); err != nil {
		return false
	}
	nvidia.GemvMLXDirect(ob, xb, w)
	ob.MarkOnGPU()
	copy(out[:outDim], ob.Data()[:outDim])
	return true
}

func qwen35MLXMLPIntoGPU(out, mlpIn []float32, gateM, upM, downM *mlx.QuantWeight, hidden, inter int) (bool, error) {
	if !qwen35GPUMLPEnabled || !qwen35GPUReady || gateM == nil || upM == nil || downM == nil {
		return false, nil
	}
	if len(mlpIn) != hidden || len(out) != hidden {
		return false, fmt.Errorf("Qwen3.5 GPU MLX MLP dims in/out=%d/%d want %d", len(mlpIn), len(out), hidden)
	}
	if gateM.InDim != hidden || gateM.OutDim != inter || upM.InDim != hidden || upM.OutDim != inter || downM.InDim != inter || downM.OutDim != hidden {
		return false, nil
	}
	gwGate, freeGate, ok := qwen35MLXMLPWeightForGPU(gateM)
	if !ok {
		return false, nil
	}
	defer freeGate()
	gwUp, freeUp, ok := qwen35MLXMLPWeightForGPU(upM)
	if !ok {
		return false, nil
	}
	defer freeUp()
	gwDown, freeDown, ok := qwen35MLXMLPWeightForGPU(downM)
	if !ok {
		return false, nil
	}
	defer freeDown()
	qwen35MLXGPUScratch.Lock()
	defer qwen35MLXGPUScratch.Unlock()
	if qwen35MLXGPUScratch.x == nil || qwen35MLXGPUScratch.xN < hidden {
		if qwen35MLXGPUScratch.x != nil {
			qwen35MLXGPUScratch.x.Free()
		}
		qwen35MLXGPUScratch.x = nvidia.NewDevBuf(hidden)
		qwen35MLXGPUScratch.xN = hidden
	}
	if qwen35MLXGPUScratch.gate == nil || qwen35MLXGPUScratch.interN < inter {
		if qwen35MLXGPUScratch.gate != nil {
			qwen35MLXGPUScratch.gate.Free()
		}
		if qwen35MLXGPUScratch.up != nil {
			qwen35MLXGPUScratch.up.Free()
		}
		qwen35MLXGPUScratch.gate = nvidia.NewDevBuf(inter)
		qwen35MLXGPUScratch.up = nvidia.NewDevBuf(inter)
		qwen35MLXGPUScratch.interN = inter
	}
	if qwen35MLXGPUScratch.out == nil || qwen35MLXGPUScratch.outN < hidden {
		if qwen35MLXGPUScratch.out != nil {
			qwen35MLXGPUScratch.out.Free()
		}
		qwen35MLXGPUScratch.out = nvidia.NewDevBuf(hidden)
		qwen35MLXGPUScratch.outN = hidden
	}
	xb := qwen35MLXGPUScratch.x
	gate := qwen35MLXGPUScratch.gate
	up := qwen35MLXGPUScratch.up
	ob := qwen35MLXGPUScratch.out
	copy(xb.Data()[:hidden], mlpIn)
	xb.MarkDirty()
	if err := xb.ToGPU(); err != nil {
		return false, err
	}
	if err := gate.EnsureGPU(); err != nil {
		return false, err
	}
	if err := up.EnsureGPU(); err != nil {
		return false, err
	}
	if err := ob.EnsureGPU(); err != nil {
		return false, err
	}
	nvidia.GemvMLXDirect(gate, xb, gwGate)
	gate.MarkOnGPU()
	nvidia.GemvMLXDirect(up, xb, gwUp)
	up.MarkOnGPU()
	if err := nvidia.F32SiLUMulBuffer(gate.GPUBuffer(), gate.GPUBuffer(), up.GPUBuffer(), inter); err != nil {
		return false, err
	}
	gate.MarkOnGPU()
	nvidia.GemvMLXDirect(ob, gate, gwDown)
	ob.MarkOnGPU()
	copy(out[:hidden], ob.Data()[:hidden])
	return true, nil
}

func qwen35MLXMLPWeightForGPU(w *mlx.QuantWeight) (*nvidia.GPUMLXWeight, func(), bool) {
	if gw, ok := qwen35CachedGPUMXWeightIfResident(w); ok {
		return gw, func() {}, true
	}
	if !qwen35GPUMLXOverflowEnabled {
		return nil, func() {}, false
	}
	gw, err := qwen35TransientGPUMXWeight(w)
	if err != nil {
		return nil, func() {}, false
	}
	return gw, gw.Free, true
}

func resetQwen35MLXGPUScratch() {
	qwen35MLXGPUScratch.Lock()
	defer qwen35MLXGPUScratch.Unlock()
	for _, b := range []*nvidia.DevBuf{qwen35MLXGPUScratch.x, qwen35MLXGPUScratch.gate, qwen35MLXGPUScratch.up, qwen35MLXGPUScratch.out} {
		if b != nil {
			b.Free()
		}
	}
	qwen35MLXGPUScratch.x = nil
	qwen35MLXGPUScratch.gate = nil
	qwen35MLXGPUScratch.up = nil
	qwen35MLXGPUScratch.out = nil
	qwen35MLXGPUScratch.xN = 0
	qwen35MLXGPUScratch.interN = 0
	qwen35MLXGPUScratch.outN = 0
}
