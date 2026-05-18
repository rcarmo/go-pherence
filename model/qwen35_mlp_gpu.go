package model

import (
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia"
)

func qwen35MLPIntoGPU(out, mlpIn []float32, gateQ, upQ, downQ *Qwen35NVFP4Weight, hidden, inter int) (bool, error) {
	if !qwen35GPUMLPEnabled || !qwen35GPUReady || gateQ == nil || upQ == nil || downQ == nil || gateQ.W == nil || upQ.W == nil || downQ.W == nil || gateQ.GPU == nil || upQ.GPU == nil || downQ.GPU == nil {
		return false, nil
	}
	if len(mlpIn) != hidden || len(out) != hidden {
		return false, fmt.Errorf("Qwen3.5 GPU MLP dims in/out=%d/%d want %d", len(mlpIn), len(out), hidden)
	}
	if gateQ.W.InDim != hidden || gateQ.W.OutDim != inter || upQ.W.InDim != hidden || upQ.W.OutDim != inter || downQ.W.InDim != inter || downQ.W.OutDim != hidden {
		return false, nil
	}
	gwGate, _, err := qwen35CachedGPUWeight(gateQ)
	if err != nil {
		return false, fmt.Errorf("mlp.gate_proj GPU cache: %w", err)
	}
	gwUp, _, err := qwen35CachedGPUWeight(upQ)
	if err != nil {
		return false, fmt.Errorf("mlp.up_proj GPU cache: %w", err)
	}
	gwDown, _, err := qwen35CachedGPUWeight(downQ)
	if err != nil {
		return false, fmt.Errorf("mlp.down_proj GPU cache: %w", err)
	}
	xBuf, gateBuf, upBuf, outBuf, unlock, err := qwen35MLPScratchBuffers(hidden, inter)
	if err != nil {
		return false, err
	}
	defer unlock()
	if err := xBuf.Upload(mlpIn); err != nil {
		return false, fmt.Errorf("upload GPU MLP input: %w", err)
	}
	if err := nvidia.GemvNVFP4Buffer(gateBuf, xBuf, gwGate); err != nil {
		return false, fmt.Errorf("GPU MLP gate GEMV: %w", err)
	}
	if err := nvidia.GemvNVFP4Buffer(upBuf, xBuf, gwUp); err != nil {
		return false, fmt.Errorf("GPU MLP up GEMV: %w", err)
	}
	if err := nvidia.F32SiLUMulBuffer(gateBuf, gateBuf, upBuf, inter); err != nil {
		return false, fmt.Errorf("GPU MLP SiLU*up: %w", err)
	}
	if err := nvidia.GemvNVFP4Buffer(outBuf, gateBuf, gwDown); err != nil {
		return false, fmt.Errorf("GPU MLP down GEMV: %w", err)
	}
	if err := outBuf.Download(out); err != nil {
		return false, fmt.Errorf("download GPU MLP output: %w", err)
	}
	return true, nil
}

func qwen35MLPScratchBuffers(hidden, inter int) (x, gate, up, out *nvidia.Buffer, unlock func(), err error) {
	qwen35MLPGPUScratch.Lock()
	unlock = func() { qwen35MLPGPUScratch.Unlock() }
	if qwen35MLPGPUScratch.x == nil || qwen35MLPGPUScratch.xN < hidden {
		if qwen35MLPGPUScratch.x != nil {
			qwen35MLPGPUScratch.x.Free()
		}
		qwen35MLPGPUScratch.x, err = nvidia.Malloc(hidden)
		if err != nil {
			unlock()
			return nil, nil, nil, nil, nil, fmt.Errorf("alloc GPU MLP input: %w", err)
		}
		qwen35MLPGPUScratch.xN = hidden
	}
	if qwen35MLPGPUScratch.gate == nil || qwen35MLPGPUScratch.interN < inter {
		if qwen35MLPGPUScratch.gate != nil {
			qwen35MLPGPUScratch.gate.Free()
		}
		if qwen35MLPGPUScratch.up != nil {
			qwen35MLPGPUScratch.up.Free()
		}
		qwen35MLPGPUScratch.gate, err = nvidia.Malloc(inter)
		if err != nil {
			unlock()
			return nil, nil, nil, nil, nil, fmt.Errorf("alloc GPU MLP gate: %w", err)
		}
		qwen35MLPGPUScratch.up, err = nvidia.Malloc(inter)
		if err != nil {
			unlock()
			return nil, nil, nil, nil, nil, fmt.Errorf("alloc GPU MLP up: %w", err)
		}
		qwen35MLPGPUScratch.interN = inter
	}
	if qwen35MLPGPUScratch.out == nil || qwen35MLPGPUScratch.outN < hidden {
		if qwen35MLPGPUScratch.out != nil {
			qwen35MLPGPUScratch.out.Free()
		}
		qwen35MLPGPUScratch.out, err = nvidia.Malloc(hidden)
		if err != nil {
			unlock()
			return nil, nil, nil, nil, nil, fmt.Errorf("alloc GPU MLP out: %w", err)
		}
		qwen35MLPGPUScratch.outN = hidden
	}
	return qwen35MLPGPUScratch.x, qwen35MLPGPUScratch.gate, qwen35MLPGPUScratch.up, qwen35MLPGPUScratch.out, unlock, nil
}
