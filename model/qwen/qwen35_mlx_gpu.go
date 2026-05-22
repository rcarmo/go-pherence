package qwen

import nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"

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
	if err := ob.ToGPU(); err != nil {
		return false
	}
	nvidia.GemvMLXDirect(ob, xb, w)
	nvidia.Sync()
	copy(out[:outDim], ob.Data()[:outDim])
	return true
}

func resetQwen35MLXGPUScratch() {
	qwen35MLXGPUScratch.Lock()
	defer qwen35MLXGPUScratch.Unlock()
	if qwen35MLXGPUScratch.x != nil {
		qwen35MLXGPUScratch.x.Free()
	}
	if qwen35MLXGPUScratch.out != nil {
		qwen35MLXGPUScratch.out.Free()
	}
	qwen35MLXGPUScratch.x = nil
	qwen35MLXGPUScratch.out = nil
	qwen35MLXGPUScratch.xN = 0
	qwen35MLXGPUScratch.outN = 0
}
