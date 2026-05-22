package main

import (
	"time"

	"github.com/rcarmo/go-pherence/backends/mlx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/model/qwen"
)

func qwen35ArgmaxMLXGPU(logits []float32, w *mlx.QuantWeight, x []float32) bool {
	if w == nil || len(logits) < w.OutDim || len(x) < w.InDim {
		return false
	}
	gw, err := qwen.CacheQwen35MLXWeight(w)
	if err != nil || gw == nil {
		return false
	}
	start := time.Now()
	xb := nvidia.NewDevBufFrom(x)
	ob := nvidia.NewDevBuf(w.OutDim)
	defer xb.Free()
	defer ob.Free()
	if xb.ToGPU() != nil || ob.ToGPU() != nil {
		return false
	}
	nvidia.GemvMLXDirect(ob, xb, gw)
	nvidia.Sync()
	copy(logits[:w.OutDim], ob.Data()[:w.OutDim])
	qwen36LMHeadStats.GPUMillis += time.Since(start).Milliseconds()
	return true
}
