package qwen

import (
	"fmt"
	llmops "github.com/rcarmo/go-pherence/model/internal/ops"
	"math"
	"sync"
	"time"

	"github.com/rcarmo/go-pherence/backends/mlx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simdnvfp4 "github.com/rcarmo/go-pherence/backends/simd/quant/nvfp4"
	"github.com/rcarmo/go-pherence/tensor"
)

var qwen35GPUEnabled bool
var qwen35GPUReady bool
var qwen35GPUVerifyRemaining int
var qwen35GPUVerifyTolerance float32 = 1e-4
var qwen35GPUVerifyCompared int64
var qwen35GPUVerifyMismatches int64
var qwen35GPUVerifyMaxDiff float32
var qwen35LinearStats Qwen35LinearStats
var qwen35LinearFailCounts = map[string]int64{}
var qwen35LinearTiming bool
var qwen35GPUMLPEnabled bool
var qwen35GPUMLXOverflowEnabled bool

var qwen35MLPGPUScratch = struct {
	sync.Mutex
	x, gate, up, out *nvidia.Buffer
	xN, interN, outN int
}{}

var qwen35MLXGPUScratch = struct {
	sync.Mutex
	x, gate, up, out *nvidia.DevBuf
	xN, interN, outN int
}{}

type Qwen35LinearStats struct {
	Calls        int64    `json:"calls"`
	GPUCalls     int64    `json:"gpu_calls"`
	CPUCalls     int64    `json:"cpu_calls"`
	GPUFailures  int64    `json:"gpu_failures,omitempty"`
	GPUMillis    int64    `json:"gpu_ms"`
	GPUUploadMS  int64    `json:"gpu_upload_ms,omitempty"`
	GPUKernelMS  int64    `json:"gpu_kernel_ms,omitempty"`
	CPUMillis    int64    `json:"cpu_ms"`
	VerifyMillis int64    `json:"verify_ms"`
	AvgGPUCallMS float64  `json:"avg_gpu_call_ms,omitempty"`
	GPUFailTop   []string `json:"gpu_fail_top,omitempty"`
}

type Qwen35GPUVerifyStats struct {
	Compared   int64   `json:"compared"`
	Mismatches int64   `json:"mismatches"`
	MaxDiff    float32 `json:"max_diff"`
	Tolerance  float32 `json:"tolerance"`
}

func SetQwen35GPUEnabled(enabled bool) {
	qwen35GPUEnabled = enabled
	qwen35GPUReady = enabled && nvidia.SgemmReady()
}

func SetQwen35LinearTiming(enabled bool)         { qwen35LinearTiming = enabled }
func SetQwen35GPUMLPEnabled(enabled bool)        { qwen35GPUMLPEnabled = enabled }
func SetQwen35GPUMXOverflowEnabled(enabled bool) { qwen35GPUMLXOverflowEnabled = enabled }

func SetQwen35GPUVerify(limit int, tolerance float32) {
	qwen35GPUVerifyRemaining = limit
	if tolerance > 0 {
		qwen35GPUVerifyTolerance = tolerance
	}
	qwen35GPUVerifyCompared = 0
	qwen35GPUVerifyMismatches = 0
	qwen35GPUVerifyMaxDiff = 0
}

func Qwen35GPUVerifyStatsSnapshot() Qwen35GPUVerifyStats {
	return Qwen35GPUVerifyStats{Compared: qwen35GPUVerifyCompared, Mismatches: qwen35GPUVerifyMismatches, MaxDiff: qwen35GPUVerifyMaxDiff, Tolerance: qwen35GPUVerifyTolerance}
}

func ResetQwen35LinearStats() {
	qwen35LinearStats = Qwen35LinearStats{}
	qwen35LinearFailCounts = map[string]int64{}
}

func Qwen35LinearStatsSnapshot() Qwen35LinearStats {
	out := qwen35LinearStats
	if out.GPUCalls > 0 {
		out.AvgGPUCallMS = float64(out.GPUMillis) / float64(out.GPUCalls)
	}
	for name, n := range qwen35LinearFailCounts {
		out.GPUFailTop = append(out.GPUFailTop, fmt.Sprintf("%s:%d", name, n))
		if len(out.GPUFailTop) >= 8 {
			break
		}
	}
	return out
}

func qwen35LinearInto(out, x []float32, dense *tensor.Tensor, q *Qwen35NVFP4Weight, m *mlx.QuantWeight, inDim, outDim int, name string) error {
	if len(out) != outDim || len(x) != inDim {
		return fmt.Errorf("%s vector dims out/x=%d/%d want %d/%d", name, len(out), len(x), outDim, inDim)
	}
	if q != nil {
		if q.W == nil {
			return fmt.Errorf("%s nil NVFP4 weight", name)
		}
		if q.W.InDim != inDim || q.W.OutDim != outDim {
			return fmt.Errorf("%s NVFP4 dims out/in=%d/%d want %d/%d", name, q.W.OutDim, q.W.InDim, outDim, inDim)
		}
		qwen35LinearStats.Calls++
		if qwen35GPUReady {
			var start time.Time
			if qwen35LinearTiming {
				start = time.Now()
			}
			gw, transient, err := qwen35CachedGPUWeight(q)
			if err != nil {
				return fmt.Errorf("%s upload/cache NVFP4 GPU: %w", name, err)
			}
			var uploadMS int64
			var kernelStart time.Time
			if qwen35LinearTiming {
				uploadMS = time.Since(start).Milliseconds()
				kernelStart = time.Now()
			}
			if transient {
				defer gw.Free()
			}
			if err := nvidia.GemvNVFP4(out, x, gw); err != nil {
				if qwen35LinearTiming {
					qwen35LinearStats.GPUMillis += time.Since(start).Milliseconds()
				}
				return fmt.Errorf("%s GPU NVFP4 GEMV: %w", name, err)
			}
			if qwen35LinearTiming {
				qwen35LinearStats.GPUKernelMS += time.Since(kernelStart).Milliseconds()
				qwen35LinearStats.GPUMillis += time.Since(start).Milliseconds()
				qwen35LinearStats.GPUUploadMS += uploadMS
			}
			qwen35LinearStats.GPUCalls++
			if qwen35GPUVerifyRemaining > 0 {
				verifyStart := time.Now()
				qwen35GPUVerifyRemaining--
				ref := make([]float32, outDim)
				simdnvfp4.GemvNVFP4(ref, x, q.W)
				var maxDiff float32
				for i := range ref {
					d := float32(math.Abs(float64(ref[i] - out[i])))
					if d > maxDiff {
						maxDiff = d
					}
				}
				qwen35GPUVerifyCompared++
				if maxDiff > qwen35GPUVerifyMaxDiff {
					qwen35GPUVerifyMaxDiff = maxDiff
				}
				qwen35LinearStats.VerifyMillis += time.Since(verifyStart).Milliseconds()
				if maxDiff > qwen35GPUVerifyTolerance {
					qwen35GPUVerifyMismatches++
					return fmt.Errorf("%s GPU NVFP4 verification max diff=%g exceeds tolerance=%g", name, maxDiff, qwen35GPUVerifyTolerance)
				}
			}
			return nil
		}
		var start time.Time
		if qwen35LinearTiming {
			start = time.Now()
		}
		simdnvfp4.GemvNVFP4(out, x, q.W)
		qwen35LinearStats.CPUCalls++
		if qwen35LinearTiming {
			qwen35LinearStats.CPUMillis += time.Since(start).Milliseconds()
		}
		return nil
	}
	if m != nil {
		if m.InDim != inDim || m.OutDim != outDim {
			return fmt.Errorf("%s MLX dims out/in=%d/%d want %d/%d", name, m.OutDim, m.InDim, outDim, inDim)
		}
		qwen35LinearStats.Calls++
		if qwen35GPUReady {
			start := time.Now()
			gw, err := qwen35CachedGPUMXWeight(m)
			transient := false
			if err != nil && qwen35GPUMLXOverflowEnabled {
				gw, err = qwen35TransientGPUMXWeight(m)
				transient = err == nil
			}
			if err == nil {
				if transient {
					defer gw.Free()
				}
				if qwen35GemvMLXGPU(out, x, gw, inDim, outDim) {
					qwen35LinearStats.GPUCalls++
					if qwen35LinearTiming {
						qwen35LinearStats.GPUMillis += time.Since(start).Milliseconds()
					}
					return nil
				}
				qwen35LinearStats.GPUFailures++
				qwen35LinearFailCounts[name]++
			}
		}
		if !mlx.GemvTo(out, x, m) {
			return fmt.Errorf("%s MLX GEMV failed", name)
		}
		qwen35LinearStats.CPUCalls++
		return nil
	}
	if dense == nil {
		return fmt.Errorf("missing %s", name)
	}
	llmops.GemvNT(out, x, dense.Data(), inDim, outDim)
	return nil
}
