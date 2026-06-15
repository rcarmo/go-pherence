package diffusiongemma

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

var f32GELUExactMulScratch = struct {
	sync.Mutex
	gate []float32
	up   []float32
}{}

func ensureF32GELUExactMulScratch(n int) (gate, up []float32) {
	if cap(f32GELUExactMulScratch.gate) < n {
		f32GELUExactMulScratch.gate = make([]float32, n)
	}
	if cap(f32GELUExactMulScratch.up) < n {
		f32GELUExactMulScratch.up = make([]float32, n)
	}
	return f32GELUExactMulScratch.gate[:n], f32GELUExactMulScratch.up[:n]
}

func freeF32GELUExactMulScratch() {
	f32GELUExactMulScratch.Lock()
	f32GELUExactMulScratch.gate = nil
	f32GELUExactMulScratch.up = nil
	f32GELUExactMulScratch.Unlock()
}

type f32GELUExactMulStats struct {
	Calls      uint64
	Elements   uint64
	DownloadNS uint64
	GELUNS     uint64
	UploadNS   uint64
}

var f32GELUExactMulCounters struct {
	calls      atomic.Uint64
	elements   atomic.Uint64
	downloadNS atomic.Uint64
	geluNS     atomic.Uint64
	uploadNS   atomic.Uint64
}

func f32GELUExactMulSnapshot() f32GELUExactMulStats {
	return f32GELUExactMulStats{
		Calls:      f32GELUExactMulCounters.calls.Load(),
		Elements:   f32GELUExactMulCounters.elements.Load(),
		DownloadNS: f32GELUExactMulCounters.downloadNS.Load(),
		GELUNS:     f32GELUExactMulCounters.geluNS.Load(),
		UploadNS:   f32GELUExactMulCounters.uploadNS.Load(),
	}
}

func (s f32GELUExactMulStats) Sub(base f32GELUExactMulStats) f32GELUExactMulStats {
	return f32GELUExactMulStats{
		Calls:      s.Calls - base.Calls,
		Elements:   s.Elements - base.Elements,
		DownloadNS: s.DownloadNS - base.DownloadNS,
		GELUNS:     s.GELUNS - base.GELUNS,
		UploadNS:   s.UploadNS - base.UploadNS,
	}
}

func resetF32GELUExactMulStats() {
	f32GELUExactMulCounters.calls.Store(0)
	f32GELUExactMulCounters.elements.Store(0)
	f32GELUExactMulCounters.downloadNS.Store(0)
	f32GELUExactMulCounters.geluNS.Store(0)
	f32GELUExactMulCounters.uploadNS.Store(0)
}

// f32GELUExactMulBuffer computes gate = exact_gelu(gate) * up in-place.
// llama.cpp's Gemma4/DiffusionGemma graph uses ggml_gelu (erf-based), not the
// tanh approximation. Until we add an exact CUDA activation kernel, keep
// opt-in device-resident MLP paths numerically aligned by making this explicit
// host boundary rather than silently using the faster gelu_tanh kernel.
func f32GELUExactMulBuffer(gate, up *gpu.Buffer, n int) error {
	if gate == nil || up == nil || n <= 0 {
		return fmt.Errorf("invalid exact GELU device activation buffers")
	}
	maxInt := int(^uint(0) >> 1)
	if n > maxInt/4 {
		return fmt.Errorf("exact GELU device activation byte-size overflow n=%d", n)
	}
	needBytes := n * 4
	if needBytes > gate.Size || needBytes > up.Size {
		return fmt.Errorf("exact GELU device activation short buffer n=%d bytes=%d gate=%d up=%d", n, needBytes, gate.Size, up.Size)
	}
	f32GELUExactMulScratch.Lock()
	defer f32GELUExactMulScratch.Unlock()
	gateHost, upHost := ensureF32GELUExactMulScratch(n)
	downloadStart := time.Now()
	if err := gate.Download(gateHost); err != nil {
		return fmt.Errorf("download exact GELU gate: %w", err)
	}
	if err := up.Download(upHost); err != nil {
		return fmt.Errorf("download exact GELU up: %w", err)
	}
	f32GELUExactMulCounters.downloadNS.Add(uint64(time.Since(downloadStart).Nanoseconds()))
	geluStart := time.Now()
	if !simd.GELUExactMulTo(gateHost, gateHost, upHost) {
		return fmt.Errorf("exact GELU activation rejected n=%d", n)
	}
	f32GELUExactMulCounters.geluNS.Add(uint64(time.Since(geluStart).Nanoseconds()))
	uploadStart := time.Now()
	if err := gate.Upload(gateHost); err != nil {
		return fmt.Errorf("upload exact GELU activation: %w", err)
	}
	f32GELUExactMulCounters.uploadNS.Add(uint64(time.Since(uploadStart).Nanoseconds()))
	f32GELUExactMulCounters.calls.Add(1)
	f32GELUExactMulCounters.elements.Add(uint64(n))
	return nil
}
