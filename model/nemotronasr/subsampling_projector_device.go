package nemotronasr

import (
	"context"
	"fmt"
	"math"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

// DeviceSubsamplingProjector keeps one projection's weights and buffers
// resident for a single CPU PCM-to-text request. The GPU executes only the
// subsampling projection. The legacy path uploads/downloads each four-row
// chunk. Explicit SharedMemory instead uses scoped native producer/consumer
// access. The request must not be used concurrently. Close releases all
// request-owned resources once their GPU use is confirmed complete.
type DeviceSubsamplingProjector struct {
	Backend               string // "ptx" or "vulkan"
	SharedMemory          bool   // explicit Vulkan-only opt-in; immutable after first use
	ready                 bool
	closed                bool
	dispatches            int // successful GPU projections; observable after Close
	stats                 ProjectionTransferStats
	ptxX, ptxW, ptxY      *ptx.Buffer
	vkOp                  *vulkan.VkLinearF32
	vkArena               *vulkan.VkTensorArena
	vkX, vkW, vkBias, vkY *vulkan.VkTensorF32
}

// ProjectionTransferStats counts application-level boundary copies and scoped
// borrows. Weight preparation is separate; driver-internal movement is unknown.
// Request objects are serial; snapshot only outside a projection callback.
type ProjectionTransferStats struct {
	InputCopyBytes, OutputCopyBytes, WeightUploadBytes uint64
	CPUWrites, CPUReads                                uint64
	Dispatches                                         int
}

func (p *DeviceSubsamplingProjector) TransferStats() ProjectionTransferStats {
	if p == nil {
		return ProjectionTransferStats{}
	}
	s := p.stats
	s.Dispatches = p.dispatches
	return s
}

func (p *DeviceSubsamplingProjector) Close() error {
	if p == nil {
		return nil
	}
	p.ready = false
	p.closed = true
	if p.ptxX != nil {
		p.ptxX.Free()
		p.ptxX = nil
	}
	if p.ptxW != nil {
		p.ptxW.Free()
		p.ptxW = nil
	}
	if p.ptxY != nil {
		p.ptxY.Free()
		p.ptxY = nil
	}
	var err error
	if p.vkArena != nil {
		err = p.vkArena.Close()
		if err != nil {
			return err // retain owners for an explicit drain then Close retry
		}
		p.vkArena = nil
	}
	if p.vkOp != nil {
		if closeErr := p.vkOp.Close(); closeErr != nil {
			return closeErr
		}
		p.vkOp = nil
	}
	p.vkX, p.vkW, p.vkBias, p.vkY = nil, nil, nil, nil
	return err
}

func (p *DeviceSubsamplingProjector) prepare(ctx context.Context, weight, bias []float32) (err error) {
	const in, out = 4352, 1024
	defer func() {
		if err != nil {
			_ = p.Close()
		}
	}()
	switch p.Backend {
	case "ptx":
		if !ptx.Init() || !ptx.SgemmReady() {
			return fmt.Errorf("Nemotron ASR PTX SGEMM unavailable")
		}
		transposed := make([]float32, len(weight))
		for col := 0; col < in; col++ {
			for row := 0; row < out; row++ {
				transposed[col*out+row] = weight[row*in+col]
			}
		}
		if p.ptxX, err = ptx.Malloc(4 * in); err != nil {
			return err
		}
		if p.ptxW, err = ptx.Malloc(len(weight)); err != nil {
			return err
		}
		if p.ptxY, err = ptx.Malloc(4 * out); err != nil {
			return err
		}
		if err = p.ptxW.Upload(transposed); err != nil {
			return err
		}
	case "vulkan":
		if !vulkan.VulkanInit() {
			return fmt.Errorf("Nemotron ASR Vulkan unavailable")
		}
		if p.vkOp, err = vulkan.NewVkLinearF32(ctx); err != nil {
			return err
		}
		if p.SharedMemory {
			p.vkArena, err = vulkan.NewVkSharedTensorArena(ctx, 24<<20)
		} else {
			p.vkArena, err = vulkan.NewVkTensorArena(ctx, 24<<20)
		}
		if err != nil {
			return err
		}
		alloc := func(shape ...int) (*vulkan.VkTensorF32, error) { return p.vkArena.AllocF32(ctx, shape...) }
		if p.vkX, err = alloc(4, in); err != nil {
			return err
		}
		if p.vkW, err = alloc(out, in); err != nil {
			return err
		}
		if p.vkBias, err = alloc(out); err != nil {
			return err
		}
		if p.vkY, err = alloc(4, out); err != nil {
			return err
		}
		if err = p.vkW.Upload(ctx, weight); err != nil {
			return err
		}
		if err = p.vkBias.Upload(ctx, bias); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unsupported Nemotron ASR projection backend %q", p.Backend)
	}
	p.stats.WeightUploadBytes += uint64(len(weight)+len(bias)) * 4
	p.ready = true
	return nil
}

func (p *DeviceSubsamplingProjector) Project(ctx context.Context, input, weight, bias []float32, rows int) ([]float32, error) {
	const in, out = 4352, 1024
	if p != nil && p.SharedMemory {
		if rows != 4 || len(input) != rows*in {
			return nil, fmt.Errorf("invalid Nemotron ASR device projection input")
		}
		var owned []float32
		err := p.ProjectScoped(ctx, weight, bias, rows, func(mapped []float32) error {
			copy(mapped, input)
			p.stats.InputCopyBytes += uint64(len(input)) * 4
			return nil
		}, func(mapped []float32) error {
			owned = append([]float32(nil), mapped...)
			p.stats.OutputCopyBytes += uint64(len(mapped)) * 4
			return nil
		})
		if err != nil {
			return nil, err
		}
		return owned, nil // compatibility API intentionally copies in/out
	}
	if p == nil || p.closed || ctx == nil || rows != 4 || len(input) != rows*in || len(weight) != out*in || len(bias) != out {
		return nil, fmt.Errorf("invalid Nemotron ASR device projection shape")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, v := range input {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR device projection input")
		}
	}
	if !p.ready {
		for _, values := range [][]float32{weight, bias} {
			for _, v := range values {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					return nil, fmt.Errorf("non-finite Nemotron ASR device projection weights")
				}
			}
		}
		if err := p.prepare(ctx, weight, bias); err != nil {
			return nil, err
		}
	}
	result := make([]float32, rows*out)
	switch p.Backend {
	case "ptx":
		if err := p.ptxX.Upload(input); err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := ptx.Sgemm(rows, out, in, 1, p.ptxX, p.ptxW, p.ptxY); err != nil {
			return nil, err
		}
		if err := ptx.SyncErr(); err != nil {
			return nil, err
		}
		if err := p.ptxY.Download(result); err != nil {
			return nil, err
		}
		for i := range result {
			result[i] += bias[i%out]
		}
	case "vulkan":
		if err := p.vkX.Upload(ctx, input); err != nil {
			return nil, err
		}
		if err := p.vkOp.Forward(ctx, p.vkY, p.vkX, p.vkW, p.vkBias); err != nil {
			return nil, err
		}
		if err := p.vkY.Download(ctx, result); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unsupported Nemotron ASR projection backend %q", p.Backend)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	p.stats.InputCopyBytes += uint64(len(input)) * 4
	p.stats.OutputCopyBytes += uint64(len(result)) * 4
	p.dispatches++
	return result, nil
}

// ScopedProjectionEnabled preserves copied Vulkan/PTX behavior unless the
// request explicitly selects shared memory. It never silently falls back.
func (p *DeviceSubsamplingProjector) ScopedProjectionEnabled() bool {
	return p != nil && p.SharedMemory
}

// ProjectScoped runs CPU producer -> GPU projection -> CPU consumer against
// native shared storage, with no per-chunk upload/download. Immutable weights
// are uploaded once. Request objects must not be used concurrently; callbacks
// must obey ScopedSubsamplingProjector's lifetime and no-reentry contract.
func (p *DeviceSubsamplingProjector) ProjectScoped(ctx context.Context, weight, bias []float32, rows int, produce, consume func([]float32) error) error {
	const in, out = 4352, 1024
	if p == nil || p.closed || !p.SharedMemory || p.Backend != "vulkan" || ctx == nil || rows != 4 || len(weight) != out*in || len(bias) != out || produce == nil || consume == nil {
		return fmt.Errorf("invalid Nemotron ASR shared projection")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if !p.ready {
		for _, values := range [][]float32{weight, bias} {
			for _, v := range values {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					return fmt.Errorf("non-finite Nemotron ASR shared projection weights")
				}
			}
		}
		if err := p.prepare(ctx, weight, bias); err != nil {
			return err
		}
	}
	if err := p.vkX.WithCPUWrite(ctx, func(input []float32) error {
		// A failed producer cannot leave stale request data for the next call.
		clear(input)
		p.stats.CPUWrites++
		if err := produce(input); err != nil {
			return err
		}
		for _, v := range input {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("non-finite Nemotron ASR shared projection input")
			}
		}
		return nil
	}); err != nil {
		return err
	}
	if err := p.vkOp.Forward(ctx, p.vkY, p.vkX, p.vkW, p.vkBias); err != nil {
		return err
	}
	if err := p.vkY.WithCPURead(ctx, func(output []float32) error {
		p.stats.CPUReads++
		for _, v := range output {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("non-finite Nemotron ASR shared projection output")
			}
		}
		return consume(output)
	}); err != nil {
		return err
	}
	p.dispatches++
	return nil
}
