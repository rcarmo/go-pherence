package nemotrondiarization

import (
	"context"
	"errors"
	"fmt"
	"math"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

// DeviceStackingProjector keeps weights and bounded buffers resident for one
// CPU PCM-to-speaker-segments request. Only stack projection runs on the GPU;
// each projected chunk includes upload, dispatch and download.
// Narrow close seams let failure tests use inert owners without a Vulkan loader.
var projectorVKArenaClose = (*vulkan.VkTensorArena).Close
var projectorVKOpClose = (*vulkan.VkLinearF32).Close

type DeviceStackingProjector struct {
	Backend               string // "ptx" or "vulkan"
	ready                 bool
	closed                bool
	terminal              error
	uncertain             bool // failed PTX driver call: later sync cannot erase uncertainty
	Dispatches            int
	ptxX, ptxW, ptxY      *ptx.Buffer
	vkOp                  *vulkan.VkLinearF32
	vkArena               *vulkan.VkTensorArena
	vkX, vkW, vkBias, vkY *vulkan.VkTensorF32
}

func (p *DeviceStackingProjector) Close() error {
	if p == nil {
		return nil
	}
	if p.closed {
		return p.terminal
	}
	p.ready = false
	p.closed = true
	if p.uncertain {
		return p.terminal // retain owners without another driver call
	}
	if p.Backend == "ptx" && (p.ptxX != nil || p.ptxW != nil || p.ptxY != nil) {
		if err := ptxCleanupSync(); err != nil {
			p.terminal = errors.Join(p.terminal, fmt.Errorf("PTX stacking close sync: %w", err))
			return p.terminal
		}
		for _, owner := range []**ptx.Buffer{&p.ptxX, &p.ptxW, &p.ptxY} {
			if *owner != nil {
				if err := ptxCleanupFree(*owner); err != nil {
					p.terminal = errors.Join(p.terminal, fmt.Errorf("PTX stacking close free: %w", err))
					return p.terminal
				}
				*owner = nil
			}
		}
	}
	if p.vkArena != nil {
		if err := projectorVKArenaClose(p.vkArena); err != nil {
			p.terminal = errors.Join(p.terminal, fmt.Errorf("Vulkan stacking arena close: %w", err))
			return p.terminal // retain failed and unvisited owners
		}
		p.vkArena = nil
	}
	if p.vkOp != nil {
		if err := projectorVKOpClose(p.vkOp); err != nil {
			p.terminal = errors.Join(p.terminal, fmt.Errorf("Vulkan stacking operator close: %w", err))
			return p.terminal
		}
		p.vkOp = nil
	}
	p.vkX, p.vkW, p.vkBias, p.vkY = nil, nil, nil, nil
	return p.terminal
}

func (p *DeviceStackingProjector) prepare(ctx context.Context, weight []float32) (err error) {
	const in, out, maxRows = stackWidth, projectedWidth, 64
	defer func() {
		if err != nil {
			err = errors.Join(err, p.Close())
		}
	}()
	switch p.Backend {
	case "ptx":
		if !ptx.Init() || !ptx.SgemmReady() {
			return fmt.Errorf("Nemotron diarization PTX SGEMM unavailable")
		}
		transposed := make([]float32, len(weight))
		for col := 0; col < in; col++ {
			for row := 0; row < out; row++ {
				transposed[col*out+row] = weight[row*in+col]
			}
		}
		if p.ptxX, err = ptx.Malloc(maxRows * in); err != nil {
			return err
		}
		if p.ptxW, err = ptx.Malloc(len(weight)); err != nil {
			return err
		}
		if p.ptxY, err = ptx.Malloc(maxRows * out); err != nil {
			return err
		}
		if err = p.ptxW.Upload(transposed); err != nil {
			p.uncertain = true
			p.terminal = fmt.Errorf("PTX stacking weight upload: %w", err)
			return err
		}
	case "vulkan":
		if !vulkan.VulkanInit() {
			return fmt.Errorf("Nemotron diarization Vulkan unavailable")
		}
		if p.vkOp, err = vulkan.NewVkLinearF32(ctx); err != nil {
			return err
		}
		if p.vkArena, err = vulkan.NewVkTensorArena(ctx, 8<<20); err != nil {
			return err
		}
		alloc := func(shape ...int) (*vulkan.VkTensorF32, error) { return p.vkArena.AllocF32(ctx, shape...) }
		if p.vkX, err = alloc(maxRows, in); err != nil {
			return err
		}
		if p.vkW, err = alloc(out, in); err != nil {
			return err
		}
		if p.vkBias, err = alloc(out); err != nil {
			return err
		}
		if p.vkY, err = alloc(maxRows, out); err != nil {
			return err
		}
		if err = p.vkW.Upload(ctx, weight); err != nil {
			return err
		}
		if err = p.vkBias.Upload(ctx, make([]float32, out)); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unsupported Nemotron diarization projection backend %q", p.Backend)
	}
	p.ready = true
	return nil
}

func (p *DeviceStackingProjector) Project(ctx context.Context, stacked, weight []float32, rows int) ([]float32, error) {
	const in, out, maxRows = stackWidth, projectedWidth, 64
	if p == nil {
		return nil, fmt.Errorf("invalid Nemotron diarization device projection shape")
	}
	if p.terminal != nil {
		return nil, p.terminal
	}
	if p.closed || ctx == nil || rows < 1 || rows > maxRows || len(stacked) != rows*in || len(weight) != out*in {
		return nil, fmt.Errorf("invalid Nemotron diarization device projection shape")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, v := range stacked {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite Nemotron diarization device projection")
		}
	}
	if !p.ready {
		for _, v := range weight {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("non-finite Nemotron diarization device weights")
			}
		}
		if err := p.prepare(ctx, weight); err != nil {
			return nil, err
		}
	}
	// Resident buffers are 64 rows. Zero the tail so the fixed-shape GPU
	// dispatch cannot consume stale rows from a preceding chunk.
	padded := make([]float32, maxRows*in)
	copy(padded, stacked)
	result := make([]float32, maxRows*out)
	switch p.Backend {
	case "ptx":
		if err := p.ptxX.Upload(padded); err != nil {
			p.uncertain = true
			p.terminal = fmt.Errorf("PTX stacking projection upload: %w", err)
			return nil, p.terminal
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := ptx.Sgemm(maxRows, out, in, 1, p.ptxX, p.ptxW, p.ptxY); err != nil {
			p.uncertain = true
			p.terminal = fmt.Errorf("PTX stacking projection launch: %w", err)
			return nil, p.terminal
		}
		if err := ptx.SyncErr(); err != nil {
			p.uncertain = true
			p.terminal = fmt.Errorf("PTX stacking projection sync: %w", err)
			return nil, p.terminal
		}
		if err := p.ptxY.Download(result); err != nil {
			p.uncertain = true
			p.terminal = fmt.Errorf("PTX stacking projection download: %w", err)
			return nil, p.terminal
		}
	case "vulkan":
		if err := p.vkX.Upload(ctx, padded); err != nil {
			return nil, err
		}
		if err := p.vkOp.Forward(ctx, p.vkY, p.vkX, p.vkW, p.vkBias); err != nil {
			return nil, err
		}
		if err := p.vkY.Download(ctx, result); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unsupported Nemotron diarization projection backend %q", p.Backend)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	p.Dispatches++
	return result[:rows*out], nil
}
