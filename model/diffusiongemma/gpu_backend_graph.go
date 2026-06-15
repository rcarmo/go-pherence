package diffusiongemma

import (
	"fmt"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
)

// DiffusionGemmaBackendGraph is an explicit GPU-resident execution graph for
// reusable DiffusionGemma subgraphs. It owns device work buffers and is meant to
// replace one-off host-returning calls with scheduled buffer-to-buffer kernels.
//
// The first implemented island is the shared dense MLP:
//
//	hidden[batch, H]
//	  -> FP8 gate/up dequant-SGEMM
//	  -> exact GELU(gate) * up (host boundary until exact CUDA GELU exists)
//	  -> FP8 down dequant-SGEMM
//	  -> hidden[batch, H]
//
// This mirrors llama.cpp's backend graph direction: keep intermediate tensors on
// the backend and only copy at explicit graph boundaries.
type DiffusionGemmaBackendGraph struct {
	batch        int
	hiddenSize   int
	intermediate int
	hidLen       int
	midLen       int
	x            *gpu.Buffer
	gate         *gpu.Buffer
	up           *gpu.Buffer
	down         *gpu.Buffer
}

func NewDiffusionGemmaBackendGraph(batch, hiddenSize, intermediate int) (*DiffusionGemmaBackendGraph, error) {
	hidLen, okHid := checked.MulInt(batch, hiddenSize)
	midLen, okMid := checked.MulInt(batch, intermediate)
	if batch <= 0 || hiddenSize <= 0 || intermediate <= 0 || !okHid || !okMid {
		return nil, fmt.Errorf("DiffusionGemma backend graph invalid dims batch=%d hidden=%d intermediate=%d", batch, hiddenSize, intermediate)
	}
	g := &DiffusionGemmaBackendGraph{batch: batch, hiddenSize: hiddenSize, intermediate: intermediate, hidLen: hidLen, midLen: midLen}
	var err error
	if g.x, err = gpu.Malloc(hidLen); err != nil {
		g.Free()
		return nil, fmt.Errorf("backend graph alloc hidden: %w", err)
	}
	if g.gate, err = gpu.Malloc(midLen); err != nil {
		g.Free()
		return nil, fmt.Errorf("backend graph alloc gate: %w", err)
	}
	if g.up, err = gpu.Malloc(midLen); err != nil {
		g.Free()
		return nil, fmt.Errorf("backend graph alloc up: %w", err)
	}
	if g.down, err = gpu.Malloc(hidLen); err != nil {
		g.Free()
		return nil, fmt.Errorf("backend graph alloc down: %w", err)
	}
	return g, nil
}

func (g *DiffusionGemmaBackendGraph) Free() {
	if g == nil {
		return
	}
	if g.x != nil {
		g.x.Free()
		g.x = nil
	}
	if g.gate != nil {
		g.gate.Free()
		g.gate = nil
	}
	if g.up != nil {
		g.up.Free()
		g.up = nil
	}
	if g.down != nil {
		g.down.Free()
		g.down = nil
	}
}

func (g *DiffusionGemmaBackendGraph) Compatible(batch, hiddenSize, intermediate int) bool {
	return g != nil && g.batch == batch && g.hiddenSize == hiddenSize && g.intermediate == intermediate && g.x != nil && g.gate != nil && g.up != nil && g.down != nil
}

func (g *DiffusionGemmaBackendGraph) DenseMLP(hidden []float32, fl *GPUFP8Layer) error {
	if g == nil || fl == nil || fl.Gate == nil || fl.Up == nil || fl.Down == nil {
		return fmt.Errorf("DiffusionGemma backend graph dense MLP missing graph or linears")
	}
	if len(hidden) < g.hidLen {
		return fmt.Errorf("DiffusionGemma backend graph dense MLP hidden=%d want %d", len(hidden), g.hidLen)
	}
	if fl.Gate.InDim != g.hiddenSize || fl.Up.InDim != g.hiddenSize || fl.Gate.OutDim != g.intermediate || fl.Up.OutDim != g.intermediate || fl.Down.InDim != g.intermediate || fl.Down.OutDim != g.hiddenSize {
		return fmt.Errorf("DiffusionGemma backend graph dense MLP shape mismatch gate=[%d,%d] up=[%d,%d] down=[%d,%d] graph=[batch=%d hidden=%d intermediate=%d]", fl.Gate.OutDim, fl.Gate.InDim, fl.Up.OutDim, fl.Up.InDim, fl.Down.OutDim, fl.Down.InDim, g.batch, g.hiddenSize, g.intermediate)
	}
	hiddenIn := hidden[:g.hidLen]
	var qHidden []float32
	if diffusionGemmaFP8DynamicActivationEnabled() {
		qHidden = quantizeDynamicTokenBatch(qHidden, hiddenIn, g.batch, g.hiddenSize)
		hiddenIn = qHidden
	}
	if err := g.x.Upload(hiddenIn); err != nil {
		return fmt.Errorf("backend graph upload hidden: %w", err)
	}
	if fl.GateT != nil && fl.UpT != nil {
		if err := gpu.Sgemm(g.batch, fl.Gate.OutDim, fl.Gate.InDim, 1, g.x, fl.GateT, g.gate); err != nil {
			return fmt.Errorf("backend graph gate resident SGEMM: %w", err)
		}
		if err := gpu.Sgemm(g.batch, fl.Up.OutDim, fl.Up.InDim, 1, g.x, fl.UpT, g.up); err != nil {
			return fmt.Errorf("backend graph up resident SGEMM: %w", err)
		}
	} else {
		if err := gpu.GemmFP8E4M3ViaSgemmBuffer(g.gate, g.x, g.batch, fl.Gate); err != nil {
			return fmt.Errorf("backend graph gate SGEMM: %w", err)
		}
		if err := gpu.GemmFP8E4M3ViaSgemmBuffer(g.up, g.x, g.batch, fl.Up); err != nil {
			return fmt.Errorf("backend graph up SGEMM: %w", err)
		}
	}
	if err := f32GELUExactMulBuffer(g.gate, g.up, g.midLen); err != nil {
		return fmt.Errorf("backend graph exact GELU activation: %w", err)
	}
	if diffusionGemmaFP8DynamicActivationEnabled() {
		act := make([]float32, g.midLen)
		if err := g.gate.Download(act); err != nil {
			return fmt.Errorf("backend graph download activation for dynamic FP8 down: %w", err)
		}
		qAct := quantizeDynamicTokenBatch(nil, act, g.batch, g.intermediate)
		if err := g.gate.Upload(qAct); err != nil {
			return fmt.Errorf("backend graph upload dynamic FP8 down activation: %w", err)
		}
	}
	if fl.DownT != nil {
		if err := gpu.Sgemm(g.batch, fl.Down.OutDim, fl.Down.InDim, 1, g.gate, fl.DownT, g.down); err != nil {
			return fmt.Errorf("backend graph down resident SGEMM: %w", err)
		}
	} else if err := gpu.GemmFP8E4M3ViaSgemmBuffer(g.down, g.gate, g.batch, fl.Down); err != nil {
		return fmt.Errorf("backend graph down SGEMM: %w", err)
	}
	if err := g.down.Download(hidden[:g.hidLen]); err != nil {
		return fmt.Errorf("backend graph download dense MLP: %w", err)
	}
	return nil
}
