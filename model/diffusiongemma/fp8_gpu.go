package diffusiongemma

import (
	"fmt"
	"os"
	"strings"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// GPUFP8Layer holds GPU-resident FP8 projection weights for one layer.
type GPUFP8Layer struct {
	Q    *gpu.GPUFP8E4M3Linear
	K    *gpu.GPUFP8E4M3Linear
	V    *gpu.GPUFP8E4M3Linear
	O    *gpu.GPUFP8E4M3Linear
	Gate *gpu.GPUFP8E4M3Linear
	Up   *gpu.GPUFP8E4M3Linear
	Down *gpu.GPUFP8E4M3Linear
	// Dequantized/transposed F32 dense-MLP weights [InDim, OutDim] for SGEMM.
	// These avoid per-step FP8 dequant+transpose for the shared MLP.
	GateT *gpu.Buffer
	UpT   *gpu.Buffer
	DownT *gpu.Buffer
}

func (l *GPUFP8Layer) Free() {
	if l == nil {
		return
	}
	if l.Q != nil {
		l.Q.Free()
		l.Q = nil
	}
	if l.K != nil {
		l.K.Free()
		l.K = nil
	}
	if l.V != nil {
		l.V.Free()
		l.V = nil
	}
	if l.O != nil {
		l.O.Free()
		l.O = nil
	}
	if l.Gate != nil {
		l.Gate.Free()
		l.Gate = nil
	}
	if l.Up != nil {
		l.Up.Free()
		l.Up = nil
	}
	if l.Down != nil {
		l.Down.Free()
		l.Down = nil
	}
	if l.GateT != nil {
		l.GateT.Free()
		l.GateT = nil
	}
	if l.UpT != nil {
		l.UpT.Free()
		l.UpT = nil
	}
	if l.DownT != nil {
		l.DownT.Free()
		l.DownT = nil
	}
}

// GPUFP8Model holds all GPU-resident FP8 layer weights.
type GPUFP8Model struct {
	Layers []GPUFP8Layer
}

func (m *GPUFP8Model) Free() {
	if m == nil {
		return
	}
	for i := range m.Layers {
		m.Layers[i].Free()
	}
}

// UploadFP8Layers uploads all layer projections to GPU as FP8 linears.
func UploadFP8Layers(fp8 *FP8TextWeights) (*GPUFP8Model, error) {
	if fp8 == nil {
		return nil, fmt.Errorf("nil FP8 weights")
	}
	model := &GPUFP8Model{Layers: make([]GPUFP8Layer, len(fp8.Layers))}
	cleanup := func() {
		for i := range model.Layers {
			model.Layers[i].Free()
		}
	}
	for i, lw := range fp8.Layers {
		var err error
		model.Layers[i].Q, err = gpu.UploadFP8E4M3Linear(lw.QWeight, lw.QScale, nil, lw.QShape[0], lw.QShape[1])
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("layer %d Q upload: %w", i, err)
		}
		model.Layers[i].K, err = gpu.UploadFP8E4M3Linear(lw.KWeight, lw.KScale, nil, lw.KShape[0], lw.KShape[1])
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("layer %d K upload: %w", i, err)
		}
		if lw.VWeight != nil {
			model.Layers[i].V, err = gpu.UploadFP8E4M3Linear(lw.VWeight, lw.VScale, nil, lw.VShape[0], lw.VShape[1])
			if err != nil {
				cleanup()
				return nil, fmt.Errorf("layer %d V upload: %w", i, err)
			}
		}
		model.Layers[i].O, err = gpu.UploadFP8E4M3Linear(lw.OWeight, lw.OScale, nil, lw.OShape[0], lw.OShape[1])
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("layer %d O upload: %w", i, err)
		}
		model.Layers[i].Gate, err = gpu.UploadFP8E4M3Linear(lw.GateWeight, lw.GateScale, nil, lw.GateShape[0], lw.GateShape[1])
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("layer %d gate upload: %w", i, err)
		}
		model.Layers[i].Up, err = gpu.UploadFP8E4M3Linear(lw.UpWeight, lw.UpScale, nil, lw.UpShape[0], lw.UpShape[1])
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("layer %d up upload: %w", i, err)
		}
		model.Layers[i].Down, err = gpu.UploadFP8E4M3Linear(lw.DownWeight, lw.DownScale, nil, lw.DownShape[0], lw.DownShape[1])
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("layer %d down upload: %w", i, err)
		}
		if diffusionGemmaDenseTransposeCacheEnabled() {
			model.Layers[i].GateT, err = gpu.DequantTransposeFP8E4M3Linear(model.Layers[i].Gate)
			if err != nil {
				cleanup()
				return nil, fmt.Errorf("layer %d gate transpose upload: %w", i, err)
			}
			model.Layers[i].UpT, err = gpu.DequantTransposeFP8E4M3Linear(model.Layers[i].Up)
			if err != nil {
				cleanup()
				return nil, fmt.Errorf("layer %d up transpose upload: %w", i, err)
			}
			model.Layers[i].DownT, err = gpu.DequantTransposeFP8E4M3Linear(model.Layers[i].Down)
			if err != nil {
				cleanup()
				return nil, fmt.Errorf("layer %d down transpose upload: %w", i, err)
			}
		}
	}
	return model, nil
}

func diffusionGemmaDenseTransposeCacheEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_DENSE_TRANSPOSE_CACHE")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func (m *GPUFP8Model) DenseTransposeBytes() int64 {
	if m == nil {
		return 0
	}
	var total int64
	for i := range m.Layers {
		l := &m.Layers[i]
		for _, b := range []*gpu.Buffer{l.GateT, l.UpT, l.DownT} {
			if b != nil {
				total += int64(b.Size)
			}
		}
	}
	return total
}

// FP8GemvQ runs FP8 GEMV for Q projection: out = Q_fp8 × x
func (l *GPUFP8Layer) FP8GemvQ(out, x []float32) error {
	return gpu.GemvFP8E4M3(out, x, l.Q)
}

func (l *GPUFP8Layer) FP8GemvK(out, x []float32) error {
	return gpu.GemvFP8E4M3(out, x, l.K)
}

func (l *GPUFP8Layer) FP8GemvV(out, x []float32) error {
	return gpu.GemvFP8E4M3(out, x, l.V)
}

func (l *GPUFP8Layer) FP8GemvO(out, x []float32) error {
	return gpu.GemvFP8E4M3(out, x, l.O)
}

func (l *GPUFP8Layer) FP8GemvGate(out, x []float32) error {
	return gpu.GemvFP8E4M3(out, x, l.Gate)
}

func (l *GPUFP8Layer) FP8GemvUp(out, x []float32) error {
	return gpu.GemvFP8E4M3(out, x, l.Up)
}

func (l *GPUFP8Layer) FP8GemvDown(out, x []float32) error {
	return gpu.GemvFP8E4M3(out, x, l.Down)
}
