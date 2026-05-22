package qwen

import (
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// Qwen35GPUForwardState is a GPU-resident mirror of Qwen35BaseForwardState.
// It is intended as the L0/hot tier for prompt-state reuse; CPU cache entries
// remain the source of truth until the inference path can consume these buffers
// directly.
type Qwen35GPUForwardState struct {
	Pos int

	FullK []*nvidia.Buffer
	FullV []*nvidia.Buffer

	LinearConv []*nvidia.Buffer
	LinearSSM  []*nvidia.Buffer
	LinearPos  []int

	Bytes int64
}

func qwen35UploadFloatBuffer(x []float32) (*nvidia.Buffer, int64, error) {
	if len(x) == 0 {
		return nil, 0, nil
	}
	b, err := nvidia.Malloc(len(x))
	if err != nil {
		return nil, 0, err
	}
	if err := b.Upload(x); err != nil {
		b.Free()
		return nil, 0, err
	}
	return b, int64(len(x)) * 4, nil
}

func qwen35DownloadFloatBuffer(b *nvidia.Buffer) ([]float32, error) {
	if b == nil {
		return nil, nil
	}
	if b.Size%4 != 0 {
		return nil, fmt.Errorf("Qwen GPU state buffer size %d is not float32-aligned", b.Size)
	}
	out := make([]float32, b.Size/4)
	if err := b.Download(out); err != nil {
		return nil, err
	}
	return out, nil
}

// UploadQwen35ForwardStateGPU uploads a CPU Qwen forward-state snapshot to GPU
// buffers. The returned object owns all buffers and must be freed.
func UploadQwen35ForwardStateGPU(s Qwen35BaseForwardState) (*Qwen35GPUForwardState, error) {
	if !nvidia.SgemmReady() {
		return nil, fmt.Errorf("GPU not available")
	}
	g := &Qwen35GPUForwardState{Pos: s.Pos, FullK: make([]*nvidia.Buffer, len(s.FullK)), FullV: make([]*nvidia.Buffer, len(s.FullV)), LinearConv: make([]*nvidia.Buffer, len(s.Linear)), LinearSSM: make([]*nvidia.Buffer, len(s.Linear)), LinearPos: make([]int, len(s.Linear))}
	cleanup := true
	defer func() {
		if cleanup {
			g.Free()
		}
	}()
	for i, row := range s.FullK {
		b, bytes, err := qwen35UploadFloatBuffer(row)
		if err != nil {
			return nil, fmt.Errorf("upload full K layer %d: %w", i, err)
		}
		g.FullK[i] = b
		g.Bytes += bytes
	}
	for i, row := range s.FullV {
		b, bytes, err := qwen35UploadFloatBuffer(row)
		if err != nil {
			return nil, fmt.Errorf("upload full V layer %d: %w", i, err)
		}
		g.FullV[i] = b
		g.Bytes += bytes
	}
	for i, lin := range s.Linear {
		g.LinearPos[i] = lin.Pos
		b, bytes, err := qwen35UploadFloatBuffer(lin.Conv)
		if err != nil {
			return nil, fmt.Errorf("upload linear conv layer %d: %w", i, err)
		}
		g.LinearConv[i] = b
		g.Bytes += bytes
		b, bytes, err = qwen35UploadFloatBuffer(lin.SSM)
		if err != nil {
			return nil, fmt.Errorf("upload linear ssm layer %d: %w", i, err)
		}
		g.LinearSSM[i] = b
		g.Bytes += bytes
	}
	cleanup = false
	return g, nil
}

// DownloadQwen35ForwardStateGPU downloads a GPU state mirror back to CPU slices.
func DownloadQwen35ForwardStateGPU(g *Qwen35GPUForwardState) (Qwen35BaseForwardState, error) {
	if g == nil {
		return Qwen35BaseForwardState{}, fmt.Errorf("nil Qwen GPU forward state")
	}
	out := Qwen35BaseForwardState{Pos: g.Pos, FullK: make([][]float32, len(g.FullK)), FullV: make([][]float32, len(g.FullV)), Linear: make([]Qwen35LinearAttentionState, len(g.LinearConv))}
	for i, b := range g.FullK {
		row, err := qwen35DownloadFloatBuffer(b)
		if err != nil {
			return Qwen35BaseForwardState{}, fmt.Errorf("download full K layer %d: %w", i, err)
		}
		out.FullK[i] = row
	}
	for i, b := range g.FullV {
		row, err := qwen35DownloadFloatBuffer(b)
		if err != nil {
			return Qwen35BaseForwardState{}, fmt.Errorf("download full V layer %d: %w", i, err)
		}
		out.FullV[i] = row
	}
	for i := range out.Linear {
		if i < len(g.LinearPos) {
			out.Linear[i].Pos = g.LinearPos[i]
		}
		row, err := qwen35DownloadFloatBuffer(g.LinearConv[i])
		if err != nil {
			return Qwen35BaseForwardState{}, fmt.Errorf("download linear conv layer %d: %w", i, err)
		}
		out.Linear[i].Conv = row
		if i < len(g.LinearSSM) {
			row, err = qwen35DownloadFloatBuffer(g.LinearSSM[i])
			if err != nil {
				return Qwen35BaseForwardState{}, fmt.Errorf("download linear ssm layer %d: %w", i, err)
			}
			out.Linear[i].SSM = row
		}
	}
	return out, nil
}

// Free releases all GPU buffers owned by the state. It is idempotent.
func (g *Qwen35GPUForwardState) Free() {
	if g == nil {
		return
	}
	for _, b := range g.FullK {
		if b != nil {
			b.Free()
		}
	}
	for _, b := range g.FullV {
		if b != nil {
			b.Free()
		}
	}
	for _, b := range g.LinearConv {
		if b != nil {
			b.Free()
		}
	}
	for _, b := range g.LinearSSM {
		if b != nil {
			b.Free()
		}
	}
	g.FullK, g.FullV, g.LinearConv, g.LinearSSM = nil, nil, nil, nil
	g.LinearPos = nil
	g.Bytes = 0
}
