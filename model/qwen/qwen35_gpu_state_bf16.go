package qwen

import (
	"fmt"
	"math"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// Qwen35GPUForwardStateBF16 stores a Qwen forward state in BF16-packed uint16
// form on the GPU. It is a compressed hot-tier cache format; it is not consumed
// by inference directly yet.
type Qwen35GPUForwardStateBF16 struct {
	Pos         int
	FullK       []*nvidia.Buffer
	FullV       []*nvidia.Buffer
	LinearConv  []*nvidia.Buffer
	LinearSSM   []*nvidia.Buffer
	LinearPos   []int
	LengthsK    []int
	LengthsV    []int
	LengthsConv []int
	LengthsSSM  []int
	Bytes       int64
}

func f32ToBF16Bits(x float32) uint16 {
	bits := math.Float32bits(x)
	return uint16((bits + 0x8000) >> 16)
}

func bf16BitsToF32(x uint16) float32 { return math.Float32frombits(uint32(x) << 16) }

func qwen35UploadBF16Buffer(x []float32) (*nvidia.Buffer, int64, error) {
	if len(x) == 0 {
		return nil, 0, nil
	}
	packed := make([]byte, len(x)*2)
	for i, v := range x {
		b := f32ToBF16Bits(v)
		packed[i*2] = byte(b)
		packed[i*2+1] = byte(b >> 8)
	}
	slots := (len(packed) + 3) / 4
	b, err := nvidia.Malloc(slots)
	if err != nil {
		return nil, 0, err
	}
	if err := b.UploadBytes(packed); err != nil {
		b.Free()
		return nil, 0, err
	}
	return b, int64(len(packed)), nil
}

func qwen35DownloadBF16Buffer(b *nvidia.Buffer, length int) ([]float32, error) {
	if b == nil || length == 0 {
		return nil, nil
	}
	if length < 0 {
		return nil, fmt.Errorf("negative BF16 length %d", length)
	}
	raw := make([]byte, length*2)
	if len(raw) > b.Size {
		return nil, fmt.Errorf("BF16 download bytes=%d exceeds buffer size=%d", len(raw), b.Size)
	}
	if err := b.DownloadBytes(raw); err != nil {
		return nil, err
	}
	out := make([]float32, length)
	for i := range out {
		bits := uint16(raw[i*2]) | uint16(raw[i*2+1])<<8
		out[i] = bf16BitsToF32(bits)
	}
	return out, nil
}

func UploadQwen35ForwardStateGPUBF16(s Qwen35BaseForwardState) (*Qwen35GPUForwardStateBF16, error) {
	if !nvidia.SgemmReady() {
		return nil, fmt.Errorf("GPU not available")
	}
	g := &Qwen35GPUForwardStateBF16{Pos: s.Pos, FullK: make([]*nvidia.Buffer, len(s.FullK)), FullV: make([]*nvidia.Buffer, len(s.FullV)), LinearConv: make([]*nvidia.Buffer, len(s.Linear)), LinearSSM: make([]*nvidia.Buffer, len(s.Linear)), LinearPos: make([]int, len(s.Linear)), LengthsK: make([]int, len(s.FullK)), LengthsV: make([]int, len(s.FullV)), LengthsConv: make([]int, len(s.Linear)), LengthsSSM: make([]int, len(s.Linear))}
	cleanup := true
	defer func() {
		if cleanup {
			g.Free()
		}
	}()
	for i, row := range s.FullK {
		g.LengthsK[i] = len(row)
		b, bytes, err := qwen35UploadBF16Buffer(row)
		if err != nil {
			return nil, err
		}
		g.FullK[i] = b
		g.Bytes += bytes
	}
	for i, row := range s.FullV {
		g.LengthsV[i] = len(row)
		b, bytes, err := qwen35UploadBF16Buffer(row)
		if err != nil {
			return nil, err
		}
		g.FullV[i] = b
		g.Bytes += bytes
	}
	for i, lin := range s.Linear {
		g.LinearPos[i] = lin.Pos
		g.LengthsConv[i] = len(lin.Conv)
		b, bytes, err := qwen35UploadBF16Buffer(lin.Conv)
		if err != nil {
			return nil, err
		}
		g.LinearConv[i] = b
		g.Bytes += bytes
		g.LengthsSSM[i] = len(lin.SSM)
		b, bytes, err = qwen35UploadBF16Buffer(lin.SSM)
		if err != nil {
			return nil, err
		}
		g.LinearSSM[i] = b
		g.Bytes += bytes
	}
	cleanup = false
	return g, nil
}

func DownloadQwen35ForwardStateGPUBF16(g *Qwen35GPUForwardStateBF16) (Qwen35BaseForwardState, error) {
	if g == nil {
		return Qwen35BaseForwardState{}, fmt.Errorf("nil Qwen BF16 GPU forward state")
	}
	out := Qwen35BaseForwardState{Pos: g.Pos, FullK: make([][]float32, len(g.FullK)), FullV: make([][]float32, len(g.FullV)), Linear: make([]Qwen35LinearAttentionState, len(g.LinearConv))}
	for i, b := range g.FullK {
		row, err := qwen35DownloadBF16Buffer(b, g.LengthsK[i])
		if err != nil {
			return Qwen35BaseForwardState{}, err
		}
		out.FullK[i] = row
	}
	for i, b := range g.FullV {
		row, err := qwen35DownloadBF16Buffer(b, g.LengthsV[i])
		if err != nil {
			return Qwen35BaseForwardState{}, err
		}
		out.FullV[i] = row
	}
	for i := range out.Linear {
		if i < len(g.LinearPos) {
			out.Linear[i].Pos = g.LinearPos[i]
		}
		row, err := qwen35DownloadBF16Buffer(g.LinearConv[i], g.LengthsConv[i])
		if err != nil {
			return Qwen35BaseForwardState{}, err
		}
		out.Linear[i].Conv = row
		row, err = qwen35DownloadBF16Buffer(g.LinearSSM[i], g.LengthsSSM[i])
		if err != nil {
			return Qwen35BaseForwardState{}, err
		}
		out.Linear[i].SSM = row
	}
	return out, nil
}

func (g *Qwen35GPUForwardStateBF16) Free() {
	if g == nil {
		return
	}
	for _, b := range append(append(append(g.FullK, g.FullV...), g.LinearConv...), g.LinearSSM...) {
		if b != nil {
			b.Free()
		}
	}
	g.FullK, g.FullV, g.LinearConv, g.LinearSSM = nil, nil, nil, nil
	g.Bytes = 0
}
