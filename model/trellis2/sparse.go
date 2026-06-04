package trellis2

import (
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// SparseTensor is the first native TRELLIS.2 execution surface in go-pherence.
// It mirrors the upstream SparseTensor fixture shape: one feature row per sparse
// coordinate, with coordinates ordered [batch, x, y, z]. It is deliberately
// backend-neutral; sparse conv/attention kernels can build on this layout.
type SparseTensor struct {
	Coords []int32
	Feats  []float32
	Rows   int
	Dim    int
}

func NewSparseTensor(coords []int32, feats []float32, rows, dim int) (SparseTensor, error) {
	st := SparseTensor{Coords: coords, Feats: feats, Rows: rows, Dim: dim}
	if err := st.Validate(); err != nil {
		return SparseTensor{}, err
	}
	return st, nil
}

func (s SparseTensor) Validate() error {
	if s.Rows < 0 || s.Dim <= 0 {
		return fmt.Errorf("trellis2 sparse tensor: invalid shape rows=%d dim=%d", s.Rows, s.Dim)
	}
	if len(s.Coords) < s.Rows*4 {
		return fmt.Errorf("trellis2 sparse tensor: short coords len=%d want=%d", len(s.Coords), s.Rows*4)
	}
	if len(s.Feats) < s.Rows*s.Dim {
		return fmt.Errorf("trellis2 sparse tensor: short feats len=%d want=%d", len(s.Feats), s.Rows*s.Dim)
	}
	for i := 0; i < s.Rows*4; i++ {
		if s.Coords[i] < 0 {
			return fmt.Errorf("trellis2 sparse tensor: negative coord at %d", i)
		}
	}
	return nil
}

func (s SparseTensor) Coord(row int) ([4]int32, error) {
	if row < 0 || row >= s.Rows {
		return [4]int32{}, fmt.Errorf("trellis2 sparse tensor: row %d out of range [0,%d)", row, s.Rows)
	}
	base := row * 4
	return [4]int32{s.Coords[base], s.Coords[base+1], s.Coords[base+2], s.Coords[base+3]}, nil
}

func (s SparseTensor) FeatureRow(row int) ([]float32, error) {
	if row < 0 || row >= s.Rows {
		return nil, fmt.Errorf("trellis2 sparse tensor: row %d out of range [0,%d)", row, s.Rows)
	}
	base := row * s.Dim
	return s.Feats[base : base+s.Dim], nil
}

// SparseLinearFloat32 applies y = feats @ weight^T + bias over active sparse
// feature rows. weight is row-major [outDim,inDim]. The output preserves coords
// and returns feature rows [rows,outDim].
func SparseLinearFloat32(src SparseTensor, weight, bias []float32, outDim int) (SparseTensor, error) {
	if err := src.Validate(); err != nil {
		return SparseTensor{}, err
	}
	if outDim <= 0 {
		return SparseTensor{}, fmt.Errorf("trellis2 sparse linear: invalid outDim %d", outDim)
	}
	if len(weight) < outDim*src.Dim {
		return SparseTensor{}, fmt.Errorf("trellis2 sparse linear: short weight len=%d want=%d", len(weight), outDim*src.Dim)
	}
	if bias != nil && len(bias) < outDim {
		return SparseTensor{}, fmt.Errorf("trellis2 sparse linear: short bias len=%d want=%d", len(bias), outDim)
	}
	outFeats := make([]float32, src.Rows*outDim)
	if src.Rows > 0 {
		if !simd.SgemmNTTo(outFeats, src.Feats[:src.Rows*src.Dim], weight[:outDim*src.Dim], src.Rows, outDim, src.Dim, 1, src.Dim, src.Dim, outDim) {
			return SparseTensor{}, fmt.Errorf("trellis2 sparse linear: SIMD SGEMM rejected validated tensors")
		}
		if bias != nil {
			if !simd.AddBiasRowsTo(outFeats, bias[:outDim], src.Rows, outDim) {
				return SparseTensor{}, fmt.Errorf("trellis2 sparse linear: SIMD bias rejected validated tensors")
			}
		}
	}
	coords := append([]int32(nil), src.Coords[:src.Rows*4]...)
	return SparseTensor{Coords: coords, Feats: outFeats, Rows: src.Rows, Dim: outDim}, nil
}
