package gguf

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
)

var ErrUnsupportedBatchProjection = errors.New("unsupported quant batch projection")

// ProjectBatchF32To projects a row-major batch of F32 activations through the
// quantized matrix into row-major F32 outputs.
func (m *QuantMatrix) ProjectBatchF32To(dst, x []float32, batch int) error {
	if m == nil {
		return fmt.Errorf("nil quant matrix")
	}
	if batch <= 0 {
		return fmt.Errorf("quant matrix %s: invalid batch=%d", m.Name, batch)
	}
	if len(x) < batch*m.InDim || len(dst) < batch*m.OutDim {
		return fmt.Errorf("quant matrix %s: x=%d dst=%d want at least %d/%d", m.Name, len(x), len(dst), batch*m.InDim, batch*m.OutDim)
	}
	switch m.QType {
	case QuantQ4_0:
		return m.projectBatchQ4_0To(dst, x, batch)
	case QuantQ2_K:
		return m.projectBatchQ2KTo(dst, x, batch)
	case QuantQ3_K:
		return m.projectBatchQ3KTo(dst, x, batch)
	case QuantQ4_K:
		return m.projectBatchQ4KTo(dst, x, batch)
	case QuantQ5_0:
		return m.projectBatchQ5_0To(dst, x, batch)
	case QuantQ8_0:
		return m.projectBatchQ8_0To(dst, x, batch)
	case QuantQ6_K:
		return m.projectBatchQ6KTo(dst, x, batch)
	default:
		return fmt.Errorf("quant matrix %s: %w: type %s", m.Name, ErrUnsupportedBatchProjection, m.QType)
	}
}

func (m *QuantMatrix) projectBatchQ4_0To(dst, x []float32, batch int) error {
	// Keep decode and B2/B4 on the established GEMV path. The shared
	// row-parallel traversal is consistently beneficial at B8 across measured
	// E4B shapes, while smaller batches are noisy or regress wide MLP matrices.
	if batch < 8 {
		return fmt.Errorf("quant matrix %s: %w: Q4_0 batch=%d needs B8 shape dispatch", m.Name, ErrUnsupportedBatchProjection, batch)
	}
	if m.InDim%qk8_0 != 0 {
		return fmt.Errorf("quant matrix %s: Q4_0 in=%d not multiple of %d", m.Name, m.InDim, qk8_0)
	}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return err
	}
	q8 := make([][]q8_0Block, batch)
	for pos := 0; pos < batch; pos++ {
		blocks, err := QuantizeQ8_0(x[pos*m.InDim : (pos+1)*m.InDim])
		if err != nil {
			return err
		}
		q8[pos] = blocks
	}
	if !gemvRowsParallel(m.OutDim, rowBytes, func(r int) bool {
		rowRaw := m.Raw[r*rowBytes : (r+1)*rowBytes]
		for pos := 0; pos < batch; pos++ {
			v, err := DotQ4_0Q8_0(rowRaw, q8[pos], m.InDim)
			if err != nil {
				return false
			}
			dst[pos*m.OutDim+r] = v
		}
		return true
	}) {
		return fmt.Errorf("quant matrix %s: projection failed", m.Name)
	}
	return nil
}

func (m *QuantMatrix) projectBatchQ5_0To(dst, x []float32, batch int) error {
	if m.InDim%qk8_0 != 0 {
		return fmt.Errorf("quant matrix %s: Q5_0 in=%d not multiple of %d", m.Name, m.InDim, qk8_0)
	}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return err
	}
	q8 := make([][]q8_0Block, batch)
	for pos := 0; pos < batch; pos++ {
		blocks, err := QuantizeQ8_0(x[pos*m.InDim : (pos+1)*m.InDim])
		if err != nil {
			return err
		}
		q8[pos] = blocks
	}
	if !gemvRowsParallel(m.OutDim, rowBytes, func(r int) bool {
		rowRaw := m.Raw[r*rowBytes : (r+1)*rowBytes]
		for pos := 0; pos < batch; pos++ {
			v, err := DotQ5_0Q8_0(rowRaw, q8[pos], m.InDim)
			if err != nil {
				return false
			}
			dst[pos*m.OutDim+r] = v
		}
		return true
	}) {
		return fmt.Errorf("quant matrix %s: projection failed", m.Name)
	}
	return nil
}

func (m *QuantMatrix) projectBatchQ8_0To(dst, x []float32, batch int) error {
	if m.InDim%qk8_0 != 0 {
		return fmt.Errorf("quant matrix %s: Q8_0 in=%d not multiple of %d", m.Name, m.InDim, qk8_0)
	}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return err
	}
	q8 := make([][]q8_0Block, batch)
	for pos := 0; pos < batch; pos++ {
		blocks, err := QuantizeQ8_0(x[pos*m.InDim : (pos+1)*m.InDim])
		if err != nil {
			return err
		}
		q8[pos] = blocks
	}
	if !gemvRowsParallel(m.OutDim, rowBytes, func(r int) bool {
		rowRaw := m.Raw[r*rowBytes : (r+1)*rowBytes]
		for pos := 0; pos < batch; pos++ {
			v, err := DotQ8_0Q8_0(rowRaw, q8[pos], m.InDim)
			if err != nil {
				return false
			}
			dst[pos*m.OutDim+r] = v
		}
		return true
	}) {
		return fmt.Errorf("quant matrix %s: projection failed", m.Name)
	}
	return nil
}

func (m *QuantMatrix) projectBatchQ6KTo(dst, x []float32, batch int) error {
	if m.InDim%qkK != 0 {
		return fmt.Errorf("quant matrix %s: Q6_K in=%d not multiple of %d", m.Name, m.InDim, qkK)
	}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return err
	}
	q8 := make([][]q8KBlock, batch)
	for pos := 0; pos < batch; pos++ {
		blocks, err := QuantizeQ8K(x[pos*m.InDim : (pos+1)*m.InDim])
		if err != nil {
			return err
		}
		q8[pos] = blocks
	}
	if !gemvRowsParallel(m.OutDim, rowBytes, func(r int) bool {
		rowRaw := m.Raw[r*rowBytes : (r+1)*rowBytes]
		for pos := 0; pos < batch; pos++ {
			v, err := DotQ6KQ8K(rowRaw, q8[pos], m.InDim)
			if err != nil {
				return false
			}
			dst[pos*m.OutDim+r] = v
		}
		return true
	}) {
		return fmt.Errorf("quant matrix %s: projection failed", m.Name)
	}
	return nil
}

func (m *QuantMatrix) projectBatchQ4KTo(dst, x []float32, batch int) error {
	if m.InDim%experimentalQ4KBlockElems != 0 {
		return fmt.Errorf("quant matrix %s: Q4_K in=%d not multiple of %d", m.Name, m.InDim, experimentalQ4KBlockElems)
	}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return err
	}
	fullRowTiles := m.OutDim / experimentalQ4K8x8Rows
	fullAct4Groups := batch / 4

	rowTiles := make([]experimentalQ4K8x8Tile, fullRowTiles)
	for tileIdx := 0; tileIdx < fullRowTiles; tileIdx++ {
		raw := m.Raw[tileIdx*experimentalQ4K8x8Rows*rowBytes : (tileIdx+1)*experimentalQ4K8x8Rows*rowBytes]
		tile, err := newExperimentalQ4K8x8Tile(raw, m.InDim)
		if err != nil {
			return err
		}
		rowTiles[tileIdx] = *tile
	}

	actGroups := make([]experimentalQ8K4x8Group, fullAct4Groups)
	for groupIdx := 0; groupIdx < fullAct4Groups; groupIdx++ {
		groupActs := x[groupIdx*4*m.InDim : (groupIdx+1)*4*m.InDim]
		group, err := newExperimentalQ8K4x8Group(groupActs, m.InDim)
		if err != nil {
			return err
		}
		actGroups[groupIdx] = *group
	}
	if fullAct4Groups >= 2 {
		// Preserve an explicit 8-activation counter for regression tests; the work
		// is executed as reusable 4-row quant groups consumed in consecutive pairs.
		atomic.AddUint64(&experimentalQ4K8x8Stats.quant8x, uint64(fullAct4Groups/2))
	}

	if len(rowTiles) > 0 && len(actGroups) > 0 {
		if err := runExperimentalQ4KTilePairs(rowTiles, actGroups, dst, m.OutDim); err != nil {
			return fmt.Errorf("quant matrix %s: %w", m.Name, err)
		}
	}

	dequantRow := make([]float32, m.InDim)
	for row0 := fullRowTiles * experimentalQ4K8x8Rows; row0 < m.OutDim; row0++ {
		if err := m.DequantRowTo(dequantRow, row0); err != nil {
			return err
		}
		for pos := 0; pos < batch; pos++ {
			dst[pos*m.OutDim+row0] = dotF32(dequantRow, x[pos*m.InDim:(pos+1)*m.InDim])
		}
	}
	for tileIdx := 0; tileIdx < fullRowTiles; tileIdx++ {
		row0 := tileIdx * experimentalQ4K8x8Rows
		for r := 0; r < experimentalQ4K8x8Rows; r++ {
			if err := m.DequantRowTo(dequantRow, row0+r); err != nil {
				return err
			}
			for groupIdx := fullAct4Groups; groupIdx < batch; groupIdx++ {
				act := x[groupIdx*m.InDim : (groupIdx+1)*m.InDim]
				dst[groupIdx*m.OutDim+row0+r] = dotF32(dequantRow, act)
			}
		}
	}
	return nil
}

func runExperimentalQ4KTilePairs(rowTiles []experimentalQ4K8x8Tile, actGroups []experimentalQ8K4x8Group, dst []float32, outDim int) error {
	jobs := len(rowTiles) * len(actGroups)
	if jobs == 0 {
		return nil
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || jobs < 8 {
		var block [experimentalQ4K8x8Rows * 4]float32
		for tileIdx := range rowTiles {
			for groupIdx := range actGroups {
				storeExperimentalQ4KTilePair(dst, outDim, tileIdx, groupIdx, &rowTiles[tileIdx], &actGroups[groupIdx], block[:])
			}
		}
		return nil
	}
	if workers > jobs {
		workers = jobs
	}
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			var block [experimentalQ4K8x8Rows * 4]float32
			for job := worker; job < jobs; job += workers {
				tileIdx := job / len(actGroups)
				groupIdx := job % len(actGroups)
				storeExperimentalQ4KTilePair(dst, outDim, tileIdx, groupIdx, &rowTiles[tileIdx], &actGroups[groupIdx], block[:])
			}
		}(worker)
	}
	wg.Wait()
	return nil
}

func storeExperimentalQ4KTilePair(dst []float32, outDim, tileIdx, groupIdx int, tile *experimentalQ4K8x8Tile, acts *experimentalQ8K4x8Group, block []float32) {
	gemmExperimentalQ4K8x8Q8KTo(block, tile.k, tile.blocks, acts.blocks, 4, experimentalQ4K8x8Rows)
	row0 := tileIdx * experimentalQ4K8x8Rows
	pos0 := groupIdx * 4
	for actRow := 0; actRow < 4; actRow++ {
		base := (pos0 + actRow) * outDim
		for weightRow := 0; weightRow < experimentalQ4K8x8Rows; weightRow++ {
			dst[base+row0+weightRow] = block[actRow*experimentalQ4K8x8Rows+weightRow]
		}
	}
}

func dotF32(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
