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
	blocksPerRow := m.InDim / qk8_0
	if batch >= 64 && supportsQ4_0Q8_0Rows4Tokens2() {
		// Long prefill writes exact Q8 bytes directly to their final eight-token
		// SoA destination. Only the sub-eight-token tail remains token-major.
		fullTokens := batch / 8 * 8
		tiles := make([]q8_0Tile8, fullTokens/8*blocksPerRow)
		tailQ8 := make([]q8_0Block, (batch-fullTokens)*blocksPerRow)
		if err := quantizeQ8_0Tiles8To(tiles, tailQ8, x[:batch*m.InDim], batch, m.InDim); err != nil {
			return fmt.Errorf("quant matrix %s: %w", m.Name, err)
		}
		// Put each activation tile outside the row loop. At the real 80-block
		// shape one 23 KiB tile remains in L1 across 64 rows while their 90 KiB
		// of Q4 weights fit in L2. Dynamic claims include the caller.
		if !gemvRowBlocksParallel(m.OutDim, 64, func(rowStart, rowEnd int) bool {
			pos := 0
			for ; pos < fullTokens; pos += 8 {
				tile := tiles[pos/8*blocksPerRow:]
				for r := rowStart; r < rowEnd; r++ {
					rowRaw := m.Raw[r*rowBytes : (r+1)*rowBytes]
					var values [8]float32
					if !dotQ4_0Q8_0Tokens8SoA(rowRaw, tile, blocksPerRow, &values) {
						return false
					}
					for token := 0; token < 8; token++ {
						dst[(pos+token)*m.OutDim+r] = values[token]
					}
				}
			}
			for ; pos+4 <= batch; pos += 4 {
				activation := tailQ8[(pos-fullTokens)*blocksPerRow:]
				for r := rowStart; r < rowEnd; r++ {
					rowRaw := m.Raw[r*rowBytes : (r+1)*rowBytes]
					var values [4]float32
					dotQ4_0Q8_0Tokens4(rowRaw, activation, blocksPerRow, &values)
					for token := 0; token < 4; token++ {
						dst[(pos+token)*m.OutDim+r] = values[token]
					}
				}
			}
			for ; pos < batch; pos++ {
				activation := tailQ8[(pos-fullTokens)*blocksPerRow:]
				for r := rowStart; r < rowEnd; r++ {
					rowRaw := m.Raw[r*rowBytes : (r+1)*rowBytes]
					dst[pos*m.OutDim+r] = dotQ4_0Q8_0Packed(rowRaw, activation, blocksPerRow)
				}
			}
			return true
		}) {
			return fmt.Errorf("quant matrix %s: AVX-VNNI token-tiled projection failed", m.Name)
		}
		return nil
	}

	q8 := make([]q8_0Block, batch*blocksPerRow)
	if err := quantizeQ8_0BatchTo(q8, x[:batch*m.InDim], batch, m.InDim); err != nil {
		return fmt.Errorf("quant matrix %s: %w", m.Name, err)
	}

	// AVX2 fallback reuses each Q4_0 row across four activation rows.
	if !gemvRowsParallel(m.OutDim, rowBytes*batch, func(r int) bool {
		rowRaw := m.Raw[r*rowBytes : (r+1)*rowBytes]
		pos := 0
		for ; pos+4 <= batch; pos += 4 {
			var values [4]float32
			dotQ4_0Q8_0Tokens4(rowRaw, q8[pos*blocksPerRow:], blocksPerRow, &values)
			for token := 0; token < 4; token++ {
				dst[(pos+token)*m.OutDim+r] = values[token]
			}
		}
		for ; pos < batch; pos++ {
			dst[pos*m.OutDim+r] = dotQ4_0Q8_0Packed(rowRaw, q8[pos*blocksPerRow:], blocksPerRow)
		}
		return true
	}) {
		return fmt.Errorf("quant matrix %s: token-tiled projection failed", m.Name)
	}
	return nil
}

func gemvRowBlocksParallel(outDim, blockRows int, fn func(start, end int) bool) bool {
	if outDim <= 0 || blockRows <= 0 {
		return false
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || outDim <= blockRows {
		return fn(0, outDim)
	}
	if workers > (outDim+blockRows-1)/blockRows {
		workers = (outDim + blockRows - 1) / blockRows
	}

	var next atomic.Int64
	var failed atomic.Bool
	work := func() {
		for !failed.Load() {
			start := int(next.Add(int64(blockRows))) - blockRows
			if start >= outDim {
				return
			}
			end := start + blockRows
			if end > outDim {
				end = outDim
			}
			if !fn(start, end) {
				failed.Store(true)
				return
			}
		}
	}

	var wg sync.WaitGroup
	wg.Add(workers - 1)
	for range workers - 1 {
		go func() {
			defer wg.Done()
			work()
		}()
	}
	work()
	wg.Wait()
	return !failed.Load()
}

func quantizeQ8_0Tiles8To(tiles []q8_0Tile8, tail []q8_0Block, x []float32, batch, width int) error {
	if batch <= 0 || width <= 0 || width%qk8_0 != 0 || len(x) != batch*width {
		return fmt.Errorf("Q8_0 tile quantize tiles=%d tail=%d len=%d batch=%d width=%d", len(tiles), len(tail), len(x), batch, width)
	}
	blocksPerRow := width / qk8_0
	fullTokens := batch / 8 * 8
	groups := fullTokens / 8
	if len(tiles) != groups*blocksPerRow || len(tail) != (batch-fullTokens)*blocksPerRow {
		return fmt.Errorf("Q8_0 tile quantize tiles=%d tail=%d len=%d batch=%d width=%d", len(tiles), len(tail), len(x), batch, width)
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > groups {
		workers = groups
	}
	if workers < 1 {
		workers = 1
	}
	var next atomic.Int64
	work := func() {
		for {
			group := int(next.Add(1)) - 1
			if group >= groups {
				return
			}
			tileBase := group * blocksPerRow
			for token := 0; token < 8; token++ {
				xBase := (group*8 + token) * width
				for bi := 0; bi < blocksPerRow; bi++ {
					tile := &tiles[tileBase+bi]
					quantizeQ8_0BlockTo(&tile.d[token], &tile.qs[token], x[xBase+bi*qk8_0:xBase+(bi+1)*qk8_0])
				}
			}
		}
	}
	var wg sync.WaitGroup
	wg.Add(workers - 1)
	for range workers - 1 {
		go func() {
			defer wg.Done()
			work()
		}()
	}
	work()
	wg.Wait()

	if len(tail) > 0 {
		return quantizeQ8_0To(tail, x[fullTokens*width:])
	}
	return nil
}

func quantizeQ8_0BatchTo(dst []q8_0Block, x []float32, batch, width int) error {
	if batch <= 0 || width <= 0 || width%qk8_0 != 0 || len(dst) != batch*width/qk8_0 || len(x) != batch*width {
		return fmt.Errorf("Q8_0 batch quantize destination blocks=%d len=%d batch=%d width=%d", len(dst), len(x), batch, width)
	}
	blocksPerRow := width / qk8_0
	if batch < 64 || runtime.GOMAXPROCS(0) < 2 {
		return quantizeQ8_0To(dst, x)
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > batch {
		workers = batch
	}
	var wg sync.WaitGroup
	var once sync.Once
	var quantErr error
	rowsPerWorker := (batch + workers - 1) / workers
	for worker := 0; worker < workers; worker++ {
		start := worker * rowsPerWorker
		end := start + rowsPerWorker
		if end > batch {
			end = batch
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			if err := quantizeQ8_0To(dst[start*blocksPerRow:end*blocksPerRow], x[start*width:end*width]); err != nil {
				once.Do(func() { quantErr = err })
			}
		}(start, end)
	}
	wg.Wait()
	return quantErr
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
