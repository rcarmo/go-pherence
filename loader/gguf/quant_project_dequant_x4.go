package gguf

import (
	"fmt"
	"runtime"
	"sync"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

const (
	quantProjectXTile             = 4
	quantProjectMinParallelOutDim = 1024
	quantProjectMinRowsPerWorker  = 256
	quantProjectMaxWorkers        = 4
)

type quantProjectDequantOneWeightRow func(dst []float32, raw []byte, n int) error

func (m *QuantMatrix) projectBatchQ2KTo(dst, x []float32, batch int) error {
	return m.projectBatchDequantSdotx4To(dst, x, batch, qkK, dequantRowQ2KTo)
}

func (m *QuantMatrix) projectBatchQ3KTo(dst, x []float32, batch int) error {
	return m.projectBatchDequantSdotx4To(dst, x, batch, qkK, dequantRowQ3KTo)
}

// Shared dequant-once+x4 paths for formats where benchmarked direct quant dots
// are slower across prompt rows.
func (m *QuantMatrix) projectBatchQ5_0DequantSdotx4To(dst, x []float32, batch int) error {
	return m.projectBatchDequantSdotx4To(dst, x, batch, qk8_0, dequantRowQ5_0To)
}

func (m *QuantMatrix) projectBatchQ8_0DequantSdotx4To(dst, x []float32, batch int) error {
	return m.projectBatchDequantSdotx4To(dst, x, batch, qk8_0, dequantRowQ8_0To)
}

func (m *QuantMatrix) projectBatchQ6KDequantSdotx4To(dst, x []float32, batch int) error {
	return m.projectBatchDequantSdotx4To(dst, x, batch, qkK, dequantRowQ6KTo)
}

// projectBatchDequantSdotx4To dequantizes each weight row once into worker-owned
// scratch, reuses it across up to four activation rows via simd.Sdotx4, and
// finishes any batch tail with scalar dot products. Output rows may be split
// across a small bounded number of goroutines.
func (m *QuantMatrix) projectBatchDequantSdotx4To(dst, x []float32, batch, blockElems int, dequant quantProjectDequantOneWeightRow) error {
	if m.InDim%blockElems != 0 {
		return fmt.Errorf("quant matrix %s: %s in=%d not multiple of %d", m.Name, m.QType, m.InDim, blockElems)
	}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return err
	}
	workers := quantProjectParallelWorkers(m.OutDim)
	if workers <= 1 {
		return m.projectBatchDequantSdotx4Range(dst, x, batch, 0, m.OutDim, rowBytes, dequant, make([]float32, m.InDim))
	}

	scratch := make([]float32, workers*m.InDim)
	chunk := (m.OutDim + workers - 1) / workers
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	for worker := 0; worker < workers; worker++ {
		start := worker * chunk
		end := start + chunk
		if end > m.OutDim {
			end = m.OutDim
		}
		if start >= end {
			break
		}
		wf32 := scratch[worker*m.InDim : (worker+1)*m.InDim]
		wg.Add(1)
		go func(start, end int, wf32 []float32) {
			defer wg.Done()
			if err := m.projectBatchDequantSdotx4Range(dst, x, batch, start, end, rowBytes, dequant, wf32); err != nil {
				errCh <- err
			}
		}(start, end, wf32)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return err
		}
	}
	return nil
}

func (m *QuantMatrix) projectBatchDequantSdotx4Range(dst, x []float32, batch, rowStart, rowEnd, rowBytes int, dequant quantProjectDequantOneWeightRow, wf32 []float32) error {
	rowScratch := wf32[:m.InDim]
	for r := rowStart; r < rowEnd; r++ {
		start := r * rowBytes
		end := start + rowBytes
		if end > len(m.Raw) {
			return fmt.Errorf("quant matrix %s: row %d raw short", m.Name, r)
		}
		if err := dequant(rowScratch, m.Raw[start:end], m.InDim); err != nil {
			return fmt.Errorf("quant matrix %s: row %d: %w", m.Name, r, err)
		}
		b := 0
		for ; b+quantProjectXTile <= batch; b += quantProjectXTile {
			xRows := x[b*m.InDim:]
			d0, d1, d2, d3, ok := simd.Sdotx4(rowScratch, xRows, m.InDim)
			if !ok {
				break
			}
			dst[(b+0)*m.OutDim+r] = d0
			dst[(b+1)*m.OutDim+r] = d1
			dst[(b+2)*m.OutDim+r] = d2
			dst[(b+3)*m.OutDim+r] = d3
		}
		for ; b < batch; b++ {
			dst[b*m.OutDim+r] = dotF32(rowScratch, x[b*m.InDim:(b+1)*m.InDim])
		}
	}
	return nil
}

func quantProjectParallelWorkers(outDim int) int {
	if outDim < quantProjectMinParallelOutDim {
		return 1
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > quantProjectMaxWorkers {
		workers = quantProjectMaxWorkers
	}
	maxByRows := outDim / quantProjectMinRowsPerWorker
	if workers > maxByRows {
		workers = maxByRows
	}
	if workers < 2 {
		return 1
	}
	return workers
}
