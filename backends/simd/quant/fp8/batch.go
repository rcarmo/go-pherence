package fp8

import (
	"fmt"
	"runtime"
	"sync"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func (l Linear) GemvToDynamicToken(x []float32, out []float32, scratch []float32) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if len(x) < l.InDim || len(out) < l.OutDim || len(scratch) < l.InDim {
		return fmt.Errorf("fp8 dynamic activation buffers x=%d/%d out=%d/%d scratch=%d/%d", len(x), l.InDim, len(out), l.OutDim, len(scratch), l.InDim)
	}
	QuantizeTokenE4M3DequantTo(scratch[:l.InDim], x[:l.InDim])
	return l.GemvTo(scratch[:l.InDim], out[:l.OutDim])
}

func (l Linear) BatchGemvToBufDynamicToken(x []float32, out []float32, batch int, wf32 []float32, xq []float32) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if batch <= 0 {
		return nil
	}
	needX, okX := checkedMulInt(batch, l.InDim)
	needOut, okOut := checkedMulInt(batch, l.OutDim)
	if !okX || !okOut {
		return fmt.Errorf("fp8 dynamic batch size overflow batch=%d in=%d out=%d", batch, l.InDim, l.OutDim)
	}
	if len(x) < needX || len(out) < needOut || len(wf32) < l.InDim || len(xq) < needX {
		return fmt.Errorf("fp8 dynamic batch buffers x=%d/%d out=%d/%d wf32=%d/%d xq=%d/%d", len(x), needX, len(out), needOut, len(wf32), l.InDim, len(xq), needX)
	}
	for b := 0; b < batch; b++ {
		QuantizeTokenE4M3DequantTo(xq[b*l.InDim:(b+1)*l.InDim], x[b*l.InDim:(b+1)*l.InDim])
	}
	return l.BatchGemvToBuf(xq[:needX], out[:needOut], batch, wf32)
}

const (
	batchGemvXTile             = 4
	batchGemvMinParallelOutDim = 1024
	batchGemvMinRowsPerWorker  = 256
	batchGemvMaxWorkers        = 4
)

// BatchGemvTo computes out[b*OutDim + r] = scale[r] * dot(x[b*InDim:], W[r*InDim:]) + bias[r]
// for all batch elements b and output rows r.
//
// For batch > 1, it dequantizes each FP8 weight row to F32 once and reuses that
// decoded row across the whole batch. Large output dimensions may also tile the
// output-row space across a small bounded number of goroutines.
func (l Linear) BatchGemvTo(x []float32, out []float32, batch int) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if batch <= 0 {
		return nil
	}
	needX, okX := checkedMulInt(batch, l.InDim)
	needOut, okOut := checkedMulInt(batch, l.OutDim)
	if !okX || !okOut {
		return fmt.Errorf("fp8 batch size overflow batch=%d in=%d out=%d", batch, l.InDim, l.OutDim)
	}
	if len(x) < needX || len(out) < needOut {
		return fmt.Errorf("fp8 batch buffers x=%d/%d out=%d/%d", len(x), needX, len(out), needOut)
	}

	if batch == 1 {
		batchGemvGather1(l, x[:l.InDim], out[:l.OutDim])
		return nil
	}

	workers := batchGemvParallelWorkers(l.OutDim)
	if workers <= 1 {
		wf32 := make([]float32, l.InDim)
		batchGemvDequantOnce(l, x, out, batch, 0, l.OutDim, wf32)
		return nil
	}

	wf32 := make([]float32, workers*l.InDim)
	batchGemvDequantOnceParallel(l, x, out, batch, workers, wf32)
	return nil
}

// BatchGemvToBuf is like BatchGemvTo but accepts a pre-allocated scratch buffer
// for the dequantized weight row (length >= InDim). This avoids allocation
// overhead when called repeatedly in a hot loop and preserves the single-row
// scratch contract for callers that rely on zero-allocation steady-state use.
func (l Linear) BatchGemvToBuf(x []float32, out []float32, batch int, wf32 []float32) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if batch <= 0 {
		return nil
	}
	needX, okX := checkedMulInt(batch, l.InDim)
	needOut, okOut := checkedMulInt(batch, l.OutDim)
	if !okX || !okOut {
		return fmt.Errorf("fp8 batch size overflow batch=%d in=%d out=%d", batch, l.InDim, l.OutDim)
	}
	if len(x) < needX || len(out) < needOut || len(wf32) < l.InDim {
		return fmt.Errorf("fp8 batch buffers x=%d/%d out=%d/%d wf32=%d/%d", len(x), needX, len(out), needOut, len(wf32), l.InDim)
	}

	if batch == 1 {
		batchGemvGather1(l, x[:l.InDim], out[:l.OutDim])
		return nil
	}

	batchGemvDequantOnce(l, x, out, batch, 0, l.OutDim, wf32)
	return nil
}

func batchGemvParallelWorkers(outDim int) int {
	if outDim < batchGemvMinParallelOutDim {
		return 1
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > batchGemvMaxWorkers {
		workers = batchGemvMaxWorkers
	}
	maxByRows := outDim / batchGemvMinRowsPerWorker
	if workers > maxByRows {
		workers = maxByRows
	}
	if workers < 2 {
		return 1
	}
	return workers
}

func batchGemvGather1(l Linear, xRow []float32, oRow []float32) {
	for r := 0; r < l.OutDim; r++ {
		scale := l.scaleForRow(r)
		base := r * l.InDim
		acc := dotE4M3(xRow, l.Weight[base:base+l.InDim])
		oRow[r] = acc * scale
		if l.Bias != nil {
			oRow[r] += l.Bias[r]
		}
	}
}

func batchGemvDequantOnceParallel(l Linear, x []float32, out []float32, batch, workers int, wf32 []float32) {
	if workers <= 1 {
		batchGemvDequantOnce(l, x, out, batch, 0, l.OutDim, wf32)
		return
	}
	chunk := (l.OutDim + workers - 1) / workers
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		start := worker * chunk
		end := start + chunk
		if end > l.OutDim {
			end = l.OutDim
		}
		if start >= end {
			break
		}
		scratch := wf32[worker*l.InDim : (worker+1)*l.InDim]
		wg.Add(1)
		go func(rs, re int, scratch []float32) {
			defer wg.Done()
			batchGemvDequantOnce(l, x, out, batch, rs, re, scratch)
		}(start, end, scratch)
	}
	wg.Wait()
}

// batchGemvDequantOnce processes output rows [rStart, rEnd) for all batch elements.
// Each weight row is dequantized to F32 once, then the same decoded row is reused
// across up to four activation rows at a time to share weight-row loads.
func batchGemvDequantOnce(l Linear, x []float32, out []float32, batch, rStart, rEnd int, wf32 []float32) {
	for r := rStart; r < rEnd; r++ {
		scale := l.scaleForRow(r)
		wRow := l.Weight[r*l.InDim : (r+1)*l.InDim]
		for j := 0; j < l.InDim; j++ {
			wf32[j] = e4m3LUT[wRow[j]] * scale
		}

		bias := float32(0)
		if l.Bias != nil {
			bias = l.Bias[r]
		}

		b := 0
		for ; b+batchGemvXTile <= batch; b += batchGemvXTile {
			xRows := x[b*l.InDim:]
			d0, d1, d2, d3, ok := simd.Sdotx4(wf32[:l.InDim], xRows, l.InDim)
			if !ok {
				break
			}
			out[(b+0)*l.OutDim+r] = d0 + bias
			out[(b+1)*l.OutDim+r] = d1 + bias
			out[(b+2)*l.OutDim+r] = d2 + bias
			out[(b+3)*l.OutDim+r] = d3 + bias
		}
		for ; b < batch; b++ {
			xRow := x[b*l.InDim : (b+1)*l.InDim]
			out[b*l.OutDim+r] = simd.Sdot(wf32[:l.InDim], xRow) + bias
		}
	}
}
