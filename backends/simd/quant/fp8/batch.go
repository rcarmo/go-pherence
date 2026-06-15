package fp8

import (
	"fmt"

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

// BatchGemvTo computes out[b*OutDim + r] = scale[r] * dot(x[b*InDim:], W[r*InDim:]) + bias[r]
// for all batch elements b and output rows r.
//
// For batch > 1, uses a "dequant-once" strategy: each FP8 weight row is
// decoded to F32 once, then a fast F32 SIMD dot product (AVX2 VFMADD) is
// used for every batch element. This avoids the expensive VGATHERDPS
// (~20 cyc/8 elems) per batch element, replacing it with VFMADD (~4 cyc/8 elems).
//
// Thread-safety: this function does NOT launch goroutines internally.
// Callers are expected to manage parallelism at a higher level.
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

	// Single-element batch: original gather-dot path (no dequant benefit)
	if batch == 1 {
		xRow := x[:l.InDim]
		oRow := out[:l.OutDim]
		for r := 0; r < l.OutDim; r++ {
			scale := l.scaleForRow(r)
			base := r * l.InDim
			acc := dotE4M3(xRow, l.Weight[base:base+l.InDim])
			oRow[r] = acc * scale
			if l.Bias != nil {
				oRow[r] += l.Bias[r]
			}
		}
		return nil
	}

	// Multi-element batch: dequant each weight row to F32 once, then use fast
	// F32 SIMD dot for all batch elements.
	wf32 := make([]float32, l.InDim)
	batchGemvDequantOnce(l, x, out, batch, 0, l.OutDim, wf32)
	return nil
}

// BatchGemvToBuf is like BatchGemvTo but accepts a pre-allocated scratch buffer
// for the dequantized weight row (length >= InDim). This avoids allocation
// overhead when called repeatedly in a hot loop.
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
		xRow := x[:l.InDim]
		oRow := out[:l.OutDim]
		for r := 0; r < l.OutDim; r++ {
			scale := l.scaleForRow(r)
			base := r * l.InDim
			acc := dotE4M3(xRow, l.Weight[base:base+l.InDim])
			oRow[r] = acc * scale
			if l.Bias != nil {
				oRow[r] += l.Bias[r]
			}
		}
		return nil
	}

	batchGemvDequantOnce(l, x, out, batch, 0, l.OutDim, wf32)
	return nil
}

// batchGemvDequantOnce processes output rows [rStart, rEnd) for all batch elements.
// Each weight row is dequantized to F32 ONCE, then fast F32 dot is used for every batch element.
// wf32 is a caller-provided scratch buffer of length >= InDim.
func batchGemvDequantOnce(l Linear, x []float32, out []float32, batch, rStart, rEnd int, wf32 []float32) {
	for r := rStart; r < rEnd; r++ {
		// Dequant weight row from FP8 E4M3 to F32 (includes scale)
		scale := l.scaleForRow(r)
		wRow := l.Weight[r*l.InDim : (r+1)*l.InDim]
		for j := 0; j < l.InDim; j++ {
			wf32[j] = e4m3LUT[wRow[j]] * scale
		}

		bias := float32(0)
		if l.Bias != nil {
			bias = l.Bias[r]
		}

		// Fast F32 dot product for each batch element (AVX2 VFMADD)
		for b := 0; b < batch; b++ {
			xRow := x[b*l.InDim : (b+1)*l.InDim]
			out[b*l.OutDim+r] = simd.Sdot(wf32[:l.InDim], xRow) + bias
		}
	}
}
