package nvidia

import (
	"fmt"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// BatchedFP8QKV runs Q, K, V FP8 GEMM projections sharing a single GPU
// upload of the hidden activations. When batch > 1 and SGEMM is available,
// it uses dequant + SGEMM which is dramatically faster than the per-element
// FP8 GEMM kernel for large batches.
//
// All weight matrices must have the same InDim (hidden size).
// If wV is nil, vOut receives a copy of kOut.
func BatchedFP8QKV(qOut, kOut, vOut, hidden []float32, batch int, wQ, wK, wV *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(wQ) || !validGPUFP8E4M3Linear(wK) || (wV != nil && !validGPUFP8E4M3Linear(wV)) {
		return fmt.Errorf("invalid FP8 QKV weight matrices")
	}
	if batch <= 0 {
		return fmt.Errorf("invalid FP8 QKV batch=%d", batch)
	}
	if wQ.InDim != wK.InDim {
		return fmt.Errorf("FP8 QKV InDim mismatch Q=%d K=%d", wQ.InDim, wK.InDim)
	}
	if wV != nil && wV.InDim != wQ.InDim {
		return fmt.Errorf("FP8 QKV InDim mismatch V=%d Q=%d", wV.InDim, wQ.InDim)
	}

	inLen, okIn := checked.MulInt(batch, wQ.InDim)
	qLen, okQ := checked.MulInt(batch, wQ.OutDim)
	kLen, okK := checked.MulInt(batch, wK.OutDim)
	vLen := kLen
	okV := true
	if wV != nil {
		vLen, okV = checked.MulInt(batch, wV.OutDim)
	}
	if !okIn || !okQ || !okK || !okV {
		return fmt.Errorf("FP8 QKV buffer size overflow batch=%d Q=[%d,%d] K=[%d,%d]", batch, wQ.OutDim, wQ.InDim, wK.OutDim, wK.InDim)
	}

	if len(hidden) < inLen || len(qOut) < qLen || len(kOut) < kLen || len(vOut) < vLen {
		return fmt.Errorf("FP8 QKV buffer size mismatch hidden=%d/%d q=%d/%d k=%d/%d v=%d/%d", len(hidden), inLen, len(qOut), qLen, len(kOut), kLen, len(vOut), vLen)
	}

	// For batch > 1, prefer dequant+SGEMM (much faster than per-element FP8 GEMM)
	if batch > 1 && SgemmReady() && !wQ.HasBias && !wK.HasBias && (wV == nil || !wV.HasBias) {
		return batchedFP8QKVViaSgemm(qOut, kOut, vOut, hidden, batch, wQ, wK, wV)
	}

	// Fallback: individual GEMMs (still batched for transfer)
	return batchedFP8QKVDirect(qOut, kOut, vOut, hidden, batch, wQ, wK, wV)
}

// batchedFP8QKVViaSgemm dequantizes FP8→F32 + transposes each weight on GPU,
// then runs SGEMM for each projection, sharing one upload of hidden.
func batchedFP8QKVViaSgemm(qOut, kOut, vOut, hidden []float32, batch int, wQ, wK, wV *GPUFP8E4M3Linear) error {
	inLen := batch * wQ.InDim
	qLen := batch * wQ.OutDim
	kLen := batch * wK.OutDim

	// Upload hidden activations once
	xBuf, err := Malloc(inLen)
	if err != nil {
		return fmt.Errorf("alloc FP8 QKV SGEMM input: %w", err)
	}
	defer xBuf.Free()
	if err := xBuf.Upload(hidden[:inLen]); err != nil {
		return fmt.Errorf("upload FP8 QKV SGEMM input: %w", err)
	}

	// Helper: dequant weight, SGEMM, download result
	runProj := func(name string, out []float32, outLen int, w *GPUFP8E4M3Linear) error {
		// Dequant + transpose weight on GPU: FP8[OutDim, InDim] → F32[InDim, OutDim]
		wtBuf, err := Malloc(w.InDim * w.OutDim)
		if err != nil {
			return fmt.Errorf("alloc FP8 %s SGEMM weight: %w", name, err)
		}
		defer wtBuf.Free()
		if err := dequantTransposeFP8E4M3(wtBuf, w); err != nil {
			return fmt.Errorf("dequant FP8 %s weight: %w", name, err)
		}

		// Output buffer
		outBuf, err := Malloc(outLen)
		if err != nil {
			return fmt.Errorf("alloc FP8 %s SGEMM output: %w", name, err)
		}
		defer outBuf.Free()

		// SGEMM: C[batch, OutDim] = A[batch, InDim] × B[InDim, OutDim]
		if err := Sgemm(batch, w.OutDim, w.InDim, 1, xBuf, wtBuf, outBuf); err != nil {
			return fmt.Errorf("FP8 %s SGEMM: %w", name, err)
		}

		// Download result
		if err := outBuf.Download(out[:outLen]); err != nil {
			return fmt.Errorf("download FP8 %s: %w", name, err)
		}
		return nil
	}

	// Q projection
	if err := runProj("Q", qOut, qLen, wQ); err != nil {
		return err
	}

	// K projection
	if err := runProj("K", kOut, kLen, wK); err != nil {
		return err
	}

	// V projection
	if wV != nil {
		vLen := batch * wV.OutDim
		if err := runProj("V", vOut, vLen, wV); err != nil {
			return err
		}
	} else {
		copy(vOut[:kLen], kOut[:kLen])
	}

	return nil
}

// batchedFP8QKVDirect uses the native FP8 GEMM kernel, sharing one input upload.
func batchedFP8QKVDirect(qOut, kOut, vOut, hidden []float32, batch int, wQ, wK, wV *GPUFP8E4M3Linear) error {
	inLen := batch * wQ.InDim
	qLen := batch * wQ.OutDim
	kLen := batch * wK.OutDim

	// Upload hidden once
	xBuf, err := Malloc(inLen)
	if err != nil {
		return fmt.Errorf("alloc FP8 QKV input: %w", err)
	}
	defer xBuf.Free()
	if err := xBuf.Upload(hidden[:inLen]); err != nil {
		return fmt.Errorf("upload FP8 QKV input: %w", err)
	}

	// Allocate output buffer — reuse the largest for all 3
	maxOut := qLen
	if kLen > maxOut {
		maxOut = kLen
	}
	vLen := kLen
	if wV != nil {
		vLen = batch * wV.OutDim
		if vLen > maxOut {
			maxOut = vLen
		}
	}
	outBuf, err := Malloc(maxOut)
	if err != nil {
		return fmt.Errorf("alloc FP8 QKV output: %w", err)
	}
	defer outBuf.Free()

	// Q
	if err := GemmFP8E4M3Buffer(outBuf, xBuf, batch, wQ); err != nil {
		return fmt.Errorf("FP8 Q GEMM: %w", err)
	}
	if err := outBuf.Download(qOut[:qLen]); err != nil {
		return fmt.Errorf("download FP8 Q: %w", err)
	}

	// K
	if err := GemmFP8E4M3Buffer(outBuf, xBuf, batch, wK); err != nil {
		return fmt.Errorf("FP8 K GEMM: %w", err)
	}
	if err := outBuf.Download(kOut[:kLen]); err != nil {
		return fmt.Errorf("download FP8 K: %w", err)
	}

	// V
	if wV != nil {
		if err := GemmFP8E4M3Buffer(outBuf, xBuf, batch, wV); err != nil {
			return fmt.Errorf("FP8 V GEMM: %w", err)
		}
		if err := outBuf.Download(vOut[:vLen]); err != nil {
			return fmt.Errorf("download FP8 V: %w", err)
		}
	} else {
		copy(vOut[:kLen], kOut[:kLen])
	}

	return nil
}

// BatchedFP8OProj runs the O projection FP8 GEMM. For batch > 1 it uses
// dequant+SGEMM for better throughput.
func BatchedFP8OProj(out, attn []float32, batch int, wO *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(wO) {
		return fmt.Errorf("invalid FP8 O projection weight")
	}
	if batch <= 0 {
		return fmt.Errorf("invalid FP8 O projection batch=%d", batch)
	}
	inLen, okIn := checked.MulInt(batch, wO.InDim)
	outLen, okOut := checked.MulInt(batch, wO.OutDim)
	if !okIn || !okOut || len(attn) < inLen || len(out) < outLen {
		return fmt.Errorf("FP8 O projection buffer mismatch attn=%d/%d out=%d/%d", len(attn), inLen, len(out), outLen)
	}
	if batch > 1 && SgemmReady() && !wO.HasBias {
		return gemmFP8E4M3ViaSgemm(out, attn, batch, wO)
	}
	return GemmFP8E4M3(out, attn, batch, wO)
}
