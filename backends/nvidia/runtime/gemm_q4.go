package nvidia

// Batched Q4 GEMM: process multiple tokens in one kernel launch.
//
// Standard GEMV: [1 × inDim] × [inDim × outDim] → [1 × outDim]
// Batched GEMM:  [B × inDim] × [inDim × outDim] → [B × outDim]
//
// For prefill, B = prompt length. Each row reads the SAME weight matrix,
// so weight data is fetched once from VRAM and reused across all B rows.
// This turns memory-bound GEMV into compute-bound GEMM.
//
// The kernel dequantizes INT4 weights on-the-fly (same as gemv_q4sym)
// but accumulates B dot products per weight column.

import (
	"github.com/rcarmo/go-pherence/internal/checked"
	"unsafe"

	simdq4 "github.com/rcarmo/go-pherence/backends/simd/quant/q4"
)

// GemmQ4 performs batched matrix multiply: out[B×outDim] = input[B×inDim] × W_q4[inDim×outDim]
// where W is INT4 quantized with group scales.
func GemmQ4(out, input *DevBuf, w *GPUQuantWeight, B int) {
	if !validGPUQuantWeight(w) || input == nil || out == nil || B <= 0 {
		return
	}
	inLen, okIn := checked.MulInt(B, w.InDim)
	outLen, okOut := checked.MulInt(B, w.OutDim)
	if !okIn || !okOut || input.n < inLen || out.n < outLen {
		return
	}
	if !q4Ready || fnGemmQ4 == 0 || !fitsUint32(B) || !fitsUint32(w.InDim) || !fitsUint32(w.OutDim) || !fitsUint32(w.Groups) || !tryGPU(input, out) {
		gemmQ4CPU(out, input, w, B)
		return
	}
	EnsureContext()

	batchSize := uint32(B)
	inDim := uint32(w.InDim)
	outDim := uint32(w.OutDim)
	groups := uint32(w.Groups)

	// Grid: one block per (4 output columns, batch row) tile
	// Block: 128 threads = 4 warps, one warp per output column
	gridX, okGrid := grid1DFor(w.OutDim, 4)
	if !okGrid {
		gemmQ4CPU(out, input, w, B)
		return
	}
	gridY := batchSize

	if err := LaunchKernel(fnGemmQ4, gridX, gridY, 1, 128, 1, 1, 0,
		unsafe.Pointer(&input.gpu.Ptr),
		unsafe.Pointer(&w.QWeight.Ptr),
		unsafe.Pointer(&w.GIdx.Ptr),
		unsafe.Pointer(&w.Scales.Ptr),
		unsafe.Pointer(&out.gpu.Ptr),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&groups),
		unsafe.Pointer(&batchSize),
	); err == nil {
		out.dev = GPU_DEVICE
		return
	}
	gemmQ4CPU(out, input, w, B)
}

func gemmQ4CPU(out, input *DevBuf, w *GPUQuantWeight, B int) {
	if !validGPUQuantWeight(w) || input == nil || out == nil || B <= 0 {
		return
	}
	inLen, okIn := checked.MulInt(B, w.InDim)
	outLen, okOut := checked.MulInt(B, w.OutDim)
	if !okIn || !okOut || input.n < inLen || out.n < outLen {
		return
	}
	input.ToCPU()
	out.ToCPU()
	qwN, scN, giN, ok := q4BufferElementCounts(w)
	if !ok {
		return
	}
	qw := make([]int32, qwN)
	sc := make([]float32, scN)
	gi := make([]int32, giN)
	if err := w.QWeight.Download(int32ToFloat32(qw)); err != nil {
		return
	}
	if err := w.Scales.Download(sc); err != nil {
		return
	}
	if err := w.GIdx.Download(int32ToFloat32(gi)); err != nil {
		return
	}
	simdq4.GemmSym(out.cpu[:outLen], input.cpu[:inLen], B, qw, gi, sc, w.InDim, w.OutDim)
}

var fnGemmQ4 CUfunction

func init() {
	// Will be extracted from mega module
}

// BatchGEMMReady returns true if batched GEMM kernel is available.
func BatchGEMMReady() bool {
	loadMegaModule()
	return fnGemmQ4 != 0
}

// GemvQ4OrGemm dispatches to GEMV (B=1) or batched GEMM (B>1).
func GemvQ4OrGemm(out, input *DevBuf, w *GPUQuantWeight, B int) {
	if B <= 1 {
		GemvQ4(out, input, w)
		return
	}
	GemmQ4(out, input, w, B)
}
