package nvfp4

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"

	simdnvfp4 "github.com/rcarmo/go-pherence/backends/simd/quant/nvfp4"
)

var fnNVFP4DequantF32 CUfunction
var fnNVFP4GemvF32 CUfunction

var nvfp4Scratch = struct {
	sync.Mutex
	x    *Buffer
	out  *Buffer
	xN   int
	outN int
	xPtr uintptr
	xLen int
}{}

// NVFP4KernelKind identifies the packed NVFP4 kernel family a caller wants.
// The interface is intentionally defined before native dispatch exists so the
// fallback and future tensor-core paths agree on dimensions and buffer layout.
type NVFP4KernelKind int

const (
	NVFP4KernelGEMV NVFP4KernelKind = iota
	NVFP4KernelGEMM
)

// NVFP4KernelSpec describes a packed NVFP4 multiply request.
//
// Layout contract:
//   - W is row-major [OutDim, InDim] packed as U8 [OutDim, InDim/2]
//   - WeightScale is row-major F8_E4M3 [OutDim, InDim/GroupSize]
//   - X is F32 row-major [Batch, InDim] for GEMM, or [1, InDim] for GEMV
//   - Out is F32 row-major [Batch, OutDim]
//   - GroupSize is currently required to be 16 for ModelOpt NVFP4 native paths
//
// Native Blackwell/tensor-core dispatch can use this spec without changing the
// public fallback entry points.
type NVFP4KernelSpec struct {
	Kind      NVFP4KernelKind
	OutDim    int
	InDim     int
	Batch     int
	Groups    int
	GroupSize int
}

// NativeNVFP4TensorCoreSupported reports whether the active CUDA device is new
// enough for native NVFP4 tensor-core work. Blackwell-class GPUs are expected to
// expose compute capability 10.x or newer. This is a capability gate only; the
// native kernel path remains disabled until implemented and validated.
func NativeNVFP4TensorCoreSupported() bool {
	if !Available() {
		return false
	}
	major, minor := ComputeCapability()
	return supportsNativeNVFP4TensorCore(major, minor)
}

func supportsNativeNVFP4TensorCore(major, minor int) bool {
	return major >= 10
}

// ValidateNVFP4KernelSpec checks packed NVFP4 GEMV/GEMM dimensions without
// requiring GPU buffers. It is the shared shape gate for future native kernels.
func ValidateNVFP4KernelSpec(spec NVFP4KernelSpec) error {
	if spec.Kind != NVFP4KernelGEMV && spec.Kind != NVFP4KernelGEMM {
		return fmt.Errorf("invalid NVFP4 kernel kind %d", spec.Kind)
	}
	if spec.OutDim <= 0 || spec.InDim <= 0 || spec.Batch <= 0 || spec.Groups <= 0 || spec.GroupSize <= 0 {
		return fmt.Errorf("invalid NVFP4 kernel dims out=%d in=%d batch=%d groups=%d groupSize=%d", spec.OutDim, spec.InDim, spec.Batch, spec.Groups, spec.GroupSize)
	}
	if spec.OutDim > math.MaxUint32 || spec.InDim > math.MaxUint32 || spec.Batch > math.MaxUint32 || spec.Groups > math.MaxUint32 || spec.GroupSize > math.MaxUint32 {
		return fmt.Errorf("NVFP4 kernel dims exceed CUDA u32 interface out=%d in=%d batch=%d groups=%d groupSize=%d", spec.OutDim, spec.InDim, spec.Batch, spec.Groups, spec.GroupSize)
	}
	if spec.Kind == NVFP4KernelGEMV && spec.Batch != 1 {
		return fmt.Errorf("NVFP4 GEMV batch=%d, want 1", spec.Batch)
	}
	if spec.InDim%2 != 0 || spec.InDim%spec.GroupSize != 0 || spec.Groups != spec.InDim/spec.GroupSize {
		return fmt.Errorf("NVFP4 kernel group layout mismatch in=%d groups=%d groupSize=%d", spec.InDim, spec.Groups, spec.GroupSize)
	}
	if spec.GroupSize != 16 {
		return fmt.Errorf("NVFP4 native kernels require groupSize=16, got %d", spec.GroupSize)
	}
	if _, ok := checkedMulInt(spec.OutDim, spec.InDim/2); !ok {
		return fmt.Errorf("NVFP4 packed weight bytes overflow out=%d in=%d", spec.OutDim, spec.InDim)
	}
	if _, ok := checkedMulInt(spec.OutDim, spec.Groups); !ok {
		return fmt.Errorf("NVFP4 scale bytes overflow out=%d groups=%d", spec.OutDim, spec.Groups)
	}
	if _, ok := checkedMulInt(spec.Batch, spec.InDim); !ok {
		return fmt.Errorf("NVFP4 input elements overflow batch=%d in=%d", spec.Batch, spec.InDim)
	}
	if _, ok := checkedMulInt(spec.Batch, spec.OutDim); !ok {
		return fmt.Errorf("NVFP4 output elements overflow batch=%d out=%d", spec.Batch, spec.OutDim)
	}
	return nil
}

// GPUNVFP4Weight is the GPU-resident representation for ModelOpt/NVFP4
// weights. It is deliberately separate from GPTQ/MLX upload structures because
// NVFP4 uses U8 packed FP4 weights plus F8_E4M3 per-block scales and scalar
// scale metadata, not affine INT4 or GPTQ group-index metadata.
type GPUNVFP4Weight struct {
	Weight        *Buffer // raw U8 packed FP4 bytes, padded to float32 allocation granularity
	WeightScale   *Buffer // raw F8_E4M3 scale bytes, padded to float32 allocation granularity
	WeightScale2  float32
	InputScale    float32
	HasInputScale bool
	OutDim        int
	InDim         int
	Groups        int
	GroupSize     int
	WeightBytes   int
	ScaleBytes    int
}

// UploadNVFP4Weight uploads the observed NVFP4 packed representation to GPU
// memory without converting it to MLX/GPTQ. Kernel dispatch is added separately.
func UploadNVFP4Weight(qw *simdnvfp4.NVFP4Weight) (*GPUNVFP4Weight, error) {
	var w *GPUNVFP4Weight
	if err := UploadNVFP4WeightReuse(&w, qw); err != nil {
		return nil, err
	}
	return w, nil
}

// UploadNVFP4WeightReuse uploads qw into *dst, reusing sufficiently large
// device buffers when possible. This is intended for transient streamed weights
// where dimensions change but allocation churn is more expensive than copies.
func UploadNVFP4WeightReuse(dst **GPUNVFP4Weight, qw *simdnvfp4.NVFP4Weight) error {
	if err := simdnvfp4.ValidateNVFP4Weight(qw); err != nil {
		return err
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	EnsureContext()

	weightBytes, scaleBytes, err := nvfp4RequiredBytes(qw.OutDim, qw.InDim, qw.Groups)
	if err != nil {
		return err
	}
	w := *dst
	if w == nil {
		w = &GPUNVFP4Weight{}
		*dst = w
	}
	w.WeightScale2 = qw.WeightScale2
	w.InputScale = qw.InputScale
	w.HasInputScale = qw.HasInputScale
	w.OutDim = qw.OutDim
	w.InDim = qw.InDim
	w.Groups = qw.Groups
	w.GroupSize = qw.GroupSize
	w.WeightBytes = weightBytes
	w.ScaleBytes = scaleBytes

	if w.Weight == nil || !hasPaddedByteCapacity(w.Weight.Size, weightBytes) {
		if w.Weight != nil {
			w.Weight.Free()
			w.Weight = nil
		}
		wb, err := Malloc(f32SlotsForBytes(weightBytes))
		if err != nil {
			return fmt.Errorf("alloc NVFP4 weight (%d bytes): %w", weightBytes, err)
		}
		w.Weight = wb
	}
	if err := w.Weight.UploadBytes(qw.Weight[:weightBytes]); err != nil {
		return fmt.Errorf("upload NVFP4 weight: %w", err)
	}

	if w.WeightScale == nil || !hasPaddedByteCapacity(w.WeightScale.Size, scaleBytes) {
		if w.WeightScale != nil {
			w.WeightScale.Free()
			w.WeightScale = nil
		}
		sb, err := Malloc(f32SlotsForBytes(scaleBytes))
		if err != nil {
			return fmt.Errorf("alloc NVFP4 weight_scale (%d bytes): %w", scaleBytes, err)
		}
		w.WeightScale = sb
	}
	if err := w.WeightScale.UploadBytes(qw.WeightScale[:scaleBytes]); err != nil {
		return fmt.Errorf("upload NVFP4 weight_scale: %w", err)
	}
	return nil
}

func uploadNVFP4WeightFresh(qw *simdnvfp4.NVFP4Weight) (*GPUNVFP4Weight, error) {
	if err := simdnvfp4.ValidateNVFP4Weight(qw); err != nil {
		return nil, err
	}
	if !SgemmReady() {
		return nil, fmt.Errorf("GPU not available")
	}
	EnsureContext()

	weightBytes, scaleBytes, err := nvfp4RequiredBytes(qw.OutDim, qw.InDim, qw.Groups)
	if err != nil {
		return nil, err
	}
	w := &GPUNVFP4Weight{
		WeightScale2:  qw.WeightScale2,
		InputScale:    qw.InputScale,
		HasInputScale: qw.HasInputScale,
		OutDim:        qw.OutDim,
		InDim:         qw.InDim,
		Groups:        qw.Groups,
		GroupSize:     qw.GroupSize,
		WeightBytes:   weightBytes,
		ScaleBytes:    scaleBytes,
	}

	wb, err := Malloc(f32SlotsForBytes(weightBytes))
	if err != nil {
		return nil, fmt.Errorf("alloc NVFP4 weight (%d bytes): %w", weightBytes, err)
	}
	w.Weight = wb
	if err := wb.UploadBytes(qw.Weight[:weightBytes]); err != nil {
		w.Free()
		return nil, fmt.Errorf("upload NVFP4 weight: %w", err)
	}

	sb, err := Malloc(f32SlotsForBytes(scaleBytes))
	if err != nil {
		w.Free()
		return nil, fmt.Errorf("alloc NVFP4 weight_scale (%d bytes): %w", scaleBytes, err)
	}
	w.WeightScale = sb
	if err := sb.UploadBytes(qw.WeightScale[:scaleBytes]); err != nil {
		w.Free()
		return nil, fmt.Errorf("upload NVFP4 weight_scale: %w", err)
	}
	return w, nil
}

func (w *GPUNVFP4Weight) Free() {
	if w == nil {
		return
	}
	if w.Weight != nil {
		w.Weight.Free()
		w.Weight = nil
	}
	if w.WeightScale != nil {
		w.WeightScale.Free()
		w.WeightScale = nil
	}
}

// DequantNVFP4ToF32 materializes a GPU-resident NVFP4 weight as row-major F32.
// It first tries the CUDA dequant kernel, then falls back to downloading raw
// buffers and reusing the SIMD NVFP4 reference dequantizer.
func DequantNVFP4ToF32(w *GPUNVFP4Weight) ([]float32, error) {
	if !validGPUNVFP4Weight(w) {
		return nil, fmt.Errorf("invalid GPU NVFP4 weight")
	}
	if out, ok := dequantNVFP4ToF32CUDA(w); ok {
		return out, nil
	}
	qw, err := downloadNVFP4Weight(w)
	if err != nil {
		return nil, err
	}
	out := simdnvfp4.DequantNVFP4(qw)
	if out == nil {
		return nil, fmt.Errorf("dequantize NVFP4 fallback failed")
	}
	return out, nil
}

// GemvNVFP4 computes dense out[outDim] = W_nvfp4[outDim,inDim] · x[inDim].
// It keeps the hot path on-device: packed NVFP4 is dequantized into a transient
// GPU F32 matrix and multiplied with SGEMM, downloading only the output vector.
// If any CUDA step fails, callers can fall back to the CPU SIMD NVFP4 path.
func GemvNVFP4(out, x []float32, w *GPUNVFP4Weight) error {
	if !validGPUNVFP4Weight(w) {
		return fmt.Errorf("invalid GPU NVFP4 weight")
	}
	if len(out) < w.OutDim || len(x) < w.InDim {
		return fmt.Errorf("invalid NVFP4 GEMV buffers out=%d/%d x=%d/%d", len(out), w.OutDim, len(x), w.InDim)
	}
	if SgemmReady() {
		if err := gemvNVFP4PackedCUDA(out, x, w); err == nil {
			return nil
		} else {
			debugf("[gpu] NVFP4 packed GEMV CUDA fallback: %v\n", err)
		}
		if err := gemvNVFP4CUDA(out, x, w); err == nil {
			return nil
		} else {
			debugf("[gpu] NVFP4 GEMV CUDA fallback: %v\n", err)
		}
	}
	qw, err := downloadNVFP4Weight(w)
	if err != nil {
		return err
	}
	if !simdnvfp4.GemvNVFP4To(out[:w.OutDim], x[:w.InDim], qw) {
		return fmt.Errorf("NVFP4 CPU GEMV fallback failed")
	}
	return nil
}

func downloadNVFP4Weight(w *GPUNVFP4Weight) (*simdnvfp4.NVFP4Weight, error) {
	if !validGPUNVFP4Weight(w) {
		return nil, fmt.Errorf("invalid GPU NVFP4 weight")
	}
	weightPacked := make([]float32, f32SlotsForBytes(w.WeightBytes))
	if err := w.Weight.Download(weightPacked); err != nil {
		return nil, fmt.Errorf("download NVFP4 weight: %w", err)
	}
	scalePacked := make([]float32, f32SlotsForBytes(w.ScaleBytes))
	if err := w.WeightScale.Download(scalePacked); err != nil {
		return nil, fmt.Errorf("download NVFP4 weight_scale: %w", err)
	}
	qw := &simdnvfp4.NVFP4Weight{
		Weight:        float32PackedAsBytes(weightPacked, w.WeightBytes),
		WeightScale:   float32PackedAsBytes(scalePacked, w.ScaleBytes),
		WeightScale2:  w.WeightScale2,
		InputScale:    w.InputScale,
		HasInputScale: w.HasInputScale,
		OutDim:        w.OutDim,
		InDim:         w.InDim,
		Groups:        w.Groups,
		GroupSize:     w.GroupSize,
	}
	if err := simdnvfp4.ValidateNVFP4Weight(qw); err != nil {
		return nil, err
	}
	return qw, nil
}

// GemmNVFP4 computes dense out[batch,outDim] = x[batch,inDim] @ W_nvfp4^T.
// Native packed GEMM is not implemented yet; this correctness-first entrypoint
// uses the packed SIMD NVFP4 reference as the CPU fallback contract future CUDA
// kernels must match.
func GemmNVFP4(out, x []float32, batch int, w *GPUNVFP4Weight) error {
	if !validGPUNVFP4Weight(w) {
		return fmt.Errorf("invalid GPU NVFP4 weight")
	}
	spec := NVFP4KernelSpec{Kind: NVFP4KernelGEMM, OutDim: w.OutDim, InDim: w.InDim, Batch: batch, Groups: w.Groups, GroupSize: w.GroupSize}
	if err := ValidateNVFP4KernelSpec(spec); err != nil {
		return err
	}
	xLen, okX := checkedMulInt(batch, w.InDim)
	outLen, okOut := checkedMulInt(batch, w.OutDim)
	if !okX || !okOut || len(x) < xLen || len(out) < outLen {
		return fmt.Errorf("invalid NVFP4 GEMM buffers out=%d/%d x=%d/%d", len(out), outLen, len(x), xLen)
	}
	qw, err := downloadNVFP4Weight(w)
	if err != nil {
		return err
	}
	if !simdnvfp4.GemmNVFP4(out[:outLen], x[:xLen], batch, qw) {
		return fmt.Errorf("NVFP4 CPU GEMM fallback failed")
	}
	return nil
}

func GemvNVFP4Buffer(outBuf, xBuf *Buffer, w *GPUNVFP4Weight) error {
	if !validGPUNVFP4Weight(w) || outBuf == nil || xBuf == nil {
		return fmt.Errorf("invalid NVFP4 GEMV device buffers")
	}
	if _, err := checkedByteSize(w.OutDim, outBuf.Size); err != nil {
		return fmt.Errorf("invalid NVFP4 GEMV output buffer: %w", err)
	}
	if _, err := checkedByteSize(w.InDim, xBuf.Size); err != nil {
		return fmt.Errorf("invalid NVFP4 GEMV input buffer: %w", err)
	}
	if fnNVFP4GemvF32 == 0 || !megaModuleOK {
		return fmt.Errorf("NVFP4 packed GEMV kernel not available")
	}
	if !fitsUint32(w.OutDim) || !fitsUint32(w.InDim) || !fitsUint32(w.GroupSize) {
		return fmt.Errorf("NVFP4 packed GEMV dims exceed CUDA u32 interface")
	}
	outDim := uint32(w.OutDim)
	inDim := uint32(w.InDim)
	groupSize := uint32(w.GroupSize)
	return LaunchKernel(fnNVFP4GemvF32, uint32(w.OutDim), 1, 1, 128, 1, 1, 128*4,
		unsafe.Pointer(&w.Weight.Ptr),
		unsafe.Pointer(&w.WeightScale.Ptr),
		unsafe.Pointer(&xBuf.Ptr),
		unsafe.Pointer(&outBuf.Ptr),
		unsafe.Pointer(&w.WeightScale2),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&groupSize))
}

func F32SiLUMulBuffer(out, a, b *Buffer, n int) error {
	if fnFusedSiLUMul == 0 || out == nil || a == nil || b == nil || n <= 0 || !fitsUint32(n) {
		return fmt.Errorf("invalid F32 SiLU*Mul device buffers")
	}
	if _, err := checkedByteSize(n, out.Size); err != nil {
		return fmt.Errorf("invalid F32 SiLU*Mul output buffer: %w", err)
	}
	if _, err := checkedByteSize(n, a.Size); err != nil {
		return fmt.Errorf("invalid F32 SiLU*Mul input A buffer: %w", err)
	}
	if _, err := checkedByteSize(n, b.Size); err != nil {
		return fmt.Errorf("invalid F32 SiLU*Mul input B buffer: %w", err)
	}
	grid, okGrid := grid1DFor(n, 256)
	if !okGrid {
		return fmt.Errorf("invalid F32 SiLU*Mul grid")
	}
	nn := uint32(n)
	return LaunchKernel(fnFusedSiLUMul, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&a.Ptr), unsafe.Pointer(&b.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&nn))
}

func gemvNVFP4PackedCUDA(out, x []float32, w *GPUNVFP4Weight) error {
	if fnNVFP4GemvF32 == 0 || !megaModuleOK {
		return fmt.Errorf("NVFP4 packed GEMV kernel not available")
	}
	if !fitsUint32(w.OutDim) || !fitsUint32(w.InDim) || !fitsUint32(w.GroupSize) {
		return fmt.Errorf("NVFP4 packed GEMV dims exceed CUDA u32 interface")
	}
	xBuf, outBuf, unlock, err := nvfp4ScratchBuffers(w.InDim, w.OutDim)
	if err != nil {
		return err
	}
	defer unlock()
	xSlice := x[:w.InDim]
	xPtr := uintptr(unsafe.Pointer(unsafe.SliceData(xSlice)))
	if nvfp4Scratch.xPtr != xPtr || nvfp4Scratch.xLen != w.InDim {
		if err := xBuf.Upload(xSlice); err != nil {
			return fmt.Errorf("upload NVFP4 packed GEMV input: %w", err)
		}
		nvfp4Scratch.xPtr = xPtr
		nvfp4Scratch.xLen = w.InDim
	}
	outDim := uint32(w.OutDim)
	inDim := uint32(w.InDim)
	groupSize := uint32(w.GroupSize)
	if err := LaunchKernel(fnNVFP4GemvF32, uint32(w.OutDim), 1, 1, 128, 1, 1, 128*4,
		unsafe.Pointer(&w.Weight.Ptr),
		unsafe.Pointer(&w.WeightScale.Ptr),
		unsafe.Pointer(&xBuf.Ptr),
		unsafe.Pointer(&outBuf.Ptr),
		unsafe.Pointer(&w.WeightScale2),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&groupSize)); err != nil {
		return fmt.Errorf("launch NVFP4 packed GEMV: %w", err)
	}
	if err := outBuf.Download(out[:w.OutDim]); err != nil {
		return fmt.Errorf("download NVFP4 packed GEMV output: %w", err)
	}
	return nil
}

func nvfp4ScratchBuffers(inDim, outDim int) (*Buffer, *Buffer, func(), error) {
	nvfp4Scratch.Lock()
	unlock := func() { nvfp4Scratch.Unlock() }
	if nvfp4Scratch.x == nil || nvfp4Scratch.xN < inDim {
		if nvfp4Scratch.x != nil {
			nvfp4Scratch.x.Free()
		}
		buf, err := Malloc(inDim)
		if err != nil {
			unlock()
			return nil, nil, nil, fmt.Errorf("alloc NVFP4 packed GEMV input scratch: %w", err)
		}
		nvfp4Scratch.x = buf
		nvfp4Scratch.xN = inDim
	}
	if nvfp4Scratch.out == nil || nvfp4Scratch.outN < outDim {
		if nvfp4Scratch.out != nil {
			nvfp4Scratch.out.Free()
		}
		buf, err := Malloc(outDim)
		if err != nil {
			unlock()
			return nil, nil, nil, fmt.Errorf("alloc NVFP4 packed GEMV output scratch: %w", err)
		}
		nvfp4Scratch.out = buf
		nvfp4Scratch.outN = outDim
	}
	return nvfp4Scratch.x, nvfp4Scratch.out, unlock, nil
}

func gemvNVFP4CUDA(out, x []float32, w *GPUNVFP4Weight) error {
	weights, err := dequantNVFP4ToF32GPU(w)
	if err != nil {
		return err
	}
	defer weights.Free()
	xBuf, err := Malloc(w.InDim)
	if err != nil {
		return fmt.Errorf("alloc NVFP4 GEMV input: %w", err)
	}
	defer xBuf.Free()
	if err := xBuf.Upload(x[:w.InDim]); err != nil {
		return fmt.Errorf("upload NVFP4 GEMV input: %w", err)
	}
	outBuf, err := Malloc(w.OutDim)
	if err != nil {
		return fmt.Errorf("alloc NVFP4 GEMV output: %w", err)
	}
	defer outBuf.Free()
	if err := Sgemm(w.OutDim, 1, w.InDim, 1.0, weights, xBuf, outBuf); err != nil {
		return fmt.Errorf("NVFP4 GEMV SGEMM: %w", err)
	}
	if err := SyncErr(); err != nil {
		return fmt.Errorf("NVFP4 GEMV sync: %w", err)
	}
	if err := outBuf.Download(out[:w.OutDim]); err != nil {
		return fmt.Errorf("download NVFP4 GEMV output: %w", err)
	}
	return nil
}

func gemvNVFP4F32(out, x []float32, outDim, inDim int, weights []float32) error {
	wantWeights, ok := checkedMulInt(outDim, inDim)
	if outDim <= 0 || inDim <= 0 || !ok || len(out) < outDim || len(x) < inDim || len(weights) < wantWeights {
		return fmt.Errorf("invalid NVFP4 F32 GEMV buffers out=%d/%d x=%d/%d weights=%d/%d", len(out), outDim, len(x), inDim, len(weights), wantWeights)
	}
	for row := 0; row < outDim; row++ {
		sum := float32(0)
		rowOff := row * inDim
		for col := 0; col < inDim; col++ {
			sum += weights[rowOff+col] * x[col]
		}
		out[row] = sum
	}
	return nil
}

func dequantNVFP4ToF32CUDA(w *GPUNVFP4Weight) ([]float32, bool) {
	outBuf, err := dequantNVFP4ToF32GPU(w)
	if err != nil {
		debugf("[gpu] NVFP4 CUDA dequant fallback: %v\n", err)
		return nil, false
	}
	defer outBuf.Free()
	outLen, _ := checkedMulInt(w.OutDim, w.InDim)
	out := make([]float32, outLen)
	if err := outBuf.Download(out); err != nil {
		debugf("[gpu] NVFP4 CUDA dequant download fallback: %v\n", err)
		return nil, false
	}
	return out, true
}

func dequantNVFP4ToF32GPU(w *GPUNVFP4Weight) (*Buffer, error) {
	if fnNVFP4DequantF32 == 0 || !megaModuleOK {
		return nil, fmt.Errorf("NVFP4 dequant kernel not available")
	}
	outLen, ok := checkedMulInt(w.OutDim, w.InDim)
	if !ok || !fitsUint32(outLen) || !fitsUint32(w.InDim) || !fitsUint32(w.GroupSize) {
		return nil, fmt.Errorf("invalid NVFP4 dequant dims out=%d in=%d group=%d", w.OutDim, w.InDim, w.GroupSize)
	}
	outBuf, err := Malloc(outLen)
	if err != nil {
		return nil, fmt.Errorf("alloc NVFP4 dequant output: %w", err)
	}
	total := uint32(outLen)
	inDim := uint32(w.InDim)
	groupSize := uint32(w.GroupSize)
	grid := (total + 255) / 256
	if err := LaunchKernel(fnNVFP4DequantF32, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&w.Weight.Ptr),
		unsafe.Pointer(&w.WeightScale.Ptr),
		unsafe.Pointer(&outBuf.Ptr),
		unsafe.Pointer(&w.WeightScale2),
		unsafe.Pointer(&total),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&groupSize)); err != nil {
		outBuf.Free()
		return nil, fmt.Errorf("launch NVFP4 dequant: %w", err)
	}
	return outBuf, nil
}

func validGPUNVFP4Weight(w *GPUNVFP4Weight) bool {
	if w == nil || w.Weight == nil || w.WeightScale == nil || w.OutDim <= 0 || w.InDim <= 0 || w.Groups <= 0 || w.GroupSize <= 0 {
		return false
	}
	if w.InDim%2 != 0 || w.InDim%w.GroupSize != 0 || w.Groups != w.InDim/w.GroupSize {
		return false
	}
	weightBytes, scaleBytes, err := nvfp4RequiredBytes(w.OutDim, w.InDim, w.Groups)
	if err != nil || w.WeightBytes != weightBytes || w.ScaleBytes != scaleBytes {
		return false
	}
	return hasPaddedByteCapacity(w.Weight.Size, weightBytes) && hasPaddedByteCapacity(w.WeightScale.Size, scaleBytes)
}

func nvfp4RequiredBytes(outDim, inDim, groups int) (int, int, error) {
	if outDim <= 0 || inDim <= 0 || groups <= 0 || inDim%2 != 0 {
		return 0, 0, fmt.Errorf("invalid NVFP4 byte dims out=%d in=%d groups=%d", outDim, inDim, groups)
	}
	weightBytes, ok := checkedMulInt(outDim, inDim/2)
	if !ok {
		return 0, 0, fmt.Errorf("NVFP4 weight byte size overflows out=%d in=%d", outDim, inDim)
	}
	scaleBytes, ok := checkedMulInt(outDim, groups)
	if !ok {
		return 0, 0, fmt.Errorf("NVFP4 scale byte size overflows out=%d groups=%d", outDim, groups)
	}
	return weightBytes, scaleBytes, nil
}

func f32SlotsForBytes(n int) int {
	if n <= 0 {
		return 0
	}
	return n/4 + boolToInt(n%4 != 0)
}

func hasPaddedByteCapacity(sizeBytes, requiredBytes int) bool {
	if requiredBytes <= 0 {
		return sizeBytes >= 0
	}
	maxInt := int(^uint(0) >> 1)
	slots := f32SlotsForBytes(requiredBytes)
	if slots > maxInt/4 {
		return false
	}
	return sizeBytes >= slots*4
}

func bytesAsFloat32Padded(data []byte) []float32 {
	out := make([]float32, f32SlotsForBytes(len(data)))
	for i := range out {
		off := i * 4
		if off+4 <= len(data) {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off : off+4]))
		} else {
			var tmp [4]byte
			copy(tmp[:], data[off:])
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(tmp[:]))
		}
	}
	return out
}

func float32PackedAsBytes(data []float32, n int) []byte {
	if n <= 0 || len(data) == 0 {
		return nil
	}
	maxBytes, ok := checkedMulInt(len(data), 4)
	if !ok || n > maxBytes {
		return nil
	}
	out := make([]byte, maxBytes)
	for i, f := range data {
		binary.LittleEndian.PutUint32(out[i*4:i*4+4], math.Float32bits(f))
	}
	return out[:n]
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}
