package nvidia

import (
	"fmt"
	"sync"
	"unsafe"

	simdfp8 "github.com/rcarmo/go-pherence/backends/simd/quant/fp8"
)

var fnFP8E4M3GemvF32 CUfunction

var fp8Scratch = struct {
	sync.Mutex
	x    *Buffer
	out  *Buffer
	xN   int
	outN int
	xPtr uintptr
	xLen int
}{}

// GPUFP8E4M3Linear is a GPU-resident row-major [OutDim, InDim] F8_E4M3
// linear weight with F32 per-tensor or per-row scale and optional F32 bias.
// It mirrors backends/simd/quant/fp8.Linear so Ideogram Qwen/DiT projections
// can move from CPU LUT GEMV to a CUDA kernel without changing tensor layout.
type GPUFP8E4M3Linear struct {
	Weight      *Buffer // raw E4M3 bytes, padded to float32 allocation granularity
	Scale       *Buffer // F32, length 1 or OutDim
	Bias        *Buffer // optional F32, length OutDim
	OutDim      int
	InDim       int
	ScaleLen    int
	HasBias     bool
	WeightBytes int
}

// UploadFP8E4M3Linear uploads an F8_E4M3 row-major linear to GPU memory. Scale
// must be length 1 or OutDim; bias may be nil or OutDim.
func UploadFP8E4M3Linear(weight []byte, scale []float32, bias []float32, outDim, inDim int) (*GPUFP8E4M3Linear, error) {
	var dst *GPUFP8E4M3Linear
	if err := UploadFP8E4M3LinearReuse(&dst, weight, scale, bias, outDim, inDim); err != nil {
		return nil, err
	}
	return dst, nil
}

// UploadFP8E4M3LinearReuse uploads into *dst, reusing existing GPU buffers when
// capacity allows. This is intended for later streamed Ideogram weights.
func UploadFP8E4M3LinearReuse(dst **GPUFP8E4M3Linear, weight []byte, scale []float32, bias []float32, outDim, inDim int) error {
	lin := simdfp8.Linear{OutDim: outDim, InDim: inDim, Weight: weight, Scale: scale, Bias: bias}
	if err := lin.Validate(); err != nil {
		return err
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	EnsureContext()

	weightBytes, ok := checkedMulInt(outDim, inDim)
	if !ok {
		return fmt.Errorf("FP8 E4M3 weight size overflow out=%d in=%d", outDim, inDim)
	}
	w := *dst
	if w == nil {
		w = &GPUFP8E4M3Linear{}
		*dst = w
	}
	w.OutDim = outDim
	w.InDim = inDim
	w.ScaleLen = len(scale)
	w.HasBias = bias != nil
	w.WeightBytes = weightBytes

	if w.Weight == nil || !hasPaddedByteCapacity(w.Weight.Size, weightBytes) {
		if w.Weight != nil {
			w.Weight.Free()
			w.Weight = nil
		}
		buf, err := Malloc(f32SlotsForBytes(weightBytes))
		if err != nil {
			return fmt.Errorf("alloc FP8 E4M3 weight (%d bytes): %w", weightBytes, err)
		}
		w.Weight = buf
	}
	if err := w.Weight.UploadBytes(weight[:weightBytes]); err != nil {
		return fmt.Errorf("upload FP8 E4M3 weight: %w", err)
	}

	if w.Scale == nil || w.Scale.Size < len(scale)*4 {
		if w.Scale != nil {
			w.Scale.Free()
			w.Scale = nil
		}
		buf, err := Malloc(len(scale))
		if err != nil {
			return fmt.Errorf("alloc FP8 E4M3 scale: %w", err)
		}
		w.Scale = buf
	}
	if err := w.Scale.Upload(scale); err != nil {
		return fmt.Errorf("upload FP8 E4M3 scale: %w", err)
	}

	if bias == nil {
		if w.Bias != nil {
			w.Bias.Free()
			w.Bias = nil
		}
		return nil
	}
	if w.Bias == nil || w.Bias.Size < outDim*4 {
		if w.Bias != nil {
			w.Bias.Free()
			w.Bias = nil
		}
		buf, err := Malloc(outDim)
		if err != nil {
			return fmt.Errorf("alloc FP8 E4M3 bias: %w", err)
		}
		w.Bias = buf
	}
	if err := w.Bias.Upload(bias[:outDim]); err != nil {
		return fmt.Errorf("upload FP8 E4M3 bias: %w", err)
	}
	return nil
}

func (w *GPUFP8E4M3Linear) Free() {
	if w == nil {
		return
	}
	if w.Weight != nil {
		w.Weight.Free()
		w.Weight = nil
	}
	if w.Scale != nil {
		w.Scale.Free()
		w.Scale = nil
	}
	if w.Bias != nil {
		w.Bias.Free()
		w.Bias = nil
	}
}

// GemvFP8E4M3 computes dense out[OutDim] = dequant(W) · x with row/tensor
// scale and optional bias. It uses the CUDA direct FP8 GEMV kernel when
// available and falls back to the CPU fp8 backend otherwise.
func GemvFP8E4M3(out, x []float32, w *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(w) {
		return fmt.Errorf("invalid GPU FP8 E4M3 linear")
	}
	if len(out) < w.OutDim || len(x) < w.InDim {
		return fmt.Errorf("invalid FP8 E4M3 GEMV buffers out=%d/%d x=%d/%d", len(out), w.OutDim, len(x), w.InDim)
	}
	if SgemmReady() {
		if err := gemvFP8E4M3CUDA(out, x, w); err == nil {
			return nil
		} else {
			debugf("[gpu] FP8 E4M3 GEMV CUDA fallback: %v\n", err)
		}
	}
	lin, err := downloadFP8E4M3Linear(w)
	if err != nil {
		return err
	}
	return lin.GemvTo(x[:w.InDim], out[:w.OutDim])
}

// GemvFP8E4M3Buffer computes into GPU-resident buffers. It is the lower-level
// primitive intended for Ideogram graph wiring once activations stay on-device.
func GemvFP8E4M3Buffer(outBuf, xBuf *Buffer, w *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(w) || outBuf == nil || xBuf == nil {
		return fmt.Errorf("invalid FP8 E4M3 GEMV device buffers")
	}
	if _, err := checkedByteSize(w.OutDim, outBuf.Size); err != nil {
		return fmt.Errorf("invalid FP8 E4M3 GEMV output buffer: %w", err)
	}
	if _, err := checkedByteSize(w.InDim, xBuf.Size); err != nil {
		return fmt.Errorf("invalid FP8 E4M3 GEMV input buffer: %w", err)
	}
	if fnFP8E4M3GemvF32 == 0 || !megaModuleOK {
		return fmt.Errorf("FP8 E4M3 GEMV kernel not available")
	}
	if !fitsUint32(w.OutDim) || !fitsUint32(w.InDim) || !fitsUint32(w.ScaleLen) {
		return fmt.Errorf("FP8 E4M3 GEMV dims exceed CUDA u32 interface")
	}
	biasPtr := CUdeviceptr(0)
	hasBias := uint32(0)
	if w.Bias != nil {
		biasPtr = w.Bias.Ptr
		hasBias = 1
	}
	outDim := uint32(w.OutDim)
	inDim := uint32(w.InDim)
	scaleLen := uint32(w.ScaleLen)
	return LaunchKernel(fnFP8E4M3GemvF32, uint32(w.OutDim), 1, 1, 128, 1, 1, 128*4,
		unsafe.Pointer(&w.Weight.Ptr),
		unsafe.Pointer(&w.Scale.Ptr),
		unsafe.Pointer(&biasPtr),
		unsafe.Pointer(&xBuf.Ptr),
		unsafe.Pointer(&outBuf.Ptr),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&scaleLen),
		unsafe.Pointer(&hasBias))
}

func gemvFP8E4M3CUDA(out, x []float32, w *GPUFP8E4M3Linear) error {
	xBuf, outBuf, unlock, err := fp8ScratchBuffers(w.InDim, w.OutDim)
	if err != nil {
		return err
	}
	defer unlock()
	xSlice := x[:w.InDim]
	xPtr := uintptr(unsafe.Pointer(unsafe.SliceData(xSlice)))
	if fp8Scratch.xPtr != xPtr || fp8Scratch.xLen != w.InDim {
		if err := xBuf.Upload(xSlice); err != nil {
			return fmt.Errorf("upload FP8 E4M3 GEMV input: %w", err)
		}
		fp8Scratch.xPtr = xPtr
		fp8Scratch.xLen = w.InDim
	}
	if err := GemvFP8E4M3Buffer(outBuf, xBuf, w); err != nil {
		return err
	}
	if err := outBuf.Download(out[:w.OutDim]); err != nil {
		return fmt.Errorf("download FP8 E4M3 GEMV output: %w", err)
	}
	return nil
}

func fp8ScratchBuffers(inDim, outDim int) (*Buffer, *Buffer, func(), error) {
	fp8Scratch.Lock()
	unlock := func() { fp8Scratch.Unlock() }
	if fp8Scratch.x == nil || fp8Scratch.xN < inDim {
		if fp8Scratch.x != nil {
			fp8Scratch.x.Free()
		}
		buf, err := Malloc(inDim)
		if err != nil {
			unlock()
			return nil, nil, nil, fmt.Errorf("alloc FP8 E4M3 GEMV input scratch: %w", err)
		}
		fp8Scratch.x = buf
		fp8Scratch.xN = inDim
		fp8Scratch.xPtr = 0
		fp8Scratch.xLen = 0
	}
	if fp8Scratch.out == nil || fp8Scratch.outN < outDim {
		if fp8Scratch.out != nil {
			fp8Scratch.out.Free()
		}
		buf, err := Malloc(outDim)
		if err != nil {
			unlock()
			return nil, nil, nil, fmt.Errorf("alloc FP8 E4M3 GEMV output scratch: %w", err)
		}
		fp8Scratch.out = buf
		fp8Scratch.outN = outDim
	}
	return fp8Scratch.x, fp8Scratch.out, unlock, nil
}

func downloadFP8E4M3Linear(w *GPUFP8E4M3Linear) (*simdfp8.Linear, error) {
	if !validGPUFP8E4M3Linear(w) {
		return nil, fmt.Errorf("invalid GPU FP8 E4M3 linear")
	}
	weightPacked := make([]float32, f32SlotsForBytes(w.WeightBytes))
	if err := w.Weight.Download(weightPacked); err != nil {
		return nil, fmt.Errorf("download FP8 E4M3 weight: %w", err)
	}
	scale := make([]float32, w.ScaleLen)
	if err := w.Scale.Download(scale); err != nil {
		return nil, fmt.Errorf("download FP8 E4M3 scale: %w", err)
	}
	var bias []float32
	if w.HasBias {
		bias = make([]float32, w.OutDim)
		if err := w.Bias.Download(bias); err != nil {
			return nil, fmt.Errorf("download FP8 E4M3 bias: %w", err)
		}
	}
	lin := &simdfp8.Linear{OutDim: w.OutDim, InDim: w.InDim, Weight: float32PackedAsBytes(weightPacked, w.WeightBytes), Scale: scale, Bias: bias}
	if err := lin.Validate(); err != nil {
		return nil, err
	}
	return lin, nil
}

func validGPUFP8E4M3Linear(w *GPUFP8E4M3Linear) bool {
	if w == nil || w.Weight == nil || w.Scale == nil || w.OutDim <= 0 || w.InDim <= 0 || (w.ScaleLen != 1 && w.ScaleLen != w.OutDim) {
		return false
	}
	weightBytes, ok := checkedMulInt(w.OutDim, w.InDim)
	if !ok || w.WeightBytes != weightBytes || !hasPaddedByteCapacity(w.Weight.Size, weightBytes) || w.Scale.Size < w.ScaleLen*4 {
		return false
	}
	if w.HasBias {
		return w.Bias != nil && w.Bias.Size >= w.OutDim*4
	}
	return true
}
