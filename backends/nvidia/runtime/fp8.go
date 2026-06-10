package nvidia

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"unsafe"

	simdfp8 "github.com/rcarmo/go-pherence/backends/simd/quant/fp8"
)

var fnFP8E4M3GemvF32 CUfunction
var fnFP8E4M3GemmF32 CUfunction
var fnFP8E4M3DequantTransposeF32 CUfunction

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

// GemmFP8E4M3 computes dense row-major out[batch,OutDim] = x[batch,InDim] · W^T.
func GemmFP8E4M3(out, x []float32, batch int, w *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(w) {
		return fmt.Errorf("invalid GPU FP8 E4M3 linear")
	}
	if batch <= 0 {
		return fmt.Errorf("invalid FP8 E4M3 GEMM batch=%d", batch)
	}
	inLen, ok := checkedMulInt(batch, w.InDim)
	if !ok {
		return fmt.Errorf("FP8 E4M3 GEMM input size overflow batch=%d in=%d", batch, w.InDim)
	}
	outLen, ok := checkedMulInt(batch, w.OutDim)
	if !ok {
		return fmt.Errorf("FP8 E4M3 GEMM output size overflow batch=%d out=%d", batch, w.OutDim)
	}
	if len(x) < inLen || len(out) < outLen {
		return fmt.Errorf("invalid FP8 E4M3 GEMM buffers out=%d/%d x=%d/%d", len(out), outLen, len(x), inLen)
	}
	if SgemmReady() {
		if fp8SgemmEnabled() && !w.HasBias {
			if err := gemmFP8E4M3ViaSgemm(out, x, batch, w); err == nil {
				return nil
			} else {
				debugf("[gpu] FP8 E4M3 SGEMM fallback: %v\n", err)
			}
		}
		if err := gemmFP8E4M3CUDA(out, x, batch, w); err == nil {
			return nil
		} else {
			debugf("[gpu] FP8 E4M3 GEMM CUDA fallback: %v\n", err)
		}
	}
	lin, err := downloadFP8E4M3Linear(w)
	if err != nil {
		return err
	}
	for b := 0; b < batch; b++ {
		if err := lin.GemvTo(x[b*w.InDim:(b+1)*w.InDim], out[b*w.OutDim:(b+1)*w.OutDim]); err != nil {
			return err
		}
	}
	return nil
}

func GemmFP8E4M3Buffer(outBuf, xBuf *Buffer, batch int, w *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(w) || outBuf == nil || xBuf == nil {
		return fmt.Errorf("invalid FP8 E4M3 GEMM device buffers")
	}
	inLen, ok := checkedMulInt(batch, w.InDim)
	if !ok {
		return fmt.Errorf("FP8 E4M3 GEMM input size overflow")
	}
	outLen, ok := checkedMulInt(batch, w.OutDim)
	if !ok {
		return fmt.Errorf("FP8 E4M3 GEMM output size overflow")
	}
	if _, err := checkedByteSize(outLen, outBuf.Size); err != nil {
		return fmt.Errorf("invalid FP8 E4M3 GEMM output buffer: %w", err)
	}
	if _, err := checkedByteSize(inLen, xBuf.Size); err != nil {
		return fmt.Errorf("invalid FP8 E4M3 GEMM input buffer: %w", err)
	}
	if fnFP8E4M3GemmF32 == 0 || !megaModuleOK {
		return fmt.Errorf("FP8 E4M3 GEMM kernel not available")
	}
	if !fitsUint32(w.OutDim) || !fitsUint32(w.InDim) || !fitsUint32(w.ScaleLen) || !fitsUint32(batch) {
		return fmt.Errorf("FP8 E4M3 GEMM dims exceed CUDA u32 interface")
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
	batchU := uint32(batch)
	return LaunchKernel(fnFP8E4M3GemmF32, uint32(w.OutDim), batchU, 1, 128, 1, 1, 128*4,
		unsafe.Pointer(&w.Weight.Ptr),
		unsafe.Pointer(&w.Scale.Ptr),
		unsafe.Pointer(&biasPtr),
		unsafe.Pointer(&xBuf.Ptr),
		unsafe.Pointer(&outBuf.Ptr),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&scaleLen),
		unsafe.Pointer(&hasBias),
		unsafe.Pointer(&batchU))
}

// Gemm2FP8E4M3SameInput computes two row-major FP8 GEMMs sharing one uploaded
// activation matrix. This is intended for SwiGLU-style projection pairs such as
// Ideogram/Qwen W1+W3, where both linears consume the exact same [batch,InDim]
// input. It reduces host-device traffic without changing numerical kernels.
func GemmDiTLayerIslandsFP8E4M3(hidden []float32, attnN1, scaleMSA, gateMSA, normQ, normK, attnN2, ffnN1, scaleMLP, gateMLP, ffnN2, cos, sin []float32, tokens, heads, headDim int, scaleAttn, normEps float32, qkv, o, w1, w3, w2 *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(qkv) || !validGPUFP8E4M3Linear(o) || !validGPUFP8E4M3Linear(w1) || !validGPUFP8E4M3Linear(w3) || !validGPUFP8E4M3Linear(w2) {
		return fmt.Errorf("invalid GPU FP8 E4M3 full-layer island weights")
	}
	emb := heads * headDim
	n := tokens * emb
	tableLen := tokens * (headDim / 2)
	if tokens <= 0 || heads <= 0 || headDim <= 0 || len(hidden) < n || len(attnN1) < emb || len(scaleMSA) < emb || len(gateMSA) < emb || len(normQ) < headDim || len(normK) < headDim || len(attnN2) < emb || len(ffnN1) < emb || len(scaleMLP) < emb || len(gateMLP) < emb || len(ffnN2) < emb || len(cos) < tableLen || len(sin) < tableLen {
		return fmt.Errorf("invalid FP8 full-layer island buffers")
	}
	alloc := func(name string, size int) (*Buffer, error) {
		b, err := Malloc(size)
		if err != nil {
			return nil, fmt.Errorf("alloc FP8 full-layer %s: %w", name, err)
		}
		return b, nil
	}
	hiddenBuf, err := alloc("hidden", n)
	if err != nil {
		return err
	}
	defer hiddenBuf.Free()
	normedBuf, err := alloc("normed", n)
	if err != nil {
		return err
	}
	defer normedBuf.Free()
	attnN1Buf, err := alloc("attnN1", emb)
	if err != nil {
		return err
	}
	defer attnN1Buf.Free()
	scaleMSABuf, err := alloc("scaleMSA", emb)
	if err != nil {
		return err
	}
	defer scaleMSABuf.Free()
	gateMSABuf, err := alloc("gateMSA", emb)
	if err != nil {
		return err
	}
	defer gateMSABuf.Free()
	normQBuf, err := alloc("normQ", headDim)
	if err != nil {
		return err
	}
	defer normQBuf.Free()
	normKBuf, err := alloc("normK", headDim)
	if err != nil {
		return err
	}
	defer normKBuf.Free()
	cosBuf, err := alloc("cos", tableLen)
	if err != nil {
		return err
	}
	defer cosBuf.Free()
	sinBuf, err := alloc("sin", tableLen)
	if err != nil {
		return err
	}
	defer sinBuf.Free()
	attnN2Buf, err := alloc("attnN2", emb)
	if err != nil {
		return err
	}
	defer attnN2Buf.Free()
	ffnN1Buf, err := alloc("ffnN1", emb)
	if err != nil {
		return err
	}
	defer ffnN1Buf.Free()
	scaleMLPBuf, err := alloc("scaleMLP", emb)
	if err != nil {
		return err
	}
	defer scaleMLPBuf.Free()
	gateMLPBuf, err := alloc("gateMLP", emb)
	if err != nil {
		return err
	}
	defer gateMLPBuf.Free()
	ffnN2Buf, err := alloc("ffnN2", emb)
	if err != nil {
		return err
	}
	defer ffnN2Buf.Free()
	if err := hiddenBuf.Upload(hidden[:n]); err != nil {
		return fmt.Errorf("upload FP8 full-layer hidden: %w", err)
	}
	if err := attnN1Buf.Upload(attnN1[:emb]); err != nil {
		return fmt.Errorf("upload FP8 full-layer attnN1: %w", err)
	}
	if err := scaleMSABuf.Upload(scaleMSA[:emb]); err != nil {
		return fmt.Errorf("upload FP8 full-layer scaleMSA: %w", err)
	}
	if err := gateMSABuf.Upload(gateMSA[:emb]); err != nil {
		return fmt.Errorf("upload FP8 full-layer gateMSA: %w", err)
	}
	if err := normQBuf.Upload(normQ[:headDim]); err != nil {
		return fmt.Errorf("upload FP8 full-layer normQ: %w", err)
	}
	if err := normKBuf.Upload(normK[:headDim]); err != nil {
		return fmt.Errorf("upload FP8 full-layer normK: %w", err)
	}
	if err := cosBuf.Upload(cos[:tableLen]); err != nil {
		return fmt.Errorf("upload FP8 full-layer cos: %w", err)
	}
	if err := sinBuf.Upload(sin[:tableLen]); err != nil {
		return fmt.Errorf("upload FP8 full-layer sin: %w", err)
	}
	if err := attnN2Buf.Upload(attnN2[:emb]); err != nil {
		return fmt.Errorf("upload FP8 full-layer attnN2: %w", err)
	}
	if err := ffnN1Buf.Upload(ffnN1[:emb]); err != nil {
		return fmt.Errorf("upload FP8 full-layer ffnN1: %w", err)
	}
	if err := scaleMLPBuf.Upload(scaleMLP[:emb]); err != nil {
		return fmt.Errorf("upload FP8 full-layer scaleMLP: %w", err)
	}
	if err := gateMLPBuf.Upload(gateMLP[:emb]); err != nil {
		return fmt.Errorf("upload FP8 full-layer gateMLP: %w", err)
	}
	if err := ffnN2Buf.Upload(ffnN2[:emb]); err != nil {
		return fmt.Errorf("upload FP8 full-layer ffnN2: %w", err)
	}
	if err := IdeogramRMSNormRowsBuffer(normedBuf, hiddenBuf, attnN1Buf, scaleMSABuf, tokens, emb, normEps, true); err != nil {
		return fmt.Errorf("FP8 full-layer attn prenorm: %w", err)
	}
	if err := GemmQKVAttentionOResidualFP8E4M3Buffer(hiddenBuf, normedBuf, gateMSABuf, normQBuf, normKBuf, attnN2Buf, cosBuf, sinBuf, tokens, heads, headDim, scaleAttn, qkv, o); err != nil {
		return err
	}
	if err := IdeogramRMSNormRowsBuffer(normedBuf, hiddenBuf, ffnN1Buf, scaleMLPBuf, tokens, emb, normEps, true); err != nil {
		return fmt.Errorf("FP8 full-layer mlp prenorm: %w", err)
	}
	if err := GemmSwiGLUResidualFP8E4M3Buffer(hiddenBuf, normedBuf, gateMLPBuf, ffnN2Buf, tokens, w1, w3, w2); err != nil {
		return err
	}
	if err := hiddenBuf.Download(hidden[:n]); err != nil {
		return fmt.Errorf("download FP8 full-layer hidden: %w", err)
	}
	return nil
}

func GemmQKVAttentionOResidualFP8E4M3Buffer(hiddenBuf, xBuf, gateBuf, normQBuf, normKBuf, normOutBuf, cosBuf, sinBuf *Buffer, tokens, heads, headDim int, scale float32, wqkv, wo *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(wqkv) || !validGPUFP8E4M3Linear(wo) || tokens <= 0 || heads <= 0 || headDim <= 0 {
		return fmt.Errorf("invalid GPU FP8 E4M3 QKV attention/O buffer inputs")
	}
	emb := heads * headDim
	outLen := tokens * emb
	qkvLen := tokens * 3 * emb
	scoreLen := heads * tokens * tokens
	bufs, unlock, err := ideogramScratchBuffers(qkvLen, outLen, outLen, outLen, outLen, scoreLen, scoreLen, outLen)
	if err != nil {
		return err
	}
	defer unlock()
	qkvBuf, qBuf, kBuf, vBuf, attnBuf, scoreBuf, probBuf, oprojBuf := bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], bufs[5], bufs[6], bufs[7]
	if err := GemmFP8E4M3Buffer(qkvBuf, xBuf, tokens, wqkv); err != nil {
		return fmt.Errorf("FP8 QKV/O buffer QKV: %w", err)
	}
	if err := IdeogramSplitQKVBuffer(qkvBuf, qBuf, kBuf, vBuf, tokens, emb); err != nil {
		return err
	}
	if err := IdeogramRMSNormRowsBuffer(qBuf, qBuf, normQBuf, nil, tokens*heads, headDim, 1e-5, false); err != nil {
		return fmt.Errorf("FP8 QKV/O buffer normQ: %w", err)
	}
	if err := IdeogramRMSNormRowsBuffer(kBuf, kBuf, normKBuf, nil, tokens*heads, headDim, 1e-5, false); err != nil {
		return fmt.Errorf("FP8 QKV/O buffer normK: %w", err)
	}
	if err := IdeogramMRoPEBuffer(qBuf, cosBuf, sinBuf, tokens, heads, headDim); err != nil {
		return fmt.Errorf("FP8 QKV/O buffer mropeQ: %w", err)
	}
	if err := IdeogramMRoPEBuffer(kBuf, cosBuf, sinBuf, tokens, heads, headDim); err != nil {
		return fmt.Errorf("FP8 QKV/O buffer mropeK: %w", err)
	}
	if err := IdeogramAttentionScoresBuffer(scoreBuf, qBuf, kBuf, tokens, heads, headDim, scale); err != nil {
		return err
	}
	if err := SoftmaxRowsBuffer(probBuf, scoreBuf, heads*tokens, tokens); err != nil {
		return err
	}
	if err := IdeogramAttentionValuesBuffer(attnBuf, probBuf, vBuf, tokens, heads, headDim); err != nil {
		return err
	}
	if err := GemmFP8E4M3Buffer(oprojBuf, attnBuf, tokens, wo); err != nil {
		return fmt.Errorf("FP8 QKV/O buffer O: %w", err)
	}
	if err := IdeogramRMSNormRowsBuffer(oprojBuf, oprojBuf, normOutBuf, nil, tokens, emb, 1e-5, false); err != nil {
		return fmt.Errorf("FP8 QKV/O buffer post norm: %w", err)
	}
	if err := IdeogramGatedResidualRowsBuffer(hiddenBuf, oprojBuf, gateBuf, tokens, emb); err != nil {
		return fmt.Errorf("FP8 QKV/O buffer residual: %w", err)
	}
	_ = outLen
	return nil
}

func GemmQKVAttentionOResidualFP8E4M3(hidden, x, gate, normQ, normK, normOut, cos, sin []float32, tokens, heads, headDim int, scale float32, wqkv, wo *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(wqkv) || !validGPUFP8E4M3Linear(wo) || tokens <= 0 || heads <= 0 || headDim <= 0 {
		return fmt.Errorf("invalid GPU FP8 E4M3 QKV attention/O inputs")
	}
	emb := heads * headDim
	inLen := tokens * wqkv.InDim
	outLen := tokens * emb
	qkvLen := tokens * 3 * emb
	scoreLen := heads * tokens * tokens
	tableLen := tokens * (headDim / 2)
	if wqkv.OutDim != 3*emb || wo.InDim != emb || wo.OutDim != emb || len(x) < inLen || len(hidden) < outLen || len(gate) < emb || len(normQ) < headDim || len(normK) < headDim || len(normOut) < emb || len(cos) < tableLen || len(sin) < tableLen {
		return fmt.Errorf("invalid FP8 QKV attention/O buffers")
	}
	bufs, unlock, err := ideogramScratchBuffers(inLen, qkvLen, outLen, outLen, outLen, outLen, scoreLen, scoreLen, emb, headDim, headDim, tableLen, tableLen, outLen, outLen, emb)
	if err != nil {
		return err
	}
	defer unlock()
	xBuf, qkvBuf, qBuf, kBuf, vBuf, attnBuf, scoreBuf, probBuf, normOutBuf, nqBuf, nkBuf, cosBuf, sinBuf, oprojBuf, hiddenBuf, gateBuf := bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], bufs[5], bufs[6], bufs[7], bufs[8], bufs[9], bufs[10], bufs[11], bufs[12], bufs[13], bufs[14], bufs[15]
	if err := xBuf.Upload(x[:inLen]); err != nil {
		return fmt.Errorf("upload FP8 QKV/O input: %w", err)
	}
	if err := hiddenBuf.Upload(hidden[:outLen]); err != nil {
		return fmt.Errorf("upload FP8 QKV/O hidden: %w", err)
	}
	if err := gateBuf.Upload(gate[:emb]); err != nil {
		return fmt.Errorf("upload FP8 QKV/O gate: %w", err)
	}
	if err := normOutBuf.Upload(normOut[:emb]); err != nil {
		return fmt.Errorf("upload FP8 QKV/O post norm: %w", err)
	}
	if err := nqBuf.Upload(normQ[:headDim]); err != nil {
		return fmt.Errorf("upload FP8 QKV/O normQ: %w", err)
	}
	if err := nkBuf.Upload(normK[:headDim]); err != nil {
		return fmt.Errorf("upload FP8 QKV/O normK: %w", err)
	}
	if err := cosBuf.Upload(cos[:tableLen]); err != nil {
		return fmt.Errorf("upload FP8 QKV/O cos: %w", err)
	}
	if err := sinBuf.Upload(sin[:tableLen]); err != nil {
		return fmt.Errorf("upload FP8 QKV/O sin: %w", err)
	}
	if err := GemmFP8E4M3Buffer(qkvBuf, xBuf, tokens, wqkv); err != nil {
		return fmt.Errorf("FP8 QKV/O QKV: %w", err)
	}
	if err := IdeogramSplitQKVBuffer(qkvBuf, qBuf, kBuf, vBuf, tokens, emb); err != nil {
		return err
	}
	if err := IdeogramRMSNormRowsBuffer(qBuf, qBuf, nqBuf, nil, tokens*heads, headDim, 1e-5, false); err != nil {
		return fmt.Errorf("FP8 QKV/O normQ: %w", err)
	}
	if err := IdeogramRMSNormRowsBuffer(kBuf, kBuf, nkBuf, nil, tokens*heads, headDim, 1e-5, false); err != nil {
		return fmt.Errorf("FP8 QKV/O normK: %w", err)
	}
	if err := IdeogramMRoPEBuffer(qBuf, cosBuf, sinBuf, tokens, heads, headDim); err != nil {
		return fmt.Errorf("FP8 QKV/O mropeQ: %w", err)
	}
	if err := IdeogramMRoPEBuffer(kBuf, cosBuf, sinBuf, tokens, heads, headDim); err != nil {
		return fmt.Errorf("FP8 QKV/O mropeK: %w", err)
	}
	if err := IdeogramAttentionScoresBuffer(scoreBuf, qBuf, kBuf, tokens, heads, headDim, scale); err != nil {
		return err
	}
	if err := SoftmaxRowsBuffer(probBuf, scoreBuf, heads*tokens, tokens); err != nil {
		return err
	}
	if err := IdeogramAttentionValuesBuffer(attnBuf, probBuf, vBuf, tokens, heads, headDim); err != nil {
		return err
	}
	if err := GemmFP8E4M3Buffer(oprojBuf, attnBuf, tokens, wo); err != nil {
		return fmt.Errorf("FP8 QKV/O O: %w", err)
	}
	if err := IdeogramRMSNormRowsBuffer(oprojBuf, oprojBuf, normOutBuf, nil, tokens, emb, 1e-5, false); err != nil {
		return fmt.Errorf("FP8 QKV/O post norm: %w", err)
	}
	if err := IdeogramGatedResidualRowsBuffer(hiddenBuf, oprojBuf, gateBuf, tokens, emb); err != nil {
		return fmt.Errorf("FP8 QKV/O residual: %w", err)
	}
	if err := hiddenBuf.Download(hidden[:outLen]); err != nil {
		return fmt.Errorf("download FP8 QKV/O hidden: %w", err)
	}
	return nil
}

func GemmQKVAttentionFP8E4M3(out, x, normQ, normK, cos, sin []float32, tokens, heads, headDim int, scale float32, wqkv *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(wqkv) || tokens <= 0 || heads <= 0 || headDim <= 0 {
		return fmt.Errorf("invalid GPU FP8 E4M3 QKV attention inputs")
	}
	emb := heads * headDim
	inLen := tokens * wqkv.InDim
	outLen := tokens * emb
	qkvLen := tokens * 3 * emb
	scoreLen := heads * tokens * tokens
	tableLen := tokens * (headDim / 2)
	if wqkv.OutDim != 3*emb || len(x) < inLen || len(out) < outLen || len(normQ) < headDim || len(normK) < headDim || len(cos) < tableLen || len(sin) < tableLen {
		return fmt.Errorf("invalid FP8 QKV attention buffers")
	}
	bufs, unlock, err := ideogramScratchBuffers(inLen, qkvLen, outLen, outLen, outLen, outLen, scoreLen, scoreLen, headDim, headDim, tableLen, tableLen)
	if err != nil {
		return err
	}
	defer unlock()
	xBuf, qkvBuf, qBuf, kBuf, vBuf, outBuf, scoreBuf, probBuf, nqBuf, nkBuf, cosBuf, sinBuf := bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], bufs[5], bufs[6], bufs[7], bufs[8], bufs[9], bufs[10], bufs[11]
	if err := xBuf.Upload(x[:inLen]); err != nil {
		return fmt.Errorf("upload FP8 QKV attention input: %w", err)
	}
	if err := nqBuf.Upload(normQ[:headDim]); err != nil {
		return fmt.Errorf("upload FP8 QKV normQ: %w", err)
	}
	if err := nkBuf.Upload(normK[:headDim]); err != nil {
		return fmt.Errorf("upload FP8 QKV normK: %w", err)
	}
	if err := cosBuf.Upload(cos[:tableLen]); err != nil {
		return fmt.Errorf("upload FP8 QKV cos: %w", err)
	}
	if err := sinBuf.Upload(sin[:tableLen]); err != nil {
		return fmt.Errorf("upload FP8 QKV sin: %w", err)
	}
	if err := GemmFP8E4M3Buffer(qkvBuf, xBuf, tokens, wqkv); err != nil {
		return fmt.Errorf("FP8 QKV attention QKV: %w", err)
	}
	if err := IdeogramSplitQKVBuffer(qkvBuf, qBuf, kBuf, vBuf, tokens, emb); err != nil {
		return err
	}
	if err := IdeogramRMSNormRowsBuffer(qBuf, qBuf, nqBuf, nil, tokens*heads, headDim, 1e-5, false); err != nil {
		return fmt.Errorf("FP8 QKV attention normQ: %w", err)
	}
	if err := IdeogramRMSNormRowsBuffer(kBuf, kBuf, nkBuf, nil, tokens*heads, headDim, 1e-5, false); err != nil {
		return fmt.Errorf("FP8 QKV attention normK: %w", err)
	}
	if err := IdeogramMRoPEBuffer(qBuf, cosBuf, sinBuf, tokens, heads, headDim); err != nil {
		return fmt.Errorf("FP8 QKV attention mropeQ: %w", err)
	}
	if err := IdeogramMRoPEBuffer(kBuf, cosBuf, sinBuf, tokens, heads, headDim); err != nil {
		return fmt.Errorf("FP8 QKV attention mropeK: %w", err)
	}
	if err := IdeogramAttentionScoresBuffer(scoreBuf, qBuf, kBuf, tokens, heads, headDim, scale); err != nil {
		return err
	}
	if err := SoftmaxRowsBuffer(probBuf, scoreBuf, heads*tokens, tokens); err != nil {
		return err
	}
	if err := IdeogramAttentionValuesBuffer(outBuf, probBuf, vBuf, tokens, heads, headDim); err != nil {
		return err
	}
	if err := outBuf.Download(out[:outLen]); err != nil {
		return fmt.Errorf("download FP8 QKV attention output: %w", err)
	}
	return nil
}

func GemmSwiGLUResidualFP8E4M3Buffer(hiddenBuf, xBuf, gateBuf, normOutBuf *Buffer, batch int, w1, w3, w2 *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(w1) || !validGPUFP8E4M3Linear(w3) || !validGPUFP8E4M3Linear(w2) {
		return fmt.Errorf("invalid GPU FP8 E4M3 SwiGLU residual buffer linear set")
	}
	if batch <= 0 || w1.InDim != w3.InDim || w1.OutDim != w3.OutDim || w2.InDim != w1.OutDim {
		return fmt.Errorf("invalid FP8 E4M3 SwiGLU residual buffer dims")
	}
	interLen := batch * w1.OutDim
	outLen := batch * w2.OutDim
	bufs, unlock, err := ideogramScratchBuffers(interLen, interLen, outLen)
	if err != nil {
		return err
	}
	defer unlock()
	gBuf, uBuf, downBuf := bufs[0], bufs[1], bufs[2]
	if err := GemmFP8E4M3Buffer(gBuf, xBuf, batch, w1); err != nil {
		return fmt.Errorf("FP8 SwiGLU residual buffer W1: %w", err)
	}
	if err := GemmFP8E4M3Buffer(uBuf, xBuf, batch, w3); err != nil {
		return fmt.Errorf("FP8 SwiGLU residual buffer W3: %w", err)
	}
	if err := F32SiLUMulBuffer(gBuf, gBuf, uBuf, interLen); err != nil {
		return fmt.Errorf("FP8 SwiGLU residual buffer SiLU*Mul: %w", err)
	}
	if err := GemmFP8E4M3Buffer(downBuf, gBuf, batch, w2); err != nil {
		return fmt.Errorf("FP8 SwiGLU residual buffer W2: %w", err)
	}
	if err := IdeogramRMSNormRowsBuffer(downBuf, downBuf, normOutBuf, nil, batch, w2.OutDim, 1e-5, false); err != nil {
		return fmt.Errorf("FP8 SwiGLU residual buffer post norm: %w", err)
	}
	if err := IdeogramGatedResidualRowsBuffer(hiddenBuf, downBuf, gateBuf, batch, w2.OutDim); err != nil {
		return fmt.Errorf("FP8 SwiGLU residual buffer gated: %w", err)
	}
	_ = outLen
	return nil
}

func GemmSwiGLUResidualFP8E4M3(hidden, x, gate, normOut []float32, batch int, w1, w3, w2 *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(w1) || !validGPUFP8E4M3Linear(w3) || !validGPUFP8E4M3Linear(w2) {
		return fmt.Errorf("invalid GPU FP8 E4M3 SwiGLU residual linear set")
	}
	if batch <= 0 || w1.InDim != w3.InDim || w1.OutDim != w3.OutDim || w2.InDim != w1.OutDim {
		return fmt.Errorf("invalid FP8 E4M3 SwiGLU residual dims batch=%d w1=%dx%d w3=%dx%d w2=%dx%d", batch, w1.OutDim, w1.InDim, w3.OutDim, w3.InDim, w2.OutDim, w2.InDim)
	}
	inLen := batch * w1.InDim
	interLen := batch * w1.OutDim
	outLen := batch * w2.OutDim
	if len(x) < inLen || len(hidden) < outLen || len(gate) < w2.OutDim || len(normOut) < w2.OutDim {
		return fmt.Errorf("invalid FP8 E4M3 SwiGLU residual buffers")
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	bufs, unlock, err := ideogramScratchBuffers(inLen, interLen, interLen, outLen, outLen, w2.OutDim, w2.OutDim)
	if err != nil {
		return err
	}
	defer unlock()
	xBuf, gBuf, uBuf, downBuf, hiddenBuf, gateBuf, normBuf := bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], bufs[5], bufs[6]
	if err := xBuf.Upload(x[:inLen]); err != nil {
		return fmt.Errorf("upload FP8 SwiGLU residual input: %w", err)
	}
	if err := hiddenBuf.Upload(hidden[:outLen]); err != nil {
		return fmt.Errorf("upload FP8 SwiGLU residual hidden: %w", err)
	}
	if err := gateBuf.Upload(gate[:w2.OutDim]); err != nil {
		return fmt.Errorf("upload FP8 SwiGLU residual gate: %w", err)
	}
	if err := normBuf.Upload(normOut[:w2.OutDim]); err != nil {
		return fmt.Errorf("upload FP8 SwiGLU residual norm: %w", err)
	}
	if err := GemmFP8E4M3Buffer(gBuf, xBuf, batch, w1); err != nil {
		return fmt.Errorf("FP8 SwiGLU residual W1: %w", err)
	}
	if err := GemmFP8E4M3Buffer(uBuf, xBuf, batch, w3); err != nil {
		return fmt.Errorf("FP8 SwiGLU residual W3: %w", err)
	}
	if err := F32SiLUMulBuffer(gBuf, gBuf, uBuf, interLen); err != nil {
		return fmt.Errorf("FP8 SwiGLU residual SiLU*Mul: %w", err)
	}
	if err := GemmFP8E4M3Buffer(downBuf, gBuf, batch, w2); err != nil {
		return fmt.Errorf("FP8 SwiGLU residual W2: %w", err)
	}
	if err := IdeogramRMSNormRowsBuffer(downBuf, downBuf, normBuf, nil, batch, w2.OutDim, 1e-5, false); err != nil {
		return fmt.Errorf("FP8 SwiGLU residual post norm: %w", err)
	}
	if err := IdeogramGatedResidualRowsBuffer(hiddenBuf, downBuf, gateBuf, batch, w2.OutDim); err != nil {
		return fmt.Errorf("FP8 SwiGLU residual gated: %w", err)
	}
	if err := hiddenBuf.Download(hidden[:outLen]); err != nil {
		return fmt.Errorf("download FP8 SwiGLU residual hidden: %w", err)
	}
	return nil
}

func GemmSwiGLUFP8E4M3(out, x []float32, batch int, w1, w3, w2 *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(w1) || !validGPUFP8E4M3Linear(w3) || !validGPUFP8E4M3Linear(w2) {
		return fmt.Errorf("invalid GPU FP8 E4M3 SwiGLU linear set")
	}
	if batch <= 0 || w1.InDim != w3.InDim || w1.OutDim != w3.OutDim || w2.InDim != w1.OutDim {
		return fmt.Errorf("invalid FP8 E4M3 SwiGLU dims batch=%d w1=%dx%d w3=%dx%d w2=%dx%d", batch, w1.OutDim, w1.InDim, w3.OutDim, w3.InDim, w2.OutDim, w2.InDim)
	}
	inLen := batch * w1.InDim
	interLen := batch * w1.OutDim
	outLen := batch * w2.OutDim
	if len(x) < inLen || len(out) < outLen {
		return fmt.Errorf("invalid FP8 E4M3 SwiGLU buffers x=%d/%d out=%d/%d", len(x), inLen, len(out), outLen)
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	bufs, unlock, err := ideogramScratchBuffers(inLen, interLen, interLen, outLen)
	if err != nil {
		return err
	}
	defer unlock()
	xBuf, gBuf, uBuf, outBuf := bufs[0], bufs[1], bufs[2], bufs[3]
	if err := xBuf.Upload(x[:inLen]); err != nil {
		return fmt.Errorf("upload FP8 SwiGLU input: %w", err)
	}
	if err := GemmFP8E4M3Buffer(gBuf, xBuf, batch, w1); err != nil {
		return fmt.Errorf("FP8 SwiGLU W1: %w", err)
	}
	if err := GemmFP8E4M3Buffer(uBuf, xBuf, batch, w3); err != nil {
		return fmt.Errorf("FP8 SwiGLU W3: %w", err)
	}
	if err := F32SiLUMulBuffer(gBuf, gBuf, uBuf, interLen); err != nil {
		return fmt.Errorf("FP8 SwiGLU SiLU*Mul: %w", err)
	}
	if err := GemmFP8E4M3Buffer(outBuf, gBuf, batch, w2); err != nil {
		return fmt.Errorf("FP8 SwiGLU W2: %w", err)
	}
	if err := outBuf.Download(out[:outLen]); err != nil {
		return fmt.Errorf("download FP8 SwiGLU output: %w", err)
	}
	return nil
}

func Gemm2FP8E4M3SameInput(outA, outB, x []float32, batch int, wA, wB *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(wA) || !validGPUFP8E4M3Linear(wB) {
		return fmt.Errorf("invalid GPU FP8 E4M3 linear pair")
	}
	if batch <= 0 {
		return fmt.Errorf("invalid FP8 E4M3 GEMM2 batch=%d", batch)
	}
	if wA.InDim != wB.InDim {
		return fmt.Errorf("FP8 E4M3 GEMM2 input dim mismatch %d != %d", wA.InDim, wB.InDim)
	}
	inLen, ok := checkedMulInt(batch, wA.InDim)
	if !ok {
		return fmt.Errorf("FP8 E4M3 GEMM2 input size overflow batch=%d in=%d", batch, wA.InDim)
	}
	outALen, ok := checkedMulInt(batch, wA.OutDim)
	if !ok {
		return fmt.Errorf("FP8 E4M3 GEMM2 output A size overflow")
	}
	outBLen, ok := checkedMulInt(batch, wB.OutDim)
	if !ok {
		return fmt.Errorf("FP8 E4M3 GEMM2 output B size overflow")
	}
	if len(x) < inLen || len(outA) < outALen || len(outB) < outBLen {
		return fmt.Errorf("invalid FP8 E4M3 GEMM2 buffers x=%d/%d outA=%d/%d outB=%d/%d", len(x), inLen, len(outA), outALen, len(outB), outBLen)
	}
	if !SgemmReady() {
		linA, err := downloadFP8E4M3Linear(wA)
		if err != nil {
			return err
		}
		linB, err := downloadFP8E4M3Linear(wB)
		if err != nil {
			return err
		}
		for b := 0; b < batch; b++ {
			if err := linA.GemvTo(x[b*wA.InDim:(b+1)*wA.InDim], outA[b*wA.OutDim:(b+1)*wA.OutDim]); err != nil {
				return err
			}
			if err := linB.GemvTo(x[b*wB.InDim:(b+1)*wB.InDim], outB[b*wB.OutDim:(b+1)*wB.OutDim]); err != nil {
				return err
			}
		}
		return nil
	}
	xBuf, outABuf, unlock, err := fp8ScratchBuffers(inLen, outALen)
	if err != nil {
		return err
	}
	defer unlock()
	outBBuf, err := Malloc(outBLen)
	if err != nil {
		return fmt.Errorf("alloc FP8 E4M3 GEMM2 output B: %w", err)
	}
	defer outBBuf.Free()
	if err := xBuf.Upload(x[:inLen]); err != nil {
		return fmt.Errorf("upload FP8 E4M3 GEMM2 input: %w", err)
	}
	if err := GemmFP8E4M3Buffer(outABuf, xBuf, batch, wA); err != nil {
		return fmt.Errorf("GEMM2 A: %w", err)
	}
	if err := GemmFP8E4M3Buffer(outBBuf, xBuf, batch, wB); err != nil {
		return fmt.Errorf("GEMM2 B: %w", err)
	}
	if err := outABuf.Download(outA[:outALen]); err != nil {
		return fmt.Errorf("download FP8 E4M3 GEMM2 output A: %w", err)
	}
	if err := outBBuf.Download(outB[:outBLen]); err != nil {
		return fmt.Errorf("download FP8 E4M3 GEMM2 output B: %w", err)
	}
	return nil
}

func fp8SgemmEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_NVIDIA_FP8_SGEMM")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func gemmFP8E4M3ViaSgemm(out, x []float32, batch int, w *GPUFP8E4M3Linear) error {
	if fnFP8E4M3DequantTransposeF32 == 0 || !megaModuleOK {
		return fmt.Errorf("FP8 E4M3 dequant-transpose kernel not available")
	}
	inLen := batch * w.InDim
	outLen := batch * w.OutDim
	weightLen := w.InDim * w.OutDim
	xBuf, err := Malloc(inLen)
	if err != nil {
		return fmt.Errorf("alloc FP8 SGEMM input: %w", err)
	}
	defer xBuf.Free()
	wtBuf, err := Malloc(weightLen)
	if err != nil {
		return fmt.Errorf("alloc FP8 SGEMM dequant weight: %w", err)
	}
	defer wtBuf.Free()
	outBuf, err := Malloc(outLen)
	if err != nil {
		return fmt.Errorf("alloc FP8 SGEMM output: %w", err)
	}
	defer outBuf.Free()
	if err := xBuf.Upload(x[:inLen]); err != nil {
		return fmt.Errorf("upload FP8 SGEMM input: %w", err)
	}
	if err := dequantTransposeFP8E4M3(wtBuf, w); err != nil {
		return err
	}
	if err := Sgemm(batch, w.OutDim, w.InDim, 1, xBuf, wtBuf, outBuf); err != nil {
		return fmt.Errorf("FP8 SGEMM: %w", err)
	}
	if err := outBuf.Download(out[:outLen]); err != nil {
		return fmt.Errorf("download FP8 SGEMM output: %w", err)
	}
	return nil
}

func dequantTransposeFP8E4M3(dst *Buffer, w *GPUFP8E4M3Linear) error {
	if !validGPUFP8E4M3Linear(w) || dst == nil {
		return fmt.Errorf("invalid FP8 dequant-transpose input")
	}
	weightLen := w.InDim * w.OutDim
	if _, err := checkedByteSize(weightLen, dst.Size); err != nil {
		return fmt.Errorf("invalid FP8 dequant-transpose output: %w", err)
	}
	if !fitsUint32(w.OutDim) || !fitsUint32(w.InDim) || !fitsUint32(w.ScaleLen) {
		return fmt.Errorf("FP8 dequant-transpose dims exceed CUDA u32 interface")
	}
	outDim := uint32(w.OutDim)
	inDim := uint32(w.InDim)
	scaleLen := uint32(w.ScaleLen)
	blocks := uint32((weightLen + 255) / 256)
	return LaunchKernel(fnFP8E4M3DequantTransposeF32, blocks, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&w.Weight.Ptr),
		unsafe.Pointer(&w.Scale.Ptr),
		unsafe.Pointer(&dst.Ptr),
		unsafe.Pointer(&outDim),
		unsafe.Pointer(&inDim),
		unsafe.Pointer(&scaleLen))
}

func gemmFP8E4M3CUDA(out, x []float32, batch int, w *GPUFP8E4M3Linear) error {
	inLen := batch * w.InDim
	outLen := batch * w.OutDim
	xBuf, outBuf, unlock, err := fp8ScratchBuffers(inLen, outLen)
	if err != nil {
		return err
	}
	defer unlock()
	if err := xBuf.Upload(x[:inLen]); err != nil {
		return fmt.Errorf("upload FP8 E4M3 GEMM input: %w", err)
	}
	if err := GemmFP8E4M3Buffer(outBuf, xBuf, batch, w); err != nil {
		return err
	}
	if err := outBuf.Download(out[:outLen]); err != nil {
		return fmt.Errorf("download FP8 E4M3 GEMM output: %w", err)
	}
	return nil
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
