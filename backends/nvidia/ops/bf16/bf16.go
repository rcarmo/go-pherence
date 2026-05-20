package bf16

// BF16 CUDA PTX kernels for sm_80+ (Ampere and newer).
//
// On Ampere+, BF16 is natively supported:
//   - ld.global.b16 / st.global.b16 for BF16 load/store
//   - cvt.f32.bf16 / cvt.bf16.f32 for conversion
//   - FMA in F32 with BF16 inputs
//
// These kernels accept BF16 (uint16) buffers directly,
// halving memory bandwidth vs F32 kernels.

import (
	"sync"
	"unsafe"
)

var fnBF16RMSNorm CUfunction
var fnBF16RMSNormNoScale CUfunction
var fnBF16VecAdd CUfunction
var fnBF16SiLUMul CUfunction
var fnBF16GELUTanhMul CUfunction
var fnBF16Gemv CUfunction
var fnBF16LMHead CUfunction

var bf16LMHeadScratch = struct {
	sync.Mutex
	x    *Buffer
	out  *Buffer
	xN   int
	outN int
}{}

// DevBF16RMSNorm applies RMSNorm on BF16 data: x[i] = BF16(F32(x[i]) * invRMS * F32(w[i]))
func DevBF16RMSNorm(x, w *Buffer, n int, eps float32) bool {
	if fnBF16RMSNorm == 0 || !validBF16Buffer(x, n) || !validBF16Buffer(w, n) {
		return false
	}
	EnsureContext()
	nn := uint32(n)
	return LaunchKernel(fnBF16RMSNorm, 1, 1, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&x.Ptr),
		unsafe.Pointer(&w.Ptr),
		unsafe.Pointer(&nn),
		unsafe.Pointer(&eps)) == nil
}

// DevBF16RMSNormNoScale applies RMSNormNoScale on BF16 data: x[i] = BF16(F32(x[i]) * invRMS).
func DevBF16RMSNormNoScale(x *Buffer, n int, eps float32) bool {
	if fnBF16RMSNormNoScale == 0 || !validBF16Buffer(x, n) {
		return false
	}
	EnsureContext()
	nn := uint32(n)
	return LaunchKernel(fnBF16RMSNormNoScale, 1, 1, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&x.Ptr),
		unsafe.Pointer(&nn),
		unsafe.Pointer(&eps)) == nil
}

// DevBF16VecAdd computes dst[i] = BF16(F32(a[i]) + F32(b[i]))
func DevBF16VecAdd(dst, a, b *Buffer, n int) bool {
	if fnBF16VecAdd == 0 || !validBF16Buffer(dst, n) || !validBF16Buffer(a, n) || !validBF16Buffer(b, n) {
		return false
	}
	EnsureContext()
	grid, okGrid := grid1DFor(n, 256)
	if !okGrid {
		return false
	}
	nn := uint32(n)
	return LaunchKernel(fnBF16VecAdd, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&a.Ptr),
		unsafe.Pointer(&b.Ptr),
		unsafe.Pointer(&dst.Ptr),
		unsafe.Pointer(&nn)) == nil
}

// DevBF16SiLUMul computes dst[i] = BF16(SiLU(F32(gate[i])) * F32(up[i])).
func DevBF16SiLUMul(dst, gate, up *Buffer, n int) bool {
	if fnBF16SiLUMul == 0 || !validBF16Buffer(dst, n) || !validBF16Buffer(gate, n) || !validBF16Buffer(up, n) {
		return false
	}
	EnsureContext()
	grid, okGrid := grid1DFor(n, 256)
	if !okGrid {
		return false
	}
	nn := uint32(n)
	return LaunchKernel(fnBF16SiLUMul, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&gate.Ptr),
		unsafe.Pointer(&up.Ptr),
		unsafe.Pointer(&dst.Ptr),
		unsafe.Pointer(&nn)) == nil
}

// DevBF16GELUTanhMul computes gate[i] = BF16(GELUTanh(F32(gate[i])) * F32(up[i])) in-place.
func DevBF16GELUTanhMul(gate, up *Buffer, n int) bool {
	if fnBF16GELUTanhMul == 0 || !validBF16Buffer(gate, n) || !validBF16Buffer(up, n) {
		return false
	}
	EnsureContext()
	grid, okGrid := grid1DFor(n, 256)
	if !okGrid {
		return false
	}
	nn := uint32(n)
	return LaunchKernel(fnBF16GELUTanhMul, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&gate.Ptr),
		unsafe.Pointer(&up.Ptr),
		unsafe.Pointer(&nn)) == nil
}

func UploadBF16LMHead(weightRaw []byte, vocab, h int) (*Buffer, error) {
	if !SgemmReady() {
		return nil, nil
	}
	weightBytes, ok := checkedBF16MatrixBytes(vocab, h)
	if !ok || len(weightRaw) < weightBytes {
		return nil, nil
	}
	wBuf, err := Malloc(f32SlotsForBytes(weightBytes))
	if err != nil {
		return nil, err
	}
	if err := wBuf.UploadBytes(weightRaw[:weightBytes]); err != nil {
		wBuf.Free()
		return nil, err
	}
	return wBuf, nil
}

func BF16LMHeadWithBuffer(logits []float32, wBuf *Buffer, x []float32, vocab, h int) error {
	if fnBF16LMHead == 0 || !SgemmReady() {
		return nil
	}
	weightBytes, ok := checkedBF16MatrixBytes(vocab, h)
	if !ok || wBuf == nil || !hasPaddedByteCapacity(wBuf.Size, weightBytes) || len(x) < h || len(logits) < vocab || !fitsUint32(vocab) || !fitsUint32(h) {
		return nil
	}
	xBuf, outBuf, unlock, err := bf16LMHeadScratchBuffers(h, vocab)
	if err != nil {
		return err
	}
	defer unlock()
	if err := xBuf.Upload(x[:h]); err != nil {
		return err
	}
	vv := uint32(vocab)
	hh := uint32(h)
	gridX := uint32(vocab)
	gridY := uint32(1)
	if vocab > 65535 {
		gridX = 65535
		gridY = uint32((vocab + 65534) / 65535)
	}
	if err := LaunchKernel(fnBF16LMHead, gridX, gridY, 1, 128, 1, 1, 128*4,
		unsafe.Pointer(&wBuf.Ptr), unsafe.Pointer(&xBuf.Ptr), unsafe.Pointer(&outBuf.Ptr), unsafe.Pointer(&vv), unsafe.Pointer(&hh)); err != nil {
		return err
	}
	return outBuf.Download(logits[:vocab])
}

func bf16LMHeadScratchBuffers(h, vocab int) (*Buffer, *Buffer, func(), error) {
	bf16LMHeadScratch.Lock()
	unlock := func() { bf16LMHeadScratch.Unlock() }
	if bf16LMHeadScratch.x == nil || bf16LMHeadScratch.xN < h {
		if bf16LMHeadScratch.x != nil {
			bf16LMHeadScratch.x.Free()
		}
		buf, err := Malloc(h)
		if err != nil {
			unlock()
			return nil, nil, nil, err
		}
		bf16LMHeadScratch.x = buf
		bf16LMHeadScratch.xN = h
	}
	if bf16LMHeadScratch.out == nil || bf16LMHeadScratch.outN < vocab {
		if bf16LMHeadScratch.out != nil {
			bf16LMHeadScratch.out.Free()
		}
		buf, err := Malloc(vocab)
		if err != nil {
			unlock()
			return nil, nil, nil, err
		}
		bf16LMHeadScratch.out = buf
		bf16LMHeadScratch.outN = vocab
	}
	return bf16LMHeadScratch.x, bf16LMHeadScratch.out, unlock, nil
}

func FreeBF16LMHeadScratch() {
	bf16LMHeadScratch.Lock()
	defer bf16LMHeadScratch.Unlock()
	if bf16LMHeadScratch.x != nil {
		bf16LMHeadScratch.x.Free()
		bf16LMHeadScratch.x = nil
	}
	if bf16LMHeadScratch.out != nil {
		bf16LMHeadScratch.out.Free()
		bf16LMHeadScratch.out = nil
	}
	bf16LMHeadScratch.xN = 0
	bf16LMHeadScratch.outN = 0
}

func BF16LMHead(logits []float32, weightRaw []byte, x []float32, vocab, h int) error {
	wBuf, err := UploadBF16LMHead(weightRaw, vocab, h)
	if err != nil || wBuf == nil {
		return err
	}
	defer wBuf.Free()
	return BF16LMHeadWithBuffer(logits, wBuf, x, vocab, h)
}

func checkedBF16MatrixBytes(rows, cols int) (int, bool) {
	elements, ok := checkedMulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if elements > maxInt/2 {
		return 0, false
	}
	return elements * 2, true
}

func validBF16Buffer(b *Buffer, n int) bool {
	if b == nil || b.Ptr == 0 || n <= 0 || !fitsUint32(n) {
		return false
	}
	maxInt := int(^uint(0) >> 1)
	if n > maxInt/2 {
		return false
	}
	return b.Size >= n*2
}
