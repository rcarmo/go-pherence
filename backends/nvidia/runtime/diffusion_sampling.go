package nvidia

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/nvidia/ptx"
	"github.com/rcarmo/go-pherence/internal/checked"
)

var diffusionSampleState = struct {
	sync.Mutex
	fn        CUfunction
	softmaxFn CUfunction
}{}

func ensureDiffusionSoftmaxRows() error {
	if diffusionSampleState.softmaxFn != 0 {
		return nil
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU runtime not ready")
	}
	fn, err := LoadPTX(ptx.DiffusionSoftmaxRowsPTX, "diffusion_softmax_rows_f32")
	if err != nil {
		return err
	}
	diffusionSampleState.softmaxFn = fn
	return nil
}

func ensureDiffusionDenseSample() error {
	if diffusionSampleState.fn != 0 {
		return nil
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU runtime not ready")
	}
	fn, err := LoadPTX(ptx.DiffusionDenseSamplePTX, "diffusion_dense_sample_f32")
	if err != nil {
		return err
	}
	diffusionSampleState.fn = fn
	return nil
}

func DiffusionSoftmaxRows(logits, probs *Buffer, positions, vocab int, invTemp float32) error {
	if logits == nil || probs == nil || positions <= 0 || vocab <= 0 || !fitsUint32(positions) || !fitsUint32(vocab) {
		return fmt.Errorf("invalid diffusion softmax inputs positions=%d vocab=%d", positions, vocab)
	}
	total, ok := checked.MulInt(positions, vocab)
	if !ok {
		return fmt.Errorf("diffusion softmax size overflow positions=%d vocab=%d", positions, vocab)
	}
	if _, err := checkedByteSize(total, logits.Size); err != nil {
		return fmt.Errorf("invalid diffusion softmax logits: %w", err)
	}
	if _, err := checkedByteSize(total, probs.Size); err != nil {
		return fmt.Errorf("invalid diffusion softmax probs: %w", err)
	}
	diffusionSampleState.Lock()
	defer diffusionSampleState.Unlock()
	if err := ensureDiffusionSoftmaxRows(); err != nil {
		return err
	}
	vv := uint32(vocab)
	return LaunchKernel(diffusionSampleState.softmaxFn, uint32(positions), 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&logits.Ptr), unsafe.Pointer(&probs.Ptr), unsafe.Pointer(&vv), unsafe.Pointer(&invTemp))
}

// DiffusionDenseSample reads device-resident row-major logits [positions,vocab]
// and returns per-position argmax, entropy and one multinomial sample. It mirrors
// llama.cpp's CUDA diffusion_dense_sample_kernel and copies back only O(C) data.
func DiffusionDenseSample(logits *Buffer, uniforms []float32, positions, vocab int, invTemp float32) ([]int, []float64, []int, error) {
	if logits == nil || positions <= 0 || vocab <= 0 || len(uniforms) < positions || !fitsUint32(positions) || !fitsUint32(vocab) {
		return nil, nil, nil, fmt.Errorf("invalid diffusion sampler inputs positions=%d vocab=%d uniforms=%d", positions, vocab, len(uniforms))
	}
	logitElems, ok := checked.MulInt(positions, vocab)
	if !ok {
		return nil, nil, nil, fmt.Errorf("diffusion sampler logits size overflow positions=%d vocab=%d", positions, vocab)
	}
	if _, err := checkedByteSize(logitElems, logits.Size); err != nil {
		return nil, nil, nil, fmt.Errorf("invalid diffusion sampler logits buffer: %w", err)
	}
	diffusionSampleState.Lock()
	defer diffusionSampleState.Unlock()
	if err := ensureDiffusionDenseSample(); err != nil {
		return nil, nil, nil, err
	}
	uBuf, err := Malloc(positions)
	if err != nil {
		return nil, nil, nil, err
	}
	defer uBuf.Free()
	argBuf, err := Malloc(positions)
	if err != nil {
		return nil, nil, nil, err
	}
	defer argBuf.Free()
	entBuf, err := Malloc(positions)
	if err != nil {
		return nil, nil, nil, err
	}
	defer entBuf.Free()
	samBuf, err := Malloc(positions)
	if err != nil {
		return nil, nil, nil, err
	}
	defer samBuf.Free()
	if err := uBuf.Upload(uniforms[:positions]); err != nil {
		return nil, nil, nil, err
	}
	vv := uint32(vocab)
	if err := LaunchKernel(diffusionSampleState.fn, uint32(positions), 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&logits.Ptr), unsafe.Pointer(&uBuf.Ptr), unsafe.Pointer(&argBuf.Ptr), unsafe.Pointer(&entBuf.Ptr), unsafe.Pointer(&samBuf.Ptr), unsafe.Pointer(&vv), unsafe.Pointer(&invTemp)); err != nil {
		return nil, nil, nil, err
	}
	argRaw := make([]byte, positions*4)
	samRaw := make([]byte, positions*4)
	ent32 := make([]float32, positions)
	if err := argBuf.DownloadBytes(argRaw); err != nil {
		return nil, nil, nil, err
	}
	if err := samBuf.DownloadBytes(samRaw); err != nil {
		return nil, nil, nil, err
	}
	if err := entBuf.Download(ent32); err != nil {
		return nil, nil, nil, err
	}
	argmax := make([]int, positions)
	sampled := make([]int, positions)
	entropy := make([]float64, positions)
	for i := 0; i < positions; i++ {
		argmax[i] = int(binary.LittleEndian.Uint32(argRaw[i*4 : i*4+4]))
		sampled[i] = int(binary.LittleEndian.Uint32(samRaw[i*4 : i*4+4]))
		entropy[i] = float64(ent32[i])
		if math.IsNaN(entropy[i]) || argmax[i] < 0 || argmax[i] >= vocab || sampled[i] < 0 || sampled[i] >= vocab {
			return nil, nil, nil, fmt.Errorf("invalid diffusion sampler output row=%d argmax=%d sampled=%d entropy=%g", i, argmax[i], sampled[i], entropy[i])
		}
	}
	return argmax, entropy, sampled, nil
}
