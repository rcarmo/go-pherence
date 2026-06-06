package nvidia

import (
	"fmt"
	"unsafe"
)

var fnIdeogramCFGStepF32 CUfunction
var fnIdeogramLayerNormNoAffineF32 CUfunction

// IdeogramCFGStepBuffer fuses asymmetric CFG and FlowMatch Euler update on
// GPU-resident F32 buffers:
//
//	out = latents + sigma * (uncond + guidance * (cond - uncond))
//
// All buffers must hold at least n float32 elements.
func IdeogramCFGStepBuffer(out, latents, cond, uncond *Buffer, n int, guidance, sigma float32) error {
	if fnIdeogramCFGStepF32 == 0 || !megaModuleOK || out == nil || latents == nil || cond == nil || uncond == nil || n <= 0 || !fitsUint32(n) {
		return fmt.Errorf("invalid Ideogram CFG/step device buffers")
	}
	if _, err := checkedByteSize(n, out.Size); err != nil {
		return fmt.Errorf("invalid Ideogram CFG/step output buffer: %w", err)
	}
	if _, err := checkedByteSize(n, latents.Size); err != nil {
		return fmt.Errorf("invalid Ideogram CFG/step latent buffer: %w", err)
	}
	if _, err := checkedByteSize(n, cond.Size); err != nil {
		return fmt.Errorf("invalid Ideogram CFG/step conditional buffer: %w", err)
	}
	if _, err := checkedByteSize(n, uncond.Size); err != nil {
		return fmt.Errorf("invalid Ideogram CFG/step unconditional buffer: %w", err)
	}
	grid, ok := grid1DFor(n, 256)
	if !ok {
		return fmt.Errorf("invalid Ideogram CFG/step grid")
	}
	nn := uint32(n)
	return LaunchKernel(fnIdeogramCFGStepF32, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&latents.Ptr),
		unsafe.Pointer(&cond.Ptr),
		unsafe.Pointer(&uncond.Ptr),
		unsafe.Pointer(&out.Ptr),
		unsafe.Pointer(&guidance),
		unsafe.Pointer(&sigma),
		unsafe.Pointer(&nn))
}

// IdeogramLayerNormNoAffineBuffer computes row-wise non-affine LayerNorm on
// GPU-resident F32 buffers. x/out are row-major [rows, cols].
func IdeogramLayerNormNoAffineBuffer(out, x *Buffer, rows, cols int, eps float32) error {
	if fnIdeogramLayerNormNoAffineF32 == 0 || !megaModuleOK || out == nil || x == nil || rows <= 0 || cols <= 0 || !fitsUint32(rows) || !fitsUint32(cols) {
		return fmt.Errorf("invalid Ideogram LayerNorm device buffers")
	}
	n, ok := checkedMulInt(rows, cols)
	if !ok {
		return fmt.Errorf("Ideogram LayerNorm element count overflow rows=%d cols=%d", rows, cols)
	}
	if _, err := checkedByteSize(n, out.Size); err != nil {
		return fmt.Errorf("invalid Ideogram LayerNorm output buffer: %w", err)
	}
	if _, err := checkedByteSize(n, x.Size); err != nil {
		return fmt.Errorf("invalid Ideogram LayerNorm input buffer: %w", err)
	}
	rr := uint32(rows)
	cc := uint32(cols)
	return LaunchKernel(fnIdeogramLayerNormNoAffineF32, rr, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&x.Ptr),
		unsafe.Pointer(&out.Ptr),
		unsafe.Pointer(&rr),
		unsafe.Pointer(&cc),
		unsafe.Pointer(&eps))
}

// IdeogramLayerNormNoAffine computes row-wise non-affine LayerNorm using
// temporary device buffers. It is intended for correctness validation and early
// model wiring before the full Ideogram graph is GPU-resident.
func IdeogramLayerNormNoAffine(out, x []float32, rows, cols int, eps float32) error {
	n, ok := checkedMulInt(rows, cols)
	if !ok || rows <= 0 || cols <= 0 || len(out) < n || len(x) < n {
		return fmt.Errorf("invalid Ideogram LayerNorm host buffers out=%d/%d x=%d/%d", len(out), n, len(x), n)
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	EnsureContext()
	xBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram LayerNorm input: %w", err)
	}
	defer xBuf.Free()
	outBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram LayerNorm output: %w", err)
	}
	defer outBuf.Free()
	if err := xBuf.Upload(x[:n]); err != nil {
		return fmt.Errorf("upload Ideogram LayerNorm input: %w", err)
	}
	if err := IdeogramLayerNormNoAffineBuffer(outBuf, xBuf, rows, cols, eps); err != nil {
		return err
	}
	if err := outBuf.Download(out[:n]); err != nil {
		return fmt.Errorf("download Ideogram LayerNorm output: %w", err)
	}
	return nil
}

// IdeogramCFGStep computes the fused CFG+Euler update using temporary device
// buffers. It is a convenience/validation wrapper for the current Ideogram
// pipeline, which still returns DiT velocities as CPU slices.
func IdeogramCFGStep(out, latents, cond, uncond []float32, guidance, sigma float32) error {
	if len(out) == 0 || len(latents) != len(out) || len(cond) != len(out) || len(uncond) != len(out) {
		return fmt.Errorf("invalid Ideogram CFG/step host buffers out=%d latents=%d cond=%d uncond=%d", len(out), len(latents), len(cond), len(uncond))
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	EnsureContext()
	n := len(out)
	latBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram CFG latent: %w", err)
	}
	defer latBuf.Free()
	condBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram CFG conditional: %w", err)
	}
	defer condBuf.Free()
	uncondBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram CFG unconditional: %w", err)
	}
	defer uncondBuf.Free()
	outBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc Ideogram CFG output: %w", err)
	}
	defer outBuf.Free()
	if err := latBuf.Upload(latents); err != nil {
		return fmt.Errorf("upload Ideogram CFG latent: %w", err)
	}
	if err := condBuf.Upload(cond); err != nil {
		return fmt.Errorf("upload Ideogram CFG conditional: %w", err)
	}
	if err := uncondBuf.Upload(uncond); err != nil {
		return fmt.Errorf("upload Ideogram CFG unconditional: %w", err)
	}
	if err := IdeogramCFGStepBuffer(outBuf, latBuf, condBuf, uncondBuf, n, guidance, sigma); err != nil {
		return err
	}
	if err := outBuf.Download(out); err != nil {
		return fmt.Errorf("download Ideogram CFG output: %w", err)
	}
	return nil
}

// F32RMSNormBuffer computes out = RMSNorm(x) * weight on GPU-resident F32
// buffers. It exposes the existing NVIDIA RMSNorm kernel for model code that
// uses low-level Buffer rather than DevBuf.
func F32RMSNormBuffer(out, x, weight *Buffer, n int, eps float32) error {
	if fnRmsNorm == 0 || !megaModuleOK || out == nil || x == nil || weight == nil || n <= 0 || !fitsUint32(n) {
		return fmt.Errorf("invalid F32 RMSNorm device buffers")
	}
	if _, err := checkedByteSize(n, out.Size); err != nil {
		return fmt.Errorf("invalid F32 RMSNorm output buffer: %w", err)
	}
	if _, err := checkedByteSize(n, x.Size); err != nil {
		return fmt.Errorf("invalid F32 RMSNorm input buffer: %w", err)
	}
	if _, err := checkedByteSize(n, weight.Size); err != nil {
		return fmt.Errorf("invalid F32 RMSNorm weight buffer: %w", err)
	}
	nn := uint32(n)
	return LaunchKernel(fnRmsNorm, 1, 1, 1, 256, 1, 1, 256*4,
		unsafe.Pointer(&x.Ptr),
		unsafe.Pointer(&weight.Ptr),
		unsafe.Pointer(&out.Ptr),
		unsafe.Pointer(&nn),
		unsafe.Pointer(&eps))
}

// F32RMSNorm computes out = RMSNorm(x) * weight through temporary device
// buffers. It is a correctness/wiring wrapper for early GPU conversion stages.
func F32RMSNorm(out, x, weight []float32, eps float32) error {
	if len(out) == 0 || len(x) != len(out) || len(weight) != len(out) {
		return fmt.Errorf("invalid F32 RMSNorm host buffers out=%d x=%d weight=%d", len(out), len(x), len(weight))
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	EnsureContext()
	n := len(out)
	xBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 RMSNorm input: %w", err)
	}
	defer xBuf.Free()
	wBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 RMSNorm weight: %w", err)
	}
	defer wBuf.Free()
	outBuf, err := Malloc(n)
	if err != nil {
		return fmt.Errorf("alloc F32 RMSNorm output: %w", err)
	}
	defer outBuf.Free()
	if err := xBuf.Upload(x); err != nil {
		return fmt.Errorf("upload F32 RMSNorm input: %w", err)
	}
	if err := wBuf.Upload(weight); err != nil {
		return fmt.Errorf("upload F32 RMSNorm weight: %w", err)
	}
	if err := F32RMSNormBuffer(outBuf, xBuf, wBuf, n, eps); err != nil {
		return err
	}
	if err := outBuf.Download(out); err != nil {
		return fmt.Errorf("download F32 RMSNorm output: %w", err)
	}
	return nil
}
