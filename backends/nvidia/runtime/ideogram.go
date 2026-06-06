package nvidia

import (
	"fmt"
	"unsafe"
)

var fnIdeogramCFGStepF32 CUfunction

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
