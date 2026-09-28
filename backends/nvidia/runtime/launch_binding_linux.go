//go:build linux && (amd64 || arm64)

package nvidia

import (
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// bindFixedCUDAKernelLauncher replaces only the high-frequency launch binding.
// Missing symbols leave the existing typed binding/fallback untouched. Driver
// serialisation, context and stream selection remain the callers' responsibility.
func bindFixedCUDAKernelLauncher(lib uintptr) {
	entry, err := purego.Dlsym(lib, "cuLaunchKernel")
	if err == nil && entry != 0 {
		cuLaunchKernel = fixedCUDAKernelLauncher(entry)
	}
}

// CUDA's signature contains only integer/pointer arguments. Explicit widening
// preserves unsigned 32-bit dimensions, including stack-passed slots, and keeps
// purego's supported SyscallN ABI trampoline without reflective call marshaling.
func fixedCUDAKernelLauncher(entry uintptr) func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
	// All runtime call sites hold cudaMu and pin their thread. This binding is
	// intentionally non-reentrant, like the underlying shared context scope.
	// Keeping the pointer roots here also forces incoming stack tables to escape
	// before converting them to uintptr; no movable stack address is retained.
	state := &struct {
		words       [11]uintptr
		args, extra unsafe.Pointer
	}{}
	return func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		state.args, state.extra = args, extra
		state.words = [11]uintptr{uintptr(fn), uintptr(gx), uintptr(gy), uintptr(gz), uintptr(bx), uintptr(by), uintptr(bz), uintptr(shared), stream, uintptr(args), uintptr(extra)}
		result, _, _ := purego.SyscallN(entry, state.words[:]...)
		// CUDA consumes host parameters before returning, including error paths.
		runtime.KeepAlive(state.args)
		runtime.KeepAlive(state.extra)
		state.args, state.extra = nil, nil
		clear(state.words[:])
		return CUresult(result)
	}
}
