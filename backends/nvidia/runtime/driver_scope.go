package nvidia

import "runtime"

// lockDriver owns the thread-local CUDA context through the complete driver
// operation. It is not reentrant; *Locked helpers require this scope already.
// Global Shutdown still requires a quiescent application (no live inference).
func lockDriver() func() {
	runtime.LockOSThread()
	cudaMu.Lock()
	ensureContextLocked()
	return func() { cudaMu.Unlock(); runtime.UnlockOSThread() }
}

// Reset lazy functions before unloading their modules; otherwise reinitialised
// processes can reuse CUfunction handles from the destroyed context.
func resetLazyKernelHandles() {
	argmaxMu.Lock()
	fnArgmaxF32 = 0
	argmaxMu.Unlock()
	diffusionSampleState.Lock()
	diffusionSampleState.fn = 0
	diffusionSampleState.softmaxFn = 0
	diffusionSampleState.Unlock()
}

func unloadModule(mod CUmodule) {
	if mod == 0 {
		return
	}
	release := lockDriver()
	defer release()
	if cuCtxSynchronize != nil {
		_ = cuCtxSynchronize()
	}
	if cuModuleUnload != nil {
		_ = cuModuleUnload(mod)
	}
}

// prewarmAllocator is optional; use the same checked context scope as real
// allocations rather than raw driver calls from lazy module initialisers.
func prewarmAllocator() {
	release := lockDriver()
	defer release()
	prewarmAllocatorLocked()
}
func prewarmAllocatorLocked() {
	if cuMemAlloc == nil || cuMemFree == nil {
		return
	}
	var ptr CUdeviceptr
	if cuMemAlloc(&ptr, 64*1024*1024) == CUDA_SUCCESS {
		_ = cuMemFree(ptr)
	}
}
