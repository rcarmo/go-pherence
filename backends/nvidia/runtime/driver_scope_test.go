//go:build linux

package nvidia

import (
	"runtime"
	"sync"
	"syscall"
	"testing"
	"unsafe"
)

func TestStreamModuleAndAsyncDriverScope(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldLoad, oldLoadEx, oldGet, oldUnload := cuCtxSetCurrent, cuModuleLoadData, cuModuleLoadDataEx, cuModuleGetFunction, cuModuleUnload
	oldLaunch, oldCopy, oldAlloc, oldFree, oldSync := cuLaunchKernel, cuMemcpyDtoDAsync, cuMemAlloc, cuMemFree, cuCtxSynchronize
	oldCreate, oldDestroy, oldEC, oldED, oldER, oldWait := cuStreamCreate, cuStreamDestroy, cuEventCreate, cuEventDestroy, cuEventRecord, cuStreamWaitEvent
	oldReady, oldPrefetch, oldCapture, oldActive, oldCE, oldPE := streamsReady, prefetchStream, graphCaptureStream, captureLaunchStream, computeEvent, prefetchEvent
	oldExtra := extraModules
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuModuleLoadData, cuModuleLoadDataEx, cuModuleGetFunction, cuModuleUnload = oldSet, oldLoad, oldLoadEx, oldGet, oldUnload
		cuLaunchKernel, cuMemcpyDtoDAsync, cuMemAlloc, cuMemFree, cuCtxSynchronize = oldLaunch, oldCopy, oldAlloc, oldFree, oldSync
		cuStreamCreate, cuStreamDestroy, cuEventCreate, cuEventDestroy, cuEventRecord, cuStreamWaitEvent = oldCreate, oldDestroy, oldEC, oldED, oldER, oldWait
		streamsReady, prefetchStream, graphCaptureStream, captureLaunchStream, computeEvent, prefetchEvent = oldReady, oldPrefetch, oldCapture, oldActive, oldCE, oldPE
		extraModules = oldExtra
	}()
	gpuOK = true
	gpuCtx = 123
	streamsReady = false
	captureLaunchStream = 0
	extraModules = nil
	tid := 0
	calls := 0
	cuCtxSetCurrent = func(CUcontext) CUresult { tid = syscall.Gettid(); runtime.Gosched(); return CUDA_SUCCESS }
	check := func() {
		t.Helper()
		calls++
		if cudaMu.TryLock() {
			cudaMu.Unlock()
			t.Error("raw driver call outside lock")
		}
		if syscall.Gettid() != tid {
			t.Error("wrong context thread")
		}
		runtime.Gosched()
		if syscall.Gettid() != tid {
			t.Error("unpinned driver scope")
		}
	}
	cuMemAlloc = func(p *CUdeviceptr, n uint64) CUresult { check(); *p = 3; return CUDA_SUCCESS }
	cuMemFree = func(CUdeviceptr) CUresult { check(); return CUDA_SUCCESS }
	cuModuleLoadDataEx = nil
	cuModuleLoadData = func(m *CUmodule, p unsafe.Pointer) CUresult { check(); *m = 8; return CUDA_SUCCESS }
	cuModuleGetFunction = func(fn *CUfunction, m CUmodule, p unsafe.Pointer) CUresult { check(); *fn = 9; return CUDA_SUCCESS }
	cuModuleUnload = func(CUmodule) CUresult { check(); return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { check(); return CUDA_SUCCESS }
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		check()
		return CUDA_SUCCESS
	}
	cuMemcpyDtoDAsync = func(dst, src CUdeviceptr, n uint64, stream uintptr) CUresult {
		check()
		if stream != 7 {
			t.Error("async copy ignored capture stream")
		}
		return CUDA_SUCCESS
	}
	cuStreamCreate = func(s *CUstream, flags uint32) CUresult { check(); *s = 1; return CUDA_SUCCESS }
	cuStreamDestroy = func(CUstream) CUresult { check(); return CUDA_SUCCESS }
	cuEventCreate = func(e *CUevent, flags uint32) CUresult { check(); *e = 2; return CUDA_SUCCESS }
	cuEventDestroy = func(CUevent) CUresult { check(); return CUDA_SUCCESS }
	cuEventRecord = func(CUevent, CUstream) CUresult { check(); return CUDA_SUCCESS }
	cuStreamWaitEvent = func(CUstream, CUevent, uint32) CUresult { check(); return CUDA_SUCCESS }
	release := lockDriver()
	if err := initStreamsLocked(); err != nil {
		t.Fatal(err)
	}
	release()
	prewarmAllocator()
	if _, err := LoadPTX("test", "entry"); err != nil {
		t.Fatal(err)
	}
	MarkComputeDone()
	WaitPrefetch()
	if err := LaunchKernelOnStream(9, 1, 1, 1, 1, 1, 1, 0, 1); err != nil {
		t.Fatal(err)
	}
	captureLaunchStream = 7
	if err := copyDtoDAsync(1, 2, 16); err != nil {
		t.Fatal(err)
	}
	captureLaunchStream = 0
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := LoadPTX("test", "entry"); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if len(extraModules) != 9 {
		t.Fatal("module ownership lost", len(extraModules))
	}
	unloadModule(8)
	shutdownStreams()
	if calls < 30 {
		t.Fatal("callbacks not exercised", calls)
	}
}

func TestLazyFunctionsInvalidatedBeforeContextReplacement(t *testing.T) {
	oldArgmax, oldSample, oldSoftmax := fnArgmaxF32, diffusionSampleState.fn, diffusionSampleState.softmaxFn
	defer func() {
		fnArgmaxF32, diffusionSampleState.fn, diffusionSampleState.softmaxFn = oldArgmax, oldSample, oldSoftmax
	}()
	fnArgmaxF32 = 1
	diffusionSampleState.fn = 2
	diffusionSampleState.softmaxFn = 3
	resetLazyKernelHandles()
	if fnArgmaxF32 != 0 || diffusionSampleState.fn != 0 || diffusionSampleState.softmaxFn != 0 {
		t.Fatal("stale module handle retained")
	}
}

func TestCopyDtoDCaptureAndZeroOverflow(t *testing.T) {
	oldSync, oldAsync, oldSet, oldCapture, oldCtx := cuMemcpyDtoD, cuMemcpyDtoDAsync, cuCtxSetCurrent, captureLaunchStream, gpuCtx
	defer func() {
		cuMemcpyDtoD, cuMemcpyDtoDAsync, cuCtxSetCurrent, captureLaunchStream, gpuCtx = oldSync, oldAsync, oldSet, oldCapture, oldCtx
	}()
	gpuCtx = 0
	syncCalls, asyncCalls := 0, 0
	cuMemcpyDtoD = func(dst, src CUdeviceptr, n uint64) CUresult { syncCalls++; return CUDA_SUCCESS }
	cuMemcpyDtoDAsync = func(dst, src CUdeviceptr, n uint64, stream uintptr) CUresult {
		asyncCalls++
		if stream != 7 {
			t.Fatalf("stream=%d", stream)
		}
		return CUDA_SUCCESS
	}
	captureLaunchStream = 0
	if err := CopyDtoD(1, 2, 4); err != nil {
		t.Fatal(err)
	}
	captureLaunchStream = 7
	if err := CopyDtoD(1, 2, 4); err != nil {
		t.Fatal(err)
	}
	if syncCalls != 1 || asyncCalls != 1 {
		t.Fatal(syncCalls, asyncCalls)
	}
	cuMemcpyDtoDAsync = nil
	if err := CopyDtoD(1, 2, 4); err == nil {
		t.Fatal("missing capture API accepted")
	}
	if err := ZeroFloat32Buffer(&Buffer{Ptr: 1, Size: 4}, int(^uint(0)>>1)/2+1); err == nil {
		t.Fatal("overflowing zero count accepted")
	}
}
