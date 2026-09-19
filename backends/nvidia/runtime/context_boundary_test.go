//go:build linux

package nvidia

import (
	"runtime"
	"sync"
	"syscall"
	"testing"
)

// Model-free regression for the context/call critical section. The fake driver
// checks both CUDA serialisation and OS-thread affinity; no GPU is initialised.
func TestContextAndDriverCallShareCriticalSection(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldCopy, oldInfo, oldSync := cuCtxSetCurrent, cuMemcpyDtoD, cuMemGetInfo, cuCtxSynchronize
	oldZero := cuMemsetD32
	oldStreams := streamsReady
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuMemcpyDtoD, cuMemGetInfo, cuCtxSynchronize = oldSet, oldCopy, oldInfo, oldSync
		streamsReady = oldStreams
		cuMemsetD32 = oldZero
	}()
	gpuOK = true
	gpuCtx = 123
	streamsReady = false
	tid := 0
	calls := 0
	cuCtxSetCurrent = func(ctx CUcontext) CUresult {
		if ctx != 123 {
			t.Error("wrong context")
		}
		tid = syscall.Gettid()
		runtime.Gosched()
		return CUDA_SUCCESS
	}
	check := func() {
		t.Helper()
		calls++
		if cudaMu.TryLock() {
			cudaMu.Unlock()
			t.Error("driver call is outside context serialisation lock")
		}
		if syscall.Gettid() != tid {
			t.Error("thread changed after context selection")
		}
		runtime.Gosched()
		if syscall.Gettid() != tid {
			t.Error("driver call not pinned to context thread")
		}
	}
	cuMemcpyDtoD = func(dst, src CUdeviceptr, n uint64) CUresult { check(); return CUDA_SUCCESS }
	cuMemGetInfo = func(free, total *uint64) CUresult { check(); *free = 12; *total = 24; return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { check(); return CUDA_SUCCESS }
	cuMemsetD32 = func(CUdeviceptr, uint32, uint64) CUresult { check(); return CUDA_SUCCESS }
	for range 8 {
		if err := CopyDtoD(1, 2, 16); err != nil {
			t.Fatal(err)
		}
		free, total := MemInfo()
		if free != 12 || total != 24 {
			t.Fatal("memory info")
		}
		SyncAll()
		if err := ZeroFloat32Buffer(&Buffer{Ptr: 1, Size: 16}, 4); err != nil {
			t.Fatal(err)
		}
	}
	if calls != 32 {
		t.Fatal("driver callbacks not exercised", calls)
	}
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 10 {
				if err := CopyDtoD(1, 2, 16); err != nil {
					t.Error(err)
				}
				MemInfo()
				SyncAll()
			}
		}()
	}
	wg.Wait()
	if calls != 152 {
		t.Fatal("concurrent callbacks not exercised", calls)
	}
	cuMemGetInfo = func(free, total *uint64) CUresult { *free = 999; *total = 999; return 201 }
	if free, total := MemInfo(); free != 0 || total != 0 {
		t.Fatal("memory query failure must not return stale capacity")
	}
}
