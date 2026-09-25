package nvidia

import (
	"math"
	"testing"
	"unsafe"
)

func TestLaunchBatchValidationAndScope(t *testing.T) {
	oldOK := gpuOK
	defer func() { gpuOK = oldOK }()
	gpuOK = true
	oldCtx, oldSet, oldLaunch, oldSync, oldCapture := gpuCtx, cuCtxSetCurrent, cuLaunchKernel, cuCtxSynchronize, captureLaunchStream
	defer func() {
		gpuCtx, cuCtxSetCurrent, cuLaunchKernel, cuCtxSynchronize, captureLaunchStream = oldCtx, oldSet, oldLaunch, oldSync, oldCapture
	}()
	t.Setenv("GO_PHERENCE_CUDA_LAUNCH_CHECK", "")
	gpuCtx = 1
	captureLaunchStream = 0
	sets, calls, syncs := 0, 0, 0
	fail := false
	check := func() {
		if cudaMu.TryLock() {
			cudaMu.Unlock()
			t.Error("unlocked driver call")
		}
	}
	cuCtxSetCurrent = func(CUcontext) CUresult { check(); sets++; return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { check(); syncs++; return CUDA_SUCCESS }
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, sm uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		check()
		calls++
		if stream != uintptr(captureLaunchStream) {
			t.Error("wrong stream")
		}
		pointers := unsafe.Slice((*unsafe.Pointer)(args), 2)
		if *(*uint64)(pointers[0]) != 1234 || math.Float32frombits(*(*uint32)(pointers[1])) != 0.125 {
			t.Error("argument bits")
		}
		if fail {
			return 1
		}
		return CUDA_SUCCESS
	}
	cmds := make([]KernelLaunch, 2)
	for i := range cmds {
		cmds[i] = KernelLaunch{Function: 1, Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{32, 1, 1}, ArgCount: 2, Args: [8]uint64{1234, uint64(math.Float32bits(0.125))}}
	}
	if err := LaunchBatch(cmds); err != nil || calls != 2 || sets != 1 {
		t.Fatal("batch failed", err, calls, sets)
	}
	if a := testing.AllocsPerRun(10, func() {
		if err := LaunchBatch(cmds); err != nil {
			panic(err)
		}
	}); a != 0 {
		t.Fatal("batch allocates", a)
	}
	calls = 0
	cmds[1].ArgCount = 9
	if err := LaunchBatch(cmds); err == nil || calls != 0 {
		t.Fatal("partial validation")
	}
	cmds[1].ArgCount = 2
	fail = true
	if err := LaunchBatch(cmds); err == nil || calls != 1 {
		t.Fatal("driver error", err)
	}
	fail = false
	t.Setenv("GO_PHERENCE_CUDA_LAUNCH_CHECK", "1")
	if err := LaunchBatch(cmds); err != nil || syncs != 2 {
		t.Fatal("diagnostic sync", err, syncs)
	}
	captureLaunchStream = 7
	if err := LaunchBatch(cmds); err != nil || syncs != 2 {
		t.Fatal("capture synchronised", err, syncs)
	}
	captureLaunchStream = 0
	cuCtxSynchronize = func() CUresult { return 2 }
	if err := LaunchBatch(cmds); err == nil {
		t.Fatal("sync failure ignored")
	}
	t.Setenv("GO_PHERENCE_CUDA_LAUNCH_CHECK", "")
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	cmds[0].ArgCount = 0
	if err := LaunchBatch(cmds[:1]); err != nil {
		t.Fatal("no-arg launch", err)
	}
	if err := LaunchBatch(nil); err != nil {
		t.Fatal(err)
	}
	cuLaunchKernel = nil
	if err := LaunchBatch(cmds); err == nil {
		t.Fatal("missing driver accepted")
	}
}
