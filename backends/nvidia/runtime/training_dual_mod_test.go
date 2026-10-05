package nvidia

import (
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineDualModABI(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	mode, kind := uint32(0), uint32(0)
	calls := 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		a := unsafe.Slice((*unsafe.Pointer)(args), 10)
		if fn != 9 || gx != 1 || *(*uint32)(a[7]) != 3 || *(*uint32)(a[8]) != kind || *(*uint32)(a[9]) != mode {
			t.Fatal("mod ABI")
		}
		return CUDA_SUCCESS
	}
	op := &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_mod_dual_backward": 9}}}
	b := make([]*Buffer, 7)
	for i := range b {
		b[i] = &Buffer{Ptr: CUdeviceptr(4096 * (i + 1)), Size: 24}
	}
	for kind = 0; kind < 2; kind++ {
		for mode = 0; mode < 4; mode++ {
			if err := op.ModDualBackward(b[6], b[0], b[1], b[2], b[3], b[4], b[5], 3, kind, mode); err != nil {
				t.Fatal(err)
			}
		}
	}
	if calls != 8 {
		t.Fatal(calls)
	}
	if op.ModDualBackward(b[0], b[0], b[1], b[2], b[3], b[4], b[5], 3, 1, 0) == nil || calls != 8 {
		t.Fatal("alias launched")
	}
}
