package nvidia

import (
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineDualNormForwardABI(t *testing.T) {
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
		a := unsafe.Slice((*unsafe.Pointer)(args), 11)
		if fn != 10 || gx != 1 || *(*uint32)(a[6]) != 3 || *(*uint32)(a[7]) != 7 || *(*uint32)(a[8]) != kind || *(*uint32)(a[9]) != mode || *(*float32)(a[10]) != float32(1e-4) {
			t.Fatal("forward ABI")
		}
		return CUDA_SUCCESS
	}
	op := &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_norm_dual_forward": 10}}}
	b := make([]*Buffer, 6)
	for i := range b {
		b[i] = &Buffer{Ptr: CUdeviceptr(4096 * (i + 1)), Size: 84}
	}
	for kind = 0; kind < 2; kind++ {
		for mode = 0; mode < 3; mode++ {
			if err := op.NormDualForward(b[5], b[0], b[1], b[2], b[3], b[4], 3, 7, kind, mode, 1e-4); err != nil {
				t.Fatal(err)
			}
		}
	}
	if calls != 6 {
		t.Fatal(calls)
	}
	if op.NormDualForward(b[4], b[0], b[1], b[2], b[3], b[4], 3, 7, 0, 0, 1e-4) == nil || calls != 6 {
		t.Fatal("forward alias launched")
	}
}
