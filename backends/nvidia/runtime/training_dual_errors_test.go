package nvidia

import (
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineDualErrors(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { return 201 }
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	b := make([]*Buffer, 7)
	for i := range b {
		b[i] = &Buffer{Ptr: CUdeviceptr(4096 * (i + 1)), Size: 140}
	}
	good := &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_linear_dual_backward": 7, "training_norm_dual_backward": 8, "training_mod_dual_backward": 9, "training_norm_dual_forward": 10}}}
	invoke := func(op *TrainingF32) []error {
		return []error{op.LinearDualBackward(b[6], b[0], b[1], b[2], b[3], b[4], 3, 5, 7, 0), op.NormDualBackward(b[6], b[0], b[1], b[2], b[3], b[4], b[5], 3, 5, 0, 0), op.ModDualBackward(b[6], b[0], b[1], b[2], b[3], b[4], b[5], 3, 0, 0), op.NormDualForward(b[6], b[0], b[1], b[2], b[3], b[4], 3, 5, 0, 0, 1e-4)}
	}
	for _, op := range []*TrainingF32{nil, {}, good} {
		for _, err := range invoke(op) {
			if err == nil {
				t.Fatal("nil/closed/sync failure accepted")
			}
		}
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return 201
	}
	for _, err := range invoke(good) {
		if err == nil {
			t.Fatal("launch failure accepted")
		}
	}
	if good.LinearDualBackward(b[6], b[0], b[1], b[2], b[3], b[4], 0, 5, 7, 0) == nil || good.LinearDualBackward(b[6], b[0], b[1], b[2], b[3], b[4], 3, 5, 7, 4) == nil || good.ModDualBackward(b[6], b[0], b[1], b[2], b[3], b[4], b[5], 3, 2, 0) == nil || good.NormDualForward(b[6], b[0], b[1], b[2], b[3], b[4], 3, 5, 0, 0, 0) == nil {
		t.Fatal("invalid parameters")
	}
	b[6].Size = 0
	for _, err := range invoke(good) {
		if err == nil {
			t.Fatal("short output accepted")
		}
	}
	b[6] = b[0]
	for _, err := range invoke(good) {
		if err == nil {
			t.Fatal("output alias accepted")
		}
	}
}
