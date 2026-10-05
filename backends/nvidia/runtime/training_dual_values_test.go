package nvidia

import (
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineDualValueABI(t *testing.T) {
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
		if fn == 11 {
			a := unsafe.Slice((*unsafe.Pointer)(args), 9)
			if *(*uint32)(a[5]) != 3 || *(*uint32)(a[6]) != 5 || *(*uint32)(a[7]) != 7 || *(*uint32)(a[8]) != mode {
				t.Fatal("linear forward ABI")
			}
		} else if fn == 12 {
			a := unsafe.Slice((*unsafe.Pointer)(args), 5)
			if *(*uint32)(a[3]) != 3 || *(*uint32)(a[4]) != mode {
				t.Fatal("SiLU forward ABI")
			}
		} else if fn == 13 {
			a := unsafe.Slice((*unsafe.Pointer)(args), 8)
			if *(*uint32)(a[5]) != 3 || *(*uint32)(a[6]) != kind || *(*uint32)(a[7]) != mode {
				t.Fatal("mod forward ABI")
			}
		} else {
			t.Fatal("function")
		}
		return CUDA_SUCCESS
	}
	op := &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_linear_dual_forward": 11, "training_silu_dual_forward": 12, "training_mod_dual_forward": 13}}}
	b := make([]*Buffer, 5)
	for i := range b {
		b[i] = &Buffer{Ptr: CUdeviceptr(4096 * (i + 1)), Size: 140}
	}
	for mode = 0; mode < 2; mode++ {
		if err := op.LinearDualForward(b[4], b[0], b[1], b[2], b[3], 3, 5, 7, mode); err != nil {
			t.Fatal(err)
		}
		if err := op.SiLUDualForward(b[4], b[0], b[1], 3, mode); err != nil {
			t.Fatal(err)
		}
		for kind = 0; kind < 2; kind++ {
			if err := op.ModDualForward(b[4], b[0], b[1], b[2], b[3], 3, kind, mode); err != nil {
				t.Fatal(err)
			}
		}
	}
	if calls != 8 {
		t.Fatal(calls)
	}
	if op.SiLUDualForward(b[0], b[0], b[1], 3, 0) == nil || calls != 8 {
		t.Fatal("alias launched")
	}
}
