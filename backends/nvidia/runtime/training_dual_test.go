package nvidia

import (
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineDualNormLinearABI(t *testing.T) {
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
		if gx != 1 || bx != 256 {
			t.Fatal("dual launch geometry")
		}
		if fn == 7 {
			a := unsafe.Slice((*unsafe.Pointer)(args), 10)
			if *(*uint32)(a[6]) != 3 || *(*uint32)(a[7]) != 5 || *(*uint32)(a[8]) != 7 || *(*uint32)(a[9]) != mode {
				t.Fatal("dual linear ABI")
			}
		} else if fn == 8 {
			a := unsafe.Slice((*unsafe.Pointer)(args), 11)
			if *(*uint32)(a[7]) != 3 || *(*uint32)(a[8]) != 5 || *(*uint32)(a[9]) != kind || *(*uint32)(a[10]) != mode {
				t.Fatal("dual norm ABI")
			}
		} else {
			t.Fatal("wrong function")
		}
		return CUDA_SUCCESS
	}
	op := &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_linear_dual_backward": 7, "training_norm_dual_backward": 8}}}
	b := make([]*Buffer, 7)
	for i := range b {
		b[i] = &Buffer{Ptr: CUdeviceptr(4096 * (i + 1)), Size: 140}
	}
	for mode = 0; mode < 4; mode++ {
		if err := op.LinearDualBackward(b[6], b[0], b[1], b[2], b[3], b[4], 3, 5, 7, mode); err != nil {
			t.Fatal(err)
		}
	}
	for kind = 0; kind < 2; kind++ {
		for mode = 0; mode < 4; mode++ {
			err := op.NormDualBackward(b[6], b[0], b[1], b[2], b[3], b[4], b[5], 3, 5, kind, mode)
			if kind == 1 && mode == 3 {
				if err == nil {
					t.Fatal("variance bias admitted")
				}
			} else if err != nil {
				t.Fatal(err)
			}
		}
	}
	if calls != 11 {
		t.Fatal(calls)
	}
	if op.NormDualBackward(b[0], b[0], b[1], b[2], b[3], b[4], b[5], 3, 5, 0, 0) == nil || calls != 11 {
		t.Fatal("dual alias launched")
	}
}
