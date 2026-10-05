package nvidia

import (
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineDualSiLU(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	calls := 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		table := unsafe.Slice((*unsafe.Pointer)(args), 7)
		if fn != 6 || gx != 2 || *(*uint32)(table[6]) != 257 {
			t.Fatal("dual ABI")
		}
		return CUDA_SUCCESS
	}
	op := &TrainingF32{module: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_silu_dual_backward": 6}}}
	b := make([]*Buffer, 6)
	for i := range b {
		b[i] = &Buffer{Ptr: CUdeviceptr(4096 * (i + 1)), Size: 257 * 4}
	}
	if err := op.SiLUDualBackward(b[4], b[5], b[0], b[1], b[2], b[3], 257); err != nil {
		t.Fatal(err)
	}
	if op.SiLUDualBackward(b[4], b[4], b[0], b[1], b[2], b[3], 257) == nil || calls != 1 {
		t.Fatal("dual alias admitted")
	}
}
