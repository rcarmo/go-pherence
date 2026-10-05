package nvidia

import (
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineSiLULaunch(t *testing.T) {
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
		table := unsafe.Slice((*unsafe.Pointer)(args), 4)
		if fn != 3 || gx != 2 || *(*uint32)(table[3]) != 257 {
			t.Fatal("SiLU ABI")
		}
		return CUDA_SUCCESS
	}
	op := &TrainingF32{module: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_silu_backward": 3}}}
	x, dy, dst := &Buffer{Ptr: 4096, Size: 257 * 4}, &Buffer{Ptr: 8192, Size: 257 * 4}, &Buffer{Ptr: 12288, Size: 257 * 4}
	if err := op.SiLUBackward(dst, x, dy, 257); err != nil {
		t.Fatal(err)
	}
	if op.SiLUBackward(x, x, dy, 257) == nil || op.SiLUBackward(dst, x, dy, 258) == nil || calls != 1 {
		t.Fatal("alias/extent admitted")
	}
}
func TestPTXTrainingOfflineLinearLaunch(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	syncs, calls := 0, 0
	cuCtxSynchronize = func() CUresult { syncs++; return CUDA_SUCCESS }
	mode := uint32(0)
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		if fn != 2 || gx != 1 || gy != 1 || gz != 1 || bx != 256 {
			t.Fatal("linear launch geometry")
		}
		table := unsafe.Slice((*unsafe.Pointer)(args), 8)
		if *(*uint32)(table[4]) != 3 || *(*uint32)(table[5]) != 5 || *(*uint32)(table[6]) != 7 || *(*uint32)(table[7]) != mode {
			t.Fatal("linear argument order")
		}
		return CUDA_SUCCESS
	}
	op := &TrainingF32{module: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_linear_backward": 2}}}
	x, w, dy, dst := &Buffer{Ptr: 4096, Size: 60}, &Buffer{Ptr: 8192, Size: 140}, &Buffer{Ptr: 12288, Size: 84}, &Buffer{Ptr: 16384, Size: 140}
	for mode = 0; mode < 3; mode++ {
		if err := op.LinearBackward(dst, x, w, dy, 3, 5, 7, mode); err != nil {
			t.Fatal(err)
		}
	}
	if calls != 3 || syncs != 3 {
		t.Fatal(calls, syncs)
	}
	for _, test := range []struct {
		dst     *Buffer
		m, k, n int
		mode    uint32
	}{{x, 3, 5, 7, 0}, {&Buffer{Ptr: 16384, Size: 1}, 3, 5, 7, 1}, {dst, 0, 5, 7, 0}, {dst, 3, 5, 7, 3}, {&Buffer{Ptr: 4100, Size: 140}, 3, 5, 7, 0}} {
		if op.LinearBackward(test.dst, x, w, dy, test.m, test.k, test.n, test.mode) == nil {
			t.Fatal("invalid linear call admitted")
		}
	}
	if calls != 3 {
		t.Fatal("invalid call launched")
	}
}
