package nvidia

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineTimeEmbedding(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	makeOp := func() *TrainingF32 {
		return &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_time_embedding": 21}}}
	}
	f, time, gv, gt, dst := &Buffer{Ptr: 4096, Size: 257 * 4}, &Buffer{Ptr: 8192, Size: 8}, &Buffer{Ptr: 16384, Size: 514 * 4}, &Buffer{Ptr: 24576, Size: 514 * 4}, &Buffer{Ptr: 32768, Size: 514 * 4}
	mode, calls := uint32(0), 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		a := unsafe.Slice((*unsafe.Pointer)(args), 7)
		groups := uint32(3)
		if mode == 2 {
			groups = 1
		}
		if fn != 21 || gx != groups || bx != 256 || *(*uint32)(a[5]) != 257 || *(*uint32)(a[6]) != mode {
			t.Fatal("time ABI")
		}
		for i, b := range []*Buffer{f, time, gv, gt, dst} {
			if *(*CUdeviceptr)(a[i]) != b.Ptr {
				t.Fatal("pointer")
			}
		}
		return CUDA_SUCCESS
	}
	for mode = 0; mode < 3; mode++ {
		if mode == 2 {
			dst.Size = 4
		}
		if makeOp().TimeEmbedding(dst, f, time, gv, gt, 257, mode) != nil {
			t.Fatal("valid")
		}
	}
	if makeOp().TimeEmbedding(dst, f, time, gv, gt, 0, 2) == nil || makeOp().TimeEmbedding(f, f, time, gv, gt, 257, 2) == nil || makeOp().TimeEmbedding(dst, nil, time, gv, gt, 257, 2) == nil || calls != 3 {
		t.Fatal("bad admission")
	}
	for _, bad := range []*TrainingF32{nil, {}, {dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{}}}} {
		if bad.TimeEmbedding(dst, f, time, gv, gt, 257, 2) == nil {
			t.Fatal("owner")
		}
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		calls++
		return 201
	}
	op := makeOp()
	if op.TimeEmbedding(dst, f, time, gv, gt, 257, 2) == nil || op.poison == nil {
		t.Fatal("launch")
	}
	before := calls
	if op.TimeEmbedding(dst, f, time, gv, gt, 257, 2) == nil || before != calls {
		t.Fatal("retry")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	cuCtxSynchronize = func() CUresult { return 201 }
	op = makeOp()
	if op.TimeEmbedding(dst, f, time, gv, gt, 257, 2) == nil || op.poison == nil {
		t.Fatal("sync")
	}
	if makeOp().admitTapeKernels(train.Tape{Commands: []train.Command{{Op: "time-embedding"}}}) != nil {
		t.Fatal("preflight")
	}
	tape := &TrainingTape{op: &TrainingF32{}, buffers: []*Buffer{f, time, gv, gt, dst}, shapes: [][]int{{257}, {2}, {514}, {514}, {1}}}
	if tape.command(train.Command{Op: "time-embedding", Inputs: []int{0, 1, 2, 3}, Output: 4, Mode: 2}) == nil {
		t.Fatal("closed tape")
	}
}
