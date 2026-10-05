package nvidia

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineLSDLoss(t *testing.T) {
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
		return &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_lsd_loss": 21}}}
	}
	p, d, l, out := &Buffer{Ptr: 4096, Size: 257 * 4}, &Buffer{Ptr: 8192, Size: 257 * 4}, &Buffer{Ptr: 16384, Size: 4}, &Buffer{Ptr: 24576, Size: 3 * 4}
	mode, calls := uint32(0), 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		a := unsafe.Slice((*unsafe.Pointer)(args), 6)
		groups := uint32(1)
		if mode == 1 {
			groups = 2
		}
		if fn != 21 || gx != groups || bx != 256 || *(*uint32)(a[4]) != 257 || *(*uint32)(a[5]) != mode {
			t.Fatal("LSD ABI")
		}
		for i, b := range []*Buffer{p, d, l, out} {
			if *(*CUdeviceptr)(a[i]) != b.Ptr {
				t.Fatal("pointer")
			}
		}
		return CUDA_SUCCESS
	}
	for mode = 0; mode < 3; mode++ {
		if mode == 1 {
			out.Size = 257 * 4
		}
		if makeOp().LSDLoss(out, p, d, l, 257, mode) != nil {
			t.Fatal("valid")
		}
	}
	if makeOp().LSDLoss(out, p, d, l, 0, 1) == nil || makeOp().LSDLoss(p, p, d, l, 257, 1) == nil || makeOp().LSDLoss(out, nil, d, l, 257, 1) == nil || calls != 3 {
		t.Fatal("admission")
	}
	for _, bad := range []*TrainingF32{nil, {}, {dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{}}}} {
		if bad.LSDLoss(out, p, d, l, 257, 1) == nil {
			t.Fatal("owner")
		}
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		calls++
		return 201
	}
	op := makeOp()
	if op.LSDLoss(out, p, d, l, 257, 1) == nil || op.poison == nil {
		t.Fatal("launch")
	}
	before := calls
	if op.LSDLoss(out, p, d, l, 257, 1) == nil || before != calls {
		t.Fatal("retry")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	cuCtxSynchronize = func() CUresult { return 201 }
	op = makeOp()
	if op.LSDLoss(out, p, d, l, 257, 1) == nil || op.poison == nil {
		t.Fatal("sync")
	}
	if makeOp().admitTapeKernels(train.Tape{Commands: []train.Command{{Op: "lsd-loss"}}}) != nil {
		t.Fatal("preflight")
	}
	tape := &TrainingTape{op: &TrainingF32{}, buffers: []*Buffer{p, d, l, out}, shapes: [][]int{{257}, {257}, {1}, {257}}}
	if tape.command(train.Command{Op: "lsd-loss", Inputs: []int{0, 1, 2}, Output: 3, Mode: 1}) == nil {
		t.Fatal("closed tape")
	}
}
