package nvidia

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineReLU(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	b := []*Buffer{{Ptr: 4096, Size: 1028}, {Ptr: 16384, Size: 1028}, {Ptr: 32768, Size: 1028}}
	makeOp := func() *TrainingF32 {
		return &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_relu": 21}}}
	}
	mode, calls := uint32(0), 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		a := unsafe.Slice((*unsafe.Pointer)(args), 5)
		if fn != 21 || gx != 2 || bx != 256 || *(*uint32)(a[3]) != 257 || *(*uint32)(a[4]) != mode {
			t.Fatal("ReLU ABI")
		}
		for i := 0; i < 3; i++ {
			if *(*CUdeviceptr)(a[i]) != b[i].Ptr {
				t.Fatal("ReLU pointer")
			}
		}
		return CUDA_SUCCESS
	}
	op := makeOp()
	for mode = 0; mode < 2; mode++ {
		if op.ReLU(b[2], b[0], b[1], 257, mode) != nil {
			t.Fatal("valid launch")
		}
	}
	if op.ReLU(b[0], b[0], b[1], 257, 0) == nil || op.ReLU(b[2], nil, b[1], 257, 0) == nil || op.ReLU(b[2], b[0], b[1], 0, 0) == nil || calls != 2 {
		t.Fatal("invalid launch")
	}
	for _, bad := range []*TrainingF32{nil, {}, {dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{}}}} {
		if bad.ReLU(b[2], b[0], b[1], 257, 0) == nil {
			t.Fatal("invalid kernel")
		}
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		calls++
		return 201
	}
	op = makeOp()
	if op.ReLU(b[2], b[0], b[1], 257, 0) == nil || op.poison == nil {
		t.Fatal("failure not terminal")
	}
	before := calls
	if op.ReLU(b[2], b[0], b[1], 257, 0) == nil || calls != before {
		t.Fatal("terminal retry")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	cuCtxSynchronize = func() CUresult { return 201 }
	op = makeOp()
	if op.ReLU(b[2], b[0], b[1], 257, 0) == nil || op.poison == nil {
		t.Fatal("sync failure")
	}
	if makeOp().admitTapeKernels(train.Tape{Commands: []train.Command{{Op: "relu"}}}) != nil {
		t.Fatal("kernel preflight")
	}
	tape := &TrainingTape{op: &TrainingF32{}, buffers: b, shapes: [][]int{{257}, {257}, {257}}}
	if tape.command(train.Command{Op: "relu", Inputs: []int{0, 1}, Output: 2}) == nil {
		t.Fatal("closed tape")
	}
}
