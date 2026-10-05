package nvidia

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineTransformerPrimitives(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	b := []*Buffer{{Ptr: 4096, Size: 4096}, {Ptr: 16384, Size: 4096}, {Ptr: 32768, Size: 4096}}
	makeOp := func() *TrainingF32 {
		return &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_gelu": 17, "training_row": 18}}}
	}
	op := makeOp()
	mode, calls := uint32(0), 0
	operation := "gelu"
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		count := 5
		if operation == "row" {
			count = 8
		}
		a := unsafe.Slice((*unsafe.Pointer)(args), count)
		for i := 0; i < 3; i++ {
			if *(*CUdeviceptr)(a[i]) != b[i].Ptr {
				t.Fatal("primitive pointers")
			}
		}
		if operation == "gelu" {
			if fn != 17 || gx != 2 || *(*uint32)(a[3]) != 257 || *(*uint32)(a[4]) != mode {
				t.Fatal("GELU ABI")
			}
		} else {
			parts, slot := uint32(1), uint32(0)
			if mode < 2 {
				parts, slot = 3, 2
			}
			if fn != 18 || gx != 1 || *(*uint32)(a[3]) != 3 || *(*uint32)(a[4]) != 5 || *(*uint32)(a[5]) != parts || *(*uint32)(a[6]) != slot || *(*uint32)(a[7]) != mode {
				t.Fatal("row ABI")
			}
		}
		return CUDA_SUCCESS
	}
	for mode = 0; mode < 2; mode++ {
		if err := op.GELU(b[2], b[0], b[1], 257, mode); err != nil {
			t.Fatal(err)
		}
	}
	operation = "row"
	for mode = 0; mode < 4; mode++ {
		parts, slot := 1, 0
		if mode < 2 {
			parts, slot = 3, 2
		}
		if err := op.Row(b[2], b[0], b[1], 3, 5, parts, slot, mode); err != nil {
			t.Fatal(err)
		}
	}
	before := calls
	for _, bad := range []*TrainingF32{nil, {}, {dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{}}}} {
		if bad.GELU(b[2], b[0], b[1], 257, 0) == nil || bad.Row(b[2], b[0], b[1], 3, 5, 1, 0, 2) == nil {
			t.Fatal("invalid owner")
		}
	}
	if op.GELU(b[0], b[0], b[1], 257, 0) == nil || op.GELU(b[2], nil, b[1], 257, 0) == nil || op.GELU(b[2], b[0], b[1], 0, 0) == nil || op.Row(b[0], b[0], b[1], 3, 5, 1, 0, 2) == nil || op.Row(b[2], nil, b[1], 3, 5, 1, 0, 2) == nil || op.Row(b[2], b[0], b[1], 0, 5, 1, 0, 2) == nil || calls != before {
		t.Fatal("invalid extent launched")
	}
	for _, kind := range []string{"gelu", "row"} {
		invoke := func(op *TrainingF32) error {
			if kind == "gelu" {
				return op.GELU(b[2], b[0], b[1], 257, 0)
			}
			return op.Row(b[2], b[0], b[1], 3, 5, 1, 0, 2)
		}
		cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
			calls++
			return 201
		}
		op = makeOp()
		if invoke(op) == nil || op.poison == nil {
			t.Fatal("launch failure not terminal")
		}
		before = calls
		if invoke(op) == nil || calls != before {
			t.Fatal("retry")
		}
		cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
			return CUDA_SUCCESS
		}
		cuCtxSynchronize = func() CUresult { return 201 }
		op = makeOp()
		if invoke(op) == nil || op.poison == nil {
			t.Fatal("sync failure not terminal")
		}
		cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	}
	tape := &TrainingTape{op: &TrainingF32{}, buffers: b, shapes: [][]int{{257}, {257}, {257}}}
	for _, c := range []train.Command{{Op: "gelu", Inputs: []int{0, 1}, Output: 2}, {Op: "row", Inputs: []int{0, 1}, Output: 2, Rows: 3, Width: 5, Parts: 1, Mode: 2}} {
		if tape.command(c) == nil {
			t.Fatal("closed tape routed")
		}
	}
	op = makeOp()
	if err := op.admitTapeKernels(train.Tape{Commands: []train.Command{{Op: "gelu"}, {Op: "row"}}}); err != nil {
		t.Fatal(err)
	}
}
