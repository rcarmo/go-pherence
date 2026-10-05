package nvidia

import (
	"errors"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineEOSAdmission(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	op := &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_eos_admission": 21}}}
	hidden, weight, bias, out := &Buffer{Ptr: 4096, Size: 24}, &Buffer{Ptr: 8192, Size: 8}, &Buffer{Ptr: 12288, Size: 4}, &Buffer{Ptr: 16384, Size: 4}
	calls := 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, _ unsafe.Pointer) CUresult {
		a := unsafe.Slice((*unsafe.Pointer)(args), 7)
		calls++
		_, budget, _ := train.EOSAdmissionShape(3, 2)
		if fn != 21 || gx != 1 || bx != 256 || len(a) != 7 {
			t.Fatal("admission launch")
		}
		for i, want := range []uint32{3, 2, budget} {
			if *(*uint32)(a[i+4]) != want {
				t.Fatal("ABI")
			}
		}
		return 0
	}
	cuCtxSynchronize = func() CUresult { return 0 }
	if err := op.EOSAdmission(out, hidden, weight, bias, 3, 2); err != nil || calls != 1 {
		t.Fatal(err)
	}
	for _, kind := range []string{"rows", "extent", "alias", "nil", "kernel"} {
		var err error
		switch kind {
		case "rows":
			err = op.EOSAdmission(out, hidden, weight, bias, 0, 2)
		case "extent":
			err = op.EOSAdmission(out, weight, weight, bias, 3, 2)
		case "alias":
			err = op.EOSAdmission(bias, hidden, weight, bias, 3, 2)
		case "nil":
			err = (*TrainingF32)(nil).EOSAdmission(out, hidden, weight, bias, 3, 2)
		case "kernel":
			err = (&TrainingF32{dualModule: &PTXModule{handle: 1}}).EOSAdmission(out, hidden, weight, bias, 3, 2)
		}
		if err == nil {
			t.Fatal("admission", kind)
		}
	}
	for _, failure := range []string{"launch", "sync"} {
		fresh := &TrainingF32{dualModule: op.dualModule}
		cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
			calls++
			if failure == "launch" {
				return 201
			}
			return 0
		}
		cuCtxSynchronize = func() CUresult {
			if failure == "sync" {
				return 201
			}
			return 0
		}
		if err := fresh.EOSAdmission(out, hidden, weight, bias, 3, 2); err == nil || fresh.poison == nil {
			t.Fatal("failure")
		}
		before := calls
		if fresh.EOSAdmission(out, hidden, weight, bias, 3, 2) == nil || calls != before {
			t.Fatal("retry")
		}
	}
	terminal := &TrainingF32{poison: errors.New("terminal")}
	if terminal.EOSAdmission(out, hidden, weight, bias, 3, 2) == nil {
		t.Fatal("poison")
	}
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{6}, Values: make([]float32, 6)}, {Shape: []int{2}, Values: make([]float32, 2)}, {Shape: []int{1}, Values: []float32{0}}, {Shape: []int{1}}}, Commands: []train.Command{{Op: "eos-admission", Inputs: []int{0, 1, 2}, Output: 3, Rows: 3, Width: 2}}, Outputs: []int{3}}
	if op.admitTapeKernels(tape) != nil || (&TrainingF32{dualModule: &PTXModule{handle: 1}}).admitTapeKernels(tape) == nil {
		t.Fatal("preflight")
	}
}
