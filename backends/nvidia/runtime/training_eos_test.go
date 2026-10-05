package nvidia

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineEOS(t *testing.T) {
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
		return &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_eos": 21}}}
	}
	source, ids, dst := &Buffer{Ptr: 4096, Size: 3 * 4}, &Buffer{Ptr: 8192, Size: 3 * 4}, &Buffer{Ptr: 16384, Size: 3 * 4}
	mode, calls := uint32(0), 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		a := unsafe.Slice((*unsafe.Pointer)(args), 7)
		if fn != 21 || gx != 1 || bx != 256 || *(*uint32)(a[3]) != 3 || *(*uint32)(a[4]) != 2 || *(*uint32)(a[5]) != mode || *(*float32)(a[6]) != .1 {
			t.Fatal("eos ABI")
		}
		if *(*CUdeviceptr)(a[0]) != source.Ptr || *(*CUdeviceptr)(a[1]) != ids.Ptr || *(*CUdeviceptr)(a[2]) != dst.Ptr {
			t.Fatal("pointers")
		}
		return CUDA_SUCCESS
	}
	for mode = 0; mode < 3; mode++ {
		if mode == 2 {
			dst.Size = 4
		}
		if makeOp().EOS(dst, source, ids, 3, 2, mode, .1) != nil {
			t.Fatal("valid")
		}
	}
	if makeOp().EOS(dst, source, ids, 0, 2, 0, .1) == nil || makeOp().EOS(source, source, ids, 3, 2, 2, .1) == nil || makeOp().EOS(dst, nil, ids, 3, 2, 2, .1) == nil || calls != 3 {
		t.Fatal("bad admission")
	}
	for _, bad := range []*TrainingF32{nil, {}, {dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{}}}} {
		if bad.EOS(dst, source, ids, 3, 2, 2, .1) == nil {
			t.Fatal("bad owner")
		}
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		calls++
		return 201
	}
	op := makeOp()
	if op.EOS(dst, source, ids, 3, 2, 2, .1) == nil || op.poison == nil {
		t.Fatal("launch not terminal")
	}
	before := calls
	if op.EOS(dst, source, ids, 3, 2, 2, .1) == nil || before != calls {
		t.Fatal("retry")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	cuCtxSynchronize = func() CUresult { return 201 }
	op = makeOp()
	if op.EOS(dst, source, ids, 3, 2, 2, .1) == nil || op.poison == nil {
		t.Fatal("sync")
	}
	if makeOp().admitTapeKernels(train.Tape{Commands: []train.Command{{Op: "eos"}}}) != nil {
		t.Fatal("preflight")
	}
	tape := &TrainingTape{op: &TrainingF32{}, buffers: []*Buffer{source, ids, dst}, shapes: [][]int{{3}, {3}, {1}}}
	if tape.command(train.Command{Op: "eos", Inputs: []int{0, 1}, Output: 2, Rows: 2, Mode: 2, Scale: .1}) == nil {
		t.Fatal("closed tape")
	}
}
