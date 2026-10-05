package nvidia

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineEmbedding(t *testing.T) {
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
		return &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_embedding": 21}}}
	}
	source, ids, dst := &Buffer{Ptr: 4096, Size: 8 * 4}, &Buffer{Ptr: 8192, Size: 3 * 4}, &Buffer{Ptr: 16384, Size: 6 * 4}
	mode, calls := uint32(0), 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		a := unsafe.Slice((*unsafe.Pointer)(args), 7)
		if fn != 21 || gx != 1 || bx != 256 || *(*uint32)(a[3]) != 3 || *(*uint32)(a[4]) != 2 || *(*uint32)(a[5]) != 4 || *(*uint32)(a[6]) != mode {
			t.Fatal("embedding ABI")
		}
		if *(*CUdeviceptr)(a[0]) != source.Ptr || *(*CUdeviceptr)(a[1]) != ids.Ptr || *(*CUdeviceptr)(a[2]) != dst.Ptr {
			t.Fatal("pointers")
		}
		return CUDA_SUCCESS
	}
	for mode = 0; mode < 4; mode++ {
		if mode != 0 {
			source.Size, dst.Size = 6*4, 8*4
		}
		if mode == 2 {
			ids.Size = 8 * 4
		}
		if mode == 3 {
			ids.Size = 10 * 4
			dst.Size = 6 * 4
		}
		if makeOp().Embedding(dst, source, ids, 3, 2, 4, mode) != nil {
			t.Fatal("valid")
		}
	}
	if makeOp().Embedding(dst, source, &Buffer{Ptr: 8192, Size: 3 * 4}, 3, 2, 4, 2) == nil || calls != 4 {
		t.Fatal("short index")
	}
	if makeOp().Embedding(&Buffer{Ptr: 16384, Size: 6 * 4}, source, &Buffer{Ptr: 8192, Size: 8 * 4}, 3, 2, 4, 3) == nil || calls != 4 {
		t.Fatal("short compact index")
	}
	ids.Size = 3 * 4
	source.Size, dst.Size = 6*4, 8*4
	if makeOp().Embedding(dst, source, ids, 0, 2, 4, 0) == nil || makeOp().Embedding(source, source, ids, 3, 2, 4, 1) == nil || makeOp().Embedding(dst, nil, ids, 3, 2, 4, 1) == nil || calls != 4 {
		t.Fatal("bad admission")
	}
	for _, bad := range []*TrainingF32{nil, {}, {dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{}}}} {
		if bad.Embedding(dst, source, ids, 3, 2, 4, 1) == nil {
			t.Fatal("bad owner")
		}
	}
	ids.Size = 10 * 4 // compact mode has the same terminal ownership contract
	dst.Size = 6 * 4
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		calls++
		return 201
	}
	op := makeOp()
	if op.Embedding(dst, source, ids, 3, 2, 4, 3) == nil || op.poison == nil {
		t.Fatal("launch not terminal")
	}
	before := calls
	if op.Embedding(dst, source, ids, 3, 2, 4, 3) == nil || before != calls {
		t.Fatal("retry")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	cuCtxSynchronize = func() CUresult { return 201 }
	op = makeOp()
	if op.Embedding(dst, source, ids, 3, 2, 4, 3) == nil || op.poison == nil {
		t.Fatal("sync")
	}
	tapeValid := &TrainingTape{op: makeOp(), buffers: []*Buffer{source, ids, dst}, shapes: [][]int{{6}, {10}, {6}}}
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	if err := tapeValid.command(train.Command{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Rows: 3, Width: 2, Parts: 4, Mode: 3}); err != nil {
		t.Fatal("indexed tape dispatch", err)
	}
	if makeOp().admitTapeKernels(train.Tape{Commands: []train.Command{{Op: "embedding"}}}) != nil {
		t.Fatal("preflight")
	}
	tape := &TrainingTape{op: &TrainingF32{}, buffers: []*Buffer{source, ids, dst}, shapes: [][]int{{6}, {3}, {8}}}
	if tape.command(train.Command{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Rows: 3, Width: 2, Parts: 4, Mode: 1}) == nil {
		t.Fatal("closed tape")
	}
}
