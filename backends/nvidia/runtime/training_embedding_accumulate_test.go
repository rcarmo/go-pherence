package nvidia

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineCompactEmbeddingAccumulate(t *testing.T) {
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
		return &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_embedding_accumulate": 29}}}
	}
	previous, compact, index, out := &Buffer{Ptr: 4096, Size: 8 * 4}, &Buffer{Ptr: 8192, Size: 6 * 4}, &Buffer{Ptr: 12288, Size: 10 * 4}, &Buffer{Ptr: 16384, Size: 8 * 4}
	calls := 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		a := unsafe.Slice((*unsafe.Pointer)(args), 8)
		if fn != 29 || gx != 1 || bx != 256 || *(*uint32)(a[4]) != 3 || *(*uint32)(a[5]) != 2 || *(*uint32)(a[6]) != 4 || *(*float32)(a[7]) != .25 {
			t.Fatal("compact accumulation ABI")
		}
		for i, want := range []CUdeviceptr{previous.Ptr, compact.Ptr, index.Ptr, out.Ptr} {
			if *(*CUdeviceptr)(a[i]) != want {
				t.Fatal("pointer", i)
			}
		}
		return CUDA_SUCCESS
	}
	op := makeOp()
	d := &PTXResidentDriver{Operator: op}
	wrap := func(x *Buffer) train.ResidentBuffer { return &ptxResidentBuffer{buffer: x} }
	if err := d.ValidateCompactEmbeddingAccumulate(context.Background(), wrap(out), wrap(previous), wrap(compact), wrap(index), 3, 2, 4, .25); err != nil || calls != 0 {
		t.Fatal("pure admission", err)
	}
	if err := d.CompactEmbeddingAccumulate(context.Background(), wrap(out), wrap(previous), wrap(compact), wrap(index), 3, 2, 4, .25); err != nil || calls != 1 {
		t.Fatal(err)
	}
	for _, bad := range []struct {
		out, prev, c, i *Buffer
		rows            int
		scale           float32
	}{
		{out, previous, compact, index, 0, .25}, {out, previous, compact, index, 3, float32(math.NaN())}, {out, previous, compact, index, 3, float32(math.Inf(1))}, {previous, previous, compact, index, 3, .25}, {out, previous, out, index, 3, .25}, {out, previous, compact, nil, 3, .25}, {out, previous, compact, &Buffer{Ptr: 12288, Size: 9 * 4}, 3, .25},
	} {
		if op.CompactEmbeddingAccumulate(bad.out, bad.prev, bad.c, bad.i, bad.rows, 2, 4, bad.scale) == nil {
			t.Fatal("bad inputs")
		}
	}
	if calls != 1 {
		t.Fatal("admission launched")
	}
	if d.CompactEmbeddingAccumulate(context.Background(), wrap(out), wrap(previous), wrap(compact), wrap(index), 0, 2, 4, .25) == nil {
		t.Fatal("driver shape")
	}
	for _, bad := range []*TrainingF32{nil, {}, {dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{}}}} {
		if bad.CompactEmbeddingAccumulate(out, previous, compact, index, 3, 2, 4, .25) == nil {
			t.Fatal("owner")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if d.ValidateCompactEmbeddingAccumulate(ctx, wrap(out), wrap(previous), wrap(compact), wrap(index), 3, 2, 4, .25) == nil || d.ValidateCompactEmbeddingAccumulate(nil, wrap(out), wrap(previous), wrap(compact), wrap(index), 3, 2, 4, .25) == nil {
		t.Fatal("context")
	}
	if (&PTXResidentDriver{}).ValidateCompactEmbeddingAccumulate(context.Background(), wrap(out), wrap(previous), wrap(compact), wrap(index), 3, 2, 4, .25) == nil {
		t.Fatal("driver")
	}
	if d.ValidateCompactEmbeddingAccumulate(context.Background(), nil, wrap(previous), wrap(compact), wrap(index), 3, 2, 4, .25) == nil {
		t.Fatal("foreign")
	}
	if d.ValidateCompactEmbeddingAccumulate(context.Background(), wrap(out), wrap(previous), wrap(compact), wrap(&Buffer{Ptr: 12288, Size: 9 * 4}), 3, 2, 4, .25) == nil {
		t.Fatal("extent")
	}
	if d.ValidateCompactEmbeddingAccumulate(context.Background(), wrap(out), wrap(previous), wrap(compact), wrap(index), 3, 2, 4, float32(math.NaN())) == nil {
		t.Fatal("admission scale")
	}
	closed := &PTXResidentDriver{Operator: &TrainingF32{}}
	if closed.ValidateCompactEmbeddingAccumulate(context.Background(), wrap(out), wrap(previous), wrap(compact), wrap(index), 3, 2, 4, .25) == nil {
		t.Fatal("entry")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		calls++
		return 201
	}
	op = makeOp()
	if op.CompactEmbeddingAccumulate(out, previous, compact, index, 3, 2, 4, .25) == nil || op.poison == nil {
		t.Fatal("launch not terminal")
	}
	before := calls
	if op.CompactEmbeddingAccumulate(out, previous, compact, index, 3, 2, 4, .25) == nil || calls != before {
		t.Fatal("retry")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	cuCtxSynchronize = func() CUresult { return 201 }
	op = makeOp()
	if op.CompactEmbeddingAccumulate(out, previous, compact, index, 3, 2, 4, .25) == nil || op.poison == nil {
		t.Fatal("sync")
	}
	poisoned := &PTXResidentDriver{Operator: op}
	if poisoned.ValidateCompactEmbeddingAccumulate(context.Background(), wrap(out), wrap(previous), wrap(compact), wrap(index), 3, 2, 4, .25) == nil {
		t.Fatal("poison admission")
	}
}
