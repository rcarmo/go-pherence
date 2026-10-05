package nvidia

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineResidentNorm(t *testing.T) {
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
	positive := uint32(0)
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		a := unsafe.Slice((*unsafe.Pointer)(args), 4)
		if fn != 20 || gx != 1 || *(*uint32)(a[2]) != 257 || *(*uint32)(a[3]) != positive {
			t.Fatal("norm ABI")
		}
		return CUDA_SUCCESS
	}
	makeOp := func() *TrainingF32 {
		return &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_norm_stats": 20}}}
	}
	op := makeOp()
	x, dst := &Buffer{Ptr: 4096, Size: 257 * 4}, &Buffer{Ptr: 8192, Size: 24}
	for positive = 0; positive < 2; positive++ {
		if err := op.NormStats(dst, x, 257, positive == 1); err != nil {
			t.Fatal(err)
		}
	}
	if op.NormStats(x, x, 257, false) == nil || op.NormStats(dst, nil, 257, false) == nil || op.NormStats(dst, x, 0, false) == nil || calls != 2 {
		t.Fatal("invalid norm launched")
	}
	for _, bad := range []*TrainingF32{nil, {}, {dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{}}}} {
		if bad.NormStats(dst, x, 257, false) == nil {
			t.Fatal("invalid owner")
		}
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		calls++
		return 201
	}
	op = makeOp()
	if op.NormStats(dst, x, 257, false) == nil || op.poison == nil {
		t.Fatal("failed norm not terminal")
	}
	before := calls
	if op.NormStats(dst, x, 257, false) == nil || calls != before {
		t.Fatal("failed norm retried")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	cuCtxSynchronize = func() CUresult { return 201 }
	op = makeOp()
	if op.NormStats(dst, x, 257, false) == nil || op.poison == nil {
		t.Fatal("sync failure")
	}
}
func TestPTXTrainingOfflineResidentDriver(t *testing.T) {
	ctx := context.Background()
	d := &PTXResidentDriver{Operator: &TrainingF32{}}
	if _, err := d.NewBuffer(ctx, 2); err == nil {
		t.Fatal("unavailable runtime admitted")
	}
	for _, bad := range []*PTXResidentDriver{nil, {}} {
		if _, err := bad.NewBuffer(ctx, 2); err == nil {
			t.Fatal("nil operator")
		}
		if bad.Accumulate(ctx, nil, nil, nil, 1, 1) == nil || bad.NormStats(ctx, nil, nil, 1, false) == nil {
			t.Fatal("nil driver")
		}
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := d.NewBuffer(cancelled, 2); err == nil {
		t.Fatal("cancel")
	}
	if _, err := ptxResidentBuffers(nil); err == nil {
		t.Fatal("nil buffer")
	}
	b := make([]train.ResidentBuffer, 5)
	for i := range b {
		b[i] = &ptxResidentBuffer{buffer: &Buffer{Ptr: CUdeviceptr(4096 * (i + 1)), Size: 8}}
	}
	step, _ := train.NewAdamWStep(.01, .9, .999, 1e-8, .01, .9, 1, 1)
	if d.ValidateUpdate(ctx, b[0], b[1], b[2], b[3], b[4], 2, step) == nil {
		t.Fatal("missing update kernel")
	}
	d.Operator.module = &PTXModule{handle: 1, functions: map[string]CUfunction{"training_adamw": 20}}
	if err := d.ValidateUpdate(ctx, b[0], b[1], b[2], b[3], b[4], 2, step); err != nil {
		t.Fatal(err)
	}
	if d.ValidateUpdate(ctx, b[0], b[0], b[2], b[3], b[4], 2, step) == nil || d.ValidateUpdate(ctx, b[0], b[1], b[2], b[3], b[4], 3, step) == nil || d.ValidateUpdate(ctx, b[0], b[1], b[2], b[3], b[4], 0, step) == nil {
		t.Fatal("invalid alias/shape")
	}
	if d.Accumulate(cancelled, b[0], b[1], b[2], 2, 1) == nil || d.NormStats(cancelled, b[0], b[1], 2, false) == nil || d.Update(cancelled, b[0], b[1], b[2], b[3], b[4], 2, step) == nil {
		t.Fatal("cancelled operation")
	}
	if d.Accumulate(ctx, nil, b[1], b[2], 2, 1) == nil || d.NormStats(ctx, nil, b[1], 2, false) == nil {
		t.Fatal("foreign buffer")
	}
}

func TestPTXTrainingOfflineResidentTransfers(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	oldAlloc, oldFree, oldUp, oldDown := cuMemAlloc, cuMemFree, cuMemcpyHtoD, cuMemcpyDtoH
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
		cuMemAlloc, cuMemFree, cuMemcpyHtoD, cuMemcpyDtoH = oldAlloc, oldFree, oldUp, oldDown
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	allocs, frees, calls := 0, 0, 0
	cuMemAlloc = func(p *CUdeviceptr, n uint64) CUresult {
		allocs++
		*p = CUdeviceptr(4096 * allocs)
		return CUDA_SUCCESS
	}
	cuMemFree = func(CUdeviceptr) CUresult { frees++; return CUDA_SUCCESS }
	cuMemcpyHtoD = func(CUdeviceptr, unsafe.Pointer, uint64) CUresult { return CUDA_SUCCESS }
	cuMemcpyDtoH = func(p unsafe.Pointer, d CUdeviceptr, n uint64) CUresult {
		for i := range unsafe.Slice((*float32)(p), int(n/4)) {
			unsafe.Slice((*float32)(p), int(n/4))[i] = 7
		}
		return CUDA_SUCCESS
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		calls++
		return CUDA_SUCCESS
	}
	op := &TrainingF32{module: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_adamw": 1}}, dualModule: &PTXModule{handle: 2, functions: map[string]CUfunction{"training_vector": 2, "training_norm_stats": 3}}}
	d := &PTXResidentDriver{Operator: op}
	ctx := context.Background()
	b := make([]train.ResidentBuffer, 5)
	for i := range b {
		n := 2
		if i == 4 {
			n = 3
		}
		var err error
		b[i], err = d.NewBuffer(ctx, n)
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := b[0].Upload(ctx, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	output := make([]float32, 2)
	if err := b[0].Download(ctx, output); err != nil || output[0] != 7 {
		t.Fatal(err, output)
	}
	if err := d.Accumulate(ctx, b[2], b[0], b[1], 2, 1); err != nil {
		t.Fatal(err)
	}
	if err := d.NormStats(ctx, b[4], b[2], 2, false); err != nil {
		t.Fatal(err)
	}
	step, _ := train.NewAdamWStep(.01, .9, .999, 1e-8, .01, .9, 1, 1)
	if err := d.Update(ctx, b[0], b[1], b[2], b[3], b[4], 2, step); err != nil {
		t.Fatal(err)
	}
	if calls != 3 {
		t.Fatal("driver routing", calls)
	}
	for _, buffer := range b {
		if err := buffer.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if allocs != frees {
		t.Fatal("buffer owners", allocs, frees)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if b[0].Upload(cancelled, nil) == nil || b[0].Download(cancelled, nil) == nil {
		t.Fatal("cancelled transfer")
	}
}
