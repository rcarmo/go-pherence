package nvidia

import (
	"context"

	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineGlobalNorm(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	makeDriver := func() *PTXResidentDriver {
		return &PTXResidentDriver{Operator: &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_global_norm": 22}}}}
	}
	a, b, out := &ptxResidentBuffer{buffer: &Buffer{Ptr: 4096, Size: 16}}, &ptxResidentBuffer{buffer: &Buffer{Ptr: 8192, Size: 24}}, &ptxResidentBuffer{buffer: &Buffer{Ptr: 16384, Size: 16}}
	mode, calls := uint32(0), 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		v := unsafe.Slice((*unsafe.Pointer)(args), 7)
		n, count, ceiling := uint32(257), float32(513), float32(1)
		expectedB := b.buffer.Ptr
		if mode == 1 {
			n, count, ceiling = 1, 770, .25
			expectedB = a.buffer.Ptr
		}
		if fn != 22 || gx != 1 || bx != 256 || *(*uint32)(v[3]) != n || *(*uint32)(v[4]) != mode || *(*float32)(v[5]) != count || *(*float32)(v[6]) != ceiling || *(*CUdeviceptr)(v[0]) != a.buffer.Ptr || *(*CUdeviceptr)(v[1]) != expectedB || *(*CUdeviceptr)(v[2]) != out.buffer.Ptr {
			t.Fatal("global norm ABI")
		}
		return CUDA_SUCCESS
	}
	d := makeDriver()
	if err := d.ValidateNormMerge(context.Background(), out, a, b, 257, 513, 0); err != nil || calls != 0 {
		t.Fatal("preflight", err)
	}
	if err := d.NormMerge(context.Background(), out, a, b, 257, 513, 0); err != nil {
		t.Fatal(err)
	}
	mode = 2
	if err := d.NormMerge(context.Background(), out, a, b, 257, 513, 2); err != nil {
		t.Fatal(err)
	}
	if d.ValidateNormMerge(context.Background(), out, a, b, 257, 513, 1) == nil || d.NormMerge(context.Background(), out, a, b, 257, 513, 3) == nil {
		t.Fatal("invalid merge mode")
	}
	mode = 1
	out.buffer.Size = 16
	if err := d.NormClip(context.Background(), out, a, 770, .25); err != nil {
		t.Fatal(err)
	}
	if d.ValidateNormClip(context.Background(), a, a, 770, .25) == nil || d.ValidateNormMerge(context.Background(), out, a, nil, 257, 513, 0) == nil || d.ValidateNormClip(context.Background(), out, a, 0, 1) == nil || calls != 3 {
		t.Fatal("bad admission")
	}
	for _, bad := range []*PTXResidentDriver{nil, {}, {Operator: &TrainingF32{}}} {
		if bad.ValidateNormClip(context.Background(), out, a, 770, .25) == nil {
			t.Fatal("bad owner")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if d.NormClip(ctx, out, a, 770, .25) == nil {
		t.Fatal("cancel")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		calls++
		return 201
	}
	d = makeDriver()
	if d.NormClip(context.Background(), out, a, 770, .25) == nil || d.Operator.poison == nil {
		t.Fatal("launch failure")
	}
	before := calls
	if d.NormClip(context.Background(), out, a, 770, .25) == nil || calls != before {
		t.Fatal("retry")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	cuCtxSynchronize = func() CUresult { return 201 }
	d = makeDriver()
	if d.NormClip(context.Background(), out, a, 770, .25) == nil || d.Operator.poison == nil {
		t.Fatal("sync")
	}

}
