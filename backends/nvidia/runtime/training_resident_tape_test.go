package nvidia

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineResidentTapeBinding(t *testing.T) {
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
	allocations, uploads, downloads, frees := 0, 0, 0, 0
	cuMemAlloc = func(p *CUdeviceptr, n uint64) CUresult {
		allocations++
		*p = CUdeviceptr(4096 * allocations)
		return CUDA_SUCCESS
	}
	cuMemFree = func(p CUdeviceptr) CUresult {
		if p == 1<<20 {
			t.Fatal("borrowed parameter freed")
		}
		frees++
		return CUDA_SUCCESS
	}
	cuMemcpyHtoD = func(p CUdeviceptr, x unsafe.Pointer, n uint64) CUresult {
		if p == 1<<20 {
			t.Fatal("borrowed parameter uploaded")
		}
		uploads++
		return CUDA_SUCCESS
	}
	cuMemcpyDtoH = func(p unsafe.Pointer, d CUdeviceptr, n uint64) CUresult {
		downloads++
		for i := range unsafe.Slice((*float32)(p), int(n/4)) {
			unsafe.Slice((*float32)(p), int(n/4))[i] = 7
		}
		return CUDA_SUCCESS
	}
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		if fn == 14 {
			a := unsafe.Slice((*unsafe.Pointer)(args), 8)
			if *(*CUdeviceptr)(a[0]) != 1<<20 {
				t.Fatal("borrowed parameter not bound")
			}
		}
		return CUDA_SUCCESS
	}
	d := &PTXResidentDriver{Operator: &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_vector": 14}}}}
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{99, 99}}, {Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}}, Outputs: []int{2}}
	parameter := &ptxResidentBuffer{buffer: &Buffer{Ptr: 1 << 20, Size: 8}}
	owned, err := d.NewBoundTape(context.Background(), tape, map[int]train.ResidentBuffer{0: parameter}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if allocations != 2 || uploads != 1 {
		t.Fatal("resident copy/alloc", allocations, uploads)
	}
	if _, err := owned.OutputBuffer(2); err == nil {
		t.Fatal("output before execution")
	}
	if err := owned.RunDevice(context.Background()); err != nil {
		t.Fatal(err)
	}
	output, err := owned.OutputBuffer(2)
	if err != nil {
		t.Fatal(err)
	}
	if output.Close() == nil {
		t.Fatal("borrowed gradient freed")
	}
	if err := d.ValidateAccumulate(context.Background(), &ptxResidentBuffer{buffer: &Buffer{Ptr: 2 << 20, Size: 8}}, &ptxResidentBuffer{buffer: &Buffer{Ptr: 3 << 20, Size: 8}}, output, 2, 1); err != nil {
		t.Fatal(err)
	}
	values, err := owned.DownloadOutputs(context.Background(), []int{2})
	if err != nil || values[0][0] != 7 || downloads != 1 {
		t.Fatal(err, values, downloads)
	}
	if err := owned.RunDevice(context.Background()); err == nil {
		t.Fatal("tape replay")
	}
	if err := owned.Close(); err != nil || frees != 2 {
		t.Fatal(err, frees)
	}
	if _, err := owned.OutputBuffer(2); err == nil {
		t.Fatal("closed output")
	}
	if owned.RunDevice(context.Background()) == nil {
		t.Fatal("closed launch")
	}
	before := allocations
	for _, bindings := range []map[int]train.ResidentBuffer{{-1: parameter}, {2: parameter}, {0: &ptxResidentBuffer{buffer: &Buffer{Ptr: 1 << 20, Size: 4}}}, {0: parameter, 1: parameter}, {0: nil}} {
		if _, err := d.NewBoundTape(context.Background(), tape, bindings, 1024); err == nil {
			t.Fatal("bad borrowed input")
		}
	}
	if allocations != before {
		t.Fatal("invalid binding allocated")
	}
}
