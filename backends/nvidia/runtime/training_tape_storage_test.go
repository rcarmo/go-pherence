package nvidia

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func ptxLifetimeTape() train.Tape {
	tensors := []train.TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}}
	for i := 0; i < 4; i++ {
		tensors = append(tensors, train.TensorSpec{Shape: []int{2}})
	}
	return train.Tape{Tensors: tensors, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}, {Op: "vector", Inputs: []int{2, 0}, Output: 3, Scale: 1}, {Op: "vector", Inputs: []int{3, 1}, Output: 4, Scale: 1}, {Op: "vector", Inputs: []int{4, 1}, Output: 5, Scale: 1}}, Outputs: []int{5}}
}
func TestPTXTrainingOfflineTapeStorage(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldAlloc, oldFree := cuCtxSetCurrent, cuMemAlloc, cuMemFree
	oldH, oldD, oldSync, oldLaunch := cuMemcpyHtoD, cuMemcpyDtoH, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuMemAlloc, cuMemFree = oldSet, oldAlloc, oldFree
		cuMemcpyHtoD, cuMemcpyDtoH, cuCtxSynchronize, cuLaunchKernel = oldH, oldD, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	allocs, frees, uploads := 0, 0, 0
	ptr := uintptr(4096)
	memory := map[CUdeviceptr][]float32{}
	cuMemAlloc = func(out *CUdeviceptr, size uint64) CUresult {
		allocs++
		ptr += 4096
		*out = CUdeviceptr(ptr)
		memory[*out] = make([]float32, size/4)
		return CUDA_SUCCESS
	}
	cuMemFree = func(p CUdeviceptr) CUresult { frees++; delete(memory, p); return CUDA_SUCCESS }
	cuMemcpyHtoD = func(p CUdeviceptr, x unsafe.Pointer, size uint64) CUresult {
		uploads++
		copy(memory[p], unsafe.Slice((*float32)(x), size/4))
		return CUDA_SUCCESS
	}
	cuMemcpyDtoH = func(x unsafe.Pointer, p CUdeviceptr, size uint64) CUresult {
		copy(unsafe.Slice((*float32)(x), size/4), memory[p])
		return CUDA_SUCCESS
	}
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		a := unsafe.Slice((*unsafe.Pointer)(args), 8)
		ap, bp, op := *(*CUdeviceptr)(a[0]), *(*CUdeviceptr)(a[1]), *(*CUdeviceptr)(a[2])
		if ap == op || bp == op {
			t.Fatal("dispatch alias")
		}
		scale := *(*float32)(a[7])
		for i := range memory[op] {
			memory[op][i] = memory[ap][i] + scale*memory[bp][i]
		}
		return CUDA_SUCCESS
	}
	op := &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_vector": 7}}}
	tape := ptxLifetimeTape()
	owner, err := newTrainingTapeBound(op, tape, nil, 1<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 4 || uploads != 2 || owner.buffers[2] != owner.buffers[4] || owner.buffers[3] != owner.buffers[5] {
		t.Fatal("pooled allocation", allocs, uploads)
	}
	out, err := owner.Run()
	if err != nil || len(out) != 1 || out[0][0] != 11 || out[0][1] != 16 {
		t.Fatal("arithmetic", out, err)
	}
	if err = owner.Close(); err != nil || frees != 4 {
		t.Fatal("unique owner close", err, frees)
	}
	// Legacy constructor keeps six independent allocations.
	before := allocs
	legacy, err := newTrainingTapeBound(op, tape, nil, 1<<20)
	if err != nil || allocs-before != 6 {
		t.Fatal("legacy storage changed", err)
	}
	if err = legacy.Close(); err != nil {
		t.Fatal(err)
	}
	// Constants/borrowed parameter buffers never enter reusable intermediate slots.
	d := &PTXResidentDriver{Operator: op}
	param := &ptxResidentBuffer{buffer: &Buffer{Ptr: 1 << 20, Size: 8}}
	memory[param.buffer.Ptr] = []float32{1, 2}
	before = allocs
	bound, err := d.NewReuseBoundTape(context.Background(), tape, map[int]train.ResidentBuffer{0: param}, 1<<20)
	if err != nil || allocs-before != 3 {
		t.Fatal("borrowed storage", err)
	}
	if err = bound.RunDevice(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err = bound.OutputBuffer(2); err == nil {
		t.Fatal("dead intermediate escaped")
	}
	if _, err = bound.DownloadOutputs(context.Background(), []int{2}); err == nil {
		t.Fatal("dead intermediate downloaded")
	}
	if err = bound.Close(); err != nil {
		t.Fatal(err)
	}
	before = frees
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return 201
	}
	failed, err := newTrainingTapeBound(op, tape, nil, 1<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = failed.Run(); err == nil || failed.Close() == nil || frees != before {
		t.Fatal("uncertain storage freed")
	}
	if _, err = failed.Run(); err == nil {
		t.Fatal("retry")
	}
}
