package nvidia

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
	"unsafe"
)

func ptxZeroConstantTape() train.Tape {
	tensors := []train.TensorSpec{{Shape: []int{2}, Values: []float32{0, 0}}, {Shape: []int{2}, Values: []float32{0, 0}}}
	for i := 0; i < 4; i++ {
		tensors = append(tensors, train.TensorSpec{Shape: []int{2}})
	}
	return train.Tape{Tensors: tensors, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}, {Op: "vector", Inputs: []int{2, 0}, Output: 3, Scale: 1}, {Op: "vector", Inputs: []int{3, 1}, Output: 4, Scale: 1}, {Op: "vector", Inputs: []int{4, 1}, Output: 5, Scale: 1}}, Outputs: []int{5}}
}
func TestPTXTrainingOfflineZeroConstants(t *testing.T) {
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
	tape := ptxZeroConstantTape()
	owner, err := newTrainingTapeBound(op, tape, nil, 1<<20, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if owner.buffers[0] != owner.buffers[1] {
		t.Fatal("zero constants not shared")
	}
	if allocs != 3 || uploads != 1 || owner.buffers[2] != owner.buffers[4] || owner.buffers[3] != owner.buffers[5] {
		t.Fatal("pooled allocation", allocs, uploads)
	}
	out, err := owner.Run()
	if err != nil || len(out) != 1 || out[0][0] != 0 || out[0][1] != 0 {
		t.Fatal("arithmetic", out, err)
	}
	if err = owner.Close(); err != nil || frees != 3 {
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
	bound, err := d.NewZeroBoundTape(context.Background(), tape, map[int]train.ResidentBuffer{0: param}, 1<<20)
	if err != nil || allocs-before != 3 {
		t.Fatal("borrowed storage", err)
	}
	if bound.(*ptxResidentTape).tape.buffers[0] == bound.(*ptxResidentTape).tape.buffers[1] {
		t.Fatal("zero parameter identity shared")
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
	failed, err := newTrainingTapeBound(op, tape, nil, 1<<20, true, true)
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

func TestPTXTrainingOfflineZeroConstantsSignedZero(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldAlloc, oldFree, oldCopy := cuCtxSetCurrent, cuMemAlloc, cuMemFree, cuMemcpyHtoD
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuMemAlloc, cuMemFree, cuMemcpyHtoD = oldSet, oldAlloc, oldFree, oldCopy
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	calls, uploads := 0, 0
	cuMemAlloc = func(out *CUdeviceptr, n uint64) CUresult {
		calls++
		*out = CUdeviceptr(calls * 4096)
		return CUDA_SUCCESS
	}
	cuMemFree = func(CUdeviceptr) CUresult { return CUDA_SUCCESS }
	cuMemcpyHtoD = func(CUdeviceptr, unsafe.Pointer, uint64) CUresult { uploads++; return CUDA_SUCCESS }
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{0}}, {Shape: []int{1}, Values: []float32{0}}, {Shape: []int{1}, Values: []float32{float32(math.Copysign(0, -1))}}}, Outputs: []int{0, 1, 2}}
	op := &TrainingF32{}
	owner, err := newTrainingTapeBound(op, tape, nil, 12, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || uploads != 2 || owner.buffers[0] != owner.buffers[1] || owner.buffers[0] == owner.buffers[2] {
		t.Fatal("signed zero semantics")
	}
	if err = owner.Close(); err != nil {
		t.Fatal(err)
	}
	// Upload uncertainty retains the unique owner rather than retrying/releasing.
	cuMemcpyHtoD = func(CUdeviceptr, unsafe.Pointer, uint64) CUresult { return 201 }
	failed, err := newTrainingTapeBound(op, tape, nil, 12, true, true)
	if err == nil || failed == nil || failed.terminal == nil || failed.Close() == nil {
		t.Fatal("failed upload owner")
	}
}
