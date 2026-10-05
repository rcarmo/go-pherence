package nvidia

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineTapeOwnership(t *testing.T) {
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
	allocs, frees := 0, 0
	cuMemAlloc = func(p *CUdeviceptr, n uint64) CUresult {
		allocs++
		*p = CUdeviceptr(4096 * allocs)
		return CUDA_SUCCESS
	}
	cuMemFree = func(CUdeviceptr) CUresult { frees++; return CUDA_SUCCESS }
	cuMemcpyHtoD = func(CUdeviceptr, unsafe.Pointer, uint64) CUresult { return CUDA_SUCCESS }
	cuMemcpyDtoH = func(p unsafe.Pointer, d CUdeviceptr, n uint64) CUresult {
		for i := range unsafe.Slice((*float32)(p), int(n/4)) {
			unsafe.Slice((*float32)(p), int(n/4))[i] = 42
		}
		return CUDA_SUCCESS
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}, {Shape: []int{2}}, {Shape: []int{1, 2}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}, {Op: "reshape", Inputs: []int{2}, Output: 3}}, Outputs: []int{3}}
	op := &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_vector": 14}}}
	owned, err := NewTrainingTape(op, tape, 1024)
	if err != nil || allocs != 3 {
		t.Fatal(err, allocs)
	}
	output, err := owned.Run()
	if err != nil || output[0][0] != 42 {
		t.Fatal(err, output)
	}
	if err := owned.Close(); err != nil || frees != 3 {
		t.Fatal(err, frees)
	}
	// Native launch failure: no free, no retry; resource owner remains reachable.
	owned, err = NewTrainingTape(op, tape, 1024)
	if err != nil {
		t.Fatal(err)
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return 201
	}
	if _, err := owned.Run(); err == nil {
		t.Fatal("failure accepted")
	}
	if err := owned.Close(); err == nil || frees != 3 {
		t.Fatal("uncertain storage freed", err, frees)
	}
	if _, err := owned.Run(); err == nil {
		t.Fatal("terminal tape retried")
	}
}
func TestPTXTrainingOfflineConstantTape(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldAlloc, oldFree, oldUp, oldDown, oldLaunch := cuCtxSetCurrent, cuMemAlloc, cuMemFree, cuMemcpyHtoD, cuMemcpyDtoH, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuMemAlloc, cuMemFree, cuMemcpyHtoD, cuMemcpyDtoH, cuLaunchKernel = oldSet, oldAlloc, oldFree, oldUp, oldDown, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	allocs, frees, launches := 0, 0, 0
	var value float32
	cuMemAlloc = func(p *CUdeviceptr, n uint64) CUresult { allocs++; *p = 4096; return CUDA_SUCCESS }
	cuMemFree = func(CUdeviceptr) CUresult { frees++; return CUDA_SUCCESS }
	cuMemcpyHtoD = func(_ CUdeviceptr, p unsafe.Pointer, _ uint64) CUresult { value = *(*float32)(p); return CUDA_SUCCESS }
	cuMemcpyDtoH = func(p unsafe.Pointer, _ CUdeviceptr, _ uint64) CUresult { *(*float32)(p) = value; return CUDA_SUCCESS }
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		launches++
		return 201
	}
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{7}}}, Outputs: []int{0}}
	owner, err := NewTrainingTape(&TrainingF32{}, tape, 4)
	if err != nil {
		t.Fatal(err)
	}
	out, err := owner.Run()
	if err != nil || out[0][0] != 7 || launches != 0 || allocs != 1 {
		t.Fatal("constant tape", out, err, launches, allocs)
	}
	if err := owner.Close(); err != nil || frees != 1 {
		t.Fatal(err, frees)
	}
}
