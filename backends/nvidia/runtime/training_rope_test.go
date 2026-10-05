package nvidia

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineRoPE(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	mode, calls := uint32(0), 0
	x, table, dst := &Buffer{Ptr: 4096, Size: 4 * 13 * 3 * 14}, &Buffer{Ptr: 16384, Size: 4 * 13 * 14}, &Buffer{Ptr: 32768, Size: 4 * 13 * 3 * 14}
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		a := unsafe.Slice((*unsafe.Pointer)(args), 7)
		if fn != 15 || gx != 2 || gy != 1 || gz != 1 || bx != 256 || by != 1 || bz != 1 || shared != 0 || stream != 0 || *(*CUdeviceptr)(a[0]) != x.Ptr || *(*CUdeviceptr)(a[1]) != table.Ptr || *(*CUdeviceptr)(a[2]) != dst.Ptr || *(*uint32)(a[3]) != 13 || *(*uint32)(a[4]) != 3 || *(*uint32)(a[5]) != 14 || *(*uint32)(a[6]) != mode {
			t.Fatal("RoPE ABI")
		}
		return CUDA_SUCCESS
	}
	makeOp := func() *TrainingF32 {
		return &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_rope": 15}}}
	}
	op := makeOp()
	for mode = 0; mode < 2; mode++ {
		if err := op.RoPE(dst, x, table, 13, 3, 14, mode); err != nil {
			t.Fatal(err)
		}
	}
	if calls != 2 {
		t.Fatal(calls)
	}
	for _, bad := range []*TrainingF32{nil, {}, {dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{}}}} {
		if err := bad.RoPE(dst, x, table, 13, 3, 14, 0); err == nil {
			t.Fatal("unowned kernel")
		}
	}
	if op.RoPE(dst, x, table, 13, 3, 13, 0) == nil || op.RoPE(x, x, table, 13, 3, 14, 0) == nil || op.RoPE(table, x, table, 13, 3, 14, 0) == nil || op.RoPE(dst, nil, table, 13, 3, 14, 0) == nil || calls != 2 {
		t.Fatal("invalid call launched")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		calls++
		return 201
	}
	if err := op.RoPE(dst, x, table, 13, 3, 14, 0); err == nil || op.poison == nil {
		t.Fatal("launch failure not terminal")
	}
	before := calls
	if err := op.RoPE(dst, x, table, 13, 3, 14, 0); err == nil || calls != before {
		t.Fatal("uncertain rotation retried")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	cuCtxSynchronize = func() CUresult { return 201 }
	op = makeOp()
	if err := op.RoPE(dst, x, table, 13, 3, 14, 1); err == nil || op.poison == nil {
		t.Fatal("sync failure not terminal")
	}
}
func TestPTXTrainingOfflineRoPETape(t *testing.T) {
	// Tape command derives all three dimensions from retained metadata; no
	// allocation, launch or driver symbols are used by this closed-owner check.
	tape := &TrainingTape{op: &TrainingF32{}, buffers: []*Buffer{{Ptr: 4096, Size: 48}, {Ptr: 8192, Size: 24}, {Ptr: 16384, Size: 48}}, shapes: [][]int{{1, 2, 6}, {1, 3, 2}, {1, 2, 6}}}
	if err := tape.command(train.Command{Op: "rope", Inputs: []int{0, 1}, Output: 2, Mode: 1}); err == nil {
		t.Fatal("closed tape operator admitted")
	}
}
