package nvidia

import (
	"testing"
	"unsafe"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// Only Go fake driver calls. Do not Init or load any native module/device.
func TestPTXTrainingOfflineLaunchContract(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	calls, syncs := 0, 0
	cuCtxSynchronize = func() CUresult { syncs++; return CUDA_SUCCESS }
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		if gx != 2 || gy != 1 || gz != 1 || bx != 256 || by != 1 || bz != 1 || shared != 0 {
			t.Fatal("launch geometry")
		}
		table := unsafe.Slice((*unsafe.Pointer)(args), 15)
		if *(*uint32)(table[5]) != 257 || *(*float32)(table[14]) != .5 {
			t.Fatal("argument order")
		}
		return CUDA_SUCCESS
	}
	op := &TrainingF32{module: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_adamw": 1, "training_linear_backward": 2}}}
	p, _ := train.NewAdamWStep(.01, .9, .999, 1e-8, .01, .9, .5, 1)
	inputs := make([]*Buffer, 5)
	for i := range inputs {
		inputs[i] = &Buffer{Ptr: CUdeviceptr(0x1000 + i*0x1000), Size: 257 * 4}
	}
	if err := op.AdamW(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], 257, p); err != nil || calls != 1 || syncs != 1 {
		t.Fatal(err, calls, syncs)
	}
	for i := 0; i < 5; i++ {
		for j := i + 1; j < 5; j++ {
			bad := append([]*Buffer(nil), inputs...)
			bad[j] = bad[i]
			if err := op.AdamW(bad[0], bad[1], bad[2], bad[3], bad[4], 257, p); err == nil {
				t.Fatal("alias admitted")
			}
		}
	}
	if calls != 1 {
		t.Fatal("invalid call launched")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		calls++
		return 201
	}
	if err := op.AdamW(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], 257, p); err == nil || syncs != 1 {
		t.Fatal("launch failure certified", err)
	}
	before := calls
	if err := op.AdamW(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], 257, p); err == nil || calls != before {
		t.Fatal("uncertain update retried", err, calls)
	}
}
func TestPTXTrainingOfflineExtents(t *testing.T) {
	p, _ := train.NewAdamWStep(.01, .9, .999, 1e-8, 0, 0, 1, 1)
	if (*TrainingF32)(nil).AdamW(nil, nil, nil, nil, nil, 1, p) == nil {
		t.Fatal("nil admitted")
	}
	for _, b := range []*Buffer{nil, {}, {Ptr: 1, Size: 4}, {Ptr: 4, Size: 3}, {Ptr: CUdeviceptr(^uint64(0) - 3), Size: 8}} {
		if trainingExtent(b, 1) {
			t.Fatal("bad extent", b)
		}
	}
	op := &TrainingF32{}
	x, w, dy, dst := &Buffer{Ptr: 4096, Size: 60}, &Buffer{Ptr: 8192, Size: 140}, &Buffer{Ptr: 12288, Size: 84}, &Buffer{Ptr: 16384, Size: 140}
	for mode := uint32(0); mode < 4; mode++ {
		if op.LinearBackward(dst, x, w, dy, 3, 5, 7, mode) == nil {
			t.Fatal("closed/mode admitted")
		}
	}
	if op.LinearBackward(x, x, w, dy, 3, 5, 7, 0) == nil {
		t.Fatal("alias admitted")
	}
}
