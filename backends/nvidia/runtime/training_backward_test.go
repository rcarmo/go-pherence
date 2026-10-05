package nvidia

import (
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineAttentionPass(t *testing.T) {
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
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		table := unsafe.Slice((*unsafe.Pointer)(args), 13)
		if *(*uint32)(table[11]) != uint32(calls) {
			t.Fatal("pass stage order")
		}
		calls++
		return CUDA_SUCCESS
	}
	op := &TrainingF32{module: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_attention_backward": 5}}}
	b := make([]*Buffer, 10)
	for i := range b {
		b[i] = &Buffer{Ptr: CUdeviceptr(4096 * (i + 1)), Size: 18 * 4}
	}
	if err := op.AttentionBackwardPass(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], 3, 2, 2, 2, .5); err != nil || calls != 5 {
		t.Fatal(err, calls)
	}
	// Any stage error stops the pass immediately; no follow-on derivatives
	// are launched and partially written scratch is not treated as complete.
	calls = 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		if calls == 2 {
			return 201
		}
		return CUDA_SUCCESS
	}
	if err := op.AttentionBackwardPass(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], 3, 2, 2, 2, .5); err == nil || calls != 2 {
		t.Fatal("failed pass continued", err, calls)
	}
	calls = 5
	b[4] = b[0]
	if err := op.AttentionBackwardPass(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], 3, 2, 2, 2, .5); err == nil || calls != 5 {
		t.Fatal("pass alias launched", err, calls)
	}
}
func TestPTXTrainingOfflineBackwardABI(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	syncs := 0
	cuCtxSynchronize = func() CUresult { syncs++; return CUDA_SUCCESS }
	calls := 0
	mode := uint32(0)
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		if gx != 1 || gy != 1 || gz != 1 || bx != 256 {
			t.Fatal("backward grid")
		}
		if fn == 4 {
			table := unsafe.Slice((*unsafe.Pointer)(args), 8)
			if *(*uint32)(table[5]) != 3 || *(*uint32)(table[6]) != 5 || *(*uint32)(table[7]) != mode {
				t.Fatal("norm ABI")
			}
		} else if fn == 5 {
			table := unsafe.Slice((*unsafe.Pointer)(args), 13)
			if *(*uint32)(table[7]) != 3 || *(*uint32)(table[8]) != 2 || *(*uint32)(table[9]) != 2 || *(*uint32)(table[10]) != 2 || *(*uint32)(table[11]) != mode || *(*float32)(table[12]) != .5 {
				t.Fatal("attention ABI")
			}
		} else {
			t.Fatal("wrong function")
		}
		return CUDA_SUCCESS
	}
	op := &TrainingF32{module: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_layernorm_backward": 4, "training_attention_backward": 5}}}
	buffer := func(i, n int) *Buffer { return &Buffer{Ptr: CUdeviceptr((i + 1) * 4096), Size: n * 4} }
	x, inv, w, dy, dst := buffer(0, 15), buffer(1, 3), buffer(2, 5), buffer(3, 15), buffer(4, 18)
	for mode = 0; mode < 3; mode++ {
		if err := op.LayerNormBackward(dst, x, inv, w, dy, 3, 5, mode); err != nil {
			t.Fatal(err)
		}
	}
	q, k, v, p, dout, aux := buffer(0, 12), buffer(1, 12), buffer(2, 12), buffer(3, 18), buffer(5, 12), buffer(6, 18)
	for mode = 0; mode < 5; mode++ {
		if err := op.AttentionBackward(dst, q, k, v, p, dout, aux, 3, 2, 2, 2, mode, .5); err != nil {
			t.Fatal(err)
		}
	}
	if calls != 8 || syncs != 8 {
		t.Fatal(calls, syncs)
	}
	if op.LayerNormBackward(x, x, inv, w, dy, 3, 5, 0) == nil || op.AttentionBackward(q, q, k, v, p, dout, aux, 3, 2, 2, 2, 2, .5) == nil || op.AttentionBackward(dst, q, k, v, p, dout, aux, 3, 2, 2, 4, 2, .5) == nil || calls != 8 {
		t.Fatal("invalid call reached driver")
	}
}
