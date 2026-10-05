package nvidia

import (
	"errors"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineAttentionForward(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	b := make([]*Buffer, 5)
	for i := range b {
		b[i] = &Buffer{Ptr: CUdeviceptr(8192 * (i + 1)), Size: 4 * 13 * 14 * 3}
	}
	mode, calls := uint32(0), 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		a := unsafe.Slice((*unsafe.Pointer)(args), 11)
		groups := uint32(2)
		if mode == 1 {
			groups = 1
		}
		if mode == 2 {
			groups = 3
		}
		if fn != 16 || gx != groups || *(*uint32)(a[5]) != 13 || *(*uint32)(a[6]) != 3 || *(*uint32)(a[7]) != 14 || *(*uint32)(a[8]) != 2 || *(*uint32)(a[9]) != mode || *(*float32)(a[10]) != .25 {
			t.Fatal("forward attention ABI")
		}
		for i := 0; i < 5; i++ {
			if *(*CUdeviceptr)(a[i]) != b[i].Ptr {
				t.Fatal("forward pointer ABI")
			}
		}
		return CUDA_SUCCESS
	}
	makeOp := func() *TrainingF32 {
		return &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_attention_forward": 16}}}
	}
	op := makeOp()
	for mode = 0; mode < 3; mode++ {
		if err := op.AttentionForward(b[4], b[0], b[1], b[2], b[3], 13, 3, 14, 2, mode, .25); err != nil {
			t.Fatal(err)
		}
	}
	if calls != 3 {
		t.Fatal(calls)
	}
	for _, bad := range []*TrainingF32{nil, {}, {dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{}}}} {
		if err := bad.AttentionForward(b[4], b[0], b[1], b[2], b[3], 13, 3, 14, 2, 0, .25); err == nil {
			t.Fatal("bad operator")
		}
	}
	if op.AttentionForward(b[0], b[0], b[1], b[2], b[3], 13, 3, 14, 2, 0, .25) == nil || op.AttentionForward(b[4], nil, b[1], b[2], b[3], 13, 3, 14, 2, 0, .25) == nil || op.AttentionForward(b[4], b[0], b[1], b[2], b[3], 13, 3, 14, 14, 0, .25) == nil || calls != 3 {
		t.Fatal("invalid forward launched")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		calls++
		return 201
	}
	if err := op.AttentionForward(b[4], b[0], b[1], b[2], b[3], 13, 3, 14, 2, 0, .25); err == nil || op.poison == nil {
		t.Fatal("failure not terminal")
	}
	before := calls
	if err := op.AttentionForward(b[4], b[0], b[1], b[2], b[3], 13, 3, 14, 2, 0, .25); err == nil || calls != before {
		t.Fatal("terminal forward retried")
	}
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	cuCtxSynchronize = func() CUresult { return 201 }
	op = makeOp()
	if err := op.AttentionForward(b[4], b[0], b[1], b[2], b[3], 13, 3, 14, 2, 2, .25); err == nil || op.poison == nil {
		t.Fatal("sync failure not terminal")
	}
	tape := &TrainingTape{op: &TrainingF32{}, buffers: b, shapes: [][]int{{13, 3, 14}, {13, 3, 14}, {13, 3, 14}, {3, 13, 13}, {3, 13, 13}}}
	if err := tape.command(train.Command{Op: "attention-forward", Inputs: []int{0, 1, 2, 3}, Output: 4, Scale: .25}); err == nil {
		t.Fatal("closed tape admitted")
	}
}

func TestPTXTrainingOfflineAttentionTapePreflight(t *testing.T) {
	dualNames := map[string]CUfunction{"training_rope": 1, "training_attention_forward": 2}
	op := &TrainingF32{dualModule: &PTXModule{handle: 1, functions: dualNames}}
	tape := train.Tape{Commands: []train.Command{{Op: "rope"}, {Op: "attention-forward"}, {Op: "attention-backward"}}}
	if err := op.admitTapeKernels(tape); err == nil {
		t.Fatal("missing backward module admitted")
	}
	op.module = &PTXModule{handle: 2, functions: map[string]CUfunction{"training_attention_backward": 3}}
	if err := op.admitTapeKernels(tape); err != nil {
		t.Fatal(err)
	}
	// Cover every admitted operation without any CUDA symbol calls.
	for _, entry := range []struct{ op, name string }{{"linear-forward", "training_linear_dual_forward"}, {"linear-backward", "training_linear_dual_backward"}, {"silu-forward", "training_silu_dual_forward"}, {"silu-backward", "training_silu_dual_adjoint"}, {"norm-forward", "training_norm_dual_forward"}, {"norm-backward", "training_norm_dual_backward"}, {"mod-forward", "training_mod_dual_forward"}, {"mod-backward", "training_mod_dual_backward"}, {"vector", "training_vector"}} {
		dualNames[entry.name] = 4
		tape.Commands = append(tape.Commands, train.Command{Op: entry.op})
	}
	tape.Commands = append(tape.Commands, train.Command{Op: "reshape"})
	if err := op.admitTapeKernels(tape); err != nil {
		t.Fatal(err)
	}
	delete(dualNames, "training_rope")
	if err := op.admitTapeKernels(tape); err == nil {
		t.Fatal("missing first entry")
	}
	dualNames["training_rope"] = 1
	op.poison = errors.New("uncertain")
	if op.admitTapeKernels(tape) == nil {
		t.Fatal("terminal owner")
	}
	op.poison = nil
	tape.Commands = append(tape.Commands, train.Command{Op: "unknown"})
	if op.admitTapeKernels(tape) == nil {
		t.Fatal("unknown entry")
	}
	var nilOp *TrainingF32
	if nilOp.admitTapeKernels(tape) == nil {
		t.Fatal("nil owner")
	}
}
