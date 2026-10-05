package vulkan

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingEOSAdmission(t *testing.T) {
	info, err := InspectVulkanShader(spirvTrainingEOSAdmission)
	if err != nil || info.StorageBindings != 15 || info.PushBytes != 12 || info.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal(info, err)
	}
	_, k, _ := newLifetimeMock(t)
	k.numBuffers = 4
	k.pushSize = 12
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{eosAdmission: k}
	ctx := context.Background()
	hidden, weight, bias, out := linearMetadataTensor(1, 6), linearMetadataTensor(2, 2), linearMetadataTensor(3, 1), linearMetadataTensor(4, 1)
	stage, err := op.EOSAdmissionStage(ctx, hidden, weight, bias, out, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	_, budget, _ := train.EOSAdmissionShape(3, 2)
	for i, want := range []uint32{3, 2, budget} {
		if stage.PushWords[i] != want {
			t.Fatal("push")
		}
	}
	if stage.Groups != [3]uint32{1, 1, 1} {
		t.Fatal("groups")
	}
	for _, kind := range []string{"nil", "owner", "rows", "extent", "alias", "cancel"} {
		var err error
		switch kind {
		case "nil":
			_, err = (*VkTrainingF32)(nil).EOSAdmissionStage(ctx, hidden, weight, bias, out, 3, 2)
		case "owner":
			_, err = (&VkTrainingF32{}).EOSAdmissionStage(ctx, hidden, weight, bias, out, 3, 2)
		case "rows":
			_, err = op.EOSAdmissionStage(ctx, hidden, weight, bias, out, 0, 2)
		case "extent":
			_, err = op.EOSAdmissionStage(ctx, weight, weight, bias, out, 3, 2)
		case "alias":
			_, err = op.EOSAdmissionStage(ctx, hidden, weight, bias, bias, 3, 2)
		case "cancel":
			cancelled, cancel := context.WithCancel(ctx)
			cancel()
			_, err = op.EOSAdmissionStage(cancelled, hidden, weight, bias, out, 3, 2)
		}
		if err == nil {
			t.Fatal("bad stage", kind)
		}
	}
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{6}, Values: make([]float32, 6)}, {Shape: []int{2}, Values: make([]float32, 2)}, {Shape: []int{1}, Values: []float32{0}}, {Shape: []int{1}}}, Commands: []train.Command{{Op: "eos-admission", Inputs: []int{0, 1, 2}, Output: 3, Rows: 3, Width: 2}}, Outputs: []int{3}}
	if _, err := op.tapeStage(ctx, []*VkTensorF32{hidden, weight, bias, out}, tape.Commands[0]); err != nil {
		t.Fatal("tape dispatch", err)
	}
	if op.admitTapeKernels(tape) != nil || (&VkTrainingF32{}).admitTapeKernels(tape) == nil {
		t.Fatal("preflight")
	}
}

func TestVulkanOfflineTrainingEOSAdmissionLateConstruction(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	calls := 0
	op, err := newVkTrainingF32Locked(func([]byte, int, int) (*VkComputeKernel, error) {
		calls++
		if calls == 29 {
			return nil, fmt.Errorf("EOS admission create")
		}
		return kernel, nil
	})
	if err == nil || op != nil || calls != 29 || !kernel.closed {
		t.Fatal("late construction rollback", err, calls)
	}
}
