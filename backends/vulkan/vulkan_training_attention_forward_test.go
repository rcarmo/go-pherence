package vulkan

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"reflect"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingAttentionForward(t *testing.T) {
	c, err := InspectVulkanShader(spirvTrainingAttentionForward)
	if err != nil || c.LocalSize != [3]uint32{256, 1, 1} || c.PushBytes != 24 || c.StorageBindings != 31 || c.SharedBytes != 0 {
		t.Fatal(c, err)
	}
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 5
	kernel.pushSize = 24
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{attentionForward: kernel}
	q, k, v, aux := linearMetadataTensor(1, 13, 3, 14), linearMetadataTensor(2, 13, 3, 14), linearMetadataTensor(3, 13, 3, 14), linearMetadataTensor(4, 3, 13, 13)
	dst := linearMetadataTensor(5, 3, 13, 13)
	for mode := uint32(0); mode < 3; mode++ {
		if mode == 2 {
			dst = linearMetadataTensor(5, 13, 3, 14)
		}
		s, err := op.AttentionForwardStage(context.Background(), dst, q, k, v, aux, 2, mode, .25)
		groups := uint32(2)
		if mode == 1 {
			groups = 1
		}
		if mode == 2 {
			groups = 3
		}
		if err != nil || s.Groups != [3]uint32{groups, 1, 1} || !reflect.DeepEqual(s.PushWords, []uint32{13, 3, 14, 2, mode, math.Float32bits(.25)}) {
			t.Fatal(s, err)
		}
	}
	for _, bad := range []*VkTrainingF32{nil, {}} {
		if _, err := bad.AttentionForwardStage(context.Background(), dst, q, k, v, aux, 2, 2, .25); err == nil {
			t.Fatal("operator")
		}
	}
	for _, inputs := range [][5]*VkTensorF32{{q, q, k, v, aux}, {dst, nil, k, v, aux}, {dst, q, nil, v, aux}, {dst, q, k, nil, aux}, {dst, q, k, v, nil}, {nil, q, k, v, aux}, {dst, q, k, v, linearMetadataTensor(6, 3, 12, 13)}, {linearMetadataTensor(6, 3, 13, 13), q, k, v, aux}} {
		if _, err := op.AttentionForwardStage(context.Background(), inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], 2, 2, .25); err == nil {
			t.Fatal("invalid alias/shape")
		}
	}
	if _, err := op.AttentionForwardStage(context.Background(), dst, q, k, v, aux, 14, 2, .25); err == nil {
		t.Fatal("context")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := op.AttentionForwardStage(ctx, dst, q, k, v, aux, 2, 2, .25); err == nil {
		t.Fatal("cancelled")
	}
	if _, err := op.tapeStage(context.Background(), []*VkTensorF32{q, k, v, aux, dst}, train.Command{Op: "attention-forward", Inputs: []int{0, 1, 2, 3}, Output: 4, Mode: 2, Offset: 2, Scale: .25}); err != nil {
		t.Fatal("tape routing", err)
	}
}

func TestVulkanOfflineTrainingAttentionTapePreflight(t *testing.T) {
	kernel := &VkComputeKernel{}
	op := &VkTrainingF32{rope: kernel, attentionForward: kernel}
	tape := train.Tape{Commands: []train.Command{{Op: "rope"}, {Op: "attention-forward"}, {Op: "attention-backward"}}}
	if op.admitTapeKernels(tape) == nil {
		t.Fatal("missing backward owner")
	}
	op.attention = kernel
	op.linearDualForward, op.linearDual, op.siluDualForward, op.siluDualAdjoint = kernel, kernel, kernel, kernel
	op.normDualForward, op.normDual, op.modDualForward, op.modDual, op.vector = kernel, kernel, kernel, kernel, kernel
	for _, name := range []string{"linear-forward", "linear-backward", "silu-forward", "silu-backward", "norm-forward", "norm-backward", "mod-forward", "mod-backward", "vector", "reshape"} {
		tape.Commands = append(tape.Commands, train.Command{Op: name})
	}
	if err := op.admitTapeKernels(tape); err != nil {
		t.Fatal(err)
	}
	kernel.closed = true
	if op.admitTapeKernels(tape) == nil {
		t.Fatal("closed owner")
	}
	kernel.closed = false
	tape.Commands = append(tape.Commands, train.Command{Op: "unknown"})
	if op.admitTapeKernels(tape) == nil {
		t.Fatal("unknown entry")
	}
	var nilOp *VkTrainingF32
	if nilOp.admitTapeKernels(tape) == nil {
		t.Fatal("nil owner")
	}
}
