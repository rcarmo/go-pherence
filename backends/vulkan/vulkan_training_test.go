package vulkan

import (
	"context"
	"math"
	"reflect"
	"testing"
	"unsafe"

	train "github.com/rcarmo/go-pherence/backends/training"
)

func TestVulkanOfflineTrainingShaderContracts(t *testing.T) {
	for _, tc := range []struct {
		code          []byte
		buffers, push uint32
	}{{spirvTrainingAdamW, 31, 40}, {spirvTrainingLinearBackward, 15, 16}, {spirvTrainingSiLUBackward, 7, 4}} {
		c, err := InspectVulkanShader(tc.code)
		if err != nil || c.LocalSize != [3]uint32{256, 1, 1} || c.SharedBytes != 0 || c.StorageBindings != tc.buffers || c.PushBytes != tc.push {
			t.Fatal(c, err)
		}
	}
	// Sqrt is the only newly admitted extended instruction. Arbitrary
	// unary GLSL instructions remain outside the pipeline contract.
	for _, instruction := range []uint32{30, 33, 43} {
		bad := editSPIRV(spirvTrainingAdamW, 12, func(a []uint32) { a[4] = instruction })
		if _, err := InspectVulkanShader(bad); err == nil {
			t.Fatal("unrelated extended instruction admitted", instruction)
		}
	}
}
func TestVulkanOfflineTrainingAdamWBindings(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	kernel.numBuffers = 5
	kernel.pushSize = 40
	op := &VkTrainingF32{adam: kernel}
	p, err := train.NewAdamWStep(.01, .9, .999, 1e-8, .01, .9, .5, 2)
	if err != nil {
		t.Fatal(err)
	}
	makeInputs := func() []*VkTensorF32 {
		out := make([]*VkTensorF32, 5)
		for i := range out {
			out[i] = linearMetadataTensor(i+1, 257)
		}
		return out
	}
	inputs := makeInputs()
	s, err := op.AdamWStage(context.Background(), inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], p)
	if err != nil || s.Groups != [3]uint32{2, 1, 1} || len(s.PushWords) != 10 || s.PushWords[9] != math.Float32bits(.5) {
		t.Fatal(s, err)
	}
	for i := range inputs {
		bad := append([]*VkTensorF32(nil), inputs...)
		bad[i] = nil
		if _, err := op.AdamWStage(context.Background(), bad[0], bad[1], bad[2], bad[3], bad[4], p); err == nil {
			t.Fatal("nil admitted", i)
		}
	}
	for i := 0; i < 5; i++ {
		for j := i + 1; j < 5; j++ {
			bad := append([]*VkTensorF32(nil), inputs...)
			bad[j] = bad[i]
			if _, err := op.AdamWStage(context.Background(), bad[0], bad[1], bad[2], bad[3], bad[4], p); err == nil {
				t.Fatal("alias admitted", i, j)
			}
		}
	}
	inputs[2].shape[0]--
	if _, err := op.AdamWStage(context.Background(), inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], p); err == nil {
		t.Fatal("shape mismatch admitted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	inputs = makeInputs()
	if _, err := op.AdamWStage(ctx, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], p); err == nil {
		t.Fatal("cancel admitted")
	}
	p.GradientScale = -1
	if _, err := op.AdamWStage(context.Background(), inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], p); err == nil {
		t.Fatal("invalid constants")
	}
}
func TestVulkanOfflineTrainingSiLU(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	kernel.numBuffers = 3
	kernel.pushSize = 4
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{silu: kernel}
	x, dy, dst := linearMetadataTensor(1, 257), linearMetadataTensor(2, 257), linearMetadataTensor(3, 257)
	stage, err := op.SiLUBackwardStage(context.Background(), dst, x, dy)
	if err != nil || stage.Groups != [3]uint32{2, 1, 1} {
		t.Fatal(stage, err)
	}
	if _, err := op.SiLUBackwardStage(context.Background(), x, x, dy); err == nil {
		t.Fatal("alias admitted")
	}
	dy.shape[0] = 256
	if _, err := op.SiLUBackwardStage(context.Background(), dst, x, dy); err == nil {
		t.Fatal("shape admitted")
	}
}
func TestVulkanOfflineTrainingLinearBindings(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	kernel.numBuffers = 4
	kernel.pushSize = 16
	op := &VkTrainingF32{linear: kernel}
	x, w, dy := linearMetadataTensor(1, 3, 5), linearMetadataTensor(2, 7, 5), linearMetadataTensor(3, 3, 7)
	outputs := []*VkTensorF32{linearMetadataTensor(4, 3, 5), linearMetadataTensor(5, 7, 5), linearMetadataTensor(6, 7)}
	for mode, dst := range outputs {
		s, err := op.LinearBackwardStage(context.Background(), dst, x, w, dy, uint32(mode))
		if err != nil || !reflect.DeepEqual(s.PushWords, []uint32{3, 5, 7, uint32(mode)}) {
			t.Fatal(mode, s, err)
		}
	}
	for _, mode := range []uint32{0, 1, 2, 3} {
		if _, err := op.LinearBackwardStage(context.Background(), x, x, w, dy, mode); err == nil {
			t.Fatal("alias/rank/mode admitted", mode)
		}
	}
	dy.shape[0] = 4
	if _, err := op.LinearBackwardStage(context.Background(), outputs[0], x, w, dy, 0); err == nil {
		t.Fatal("input mismatch")
	}
}
