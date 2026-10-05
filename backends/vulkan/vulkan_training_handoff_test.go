package vulkan

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingHandoffPreflight(t *testing.T) {
	_, k, _ := newLifetimeMock(t)
	k.numBuffers = 3
	k.pushSize = 20
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	d := &VulkanResidentDriver{Operator: &VkTrainingF32{vector: k}}
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}, {Shape: []int{2}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}}, Outputs: []int{2}}
	a, b := &vkResidentBuffer{tensor: linearMetadataTensor(1, 2)}, &vkResidentBuffer{tensor: linearMetadataTensor(2, 2)}
	if err := d.ValidateHandoffTape(context.Background(), tape, map[int]train.ResidentBuffer{0: a, 1: b}, 100); err != nil {
		t.Fatal(err)
	}
	external, err := train.WithExternalTapeInputs(tape, []int{0}, 100)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.ValidateHandoffTape(context.Background(), external, map[int]train.ResidentBuffer{0: a, 1: b}, 100); err != nil {
		t.Fatal(err)
	}
	if d.ValidateHandoffTape(context.Background(), external, nil, 100) == nil {
		t.Fatal("missing external preflight")
	}
	if owner, err := newVkTrainingTapeBound(context.Background(), d.Operator, external, nil, 100); err == nil || owner != nil {
		t.Fatal("unbound external allocated native storage")
	}
	dead := external
	dead.Tensors = append(append([]train.TensorSpec(nil), external.Tensors...), train.TensorSpec{Shape: []int{1}, External: true})
	if d.ValidateHandoffTape(context.Background(), dead, map[int]train.ResidentBuffer{0: a}, 100) == nil {
		t.Fatal("dead external omitted")
	}
	for _, bindings := range []map[int]train.ResidentBuffer{{-1: a}, {2: a}, {0: nil}, {0: &vkResidentBuffer{tensor: linearMetadataTensor(3, 1)}}, {0: a, 1: a}, {0: &vkResidentBuffer{tensor: linearMetadataTensor(3, 2), sourceTape: &VkTrainingTape{closed: true}}}} {
		if d.ValidateHandoffTape(context.Background(), tape, bindings, 100) == nil {
			t.Fatal("bad bindings")
		}
	}
	if (*VulkanResidentDriver)(nil).ValidateHandoffTape(context.Background(), tape, nil, 100) == nil || d.ValidateHandoffTape(nil, tape, nil, 100) == nil {
		t.Fatal("nil")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if d.ValidateHandoffTape(ctx, tape, nil, 100) == nil {
		t.Fatal("cancel")
	}
	invalid := tape
	invalid.Tensors = append(append([]train.TensorSpec(nil), tape.Tensors...), train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
	if d.ValidateHandoffTape(context.Background(), invalid, nil, 100) == nil {
		t.Fatal("full source")
	}
	if (&VulkanResidentDriver{Operator: &VkTrainingF32{}}).ValidateHandoffTape(context.Background(), tape, nil, 100) == nil {
		t.Fatal("kernel")
	}
	d.terminal = context.Canceled
	if d.ValidateHandoffTape(context.Background(), tape, nil, 100) == nil {
		t.Fatal("terminal")
	}
}

func TestVulkanOfflineTrainingHandoffExternalOwner(t *testing.T) {
	newLifetimeMock(t)
	source := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, External: true}}, Outputs: []int{0}}
	borrowed := linearMetadataTensor(1, 2)
	owner, err := newVkTrainingTapeBound(context.Background(), &VkTrainingF32{}, source, map[int]*VkTensorF32{0: borrowed}, 8, true, true)
	if err != nil || owner == nil || len(owner.arenas) != 0 || len(owner.plans) != 0 || owner.tensors[0].Elements() != 2 {
		t.Fatal("borrowed-only construction", err)
	}
	if err := owner.runDevice(context.Background()); err != nil {
		t.Fatal(err)
	}
	if owner.Close() != nil || borrowed.Elements() != 2 {
		t.Fatal("external source freed")
	}
}
