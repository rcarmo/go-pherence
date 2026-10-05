package vulkan

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
)

func TestVulkanOfflineResidentTapeBinding(t *testing.T) {
	mock, k, _, _ := newPlanMock(t)
	k.numBuffers = 3
	k.pushSize = 20
	op := &VkTrainingF32{vector: k}
	d := &VulkanResidentDriver{Operator: op}
	ctx := context.Background()
	arena := mustArena(t, 8)
	parameter := mustTensor(t, arena, 2)
	if err := parameter.Upload(ctx, []float32{.3, -.7}); err != nil {
		t.Fatal(err)
	}
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{99, 99}}, {Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}}, Outputs: []int{2}}
	input := &vkResidentBuffer{arena: arena, tensor: parameter}
	owned, err := d.NewBoundTape(ctx, tape, map[int]train.ResidentBuffer{0: input}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	concrete := owned.(*vkResidentTape)
	if len(concrete.tape.arenas) != 2 || concrete.tape.tensors[0].arena != parameter.arena {
		t.Fatal("borrowed allocation not shared")
	}
	if _, err := owned.OutputBuffer(2); err == nil {
		t.Fatal("output before execution")
	}
	if err := owned.RunDevice(ctx); err != nil {
		t.Fatal(err)
	}
	output, err := owned.OutputBuffer(2)
	if err != nil {
		t.Fatal(err)
	}
	if output.Close() == nil {
		t.Fatal("borrowed gradient closed")
	}
	values, err := owned.DownloadOutputs(ctx, []int{2})
	if err != nil || len(values[0]) != 2 {
		t.Fatal(err)
	}
	if owned.RunDevice(ctx) == nil {
		t.Fatal("tape replay")
	}
	if err := owned.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := parameter.ReshapeF32(ctx, 2); err != nil {
		t.Fatal("borrowed parameter freed", err)
	}
	if _, err := owned.OutputBuffer(2); err == nil {
		t.Fatal("closed output")
	}
	for _, bindings := range []map[int]train.ResidentBuffer{{-1: input}, {2: input}, {0: nil}, {0: input, 1: input}} {
		if _, err := d.NewBoundTape(ctx, tape, bindings, 1024); err == nil {
			t.Fatal("invalid borrowed inputs")
		}
	}
	// ValidateAccumulate accepts reshaped tape gradients, including matrices.
	a, b, out := &vkResidentBuffer{tensor: linearMetadataTensor(1, 2)}, &vkResidentBuffer{tensor: linearMetadataTensor(2, 1, 2)}, &vkResidentBuffer{tensor: linearMetadataTensor(3, 2)}
	if err := d.ValidateAccumulate(ctx, out, a, b, 2, 1); err != nil {
		t.Fatal(err)
	}
	if d.ValidateAccumulate(ctx, a, a, b, 2, 1) == nil {
		t.Fatal("output alias")
	}
	mock.failure = "submit"
	owned, err = d.NewBoundTape(ctx, tape, map[int]train.ResidentBuffer{0: input}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if err := owned.RunDevice(ctx); err == nil {
		t.Fatal("failed submit")
	}
	if owned.Close() == nil {
		t.Fatal("uncertain resources closed")
	}
}
