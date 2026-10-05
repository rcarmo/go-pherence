package vulkan

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
)

func TestVulkanOfflineTrainingTapeOwnership(t *testing.T) {
	mock, k, _, _ := newPlanMock(t)
	k.numBuffers = 3
	k.pushSize = 20
	op := &VkTrainingF32{vector: k}
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}, {Shape: []int{2}}, {Shape: []int{1, 2}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}, {Op: "reshape", Inputs: []int{2}, Output: 3}}, Outputs: []int{3}}
	owner, err := NewVkTrainingTape(context.Background(), op, tape, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if len(owner.arenas) != 3 || len(owner.plans) != 1 || owner.tensors[2].arena != owner.tensors[3].arena {
		t.Fatal("reshape retained allocation")
	}
	if _, err := owner.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	mock.failure = "sets"
	owner, err = NewVkTrainingTape(context.Background(), op, tape, 1024)
	if err == nil || owner != nil {
		t.Fatal("construction rollback", owner, err)
	}
}
func TestVulkanOfflineTrainingReshape(t *testing.T) {
	_, _, _ = newLifetimeMock(t)
	mockMemory(t)
	a := mustArena(t, 64)
	x := mustTensor(t, a, 2, 2)
	view, err := x.ReshapeF32(context.Background(), 4)
	if err != nil || view.arena != x.arena || view.rank != 1 || x.rank != 2 {
		t.Fatal(view, err)
	}
	if _, err := x.ReshapeF32(context.Background(), 5); err == nil {
		t.Fatal("element mismatch")
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := view.ReshapeF32(context.Background(), 2, 2); err == nil {
		t.Fatal("closed reshape")
	}
}
func TestVulkanOfflineTrainingConstantTape(t *testing.T) {
	_, _, _ = newLifetimeMock(t)
	mockMemory(t)
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{7}}}, Outputs: []int{0}}
	owner, err := NewVkTrainingTape(context.Background(), &VkTrainingF32{}, tape, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(owner.plans) != 0 || len(owner.arenas) != 1 {
		t.Fatal("constant tape dispatch/allocation")
	}
	values, err := owner.Run(context.Background())
	if err != nil || len(values) != 1 || len(values[0]) != 1 || values[0][0] != 7 {
		t.Fatal(values, err)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
}
