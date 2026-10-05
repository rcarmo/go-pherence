package vulkan

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
)

func vkLifetimeTape() train.Tape {
	t := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}}, Outputs: []int{5}}
	for i := 0; i < 4; i++ {
		t.Tensors = append(t.Tensors, train.TensorSpec{Shape: []int{2}})
	}
	t.Commands = []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}, {Op: "vector", Inputs: []int{2, 0}, Output: 3, Scale: 1}, {Op: "vector", Inputs: []int{3, 1}, Output: 4, Scale: 1}, {Op: "vector", Inputs: []int{4, 1}, Output: 5, Scale: 1}}
	return t
}
func TestVulkanOfflineTapeStorage(t *testing.T) {
	_, k, _, _ := newPlanMock(t)
	k.numBuffers = 3
	k.pushSize = 20
	op := &VkTrainingF32{vector: k}
	tape := vkLifetimeTape()
	owner, err := newVkTrainingTapeBound(context.Background(), op, tape, nil, 1<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(owner.arenas) != 4 || len(owner.plans) != 1 || owner.tensors[2].arena != owner.tensors[4].arena || owner.tensors[3].arena != owner.tensors[5].arena {
		t.Fatal("equal-size pooled storage")
	}
	for _, c := range tape.Commands {
		out := owner.tensors[c.Output]
		for _, id := range c.Inputs {
			if out.arena == owner.tensors[id].arena {
				t.Fatal("dispatch alias")
			}
		}
	}
	// Plan execution verifies acquire/between/release barrier ABI through mock;
	// these include compute shader writes -> reads AND writes for reuse.
	if err = owner.runDevice(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err = owner.Close(); err != nil {
		t.Fatal("unique owner close", err)
	}
	legacy, err := newVkTrainingTapeBound(context.Background(), op, tape, nil, 1<<20)
	if err != nil || len(legacy.arenas) != 6 {
		t.Fatal("legacy storage changed", err)
	}
	if err = legacy.Close(); err != nil {
		t.Fatal(err)
	}
	// Shape aliases propagate terminal retention through reshape chains.
	tape.Tensors = append(tape.Tensors, train.TensorSpec{Shape: []int{1, 2}}, train.TensorSpec{Shape: []int{2}})
	tape.Commands = append(tape.Commands, train.Command{Op: "reshape", Inputs: []int{2}, Output: 6}, train.Command{Op: "reshape", Inputs: []int{6}, Output: 7})
	tape.Outputs = []int{5, 7}
	pinned, err := newVkTrainingTapeBound(context.Background(), op, tape, nil, 1<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	if pinned.tensors[2].arena != pinned.tensors[7].arena || pinned.tensors[2].arena == pinned.tensors[4].arena {
		t.Fatal("alias root overwritten")
	}
	if err = pinned.Close(); err != nil {
		t.Fatal(err)
	}
}
func TestVulkanOfflineTapeStorageRetention(t *testing.T) {
	mock, k, _, _ := newPlanMock(t)
	k.numBuffers = 3
	k.pushSize = 20
	op := &VkTrainingF32{vector: k}
	ctx := context.Background()
	arena, err := NewVkTensorArena(ctx, 8)
	if err != nil {
		t.Fatal(err)
	}
	param, err := arena.AllocF32(ctx, 2)
	if err != nil {
		t.Fatal(err)
	}
	d := &VulkanResidentDriver{Operator: op}
	bound, err := d.NewReuseBoundTape(ctx, vkLifetimeTape(), map[int]train.ResidentBuffer{0: &vkResidentBuffer{tensor: param}}, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	owner := bound.(*vkResidentTape).tape
	if len(owner.arenas) != 3 || owner.tensors[0].arena != param.arena {
		t.Fatal("borrowed parameter not pinned")
	}
	if err = bound.RunDevice(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err = bound.OutputBuffer(2); err == nil {
		t.Fatal("dead intermediate escaped")
	}
	if _, err = bound.DownloadOutputs(ctx, []int{2}); err == nil {
		t.Fatal("dead intermediate downloaded")
	}
	if err = bound.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = param.ReshapeF32(ctx, 2); err != nil {
		t.Fatal("borrowed owner freed")
	}
	mock.failure = "sets"
	rollback, err := newVkTrainingTapeBound(ctx, op, vkLifetimeTape(), nil, 1<<20, true)
	if err == nil || rollback != nil {
		t.Fatal("construction cleanup", err)
	}
	mock.failure = ""
	failed, err := newVkTrainingTapeBound(ctx, op, vkLifetimeTape(), nil, 1<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	mock.failure = "submit"
	if err = failed.runDevice(ctx); err == nil || failed.terminal == nil || failed.Close() == nil {
		t.Fatal("submitted failure retained")
	}
	for _, a := range failed.arenas {
		if a.state.buffer.closed {
			t.Fatal("uncertain allocation freed")
		}
	}
	if err = failed.runDevice(ctx); err == nil {
		t.Fatal("retry")
	}
}

func TestVulkanOfflineTapeStoragePlanChunks(t *testing.T) {
	_, k, _, _ := newPlanMock(t)
	k.numBuffers = 3
	k.pushSize = 20
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}}}
	for i := 0; i < 65; i++ {
		id := len(tape.Tensors)
		tape.Tensors = append(tape.Tensors, train.TensorSpec{Shape: []int{2}})
		source := id - 1
		if i == 0 {
			source = 0
		}
		tape.Commands = append(tape.Commands, train.Command{Op: "vector", Inputs: []int{source, 1}, Output: id, Scale: 1})
	}
	tape.Outputs = []int{66}
	owner, err := newVkTrainingTapeBound(context.Background(), &VkTrainingF32{vector: k}, tape, nil, 1<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(owner.plans) != 2 || len(owner.arenas) != 4 {
		t.Fatal("pooled multi-plan", len(owner.plans), len(owner.arenas))
	}
	if err = owner.runDevice(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err = owner.Close(); err != nil {
		t.Fatal(err)
	}
}
