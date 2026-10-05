package pockettts

import (
	"context"
	"errors"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

type closableHostFlowExecutor struct {
	hostFlowTapeExecutor
	closes int
	err    error
}

func (e *closableHostFlowExecutor) Close() error { e.closes++; return e.err }
func TestTrainingFlowBackendOfflineOwnership(t *testing.T) {
	sentinel := errors.New("retained resource")
	if _, err := (&FlowTapeTrainingBackend{Executor: hostFlowTapeExecutor{t}, MaxBytes: 1}).EvaluateFlow(tinyTrainableFlowHead(), []float32{.2, -.3, .5}, []float32{.25, .8}, []float32{.3, -.2}, 1, []float32{.1, .2}, []float32{.3, -.4}); err == nil {
		t.Fatal("over-budget tape executed")
	}
	owner := &closableHostFlowExecutor{hostFlowTapeExecutor: hostFlowTapeExecutor{t}, err: sentinel}
	backend := &FlowTapeTrainingBackend{Executor: owner, MaxBytes: 64 << 20}
	if err := backend.Close(); !errors.Is(err, sentinel) || owner.closes != 1 {
		t.Fatal("close lost owner error", err)
	}
	for _, b := range []*FlowTapeTrainingBackend{nil, {}, {Executor: hostFlowTapeExecutor{t}}} {
		if err := b.Close(); err != nil {
			t.Fatal(err)
		}
	}
	for _, b := range []*FlowTapeTrainingBackend{nil, {}} {
		if _, err := b.EvaluateFlow(nil, nil, nil, nil, -1, nil, nil); err == nil {
			t.Fatal("nil executor admitted")
		}
		if b.TrainingFlowBackendName() != "unknown" {
			t.Fatal("unidentified backend named")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	ptx := &PTXFlowTapeExecutor{Context: ctx, Operator: &nvidia.TrainingF32{}, MaxBytes: 1 << 20}
	vk := &VulkanFlowTapeExecutor{Context: ctx, Operator: &vulkan.VkTrainingF32{}, MaxBytes: 1 << 20}
	if _, err := ptx.ExecuteFlowTape(FlowDualTrainingTape{}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := vk.ExecuteFlowTape(FlowDualTrainingTape{}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if ptx.terminal != nil || vk.terminal != nil {
		t.Fatal("preflight cancellation poisoned executor")
	}
	// Empty host operators cannot load a driver; nil and terminal guards must
	// reject before tape creation. A terminal close must retain its operator.
	for _, e := range []*PTXFlowTapeExecutor{nil, {}, {Context: context.Background()}, {Context: context.Background(), Operator: &nvidia.TrainingF32{}, terminal: sentinel}} {
		if _, err := e.ExecuteFlowTape(FlowDualTrainingTape{}); err == nil {
			t.Fatal("bad PTX owner admitted")
		}
	}
	for _, e := range []*VulkanFlowTapeExecutor{nil, {}, {Context: context.Background()}, {Context: context.Background(), Operator: &vulkan.VkTrainingF32{}, terminal: sentinel}} {
		if _, err := e.ExecuteFlowTape(FlowDualTrainingTape{}); err == nil {
			t.Fatal("bad Vulkan owner admitted")
		}
	}
	ptx.terminal = sentinel
	vk.terminal = sentinel
	if err := ptx.Close(); !errors.Is(err, sentinel) || ptx.Operator == nil {
		t.Fatal("terminal PTX owner dropped", err)
	}
	if err := vk.Close(); !errors.Is(err, sentinel) || vk.Operator == nil {
		t.Fatal("terminal Vulkan owner dropped", err)
	}
	var nilPTX *PTXFlowTapeExecutor
	var nilVK *VulkanFlowTapeExecutor
	if nilPTX.Close() != nil || nilVK.Close() != nil || nilPTX.TrainingFlowBackendName() != "unknown" || nilVK.TrainingFlowBackendName() != "unknown" {
		t.Fatal("typed nil owner")
	}
	if (&PTXFlowTapeExecutor{}).Close() != nil || (&VulkanFlowTapeExecutor{}).Close() != nil {
		t.Fatal("empty host owner")
	}
	if (&FlowTapeTrainingBackend{Executor: ptx}).TrainingFlowBackendName() != "ptx-flow-hybrid-v1" || (&FlowTapeTrainingBackend{Executor: vk}).TrainingFlowBackendName() != "vulkan-flow-hybrid-v1" {
		t.Fatal("scope propagation")
	}
}
