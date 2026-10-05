//go:build linux

package vulkan

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"
	"unsafe"
)

// Submit real native work exactly once, cancel at its return, and retain all
// owners. No drain, resubmission, reset or recovery is performed by this test.
// Run in a dedicated process: process teardown owns the terminal native state.
func TestVulkanNativeTrainingCancelRetainsOwners(t *testing.T) {
	if os.Getenv("GO_PHERENCE_VULKAN_CANCEL_NATIVE") != "1" {
		t.Skip("explicit isolated native cancellation qualification")
	}
	captureInit(t)
	if err := VulkanSetMemoryBudget(VulkanMemoryBudget{MaxBytes: 8 << 20, MaxAllocations: 64}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	op, err := NewVkTrainingF32(ctx)
	if err != nil {
		t.Fatal(err)
	}
	arena, err := NewVkTensorArena(ctx, 4096)
	if err != nil {
		t.Fatal(err)
	}
	tensors := make([]*VkTensorF32, 3)
	for i := range tensors {
		tensors[i], err = arena.AllocF32(ctx, 256)
		if err != nil {
			t.Fatal(err)
		}
		if err = tensors[i].Upload(ctx, make([]float32, 256)); err != nil {
			t.Fatal(err)
		}
	}
	stage, err := op.SiLUBackwardStage(ctx, tensors[0], tensors[1], tensors[2])
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewVkF32Plan(ctx, []VkF32Stage{stage})
	if err != nil {
		t.Fatal(err)
	}
	submit := vkQueueSubmit
	submissions := 0
	mockVK(t, &vkQueueSubmit, func(q VkQueue, n uint32, info unsafe.Pointer, f VkFence) VkResult {
		result := submit(q, n, info, f)
		if result == VK_SUCCESS {
			submissions++
			cancel()
		}
		return result
	})
	err = plan.Run(ctx)
	if !errors.Is(err, context.Canceled) || !errors.Is(err, ErrVulkanInFlight) || submissions != 1 {
		t.Fatal("native cancellation boundary", err, submissions)
	}
	before := VulkanMemoryStats()
	if before.Bytes == 0 || before.Allocations == 0 || !before.InFlight || before.Uncertain || before.DeviceLost {
		t.Fatal("cancel retained state", before)
	}
	if err = plan.Close(); !errors.Is(err, ErrVulkanInFlight) {
		t.Fatal("pending plan destruction", err)
	}
	if err = arena.Close(); !errors.Is(err, ErrVulkanInFlight) {
		t.Fatal("pending arena destruction", err)
	}
	after := VulkanMemoryStats()
	if after.Bytes != before.Bytes || after.Allocations != before.Allocations || !after.InFlight {
		t.Fatal("cancel owner accounting changed", before, after)
	}
	t.Logf("native cancellation single-submit retained=%+v; no retry/drain/recovery; process teardown only", after)
}
