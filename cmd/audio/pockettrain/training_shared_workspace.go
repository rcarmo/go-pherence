//go:build linux

package main

import (
	"context"
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/model/pockettts"
)

// CPU planner already covers P/G/M/V/EMA, activation clones and one GC cycle.
// Shared completion retains a separate P/M/V/EMA guard (4P); checkpoints borrow
// that validated guard synchronously instead of adding a copied 5P snapshot.
// One combined allocator cap covers all coherent resident+tape memory. These
// ownership facts remove duplicate charges, not the process/runtime RSS guard.
func sharedWorkspaceHostMappingBudget(plan pockettts.TrainingShapePlan, available, deviceCap int64) (int64, error) {
	if plan.ParameterBytes <= 0 || plan.ResidentUpperBytes < 0 || deviceCap <= 0 || deviceCap > 2<<30 || available <= 0 {
		return 0, fmt.Errorf("sharedworkspace host admission inputs")
	}
	if plan.ParameterBytes > math.MaxInt64/4 {
		return 0, fmt.Errorf("sharedworkspace host state overflow")
	}
	state := 4 * plan.ParameterBytes // independent completed P/M/V/EMA boundary
	if plan.ResidentUpperBytes > math.MaxInt64-state {
		return 0, fmt.Errorf("sharedworkspace host sum overflow")
	}
	need := plan.ResidentUpperBytes + state
	if need > math.MaxInt64-deviceCap {
		return 0, fmt.Errorf("sharedworkspace mapped sum overflow")
	}
	need += deviceCap
	if need >= available {
		return 0, fmt.Errorf("sharedworkspace host/native mapped admission requires %d bytes before model mapping, ceiling=%d", need, available)
	}
	return available - need, nil
}

// The shared route has a combined package-owned native ceiling in addition to
// complete-source and resident admission. Never loosen a stricter caller cap
// or borrow an already-live allocation inventory from another owner.
func sharedWorkspaceNativeBudget(maxBytes int64, usage vulkan.VulkanMemoryUsage) (vulkan.VulkanMemoryBudget, error) {
	var budget vulkan.VulkanMemoryBudget
	if maxBytes <= 0 || maxBytes > 2<<30 {
		return budget, fmt.Errorf("sharedworkspace native ceiling")
	}
	if usage.Bytes != 0 || usage.Allocations != 0 || usage.InFlight || usage.Uncertain || usage.DeviceLost {
		return budget, fmt.Errorf("sharedworkspace requires a fresh native allocation owner")
	}
	budget = vulkan.VulkanMemoryBudget{MaxBytes: uint64(maxBytes), MaxAllocations: 4096}
	if usage.Budget.MaxBytes != 0 {
		budget.MaxBytes = min(budget.MaxBytes, usage.Budget.MaxBytes)
	}
	if usage.Budget.MaxAllocations != 0 {
		budget.MaxAllocations = min(budget.MaxAllocations, usage.Budget.MaxAllocations)
	}
	return budget, nil
}

func openSharedWorkspaceBackend(ctx context.Context, maxBytes int64) (trainingBackendOwners, func() error, error) {
	var owners trainingBackendOwners
	if !vulkan.VulkanInit() {
		return owners, nil, fmt.Errorf("Vulkan unavailable; sharedworkspace has no fallback")
	}
	budget, e := sharedWorkspaceNativeBudget(maxBytes, vulkan.VulkanMemoryStats())
	if e != nil {
		return owners, nil, e
	}
	if e = vulkan.VulkanSetMemoryBudget(budget); e != nil {
		return owners, nil, e
	}
	op, e := vulkan.NewVkTrainingF32(ctx)
	if e != nil {
		if op != nil {
			return owners, op.Close, e
		}
		return owners, nil, e
	}
	b := &pockettts.ResidentTrainingBackend{Driver: vulkan.NewVulkanSharedWorkspaceDriver(op), Scope: "vulkan-resident-sharedworkspace-hybrid-v1", MaxResidentBytes: maxBytes, MaxTapeBytes: maxBytes}
	owners.flow, owners.transformer, owners.optimizer = b, b, b
	owners.weight, owners.projection, owners.sequence, owners.eos = b, b, b, b
	owners.loss, owners.branch, owners.step = b, b, b
	return owners, func() error {
		if e := b.Close(); e != nil {
			return e
		}
		return op.Close()
	}, nil
}
