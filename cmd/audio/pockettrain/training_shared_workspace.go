//go:build linux

package main

import (
	"context"
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/model/pockettts"
)

// Conservative existing-cap accounting. This may reject a row whose exact
// live slot ledger would fit; it never excludes coherent native mapped RAM.
func sharedWorkspaceHostMappingBudget(plan pockettts.TrainingShapePlan, available, deviceCap int64) (int64, error) {
	if plan.ParameterBytes <= 0 || plan.ResidentUpperBytes < 0 || deviceCap <= 0 || deviceCap > 2<<30 || available <= 0 {
		return 0, fmt.Errorf("sharedworkspace host admission inputs")
	}
	if plan.ParameterBytes > math.MaxInt64/9 {
		return 0, fmt.Errorf("sharedworkspace host state overflow")
	}
	state := 9 * plan.ParameterBytes // checkpoint snapshot plus retained completed host boundary
	if plan.ResidentUpperBytes > math.MaxInt64-state {
		return 0, fmt.Errorf("sharedworkspace host sum overflow")
	}
	need := plan.ResidentUpperBytes + state
	if need > math.MaxInt64-2*deviceCap {
		return 0, fmt.Errorf("sharedworkspace mapped sum overflow")
	}
	need += 2 * deviceCap
	if need >= available {
		return 0, fmt.Errorf("sharedworkspace host/native mapped admission requires %d bytes before model mapping, ceiling=%d", need, available)
	}
	return available - need, nil
}

func openSharedWorkspaceBackend(ctx context.Context, maxBytes int64) (trainingBackendOwners, func() error, error) {
	var owners trainingBackendOwners
	if !vulkan.VulkanInit() {
		return owners, nil, fmt.Errorf("Vulkan unavailable; sharedworkspace has no fallback")
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
