//go:build linux

package main

import (
	"context"
	"github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/model/pockettts"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

func TestPocketTrainSharedNativeBudgetInstalled(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_CLI_BUDGET_NATIVE") != "1" {
		t.Skip("explicit native CLI budget setup")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	owners, cleanup, err := openTrainingBackend(ctx, "vulkan-resident-sharedworkspace-hybrid-v1", 8<<20)
	if err != nil {
		t.Fatal("native constructor no retry", err)
	}
	if owners.optimizer == nil || cleanup == nil {
		t.Fatal("missing shared owner")
	}
	stats := vulkan.VulkanMemoryStats()
	if stats.Budget.MaxBytes != 8<<20 || stats.Budget.MaxAllocations != 4096 || stats.Bytes != 0 || stats.Allocations != 0 {
		t.Fatal("CLI budget not installed before state", stats)
	}
	if err = cleanup(); err != nil {
		t.Fatal(err)
	}
	stats = vulkan.VulkanMemoryStats()
	if stats.Bytes != 0 || stats.Allocations != 0 || stats.InFlight || stats.Uncertain || stats.DeviceLost {
		t.Fatal("native constructor closure", stats)
	}
}

func TestPocketTrainSharedNativeBudget(t *testing.T) {
	budget, err := sharedWorkspaceNativeBudget(2<<30, vulkan.VulkanMemoryUsage{})
	if err != nil || budget.MaxBytes != 2<<30 || budget.MaxAllocations != 4096 {
		t.Fatal(budget, err)
	}
	usage := vulkan.VulkanMemoryUsage{Budget: vulkan.VulkanMemoryBudget{MaxBytes: 1 << 30, MaxAllocations: 32}}
	budget, err = sharedWorkspaceNativeBudget(2<<30, usage)
	if err != nil || budget != usage.Budget {
		t.Fatal("stricter budget loosened", budget, err)
	}
	for _, limit := range []int64{0, -1, 2<<30 + 1} {
		if _, err = sharedWorkspaceNativeBudget(limit, vulkan.VulkanMemoryUsage{}); err == nil {
			t.Fatal("invalid native cap", limit)
		}
	}
	for _, usage := range []vulkan.VulkanMemoryUsage{{Bytes: 4}, {Allocations: 1}, {InFlight: true}, {Uncertain: true}, {DeviceLost: true}} {
		if _, err = sharedWorkspaceNativeBudget(2<<30, usage); err == nil {
			t.Fatal("foreign live native owner accepted", usage)
		}
	}
}

func TestPocketTrainSharedReleasedShapeAdmission(t *testing.T) {
	raw, err := os.ReadFile("../../../model/pockettts/testdata/english-upstream.yaml")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := pockettts.ParseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	for _, shape := range []pockettts.TrainingShape{{MicroBatchRows: 1, TargetFrames: 14, VoiceFrames: 13, TextTokens: 1, GradientAccumulation: 1, FlowBatchMultiplier: 1}, {MicroBatchRows: 1, TargetFrames: 375, VoiceFrames: 62, TextTokens: 512, GradientAccumulation: 1, FlowBatchMultiplier: 1}} {
		plan, err := pockettts.PlanTrainingShape(cfg, shape, pockettts.ProductionTrainingShapeLimits(6<<30))
		if err != nil {
			if shape.TargetFrames == 14 {
				t.Fatal(err)
			}
			t.Log("joint CPU planner rejection", err)
			continue
		}
		mapping, err := sharedWorkspaceHostMappingBudget(plan, 6<<30, 2<<30)
		t.Logf("shape=%+v CPUupper=%d P=%d mapping=%d err=%v", shape, plan.ResidentUpperBytes, plan.ParameterBytes, mapping, err)
		if shape.TargetFrames == 14 && (err != nil || mapping < 219029196) {
			t.Fatal("released short row not admitted", mapping, err)
		}
		if shape.TargetFrames == 375 && err == nil {
			t.Fatal("unqualified joint host estimate unexpectedly admitted")
		}
	}
}

func TestPocketTrainSharedHostAdmission(t *testing.T) {
	p := pockettts.TrainingShapePlan{ParameterBytes: 100, ResidentUpperBytes: 1000}
	if _, e := sharedWorkspaceHostMappingBudget(p, 1400+1024, 1024); e == nil {
		t.Fatal("onebyte mapping headroom not reserved")
	}
	if n, e := sharedWorkspaceHostMappingBudget(p, 1400+1025, 1024); e != nil || n != 1 {
		t.Fatal(n, e)
	}
	for _, b := range []int64{0, -1, math.MaxInt64} {
		if _, e := sharedWorkspaceHostMappingBudget(p, 10000, b); e == nil {
			t.Fatal("invalid device cap")
		}
	}
	p.ParameterBytes = math.MaxInt64
	if _, e := sharedWorkspaceHostMappingBudget(p, math.MaxInt64, 1024); e == nil {
		t.Fatal("overflow")
	}
}

func TestPocketTrainSharedWorkspaceScope(t *testing.T) {
	pin := strings.Repeat("a", 64)
	name := "vulkan-resident-sharedworkspace-hybrid"
	scope := name + "-v1"
	if got, e := trainingBackendScope(name, true, pin, 2<<30); e != nil || got != scope {
		t.Fatal(got, e)
	}
	for _, c := range []struct {
		opt   bool
		pin   string
		bytes int64
	}{{false, pin, 1024}, {true, "", 1024}, {true, pin, 0}, {true, pin, (2 << 30) + 1}} {
		if _, e := trainingBackendScope(name, c.opt, c.pin, c.bytes); e == nil {
			t.Fatal("shared scope admission")
		}
	}
	if _, e := trainingBackendScope("ptx-resident-sharedworkspace-hybrid", true, pin, 1024); e == nil {
		t.Fatal("unimplemented PTX route")
	}
	want := hashText("base;resident-sharedworkspace-backend=" + scope + ";device-runtime=" + pin)
	if got := trainingBackendOptionsHash("base", scope, pin); got != want || got == trainingBackendOptionsHash("base", "vulkan-resident-devicehandoff-hybrid-v1", pin) {
		t.Fatal("replay identity", got)
	}
	if !strings.Contains(trainingBackendNotice(scope), "CPU copy-back") {
		t.Fatal("placement notice")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, e := openTrainingBackend(ctx, scope, 1024); e == nil {
		t.Fatal("cancel factory")
	}
}
