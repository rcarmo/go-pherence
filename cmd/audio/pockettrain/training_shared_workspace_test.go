//go:build linux

package main

import (
	"context"
	"github.com/rcarmo/go-pherence/model/pockettts"
	"math"
	"strings"
	"testing"
)

func TestPocketTrainSharedHostAdmission(t *testing.T) {
	p := pockettts.TrainingShapePlan{ParameterBytes: 100, ResidentUpperBytes: 1000}
	if _, e := sharedWorkspaceHostMappingBudget(p, 1900+2048, 1024); e == nil {
		t.Fatal("onebyte mapping headroom not reserved")
	}
	if n, e := sharedWorkspaceHostMappingBudget(p, 1900+2049, 1024); e != nil || n != 1 {
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
