package vulkan

import (
	"context"
	"math"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

func sharedVulkanMock(t *testing.T) (*VulkanSharedWorkspaceDriver, *planMock, *VkComputeKernel) {
	t.Helper()
	mock, k, _, _ := newPlanMock(t)
	k.numBuffers = 3
	k.pushSize = 20
	return NewVulkanSharedWorkspaceDriver(&VkTrainingF32{adam: k, vector: k, normStats: k, globalNorm: k, adamScaled: k}), mock, k
}
func TestVulkanSharedWorkspaceCopyAdmission(t *testing.T) {
	d, _, _ := sharedVulkanMock(t)
	ctx := context.Background()
	a, e := d.NewBuffer(ctx, 4)
	if e != nil {
		t.Fatal(e)
	}
	b, e := d.NewBuffer(ctx, 4)
	if e != nil {
		t.Fatal(e)
	}
	// Raw transport contract: preserve signed zero and subnormal representation.
	values := []float32{math.Float32frombits(0x80000000), math.Float32frombits(1), 1, -2}
	if e = a.Upload(ctx, values); e != nil {
		t.Fatal(e)
	}
	if e = d.ValidateSharedInventory(ctx, []train.ResidentBuffer{a, b}, []int{4, 4}); e != nil {
		t.Fatal(e)
	}
	if e = d.Copy(ctx, b, a, 4); e != nil {
		t.Fatal(e)
	}
	got := make([]float32, 4)
	if e = b.Download(ctx, got); e != nil {
		t.Fatal(e)
	}
	for i, z := range got {
		if math.Float32bits(z) != math.Float32bits(values[i]) {
			t.Fatal("exact copy transport", i, z)
		}
	}
	for _, test := range []struct {
		name        string
		out, source train.ResidentBuffer
		n           int
	}{{"alias", a, a, 4}, {"extent", a, b, 3}, {"nil", nil, b, 4}, {"negative", a, b, -1}} {
		t.Run(test.name, func(t *testing.T) {
			if d.ValidateCopy(ctx, test.out, test.source, test.n) == nil {
				t.Fatal("invalid copy")
			}
		})
	}
	alias := *a.(*vkSharedOwner)
	if d.ValidateSharedInventory(ctx, []train.ResidentBuffer{a, &alias}, []int{4, 4}) == nil {
		t.Fatal("wrapped physical alias")
	}
	if d.ValidateSharedInventory(ctx, []train.ResidentBuffer{a}, []int{3}) == nil || d.ValidateSharedInventory(ctx, nil, nil) == nil {
		t.Fatal("inventory")
	}
	foreign := NewVulkanSharedWorkspaceDriver(d.base.Operator)
	if foreign.ValidateCopy(ctx, a, b, 4) == nil {
		t.Fatal("foreign driver")
	}
	if _, e = d.Prefix(ctx, a, 2, nil); e == nil {
		t.Fatal("missing token")
	}
	prefix := &vkSharedPrefix{driver: d, parent: a.(*vkSharedOwner), tensor: a.(*vkSharedOwner).raw.tensor, access: &train.SharedResidentAccess{}}
	if prefix.Upload(ctx, values) == nil || prefix.Download(ctx, got) == nil || prefix.Close() == nil {
		t.Fatal("invalid/expired prefix")
	}
	if _, e = d.NewBoundTape(ctx, train.Tape{}, map[int]train.ResidentBuffer{0: prefix}, 1024); e == nil {
		t.Fatal("prefix retained")
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if d.Copy(canceled, b, a, 4) == nil {
		t.Fatal("cancel copy")
	}
	if e = a.Close(); e != nil {
		t.Fatal(e)
	}
	if d.Copy(ctx, b, a, 4) == nil {
		t.Fatal("closed source")
	}
	if e = b.Close(); e != nil {
		t.Fatal(e)
	}
}
func TestVulkanSharedWorkspaceGenericAccumulate(t *testing.T) {
	d, mock, k := sharedVulkanMock(t)
	ctx := context.Background()
	snap := train.ResidentSnapshot{Config: train.ResidentConfig{LearningRate: .001, Beta1: .9, Beta2: .99, Epsilon: 1e-8, EMADecay: .9}, Parameters: map[string]train.ResidentParameter{"a": {Parameter: []float32{1, 2, 3}, M: make([]float32, 3), V: make([]float32, 3), EMA: []float32{1, 2, 3}}, "b": {Parameter: []float32{4}, M: make([]float32, 1), V: make([]float32, 1), EMA: []float32{4}}}}
	plan, e := train.SharedResidentBytes(snap)
	if e != nil {
		t.Fatal(e)
	}
	if s, e := train.NewSharedWorkspaceResidentState(ctx, d, snap, plan.Bytes-1); e == nil || s != nil {
		t.Fatal("byte admission")
	}
	s, e := train.NewSharedWorkspaceResidentState(ctx, d, snap, plan.Bytes)
	if e != nil {
		t.Fatal(e)
	}
	if e = s.Accumulate(ctx, map[string][]float32{"a": {.1, .2, .3}, "b": {.4}}, .25); e != nil {
		t.Fatal("guarded valid prefix path", e)
	}
	if e = s.Accumulate(ctx, map[string][]float32{"a": {.2, .3, .4}}, 1); e != nil {
		t.Fatal("stable G repeated", e)
	}
	if len(mock.dispatchSets) != 3 {
		t.Fatal("unexpected accumulation dispatch", len(mock.dispatchSets))
	}
	// Norm/update preflight still honours protected wrappers; configure mock ABI.
	k.numBuffers = 2
	k.pushSize = 8
	a, e := d.NewBuffer(ctx, 3)
	if e != nil {
		t.Fatal(e)
	}
	stats, e := d.NewBuffer(ctx, 3)
	if e != nil {
		t.Fatal(e)
	}
	if e = d.ValidateNormStats(ctx, stats, a, 3, false); e != nil {
		t.Fatal(e)
	}
	if d.ValidateNormStats(ctx, a, a, 3, false) == nil {
		t.Fatal("norm alias")
	}
	if e = a.Close(); e != nil {
		t.Fatal(e)
	}
	if e = stats.Close(); e != nil {
		t.Fatal(e)
	}
	if e = s.Close(); e != nil {
		t.Fatal(e)
	}
}
func TestVulkanSharedWorkspacePartialFailureRetention(t *testing.T) {
	d, mock, _ := sharedVulkanMock(t)
	ctx := context.Background()
	snap := train.ResidentSnapshot{Config: train.ResidentConfig{LearningRate: .001, Beta1: .9, Beta2: .99, Epsilon: 1e-8, EMADecay: .9}, Parameters: map[string]train.ResidentParameter{"a": {Parameter: []float32{1}, M: []float32{0}, V: []float32{0}, EMA: []float32{1}}}}
	plan, _ := train.SharedResidentBytes(snap)
	s, e := train.NewSharedWorkspaceResidentState(ctx, d, snap, plan.Bytes)
	if e != nil {
		t.Fatal(e)
	}
	mock.failure = "submit"
	if e = s.Accumulate(ctx, map[string][]float32{"a": {.1}}, 1); e == nil {
		t.Fatal("uncertain accumulation accepted")
	}
	if s.Close() == nil || d.Close() == nil {
		t.Fatal("uncertain storage release")
	}
	count := len(mock.dispatchSets)
	if s.Accumulate(ctx, map[string][]float32{"a": {.1}}, 1) == nil || len(mock.dispatchSets) != count {
		t.Fatal("uncertain retry")
	}
}
