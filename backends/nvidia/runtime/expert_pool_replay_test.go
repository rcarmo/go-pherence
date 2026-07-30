package nvidia

import (
	"strings"
	"testing"
)

func TestNewExpertPoolReplayFixtureDerivesLayers(t *testing.T) {
	fixture, err := NewExpertPoolReplayFixture("two-layers", []int{0, 1, 4, 5}, 4)
	if err != nil {
		t.Fatal(err)
	}
	wantLayers := []int{0, 0, 1, 1}
	for i, want := range wantLayers {
		if got := fixture.Layers[i]; got != want {
			t.Fatalf("layer[%d]=%d want %d", i, got, want)
		}
	}
}

func TestExpertPoolReplayFixtureRejectsMalformedInputs(t *testing.T) {
	if _, err := NewExpertPoolReplayFixture("bad", []int{0}, 0); err == nil {
		t.Fatal("expected expertsPerLayer validation error")
	}
	fixture := ExpertPoolReplayFixture{GlobalKeys: []int{0}, Layers: nil}
	if err := fixture.Validate(); err == nil {
		t.Fatal("expected length mismatch error")
	}
	fixture = ExpertPoolReplayFixture{GlobalKeys: []int{-1}, Layers: []int{0}}
	if err := fixture.Validate(); err == nil {
		t.Fatal("expected negative key error")
	}
	fixture = ExpertPoolReplayFixture{GlobalKeys: []int{0}, Layers: []int{-1}}
	if err := fixture.Validate(); err == nil {
		t.Fatal("expected negative layer error")
	}
	fixture = ExpertPoolReplayFixture{GlobalKeys: []int{0}, Layers: []int{0}}
	if _, err := fixture.Replay(-1); err == nil {
		t.Fatal("expected negative slot error")
	}
	if _, err := fixture.ReplayWithPolicy(1, ExpertCachePolicy("broken")); err != nil {
		t.Fatalf("ReplayWithPolicy should fall back to LRU for unsupported constructor policy values: %v", err)
	}
}

func TestExpertPoolReplayDeterministicAcrossSlotSizes(t *testing.T) {
	fixture, err := NewExpertPoolReplayFixture("mixed-locality", []int{0, 1, 0, 4, 0, 1, 4, 4}, 4)
	if err != nil {
		t.Fatal(err)
	}
	stats, err := fixture.ReplaySlots(1, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(stats) != 3 {
		t.Fatalf("len(stats)=%d want 3", len(stats))
	}

	cases := []struct {
		got       ExpertPoolReplayStats
		policy    ExpertCachePolicy
		slots     int
		hits      int
		misses    int
		evictions int
		layers    []ExpertPoolReplayLayerStats
	}{
		{
			got:       stats[0],
			policy:    ExpertCachePolicyLRU,
			slots:     1,
			hits:      1,
			misses:    7,
			evictions: 6,
			layers: []ExpertPoolReplayLayerStats{
				{Layer: 0, Accesses: 5, Hits: 0, Misses: 5},
				{Layer: 1, Accesses: 3, Hits: 1, Misses: 2},
			},
		},
		{
			got:       stats[1],
			policy:    ExpertCachePolicyLRU,
			slots:     2,
			hits:      3,
			misses:    5,
			evictions: 3,
			layers: []ExpertPoolReplayLayerStats{
				{Layer: 0, Accesses: 5, Hits: 2, Misses: 3},
				{Layer: 1, Accesses: 3, Hits: 1, Misses: 2},
			},
		},
		{
			got:       stats[2],
			policy:    ExpertCachePolicyLRU,
			slots:     3,
			hits:      5,
			misses:    3,
			evictions: 0,
			layers: []ExpertPoolReplayLayerStats{
				{Layer: 0, Accesses: 5, Hits: 3, Misses: 2},
				{Layer: 1, Accesses: 3, Hits: 2, Misses: 1},
			},
		},
	}

	for _, tc := range cases {
		if tc.got.Policy != tc.policy {
			t.Fatalf("policy=%q want %q", tc.got.Policy, tc.policy)
		}
		if tc.got.Slots != tc.slots {
			t.Fatalf("slots=%d want %d", tc.got.Slots, tc.slots)
		}
		if tc.got.Hits != tc.hits || tc.got.Misses != tc.misses || tc.got.Evictions != tc.evictions {
			t.Fatalf("slots=%d got hits=%d misses=%d evictions=%d", tc.slots, tc.got.Hits, tc.got.Misses, tc.got.Evictions)
		}
		if tc.got.UniqueExperts != 3 {
			t.Fatalf("slots=%d unique=%d want 3", tc.slots, tc.got.UniqueExperts)
		}
		if len(tc.got.Layers) != len(tc.layers) {
			t.Fatalf("slots=%d layer stats=%d want %d", tc.slots, len(tc.got.Layers), len(tc.layers))
		}
		for i, want := range tc.layers {
			got := tc.got.Layers[i]
			if got != want {
				t.Fatalf("slots=%d layer[%d]=%+v want %+v", tc.slots, i, got, want)
			}
		}
	}
}

func TestExpertPoolReplayDefaultMatchesExplicitLRU(t *testing.T) {
	fixture, err := NewExpertPoolReplayFixture("default-lru", []int{0, 1, 0, 4, 0, 1, 4, 4}, 4)
	if err != nil {
		t.Fatal(err)
	}
	got, err := fixture.Replay(2)
	if err != nil {
		t.Fatal(err)
	}
	want, err := fixture.ReplayWithPolicy(2, ExpertCachePolicyLRU)
	if err != nil {
		t.Fatal(err)
	}
	if got.Fixture != want.Fixture || got.Policy != want.Policy || got.Slots != want.Slots || got.Accesses != want.Accesses || got.UniqueExperts != want.UniqueExperts || got.Hits != want.Hits || got.Misses != want.Misses || got.Evictions != want.Evictions {
		t.Fatalf("default Replay=%+v want explicit LRU %+v", got, want)
	}
	if len(got.Layers) != len(want.Layers) {
		t.Fatalf("default Replay layers=%d want %d", len(got.Layers), len(want.Layers))
	}
	for i := range got.Layers {
		if got.Layers[i] != want.Layers[i] {
			t.Fatalf("default Replay layer[%d]=%+v want %+v", i, got.Layers[i], want.Layers[i])
		}
	}
}

func TestExpertPoolReplayLFUDeterministic(t *testing.T) {
	fixture, err := NewExpertPoolReplayFixture("lfu", []int{0, 1, 0, 2, 1, 2}, 2)
	if err != nil {
		t.Fatal(err)
	}
	stats, err := fixture.ReplayWithPolicy(2, ExpertCachePolicyLFU)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Policy != ExpertCachePolicyLFU {
		t.Fatalf("policy=%q want %q", stats.Policy, ExpertCachePolicyLFU)
	}
	if stats.Hits != 1 || stats.Misses != 5 || stats.Evictions != 3 {
		t.Fatalf("LFU stats=%+v", stats)
	}
	wantLayers := []ExpertPoolReplayLayerStats{
		{Layer: 0, Accesses: 4, Hits: 1, Misses: 3},
		{Layer: 1, Accesses: 2, Hits: 0, Misses: 2},
	}
	if len(stats.Layers) != len(wantLayers) {
		t.Fatalf("len(layers)=%d want %d", len(stats.Layers), len(wantLayers))
	}
	for i, want := range wantLayers {
		if stats.Layers[i] != want {
			t.Fatalf("layer[%d]=%+v want %+v", i, stats.Layers[i], want)
		}
	}
}

func TestExpertPoolReplayReportIncludesPerLayerSummary(t *testing.T) {
	fixture, err := NewExpertPoolReplayFixture("report", []int{0, 1, 0, 4}, 4)
	if err != nil {
		t.Fatal(err)
	}
	stats, err := fixture.ReplayWithPolicy(2, ExpertCachePolicyLFU)
	if err != nil {
		t.Fatal(err)
	}
	report := stats.Report()
	for _, needle := range []string{"fixture=report", "policy=lfu", "slots=2", "L0", "L1", "accesses=", "hit_rate="} {
		if !strings.Contains(report, needle) {
			t.Fatalf("report %q missing %q", report, needle)
		}
	}
}
