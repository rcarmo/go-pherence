package nvidia

import (
	"fmt"
	"sort"
	"strings"
)

// ExpertPoolReplayFixture records a deterministic sequence of global expert-pool
// key lookups together with their source layer, so tests and benchmarks can
// replay the exact same route stream against different slot counts and cache
// policies.
type ExpertPoolReplayFixture struct {
	Name       string
	GlobalKeys []int
	Layers     []int
}

// NewExpertPoolReplayFixture derives layer IDs from global pool keys using the
// production key layout layer*expertsPerLayer+expert.
func NewExpertPoolReplayFixture(name string, globalKeys []int, expertsPerLayer int) (ExpertPoolReplayFixture, error) {
	if expertsPerLayer <= 0 {
		return ExpertPoolReplayFixture{}, fmt.Errorf("invalid expertsPerLayer %d", expertsPerLayer)
	}
	fixture := ExpertPoolReplayFixture{
		Name:       name,
		GlobalKeys: append([]int(nil), globalKeys...),
		Layers:     make([]int, len(globalKeys)),
	}
	for i, key := range fixture.GlobalKeys {
		if key < 0 {
			return ExpertPoolReplayFixture{}, fmt.Errorf("global key[%d]=%d must be non-negative", i, key)
		}
		fixture.Layers[i] = key / expertsPerLayer
	}
	return fixture, fixture.Validate()
}

// Validate checks that the fixture can be replayed deterministically.
func (f ExpertPoolReplayFixture) Validate() error {
	if len(f.GlobalKeys) != len(f.Layers) {
		return fmt.Errorf("replay fixture key/layer length mismatch keys=%d layers=%d", len(f.GlobalKeys), len(f.Layers))
	}
	for i, key := range f.GlobalKeys {
		if key < 0 {
			return fmt.Errorf("global key[%d]=%d must be non-negative", i, key)
		}
		if f.Layers[i] < 0 {
			return fmt.Errorf("layer[%d]=%d must be non-negative", i, f.Layers[i])
		}
	}
	return nil
}

// ExpertPoolReplayLayerStats holds per-layer replay results for one slot size.
type ExpertPoolReplayLayerStats struct {
	Layer    int
	Accesses int
	Hits     int
	Misses   int
}

func (s ExpertPoolReplayLayerStats) HitRate() float64 {
	if s.Accesses == 0 {
		return 0
	}
	return float64(s.Hits) / float64(s.Accesses)
}

func (s ExpertPoolReplayLayerStats) AccessShare(totalAccesses int) float64 {
	if totalAccesses <= 0 {
		return 0
	}
	return float64(s.Accesses) / float64(totalAccesses)
}

// ExpertPoolReplayStats reports the aggregate and per-layer results from
// replaying one fixture against one slot count.
type ExpertPoolReplayStats struct {
	Fixture       string
	Policy        ExpertCachePolicy
	Slots         int
	Accesses      int
	UniqueExperts int
	Hits          int
	Misses        int
	Evictions     int
	Layers        []ExpertPoolReplayLayerStats
}

func (s ExpertPoolReplayStats) HitRate() float64 {
	if s.Accesses == 0 {
		return 0
	}
	return float64(s.Hits) / float64(s.Accesses)
}

// Report formats a compact human-readable replay summary.
func (s ExpertPoolReplayStats) Report() string {
	layers := make([]string, 0, len(s.Layers))
	for _, layer := range s.Layers {
		layers = append(layers, fmt.Sprintf("L%d accesses=%d (%.2f%%) hits=%d misses=%d hit_rate=%.2f%%",
			layer.Layer,
			layer.Accesses,
			100*layer.AccessShare(s.Accesses),
			layer.Hits,
			layer.Misses,
			100*layer.HitRate(),
		))
	}
	name := s.Fixture
	if name == "" {
		name = "unnamed"
	}
	return fmt.Sprintf("fixture=%s policy=%s slots=%d accesses=%d unique=%d hits=%d misses=%d evictions=%d hit_rate=%.2f%% layers=[%s]",
		name,
		s.Policy.String(),
		s.Slots,
		s.Accesses,
		s.UniqueExperts,
		s.Hits,
		s.Misses,
		s.Evictions,
		100*s.HitRate(),
		strings.Join(layers, ", "),
	)
}

// Replay runs the fixture through the production expert-pool implementation
// using the historical default LRU policy.
func (f ExpertPoolReplayFixture) Replay(slots int) (ExpertPoolReplayStats, error) {
	return f.ReplayWithPolicy(slots, ExpertCachePolicyLRU)
}

// ReplayWithPolicy runs the fixture through the production expert-pool
// implementation using the caller-selected cache policy and dummy expert entries
// so the measured hit/miss/eviction behaviour matches the configured policy.
func (f ExpertPoolReplayFixture) ReplayWithPolicy(slots int, policy ExpertCachePolicy) (ExpertPoolReplayStats, error) {
	if slots < 0 {
		return ExpertPoolReplayStats{}, fmt.Errorf("invalid slot count %d", slots)
	}
	if err := f.Validate(); err != nil {
		return ExpertPoolReplayStats{}, err
	}

	pool := NewExpertPoolWithPolicy(slots, nil, policy)
	perLayer := make(map[int]*ExpertPoolReplayLayerStats)
	unique := make(map[int]struct{}, len(f.GlobalKeys))

	for i, key := range f.GlobalKeys {
		layerID := f.Layers[i]
		layer := perLayer[layerID]
		if layer == nil {
			layer = &ExpertPoolReplayLayerStats{Layer: layerID}
			perLayer[layerID] = layer
		}
		layer.Accesses++
		unique[key] = struct{}{}

		if pool.Get(key) != nil {
			layer.Hits++
			continue
		}
		layer.Misses++
		pool.Put(&ExpertEntry{ExpertID: key})
	}

	layers := make([]ExpertPoolReplayLayerStats, 0, len(perLayer))
	for _, layer := range perLayer {
		layers = append(layers, *layer)
	}
	sort.Slice(layers, func(i, j int) bool { return layers[i].Layer < layers[j].Layer })

	return ExpertPoolReplayStats{
		Fixture:       f.Name,
		Policy:        pool.Policy(),
		Slots:         slots,
		Accesses:      len(f.GlobalKeys),
		UniqueExperts: len(unique),
		Hits:          int(pool.Hits.Load()),
		Misses:        int(pool.Misses.Load()),
		Evictions:     int(pool.Evicts.Load()),
		Layers:        layers,
	}, nil
}

// ReplaySlots replays the same fixture against multiple slot sizes in the
// caller-supplied order using the historical default LRU policy.
func (f ExpertPoolReplayFixture) ReplaySlots(slotSizes ...int) ([]ExpertPoolReplayStats, error) {
	return f.ReplaySlotsWithPolicy(ExpertCachePolicyLRU, slotSizes...)
}

// ReplaySlotsWithPolicy replays the same fixture against multiple slot sizes in
// the caller-supplied order using one cache policy.
func (f ExpertPoolReplayFixture) ReplaySlotsWithPolicy(policy ExpertCachePolicy, slotSizes ...int) ([]ExpertPoolReplayStats, error) {
	out := make([]ExpertPoolReplayStats, 0, len(slotSizes))
	for _, slots := range slotSizes {
		stats, err := f.ReplayWithPolicy(slots, policy)
		if err != nil {
			return nil, err
		}
		out = append(out, stats)
	}
	return out, nil
}
