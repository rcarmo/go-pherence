package nvidia

import (
	"fmt"
	"testing"
)

func benchmarkExpertPoolReplayFixture(b *testing.B) ExpertPoolReplayFixture {
	b.Helper()
	const (
		layers          = 48
		expertsPerLayer = 128
		steps           = 32
		topK            = 4
	)
	keys := make([]int, 0, layers*steps*topK)
	for step := 0; step < steps; step++ {
		for layer := 0; layer < layers; layer++ {
			base := (layer*17 + step*3) % expertsPerLayer
			keys = append(keys,
				layer*expertsPerLayer+base,
				layer*expertsPerLayer+((base+1)%expertsPerLayer),
				layer*expertsPerLayer+((base+step/4+7)%expertsPerLayer),
				layer*expertsPerLayer+((layer+step*11)%expertsPerLayer),
			)
		}
	}
	fixture, err := NewExpertPoolReplayFixture("synthetic_qwen_moe_decode", keys, expertsPerLayer)
	if err != nil {
		b.Fatal(err)
	}
	return fixture
}

func benchmarkExpertPoolReplayLayerQuotaFixture(b *testing.B) ExpertPoolReplayFixture {
	b.Helper()
	const (
		layers          = 24
		expertsPerLayer = 64
		steps           = 48
		topK            = 4
	)
	keys := make([]int, 0, layers*steps*topK)
	for step := 0; step < steps; step++ {
		for layer := 0; layer < layers; layer++ {
			quota := 4
			switch {
			case layer < 8:
				quota = 8
			case layer < 16:
				quota = 16
			default:
				quota = 32
			}
			base := (layer*5 + step*7) % quota
			keys = append(keys,
				layer*expertsPerLayer+base,
				layer*expertsPerLayer+((base+1)%quota),
				layer*expertsPerLayer+((base+step/6+3)%quota),
				layer*expertsPerLayer+((base+layer+step)%quota),
			)
		}
	}
	fixture, err := NewExpertPoolReplayFixture("synthetic_layer_quota", keys, expertsPerLayer)
	if err != nil {
		b.Fatal(err)
	}
	return fixture
}

func benchmarkExpertPoolReplayPolicies(b *testing.B, fixture ExpertPoolReplayFixture, slotSizes ...int) {
	b.Helper()
	for _, policy := range []ExpertCachePolicy{ExpertCachePolicyLRU, ExpertCachePolicyLFU} {
		for _, slots := range slotSizes {
			baseline, err := fixture.ReplayWithPolicy(slots, policy)
			if err != nil {
				b.Fatal(err)
			}
			b.Run(fmt.Sprintf("policy_%s/slots_%d", policy, slots), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					stats, err := fixture.ReplayWithPolicy(slots, policy)
					if err != nil {
						b.Fatal(err)
					}
					if stats.Hits != baseline.Hits || stats.Misses != baseline.Misses || stats.Evictions != baseline.Evictions {
						b.Fatalf("non-deterministic replay: got %+v want %+v", stats, baseline)
					}
				}
				b.StopTimer()
				b.ReportMetric(100*baseline.HitRate(), "hit_pct")
				b.ReportMetric(float64(baseline.Evictions), "evictions")
			})
		}
	}
}

// Synthetic only: these route streams are cache-policy microbenchmarks, not
// real-model performance claims.
func BenchmarkExpertPoolReplaySyntheticGlobalKeys(b *testing.B) {
	benchmarkExpertPoolReplayPolicies(b, benchmarkExpertPoolReplayFixture(b), 16, 32, 64)
}

// Synthetic only: this simple layered-locality fixture approximates per-layer
// quota pressure without making claims about any production model.
func BenchmarkExpertPoolReplaySyntheticLayerQuota(b *testing.B) {
	benchmarkExpertPoolReplayPolicies(b, benchmarkExpertPoolReplayLayerQuotaFixture(b), 8, 16, 32)
}
