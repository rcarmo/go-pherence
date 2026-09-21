package sampling

import (
	"math"
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

// Slow full-sort oracle: deliberately uses separate arrays and the original
// token-order softmax summation so paired in-place sorting cannot change draws.
func referenceSample(t *testing.T, logits []float32, cfg Config, draw float64) Result {
	t.Helper()
	c := make([]candidate, 0, len(logits))
	for i, v := range logits {
		if !isExcluded(v) {
			c = append(c, candidate{int32(i), v})
		}
	}
	if cfg.TopK > 0 {
		sort.Slice(c, func(i, j int) bool { return candidateBetter(c[i], c[j]) })
		c = c[:min(cfg.TopK, len(c))]
	}
	scaled := make([]float64, len(c))
	maxV := math.Inf(-1)
	for i, v := range c {
		scaled[i] = float64(v.logit) / cfg.Temperature
		maxV = math.Max(maxV, scaled[i])
	}
	w := make([]float64, len(c))
	total := 0.0
	for i, v := range scaled {
		d := v - maxV
		if math.IsInf(v, 1) && math.IsInf(maxV, 1) {
			d = 0
		}
		w[i] = math.Exp(d)
		total += w[i]
	}
	if cfg.topPEnabled() {
		order := make([]int, len(c))
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(i, j int) bool { return candidateBetter(c[order[i]], c[order[j]]) })
		sortedC := make([]candidate, len(c))
		sortedW := make([]float64, len(w))
		for i, src := range order {
			sortedC[i] = c[src]
			sortedW[i] = w[src]
		}
		c, w = sortedC, sortedW
		threshold := cfg.TopP * total
		total = 0
		cut := len(c)
		for i, v := range w {
			total += v
			if total >= threshold {
				cut = i + 1
				break
			}
		}
		c, w = c[:cut], w[:cut]
	}
	draw = math.Max(0, math.Min(1, draw))
	sum := 0.0
	chosen := c[len(c)-1]
	for i, v := range w {
		if v <= 0 {
			continue
		}
		sum += v
		if sum >= draw*total {
			chosen = c[i]
			break
		}
	}
	return Result{TokenID: int(chosen.idx), Logit: chosen.logit, Candidates: len(c)}
}

func TestSamplingAllocationOptimisationParity(t *testing.T) {
	r := rand.New(rand.NewSource(19))
	for _, n := range []int{1, 2, 3, 7, 40, 129, 513} {
		logits := make([]float32, n)
		for i := range logits {
			logits[i] = float32(r.Intn(23) - 11)
		}
		for mode := 0; mode < 3; mode++ {
			if mode == 1 && n > 3 {
				logits[0] = float32(math.NaN())
				logits[1] = float32(math.Inf(-1))
			}
			if mode == 2 {
				logits[n-1] = float32(math.Inf(1))
			}
			original := append([]float32(nil), logits...)
			for _, k := range []int{0, 1, 2, 7, 40, n + 1} {
				for _, p := range []float64{0, 0.01, 0.5, 0.9, 1} {
					for _, temp := range []float64{0.01, 0.8, 3} {
						for _, draw := range []float64{-1, 0, math.SmallestNonzeroFloat64, 0.371, 0.5, math.Nextafter(1, 0), 1, 2} {
							cfg := Config{Temperature: temp, TopK: k, TopP: p}
							want := referenceSample(t, logits, cfg, draw)
							got, err := SampleWithDraw(logits, cfg, draw)
							if err != nil || got != want {
								t.Fatalf("n=%d mode=%d cfg=%+v draw=%g got=%+v err=%v want=%+v", n, mode, cfg, draw, got, err, want)
							}
						}
					}
				}
			}
			for i := range logits {
				if math.Float32bits(logits[i]) != math.Float32bits(original[i]) {
					t.Fatal("input mutated")
				}
			}
		}
	}
}

func TestTopKAllocationBoundAndHugeLimit(t *testing.T) {
	logits := []float32{2, 4, 4, -1}
	want := []candidate{{1, 4}, {2, 4}, {0, 2}, {3, -1}}
	got, err := boundedTopK(logits, int(^uint(0)>>1))
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatal(got, err)
	}
	logits = make([]float32, 1024)
	for i := range logits {
		logits[i] = float32(i % 71)
	}
	for _, cfg := range []Config{{Temperature: 0.8, TopK: 40}, {Temperature: 0.8, TopP: 0.9}} {
		allocs := testing.AllocsPerRun(50, func() {
			if _, err := SampleWithDraw(logits, cfg, 0.371); err != nil {
				panic(err)
			}
		})
		if allocs > 3 {
			t.Fatalf("cfg=%+v allocs=%g, want <=3 independent of K/vocabulary", cfg, allocs)
		}
	}
}
