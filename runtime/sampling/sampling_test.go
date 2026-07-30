package sampling

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

func f32(vs ...float32) []float32 { return vs }

// ---- config / error contract ----

func TestSampleEmptyLogits(t *testing.T) {
	_, err := SampleWithDraw(nil, Config{Temperature: 1}, 0)
	if !errors.Is(err, ErrEmptyLogits) {
		t.Fatalf("want ErrEmptyLogits, got %v", err)
	}
	_, err = SampleWithDraw([]float32{}, Config{Temperature: 0}, 0)
	if !errors.Is(err, ErrEmptyLogits) {
		t.Fatalf("want ErrEmptyLogits for greedy too, got %v", err)
	}
}

func TestSampleAllInvalid(t *testing.T) {
	nan := float32(math.NaN())
	ninf := float32(math.Inf(-1))
	logits := f32(nan, ninf, nan, ninf)

	if _, err := SampleWithDraw(logits, Config{Temperature: 0}, 0); !errors.Is(err, ErrAllInvalid) {
		t.Fatalf("greedy: want ErrAllInvalid, got %v", err)
	}
	if _, err := SampleWithDraw(logits, Config{Temperature: 1}, 0.5); !errors.Is(err, ErrAllInvalid) {
		t.Fatalf("topk-unlimited: want ErrAllInvalid, got %v", err)
	}
	if _, err := SampleWithDraw(logits, Config{Temperature: 1, TopK: 2}, 0.5); !errors.Is(err, ErrAllInvalid) {
		t.Fatalf("bounded topk: want ErrAllInvalid, got %v", err)
	}
}

func TestConfigValidateErrors(t *testing.T) {
	cases := []Config{
		{Temperature: 1, TopK: -1},
		{Temperature: 1, TopP: -0.1},
		{Temperature: 1, TopP: 1.1},
		{Temperature: 1, TopP: math.NaN()},
		{Temperature: math.NaN()},
	}
	for i, cfg := range cases {
		if _, err := SampleWithDraw([]float32{1, 2, 3}, cfg, 0); !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("case %d: want ErrInvalidConfig, got %v", i, err)
		}
	}
}

func TestSampleNilRandRequiresGreedyOrError(t *testing.T) {
	logits := []float32{1, 2, 3}
	if _, err := Sample(logits, Config{Temperature: 0}, nil); err != nil {
		t.Fatalf("greedy with nil rng should succeed, got %v", err)
	}
	if _, err := Sample(logits, Config{Temperature: 1}, nil); !errors.Is(err, ErrNilRand) {
		t.Fatalf("want ErrNilRand, got %v", err)
	}
}

// ---- greedy ----

func TestGreedyPicksMax(t *testing.T) {
	logits := f32(1, 5, 3, 5, 2)
	res, err := SampleWithDraw(logits, Config{Temperature: 0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.TokenID != 1 {
		t.Fatalf("want token 1 (first max, ascending id tie-break), got %d", res.TokenID)
	}
	if res.Logit != 5 {
		t.Fatalf("want logit 5, got %v", res.Logit)
	}
	if res.Candidates != 5 {
		t.Fatalf("want 5 valid candidates, got %d", res.Candidates)
	}
}

func TestGreedyNegativeTemperatureAlsoGreedy(t *testing.T) {
	logits := f32(1, 5, 3)
	res, err := SampleWithDraw(logits, Config{Temperature: -1, TopK: 1, TopP: 0.1}, 0.999)
	if err != nil {
		t.Fatal(err)
	}
	if res.TokenID != 1 {
		t.Fatalf("want greedy token 1 regardless of TopK/TopP, got %d", res.TokenID)
	}
}

func TestGreedyExcludesNaNAndNegInf(t *testing.T) {
	nan := float32(math.NaN())
	ninf := float32(math.Inf(-1))
	logits := f32(nan, ninf, 4, ninf, nan)
	res, err := SampleWithDraw(logits, Config{Temperature: 0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.TokenID != 2 {
		t.Fatalf("want token 2, got %d", res.TokenID)
	}
	if res.Candidates != 1 {
		t.Fatalf("want 1 valid candidate, got %d", res.Candidates)
	}
}

func TestGreedyPositiveInfinityTieBreaksAscending(t *testing.T) {
	pinf := float32(math.Inf(1))
	logits := f32(1, pinf, 2, pinf, 0)
	res, err := SampleWithDraw(logits, Config{Temperature: 0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.TokenID != 1 {
		t.Fatalf("want first +Inf token (id 1), got %d", res.TokenID)
	}
}

// ---- candidate ordering / ties (via boundedTopK + sort) ----

func TestBoundedTopKMatchesFullSortPrefix(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	n := 500
	logits := make([]float32, n)
	for i := range logits {
		logits[i] = float32(rng.Intn(50)) // lots of ties on purpose
	}
	for _, k := range []int{1, 2, 5, 17, 100, n, n + 10} {
		got, err := boundedTopK(logits, k)
		if err != nil {
			t.Fatal(err)
		}
		full, err := fullScan(logits)
		if err != nil {
			t.Fatal(err)
		}
		sortCandidatesDesc(full)
		want := full
		if k < len(full) {
			want = full[:k]
		}
		if len(got) != len(want) {
			t.Fatalf("k=%d: len mismatch got %d want %d", k, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("k=%d: mismatch at %d: got %+v want %+v", k, i, got[i], want[i])
			}
		}
	}
}

func TestBoundedTopKTieBreakAscendingIDAtBoundary(t *testing.T) {
	// Tokens 0..4 all share logit 1.0; only 3 fit in top-3, so ids 0,1,2 win.
	logits := f32(1, 1, 1, 1, 1)
	got, err := boundedTopK(logits, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("want 3 candidates, got %d", len(got))
	}
	for i, c := range got {
		if int(c.idx) != i {
			t.Fatalf("want ascending ids [0,1,2], got idx=%d at pos %d", c.idx, i)
		}
	}
}

func TestBoundedTopKUnlimitedZeroTopKUsesFullScan(t *testing.T) {
	logits := f32(3, 1, 4, 1, 5)
	res, err := SampleWithDraw(logits, Config{Temperature: 1, TopK: 0}, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if res.Candidates != 5 {
		t.Fatalf("want all 5 candidates eligible, got %d", res.Candidates)
	}
}

// ---- NaN / Infinity in stochastic sampling ----

func TestSampleExcludesNaNTokens(t *testing.T) {
	nan := float32(math.NaN())
	logits := f32(1, nan, 1, nan, 1)
	for draw := 0.0; draw <= 1.0; draw += 0.1 {
		res, err := SampleWithDraw(logits, Config{Temperature: 1}, draw)
		if err != nil {
			t.Fatal(err)
		}
		if res.TokenID == 1 || res.TokenID == 3 {
			t.Fatalf("NaN token %d must never be selected", res.TokenID)
		}
	}
}

func TestSampleExcludesNegativeInfinity(t *testing.T) {
	ninf := float32(math.Inf(-1))
	logits := f32(ninf, 5, ninf, 5, ninf)
	for draw := 0.0; draw <= 1.0; draw += 0.1 {
		res, err := SampleWithDraw(logits, Config{Temperature: 1}, draw)
		if err != nil {
			t.Fatal(err)
		}
		if res.TokenID != 1 && res.TokenID != 3 {
			t.Fatalf("-Inf token must never be selected, got %d", res.TokenID)
		}
	}
}

func TestSamplePositiveInfinityDeterministicUniformAmongTies(t *testing.T) {
	pinf := float32(math.Inf(1))
	logits := f32(0, pinf, 10, pinf, -3)
	// Regardless of temperature/draw, only the two +Inf tokens (1, 3) may
	// win, and the split must be exactly even (each covers half the [0,1]
	// draw range) since finite tokens carry zero probability mass.
	cfg := Config{Temperature: 0.7}
	res, err := SampleWithDraw(logits, cfg, 0.0)
	if err != nil {
		t.Fatal(err)
	}
	if res.TokenID != 1 {
		t.Fatalf("draw=0: want token 1, got %d", res.TokenID)
	}
	res, err = SampleWithDraw(logits, cfg, 0.9999999)
	if err != nil {
		t.Fatal(err)
	}
	if res.TokenID != 3 {
		t.Fatalf("draw~1: want token 3, got %d", res.TokenID)
	}
	for draw := 0.0; draw <= 1.0; draw += 0.05 {
		res, err := SampleWithDraw(logits, cfg, draw)
		if err != nil {
			t.Fatal(err)
		}
		if res.TokenID != 1 && res.TokenID != 3 {
			t.Fatalf("draw=%v: finite token must never win against +Inf, got %d", draw, res.TokenID)
		}
	}
}

func TestSamplePositiveInfinitySingleWinnerIsCertain(t *testing.T) {
	pinf := float32(math.Inf(1))
	logits := f32(100, pinf, -50, 3)
	for draw := 0.0; draw <= 1.0; draw += 0.1 {
		res, err := SampleWithDraw(logits, Config{Temperature: 2}, draw)
		if err != nil {
			t.Fatal(err)
		}
		if res.TokenID != 1 {
			t.Fatalf("draw=%v: sole +Inf token must be certain winner, got %d", draw, res.TokenID)
		}
	}
}

// ---- TopK behavior ----

func TestTopKRestrictsCandidateSet(t *testing.T) {
	logits := f32(0, 1, 2, 3, 4) // token 4 is best, then 3, 2, 1, 0
	cfg := Config{Temperature: 1, TopK: 2}
	for draw := 0.0; draw <= 1.0; draw += 0.05 {
		res, err := SampleWithDraw(logits, cfg, draw)
		if err != nil {
			t.Fatal(err)
		}
		if res.TokenID != 4 && res.TokenID != 3 {
			t.Fatalf("draw=%v: TopK=2 must restrict to tokens {3,4}, got %d", draw, res.TokenID)
		}
		if res.Candidates != 2 {
			t.Fatalf("want 2 candidates, got %d", res.Candidates)
		}
	}
}

func TestTopKZeroMeansUnlimited(t *testing.T) {
	logits := f32(0, 1, 2, 3, 4)
	res, err := SampleWithDraw(logits, Config{Temperature: 1, TopK: 0}, 0.0)
	if err != nil {
		t.Fatal(err)
	}
	if res.Candidates != 5 {
		t.Fatalf("TopK=0 should keep all 5 candidates, got %d", res.Candidates)
	}
}

func TestTopKLargerThanVocabIsHarmless(t *testing.T) {
	logits := f32(0, 1, 2)
	res, err := SampleWithDraw(logits, Config{Temperature: 1, TopK: 1000}, 0.0)
	if err != nil {
		t.Fatal(err)
	}
	if res.Candidates != 3 {
		t.Fatalf("want 3 candidates, got %d", res.Candidates)
	}
}

// ---- TopP behavior ----

func TestTopPIncludesThresholdCrossingToken(t *testing.T) {
	// Two tokens with a huge logit gap so token 1 (higher logit) dominates
	// probability mass; a small TopP should keep exactly it (the crossing
	// token), even though its own probability alone may already exceed
	// TopP.
	logits := f32(0, 20)
	res, err := SampleWithDraw(logits, Config{Temperature: 1, TopP: 0.5}, 0.0)
	if err != nil {
		t.Fatal(err)
	}
	if res.TokenID != 1 {
		t.Fatalf("want dominant token 1, got %d", res.TokenID)
	}
	if res.Candidates != 1 {
		t.Fatalf("want exactly 1 surviving candidate (the crossing token), got %d", res.Candidates)
	}
}

func TestTopPZeroAndOneDisabled(t *testing.T) {
	logits := f32(0, 1, 2, 3, 4)
	for _, p := range []float64{0, 1} {
		res, err := SampleWithDraw(logits, Config{Temperature: 1, TopP: p}, 0.0)
		if err != nil {
			t.Fatal(err)
		}
		if res.Candidates != 5 {
			t.Fatalf("TopP=%v should disable truncation, got %d candidates", p, res.Candidates)
		}
	}
}

func TestTopPOperatesOverDescendingCandidatesEvenWithoutTopK(t *testing.T) {
	// TopK disabled (0): fullScan returns ascending-id order; TopP must
	// still sort to descending logit before truncating.
	logits := f32(4, 0, 3, 1, 2) // best is token 0, then 2, 4, 3, 1
	res, err := SampleWithDraw(logits, Config{Temperature: 0.3, TopP: 0.2}, 0.0)
	if err != nil {
		t.Fatal(err)
	}
	if res.TokenID != 0 {
		t.Fatalf("want dominant token 0, got %d", res.TokenID)
	}
}

func TestTopKThenTopPComposition(t *testing.T) {
	logits := f32(0, 1, 2, 3, 100)
	// TopK=3 keeps {100,3,2} (tokens 4,3,2); TopP=0.1 with such a dominant
	// logit should further truncate to just token 4.
	res, err := SampleWithDraw(logits, Config{Temperature: 1, TopK: 3, TopP: 0.1}, 0.0)
	if err != nil {
		t.Fatal(err)
	}
	if res.TokenID != 4 {
		t.Fatalf("want token 4, got %d", res.TokenID)
	}
	if res.Candidates != 1 {
		t.Fatalf("want 1 surviving candidate, got %d", res.Candidates)
	}
}

// ---- fixed seed / reproducibility ----

func TestSampleFixedSeedIsReproducible(t *testing.T) {
	logits := make([]float32, 1000)
	src := rand.New(rand.NewSource(123))
	for i := range logits {
		logits[i] = float32(src.NormFloat64())
	}
	cfg := Config{Temperature: 0.8, TopK: 40, TopP: 0.9}

	run := func(seed int64) []int {
		rng := rand.New(rand.NewSource(seed))
		out := make([]int, 200)
		for i := range out {
			res, err := Sample(logits, cfg, rng)
			if err != nil {
				t.Fatal(err)
			}
			out[i] = res.TokenID
		}
		return out
	}

	a := run(42)
	b := run(42)
	if len(a) != len(b) {
		t.Fatal("length mismatch")
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("mismatch at %d: %d vs %d", i, a[i], b[i])
		}
	}

	c := run(43)
	same := true
	for i := range a {
		if a[i] != c[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("different seeds produced identical sequences (suspicious)")
	}
}

func TestSampleWithDrawExactReproducibility(t *testing.T) {
	logits := f32(1, 2, 3, 4, 5)
	cfg := Config{Temperature: 1, TopK: 3, TopP: 0.8}
	r1, err := SampleWithDraw(logits, cfg, 0.37)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := SampleWithDraw(logits, cfg, 0.37)
	if err != nil {
		t.Fatal(err)
	}
	if r1 != r2 {
		t.Fatalf("identical draws must produce identical results: %+v vs %+v", r1, r2)
	}
}

// ---- distribution sanity ----

func TestDistributionSanityMatchesSoftmaxProportions(t *testing.T) {
	logits := f32(0, 1, 2) // softmax favors token 2 > 1 > 0
	cfg := Config{Temperature: 1}
	rng := rand.New(rand.NewSource(99))
	counts := make(map[int]int)
	const trials = 20000
	for i := 0; i < trials; i++ {
		res, err := Sample(logits, cfg, rng)
		if err != nil {
			t.Fatal(err)
		}
		counts[res.TokenID]++
	}
	if counts[2] <= counts[1] || counts[1] <= counts[0] {
		t.Fatalf("expected monotonic counts by logit rank, got %v", counts)
	}
	// exp(0),exp(1),exp(2) -> weights ~ 1, 2.718, 7.389; total ~11.107
	wantP2 := 7.389 / 11.107
	gotP2 := float64(counts[2]) / trials
	if math.Abs(gotP2-wantP2) > 0.02 {
		t.Fatalf("token 2 empirical prob %.4f too far from expected %.4f", gotP2, wantP2)
	}
}

func TestDistributionSanityLowTemperatureConcentrates(t *testing.T) {
	logits := f32(0, 1, 2)
	cfg := Config{Temperature: 0.05}
	rng := rand.New(rand.NewSource(7))
	counts := make(map[int]int)
	const trials = 2000
	for i := 0; i < trials; i++ {
		res, err := Sample(logits, cfg, rng)
		if err != nil {
			t.Fatal(err)
		}
		counts[res.TokenID]++
	}
	if counts[2] < trials-5 {
		t.Fatalf("low temperature should concentrate almost all mass on token 2, got %v", counts)
	}
}

// ---- stable ordering across full pipeline ----

func TestOrderingIsStableAcrossRepeats(t *testing.T) {
	logits := f32(5, 5, 3, 5, 1, 3, 5)
	full, err := fullScan(logits)
	if err != nil {
		t.Fatal(err)
	}
	sortCandidatesDesc(full)
	wantIDs := []int32{0, 1, 3, 6, 2, 5, 4}
	for i, c := range full {
		if c.idx != wantIDs[i] {
			t.Fatalf("pos %d: want id %d, got %d (%+v)", i, wantIDs[i], c.idx, full)
		}
	}
}
