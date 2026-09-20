package needle

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

var decoderFixtureTokens = []int{2, 7, 4, 9, 3, 6, 5, 8, 7, 3, 5, 2}

const (
	decoderAbsTol = 1e-4
	decoderRelTol = 5e-3
)

func decoderTokens(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = decoderFixtureTokens[i%len(decoderFixtureTokens)]
	}
	return out
}

func decoderLastRow(t testing.TB, logits []float32, cols int) []float32 {
	t.Helper()
	if cols <= 0 || len(logits) < cols || len(logits)%cols != 0 {
		t.Fatalf("invalid logits shape len=%d cols=%d", len(logits), cols)
	}
	return append([]float32(nil), logits[len(logits)-cols:]...)
}

func decoderCloseEnough(got, want []float32, abs, rel float64) error {
	if len(got) != len(want) {
		return fmt.Errorf("length %d want %d", len(got), len(want))
	}
	maxErr := 0.0
	idx := 0
	for i, v := range got {
		err := math.Abs(float64(v - want[i]))
		if err > maxErr {
			maxErr = err
			idx = i
		}
		if math.IsNaN(float64(v)) || err > abs+rel*math.Abs(float64(want[i])) {
			return fmt.Errorf("mismatch at %d got %.9g want %.9g max abs %.8g at %d", i, v, want[i], maxErr, idx)
		}
	}
	return nil
}

func decoderABModel(t testing.TB) *Model {
	t.Helper()
	_, f := extended(t)
	m, err := New(&checkpoint.Checkpoint{Config: f.AB.Config, FormatVersion: 2, Tensors: f.AB.Tensors})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func decoderArchiveWindowFixture(t testing.TB, window int) *Model {
	t.Helper()
	m, _ := fixture(t)
	cfg := m.Configuration()
	if window < 1 || window >= cfg.MaxSeq {
		t.Fatalf("window %d outside 1..%d", window, cfg.MaxSeq-1)
	}
	cp := m.Checkpoint()
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(cp.Config, &raw); err != nil {
		t.Fatal(err)
	}
	for key, value := range map[string]any{
		"go_archive_decoded":   true,
		"go_archive_kv_window": window,
	} {
		b, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		raw[key] = b
	}
	var err error
	cp.Config, err = json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	out, err := New(cp)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func decoderStepParity(t testing.TB, m *Model, opts DecoderOptions, tokens []int, abs, rel float64) {
	t.Helper()
	dec, err := m.NewDecoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	if got := dec.Position(); got != 0 {
		t.Fatalf("initial position=%d want 0", got)
	}
	cacheBytes := dec.CacheBytes()
	if cacheBytes <= 0 {
		t.Fatalf("CacheBytes=%d want >0", cacheBytes)
	}
	outVocab := m.Configuration().OutVocab
	for i, tok := range tokens {
		wantAll, err := m.Forward(tokens[:i+1], opts.Execution)
		if err != nil {
			t.Fatalf("forward prefix %d: %v", i+1, err)
		}
		want := decoderLastRow(t, wantAll, outVocab)
		got, err := dec.Step(context.Background(), tok)
		if err != nil {
			t.Fatalf("step %d: %v", i+1, err)
		}
		compare(t, fmt.Sprintf("prefix %d", i+1), got, want, abs, rel)
		if got := dec.Position(); got != i+1 {
			t.Fatalf("position after step %d = %d want %d", i+1, got, i+1)
		}
		if got := dec.CacheBytes(); got != cacheBytes {
			t.Fatalf("cache resized after step %d: %d -> %d", i+1, cacheBytes, got)
		}
	}
}

func TestDecoderStepParity(t *testing.T) {
	cq := &Quantization{WeightBits: 4, ActivationBits: 8, KVBits: 8}
	cases := []struct {
		name  string
		build func(testing.TB) (*Model, DecoderOptions)
	}{
		{
			name: "needle2-fp32",
			build: func(t testing.TB) (*Model, DecoderOptions) {
				m, _ := fixtureFile(t, "needle2.json")
				return m, DecoderOptions{Capacity: len(decoderFixtureTokens)}
			},
		},
		{
			name: "needle3-fp32",
			build: func(t testing.TB) (*Model, DecoderOptions) {
				m, _ := fixtureFile(t, "needle3.json")
				return m, DecoderOptions{Capacity: len(decoderFixtureTokens)}
			},
		},
		{
			name: "needle3-cq",
			build: func(t testing.TB) (*Model, DecoderOptions) {
				m, _ := fixtureFile(t, "needle3-cq.json")
				return m, DecoderOptions{Capacity: len(decoderFixtureTokens), Execution: Options{Quant: cq}}
			},
		},
		{
			name: "needle3-ab",
			build: func(t testing.TB) (*Model, DecoderOptions) {
				return decoderABModel(t), DecoderOptions{Capacity: len(decoderFixtureTokens), Execution: Options{Quant: cq}}
			},
		},
		{
			name: "archive3",
			build: func(t testing.TB) (*Model, DecoderOptions) {
				m, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
				if err != nil {
					t.Fatal(err)
				}
				return m, DecoderOptions{Capacity: len(decoderFixtureTokens)}
			},
		},
		{
			name: "sliced3-depth2",
			build: func(t testing.TB) (*Model, DecoderOptions) {
				m, _ := extended(t)
				child, err := m.SliceDepth(2)
				if err != nil {
					t.Fatal(err)
				}
				return child, DecoderOptions{Capacity: len(decoderFixtureTokens)}
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, opts := tc.build(t)
			decoderStepParity(t, m, opts, decoderFixtureTokens, decoderAbsTol, decoderRelTol)
		})
	}
}

func TestDecoderResetReuseAndOwnedLogits(t *testing.T) {
	m, _ := fixture(t)
	opts := DecoderOptions{Capacity: len(decoderFixtureTokens)}
	dec, err := m.NewDecoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	cacheBytes := dec.CacheBytes()
	if cacheBytes <= 0 {
		t.Fatalf("CacheBytes=%d want >0", cacheBytes)
	}
	outVocab := m.Configuration().OutVocab

	want1All, err := m.Forward(decoderFixtureTokens[:1], opts.Execution)
	if err != nil {
		t.Fatal(err)
	}
	want1 := decoderLastRow(t, want1All, outVocab)
	got1, err := dec.Step(context.Background(), decoderFixtureTokens[0])
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "first step", got1, want1, decoderAbsTol, decoderRelTol)
	if len(got1) == 0 {
		t.Fatal("empty logits")
	}
	got1[0] = 123456

	want2All, err := m.Forward(decoderFixtureTokens[:2], opts.Execution)
	if err != nil {
		t.Fatal(err)
	}
	want2 := decoderLastRow(t, want2All, outVocab)
	got2, err := dec.Step(context.Background(), decoderFixtureTokens[1])
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "second step after caller mutation", got2, want2, decoderAbsTol, decoderRelTol)
	if got := dec.CacheBytes(); got != cacheBytes {
		t.Fatalf("cache resized after step: %d -> %d", cacheBytes, got)
	}

	dec.Reset()
	if got := dec.Position(); got != 0 {
		t.Fatalf("position after reset=%d want 0", got)
	}
	if got := dec.CacheBytes(); got != cacheBytes {
		t.Fatalf("cache changed across reset: %d -> %d", cacheBytes, got)
	}
	again, err := dec.Step(context.Background(), decoderFixtureTokens[0])
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "first step after reset", again, want1, decoderAbsTol, decoderRelTol)
}

func TestDecoderFailureAtomicity(t *testing.T) {
	m, _ := fixture(t)
	outVocab := m.Configuration().OutVocab

	t.Run("invalid token", func(t *testing.T) {
		dec, err := m.NewDecoder(DecoderOptions{Capacity: 3})
		if err != nil {
			t.Fatal(err)
		}
		if _, err = dec.Step(context.Background(), decoderFixtureTokens[0]); err != nil {
			t.Fatal(err)
		}
		pos, cacheBytes := dec.Position(), dec.CacheBytes()
		got, err := dec.Step(context.Background(), -1)
		if got != nil || err == nil {
			t.Fatalf("invalid token got=%v err=%v", got, err)
		}
		if dec.Position() != pos || dec.CacheBytes() != cacheBytes {
			t.Fatalf("invalid token mutated state: pos=%d cache=%d want pos=%d cache=%d", dec.Position(), dec.CacheBytes(), pos, cacheBytes)
		}
		wantAll, err := m.Forward(decoderFixtureTokens[:2], Options{})
		if err != nil {
			t.Fatal(err)
		}
		want := decoderLastRow(t, wantAll, outVocab)
		got, err = dec.Step(context.Background(), decoderFixtureTokens[1])
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "after invalid token", got, want, decoderAbsTol, decoderRelTol)
	})

	t.Run("cancelled context", func(t *testing.T) {
		dec, err := m.NewDecoder(DecoderOptions{Capacity: 3})
		if err != nil {
			t.Fatal(err)
		}
		if _, err = dec.Step(context.Background(), decoderFixtureTokens[0]); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		pos, cacheBytes := dec.Position(), dec.CacheBytes()
		got, err := dec.Step(ctx, decoderFixtureTokens[1])
		if got != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled step got=%v err=%v", got, err)
		}
		if dec.Position() != pos || dec.CacheBytes() != cacheBytes {
			t.Fatalf("cancelled step mutated state: pos=%d cache=%d want pos=%d cache=%d", dec.Position(), dec.CacheBytes(), pos, cacheBytes)
		}
		wantAll, err := m.Forward(decoderFixtureTokens[:2], Options{})
		if err != nil {
			t.Fatal(err)
		}
		want := decoderLastRow(t, wantAll, outVocab)
		got, err = dec.Step(context.Background(), decoderFixtureTokens[1])
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "after cancellation", got, want, decoderAbsTol, decoderRelTol)
	})

	t.Run("capacity overflow", func(t *testing.T) {
		dec, err := m.NewDecoder(DecoderOptions{Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		if _, err = dec.Step(context.Background(), decoderFixtureTokens[0]); err != nil {
			t.Fatal(err)
		}
		pos, cacheBytes := dec.Position(), dec.CacheBytes()
		got, err := dec.Step(context.Background(), decoderFixtureTokens[1])
		if got != nil || err == nil {
			t.Fatalf("capacity overflow got=%v err=%v", got, err)
		}
		if dec.Position() != pos || dec.CacheBytes() != cacheBytes {
			t.Fatalf("capacity overflow mutated state: pos=%d cache=%d want pos=%d cache=%d", dec.Position(), dec.CacheBytes(), pos, cacheBytes)
		}
		dec.Reset()
		if dec.Position() != 0 || dec.CacheBytes() != cacheBytes {
			t.Fatalf("reset after overflow pos=%d cache=%d want pos=0 cache=%d", dec.Position(), dec.CacheBytes(), cacheBytes)
		}
		wantAll, err := m.Forward(decoderFixtureTokens[:1], Options{})
		if err != nil {
			t.Fatal(err)
		}
		want := decoderLastRow(t, wantAll, outVocab)
		got, err = dec.Step(context.Background(), decoderFixtureTokens[0])
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "after overflow reset", got, want, decoderAbsTol, decoderRelTol)
	})
}

func TestDecoderStepWorkBudgetKeepsPosition(t *testing.T) {
	m, _ := fixture(t)
	dec, err := m.NewDecoder(DecoderOptions{Capacity: len(decoderFixtureTokens), Execution: Options{MaxWorkBytes: 1}})
	if err != nil {
		t.Fatal(err)
	}
	cacheBytes := dec.CacheBytes()
	got, err := dec.Step(context.Background(), decoderFixtureTokens[0])
	if got != nil || err == nil {
		t.Fatalf("tiny work budget got=%v err=%v", got, err)
	}
	if dec.Position() != 0 || dec.CacheBytes() != cacheBytes {
		t.Fatalf("tiny work budget mutated state: pos=%d cache=%d want pos=0 cache=%d", dec.Position(), dec.CacheBytes(), cacheBytes)
	}
}

func TestDecoderCapacityDefaultsAndAdmission(t *testing.T) {
	t.Run("explicit over max seq", func(t *testing.T) {
		m, _ := fixtureFile(t, "needle2.json")
		cfg := m.Configuration()
		if _, err := m.NewDecoder(DecoderOptions{Capacity: cfg.MaxSeq + 1}); err == nil {
			t.Fatal("accepted capacity above max_seq")
		}
	})

	t.Run("default uses max seq", func(t *testing.T) {
		m, _ := fixtureFile(t, "needle2.json")
		cfg := m.Configuration()
		dec, err := m.NewDecoder(DecoderOptions{})
		if err != nil {
			t.Fatal(err)
		}
		for i, tok := range decoderTokens(cfg.MaxSeq) {
			if _, err = dec.Step(context.Background(), tok); err != nil {
				t.Fatalf("step %d/%d: %v", i+1, cfg.MaxSeq, err)
			}
		}
		cacheBytes := dec.CacheBytes()
		got, err := dec.Step(context.Background(), decoderTokens(cfg.MaxSeq + 1)[cfg.MaxSeq])
		if got != nil || err == nil {
			t.Fatalf("overflow after default max_seq got=%v err=%v", got, err)
		}
		if dec.Position() != cfg.MaxSeq || dec.CacheBytes() != cacheBytes {
			t.Fatalf("default max_seq overflow mutated state: pos=%d cache=%d want pos=%d cache=%d", dec.Position(), dec.CacheBytes(), cfg.MaxSeq, cacheBytes)
		}
	})

	t.Run("archive window smaller than max seq", func(t *testing.T) {
		m := decoderArchiveWindowFixture(t, 4)
		if _, err := m.NewDecoder(DecoderOptions{Capacity: 5}); err == nil {
			t.Fatal("accepted capacity above archive window")
		}
		dec, err := m.NewDecoder(DecoderOptions{})
		if err != nil {
			t.Fatal(err)
		}
		for i, tok := range decoderTokens(4) {
			if _, err = dec.Step(context.Background(), tok); err != nil {
				t.Fatalf("step %d/4: %v", i+1, err)
			}
		}
		cacheBytes := dec.CacheBytes()
		got, err := dec.Step(context.Background(), decoderTokens(5)[4])
		if got != nil || err == nil {
			t.Fatalf("archive window overflow got=%v err=%v", got, err)
		}
		if dec.Position() != 4 || dec.CacheBytes() != cacheBytes {
			t.Fatalf("archive window overflow mutated state: pos=%d cache=%d want pos=4 cache=%d", dec.Position(), dec.CacheBytes(), cacheBytes)
		}
	})
}

func TestDecoderMaxCacheBytesAdmission(t *testing.T) {
	m, _ := fixture(t)
	dec, err := m.NewDecoder(DecoderOptions{Capacity: len(decoderFixtureTokens)})
	if err != nil {
		t.Fatal(err)
	}
	need := dec.CacheBytes()
	if need <= 1 {
		t.Fatalf("CacheBytes=%d want >1", need)
	}
	if _, err := m.NewDecoder(DecoderOptions{Capacity: len(decoderFixtureTokens), MaxCacheBytes: need - 1}); err == nil {
		t.Fatal("accepted MaxCacheBytes below required cache")
	}
}

func TestDecoderInvalidQuantValidation(t *testing.T) {
	cases := []struct {
		name string
		fn   func(testing.TB) error
	}{
		{
			name: "needle2 rejects cq",
			fn: func(t testing.TB) error {
				m, _ := fixtureFile(t, "needle2.json")
				_, err := m.NewDecoder(DecoderOptions{Execution: Options{Quant: &Quantization{WeightBits: 4}}})
				return err
			},
		},
		{
			name: "needle3 rejects unsupported weight bits",
			fn: func(t testing.TB) error {
				m, _ := fixtureFile(t, "needle3.json")
				_, err := m.NewDecoder(DecoderOptions{Execution: Options{Quant: &Quantization{WeightBits: 3}}})
				return err
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.fn(t); err == nil {
				t.Fatal("invalid quantization accepted")
			}
		})
	}
}

func TestDecoderConcurrentSeparateSessions(t *testing.T) {
	m, _ := fixture(t)
	wantAll, err := m.Forward(decoderFixtureTokens, Options{})
	if err != nil {
		t.Fatal(err)
	}
	want := decoderLastRow(t, wantAll, m.Configuration().OutVocab)
	const workers = 4
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			dec, err := m.NewDecoder(DecoderOptions{Capacity: len(decoderFixtureTokens)})
			if err != nil {
				errs <- err
				return
			}
			var got []float32
			for _, tok := range decoderFixtureTokens {
				got, err = dec.Step(context.Background(), tok)
				if err != nil {
					errs <- err
					return
				}
			}
			if dec.Position() != len(decoderFixtureTokens) {
				errs <- fmt.Errorf("position=%d want %d", dec.Position(), len(decoderFixtureTokens))
				return
			}
			if err := decoderCloseEnough(got, want, decoderAbsTol, decoderRelTol); err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func BenchmarkDecoderNeedle3(b *testing.B) {
	m, _ := fixture(b)
	b.Run("cached", func(b *testing.B) {
		dec, err := m.NewDecoder(DecoderOptions{Capacity: len(decoderFixtureTokens)})
		if err != nil {
			b.Fatal(err)
		}
		ctx := context.Background()
		b.ReportAllocs()
		for b.Loop() {
			dec.Reset()
			for _, tok := range decoderFixtureTokens {
				if _, err := dec.Step(ctx, tok); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
	b.Run("full-prefix", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			for i := range decoderFixtureTokens {
				if _, err := m.Forward(decoderFixtureTokens[:i+1], Options{}); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
}

func TestGenerateCachedMatchesReference(t *testing.T) {
	for _, file := range []string{"needle2.json", "needle3.json", "needle3-cq.json"} {
		m, f := fixtureFile(t, file)
		opts := Options{}
		if file == "needle3-cq.json" {
			opts.Quant = &Quantization{WeightBits: 4, ActivationBits: 8, KVBits: 8}
		}
		want, err := m.Generate(context.Background(), f.Tokens, 8, -1, opts)
		if err != nil {
			t.Fatal(err)
		}
		got, err := m.GenerateCached(context.Background(), f.Tokens, 8, -1, DecoderOptions{Execution: opts})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Fatalf("%s cached %v want %v", file, got, want)
		}
	}
}

// Cancel after the first layer boundary check: this exercises a failure after
// provisional QKV/engram work, not just pre-admission cancellation.
type cancelAfterChecks struct {
	context.Context
	checks   int
	cancelAt int
}

func (c *cancelAfterChecks) Err() error {
	c.checks++
	if c.checks >= c.cancelAt {
		return context.Canceled
	}
	return nil
}
func TestDecoderMidStepRollback(t *testing.T) {
	m, _ := fixture(t)
	dec, err := m.NewDecoder(DecoderOptions{Capacity: 8})
	if err != nil {
		t.Fatal(err)
	}
	_, err = dec.Step(context.Background(), 2)
	if err != nil {
		t.Fatal(err)
	}
	ctx := &cancelAfterChecks{Context: context.Background(), cancelAt: 3}
	if _, err = dec.Step(ctx, 7); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if dec.Position() != 1 {
		t.Fatal("cancelled step committed")
	}
	got, err := dec.Step(context.Background(), 7)
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.Forward([]int{2, 7}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "cancel rollback", got, want[len(want)-m.config.OutVocab:], 1e-4, 5e-3)
	// Fail at a small workspace after several allocations, then retry normally.
	old := dec.opts.MaxWorkBytes
	dec.opts.MaxWorkBytes = 8000
	if _, err = dec.Step(context.Background(), 4); err == nil {
		t.Fatal("expected workspace failure")
	}
	if dec.Position() != 2 {
		t.Fatal("workspace failure committed")
	}
	dec.opts.MaxWorkBytes = old
	got, err = dec.Step(context.Background(), 4)
	if err != nil {
		t.Fatal(err)
	}
	want, _ = m.Forward([]int{2, 7, 4}, Options{})
	compare(t, "budget rollback", got, want[len(want)-m.config.OutVocab:], 1e-4, 5e-3)
}

func TestDecoderQuantOptionsAreOwned(t *testing.T) {
	m, _ := fixture(t)
	q := &Quantization{WeightBits: 4, ActivationBits: 8, KVBits: 8}
	dec, err := m.NewDecoder(DecoderOptions{Capacity: 2, Execution: Options{Quant: q}})
	if err != nil {
		t.Fatal(err)
	}
	q.WeightBits = 3
	q.ActivationBits = 0
	got, err := dec.Step(context.Background(), 2)
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.Forward([]int{2}, Options{Quant: &Quantization{WeightBits: 4, ActivationBits: 8, KVBits: 8}})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "owned quant options", got, want, 1e-4, 5e-3)
}

func TestDecoderCacheBuffersAreCleared(t *testing.T) {
	m, _ := fixture(t)
	dec, err := m.NewDecoder(DecoderOptions{Capacity: 4})
	if err != nil {
		t.Fatal(err)
	}
	for _, token := range []int{2, 7, 4} {
		if _, err = dec.Step(context.Background(), token); err != nil {
			t.Fatal(err)
		}
	}
	dec.Reset()
	for _, id := range dec.ids {
		if id != 0 {
			t.Fatal("prompt retained")
		}
	}
	for _, l := range dec.layers {
		for _, r := range []rowRing{l.key, l.val, l.qraw, l.kraw, l.vraw} {
			for _, v := range r.data {
				if v != 0 {
					t.Fatal("cache not cleared")
				}
			}
		}
	}
	for _, r := range dec.engrams {
		for _, v := range r.data {
			if v != 0 {
				t.Fatal("engram history retained")
			}
		}
	}
}
