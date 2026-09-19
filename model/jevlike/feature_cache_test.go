package jevlike

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func cacheTestContract(dtype string) FeatureContract {
	return FeatureContract{Version: 1, ModelID: "test@pinned#sha256=" + strings.Repeat("a", 64), DatasetSHA256: strings.Repeat("b", 64), Backend: "synthetic-f32", TokenPolicy: "plain/no-bos/no-eos/reject-overlength", Representation: "causal/final-rmsnorm/all-token-rows", Pooling: "option-mean-f32-before-storage", DType: dtype, Width: 4, ContextTokens: 8, OptionTokens: 4}
}
func cacheTokenizer(s string) ([]int, error) {
	ids := make([]int, len(s))
	for i, b := range []byte(s) {
		ids[i] = int(b)
	}
	return ids, nil
}
func cacheSynthetic(ids []int) ([][]float32, error) {
	rows := make([][]float32, len(ids))
	for i, id := range ids {
		rows[i] = []float32{float32(id) / 128, float32(i) / 8, -0.17, 1.11}
	}
	return rows, nil
}
func TestFeatureCacheRoundtripPoolingOfflineAndIdentity(t *testing.T) {
	for _, dtype := range []string{"f32", "f16"} {
		t.Run(dtype, func(t *testing.T) {
			dir := t.TempDir()
			contract := cacheTestContract(dtype)
			calls := 0
			cache, err := OpenFeatureCache(dir, contract, 1<<20, cacheTokenizer, func(ids []int) ([][]float32, error) { calls++; return cacheSynthetic(ids) })
			if err != nil {
				t.Fatal(err)
			}
			context, err := cache.Encode("abcd", 8)
			if err != nil {
				t.Fatal(err)
			}
			option, err := cache.EncodeOption("ab", 4)
			if err != nil {
				t.Fatal(err)
			}
			again, err := cache.Encode("abcd", 8)
			if err != nil || !reflect.DeepEqual(again, context) {
				t.Fatal(err)
			}
			if calls != 2 {
				t.Fatal("encoder called on hit", calls)
			}
			// Requesting an already-pooled option returns the same vector, not a mean of
			// individually quantised token rows.
			if _, err := cache.EncodeOption("ab", 4); err != nil {
				t.Fatal(err)
			}
			if _, err := cache.Encode("123456789", 8); err == nil {
				t.Fatal("overlength truncated")
			}
			if _, err := OpenFeatureCache(dir, contract, 1<<20, cacheTokenizer, cacheSynthetic); err == nil {
				t.Fatal("two writers")
			}
			cache.Close()
			cache.Close()
			cache, err = OpenFeatureCache(dir, contract, 1<<20, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer cache.Close()
			offline, err := cache.Encode("abcd", 8)
			if err != nil || !reflect.DeepEqual(offline, context) {
				t.Fatal(err)
			}
			pooled, err := cache.EncodeOption("ab", 4)
			if err != nil || !reflect.DeepEqual(pooled, option) {
				t.Fatal(err)
			}
			if _, err := cache.Encode("new", 8); err == nil {
				t.Fatal("offline fallback")
			}
			changed := contract
			changed.ModelID = "different-same-width"
			if _, err := OpenFeatureCache(dir, changed, 1<<20, nil, nil); err == nil {
				t.Fatal("stale contract")
			}
			entries, _ := filepath.Glob(filepath.Join(dir, "*.jvf"))
			data, _ := os.ReadFile(entries[0])
			data[len(data)-1] ^= 1
			os.WriteFile(entries[0], data, 0o600)
			_, e1 := cache.Encode("abcd", 8)
			_, e2 := cache.EncodeOption("ab", 4)
			if e1 == nil && e2 == nil {
				t.Fatal("corrupt checksum accepted")
			}
		})
	}
}
func TestFeatureCacheRejectsNonfiniteAndBudget(t *testing.T) {
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), 70000} {
		c, e := OpenFeatureCache(t.TempDir(), cacheTestContract("f16"), 1<<20, cacheTokenizer, func(ids []int) ([][]float32, error) { return [][]float32{{bad, 0, 0, 0}}, nil })
		if e != nil {
			t.Fatal(e)
		}
		if _, e = c.Encode("a", 8); e == nil {
			t.Fatal("invalid features admitted", bad)
		}
		c.Close()
	}
	c, e := OpenFeatureCache(t.TempDir(), cacheTestContract("f32"), 1024, cacheTokenizer, cacheSynthetic)
	if e != nil {
		t.Fatal(e)
	}
	defer c.Close()
	if _, e = c.Encode("abcd", 8); e == nil {
		t.Fatal("budget ignored")
	}
}
func TestFeatureCheckpointRequiresExactEncoder(t *testing.T) {
	c, e := OpenFeatureCache(t.TempDir(), cacheTestContract("f32"), 1<<20, cacheTokenizer, cacheSynthetic)
	if e != nil {
		t.Fatal(e)
	}
	defer c.Close()
	head, _ := NewAttentionHead(4, 2)
	m := &FrozenScorer{Config: Config{Width: 4, Rank: 2, ContextTokens: 8, OptionTokens: 4}, Head: *head, Reference: c.FeatureReference(), Encoder: c}
	InitializeFrozenScorer(m, 7)
	checkpoint, e := FrozenCheckpoint(m)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = checkpoint.Frozen(c); e != nil {
		t.Fatal(e)
	}
	other := cacheTestContract("f32")
	other.ModelID = "other#sha256=" + strings.Repeat("c", 64)
	wrong, e := OpenFeatureCache(t.TempDir(), other, 1<<20, cacheTokenizer, cacheSynthetic)
	if e != nil {
		t.Fatal(e)
	}
	defer wrong.Close()
	if _, e = checkpoint.Frozen(wrong); e == nil {
		t.Fatal("same width wrong encoder accepted")
	}
}
func TestCachedEpochsNeverCallEncoder(t *testing.T) {
	calls := 0
	c, e := OpenFeatureCache(t.TempDir(), cacheTestContract("f32"), 1<<20, cacheTokenizer, func(ids []int) ([][]float32, error) { calls++; return cacheSynthetic(ids) })
	if e != nil {
		t.Fatal(e)
	}
	examples := []ChoiceExample{{Context: "abc", Options: []string{"a", "b"}, Label: 0}, {Context: "bcd", Options: []string{"b", "c"}, Label: 1}}
	for _, ex := range examples {
		c.Encode(ex.Context, 8)
		for _, text := range ex.Options {
			c.EncodeOption(text, 4)
		}
	}
	if calls != 5 {
		t.Fatal(calls)
	}
	dir := c.dir
	c.Close()
	offline, e := OpenFeatureCache(dir, cacheTestContract("f32"), 1<<20, nil, nil)
	if e != nil {
		t.Fatal(e)
	}
	defer offline.Close()
	head, _ := NewAttentionHead(4, 2)
	m := &FrozenScorer{Config: Config{Width: 4, Rank: 2, ContextTokens: 8, OptionTokens: 4}, Head: *head, Reference: offline.FeatureReference(), Encoder: offline}
	if _, e = TrainFrozenScorer(m, examples, examples, TrainConfig{Epochs: 2, BatchSize: 1}); e != nil {
		t.Fatal(e)
	}
	if calls != 5 || offline.Misses != 0 {
		t.Fatal("encoder used after extraction")
	}
}

func TestFeatureF16ConversionIsExplicitAndPreservesPooledOptions(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "f32")
	target := filepath.Join(root, "f16")
	c, e := OpenFeatureCache(source, cacheTestContract("f32"), 1<<20, cacheTokenizer, cacheSynthetic)
	if e != nil {
		t.Fatal(e)
	}
	rows, e := c.Encode("abc", 8)
	if e != nil {
		t.Fatal(e)
	}
	option, e := c.EncodeOption("ab", 4)
	if e != nil {
		t.Fatal(e)
	}
	c.Close()
	n, e := ConvertFeatureCache(source, target, 1<<20)
	if e != nil || n != 2 {
		t.Fatal(n, e)
	}
	out, e := OpenFeatureCache(target, cacheTestContract("f16"), 1<<20, nil, nil)
	if e != nil {
		t.Fatal(e)
	}
	defer out.Close()
	got, e := out.Encode("abc", 8)
	if e != nil {
		t.Fatal(e)
	}
	pooled, e := out.EncodeOption("ab", 4)
	if e != nil {
		t.Fatal(e)
	}
	for i := range rows {
		for j, v := range rows[i] {
			if math.Abs(float64(v-got[i][j])) > 0.001 {
				t.Fatal("bad quantisation")
			}
		}
	}
	for j, v := range option {
		if math.Abs(float64(v-pooled[j])) > 0.001 {
			t.Fatal("bad pooled quantisation")
		}
	}
	if _, e = ConvertFeatureCache(source, target, 1<<20); e == nil {
		t.Fatal("conversion overwrote existing cache")
	}
}

func TestFreshFeatureEncoderMatchesCached(t *testing.T) {
	for _, dtype := range []string{"f32", "f16"} {
		contract := cacheTestContract(dtype)
		cache, e := OpenFeatureCache(t.TempDir(), contract, 1<<20, cacheTokenizer, cacheSynthetic)
		if e != nil {
			t.Fatal(e)
		}
		live := &FeatureEncoder{Contract: contract, Tokenize: cacheTokenizer, EncodeTokens: cacheSynthetic}
		a, e := cache.Encode("abc", 8)
		if e != nil {
			t.Fatal(e)
		}
		b, e := live.Encode("abc", 8)
		if e != nil || !reflect.DeepEqual(a, b) {
			t.Fatal("fresh/cache context mismatch", e)
		}
		x, e := cache.EncodeOption("ab", 4)
		if e != nil {
			t.Fatal(e)
		}
		y, e := live.EncodeOption("ab", 4)
		if e != nil || !reflect.DeepEqual(x, y) {
			t.Fatal("fresh/cache option mismatch", e)
		}
		cache.Close()
	}
}
