package qwen3tts

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

func TestReferenceCPUSamplerMatchesPinnedSyntheticOracle(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "customvoice_0b6_ryan_hello", "sampling_probe.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Name        string    `json:"name"`
			Input       []float32 `json:"input_logits"`
			Adjusted    []float32 `json:"adjusted_logits"`
			Generated   []uint32  `json:"generated"`
			Temperature float64   `json:"temperature"`
			TopK        int       `json:"top_k"`
			TopP        float64   `json:"top_p"`
			Penalty     float64   `json:"repetition_penalty"`
			Seed        uint64    `json:"seed"`
			Tokens      []uint32  `json:"tokens"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) != 4 {
		t.Fatalf("cases=%d", len(fixture.Cases))
	}
	for _, c := range fixture.Cases {
		cfg := ReferenceSampleConfig{Temperature: c.Temperature, TopK: c.TopK, TopP: c.TopP, RepetitionPenalty: c.Penalty}
		sampler := NewReferenceCPUSampler(c.Seed)
		for i, want := range c.Tokens {
			got, err := sampler.Select(c.Input, cfg, c.Generated, false, true)
			if err != nil || got != want {
				t.Fatalf("%s draw %d got=%d want=%d err=%v", c.Name, i, got, want, err)
			}
		}
	}
}

func TestReferenceCPUSamplerValidationAndOwnership(t *testing.T) {
	cfg := ReferenceSampleConfig{Temperature: .7, TopK: 2, TopP: .9, RepetitionPenalty: 1}
	input := []float32{2, 1, -1}
	original := append([]float32(nil), input...)
	if _, err := (*ReferenceCPUSampler)(nil).Select(input, cfg, nil, false, true); err == nil {
		t.Fatal("accepted nil sampler")
	}
	for _, bad := range []ReferenceSampleConfig{
		{Temperature: .7, TopK: -1, TopP: .9, RepetitionPenalty: 1},
		{Temperature: math.NaN(), TopK: 2, TopP: .9, RepetitionPenalty: 1},
		{Temperature: math.Inf(1), TopK: 2, TopP: .9, RepetitionPenalty: 1},
		{Temperature: .7, TopK: 2, TopP: 1.1, RepetitionPenalty: 1},
		{Temperature: .7, TopK: 2, TopP: .9, RepetitionPenalty: 0},
		{Temperature: .7, TopK: 2, TopP: .9, RepetitionPenalty: math.Inf(1)},
		{Temperature: math.SmallestNonzeroFloat64, TopK: 2, TopP: .9, RepetitionPenalty: 1},
	} {
		if _, err := NewReferenceCPUSampler(42).Select(input, bad, nil, false, true); err == nil {
			t.Fatalf("accepted invalid config %+v", bad)
		}
	}
	for _, bad := range [][]float32{nil, {1, float32(math.NaN())}, {1, float32(math.Inf(1))}, {float32(math.Inf(-1)), float32(math.Inf(-1))}} {
		if _, err := NewReferenceCPUSampler(42).Select(bad, cfg, nil, false, true); err == nil {
			t.Fatalf("accepted invalid logits %v", bad)
		}
	}
	if _, err := NewReferenceCPUSampler(42).Select(input, cfg, nil, true, true); err == nil {
		t.Fatal("accepted invalid TTS vocab")
	}
	full := make([]float32, CodecVocabSize)
	full[CodecEOS] = 10
	full[42] = 2
	full[2049] = 100
	g := ReferenceSampleConfig{Temperature: 0, TopK: 1, TopP: .1, RepetitionPenalty: 1}
	if got, err := NewReferenceCPUSampler(42).Select(full, g, nil, true, false); err != nil || got != 42 {
		t.Fatalf("EOS minimum got=%d err=%v", got, err)
	}
	if got, err := NewReferenceCPUSampler(42).Select(full, g, nil, true, true); err != nil || got != CodecEOS {
		t.Fatalf("EOS allowed got=%d err=%v", got, err)
	}
	if got, err := NewReferenceCPUSampler(42).Select([]float32{2, 1}, ReferenceSampleConfig{Temperature: 0, TopK: 0, TopP: 1, RepetitionPenalty: 1}, nil, false, true); err != nil || got != 0 {
		t.Fatalf("greedy got=%d err=%v", got, err)
	}
	if !reflect.DeepEqual(input, original) {
		t.Fatal("mutated caller logits")
	}
	const workers = 8
	results := make([]uint32, workers)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = NewReferenceCPUSampler(42).Select(input, cfg, nil, false, true)
		}(i)
	}
	wg.Wait()
	for i := range results {
		if errs[i] != nil || results[i] != results[0] {
			t.Fatalf("concurrent request %d got=%d err=%v", i, results[i], errs[i])
		}
	}
}

func TestReferenceCPUSamplerReleasedPrefillStep(t *testing.T) {
	root := filepath.Join("testdata", "customvoice_0b6_ryan_hello")
	data, err := os.ReadFile(filepath.Join(root, "logits.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != CodecVocabSize*4 {
		t.Fatalf("logits bytes=%d", len(data))
	}
	logits := make([]float32, CodecVocabSize)
	for i := range logits {
		logits[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	data, err = os.ReadFile(filepath.Join(root, "released_first_sample.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Rows []struct {
			Label       string   `json:"label"`
			Seed        uint64   `json:"seed"`
			Temperature float64  `json:"temperature"`
			TopK        int      `json:"top_k"`
			TopP        float64  `json:"top_p"`
			Tokens      []uint32 `json:"tokens"`
		} `json:"rows"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Rows) != 12 {
		t.Fatalf("rows=%d", len(fixture.Rows))
	}
	original := append([]float32(nil), logits...)
	for _, c := range fixture.Rows {
		row := append([]float32(nil), logits...)
		if c.Label == "controlled_near_tie" {
			row[1221] = row[1995] - 0.25
		} else if c.Label != "released" {
			t.Fatalf("label=%q", c.Label)
		}
		sampler := NewReferenceCPUSampler(c.Seed)
		cfg := ReferenceSampleConfig{Temperature: c.Temperature, TopK: c.TopK, TopP: c.TopP, RepetitionPenalty: 1}
		for i, want := range c.Tokens {
			got, err := sampler.Select(row, cfg, nil, true, true)
			if err != nil || got != want {
				t.Fatalf("%s seed=%d draw=%d got=%d want=%d err=%v", c.Label, c.Seed, i, got, want, err)
			}
		}
	}
	if !reflect.DeepEqual(logits, original) {
		t.Fatal("mutated fixture logits")
	}
}
