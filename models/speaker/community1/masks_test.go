package community1

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"sync"
	"testing"
)

type maskFixture struct {
	Name               string
	Config             EmbeddingMaskConfig
	Segmentations      []*float32
	Masks              []float32
	MinimumCleanFrames int   `json:"minimum_clean_frames"`
	SelectedFrames     []int `json:"selected_frames"`
}
type filterFixture struct {
	Name                      string
	Config                    ClusteringFilterConfig
	Segmentations, Embeddings []*float32
	Filtered                  []float32
	ChunkIndices              []int `json:"chunk_indices"`
	SpeakerIndices            []int `json:"speaker_indices"`
}

func loadMaskFixtures(t *testing.T) ([]maskFixture, []filterFixture) {
	t.Helper()
	data, err := os.ReadFile("testdata/masks-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Schema    int
		Reference struct {
			Diarization string `json:"diarization_sha256"`
			Clustering  string `json:"clustering_sha256"`
		}
		Selections []maskFixture
		Filters    []filterFixture
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || f.Reference.Diarization != "cbb358abedef5042fcc71bb970b12a2936a16686be4bda15691a224f172656fd" || f.Reference.Clustering != "6031fb7c21277a7e9901ef2cdaed7d5cd69f7ef45508dc4b45e82ce0da3c8fba" || len(f.Selections) != 30 || len(f.Filters) != 25 {
		t.Fatal("mask oracle contract changed")
	}
	return f.Selections, f.Filters
}
func maskValues(values []*float32) []float32 {
	result := make([]float32, len(values))
	for i, value := range values {
		if value == nil {
			result[i] = float32(math.NaN())
		} else {
			result[i] = *value
		}
	}
	return result
}
func sameMaskBits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if math.Float32bits(v) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

func TestEmbeddingMasksPinnedOracle(t *testing.T) {
	cases, _ := loadMaskFixtures(t)
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			input := maskValues(c.Segmentations)
			before := append([]float32(nil), input...)
			result, err := SelectEmbeddingMasks(context.Background(), input, c.Config)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(result.Masks, c.Masks) || !reflect.DeepEqual(result.SelectedFrames, c.SelectedFrames) || result.MinimumCleanFrames != c.MinimumCleanFrames {
				t.Fatal("selection mismatch", result, c)
			}
			if !sameMaskBits(input, before) {
				t.Fatal("modified source masks")
			}
			for i := range input {
				input[i] = 7
			}
			if !reflect.DeepEqual(result.Masks, c.Masks) {
				t.Fatal("retained source alias")
			}
		})
	}
	// Explicit strict threshold: at two clean frames fallback to four total;
	// at three clean frames discard one overlapping frame.
	for _, c := range cases {
		if c.Name == "strict_equal_fallback_True" || c.Name == "strict_above_clean_True" {
			result, err := SelectEmbeddingMasks(context.Background(), maskValues(c.Segmentations), c.Config)
			if err != nil {
				t.Fatal(err)
			}
			want := c.Name == "strict_above_clean_True"
			if result.UsedOverlapExcluded[0] != want {
				t.Fatal("strict threshold changed")
			}
		}
	}
}

func TestClusteringFilterPinnedOracle(t *testing.T) {
	_, cases := loadMaskFixtures(t)
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			embeddings, seg := maskValues(c.Embeddings), maskValues(c.Segmentations)
			beforeE, beforeS := append([]float32(nil), embeddings...), append([]float32(nil), seg...)
			out, err := FilterClusteringEmbeddings(context.Background(), embeddings, seg, c.Config)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(out.Embeddings, c.Filtered) || !reflect.DeepEqual(out.ChunkIndices, c.ChunkIndices) || !reflect.DeepEqual(out.SpeakerIndices, c.SpeakerIndices) {
				t.Fatal("filter mismatch", out, c)
			}
			if !sameMaskBits(embeddings, beforeE) || !sameMaskBits(seg, beforeS) {
				t.Fatal("modified source")
			}
			for i := range embeddings {
				embeddings[i] = 99
			}
			if !reflect.DeepEqual(out.Embeddings, c.Filtered) {
				t.Fatal("unowned filtered embeddings")
			}
		})
	}
}

func TestEmbeddingMaskRejectionAndBounds(t *testing.T) {
	ctx := context.Background()
	cfg := EmbeddingMaskConfig{Frames: 2, Speakers: 2, WindowSamples: 160000, MinimumSamples: 400, ExcludeOverlap: true}
	input := []float32{1, 0, 0, 1}
	for _, kind := range []string{"zero_frames", "huge_frames", "zero_speakers", "huge_speakers", "short", "long", "soft", "negative", "inf", "samples_zero", "samples_huge", "minimum_zero", "minimum_huge"} {
		c := cfg
		values := append([]float32(nil), input...)
		switch kind {
		case "zero_frames":
			c.Frames = 0
		case "huge_frames":
			c.Frames = int(^uint(0) >> 1)
		case "zero_speakers":
			c.Speakers = 0
		case "huge_speakers":
			c.Speakers = 9
		case "short":
			values = values[:3]
		case "long":
			values = append(values, 0)
		case "soft":
			values[0] = .5
		case "negative":
			values[0] = -1
		case "inf":
			values[0] = float32(math.Inf(1))
		case "samples_zero":
			c.WindowSamples = 0
		case "samples_huge":
			c.WindowSamples = 480001
		case "minimum_zero":
			c.MinimumSamples = 0
		case "minimum_huge":
			c.MinimumSamples = 480001
		}
		if out, err := SelectEmbeddingMasks(ctx, values, c); err == nil || out != nil {
			t.Fatal("accepted", kind)
		}
	}
	// Maximum mask shape and ratio: bounded product, no big PCM allocation.
	maxCfg := EmbeddingMaskConfig{4096, 8, 1, 480000, true}
	out, err := SelectEmbeddingMasks(ctx, make([]float32, 4096*8), maxCfg)
	if err != nil || out.MinimumCleanFrames != 1966080000 {
		t.Fatal("maximum geometry", out, err)
	}
	f := ClusteringFilterConfig{1, 2, 2, 3, .2}
	emb := make([]float32, 6)
	for _, kind := range []string{"chunks", "frames", "speakers", "dimension", "cap", "length_e", "length_s", "nan_ratio", "inf_ratio", "negative_ratio", "large_ratio", "inf_embedding", "inf_seg", "soft_seg"} {
		c := f
		e := append([]float32(nil), emb...)
		s := append([]float32(nil), input...)
		switch kind {
		case "chunks":
			c.Chunks = int(^uint(0) >> 1)
		case "frames":
			c.Frames = 0
		case "speakers":
			c.Speakers = 9
		case "dimension":
			c.Dimension = 513
		case "cap":
			c.Chunks = 4096
			c.Frames = 4096
		case "length_e":
			e = e[:5]
		case "length_s":
			s = s[:3]
		case "nan_ratio":
			c.MinActiveRatio = math.NaN()
		case "inf_ratio":
			c.MinActiveRatio = math.Inf(1)
		case "negative_ratio":
			c.MinActiveRatio = -.1
		case "large_ratio":
			c.MinActiveRatio = 1.1
		case "inf_embedding":
			e[0] = float32(math.Inf(-1))
		case "inf_seg":
			s[0] = float32(math.Inf(-1))
		case "soft_seg":
			s[0] = .1
		}
		if out, err := FilterClusteringEmbeddings(ctx, e, s, c); err == nil || out != nil {
			t.Fatal("accepted filter", kind)
		}
	}
}

func TestEmbeddingMasksCancellationAndConcurrency(t *testing.T) {
	cfg := EmbeddingMaskConfig{257, 3, 160000, 400, true}
	seg := make([]float32, cfg.Frames*cfg.Speakers)
	for i := 0; i < cfg.Frames; i++ {
		seg[i*3+i%3] = 1
	}
	f := ClusteringFilterConfig{1, cfg.Frames, cfg.Speakers, 5, .2}
	emb := make([]float32, 15)
	runSelection := func(ctx context.Context) error {
		out, err := SelectEmbeddingMasks(ctx, seg, cfg)
		if err != nil && out != nil {
			t.Fatal("partial selection")
		}
		return err
	}
	runFilter := func(ctx context.Context) error {
		out, err := FilterClusteringEmbeddings(ctx, emb, seg, f)
		if err != nil && out != nil {
			t.Fatal("partial admission")
		}
		return err
	}
	for i, run := range []func(context.Context) error{runSelection, runFilter} {
		ctx := newPowersetContext(0)
		err := run(ctx)
		ctx.cancel()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("mask stage %d cancellation checkpoints %d", i, ctx.calls)
		for at := 1; at <= ctx.calls; at++ {
			cancel := newPowersetContext(at)
			err := run(cancel)
			cancel.cancel()
			if !errors.Is(err, context.Canceled) {
				t.Fatal("cancellation", i, at, err)
			}
		}
	}
	want, err := SelectEmbeddingMasks(context.Background(), seg, cfg)
	if err != nil {
		t.Fatal(err)
	}
	wantFilter, err := FilterClusteringEmbeddings(context.Background(), emb, seg, f)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := SelectEmbeddingMasks(context.Background(), seg, cfg)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Error("concurrent selection", err)
			}
			filter, err := FilterClusteringEmbeddings(context.Background(), emb, seg, f)
			if err != nil || !reflect.DeepEqual(filter, wantFilter) {
				t.Error("concurrent filter", err)
			}
		}()
	}
	wg.Wait()
}

func TestEmptyMaskBiasExcludedFromClustering(t *testing.T) {
	// Raw WeSpeaker projection remains unchanged: finite bias on empty support.
	// The admission API must not treat that as speech at the pinned default ratio.
	fixture := loadResNetFixtures(t)[0]
	model, err := NewWeSpeakerResNet34(context.Background(), fixture.Config, fixture.Weights)
	if err != nil {
		t.Fatal(err)
	}
	seg := make([]float32, 5*2)
	selected, err := SelectEmbeddingMasks(context.Background(), seg, EmbeddingMaskConfig{5, 2, 160000, 400, true})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := model.Forward(context.Background(), fixture.Input, fixture.Frames, selected.Masks, 2, 5, WeSpeakerBlockSIMD)
	if err != nil {
		t.Fatal(err)
	}
	for s := 0; s < 2; s++ {
		if raw.NonzeroFrames[s] != 0 || !reflect.DeepEqual(raw.Embeddings[s*fixture.Config.EmbedDim:(s+1)*fixture.Config.EmbedDim], fixture.Weights.Projection.Bias) {
			t.Fatal("empty raw projection changed")
		}
	}
	filtered, err := FilterClusteringEmbeddings(context.Background(), raw.Embeddings, seg, ClusteringFilterConfig{1, 5, 2, fixture.Config.EmbedDim, .2})
	if err != nil || len(filtered.Embeddings) != 0 {
		t.Fatal("admitted empty bias", err, filtered)
	}
}
