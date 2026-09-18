// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: MIT

package jevlike

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Derived from upstream MIT-licensed Jevlike DoomScorerV2._forward parity data.
type visionActionReferenceFixture struct {
	Schema            int                    `json:"schema"`
	AbsoluteTolerance float64                `json:"absolute_tolerance"`
	UpstreamSource    string                 `json:"upstream_source"`
	TorchVersion      string                 `json:"torch_version"`
	Config            visionActionConfigJSON `json:"config"`
	Positions         []float32              `json:"positions"`
	OptionEmbedding   []float32              `json:"option_embedding"`
	Heads             []AttentionHead        `json:"heads"`
	ValueNormWeight   []float32              `json:"value_norm_weight"`
	ValueNormBias     []float32              `json:"value_norm_bias"`
	ValueWeight       []float32              `json:"value_weight"`
	ValueBias         float32                `json:"value_bias"`
	RawContext        [][][]float32          `json:"raw_context"`
	OptionIDs         [][]int                `json:"option_ids"`
	Forward           visionActionForward    `json:"forward"`
	Trace             VisionActionTrace      `json:"trace"`
}

type visionActionConfigJSON struct {
	Width        int `json:"width"`
	Rank         int `json:"rank"`
	Actions      int `json:"actions"`
	Reads        int `json:"reads"`
	PatchRows    int `json:"patch_rows"`
	PatchColumns int `json:"patch_columns"`
}

func (c visionActionConfigJSON) scorerConfig() VisionActionConfig {
	return VisionActionConfig{
		Width:        c.Width,
		Rank:         c.Rank,
		Actions:      c.Actions,
		Reads:        c.Reads,
		PatchRows:    c.PatchRows,
		PatchColumns: c.PatchColumns,
	}
}

type visionActionForward struct {
	Logits [][]float32 `json:"logits"`
	Values []float32   `json:"values"`
}

func TestVisionActionPyTorchParity(t *testing.T) {
	fixture := loadVisionActionReferenceFixture(t)
	if fixture.Schema != 1 {
		t.Fatalf("fixture schema=%d want 1", fixture.Schema)
	}
	if fixture.AbsoluteTolerance <= 0 {
		t.Fatalf("fixture absolute_tolerance=%g must be positive", fixture.AbsoluteTolerance)
	}

	scorer, err := NewVisionActionScorer(fixture.Config.scorerConfig())
	if err != nil {
		t.Fatalf("NewVisionActionScorer() error = %v", err)
	}
	scorer.Positions = append([]float32(nil), fixture.Positions...)
	scorer.OptionEmbedding = append([]float32(nil), fixture.OptionEmbedding...)
	if len(fixture.Heads) != len(scorer.Heads) {
		t.Fatalf("fixture heads=%d want=%d", len(fixture.Heads), len(scorer.Heads))
	}
	for i := range fixture.Heads {
		scorer.Heads[i] = fixture.Heads[i]
	}
	scorer.ValueNormWeight = append([]float32(nil), fixture.ValueNormWeight...)
	scorer.ValueNormBias = append([]float32(nil), fixture.ValueNormBias...)
	scorer.ValueWeight = append([]float32(nil), fixture.ValueWeight...)
	scorer.ValueBias = fixture.ValueBias
	if err := scorer.Validate(); err != nil {
		t.Fatalf("scorer.Validate() error = %v", err)
	}

	selection := OptionSelection{PerBatch: fixture.OptionIDs}
	gotLogits, gotValues, err := scorer.Forward(fixture.RawContext, selection)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	compareFloat2D(t, "forward.logits", gotLogits, fixture.Forward.Logits, fixture.AbsoluteTolerance)
	compareFloat1D(t, "forward.values", gotValues, fixture.Forward.Values, fixture.AbsoluteTolerance)

	traceLogits, traceValues, trace, err := scorer.ForwardTrace(fixture.RawContext, selection)
	if err != nil {
		t.Fatalf("ForwardTrace() error = %v", err)
	}
	compareFloat2D(t, "trace.forward_logits", traceLogits, fixture.Forward.Logits, fixture.AbsoluteTolerance)
	compareFloat1D(t, "trace.forward_values", traceValues, fixture.Forward.Values, fixture.AbsoluteTolerance)
	compareFloat3D(t, "trace.query_matrix", trace.QueryMatrix, fixture.Trace.QueryMatrix, fixture.AbsoluteTolerance)
	compareFloat3D(t, "trace.key_matrix", trace.KeyMatrix, fixture.Trace.KeyMatrix, fixture.AbsoluteTolerance)
	compareFloat3D(t, "trace.value_matrix", trace.ValueMatrix, fixture.Trace.ValueMatrix, fixture.AbsoluteTolerance)
	compareFloat3D(t, "trace.attention_map", trace.AttentionMap, fixture.Trace.AttentionMap, fixture.AbsoluteTolerance)
	compareFloat2D(t, "trace.logits_matrix", trace.LogitsMatrix, fixture.Trace.LogitsMatrix, fixture.AbsoluteTolerance)
	compareFloat2D(t, "trace.probabilities", trace.Probabilities, fixture.Trace.Probabilities, fixture.AbsoluteTolerance)
	if len(fixture.Trace.Entropy) != 0 {
		compareFloat2D(t, "trace.entropy", trace.Entropy, fixture.Trace.Entropy, fixture.AbsoluteTolerance)
	}
}

func loadVisionActionReferenceFixture(t *testing.T) visionActionReferenceFixture {
	t.Helper()
	path := filepath.Join("testdata", "vision_action_reference.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", path, err)
	}
	var fixture visionActionReferenceFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatalf("json.Unmarshal(%q) error = %v", path, err)
	}
	return fixture
}

func compareFloat1D(t *testing.T, label string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len=%d want=%d", label, len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tolerance {
			t.Fatalf("%s[%d] got=%g want=%g diff=%g tol=%g", label, i, got[i], want[i], diff, tolerance)
		}
	}
}
