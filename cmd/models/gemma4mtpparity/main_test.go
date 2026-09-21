package main

import (
	"encoding/json"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/model"
)

func TestDefaultParityFixtureUsesCheckpointRoot(t *testing.T) {
	fxPath := filepath.Join("..", "..", "..", "model", "testdata", "gemma4-mtp-llamacpp-fixture.json")
	fx, err := loadParityFixture(fxPath)
	if err != nil {
		t.Fatal(err)
	}
	for name, path := range map[string]string{"main_model": fx.MainModel, "drafter": fx.Drafter} {
		if !strings.HasPrefix(path, "checkpoints/") {
			t.Errorf("%s=%q; default fixture must use the repository checkpoint root", name, path)
		}
	}
}

func TestMissingAssetsRequireExplicitFixtureOnly(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "model", "testdata", "gemma4-mtp-llamacpp-fixture.json")
	fx, err := loadParityFixture(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	fx.MainModel = filepath.Join(t.TempDir(), "missing-main.gguf")
	fx.Drafter = filepath.Join(t.TempDir(), "missing-drafter.gguf")
	if _, err := runParity(fixturePath, fx); err == nil {
		t.Fatal("missing models reported parity")
	}
	report, err := checkFixture(fixturePath, fx)
	if err != nil {
		t.Fatal(err)
	}
	if report.Matched || report.Executed || !report.FixtureValidated || len(report.Got.OutputTokens) != 0 {
		b, _ := json.MarshalIndent(report, "", "  ")
		t.Fatalf("trimmed fallback report mismatch:\n%s", b)
	}
}

func TestRunParityDefaultFixtureRestoresForceOnTheFly(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "model", "testdata", "gemma4-mtp-llamacpp-fixture.json")
	fx, err := loadParityFixture(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	old := model.ForceOnTheFly
	model.ForceOnTheFly = false
	defer func() { model.ForceOnTheFly = old }()
	fx.MainModel = filepath.Join(t.TempDir(), "missing-main.gguf")
	if _, err := runParity(fixturePath, fx); err == nil {
		t.Fatal("missing model accepted")
	}
	if model.ForceOnTheFly {
		t.Fatal("runParity leaked ForceOnTheFly=true")
	}
}

func TestMissingOrInvalidLogitProbesCannotPass(t *testing.T) {
	for _, tc := range []struct {
		got  [][]float32
		want []map[string]float64
	}{
		{nil, []map[string]float64{{"0": 1}}},
		{[][]float32{{1}}, []map[string]float64{{"2": 1}}},
		{[][]float32{{1}}, []map[string]float64{{"bad": 1}}},
		{[][]float32{{float32(math.NaN())}}, []map[string]float64{{"0": 1}}},
	} {
		d := selectedLogitDeltas("test", tc.got, tc.want, .001)
		if len(selectedLogitMismatches(d)) == 0 {
			t.Fatal("invalid probes passed", tc)
		}
		if _, err := json.Marshal(d); err != nil {
			t.Fatal("report not JSON-safe", err)
		}
	}
	valid := selectedLogitDeltas("test", [][]float32{{1}}, []map[string]float64{{"0": 1}}, .001)
	if len(valid) != 1 || len(selectedLogitMismatches(valid)) != 0 {
		t.Fatal(valid)
	}
}
