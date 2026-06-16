package main

import (
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/model"
)

func TestRunParityDefaultFixtureRestoresForceOnTheFly(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "model", "testdata", "gemma4-mtp-llamacpp-fixture.json")
	fx, err := loadParityFixture(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	old := model.ForceOnTheFly
	model.ForceOnTheFly = false
	defer func() { model.ForceOnTheFly = old }()
	report, err := runParity(fixturePath, fx)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Matched {
		t.Fatalf("default fixture did not match: %+v", report)
	}
	if model.ForceOnTheFly {
		t.Fatal("runParity leaked ForceOnTheFly=true")
	}
}
