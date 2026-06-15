package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadParityFixtureValidation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "fixture.json")
	data := []byte(`{"prompt_tokens":[1,2],"draft_count":2,"max_tokens":3,"cycle":{"input_token":2,"drafted_tokens":[3,4],"verifier_tokens":[2,3,4],"verifier_output_tokens":[3,5,6],"accepted_prefix_len":1,"bonus_token":5,"output_tokens":[3,5]}}`)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GO_PHERENCE_GEMMA4_MAIN", "main")
	t.Setenv("GO_PHERENCE_GEMMA4_MTP_DRAFTER", "drafter")
	fx, err := loadParityFixture(path)
	if err != nil {
		t.Fatal(err)
	}
	if fx.MainModel != "main" || fx.Drafter != "drafter" || fx.DraftCount != 2 || fx.MaxTokens != 3 {
		t.Fatalf("fixture=%+v", fx)
	}

	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte(`{"main_model":"m","drafter":"d","prompt_tokens":[1],"draft_count":2,"max_tokens":2}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadParityFixture(bad); err == nil {
		t.Fatal("accepted max_tokens <= draft_count")
	}
}
