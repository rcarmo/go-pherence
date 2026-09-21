package jevlike

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	backbone "github.com/rcarmo/go-pherence/model"
)

func TestDirectQwen3Reference(t *testing.T) {
	dir, refPath := os.Getenv("JEVLIKE_QWEN3_MODEL_DIR"), os.Getenv("JEVLIKE_DIRECT_REFERENCE")
	if dir == "" || refPath == "" {
		t.Skip("requires local direct-score reference")
	}
	var ref struct {
		Repository string `json:"repository"`
		Revision   string `json:"revision"`
		Template   string `json:"template_sha256"`
		Fixtures   []struct {
			Request         DirectChoiceRequest `json:"request"`
			Prompt          string              `json:"prompt"`
			Tokens          []int               `json:"tokens"`
			CandidateTokens []int               `json:"candidate_tokens"`
			Hidden          []float32           `json:"last_hidden"`
			Logits          []float32           `json:"logits"`
			Argmax          int                 `json:"argmax"`
		} `json:"fixtures"`
	}
	b, err := os.ReadFile(refPath)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	if !pinnedQwen3Reference(ref.Repository, ref.Revision) || filepath.Base(filepath.Clean(dir)) != ref.Revision || len(ref.Fixtures) < 5 {
		t.Fatal("unexpected direct fixture")
	}
	prompt, err := LoadQwen3ChoicePrompt(dir, 512)
	if err != nil {
		t.Fatal(err)
	}
	if prompt.TemplateSHA256 != ref.Template {
		t.Fatal("template mismatch")
	}
	encoder, err := backbone.NewFrozenGPUEncoder(dir, backbone.FrozenGPUOptions{MaxTokens: 512, BudgetBytes: 10 << 30, ReserveBytes: 1 << 30})
	if err != nil {
		t.Fatal(err)
	}
	defer nvidia.Shutdown() // test process owns CUDA; encoder itself must not
	defer encoder.Close()
	for index, f := range ref.Fixtures {
		text, ids, codes, err := prompt.Prepare(f.Request)
		if err != nil {
			t.Fatal(err)
		}
		if text != f.Prompt || !reflect.DeepEqual(ids, f.Tokens) || !reflect.DeepEqual(codes, f.CandidateTokens) {
			t.Fatalf("case %d prompt/token contract mismatch\nrender=%q\nreference=%q\ntokens=%v\nreference_tokens=%v\ncodes=%v reference_codes=%v", index, text, f.Prompt, ids, f.Tokens, codes, f.CandidateTokens)
		}
		start := time.Now()
		got, err := ScoreChoices(encoder, prompt, f.Request)
		if err != nil {
			t.Fatal(err)
		}
		var mx float64
		for i, v := range got.Logits {
			mx = math.Max(mx, math.Abs(float64(v)-float64(f.Logits[i])))
		}
		changed := got.SelectedID != f.Request.Candidates[f.Argmax].ID
		t.Logf("case=%d tokens=%d seconds=%.3f max_logit_delta=%g changed_decision=%v logits=%v", index, len(ids), time.Since(start).Seconds(), mx, changed, got.Logits)
		// Fixed before first native direct-score run, not fitted to outcomes.
		if mx > 0.005 {
			t.Errorf("case %d selected logits exceed 0.005 gate", index)
		}
		if changed {
			t.Errorf("case %d changed decision", index)
		}
		if index == 0 {
			profiled, timing, err := encoder.ProfileSelectedLogits(ids, codes)
			if err != nil || !reflect.DeepEqual(profiled, got.Logits) {
				t.Fatal("profiled logits changed", profiled, err)
			}
			if timing.PrefillSeconds <= 0 || timing.EmbeddingUploadSeconds <= 0 || timing.DownloadSeconds <= 0 || timing.ProjectionSeconds <= 0 || timing.UploadBytes != len(ids)*2560*4 || timing.DownloadBytes != 2560*4 {
				t.Fatal("invalid phase accounting", timing)
			}
		}
	}
}
