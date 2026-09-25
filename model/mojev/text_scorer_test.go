package mojev

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/rcarmo/go-pherence/loader/weights"
)

type nativeTextCase struct {
	Row    EncodedRow    `json:"row"`
	Logits [][]float32   `json:"logits"`
	Hidden [][][]float32 `json:"hidden"`
}
type nativeTextFixture struct {
	Schema    int                       `json:"schema"`
	Policy    string                    `json:"policy"`
	Revision  string                    `json:"source_revision"`
	ConfigSHA string                    `json:"config_sha256"`
	WeightSHA string                    `json:"weights_sha256"`
	Cases     map[string]nativeTextCase `json:"cases"`
}

func nativeFixture(t *testing.T) nativeTextFixture {
	t.Helper()
	for _, p := range []struct{ path, sha string }{
		{"../../scripts/mojev_oracle_native_text.py", "8b7c0f07b24253921af3a9ba4f84013f152401303c05c91b1d9dd3ca894dcb27"},
		{"../../scripts/mojev_oracle_repaired_text_isolation.py", "7fe9998def69d1b14fede716d750fc527e84178ef5eeacaa1d3691edbf45ed30"},
		{"testdata/native_text.json", "80b2d914a1dd1ac0a678f491b0b9f869d3fc27499dfe8d477926f9cb44178e1a"},
	} {
		b, e := os.ReadFile(p.path)
		if e != nil {
			t.Fatal(e)
		}
		if fmt.Sprintf("%x", sha256.Sum256(b)) != p.sha {
			t.Fatalf("hash mismatch %s", p.path)
		}
	}
	data, err := os.ReadFile("testdata/native_text.json")
	if err != nil {
		t.Fatal(err)
	}
	var f nativeTextFixture
	if err = json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || f.Policy != "f32-branch-local-positions-full-tree-causal-linear" || f.Revision != SourceRevision || len(f.Cases) != 8 {
		t.Fatal("fixture provenance")
	}
	return f
}
func TestNativeTextFixture(t *testing.T) {
	f := nativeFixture(t)
	base := f.Cases["base"].Logits
	if len(base) != 2 || len(f.Cases["base"].Hidden) != 4 {
		t.Fatal("fixture geometry")
	}
	for _, name := range []string{"sibling_token", "sibling_length", "question_length"} {
		if !reflect.DeepEqual(base[1], f.Cases[name].Logits[1]) {
			t.Fatalf("reference leak: %s", name)
		}
	}
	if !reflect.DeepEqual(base[0], f.Cases["other_question"].Logits[0]) {
		t.Fatal("reference question leak")
	}
}
func TestReleasedNativeTextScorer(t *testing.T) {
	f := nativeFixture(t)
	dir := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if dir == "" {
		t.Skip("set GO_PHERENCE_MOJEV_CHECKPOINT_DIR to opt in")
	}
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(data)) != f.ConfigSHA {
		t.Fatal("config hash")
	}
	file, err := os.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.New()
	_, err = io.Copy(hash, file)
	file.Close()
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", hash.Sum(nil)) != f.WeightSHA {
		t.Fatal("weight hash")
	}
	src, err := weights.OpenSafetensors(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	model, err := LoadTextScorer(src, data)
	if err != nil {
		t.Fatal(err)
	}
	if err = src.Close(); err != nil {
		t.Fatal(err)
	}
	outputs := map[string][][]float32{}
	for _, name := range []string{"base", "sibling_token", "other_question", "sibling_length", "question_length", "candidate_order", "question_order", "long_nodes"} {
		c := f.Cases[name]
		got, err := model.ScoreEncoded(c.Row)
		if err != nil {
			t.Fatal(err)
		}
		outputs[name] = got
		var maxDiff float64
		for i := range got {
			for j, v := range got[i] {
				d := math.Abs(float64(v - c.Logits[i][j]))
				maxDiff = math.Max(maxDiff, d)
				if math.IsNaN(float64(v)) || d > 3e-4 {
					t.Errorf("%s %d/%d got %.9g want %.9g diff %.6g", name, i, j, v, c.Logits[i][j], d)
				}
			}
		}
		t.Logf("%s maximum logit error %.6g", name, maxDiff)
	}
	base := outputs["base"]
	for _, name := range []string{"sibling_token", "sibling_length", "question_length"} {
		if !reflect.DeepEqual(base[1], outputs[name][1]) {
			t.Errorf("NATIVE leak: %s", name)
		}
	}
	if !reflect.DeepEqual(base[0], outputs["other_question"][0]) {
		t.Error("native question leak")
	}
	for i := range base {
		for j := range base[i] {
			if base[i][j] != outputs["candidate_order"][i][1-j] || base[i][j] != outputs["question_order"][1-i][j] {
				t.Fatal("native ordering leak")
			}
		}
	}
	b := f.Cases["base"].Row
	var maxHidden float64
	for field, question := range b.Questions {
		for option, candidate := range b.Candidates[field] {
			branch := TextBranch{IDs: append(append(append([]int{}, b.State...), question...), candidate...), StateLen: len(b.State), QuestionLen: len(question)}
			hidden, err := model.encodeBranch(branch)
			if err != nil {
				t.Fatal(err)
			}
			for i, row := range f.Cases["base"].Hidden[field*2+option] {
				for j, v := range row {
					d := math.Abs(float64(hidden[i*1024+j] - v))
					maxHidden = math.Max(maxHidden, d)
				}
			}
		}
	}
	t.Logf("all base branches max hidden error %.6g", maxHidden)
	if maxHidden > 2e-3 {
		t.Errorf("hidden drift %.6g", maxHidden)
	}
	bad := repairedTextRow()
	bad.Candidates[0][0][0] = model.meta.VocabSize
	if out, err := model.ScoreEncoded(bad); err == nil || out != nil {
		t.Fatal("invalid token accepted")
	}
	again, err := model.ScoreEncoded(b)
	if err != nil || !reflect.DeepEqual(again, base) {
		t.Fatal("request state retained")
	}
	// Two concurrent requests share only immutable loaded tensors.
	var wg sync.WaitGroup
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, e := model.ScoreEncoded(b)
			if e != nil || !reflect.DeepEqual(got, base) {
				t.Errorf("concurrent result %v %v", got, e)
			}
		}()
	}
	wg.Wait()

	for name, want := range map[string]string{"tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523", "tokenizer_config.json": "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b"} {
		b, e := os.ReadFile(filepath.Join(dir, name))
		if e != nil || fmt.Sprintf("%x", sha256.Sum256(b)) != want {
			t.Fatalf("tokenizer hash %s: %v", name, e)
		}
	}
	tok, err := tokenizer.LoadWithConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	req, err := DecodeTextRequest(strings.NewReader(`{"model":"mojev","state":"The object is red.","questions":{"color":{"type":"choice","instructions":"Color?","criteria":{"red":null,"blue":null}},"size":{"type":"choice","instructions":"Size?","criteria":{"large":null,"small":null}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	decision, err := model.ScoreText(req, tok, 128, 128)
	if err != nil {
		t.Fatal(err)
	}
	if len(decision.Answers) != 2 || decision.Usage.InputTokens <= 0 || decision.Usage.OutputTokens != 0 {
		t.Fatal("invalid real decision", decision)
	}
	req.Fields[0].Instructions = "A much longer unrelated question with different words."
	changed, err := model.ScoreText(req, tok, 128, 128)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decision.Answers["size"], changed.Answers["size"]) {
		t.Fatal("public text cross-question leak")
	}
	req.State = "<|im_start|>bad"
	if out, err := model.ScoreText(req, tok, 128, 128); err == nil || out != nil {
		t.Fatal("control token accepted")
	}
}
