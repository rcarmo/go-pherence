package mojev

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/weights"
)

type longTextFixture struct {
	Schema    int    `json:"schema"`
	Policy    string `json:"policy"`
	Revision  string `json:"source_revision"`
	WeightSHA string `json:"weights_sha256"`
	ConfigSHA string `json:"config_sha256"`
	Cases     map[string]struct {
		Row       EncodedRow    `json:"row"`
		Logits    [][]float32   `json:"logits"`
		Hidden    [][][]float32 `json:"hidden"`
		Positions [][]int       `json:"hidden_positions"`
	} `json:"cases"`
}

func longFixture(t *testing.T) longTextFixture {
	t.Helper()
	for p, want := range map[string]string{"../../scripts/mojev_oracle_long_text.py": "989bdc1d4ce4cfe7329ff23aed3285b7e350f00c976ee44ddf6ffa3da00c775d", "testdata/native_long_text.json": "5f1a5c6e891eb2bbc5c8de0ee3a87c6a41f28914302b649d136db8f71cd5a0bd"} {
		b, e := os.ReadFile(p)
		if e != nil {
			t.Fatal(e)
		}
		if fmt.Sprintf("%x", sha256.Sum256(b)) != want {
			t.Fatal("fixture hash", p)
		}
	}
	b, e := os.ReadFile("testdata/native_long_text.json")
	if e != nil {
		t.Fatal(e)
	}
	var f longTextFixture
	if e = json.Unmarshal(b, &f); e != nil {
		t.Fatal(e)
	}
	if f.Schema != 1 || f.Revision != SourceRevision || f.Policy != "f32-branch-local-positions-full-tree-causal-linear" || len(f.Cases) != 5 {
		t.Fatal("fixture provenance")
	}
	return f
}
func TestNativeLongTextFixture(t *testing.T) {
	f := longFixture(t)
	for name, c := range f.Cases {
		if len(c.Row.State) != 128 || len(c.Row.Questions) != 1 || len(c.Row.Questions[0]) != 128 || len(c.Row.Candidates) != 1 || len(c.Row.Candidates[0]) != 2 || len(c.Logits) != 1 || len(c.Logits[0]) != 2 || len(c.Hidden) != 2 || len(c.Positions) != 2 {
			t.Fatal("fixture geometry", name)
		}
		for i, ids := range c.Row.Candidates[0] {
			if len(ids) < 1 || len(ids) > 256 || len(c.Hidden[i]) != 3 || !reflect.DeepEqual(c.Positions[i], []int{127, 255, 255 + len(ids)}) {
				t.Fatal("sample geometry", name)
			}
			for _, row := range c.Hidden[i] {
				if len(row) != 1024 {
					t.Fatal("hidden width")
				}
				for _, v := range row {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatal("nonfinite fixture")
					}
				}
			}
		}
	}
	base := f.Cases["max_path"]
	for _, name := range []string{"sibling_token", "sibling_length"} {
		if base.Logits[0][0] != f.Cases[name].Logits[0][0] || !reflect.DeepEqual(base.Hidden[0], f.Cases[name].Hidden[0]) {
			t.Fatal("oracle isolation")
		}
	}
	if !reflect.DeepEqual(f.Cases["candidate_order"].Logits[0], []float32{base.Logits[0][1], base.Logits[0][0]}) {
		t.Fatal("oracle permutation")
	}
}

func TestReleasedLongTextScorer(t *testing.T) {
	backend := os.Getenv("GO_PHERENCE_MOJEV_LONG_BACKEND")
	if backend == "" {
		t.Skip("set GO_PHERENCE_MOJEV_LONG_BACKEND=simd or nvidia and checkpoint path")
	}
	if backend != "simd" && backend != "nvidia" {
		t.Fatal("invalid backend")
	}
	f := longFixture(t)
	dir := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if dir == "" {
		t.Fatal("checkpoint required")
	}
	for name, want := range map[string]string{"config.json": f.ConfigSHA, "model.safetensors": f.WeightSHA} {
		file, e := os.Open(filepath.Join(dir, name))
		if e != nil {
			t.Fatal(e)
		}
		h := sha256.New()
		_, e = io.Copy(h, file)
		file.Close()
		if e != nil || fmt.Sprintf("%x", h.Sum(nil)) != want {
			t.Fatal("asset hash", name, e)
		}
	}
	config, e := os.ReadFile(filepath.Join(dir, "config.json"))
	if e != nil {
		t.Fatal(e)
	}
	src, e := weights.OpenSafetensors(dir)
	if e != nil {
		t.Fatal(e)
	}
	defer src.Close()
	cpu, e := LoadTextScorer(src, config)
	if e != nil {
		t.Fatal(e)
	}
	src.Close()
	var score func(EncodedRow) ([][]float32, error)
	var encode TextBranchEncoder
	var encodeTree func(EncodedRow) ([]float32, error)
	if backend == "simd" {
		s, e := NewSIMDTextScorer(cpu, 512)
		if e != nil {
			t.Fatal(e)
		}
		score = s.ScoreEncoded
		encodeTree = func(row EncodedRow) ([]float32, error) {
			s.mu.Lock()
			defer s.mu.Unlock()
			n := 0
			for _, node := range [][]int{row.State, row.Questions[0], row.Candidates[0][0], row.Candidates[0][1]} {
				for _, id := range node {
					s.rows[n] = cpu.embedding[id*1024 : (id+1)*1024]
					n++
				}
			}
			out := s.hidden[:n*1024]
			if err := s.branch.ForwardTreeInto(out, s.rows[:n], 128, 128, []int{384, 512}, cpu.rope, cpu.eps); err != nil {
				return nil, err
			}
			cpu.normaliseFinal(out)
			return out, nil
		}
		encode = func(b TextBranch) ([]float32, error) { s.mu.Lock(); defer s.mu.Unlock(); return s.encodeBranch(b) }
	} else {
		g, e := NewNVIDIATextScorer(cpu, 512)
		if e != nil {
			if g != nil {
				_ = g.Close()
			}
			t.Fatal(e)
		}
		defer func() {
			if e := g.Close(); e != nil {
				t.Error(e)
			}
		}()
		score = g.ScoreEncoded
		encodeTree = func(row EncodedRow) ([]float32, error) {
			g.mu.Lock()
			defer g.mu.Unlock()
			var ids []int
			for _, node := range [][]int{row.State, row.Questions[0], row.Candidates[0][0], row.Candidates[0][1]} {
				ids = append(ids, node...)
			}
			return g.encodeTree(TextBranch{IDs: ids, StateLen: 128, QuestionLen: 128}, []int{384, 512})
		}
		encode = func(b TextBranch) ([]float32, error) { g.mu.Lock(); defer g.mu.Unlock(); return g.encodeBranch(b) }
	}
	gotCases := map[string][][]float32{}
	for _, name := range []string{"max_path", "shared_tree", "sibling_token", "sibling_length", "candidate_order"} {
		c := f.Cases[name]
		got, e := score(c.Row)
		if e != nil {
			t.Fatal(e)
		}
		gotCases[name] = got
		if len(got) != 1 || len(got[0]) != 2 {
			t.Fatal("result shape")
		}
		var maxLogit, maxHidden float64
		for i, v := range got[0] {
			maxLogit = math.Max(maxLogit, math.Abs(float64(v-c.Logits[0][i])))
		}
		for i, candidate := range c.Row.Candidates[0] {
			ids := append(append(append([]int{}, c.Row.State...), c.Row.Questions[0]...), candidate...)
			hidden, e := encode(TextBranch{IDs: ids, StateLen: 128, QuestionLen: 128})
			if e != nil {
				t.Fatal(e)
			}
			for s, pos := range c.Positions[i] {
				for j, v := range c.Hidden[i][s] {
					maxHidden = math.Max(maxHidden, math.Abs(float64(hidden[pos*1024+j]-v)))
				}
			}
		}
		if name == "shared_tree" {
			hidden, err := encodeTree(c.Row)
			if err != nil {
				t.Fatal(err)
			}
			for i := range c.Hidden {
				for sample, pos := range c.Positions[i] {
					if pos >= 256 {
						pos += i * 128
					}
					for j, v := range c.Hidden[i][sample] {
						maxHidden = math.Max(maxHidden, math.Abs(float64(hidden[pos*1024+j]-v)))
					}
				}
			}
		}
		t.Logf("%s %s max_logits=%g max_hidden=%g", backend, name, maxLogit, maxHidden)
		if math.IsNaN(maxLogit) || math.IsNaN(maxHidden) || maxLogit > 3e-4 || maxHidden > 2e-3 {
			t.Fatal("long reference parity", name, maxLogit, maxHidden)
		}
	}
	for _, name := range []string{"sibling_token", "sibling_length"} {
		if gotCases[name][0][0] != gotCases["max_path"][0][0] {
			t.Fatal("long sibling leak")
		}
	}
	base := gotCases["max_path"][0]
	if !reflect.DeepEqual(gotCases["candidate_order"][0], []float32{base[1], base[0]}) {
		t.Fatal("long order leak")
	}
}
