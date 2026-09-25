package mojev

import (
	"crypto/sha256"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/loader/weights"
)

func TestMoJevAcceleratedReleased(t *testing.T) {
	gpuEnabled, simdEnabled := os.Getenv("GO_PHERENCE_MOJEV_NVIDIA") == "1", os.Getenv("GO_PHERENCE_MOJEV_SIMD") == "1"
	if !gpuEnabled && !simdEnabled {
		t.Skip("opt-in released SIMD/PTX tests")
	}
	dir := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if dir == "" {
		t.Fatal("checkpoint required")
	}
	f := nativeFixture(t)
	for name, want := range map[string]string{"model.safetensors": f.WeightSHA, "config.json": f.ConfigSHA, "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523", "tokenizer_config.json": "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b"} {
		file, e := os.Open(filepath.Join(dir, name))
		if e != nil {
			t.Fatal(e)
		}
		h := sha256.New()
		_, e = io.Copy(h, file)
		file.Close()
		if e != nil || fmt.Sprintf("%x", h.Sum(nil)) != want {
			t.Fatal("asset hash mismatch", name, e)
		}
	}
	data, e := os.ReadFile(filepath.Join(dir, "config.json"))
	if e != nil {
		t.Fatal(e)
	}
	src, e := weights.OpenSafetensors(dir)
	if e != nil {
		t.Fatal(e)
	}
	defer src.Close()
	cpu, e := LoadTextScorer(src, data)
	if e != nil {
		t.Fatal(e)
	}
	src.Close()
	type executor interface {
		ScoreEncoded(EncodedRow) ([][]float32, error)
		ScoreText(TextRequest, *tokenizer.Tokenizer, int, int) (*TextDecision, error)
	}
	type backend struct {
		name   string
		scorer executor
		close  func() error
	}
	var backends []backend
	if simdEnabled {
		s, e := NewSIMDTextScorer(cpu, 256)
		if e != nil {
			t.Fatal(e)
		}
		backends = append(backends, backend{"simd", s, func() error { return nil }})
	}
	if gpuEnabled {
		g, e := NewNVIDIATextScorer(cpu, 256)
		if e != nil {
			t.Fatal(e)
		}
		defer g.Close()
		t.Logf("GPU resident bytes %d", g.ResidentBytes())
		backends = append(backends, backend{"ptx", g, g.Close})
	}
	for _, b := range backends {
		t.Run(b.name, func(t *testing.T) {
			outputs := map[string][][]float32{}
			for _, name := range []string{"base", "sibling_token", "other_question", "sibling_length", "question_length", "candidate_order", "question_order", "long_nodes"} {
				got, e := b.scorer.ScoreEncoded(f.Cases[name].Row)
				if e != nil {
					t.Fatal(e)
				}
				outputs[name] = got
				var diff float64
				for i := range got {
					for j, v := range got[i] {
						diff = math.Max(diff, math.Abs(float64(v-f.Cases[name].Logits[i][j])))
					}
				}
				t.Logf("%s max error %g", name, diff)
				if math.IsNaN(diff) || diff > 3e-4 {
					t.Fatal("parity failed", name, diff)
				}
			}
			// Compare actual encoder rows as well as head logits, without injected
			// model outputs. Keep thresholds fixed to the CPU reference gate.
			row := repairedTextRow()
			branch := TextBranch{IDs: append(append(append([]int{}, row.State...), row.Questions[0]...), row.Candidates[0][0]...), StateLen: len(row.State), QuestionLen: len(row.Questions[0])}
			var hidden []float32
			var err error
			switch scorer := b.scorer.(type) {
			case *SIMDTextScorer:
				hidden, err = scorer.cpu.encodeBranchWith(branch, scorer.branch)
				// Public Forward rows remain owned; Into failures are transactional.
				inputs := make([][]float32, len(branch.IDs))
				for i, id := range branch.IDs {
					inputs[i] = cpu.embedding[id*1024 : (id+1)*1024]
				}
				owned, e := scorer.branch.Forward(inputs, branch.StateLen, branch.QuestionLen, cpu.rope, cpu.eps)
				if e != nil {
					t.Fatal(e)
				}
				saved := append([]float32(nil), owned[0]...)
				dst := make([]float32, len(inputs)*1024)
				if e = scorer.branch.ForwardInto(dst, inputs, branch.StateLen, branch.QuestionLen, cpu.rope, cpu.eps); e != nil {
					t.Fatal(e)
				}
				if !reflect.DeepEqual(saved, owned[0]) || !reflect.DeepEqual(dst[:1024], saved) {
					t.Fatal("Into parity/ownership")
				}
				for i := range dst {
					dst[i] = 123
				}
				if e = scorer.branch.ForwardInto(dst, inputs, 0, branch.QuestionLen, cpu.rope, cpu.eps); e == nil {
					t.Fatal("invalid Into accepted")
				}
				for _, v := range dst {
					if v != 123 {
						t.Fatal("partial Into output")
					}
				}
				if a := testing.AllocsPerRun(1, func() {
					got, e := scorer.ScoreEncoded(repairedTextRow())
					if e != nil || got == nil {
						panic(e)
					}
				}); a > 140 {
					t.Fatalf("encoded SIMD allocation regression: %g > 140", a)
				}
			case *NVIDIATextScorer:
				scorer.mu.Lock()
				hidden, err = scorer.encodeBranch(branch)
				scorer.mu.Unlock()
			}
			if err != nil {
				t.Fatal(err)
			}
			var maxHidden float64
			for i, r := range f.Cases["base"].Hidden[0] {
				for j, v := range r {
					maxHidden = math.Max(maxHidden, math.Abs(float64(hidden[i*1024+j]-v)))
				}
			}
			t.Logf("hidden max error %g", maxHidden)
			if math.IsNaN(maxHidden) || maxHidden > 2e-3 {
				t.Fatal("hidden parity")
			}
			base := outputs["base"]
			for _, name := range []string{"sibling_token", "sibling_length", "question_length"} {
				if !reflect.DeepEqual(base[1], outputs[name][1]) {
					t.Fatal("question leak", name)
				}
			}
			for _, name := range []string{"sibling_token", "sibling_length"} {
				if base[0][0] != outputs[name][0][0] {
					t.Fatal("candidate leak", name)
				}
			}
			if !reflect.DeepEqual(base[0], outputs["other_question"][0]) {
				t.Fatal("question leak")
			}
			for field := range base {
				for c := range base[field] {
					if base[field][c] != outputs["candidate_order"][field][1-c] || base[field][c] != outputs["question_order"][1-field][c] {
						t.Fatal("order leak")
					}
				}
			}
			for repeat := 0; repeat < 5; repeat++ {
				got, e := b.scorer.ScoreEncoded(f.Cases["long_nodes"].Row)
				if e != nil || !reflect.DeepEqual(got, outputs["long_nodes"]) {
					t.Fatal("nondeterministic", repeat, e)
				}
			}
			bad := repairedTextRow()
			bad.Candidates[0][0] = []int{cpu.meta.VocabSize}
			if got, e := b.scorer.ScoreEncoded(bad); e == nil || got != nil {
				t.Fatal("bad token accepted")
			}
			bad = repairedTextRow()
			bad.State = make([]int, 256)
			if got, e := b.scorer.ScoreEncoded(bad); e == nil || got != nil {
				t.Fatal("capacity ignored")
			}
			var wg sync.WaitGroup
			for range 4 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					got, e := b.scorer.ScoreEncoded(repairedTextRow())
					if e != nil || !reflect.DeepEqual(got, base) {
						t.Error("concurrent/error recovery", e)
					}
				}()
			}
			wg.Wait()
			tok, e := tokenizer.LoadWithConfig(dir)
			if e != nil {
				t.Fatal(e)
			}
			req, e := DecodeTextRequest(strings.NewReader(`{"model":"mojev","state":"A pipe is leaking near electrical equipment.","questions":{"route":{"type":"choice","instructions":"Choose a team.","criteria":{"facilities":null,"accounts":null}},"priority":{"type":"choice","instructions":"Choose priority.","criteria":{"urgent":null,"routine":null}}}}`))
			if e != nil {
				t.Fatal(e)
			}
			answer, e := b.scorer.ScoreText(req, tok, 128, 128)
			if e != nil {
				t.Fatal(e)
			}
			req.Fields[0].Instructions = "A much longer unrelated first question."
			changed, e := b.scorer.ScoreText(req, tok, 128, 128)
			if e != nil || !reflect.DeepEqual(answer.Answers["priority"], changed.Answers["priority"]) {
				t.Fatal("text isolation", e)
			}
			if b.name == "ptx" {
				// Close races safely with scoring: either owned complete output or
				// a clean closed error, never partial output or a stale launch.
				var closeWG sync.WaitGroup
				closeWG.Add(2)
				go func() {
					defer closeWG.Done()
					got, err := b.scorer.ScoreEncoded(repairedTextRow())
					if err == nil && !reflect.DeepEqual(got, base) {
						t.Error("close race output")
					}
					if err != nil && got != nil {
						t.Error("close race partial output")
					}
				}()
				go func() {
					defer closeWG.Done()
					if err := b.close(); err != nil {
						t.Error(err)
					}
				}()
				closeWG.Wait()
				if err := b.close(); err != nil {
					t.Fatal(err)
				}
				if err := b.close(); err != nil {
					t.Fatal(err)
				}
				if got, e := b.scorer.ScoreEncoded(repairedTextRow()); e == nil || got != nil {
					t.Fatal("closed scorer usable")
				}
			}
		})
	}
}
func TestMoJevAcceleratedInvalid(t *testing.T) {
	for _, cap := range []int{-1, 0, 2, 513} {
		if s, e := NewNVIDIATextScorer(nil, cap); e == nil || s != nil {
			t.Fatal("invalid GPU accepted")
		}
		if s, e := NewSIMDTextScorer(nil, cap); e == nil || s != nil {
			t.Fatal("invalid SIMD accepted")
		}
	}
	var g *NVIDIATextScorer
	g.Close()
	if g.ResidentBytes() != 0 {
		t.Fatal("nil resident")
	}
	if out, e := g.ScoreEncoded(repairedTextRow()); e == nil || out != nil {
		t.Fatal("nil GPU")
	}
	var s *SIMDTextScorer
	if out, e := s.ScoreEncoded(repairedTextRow()); e == nil || out != nil {
		t.Fatal("nil SIMD")
	}
	if os.Getenv("GO_PHERENCE_DISABLE_NVIDIA") != "" && nvidia.Init() {
		t.Fatal("GPU unexpectedly enabled")
	}
}
