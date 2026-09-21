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
	loadertokenizer "github.com/rcarmo/go-pherence/loader/tokenizer"
	backbone "github.com/rcarmo/go-pherence/model"
)

func pinnedQwen3Reference(repository, revision string) bool {
	return repository == "Qwen/Qwen3-4B-Base" && revision == "906bfd4b4dc7f14ee4320094d8b41684abff8539" ||
		repository == "Qwen/Qwen3-4B" && revision == "1cfa9a7208912126459214e8b04321603b3df60c"
}

// Full local Qwen3 reference gate: ordinary tests never download/load weights.
func TestQwen3FrozenReference(t *testing.T) {
	dir, fixture := os.Getenv("JEVLIKE_QWEN3_MODEL_DIR"), os.Getenv("JEVLIKE_QWEN3_REFERENCE")
	if dir == "" || fixture == "" {
		t.Skip("set JEVLIKE_QWEN3_MODEL_DIR and JEVLIKE_QWEN3_REFERENCE")
	}
	var ref struct {
		Version    int    `json:"version"`
		Repository string `json:"repository"`
		Revision   string `json:"revision"`
		Width      int    `json:"width"`
		Contract   string `json:"contract"`
		Fixtures   []struct {
			Text   string      `json:"text"`
			Tokens []int       `json:"tokens"`
			Hidden [][]float32 `json:"hidden"`
		} `json:"fixtures"`
	}
	data, err := os.ReadFile(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.Version != 1 || (!pinnedQwen3Reference(ref.Repository, ref.Revision) || filepath.Base(filepath.Clean(dir)) != ref.Revision) || ref.Width != 2560 || ref.Contract != "causal/final-rmsnorm/all-token-rows/no-bos-no-eos/no-logits" || len(ref.Fixtures) < 5 {
		t.Fatal("unexpected reference contract")
	}
	started := time.Now()
	var encode func([]int) ([][]float32, error)
	var gpu *backbone.FrozenGPUEncoder
	if os.Getenv("JEVLIKE_QWEN3_BACKEND") == "gpu" {
		e, err := backbone.NewFrozenGPUEncoder(dir, backbone.FrozenGPUOptions{MaxTokens: 512, BudgetBytes: 10 << 30, ReserveBytes: 1 << 30})
		if err != nil {
			t.Fatal(err)
		}
		gpu = e
		t.Cleanup(e.Close)
		free, _ := nvidia.MemInfo()
		t.Logf("GPU stats: %+v actual_free_after_load=%d", e.Stats(), free)
		encode = e.EncodeTokenHiddenStates
	} else {
		m, err := backbone.LoadLlama(dir)
		if err != nil {
			t.Fatal(err)
		}
		encode = m.EncodeTokenHiddenStates
	}
	t.Logf("model_load_seconds=%.3f", time.Since(started).Seconds())
	tok, err := loadertokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	var maxAbs, sumSq float64
	count := 0
	for index, f := range ref.Fixtures {
		ids := tok.Encode(f.Text)
		if !reflect.DeepEqual(ids, f.Tokens) {
			t.Fatalf("case %d token mismatch: %v vs %v", index, ids, f.Tokens)
		}
		begun := time.Now()
		got, err := encode(ids)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(f.Hidden) || len(got) != len(ids) {
			t.Fatal("hidden row count mismatch")
		}
		var caseMax, caseSq float64
		worstRow, worstCol := 0, 0
		for row := range got {
			if len(got[row]) != ref.Width || len(f.Hidden[row]) != ref.Width {
				t.Fatal("width mismatch")
			}
			for col, v := range got[row] {
				d := math.Abs(float64(v) - float64(f.Hidden[row][col]))
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatal("nonfinite hidden value")
				}
				if d > caseMax {
					caseMax = d
					worstRow, worstCol = row, col
				}
				caseSq += d * d
				count++
			}
		}
		maxAbs = math.Max(maxAbs, caseMax)
		sumSq += caseSq
		rms := math.Sqrt(caseSq / float64(len(ids)*ref.Width))
		t.Logf("case=%d tokens=%d seconds=%.3f max_abs=%g rms=%g worst=[%d,%d] got=%g want=%g", index, len(ids), time.Since(begun).Seconds(), caseMax, rms, worstRow, worstCol, got[worstRow][worstCol], f.Hidden[worstRow][worstCol])
		// Fixed before running the fixture: CPU F32 should not need a
		// quantisation-scale tolerance. Fail rather than tune to observations.
		if caseMax > 0.005 || rms > 0.0002 {
			t.Errorf("case %d exceeded predeclared F32 hidden tolerance", index)
		}
	}
	t.Logf("max_abs=%g rms=%g elements=%d", maxAbs, math.Sqrt(sumSq/float64(count)), count)
	if gpu != nil {
		// Long then short re-encoding must not retain previous sequence state.
		f := ref.Fixtures[0]
		a, err := encode(f.Tokens)
		if err != nil {
			t.Fatal(err)
		}
		b, err := encode(f.Tokens)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(a, b) {
			t.Fatal("repeated request changed hidden output")
		}
		stats := gpu.Stats()
		gpu.Close()
		gpu.Close()
		free, _ := nvidia.MemInfo()
		t.Logf("free_after_close=%d free_before_load=%d", free, stats.FreeBytesAtLoad)
		if free+32*1024*1024 < stats.FreeBytesAtLoad {
			t.Fatal("encoder leaked GPU memory")
		}
		if _, err = encode(f.Tokens); err == nil {
			t.Fatal("closed encoder accepted request")
		}
	}
}
