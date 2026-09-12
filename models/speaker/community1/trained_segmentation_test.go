package community1

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type trainedSegmentationManifest struct {
	Schema        int
	ModelRevision string `json:"model_revision"`
	CheckpointSHA string `json:"checkpoint_sha256"`
	Config        SegmentationLoadConfig
	Files         map[string]struct {
		SHA256 string
		Bytes  int
	}
	Cases []struct {
		Name, File      string
		Samples, Frames int
	}
}

func trainedSegmentationAssets(t *testing.T) (string, trainedSegmentationManifest) {
	t.Helper()
	dir := os.Getenv("GO_PHERENCE_COMMUNITY1_SEGMENTATION_DIR")
	if dir == "" {
		t.Fatal("explicit converted checkpoint directory required")
	}
	data, e := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if e != nil {
		t.Fatal(e)
	}
	digest := sha256.Sum256(data)
	if hex.EncodeToString(digest[:]) != "bc8af5359abf662c93646fd00fe29bec02ab4e5998cef70a92be45af454ff1e0" {
		t.Fatal("pinned lowered manifest changed")
	}
	var m trainedSegmentationManifest
	if e = json.Unmarshal(data, &m); e != nil {
		t.Fatal(e)
	}
	if m.Schema != 1 || m.ModelRevision != "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee" || m.CheckpointSHA != "7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e" || len(m.Cases) != 4 || len(m.Files) != 6 {
		t.Fatal("manifest contract")
	}
	for name, info := range m.Files {
		if filepath.Base(name) != name {
			t.Fatal("manifest path")
		}
		data, e := os.ReadFile(filepath.Join(dir, name))
		if e != nil || len(data) != info.Bytes {
			t.Fatal("asset size", name, e)
		}
		hash := sha256.Sum256(data)
		if hex.EncodeToString(hash[:]) != info.SHA256 {
			t.Fatal("asset hash", name)
		}
	}
	return dir, m
}
func trainedTensor(t *testing.T, f *safetensors.File, name string) []float32 {
	t.Helper()
	x, _, e := f.GetFloat32(name)
	if e != nil {
		t.Fatal(e)
	}
	return append([]float32(nil), x...)
}
func trainedCompare(t *testing.T, name string, got, want []float32) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("shape", name, len(got), len(want))
	}
	maxerr := 0.
	index := 0
	exact := 0
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("nonfinite", name)
		}
		d := math.Abs(float64(v) - float64(want[i]))
		if d > maxerr {
			maxerr = d
			index = i
		}
		if math.Float32bits(v) == math.Float32bits(want[i]) {
			exact++
		}
	}
	data, _ := json.Marshal(map[string]any{"stage": name, "values": len(got), "max_abs": maxerr, "index": index, "exact": exact, "gate": 2e-4, "pass": maxerr <= 2e-4})
	t.Logf("TRAINED_SEGMENTATION_METRIC %s", data)
	return maxerr
}

// Endpoint and local-mask gates are mandatory. Every intermediate boundary is
// recorded; a separate opt-in strict mode enforces them and retains failures.
// No performance claim: reference and Go traces are separate functional runs.
func TestCommunity1TrainedSegmentation(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_COMMUNITY1_SEGMENTATION") != "1" {
		t.Skip("explicit trained Community1 CPU qualification opt-in required")
	}
	dir, manifest := trainedSegmentationAssets(t)
	ctx := context.Background()
	src, e := safetensors.Open(filepath.Join(dir, "segmentation.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	m, e := LoadSegmentationSource(ctx, src, manifest.Config)
	src.Close()
	if e != nil {
		t.Fatal(e)
	}
	fs, e := safetensors.Open(filepath.Join(dir, "filters.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	filters := trainedTensor(t, fs, "sincnet.filters")
	fs.Close()
	model, e := NewExperimentalSegmentation(ctx, m, filters)
	if e != nil {
		t.Fatal(e)
	}
	for _, c := range manifest.Cases {
		t.Run(c.Name, func(t *testing.T) {
			oracle, e := safetensors.Open(filepath.Join(dir, c.File))
			if e != nil {
				t.Fatal(e)
			}
			defer oracle.Close()
			pcm := trainedTensor(t, oracle, "pcm")
			failures := 0
			check := func(name string, x []float32) {
				if trainedCompare(t, name, x, trainedTensor(t, oracle, name)) > 2e-4 {
					failures++
				}
			}
			result, e := model.ForwardPCMObserved(ctx, pcm, SegmentationModes{SincNetSIMDFMA, LSTMSIMD, HeadSIMD}, SegmentationObservers{
				SincNet: func(stage, _, _ int, v []float32) { check(fmt.Sprintf("sincnet.%d", stage), v) },
				LSTM:    func(layer, _, _ int, v []float32) { check(fmt.Sprintf("lstm.%d", layer), v) },
				Head:    func(layer, _, _ int, v []float32) { check(fmt.Sprintf("head.%d", layer), v) },
			})
			if e != nil {
				t.Fatal(e)
			}
			grid, probs := result.Grid, result.LogProbabilities
			if grid.Frames != c.Frames || len(pcm) != c.Samples || result.Classes != 7 {
				t.Fatal("grid/classes")
			}
			endpoint := trainedCompare(t, "log_probabilities", probs, trainedTensor(t, oracle, "log_probabilities"))
			if endpoint > 2e-4 {
				t.Error("trained endpoint exceeds2e-4", endpoint)
			}
			p, e := NewPowerset(3, 2)
			if e != nil {
				t.Fatal(e)
			}
			got, e := p.Decode(ctx, probs, grid.Frames, PowersetHard)
			if e != nil {
				t.Fatal(e)
			}
			want, e := p.Decode(ctx, trainedTensor(t, oracle, "log_probabilities"), grid.Frames, PowersetHard)
			if e != nil {
				t.Fatal(e)
			}
			disagree := 0
			for frame := 0; frame < grid.Frames; frame++ {
				for j := 0; j < 3; j++ {
					if got[frame*3+j] != want[frame*3+j] {
						disagree++
						break
					}
				}
			}
			t.Logf("TRAINED_SEGMENTATION_RESULT failed_boundaries=%d hard_disagreement_frames=%d/%d", failures, disagree, grid.Frames)
			if disagree != 0 {
				t.Error("trained hard-mask disagreement", disagree)
			}
			if os.Getenv("GO_PHERENCE_TEST_COMMUNITY1_STRICT") == "1" && (failures != 0 || disagree != 0) {
				t.Fatal("strict trained segmentation qualification failed")
			}
		})
	}
}
