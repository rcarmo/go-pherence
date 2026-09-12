package community1

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
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
func TestCommunity1TrainedSincNetStage1Boundary(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_COMMUNITY1_STAGE1") != "1" {
		t.Skip("explicit trained stage1 diagnostic opt-in")
	}
	dir, manifest := trainedSegmentationAssets(t)
	ctx := context.Background()
	src, e := safetensors.Open(filepath.Join(dir, "segmentation.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	checkpoint, e := LoadSegmentationSource(ctx, src, manifest.Config)
	e = errors.Join(e, src.Close())
	if e != nil {
		t.Fatal(e)
	}
	fs, e := safetensors.Open(filepath.Join(dir, "filters.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	filters := trainedTensor(t, fs, "sincnet.filters")
	e = fs.Close()
	if e != nil {
		t.Fatal(e)
	}
	model, e := NewExperimentalSegmentation(ctx, checkpoint, filters)
	if e != nil {
		t.Fatal(e)
	}
	compare := func(a, b []float32) (float64, int) {
		if len(a) != len(b) {
			t.Fatal("shape", len(a), len(b))
		}
		mx := 0.
		bits := 0
		for i, v := range a {
			d := math.Abs(float64(v) - float64(b[i]))
			mx = math.Max(mx, d)
			if math.Float32bits(v) != math.Float32bits(b[i]) {
				bits++
			}
		}
		return mx, bits
	}
	for _, c := range manifest.Cases {
		if c.Name != "silence-1s" && c.Name != "public-10-20s" {
			continue
		}
		t.Run(c.Name, func(t *testing.T) {
			o, e := safetensors.Open(filepath.Join(dir, c.File))
			if e != nil {
				t.Fatal(e)
			}
			defer o.Close()
			input := trainedTensor(t, o, "sincnet.0")
			want := trainedTensor(t, o, "sincnet.1")
			if len(input)%80 != 0 {
				t.Fatal("input shape")
			}
			n := len(input) / 80
			wantConv, wantPool := trainedTensor(t, o, "conv.1"), trainedTensor(t, o, "pool.1")
			for _, mode := range []struct {
				name string
				mode SincNetMode
			}{{"scalar", SincNetScalar}, {"simd-dot", SincNetSIMD}, {"scalar-fma", SincNetScalarFMA}, {"simd-fma", SincNetSIMDFMA}} {
				conv, frames, e := sincNetConvolve(ctx, input, 80, n, model.frontend.weights.Conv[0].Weight, model.frontend.weights.Conv[0].Bias, 60, 5, 1, mode.mode)
				if e != nil {
					t.Fatal(e)
				}
				convErr, convBits := compare(conv, wantConv)
				pooled, pframes, e := sincNetPool(ctx, conv, 60, frames)
				if e != nil {
					t.Fatal(e)
				}
				poolErr, poolBits := compare(pooled, wantPool)
				if e = sincNetNormMode(ctx, pooled, 60, pframes, model.frontend.weights.Norm[1], mode.mode); e != nil {
					t.Fatal(e)
				}
				for i, v := range pooled {
					if v < 0 {
						pooled[i] = v * .01
					}
				}
				mx, bits := compare(pooled, want)
				t.Logf("COMMUNITY_STAGE1 fixture=%s mode=%s conv_max=%g conv_bits=%d pool_max=%g pool_bits=%d output_max=%g output_bits=%d", c.Name, mode.name, convErr, convBits, poolErr, poolBits, mx, bits)
			}
		})
	}
}

func TestCommunity1TrainedSegmentationModeBoundary(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_COMMUNITY1_BOUNDARY") != "1" {
		t.Skip("explicit trained boundary diagnostic opt-in")
	}
	dir, manifest := trainedSegmentationAssets(t)
	ctx := context.Background()
	src, e := safetensors.Open(filepath.Join(dir, "segmentation.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	m, e := LoadSegmentationSource(ctx, src, manifest.Config)
	e = errors.Join(e, src.Close())
	if e != nil {
		t.Fatal(e)
	}
	fs, e := safetensors.Open(filepath.Join(dir, "filters.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	filters := trainedTensor(t, fs, "sincnet.filters")
	e = fs.Close()
	if e != nil {
		t.Fatal(e)
	}
	model, e := NewExperimentalSegmentation(ctx, m, filters)
	if e != nil {
		t.Fatal(e)
	}
	maxdiff := func(a, b []float32) (float64, int) {
		if len(a) != len(b) {
			t.Fatal("shape")
		}
		mx := 0.
		bits := 0
		for i, v := range a {
			d := math.Abs(float64(v) - float64(b[i]))
			mx = math.Max(mx, d)
			if math.Float32bits(v) != math.Float32bits(b[i]) {
				bits++
			}
		}
		return mx, bits
	}
	for _, c := range manifest.Cases {
		if c.Name != "silence-1s" && c.Name != "public-10-20s" {
			continue
		}
		t.Run(c.Name, func(t *testing.T) {
			o, e := safetensors.Open(filepath.Join(dir, c.File))
			if e != nil {
				t.Fatal(e)
			}
			defer o.Close()
			pcm := trainedTensor(t, o, "pcm")
			runs := map[string]map[string][]float32{}
			for _, mode := range []struct {
				name string
				s    SincNetMode
				l    LSTMMode
				h    HeadMode
			}{{"scalar", SincNetScalarFMA, LSTMScalar, HeadScalar}, {"simd", SincNetSIMDFMA, LSTMSIMD, HeadSIMD}} {
				values := map[string][]float32{}
				_, e := model.ForwardPCMObserved(ctx, pcm, SegmentationModes{mode.s, mode.l, mode.h}, SegmentationObservers{SincNet: func(stage, _, _ int, v []float32) {
					values[fmt.Sprintf("sincnet.%d", stage)] = append([]float32(nil), v...)
				}, LSTM: func(layer, _, _ int, v []float32) {
					values[fmt.Sprintf("lstm.%d", layer)] = append([]float32(nil), v...)
				}, Head: func(layer, _, _ int, v []float32) {
					values[fmt.Sprintf("head.%d", layer)] = append([]float32(nil), v...)
				}})
				if e != nil {
					t.Fatal(e)
				}
				runs[mode.name] = values
			}
			for _, name := range []string{"sincnet.0", "sincnet.1", "sincnet.2", "lstm.0", "lstm.1", "lstm.2", "lstm.3", "head.0", "head.1", "head.2"} {
				a, b := runs["scalar"][name], runs["simd"][name]
				modeErr, bits := maxdiff(a, b)
				oracleErr, _ := maxdiff(b, trainedTensor(t, o, name))
				t.Logf("COMMUNITY_BOUNDARY fixture=%s stage=%s scalar_simd_max=%g differing_bits=%d oracle_max=%g", c.Name, name, modeErr, bits, oracleErr)
			}
		})
	}
}

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
