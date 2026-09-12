package community1

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type trainedEmbeddingManifest struct {
	Schema        int
	ModelRevision string `json:"model_revision"`
	CheckpointSHA string `json:"checkpoint_sha256"`
	Config        WeSpeakerResNetConfig
	Prefix        string
	Files         map[string]struct {
		SHA256 string
		Bytes  int
	}
	Cases []struct {
		Name, File      string
		Samples, Frames int
		CNNFrames       int `json:"cnn_frames"`
	}
}

func trainedEmbeddingAssets(t *testing.T) (string, trainedEmbeddingManifest) {
	t.Helper()
	dir := os.Getenv("GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR")
	if dir == "" {
		t.Fatal("converted embedding directory required")
	}
	b, e := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if e != nil {
		t.Fatal(e)
	}
	h := sha256.Sum256(b)
	if hex.EncodeToString(h[:]) != "b6b6081921bbe1b22db78a67b4692c9a5fc0434b0cfa419b818d056e5a8f8ffb" {
		t.Fatal("embedding manifest changed")
	}
	var m trainedEmbeddingManifest
	if e = json.Unmarshal(b, &m); e != nil {
		t.Fatal(e)
	}
	if m.Schema != 1 || m.ModelRevision != "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee" || m.CheckpointSHA != "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929" || len(m.Files) != 5 || len(m.Cases) != 4 || m.Prefix != "resnet" || m.Config != (WeSpeakerResNetConfig{32, 80, 256}) {
		t.Fatal("embedding contract")
	}
	for name, info := range m.Files {
		if filepath.Base(name) != name || info.Bytes <= 0 || info.Bytes > 128<<20 {
			t.Fatal("asset bounds")
		}
		b, e := os.ReadFile(filepath.Join(dir, name))
		if e != nil || len(b) != info.Bytes {
			t.Fatal(name, e)
		}
		h := sha256.Sum256(b)
		if hex.EncodeToString(h[:]) != info.SHA256 {
			t.Fatal("asset hash", name)
		}
	}
	return dir, m
}
func embeddingCompare(t *testing.T, name string, got, want []float32, atol, rtol float64) bool {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("embedding shape", name, len(got), len(want))
	}
	maxabs, maxratio := 0., 0.
	exact := 0
	pass := true
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) {
			t.Fatal("nonfinite", name)
		}
		d := math.Abs(float64(v) - float64(want[i]))
		budget := atol + rtol*math.Abs(float64(want[i]))
		maxabs = math.Max(maxabs, d)
		if budget > 0 {
			maxratio = math.Max(maxratio, d/budget)
		}
		if d > budget {
			pass = false
		}
		if math.Float32bits(v) == math.Float32bits(want[i]) {
			exact++
		}
	}
	b, _ := json.Marshal(map[string]any{"stage": name, "values": len(got), "max_abs": maxabs, "max_budget_ratio": maxratio, "atol": atol, "rtol": rtol, "exact": exact, "pass": pass})
	t.Logf("TRAINED_EMBEDDING_METRIC %s", b)
	return pass
}
func TestCommunity1TrainedEmbedding(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_COMMUNITY1_EMBEDDING") != "1" {
		t.Skip("explicit trained embedding CPU opt-in")
	}
	ctx := context.Background()
	dir, mf := trainedEmbeddingAssets(t)
	source, e := safetensors.Open(filepath.Join(dir, "embedding.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	m, e := LoadWeSpeakerResNetSource(ctx, source, mf.Config, mf.Prefix)
	source.Close()
	if e != nil {
		t.Fatal(e)
	}
	wrapper, e := NewExperimentalEmbedding(ctx, m)
	if e != nil {
		t.Fatal(e)
	}
	for _, c := range mf.Cases {
		t.Run(c.Name, func(t *testing.T) {
			o, e := safetensors.Open(filepath.Join(dir, c.File))
			if e != nil {
				t.Fatal(e)
			}
			defer o.Close()
			pcm := trainedTensor(t, o, "pcm")
			if len(pcm) != c.Samples {
				t.Fatal("PCM geometry")
			}
			failures := 0
			check := func(name string, a, b []float32, abs, rel float64) {
				if !embeddingCompare(t, name, a, b, abs, rel) {
					failures++
				}
			}
			for _, path := range []string{"reference-fbank", "go-fbank"} {
				observe := func(stage, block int, _ CHWShape, v []float32) {
					name := fmt.Sprintf("trunk.%d.%d", stage, block)
					check(path+"/"+name, v, trainedTensor(t, o, name), 2e-6, 2e-6)
				}
				var features []float32
				var shape CHWShape
				var owned *EmbeddingPCMFrames
				if path == "reference-fbank" {
					features, shape, e = m.ForwardFramesObserved(ctx, trainedTensor(t, o, "fbank"), c.Frames, WeSpeakerBlockSIMD, observe)
				} else {
					owned, e = wrapper.ForwardPCMFramesObserved(ctx, pcm, WeSpeakerBlockSIMD, EmbeddingPCMObservers{Trunk: observe, Fbank: func(stage string, _ int, v []float32) {
						if stage == "centered" {
							check("fbank", v, trainedTensor(t, o, "fbank"), 2e-4, 0)
						}
					}})
					if e == nil {
						var samples, frames int
						shape, samples, frames = owned.Shape()
						if samples != c.Samples || frames != c.Frames {
							t.Fatal("frontend geometry")
						}
					}
				}
				if e != nil {
					t.Fatal(e)
				}
				if shape != (CHWShape{256, 10, c.CNNFrames}) {
					t.Fatal("trunk geometry", shape)
				}
				for _, mask := range []string{"unweighted", "masked"} {
					var weights []float32
					speakers, maskFrames := 0, 0
					if mask == "masked" {
						weights = trainedTensor(t, o, "masks")
						speakers, maskFrames = 3, 7
					}
					observeEmbedding := func(stage string, _, _ int, v []float32) {
						key := mask + ".stats"
						if stage == "embedding" {
							key = mask + ".embedding"
						}
						want := trainedTensor(t, o, key)
						pass := embeddingCompare(t, path+"/"+key, v, want, 2e-4, 0)
						if !pass {
							failures++
							if stage == "embedding" {
								t.Error("trained embedding endpoint exceeds2e-4")
							}
						}
					}
					var result *WeSpeakerEmbeddingResult
					if owned != nil {
						result, e = wrapper.EmbedFramesObserved(ctx, owned, weights, speakers, maskFrames, WeSpeakerBlockSIMD, observeEmbedding)
					} else {
						result, e = m.ForwardEmbeddingObserved(ctx, features, shape, weights, speakers, maskFrames, WeSpeakerBlockSIMD, observeEmbedding)
					}
					if e != nil {
						t.Fatal(e)
					}
					check(path+"/"+mask+".support", result.WeightSum, trainedTensor(t, o, mask+".weight_sum"), 1e-6, 0)
					raw, dtype, _, e := o.GetRaw(mask + ".nonzero_frames")
					if e != nil || dtype != "I64" || len(raw) != 8*len(result.NonzeroFrames) {
						t.Fatal("support metadata", e)
					}
					for i, n := range result.NonzeroFrames {
						if int64(binary.LittleEndian.Uint64(raw[8*i:])) != int64(n) {
							t.Fatal("nonzero support")
						}
					}
				}
			}
			t.Logf("TRAINED_EMBEDDING_RESULT failed_comparisons=%d", failures)
			if os.Getenv("GO_PHERENCE_TEST_COMMUNITY1_EMBEDDING_STRICT") == "1" && failures != 0 {
				t.Fatal("strict trained embedding qualification failed")
			}
		})
	}
}
