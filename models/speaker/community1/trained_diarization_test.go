package community1

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"github.com/rcarmo/go-pherence/loader/audio/media"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"os"
	"path/filepath"
	"testing"
)

func TestCommunity1TrainedDiarization(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_COMMUNITY1_DIARIZATION") != "1" {
		t.Skip("explicit bounded30s trained Go diarization opt-in")
	}
	ctx := context.Background()
	sd, sm := trainedSegmentationAssets(t)
	ed, em := trainedEmbeddingAssets(t)
	source, e := safetensors.Open(filepath.Join(sd, "segmentation.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	sc, e := LoadSegmentationSource(ctx, source, sm.Config)
	source.Close()
	if e != nil {
		t.Fatal(e)
	}
	fs, e := safetensors.Open(filepath.Join(sd, "filters.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	filters := trainedTensor(t, fs, "sincnet.filters")
	fs.Close()
	seg, e := NewExperimentalSegmentation(ctx, sc, filters)
	if e != nil {
		t.Fatal(e)
	}
	source, e = safetensors.Open(filepath.Join(ed, "embedding.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	ec, e := LoadWeSpeakerResNetSource(ctx, source, em.Config, em.Prefix)
	source.Close()
	if e != nil {
		t.Fatal(e)
	}
	emb, e := NewExperimentalEmbedding(ctx, ec)
	if e != nil {
		t.Fatal(e)
	}
	pin := func(path, hash string) *os.File {
		t.Helper()
		b, e := os.ReadFile(path)
		if e != nil {
			t.Fatal(e)
		}
		sum := sha256.Sum256(b)
		if hex.EncodeToString(sum[:]) != hash {
			t.Fatal("asset hash", path)
		}
		f, e := os.Open(path)
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(func() { f.Close() })
		return f
	}
	base := os.Getenv("GO_PHERENCE_COMMUNITY1_PLDA_DIR")
	if base == "" {
		t.Fatal("explicit PLDA directory required")
	}
	x := pin(filepath.Join(base, "xvec_transform.npz"), "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f")
	p := pin(filepath.Join(base, "plda.npz"), "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255")
	xs, _ := x.Stat()
	ps, _ := p.Stat()
	prepared, e := LoadRawPLDANPZ(ctx, x, xs.Size(), p, ps.Size(), PLDAConfig{256, 128, 128})
	if e != nil {
		t.Fatal(e)
	}
	t.Logf("TRAINED_PLDA condition=%g residual=%g", prepared.ConditionInf, prepared.InverseResidual)
	path := os.Getenv("GO_PHERENCE_COMMUNITY1_PUBLIC_WAV")
	if path == "" {
		t.Fatal("explicit public wav required")
	}
	pin(path, "c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e")
	reader, e := media.OpenCanonicalPCM(ctx, path)
	if e != nil {
		t.Fatal(e)
	}
	defer reader.Close()
	if reader.Timeline().Samples != 480000 {
		t.Fatal("public sample count")
	}
	model, e := NewExperimentalDiarization(ctx, seg, emb, prepared.Model)
	if e != nil {
		t.Fatal(e)
	}
	cfg := DiarizationPCMConfig{WindowSamples: 160000, StepSamples: 16000, MinimumEmbeddingSamples: 400, ExcludeOverlap: true, MinSpeakers: 1, MaxSpeakers: 64, AHCThreshold: .6, Fa: .07, Fb: .8, Constrained: true, TiePolicy: RejectAmbiguousTies}
	// Explicit diagnostic alternative only; default retains ambiguous-tie errors.
	if os.Getenv("GO_PHERENCE_DIARIZATION_LOWEST_TIES") == "1" {
		cfg.TiePolicy = LowestIndexTies
	}
	result, e := model.RunPCMObserved(ctx, reader, 480000, cfg, SegmentationModes{SincNetSIMDFMA, LSTMSIMD, HeadSIMD}, WeSpeakerBlockSIMD, func(stage string, window int) error {
		t.Logf("DIARIZATION_STAGE %s window=%d", stage, window)
		return nil
	})
	if e != nil {
		t.Fatal(e)
	}
	if len(result.Windows) != 21 || result.Grid.Frames != 589 || result.LocalSpeakers != 3 || result.EmbeddingDimension != 256 {
		t.Fatal("trained result dimensions")
	}
	t.Logf("TRAINED_DIARIZATION_RESULT path=%s training_rows=%d clusters=%d full_turns=%d exclusive_turns=%d ambiguous_frames=%d", result.Postprocess.Path, result.Postprocess.TrainingRows, result.Postprocess.Clusters, len(result.Postprocess.FullTurns), len(result.Postprocess.ExclusiveTurns), len(result.Postprocess.Timeline.AmbiguousFrames))
	out := os.Getenv("GO_PHERENCE_COMMUNITY1_DIARIZATION_OUTPUT")
	if out != "" {
		data, e := json.MarshalIndent(struct {
			Config DiarizationPCMConfig
			Result *DiarizationPCMResult
		}{cfg, result}, "", "  ")
		if e != nil {
			t.Fatal(e)
		}
		f, e := os.OpenFile(out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if e != nil {
			t.Fatal(e)
		}
		_, e = f.Write(append(data, '\n'))
		ce := f.Close()
		if e != nil {
			t.Fatal(e)
		}
		if ce != nil {
			t.Fatal(ce)
		}
	}
}
