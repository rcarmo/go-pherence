package community1

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio/media"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func pinnedTrainedDiarizationFile(t *testing.T, path, hash string) *os.File {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(b)
	if hex.EncodeToString(sum[:]) != hash {
		t.Fatal("asset hash", path)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.Close() })
	return f
}

func loadTrainedDiarizationModels(t *testing.T, ctx context.Context) (*SegmentationCheckpoint, []float32, *WeSpeakerResNet34, *RawPLDAPreparation) {
	t.Helper()
	sd, sm := trainedSegmentationAssets(t)
	ed, em := trainedEmbeddingAssets(t)
	source, err := safetensors.Open(filepath.Join(sd, "segmentation.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	segmentation, loadErr := LoadSegmentationSource(ctx, source, sm.Config)
	err = errors.Join(loadErr, source.Close())
	if err != nil {
		t.Fatal(err)
	}
	filtersFile, err := safetensors.Open(filepath.Join(sd, "filters.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	filters := trainedTensor(t, filtersFile, "sincnet.filters")
	if err = filtersFile.Close(); err != nil {
		t.Fatal(err)
	}
	source, err = safetensors.Open(filepath.Join(ed, "embedding.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	embedding, loadErr := LoadWeSpeakerResNetSource(ctx, source, em.Config, em.Prefix)
	err = errors.Join(loadErr, source.Close())
	if err != nil {
		t.Fatal(err)
	}
	base := os.Getenv("GO_PHERENCE_COMMUNITY1_PLDA_DIR")
	if base == "" {
		t.Fatal("explicit PLDA directory required")
	}
	x := pinnedTrainedDiarizationFile(t, filepath.Join(base, "xvec_transform.npz"), "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f")
	p := pinnedTrainedDiarizationFile(t, filepath.Join(base, "plda.npz"), "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255")
	xs, err := x.Stat()
	if err != nil {
		t.Fatal(err)
	}
	ps, err := p.Stat()
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := LoadRawPLDANPZ(ctx, x, xs.Size(), p, ps.Size(), PLDAConfig{256, 128, 128})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("TRAINED_PLDA condition=%g residual=%g", prepared.ConditionInf, prepared.InverseResidual)
	return segmentation, filters, embedding, prepared
}

func openTrainedDiarizationPCM(t *testing.T, ctx context.Context) *media.PCMReader {
	t.Helper()
	path := os.Getenv("GO_PHERENCE_COMMUNITY1_PUBLIC_WAV")
	if path == "" {
		t.Fatal("explicit public wav required")
	}
	pinnedTrainedDiarizationFile(t, path, "c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e")
	reader, err := media.OpenCanonicalPCM(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	if reader.Timeline().Samples != 480000 {
		t.Fatal("public sample count")
	}
	return reader
}

func TestCommunity1TrainedDiarization(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_COMMUNITY1_DIARIZATION") != "1" {
		t.Skip("explicit bounded30s trained Go diarization opt-in")
	}
	ctx := context.Background()
	sc, filters, ec, prepared := loadTrainedDiarizationModels(t, ctx)
	seg, e := NewExperimentalSegmentation(ctx, sc, filters)
	if e != nil {
		t.Fatal(e)
	}
	emb, e := NewExperimentalEmbedding(ctx, ec)
	if e != nil {
		t.Fatal(e)
	}
	reader := openTrainedDiarizationPCM(t, ctx)
	model, e := NewExperimentalDiarization(ctx, seg, emb, prepared.Model)
	if e != nil {
		t.Fatal(e)
	}
	cfg := DiarizationPCMConfig{WindowSamples: 160000, StepSamples: 16000, MinimumEmbeddingSamples: 400, ExcludeOverlap: true, MinSpeakers: 1, MaxSpeakers: 64, AHCThreshold: .6, Fa: .07, Fb: .8, Constrained: true, TiePolicy: RejectAmbiguousTies}
	// Explicit diagnostic alternative only; default retains ambiguous-tie errors.
	if os.Getenv("GO_PHERENCE_DIARIZATION_LOWEST_TIES") == "1" {
		cfg.TiePolicy = LowestIndexTies
	}
	embeddingMode := WeSpeakerBlockSIMD
	if os.Getenv("GO_PHERENCE_TEST_COMMUNITY1_GEMM") == "1" {
		embeddingMode = WeSpeakerBlockGEMM
	}
	result, e := model.RunPCMObserved(ctx, reader, 480000, cfg, SegmentationModes{SincNetSIMDFMA, LSTMSIMD, HeadSIMD}, embeddingMode, func(stage string, window int) error {
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
