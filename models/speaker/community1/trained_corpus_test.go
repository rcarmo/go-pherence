package community1

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio/media"
)

// TestCommunity1TrainedCorpus is an explicit bounded CPU corpus gate. Unlike
// the fixed tutorial regression, it accepts any hash-pinned canonical16k WAV
// whose geometry fits PlanDiarizationWindows. It downloads nothing, chooses no
// fallback, and never runs in ordinary tests.
func TestCommunity1TrainedCorpus(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_COMMUNITY1_CORPUS") != "1" {
		t.Skip("set GO_PHERENCE_TEST_COMMUNITY1_CORPUS=1 in an authorised model/CPU window")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 10*time.Minute {
		t.Fatal("trained Community-1 corpus gate requires go test -timeout of at most 10m")
	}
	path := os.Getenv("GO_PHERENCE_COMMUNITY1_CORPUS_WAV")
	wantHash := os.Getenv("GO_PHERENCE_COMMUNITY1_CORPUS_WAV_SHA256")
	wantSamples, err := strconv.ParseInt(os.Getenv("GO_PHERENCE_COMMUNITY1_CORPUS_SAMPLES"), 10, 64)
	if path == "" || len(wantHash) != sha256.Size*2 || wantSamples < 1 {
		t.Fatal("explicit corpus WAV, lowercase SHA-256 and sample count required")
	}
	if _, err := hex.DecodeString(wantHash); err != nil || strings.ToLower(wantHash) != wantHash {
		t.Fatal("invalid corpus WAV SHA-256")
	}
	_ = pinnedTrainedDiarizationFile(t, path, wantHash)
	ctx := context.Background()
	reader, err := media.OpenCanonicalPCM(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	if int64(reader.Timeline().Samples) != wantSamples {
		t.Fatal("corpus WAV sample count", reader.Timeline().Samples)
	}
	windows, err := PlanDiarizationWindows(wantSamples, 160000, 16000)
	if err != nil {
		t.Fatal(err)
	}
	segmentation, filters, embedding, prepared := loadTrainedDiarizationModels(t, ctx)
	seg, err := NewExperimentalSegmentation(ctx, segmentation, filters)
	if err != nil {
		t.Fatal(err)
	}
	emb, err := NewExperimentalEmbedding(ctx, embedding)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewExperimentalDiarization(ctx, seg, emb, prepared.Model)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DiarizationPCMConfig{WindowSamples: 160000, StepSamples: 16000, MinimumEmbeddingSamples: 400, ExcludeOverlap: true, MinSpeakers: 1, MaxSpeakers: 64, AHCThreshold: .6, Fa: .07, Fb: .8, Constrained: true, TiePolicy: RejectAmbiguousTies}
	if os.Getenv("GO_PHERENCE_DIARIZATION_LOWEST_TIES") == "1" {
		cfg.TiePolicy = LowestIndexTies
	}
	started := time.Now()
	result, err := model.RunPCMObserved(ctx, reader, wantSamples, cfg, SegmentationModes{SincNetSIMDFMA, LSTMSIMD, HeadSIMD}, WeSpeakerBlockSIMD, func(stage string, window int) error {
		t.Logf("CORPUS_STAGE %s window=%d", stage, window)
		return nil
	})
	elapsed := time.Since(started)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Windows) != len(windows) || result.Grid.Frames != 589 || result.LocalSpeakers != 3 || result.EmbeddingDimension != 256 || result.Postprocess == nil || result.Postprocess.Timeline == nil {
		t.Fatal("trained corpus result contract")
	}
	if cfg.TiePolicy == RejectAmbiguousTies && len(result.Postprocess.Timeline.AmbiguousFrames) != 0 {
		t.Fatal("strict corpus gate published ambiguous frames")
	}
	t.Logf("TRAINED_CORPUS_RESULT samples=%d windows=%d elapsed=%s path=%s training_rows=%d clusters=%d full_turns=%d exclusive_turns=%d ambiguous_frames=%d", wantSamples, len(windows), elapsed, result.Postprocess.Path, result.Postprocess.TrainingRows, result.Postprocess.Clusters, len(result.Postprocess.FullTurns), len(result.Postprocess.ExclusiveTurns), len(result.Postprocess.Timeline.AmbiguousFrames))
	out := os.Getenv("GO_PHERENCE_COMMUNITY1_CORPUS_OUTPUT")
	if out == "" {
		return
	}
	data, err := json.MarshalIndent(struct {
		Schema       int
		InputSHA256  string
		InputSamples int64
		ElapsedNanos int64
		Config       DiarizationPCMConfig
		Result       *DiarizationPCMResult
	}{1, wantHash, wantSamples, elapsed.Nanoseconds(), cfg, result}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	output, err := os.OpenFile(out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, writeErr := output.Write(append(data, '\n'))
	closeErr := output.Close()
	if writeErr != nil {
		t.Fatal(writeErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
}
