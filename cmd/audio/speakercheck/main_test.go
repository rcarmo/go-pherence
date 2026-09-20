package main

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestScoreExpectedCanonicalizesSpeakerLabels(t *testing.T) {
	segments := []checkSegment{{Speaker: 2}, {Speaker: 2}, {Speaker: 1}, {Speaker: 1}}
	score := scoreExpected(segments, []int{1, 1, 2, 2})
	if score.Failure != "" {
		t.Fatalf("unexpected failure: %s", score.Failure)
	}
	if !score.Passed || score.ExactMatches != 4 || score.Total != 4 {
		t.Fatalf("unexpected exact score: %+v", score)
	}
	if score.Accuracy != 1 || score.PairwiseScore != 1 {
		t.Fatalf("unexpected invariant scores: %+v", score)
	}
}

func TestScoreExpectedFailsOnLengthMismatch(t *testing.T) {
	score := scoreExpected([]checkSegment{{Speaker: 1}, {Speaker: 2}}, []int{1})
	if score.Passed || score.Failure == "" {
		t.Fatalf("expected explicit failure, got %+v", score)
	}
	if !strings.Contains(score.Failure, "expected 1 labels for 2 segments") {
		t.Fatalf("unexpected failure message: %q", score.Failure)
	}
	if exitCodeForReport(checkReport{Score: score}) != 1 {
		t.Fatal("length mismatch should fail regardless of output mode")
	}
}

func TestValidateParametersRejectsNonFiniteValues(t *testing.T) {
	cases := []struct {
		name string
		args [4]float64
	}{
		{name: "threshold", args: [4]float64{math.NaN(), 0, 0, 0}},
		{name: "context", args: [4]float64{0.3, math.Inf(1), 0, 0}},
		{name: "start", args: [4]float64{0.3, 0.5, math.Inf(-1), 0}},
		{name: "duration", args: [4]float64{0.3, 0.5, 0, math.NaN()}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := validateParameters(tc.args[0], tc.args[1], tc.args[2], tc.args[3]); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestSampleOffsetClampsBeforeNarrowing(t *testing.T) {
	if got := sampleOffset(1e300, 10, 10); got != 10 {
		t.Fatalf("large finite offset = %d, want 10", got)
	}
	if got := sampleOffset(math.Inf(1), 10, 10); got != 10 {
		t.Fatalf("+Inf offset = %d, want 10", got)
	}
	if got := sampleOffset(math.NaN(), 10, 10); got != 0 {
		t.Fatalf("NaN offset = %d, want 0", got)
	}
	samples := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	got := sliceSamples(samples, 10, 0.6, 1e300)
	if len(got) != 4 || got[0] != 6 || got[3] != 9 {
		t.Fatalf("unexpected slice: %v", got)
	}
}

func TestLoadAudioSamplesFallbackUsesBoundedRunnerAndPrivateTempDir(t *testing.T) {
	origWAV := wavLoader
	origLookPath := ffmpegLookPath
	origRunner := commandRunner
	origTempDirMaker := tempDirMaker
	origRemoveAll := removeAll
	t.Cleanup(func() {
		wavLoader = origWAV
		ffmpegLookPath = origLookPath
		commandRunner = origRunner
		tempDirMaker = origTempDirMaker
		removeAll = origRemoveAll
	})

	input := filepath.Join(t.TempDir(), "input.mp3")
	var decodedPath string
	var runnerArgs []string
	var runnerLimit int
	var deadline time.Time

	wavLoader = func(path string) ([]float32, int, error) {
		switch path {
		case input:
			return nil, 0, errors.New("not a wav")
		case decodedPath:
			if filepath.Base(path) != "decoded.wav" {
				t.Fatalf("unexpected decoded file name: %s", path)
			}
			if filepath.Dir(path) == os.TempDir() {
				t.Fatalf("expected private temp directory, got %s", path)
			}
			return []float32{0.25, 0.5}, 16000, nil
		default:
			t.Fatalf("unexpected wav path: %s", path)
			return nil, 0, nil
		}
	}
	ffmpegLookPath = func(name string) (string, error) {
		if name != "ffmpeg" {
			t.Fatalf("unexpected executable lookup: %s", name)
		}
		return "/fake/ffmpeg", nil
	}
	commandRunner = func(ctx context.Context, path string, args, env []string, limit int) ([]byte, []byte, error) {
		if path != "/fake/ffmpeg" {
			t.Fatalf("unexpected command path: %s", path)
		}
		var ok bool
		deadline, ok = ctx.Deadline()
		if !ok {
			t.Fatal("expected ffmpeg deadline")
		}
		runnerArgs = append([]string(nil), args...)
		runnerLimit = limit
		decodedPath = args[len(args)-1]
		if strings.Contains(strings.Join(args, " "), " -y ") || (len(args) > 0 && args[0] == "-y") {
			t.Fatal("unexpected overwrite flag")
		}
		return nil, nil, nil
	}
	tempDirMaker = os.MkdirTemp
	removeAll = os.RemoveAll

	samples, sr, cleanup, err := loadAudioSamples(input)
	if err != nil {
		t.Fatalf("loadAudioSamples returned error: %v", err)
	}
	if len(samples) != 2 || sr != 16000 {
		t.Fatalf("unexpected decode result: len=%d sr=%d", len(samples), sr)
	}
	if cleanup == nil {
		t.Fatal("expected cleanup")
	}
	if runnerLimit != ffmpegOutputLimit {
		t.Fatalf("runner limit = %d, want %d", runnerLimit, ffmpegOutputLimit)
	}
	if time.Until(deadline) <= 0 || time.Until(deadline) > ffmpegDecodeTimeout {
		t.Fatalf("unexpected deadline: %v", deadline)
	}
	gotArgs := strings.Join(runnerArgs, "\n")
	wantArgs := strings.Join([]string{"-nostdin", "-hide_banner", "-loglevel", "error", "-i", input, "-ar", "16000", "-ac", "1", decodedPath}, "\n")
	if gotArgs != wantArgs {
		t.Fatalf("unexpected ffmpeg args: %v", runnerArgs)
	}
	if _, err := os.Stat(filepath.Dir(decodedPath)); err != nil {
		t.Fatalf("temp dir missing before cleanup: %v", err)
	}
	cleanup()
	if _, err := os.Stat(filepath.Dir(decodedPath)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("temp dir still present after cleanup: %v", err)
	}
}

func TestInvalidLabelsAndContextCannotPass(t *testing.T) {
	if _, err := parseExpectedLabels(" , "); err == nil {
		t.Fatal("empty label list")
	}
	if s := scoreExpected([]checkSegment{{Speaker: 0}}, []int{1}); s.Passed || s.Failure == "" {
		t.Fatal(s)
	}
	for _, c := range []float64{-1, 31, 1e300} {
		if validateParameters(.3, c, 0, 0) == nil {
			t.Fatal(c)
		}
	}
}
