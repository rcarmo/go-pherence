// Command speakercheck validates the speaker diarization path without running Whisper.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/internal/commandcapture"
	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/model/speaker"
)

const (
	ffmpegDecodeTimeout = 30 * time.Second
	ffmpegOutputLimit   = 64 << 10
)

var (
	wavLoader      = audio.WAV
	ffmpegLookPath = exec.LookPath
	tempDirMaker   = os.MkdirTemp
	removeAll      = os.RemoveAll
	commandRunner  = commandcapture.Run
)

type checkReport struct {
	Input     string          `json:"input"`
	Model     string          `json:"model"`
	Threshold float64         `json:"threshold"`
	Context   float64         `json:"context"`
	Start     float64         `json:"start"`
	Duration  float64         `json:"duration"`
	Segments  []checkSegment  `json:"segments"`
	Counts    map[string]int  `json:"counts"`
	Sims      []checkPairwise `json:"similarities,omitempty"`
	Score     *checkScore     `json:"score,omitempty"`
}

type checkSegment struct {
	Index    int     `json:"index"`
	Start    float64 `json:"start"`
	End      float64 `json:"end"`
	Duration float64 `json:"duration"`
	Speaker  int     `json:"speaker"`
}

type checkPairwise struct {
	I      int     `json:"i"`
	J      int     `json:"j"`
	Cosine float32 `json:"cosine"`
}

type checkScore struct {
	Expected      []int   `json:"expected"`
	ExactMatches  int     `json:"exact_matches"`
	Total         int     `json:"total"`
	Accuracy      float64 `json:"accuracy"`
	PairwiseAgree int     `json:"pairwise_agree"`
	PairwiseTotal int     `json:"pairwise_total"`
	PairwiseScore float64 `json:"pairwise_score"`
	Passed        bool    `json:"passed"`
	Failure       string  `json:"failure,omitempty"`
}

func main() {
	input := flag.String("input", "", "Input audio file (WAV directly, other formats via ffmpeg if available)")
	modelPath := flag.String("speaker-model", "checkpoints/speaker-ecapa-voxceleb.safetensors", "Converted SpeechBrain ECAPA safetensors model")
	threshold := flag.Float64("threshold", 0.3, "Cosine similarity threshold for agglomerative clustering")
	context := flag.Float64("context", 0.5, "Embedding context padding around VAD segments in seconds")
	startSec := flag.Float64("start", 0, "Start offset in seconds for spot checks")
	durationSec := flag.Float64("duration", 0, "Optional duration in seconds for spot checks")
	showSims := flag.Bool("sims", true, "Print pairwise cosine similarities")
	jsonOut := flag.Bool("json", false, "Emit machine-readable JSON")
	expect := flag.String("expect", "", "Comma-separated expected 1-based speaker labels for scored validation, e.g. 1,1,2,2")
	flag.Parse()
	if *input == "" {
		fmt.Fprintln(os.Stderr, "-input is required")
		os.Exit(2)
	}
	if err := validateParameters(*threshold, *context, *startSec, *durationSec); err != nil {
		fatalf("flags: %v", err)
	}

	dbg := os.Getenv("SPEAKER_DEBUG") != ""
	tick := time.Now()
	lap := func(name string) {
		if dbg {
			fmt.Fprintf(os.Stderr, "[t] %-12s %.2fs\n", name, time.Since(tick).Seconds())
			tick = time.Now()
		}
	}

	samples, sr, cleanup, err := loadAudioSamples(*input)
	// WAV loader returns owned samples; release the temporary file before any
	// later fatal exit (os.Exit does not run defers).
	if cleanup != nil {
		cleanup()
	}
	if err != nil {
		fatalf("audio: %v", err)
	}
	if sr != 16000 {
		samples = audio.ResampleSinc(samples, sr, 16000)
		sr = 16000
	}
	samples = sliceSamples(samples, sr, *startSec, *durationSec)
	lap("audio")
	model, err := speaker.LoadSpeechBrainECAPASafetensors(*modelPath)
	if err != nil {
		fatalf("speaker model: %v", err)
	}
	lap("model-load")
	vad := speaker.EnergyVAD(samples, sr, 25, 10, 0)
	lap("vad")
	embeddings := speaker.ExtractSpeechBrainEmbeddingsWithContext(samples, sr, vad, model, *context)
	lap("embed")
	labels := speaker.AgglomerativeCluster(embeddings, float32(*threshold))
	labels = speaker.SmoothSingletonLabels(labels, embeddings, 0.4)
	lap("cluster")

	report := buildReport(*input, *modelPath, *threshold, *context, *startSec, float64(len(samples))/float64(sr), vad, labels, embeddings, *showSims)
	if *expect != "" {
		expected, err := parseExpectedLabels(*expect)
		if err != nil {
			fatalf("expect: %v", err)
		}
		report.Score = scoreExpected(report.Segments, expected)
	}
	exitCode := exitCodeForReport(report)
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fatalf("json: %v", err)
		}
	} else {
		printTextReport(report)
	}
	if exitCode != 0 {
		os.Exit(exitCode)
	}
}

func buildReport(input, modelPath string, threshold, context, start, duration float64, vad []speaker.VADSegment, labels []int, embeddings [][]float32, includeSims bool) checkReport {
	countsInt := map[int]int{}
	for _, label := range labels {
		countsInt[label]++
	}
	keys := make([]int, 0, len(countsInt))
	for k := range countsInt {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	counts := make(map[string]int, len(keys))
	for _, k := range keys {
		counts[fmt.Sprintf("speaker%d", k+1)] = countsInt[k]
	}
	segments := make([]checkSegment, len(vad))
	for i, seg := range vad {
		speakerID := 0
		if i < len(labels) {
			speakerID = labels[i] + 1
		}
		segments[i] = checkSegment{Index: i, Start: seg.Start + start, End: seg.End + start, Duration: seg.End - seg.Start, Speaker: speakerID}
	}
	var sims []checkPairwise
	if includeSims {
		for i := 0; i < len(embeddings); i++ {
			for j := i + 1; j < len(embeddings); j++ {
				sims = append(sims, checkPairwise{I: i, J: j, Cosine: speaker.CosineSimilarity(embeddings[i], embeddings[j])})
			}
		}
	}
	return checkReport{Input: input, Model: modelPath, Threshold: threshold, Context: context, Start: start, Duration: duration, Segments: segments, Counts: counts, Sims: sims}
}

func printTextReport(report checkReport) {
	fmt.Printf("segments=%d speakers=%d threshold=%.2f context=%.2fs start=%.2fs duration=%.2fs\n", len(report.Segments), len(report.Counts), report.Threshold, report.Context, report.Start, report.Duration)
	for _, seg := range report.Segments {
		fmt.Printf("%02d %.2f-%.2f speaker=%d duration=%.2fs\n", seg.Index, seg.Start, seg.End, seg.Speaker, seg.Duration)
	}
	keys := make([]string, 0, len(report.Counts))
	for k := range report.Counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Print("counts")
	for _, k := range keys {
		fmt.Printf(" %s=%d", k, report.Counts[k])
	}
	fmt.Println()
	if report.Score != nil {
		if report.Score.Failure != "" {
			fmt.Printf("score failed: %s\n", report.Score.Failure)
		} else {
			fmt.Printf("score exact=%d/%d accuracy=%.3f pairwise=%d/%d pairwise_score=%.3f passed=%t\n", report.Score.ExactMatches, report.Score.Total, report.Score.Accuracy, report.Score.PairwiseAgree, report.Score.PairwiseTotal, report.Score.PairwiseScore, report.Score.Passed)
		}
	}
	for _, sim := range report.Sims {
		fmt.Printf("sim %02d-%02d %.3f\n", sim.I, sim.J, sim.Cosine)
	}
}

func parseExpectedLabels(value string) ([]int, error) {
	parts := strings.Split(value, ",")
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := strconv.Atoi(part)
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("invalid 1-based label %q", part)
		}
		out = append(out, v)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("expected labels must not be empty")
	}
	return out, nil
}

func scoreExpected(segments []checkSegment, expected []int) *checkScore {
	score := &checkScore{Expected: append([]int(nil), expected...), Total: len(segments)}
	if len(expected) != len(segments) {
		score.Failure = fmt.Sprintf("expected %d labels for %d segments", len(expected), len(segments))
		return score
	}
	predicted := make([]int, len(segments))
	for i, seg := range segments {
		if seg.Speaker <= 0 || expected[i] <= 0 {
			score.Failure = "speaker labels must be positive"
			return score
		}
		predicted[i] = seg.Speaker
	}
	predicted = canonicalizeLabels(predicted)
	expected = canonicalizeLabels(expected)
	for i := range predicted {
		if predicted[i] == expected[i] {
			score.ExactMatches++
		}
	}
	if score.Total == 0 {
		score.Accuracy = 1
	} else {
		score.Accuracy = float64(score.ExactMatches) / float64(score.Total)
	}
	for i := 0; i < len(predicted); i++ {
		for j := i + 1; j < len(predicted); j++ {
			score.PairwiseTotal++
			predSame := predicted[i] == predicted[j]
			expSame := expected[i] == expected[j]
			if predSame == expSame {
				score.PairwiseAgree++
			}
		}
	}
	if score.PairwiseTotal == 0 {
		score.PairwiseScore = 1
	} else {
		score.PairwiseScore = float64(score.PairwiseAgree) / float64(score.PairwiseTotal)
	}
	score.Passed = score.ExactMatches == score.Total
	return score
}

func canonicalizeLabels(labels []int) []int {
	out := make([]int, len(labels))
	seen := make(map[int]int, len(labels))
	next := 1
	for i, label := range labels {
		mapped, ok := seen[label]
		if !ok {
			mapped = next
			seen[label] = mapped
			next++
		}
		out[i] = mapped
	}
	return out
}

func exitCodeForReport(report checkReport) int {
	if report.Score == nil || report.Score.Passed {
		return 0
	}
	return 1
}

func validateParameters(threshold, context, startSec, durationSec float64) error {
	if threshold < -1 || threshold > 1 || context < 0 || context > 30 || startSec < 0 || durationSec < 0 {
		return fmt.Errorf("threshold must be -1..1, context0..30 seconds, start/duration nonnegative")
	}
	for _, param := range []struct {
		name  string
		value float64
	}{
		{name: "threshold", value: threshold},
		{name: "context", value: context},
		{name: "start", value: startSec},
		{name: "duration", value: durationSec},
	} {
		if !isFinite(param.value) {
			return fmt.Errorf("-%s must be finite", param.name)
		}
	}
	return nil
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func loadAudioSamples(path string) ([]float32, int, func(), error) {
	samples, sr, err := wavLoader(path)
	if err == nil {
		return samples, sr, nil, nil
	}
	ffmpegPath, lookErr := ffmpegLookPath("ffmpeg")
	if lookErr != nil {
		return nil, 0, nil, fmt.Errorf("wav decode failed (%v), and ffmpeg was not found for fallback decode", err)
	}
	tmpDir, tempErr := tempDirMaker("", "speakercheck-")
	if tempErr != nil {
		return nil, 0, nil, fmt.Errorf("temp dir: %w", tempErr)
	}
	cleanup := func() { _ = removeAll(tmpDir) }
	tmp := filepath.Join(tmpDir, "decoded.wav")
	ctx, cancel := context.WithTimeout(context.Background(), ffmpegDecodeTimeout)
	defer cancel()
	_, stderr, runErr := commandRunner(ctx, ffmpegPath, []string{"-nostdin", "-hide_banner", "-loglevel", "error", "-i", path, "-ar", "16000", "-ac", "1", tmp}, nil, ffmpegOutputLimit)
	if runErr != nil {
		cleanup()
		return nil, 0, nil, fmt.Errorf("ffmpeg decode: %w%s", runErr, formatCapturedStderr(stderr))
	}
	samples, sr, err = wavLoader(tmp)
	if err != nil {
		cleanup()
		return nil, 0, nil, fmt.Errorf("decoded wav: %w", err)
	}
	return samples, sr, cleanup, nil
}

func formatCapturedStderr(stderr []byte) string {
	msg := strings.TrimSpace(string(stderr))
	if msg == "" {
		return ""
	}
	return ": " + msg
}

func sliceSamples(samples []float32, sampleRate int, startSec, durationSec float64) []float32 {
	if len(samples) == 0 || sampleRate <= 0 {
		return samples
	}
	start := sampleOffset(startSec, sampleRate, len(samples))
	end := len(samples)
	if durationSec > 0 {
		end = start + sampleOffset(durationSec, sampleRate, len(samples)-start)
		if end > len(samples) {
			end = len(samples)
		}
	}
	if end < start {
		end = start
	}
	return samples[start:end]
}

func sampleOffset(seconds float64, sampleRate, maxSamples int) int {
	if sampleRate <= 0 || maxSamples <= 0 {
		return 0
	}
	if math.IsNaN(seconds) || math.IsInf(seconds, -1) || seconds <= 0 {
		return 0
	}
	maxSeconds := float64(maxSamples) / float64(sampleRate)
	if math.IsInf(seconds, 1) || seconds >= maxSeconds {
		return maxSamples
	}
	offset := seconds * float64(sampleRate)
	if offset <= 0 {
		return 0
	}
	if offset >= float64(maxSamples) {
		return maxSamples
	}
	return int(offset)
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
