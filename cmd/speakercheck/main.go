// Command speakercheck validates the speaker diarization path without running Whisper.
package main

import (
	"time"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/models/speaker"
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
}

func main() {
	input := flag.String("input", "", "Input audio file (WAV directly, other formats via ffmpeg if available)")
	modelPath := flag.String("speaker-model", "models/speaker-ecapa-voxceleb.safetensors", "Converted SpeechBrain ECAPA safetensors model")
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

	dbg := os.Getenv("SPEAKER_DEBUG") != ""
	tick := time.Now()
	lap := func(name string) {
		if dbg {
			fmt.Fprintf(os.Stderr, "[t] %-12s %.2fs\n", name, time.Since(tick).Seconds())
			tick = time.Now()
		}
	}

	samples, sr, cleanup, err := loadAudioSamples(*input)
	if cleanup != nil {
		defer cleanup()
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
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fatalf("json: %v", err)
		}
		return
	}
	printTextReport(report)
	if report.Score != nil && report.Score.PairwiseScore < 1 {
		os.Exit(1)
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
		fmt.Printf("score exact=%d/%d accuracy=%.3f pairwise=%d/%d pairwise_score=%.3f\n", report.Score.ExactMatches, report.Score.Total, report.Score.Accuracy, report.Score.PairwiseAgree, report.Score.PairwiseTotal, report.Score.PairwiseScore)
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
	return out, nil
}

func scoreExpected(segments []checkSegment, expected []int) *checkScore {
	total := len(segments)
	if len(expected) < total {
		total = len(expected)
	}
	exact := 0
	for i := 0; i < total; i++ {
		if segments[i].Speaker == expected[i] {
			exact++
		}
	}
	pairAgree, pairTotal := 0, 0
	for i := 0; i < total; i++ {
		for j := i + 1; j < total; j++ {
			pairTotal++
			predSame := segments[i].Speaker == segments[j].Speaker
			expSame := expected[i] == expected[j]
			if predSame == expSame {
				pairAgree++
			}
		}
	}
	s := &checkScore{Expected: expected, ExactMatches: exact, Total: total, PairwiseAgree: pairAgree, PairwiseTotal: pairTotal}
	if total > 0 {
		s.Accuracy = float64(exact) / float64(total)
	}
	if pairTotal > 0 {
		s.PairwiseScore = float64(pairAgree) / float64(pairTotal)
	}
	return s
}

func loadAudioSamples(path string) ([]float32, int, func(), error) {
	samples, sr, err := audio.WAV(path)
	if err == nil {
		return samples, sr, nil, nil
	}
	if _, lookErr := exec.LookPath("ffmpeg"); lookErr != nil {
		return nil, 0, nil, fmt.Errorf("wav decode failed (%v), and ffmpeg was not found for fallback decode", err)
	}
	tmp := filepath.Join(os.TempDir(), fmt.Sprintf("speakercheck-%d.wav", os.Getpid()))
	cmd := exec.Command("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-ar", "16000", "-ac", "1", tmp)
	if out, runErr := cmd.CombinedOutput(); runErr != nil {
		return nil, 0, nil, fmt.Errorf("ffmpeg decode: %v: %s", runErr, strings.TrimSpace(string(out)))
	}
	cleanup := func() { _ = os.Remove(tmp) }
	samples, sr, err = audio.WAV(tmp)
	if err != nil {
		cleanup()
		return nil, 0, nil, fmt.Errorf("decoded wav: %w", err)
	}
	return samples, sr, cleanup, nil
}

func sliceSamples(samples []float32, sampleRate int, startSec, durationSec float64) []float32 {
	if len(samples) == 0 || sampleRate <= 0 {
		return samples
	}
	start := int(startSec * float64(sampleRate))
	if start < 0 {
		start = 0
	}
	if start > len(samples) {
		start = len(samples)
	}
	end := len(samples)
	if durationSec > 0 {
		end = start + int(durationSec*float64(sampleRate))
		if end > len(samples) {
			end = len(samples)
		}
	}
	return samples[start:end]
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
