// Command speakercheck validates the speaker diarization path without running Whisper.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"

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

func main() {
	input := flag.String("input", "", "Input WAV file")
	modelPath := flag.String("speaker-model", "models/speaker-ecapa-voxceleb.safetensors", "Converted SpeechBrain ECAPA safetensors model")
	threshold := flag.Float64("threshold", 0.3, "Cosine similarity threshold for agglomerative clustering")
	context := flag.Float64("context", 0.5, "Embedding context padding around VAD segments in seconds")
	startSec := flag.Float64("start", 0, "Start offset in seconds for spot checks")
	durationSec := flag.Float64("duration", 0, "Optional duration in seconds for spot checks")
	showSims := flag.Bool("sims", true, "Print pairwise cosine similarities")
	jsonOut := flag.Bool("json", false, "Emit machine-readable JSON")
	flag.Parse()
	if *input == "" {
		fmt.Fprintln(os.Stderr, "-input is required")
		os.Exit(2)
	}

	samples, sr, err := audio.WAV(*input)
	if err != nil {
		fatalf("wav: %v", err)
	}
	if sr != 16000 {
		samples = audio.ResampleSinc(samples, sr, 16000)
		sr = 16000
	}
	samples = sliceSamples(samples, sr, *startSec, *durationSec)
	model, err := speaker.LoadSpeechBrainECAPASafetensors(*modelPath)
	if err != nil {
		fatalf("speaker model: %v", err)
	}
	vad := speaker.EnergyVAD(samples, sr, 25, 10, 0)
	embeddings := speaker.ExtractSpeechBrainEmbeddingsWithContext(samples, sr, vad, model, *context)
	labels := speaker.AgglomerativeCluster(embeddings, float32(*threshold))
	labels = speaker.SmoothSingletonLabels(labels, embeddings, 0.4)

	report := buildReport(*input, *modelPath, *threshold, *context, *startSec, float64(len(samples))/float64(sr), vad, labels, embeddings, *showSims)
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fatalf("json: %v", err)
		}
		return
	}
	printTextReport(report)
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
	for _, sim := range report.Sims {
		fmt.Printf("sim %02d-%02d %.3f\n", sim.I, sim.J, sim.Cosine)
	}
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
