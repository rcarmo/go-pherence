// Command speakercheck validates the speaker diarization path without running Whisper.
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/models/speaker"
)

func main() {
	input := flag.String("input", "", "Input WAV file")
	modelPath := flag.String("speaker-model", "models/speaker-ecapa-voxceleb.safetensors", "Converted SpeechBrain ECAPA safetensors model")
	threshold := flag.Float64("threshold", 0.3, "Cosine similarity threshold for agglomerative clustering")
	context := flag.Float64("context", 0.5, "Embedding context padding around VAD segments in seconds")
	startSec := flag.Float64("start", 0, "Start offset in seconds for spot checks")
	durationSec := flag.Float64("duration", 0, "Optional duration in seconds for spot checks")
	showSims := flag.Bool("sims", true, "Print pairwise cosine similarities")
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

	counts := map[int]int{}
	for _, label := range labels {
		counts[label]++
	}
	keys := make([]int, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	fmt.Printf("segments=%d speakers=%d threshold=%.2f context=%.2fs start=%.2fs duration=%.2fs\n", len(vad), len(keys), *threshold, *context, *startSec, float64(len(samples))/float64(sr))
	for i, seg := range vad {
		fmt.Printf("%02d %.2f-%.2f speaker=%d duration=%.2fs\n", i, seg.Start+*startSec, seg.End+*startSec, labels[i]+1, seg.End-seg.Start)
	}
	fmt.Print("counts")
	for _, k := range keys {
		fmt.Printf(" speaker%d=%d", k+1, counts[k])
	}
	fmt.Println()
	if *showSims {
		for i := 0; i < len(embeddings); i++ {
			for j := i + 1; j < len(embeddings); j++ {
				fmt.Printf("sim %02d-%02d %.3f\n", i, j, speaker.CosineSimilarity(embeddings[i], embeddings[j]))
			}
		}
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
