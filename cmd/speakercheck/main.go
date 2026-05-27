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
	model, err := speaker.LoadSpeechBrainECAPASafetensors(*modelPath)
	if err != nil {
		fatalf("speaker model: %v", err)
	}
	vad := speaker.EnergyVAD(samples, sr, 25, 10, 0)
	embeddings := speaker.ExtractSpeechBrainEmbeddingsWithContext(samples, sr, vad, model, *context)
	labels := speaker.AgglomerativeCluster(embeddings, float32(*threshold))
	labels = smoothSingletons(labels, embeddings, 0.4)

	counts := map[int]int{}
	for _, label := range labels {
		counts[label]++
	}
	keys := make([]int, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	fmt.Printf("segments=%d speakers=%d threshold=%.2f context=%.2fs\n", len(vad), len(keys), *threshold, *context)
	for i, seg := range vad {
		fmt.Printf("%02d %.2f-%.2f speaker=%d duration=%.2fs\n", i, seg.Start, seg.End, labels[i]+1, seg.End-seg.Start)
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

func smoothSingletons(labels []int, embeddings [][]float32, minAvgSim float32) []int {
	if len(labels) == 0 || len(labels) != len(embeddings) {
		return labels
	}
	out := append([]int(nil), labels...)
	for pass := 0; pass < len(out); pass++ {
		changed := false
		counts := map[int]int{}
		for _, label := range out {
			counts[label]++
		}
		for i, label := range out {
			if counts[label] != 1 {
				continue
			}
			bestLabel := label
			bestSim := float32(-2)
			for candidate, count := range counts {
				if candidate == label || count == 0 {
					continue
				}
				var sum float32
				var n int
				for j, other := range out {
					if other != candidate || i == j {
						continue
					}
					sum += speaker.CosineSimilarity(embeddings[i], embeddings[j])
					n++
				}
				if n == 0 {
					continue
				}
				avg := sum / float32(n)
				if avg > bestSim {
					bestSim = avg
					bestLabel = candidate
				}
			}
			if bestLabel != label && bestSim >= minAvgSim {
				out[i] = bestLabel
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	return renumber(out)
}

func renumber(labels []int) []int {
	remap := map[int]int{}
	next := 0
	out := make([]int, len(labels))
	for i, label := range labels {
		mapped, ok := remap[label]
		if !ok {
			mapped = next
			remap[label] = mapped
			next++
		}
		out[i] = mapped
	}
	return out
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
