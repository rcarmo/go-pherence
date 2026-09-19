package speaker

import (
	"os"
	"runtime"
	"strconv"
	"sync"
)

// ExtractSpeechBrainEmbeddings computes SpeechBrain ECAPA embeddings for VAD
// segments using SpeechBrain-compatible Fbank preprocessing.
func ExtractSpeechBrainEmbeddings(samples []float32, sampleRate int, segments []VADSegment, ecapa *SpeechBrainECAPA) [][]float32 {
	return ExtractSpeechBrainEmbeddingsWithContext(samples, sampleRate, segments, ecapa, 0.5)
}

// ExtractSpeechBrainEmbeddingsWithContext computes SpeechBrain ECAPA embeddings
// with optional context padding around each VAD segment. Short segments produce
// unstable speaker embeddings; a small amount of surrounding audio improves
// clustering while keeping transcript cue boundaries unchanged.
func ExtractSpeechBrainEmbeddingsWithContext(samples []float32, sampleRate int, segments []VADSegment, ecapa *SpeechBrainECAPA, contextSec float64) [][]float32 {
	if ecapa == nil || len(segments) == 0 {
		return nil
	}
	embeddings := make([][]float32, len(segments))
	pad := int(contextSec * float64(sampleRate))
	if pad < 0 {
		pad = 0
	}

	// Each segment's ECAPA forward is independent; the per-forward GEMMs are
	// single-threaded, so fan segments across cores to use the K3 fully. Capped
	// to avoid the all-core RVV brown-out the board exhibits under sustained load.
	one := func(i int) {
		seg := segments[i]
		startSample := int(seg.Start*float64(sampleRate)) - pad
		endSample := int(seg.End*float64(sampleRate)) + pad
		if startSample < 0 {
			startSample = 0
		}
		if endSample > len(samples) {
			endSample = len(samples)
		}
		if startSample >= endSample {
			embeddings[i] = make([]float32, 192)
			return
		}
		features := SpeechBrainFbank(samples[startSample:endSample], sampleRate)
		if len(features) != 80 || len(features[0]) == 0 {
			embeddings[i] = make([]float32, 192)
			return
		}
		frames := len(features[0])
		flat := make([]float32, 80*frames)
		for m := 0; m < 80; m++ {
			copy(flat[m*frames:], features[m])
		}
		embeddings[i] = ecapa.Embed(flat, frames)
	}

	nw := speakerWorkers()
	if nw <= 1 || len(segments) == 1 {
		for i := range segments {
			one(i)
		}
		return embeddings
	}
	var wg sync.WaitGroup
	jobs := make(chan int)
	for w := 0; w < nw; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				one(i)
			}
		}()
	}
	for i := range segments {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	return embeddings
}

// speakerWorkers bounds segment-level parallelism. Defaults to min(GOMAXPROCS, 6);
// the K3 reboots under sustained all-8-core RVV load. Override with SPEAKER_THREADS.
func speakerWorkers() int {
	if v := os.Getenv("SPEAKER_THREADS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	n := runtime.GOMAXPROCS(0)
	if n > 6 {
		n = 6
	}
	if n < 1 {
		n = 1
	}
	return n
}
