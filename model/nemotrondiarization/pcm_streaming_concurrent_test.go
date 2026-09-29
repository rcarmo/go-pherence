package nemotrondiarization

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Opt-in shared-weight stress. Each request owns a fresh frontend and speaker
// cache; neither the public output nor the mutable per-stream state is shared.
func TestReleasedPCMStreamingSharedModelConcurrent(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_CONCURRENT") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_CONCURRENT=1")
	}
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	loaded, loadErr := LoadPCMStreamingRequest(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	// The loaded compressor was validated above; each worker owns its cache.
	newRequest := func() *PCMStreamingRequest {
		return &PCMStreamingRequest{
			frontend: &PCMStackingStream{stack: StackingStream{Projection: loaded.frontend.stack.Projection}},
			window:   &StreamingWindow{Tower: loaded.window.Tower, Head: loaded.window.Head, Cache: &SpeakerCache{compressor: loaded.window.Cache.compressor}},
		}
	}
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 || len(pcm) < 32000 {
		t.Fatalf("JFK input rate=%d samples=%d err=%v", rate, len(pcm), err)
	}
	pcm = pcm[:32000]
	type result struct {
		logits []float32
		err    error
	}
	run := func(size int) result {
		s := newRequest()
		var got result
		var owned, saved []float32
		for start := 0; start < len(pcm); start += size {
			end := min(start+size, len(pcm))
			part, err := s.AppendPCM(pcm[start:end])
			if err != nil {
				got.err = err
				return got
			}
			if owned == nil && len(part) > 0 {
				owned, saved = part, append([]float32(nil), part...)
			}
			got.logits = append(got.logits, part...)
		}
		part, err := s.Finish()
		got.err = err
		got.logits = append(got.logits, part...)
		if len(owned) == 0 || !reflect.DeepEqual(owned, saved) {
			got.err = fmt.Errorf("previously returned logits absent or changed")
		}
		return got
	}
	sizes := [2]int{7979, 32000}
	var serial, parallel [2]result
	for i, size := range sizes {
		serial[i] = run(size)
		if serial[i].err != nil || len(serial[i].logits) != 199*diarizationSpeakers {
			t.Fatalf("serial size=%d logits=%d err=%v", size, len(serial[i].logits), serial[i].err)
		}
	}
	var workers sync.WaitGroup
	start := make(chan struct{})
	for i, size := range sizes {
		workers.Add(1)
		go func(i, size int) {
			defer workers.Done()
			<-start
			parallel[i] = run(size)
		}(i, size)
	}
	close(start)
	workers.Wait()
	for i, size := range sizes {
		if parallel[i].err != nil || !reflect.DeepEqual(parallel[i].logits, serial[i].logits) {
			t.Fatalf("shared-model concurrency differs size=%d logits=%d err=%v", size, len(parallel[i].logits), parallel[i].err)
		}
		t.Logf("shared-model size=%d logits_rows=%d", size, len(parallel[i].logits)/diarizationSpeakers)
	}
}
