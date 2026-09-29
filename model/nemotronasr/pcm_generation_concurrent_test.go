package nemotronasr

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
)

// Opt-in shared-weight stress: each request owns its cache, frontend and
// returned decisions. It tests concurrent use, not sustained throughput.
func TestReleasedPCMGenerationSharedModelConcurrent(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_CONCURRENT") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_CONCURRENT=1")
	}
	model := releasedPCMGenerationModel(t)
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 || len(pcm) < 80000 {
		t.Fatalf("JFK input rate=%d samples=%d err=%v", rate, len(pcm), err)
	}
	pcm = pcm[:80000]
	type result struct {
		tokens []int
		frames []int64
		err    error
	}
	run := func(size int) result {
		stream := &PCMGenerationStream{Model: model}
		var got result
		var owned []int
		var saved []int
		for start := 0; start < len(pcm); start += size {
			end := min(start+size, len(pcm))
			part, frames, err := stream.AppendPCM(context.Background(), pcm[start:end])
			if err != nil {
				got.err = err
				return got
			}
			if owned == nil && len(part) > 0 {
				owned, saved = part, append([]int(nil), part...)
			}
			got.tokens = append(got.tokens, part...)
			got.frames = append(got.frames, frames...)
		}
		part, frames, err := stream.Finish(context.Background())
		got.err = err
		got.tokens = append(got.tokens, part...)
		got.frames = append(got.frames, frames...)
		if len(owned) == 0 || !reflect.DeepEqual(owned, saved) {
			got.err = fmt.Errorf("previously returned decisions absent or changed")
		}
		return got
	}
	sizes := [2]int{4040, 80000}
	var serial, parallel [2]result
	for i, size := range sizes {
		serial[i] = run(size)
		if serial[i].err != nil || len(serial[i].tokens) == 0 || len(serial[i].tokens) != len(serial[i].frames) {
			t.Fatalf("serial size=%d decisions=%d err=%v", size, len(serial[i].tokens), serial[i].err)
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
		if parallel[i].err != nil || !reflect.DeepEqual(parallel[i].tokens, serial[i].tokens) || !reflect.DeepEqual(parallel[i].frames, serial[i].frames) {
			t.Fatalf("shared-model concurrency differs size=%d decisions=%d err=%v", size, len(parallel[i].tokens), parallel[i].err)
		}
		t.Logf("shared-model size=%d decisions=%d frames=%d", size, len(parallel[i].tokens), len(parallel[i].frames))
	}
}
