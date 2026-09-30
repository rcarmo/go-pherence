package nemotronasr

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

// Opt-in matched CPU experiment. This records reproducible decisions rather
// than relying on a benchmark's decision count as evidence of text parity.
func TestPCMGenerationOptimizationTrial(t *testing.T) {
	output := os.Getenv("GO_PHERENCE_NEMOTRON_OPT_RESULT")
	if output == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_OPT_RESULT")
	}
	model := releasedPCMGenerationModel(t)
	vocab, err := tokenizer.Load(os.Getenv("GO_PHERENCE_NEMOTRON_ASR_TOKENIZER"))
	if err != nil {
		t.Fatal(err)
	}
	jfk, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 || len(jfk) != 176000 {
		t.Fatal(rate, len(jfk), err)
	}
	podcast, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "podcast.wav"))
	if err != nil || rate != 16000 || len(podcast) < 320*rate {
		t.Fatal(rate, len(podcast), err)
	}
	podcast = podcast[300*rate : 320*rate]
	type result struct {
		Name    string
		Seconds float64
		Tokens  []int
		Frames  []int64
		Text    string
	}
	results := []result{}
	for _, clip := range []struct {
		name string
		pcm  []float32
	}{{"jfk", jfk}, {"podcast", podcast}, {"jfk-repeat", jfk}} {
		stream := &PCMGenerationStream{Model: model}
		r := result{Name: clip.name}
		start := time.Now()
		for offset := 0; offset < len(clip.pcm); offset += 80000 {
			tokens, frames, err := stream.AppendPCM(context.Background(), clip.pcm[offset:min(offset+80000, len(clip.pcm))])
			if err != nil {
				t.Fatal(err)
			}
			r.Tokens = append(r.Tokens, tokens...)
			r.Frames = append(r.Frames, frames...)
		}
		tokens, frames, err := stream.Finish(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		r.Tokens = append(r.Tokens, tokens...)
		r.Frames = append(r.Frames, frames...)
		r.Text, err = DecodeRNNTText(vocab, r.Tokens)
		if err != nil {
			t.Fatal(err)
		}
		r.Seconds = time.Since(start).Seconds()
		results = append(results, r)
		t.Logf("%s %.3fs decisions=%d text=%s", r.Name, r.Seconds, len(r.Tokens), r.Text)
	}
	data, err := json.MarshalIndent(struct {
		Threads int
		Results []result
	}{runtime.GOMAXPROCS(0), results}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(output, append(data, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
}
