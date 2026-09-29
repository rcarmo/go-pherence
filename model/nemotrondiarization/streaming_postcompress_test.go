package nemotrondiarization

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// This seeds the independently checked synthetic cache_step5 compressed
// state; it does not exercise the preceding oversized model window.
func TestReleasedStreamingWindowAfterCompressionPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	tower, err := LoadOfflineAudioTower(file)
	if err != nil {
		t.Fatal(err)
	}
	head, err := LoadOfflineHead(file)
	if err != nil {
		t.Fatal(err)
	}
	compressor, err := LoadSpeakerCompressor(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	cache, err := NewSpeakerCache(compressor)
	if err != nil {
		t.Fatal(err)
	}
	cache.speaker = readStackingFixture(t, "testdata/cache_step5_speaker.f32.gz", 264*projectedWidth)
	cache.speakerProbs = readStackingFixture(t, "testdata/cache_step5_speaker_probs.f32.gz", 264*diarizationSpeakers)
	cache.fifo = readStackingFixture(t, "testdata/cache_step5_fifo.f32.gz", 51*projectedWidth)
	cache.compressed = true
	window := &StreamingWindow{Tower: tower, Head: head, Cache: cache}
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	for step := 0; step < 2; step++ {
		chunk := append([]float32(nil), stacked[step*9*projectedWidth:(step*9+13)*projectedWidth]...)
		initial := append([]float32(nil), chunk...)
		input, logits, err := window.ForwardPrepared(chunk, 9, 4)
		if err != nil {
			t.Fatal(err)
		}
		prefix := fmt.Sprintf("testdata/jfk_postcompress_step%d_", step)
		wantInput := readStackingFixture(t, prefix+"input.f32.gz", (328+9*step)*projectedWidth)
		if !reflect.DeepEqual(input, wantInput) {
			t.Fatalf("step=%d prepared input differs", step)
		}
		wantLogits := readStackingFixture(t, prefix+"logits.f32.gz", (328+9*step)*diarizationUpsample*diarizationSpeakers)
		var maxAbs, sumAbs float64
		var outside int
		for i, value := range logits {
			delta := math.Abs(float64(value - wantLogits[i]))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(wantLogits[i])) {
				outside++
			}
		}
		mean := sumAbs / float64(len(logits))
		t.Logf("step=%d logits max_abs=%g mean_abs=%g outside=%d", step, maxAbs, mean, outside)
		if outside != 0 || mean > 1e-5 {
			t.Fatalf("step=%d logits differ from PyTorch", step)
		}
		speaker, probs, fifo, compressed := cache.Snapshot()
		if !compressed {
			t.Fatalf("step=%d lost compressed state", step)
		}
		for _, item := range []struct {
			name   string
			got    []float32
			length int
		}{
			{"speaker", speaker, 264 * projectedWidth},
			{"speaker_probs", probs, 264 * diarizationSpeakers},
			{"fifo", fifo, (60 + 9*step) * projectedWidth},
		} {
			want := readStackingFixture(t, prefix+item.name+".f32.gz", item.length)
			if !reflect.DeepEqual(item.got, want) {
				t.Fatalf("step=%d %s differs", step, item.name)
			}
		}
		if !reflect.DeepEqual(chunk, initial) {
			t.Fatal("mutated prepared chunk")
		}
	}
}
