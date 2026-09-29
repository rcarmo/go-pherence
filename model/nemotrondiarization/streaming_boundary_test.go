package nemotrondiarization

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedStreamingWindowFirstSpeakerTransferPyTorchParity(t *testing.T) {
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
	cache.fifo = readStackingFixture(t, "testdata/cache_step3_fifo.f32.gz", 264*projectedWidth)
	window := &StreamingWindow{Tower: tower, Head: head, Cache: cache}
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	for step := 0; step < 2; step++ {
		chunk := append([]float32(nil), stacked[step*9*projectedWidth:(step*9+13)*projectedWidth]...)
		initial := append([]float32(nil), chunk...)
		input, logits, err := window.ForwardPrepared(chunk, 9, 4)
		if err != nil {
			t.Fatal(err)
		}
		refInput := readStackingFixture(t, fmt.Sprintf("testdata/jfk_boundary_step%d_input.f32.gz", step), len(input))
		if !reflect.DeepEqual(input, refInput) {
			t.Fatalf("step=%d prepared input differs", step)
		}
		refLogits := readStackingFixture(t, fmt.Sprintf("testdata/jfk_boundary_step%d_logits.f32.gz", step), len(logits))
		var maxAbs, sumAbs float64
		var outside int
		for i, value := range logits {
			delta := math.Abs(float64(value - refLogits[i]))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(refLogits[i])) {
				outside++
			}
		}
		mean := sumAbs / float64(len(logits))
		t.Logf("step=%d logits max_abs=%g mean_abs=%g outside=%d", step, maxAbs, mean, outside)
		if outside != 0 || mean > 1e-5 {
			t.Fatalf("step=%d logits differ from PyTorch", step)
		}
		speaker, probs, fifo, compressed := cache.Snapshot()
		if compressed {
			t.Fatalf("step=%d unexpectedly compressed", step)
		}
		for _, item := range []struct {
			name string
			got  []float32
		}{{"speaker", speaker}, {"fifo", fifo}} {
			want := readStackingFixture(t, fmt.Sprintf("testdata/jfk_boundary_step%d_%s.f32.gz", step, item.name), len(item.got))
			if !reflect.DeepEqual(item.got, want) {
				t.Fatalf("step=%d %s differs", step, item.name)
			}
		}
		refProbs := readStackingFixture(t, fmt.Sprintf("testdata/jfk_boundary_step%d_speaker_probs.f32.gz", step), len(probs))
		var probMax float64
		for i, value := range probs {
			probMax = math.Max(probMax, math.Abs(float64(value-refProbs[i])))
		}
		t.Logf("step=%d speaker=%d fifo=%d probability_max_abs=%g", step, len(speaker)/projectedWidth, len(fifo)/projectedWidth, probMax)
		if probMax > 2e-6 {
			t.Fatalf("step=%d speaker probabilities differ", step)
		}
		if !reflect.DeepEqual(chunk, initial) {
			t.Fatal("mutated chunk")
		}
	}
}
