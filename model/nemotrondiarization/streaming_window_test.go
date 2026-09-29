package nemotrondiarization

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedStreamingWindowTwoChunksPyTorchParity(t *testing.T) {
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
	window := &StreamingWindow{Tower: tower, Head: head, Cache: cache}
	for step := 0; step < 2; step++ {
		cached := step * 9
		refInput := readStackingFixture(t, fmt.Sprintf("testdata/jfk_stream_step%d_input.f32.gz", step), (cached+13)*projectedWidth)
		chunk := append([]float32(nil), refInput[cached*projectedWidth:]...)
		initial := append([]float32(nil), chunk...)
		input, logits, err := window.ForwardPrepared(chunk, 9, 4)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(input, refInput) {
			t.Fatalf("step=%d prepared input differs", step)
		}
		ref := readStackingFixture(t, fmt.Sprintf("testdata/jfk_stream_step%d_logits.f32.gz", step), len(logits))
		var maxAbs, sumAbs float64
		var outside int
		for i, value := range logits {
			delta := math.Abs(float64(value - ref[i]))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
				outside++
			}
		}
		t.Logf("step=%d logits max_abs=%g mean_abs=%g outside=%d", step, maxAbs, sumAbs/float64(len(logits)), outside)
		if outside != 0 || sumAbs/float64(len(logits)) > 1e-5 {
			t.Fatalf("step=%d logits differ from PyTorch", step)
		}
		_, _, fifo, _ := cache.Snapshot()
		fifoWant := readStackingFixture(t, fmt.Sprintf("testdata/jfk_stream_step%d_fifo.f32.gz", step), (step+1)*9*projectedWidth)
		if !reflect.DeepEqual(fifo, fifoWant) {
			t.Fatalf("step=%d FIFO differs", step)
		}
		if !reflect.DeepEqual(chunk, initial) {
			t.Fatal("mutated prepared chunk")
		}
	}
}

func TestStreamingWindowRejectsMalformed(t *testing.T) {
	if _, _, err := (*StreamingWindow)(nil).ForwardPrepared(make([]float32, 13*projectedWidth), 9, 4); err == nil {
		t.Fatal("accepted nil window")
	}
	m := &StreamingWindow{Tower: &OfflineAudioTower{}, Head: &OfflineHead{}}
	if _, _, err := m.ForwardPrepared(make([]float32, 13*projectedWidth), 9, 4); err == nil {
		t.Fatal("accepted nil cache")
	}
}
