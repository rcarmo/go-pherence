package nemotrondiarization

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// The seeded 264-speaker/264-FIFO cache exercises the largest prepared 9+4
// window. It does not represent an audio-scheduled continuous stream.
func TestReleasedStreamingWindowMaximumCachePyTorchParity(t *testing.T) {
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
	cache.speakerProbs = readStackingFixture(t, "testdata/jfk_maxcache_seed_speaker_probs.f32.gz", 264*diarizationSpeakers)
	cache.fifo = readStackingFixture(t, "testdata/jfk_maxcache_seed_fifo.f32.gz", 264*projectedWidth)
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
		prefix := fmt.Sprintf("testdata/jfk_maxcache_step%d_", step)
		rows := maxPreparedDiarizationRows
		if step == 1 {
			rows = 328
		}
		wantInput := readStackingFixture(t, prefix+"input.f32.gz", rows*projectedWidth)
		if !reflect.DeepEqual(input, wantInput) {
			t.Fatalf("step=%d prepared input differs", step)
		}
		wantLogits := readStackingFixture(t, prefix+"logits.f32.gz", rows*diarizationUpsample*diarizationSpeakers)
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
		t.Logf("step=%d rows=%d max_abs=%g mean_abs=%g outside=%d", step, rows, maxAbs, mean, outside)
		if outside != 0 || mean > 2e-5 {
			t.Fatalf("step=%d logits differ from PyTorch", step)
		}
		mask := make([]bool, rows)
		for i := range mask {
			mask[i] = true
		}
		pooled, err := PoolSpeakerProbabilities(logits, mask)
		if err != nil {
			t.Fatal(err)
		}
		pooledRef := readStackingFixture(t, prefix+"pooled.f32.gz", rows*diarizationSpeakers)
		var pooledMax, pooledMean float64
		for i, v := range pooled {
			d := math.Abs(float64(v - pooledRef[i]))
			pooledMax = math.Max(pooledMax, d)
			pooledMean += d
		}
		pooledMean /= float64(len(pooled))
		t.Logf("step=%d pooled max_abs=%g mean_abs=%g", step, pooledMax, pooledMean)
		if pooledMax > 2.5e-5 || pooledMean > 2e-6 {
			t.Fatalf("step=%d pooled probabilities differ from PyTorch", step)
		}
		speaker, probs, fifo, compressed := cache.Snapshot()
		if !compressed {
			t.Fatal("lost compressed flag")
		}
		for _, item := range []struct {
			name   string
			got    []float32
			length int
		}{
			{"speaker", speaker, 264 * projectedWidth},
			{"speaker_probs", probs, 264 * diarizationSpeakers},
			{"fifo", fifo, (51 + 9*step) * projectedWidth},
		} {
			want := readStackingFixture(t, prefix+item.name+".f32.gz", item.length)
			if item.name != "speaker_probs" {
				if !reflect.DeepEqual(item.got, want) {
					for i := range item.got {
						if item.got[i] != want[i] {
							t.Logf("step=%d %s first mismatch index=%d row=%d got=%g want=%g", step, item.name, i, i/projectedWidth, item.got[i], want[i])
							break
						}
					}
					t.Fatalf("step=%d %s differs", step, item.name)
				}
			} else {
				var maxDelta, sumDelta float64
				var bad, firstBad int
				for i, value := range item.got {
					delta := math.Abs(float64(value - want[i]))
					maxDelta = math.Max(maxDelta, delta)
					sumDelta += delta
					// Full-tower logits differ numerically from PyTorch; pooling
					// eight sigmoids compounds that error. Exact speaker/FIFO
					// selection above still guards downstream cache order.
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 2.5e-5 {
						if bad == 0 {
							firstBad = i
						}
						bad++
					}
				}
				t.Logf("step=%d speaker probs max_abs=%g mean_abs=%g outside=%d first=%d", step, maxDelta, sumDelta/float64(len(item.got)), bad, firstBad)
				if bad > 0 || sumDelta/float64(len(item.got)) > 1e-6 {
					t.Fatalf("step=%d speaker probability %d differs got=%g want=%g", step, firstBad, item.got[firstBad], want[firstBad])
				}
			}
		}
		if !reflect.DeepEqual(chunk, initial) {
			t.Fatal("mutated prepared chunk")
		}
	}
}
