package nemotrondiarization

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedSpeakerCacheCompressedStepsPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
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
	state, err := NewSpeakerCache(compressor)
	if err != nil {
		t.Fatal(err)
	}
	for index, frames := range []int{9, 9, 9, 237, 9, 222, 9} {
		lookahead := 4
		beforeSpeaker, _, beforeFIFO, _ := state.Snapshot()
		cacheFrames := (len(beforeSpeaker) + len(beforeFIFO)) / projectedWidth
		rows := cacheFrames + frames + lookahead
		ref := readStackingFixture(t, fmt.Sprintf("testdata/cache_step%d_input.f32.gz", index), rows*projectedWidth)
		chunk := append([]float32(nil), ref[cacheFrames*projectedWidth:]...)
		prepared, err := state.Prepare(chunk, frames, lookahead)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(prepared, ref) {
			t.Fatalf("step=%d prepared embeddings differ", index)
		}
		logits := make([]float32, rows*diarizationUpsample*diarizationSpeakers)
		for i := range logits {
			logits[i] = float32(i%53-26) / 11
			if index >= 5 {
				logits[i] += float32(i/(diarizationUpsample*diarizationSpeakers))*0.00017 + float32(i%diarizationSpeakers)*0.000037
			}
		}
		mask := make([]bool, rows)
		for i := range mask {
			mask[i] = index != 2 || i != rows-1
		}
		if err := state.Update(chunk, frames, lookahead, logits, mask); err != nil {
			t.Fatalf("step=%d: %v", index, err)
		}
		speaker, probs, fifo, compressed := state.Snapshot()
		fifoWant := readStackingFixture(t, fmt.Sprintf("testdata/cache_step%d_fifo.f32.gz", index), len(fifo))
		if !reflect.DeepEqual(fifo, fifoWant) {
			t.Fatalf("step=%d FIFO differs", index)
		}
		if index >= 3 {
			speakerWant := readStackingFixture(t, fmt.Sprintf("testdata/cache_step%d_speaker.f32.gz", index), len(speaker))
			probsWant := readStackingFixture(t, fmt.Sprintf("testdata/cache_step%d_speaker_probs.f32.gz", index), len(probs))
			if len(speaker) != len(speakerWant) || (len(speaker) != 0 && !reflect.DeepEqual(speaker, speakerWant)) {
				first := -1
				for i := 0; i < min(len(speaker), len(speakerWant)); i++ {
					if speaker[i] != speakerWant[i] {
						first = i
						break
					}
				}
				if first >= 0 {
					t.Fatalf("step=%d speaker embeddings differ at %d: %g want %g", index, first, speaker[first], speakerWant[first])
				}
				t.Fatalf("step=%d speaker embedding lengths differ %d want %d", index, len(speaker), len(speakerWant))
			}
			var maxAbs float64
			for i, value := range probs {
				maxAbs = math.Max(maxAbs, math.Abs(float64(value-probsWant[i])))
			}
			t.Logf("step=%d speaker=%d fifo=%d compressed=%t probability_max_abs=%g", index, len(speaker)/projectedWidth, len(fifo)/projectedWidth, compressed, maxAbs)
			if maxAbs > 2e-6 {
				t.Fatalf("step=%d speaker probabilities differ", index)
			}
		}
		if compressed != (index >= 5) {
			t.Fatalf("step=%d compressed=%t", index, compressed)
		}
	}
}

func TestSpeakerCacheRejectsMalformedWithoutMutation(t *testing.T) {
	state, err := NewSpeakerCache(&SpeakerCompressor{silence: make([]float32, projectedWidth)})
	if err != nil {
		t.Fatal(err)
	}
	chunk := make([]float32, 13*projectedWidth)
	logits := make([]float32, 13*diarizationUpsample*diarizationSpeakers)
	mask := make([]bool, 13)
	for i := range mask {
		mask[i] = true
	}
	if err := state.Update(chunk, 9, 4, logits, mask); err != nil {
		t.Fatal(err)
	}
	a, b, c, d := state.Snapshot()
	chunk[0] = float32(math.NaN())
	if err := state.Update(chunk, 9, 4, logits, mask); err == nil {
		t.Fatal("accepted non-finite chunk")
	}
	x, y, z, e := state.Snapshot()
	if !reflect.DeepEqual(a, x) || !reflect.DeepEqual(b, y) || !reflect.DeepEqual(c, z) || d != e {
		t.Fatal("rejected update changed state")
	}
}
