package nemotrondiarization

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

func TestSpeakerCacheFirstOverflowPyTorchParity(t *testing.T) {
	var state SpeakerCacheUncompressed
	for index, frames := range []int{9, 9, 9, 237, 9} {
		lookahead := 4
		cacheFrames := (len(state.speaker) + len(state.fifo)) / projectedWidth
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
		}
		mask := make([]bool, rows)
		for i := range mask {
			mask[i] = index != 2 || i != rows-1
		}
		if err := state.Update(chunk, frames, lookahead, logits, mask); err != nil {
			t.Fatalf("step=%d: %v", index, err)
		}
		speaker, probs, fifo := state.Snapshot()
		fifoWant := readStackingFixture(t, fmt.Sprintf("testdata/cache_step%d_fifo.f32.gz", index), len(fifo))
		if !reflect.DeepEqual(fifo, fifoWant) {
			t.Fatalf("step=%d FIFO differs", index)
		}
		if index >= 3 {
			speakerWant := readStackingFixture(t, fmt.Sprintf("testdata/cache_step%d_speaker.f32.gz", index), len(speaker))
			probsWant := readStackingFixture(t, fmt.Sprintf("testdata/cache_step%d_speaker_probs.f32.gz", index), len(probs))
			if len(speaker) != len(speakerWant) || (len(speaker) != 0 && !reflect.DeepEqual(speaker, speakerWant)) {
				t.Fatalf("step=%d speaker embeddings differ", index)
			}
			var maxAbs float64
			for i, value := range probs {
				maxAbs = math.Max(maxAbs, math.Abs(float64(value-probsWant[i])))
			}
			t.Logf("step=%d speaker=%d fifo=%d probability_max_abs=%g", index, len(speaker)/projectedWidth, len(fifo)/projectedWidth, maxAbs)
			if maxAbs > 2e-6 {
				t.Fatalf("step=%d speaker probabilities differ", index)
			}
		}
	}
}

func TestSpeakerCacheRejectsCompressionWithoutMutation(t *testing.T) {
	state := SpeakerCacheUncompressed{speaker: make([]float32, 222*projectedWidth), speakerProbs: make([]float32, 222*diarizationSpeakers), fifo: make([]float32, 264*projectedWidth)}
	chunk := make([]float32, 222*projectedWidth)
	logits := make([]float32, (222+264+222)*diarizationUpsample*diarizationSpeakers)
	mask := make([]bool, 222+264+222)
	for i := range mask {
		mask[i] = true
	}
	speaker, probs, fifo := state.Snapshot()
	if err := state.Update(chunk, 222, 0, logits, mask); err == nil {
		t.Fatal("accepted unqualified compression")
	}
	a, b, c := state.Snapshot()
	if !reflect.DeepEqual(a, speaker) || !reflect.DeepEqual(b, probs) || !reflect.DeepEqual(c, fifo) {
		t.Fatal("rejected compression changed state")
	}
}
