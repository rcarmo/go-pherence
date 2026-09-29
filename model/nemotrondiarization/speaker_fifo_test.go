package nemotrondiarization

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

func TestSpeakerFIFOPreCompressionPyTorchParity(t *testing.T) {
	var cache SpeakerFIFO
	for index := 0; index < 3; index++ {
		frames, lookahead := 9, 4
		inputRows := len(cache.fifo)/projectedWidth + frames + lookahead
		ref := readStackingFixture(t, fmt.Sprintf("testdata/cache_step%d_input.f32.gz", index), inputRows*projectedWidth)
		chunk := ref[len(cache.fifo):]
		initial := append([]float32(nil), chunk...)
		got, err := cache.Prepare(chunk, frames, lookahead)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, ref) {
			t.Fatalf("step=%d prepared embeddings differ", index)
		}
		if err := cache.Update(chunk, frames, lookahead); err != nil {
			t.Fatal(err)
		}
		want := readStackingFixture(t, fmt.Sprintf("testdata/cache_step%d_fifo.f32.gz", index), (index+1)*frames*projectedWidth)
		if !reflect.DeepEqual(cache.Snapshot(), want) {
			t.Fatalf("step=%d FIFO differs", index)
		}
		if !reflect.DeepEqual(chunk, initial) {
			t.Fatalf("step=%d mutated caller input", index)
		}
	}
}

func TestSpeakerFIFORejectsWithoutMutation(t *testing.T) {
	var cache SpeakerFIFO
	chunk := make([]float32, 13*projectedWidth)
	if err := cache.Update(chunk, 9, 4); err != nil {
		t.Fatal(err)
	}
	old := cache.Snapshot()
	bad := append([]float32(nil), chunk...)
	bad[0] = float32(math.NaN())
	if err := cache.Update(bad, 9, 4); err == nil {
		t.Fatal("accepted non-finite input")
	}
	if err := cache.Update(chunk, 9, 5); err == nil {
		t.Fatal("accepted unsupported lookahead")
	}
	if err := cache.Update(chunk[:len(chunk)-1], 9, 4); err == nil {
		t.Fatal("accepted short chunk")
	}
	if !reflect.DeepEqual(cache.Snapshot(), old) {
		t.Fatal("rejected update changed FIFO")
	}
	cache.fifo = make([]float32, 260*projectedWidth)
	old = cache.Snapshot()
	if err := cache.Update(chunk, 9, 4); err == nil {
		t.Fatal("accepted unqualified compression")
	}
	if !reflect.DeepEqual(cache.Snapshot(), old) {
		t.Fatal("overflow changed FIFO")
	}
}
