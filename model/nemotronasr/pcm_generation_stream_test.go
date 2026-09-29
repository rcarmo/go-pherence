package nemotronasr

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func releasedPCMGenerationModel(t *testing.T) *PCMGenerationModel {
	t.Helper()
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	f, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	model, err := LoadPCMGenerationModel(f)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func TestPCMGenerationStreamJFKShort(t *testing.T) {
	model := releasedPCMGenerationModel(t)
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 {
		t.Fatalf("JFK WAV rate=%d err=%v", rate, err)
	}
	for _, size := range []int{397, 4040, 5520} {
		s := &PCMGenerationStream{Model: model}
		var decisions []int
		var positions []int64
		var first []int
		for offset := 0; offset < 13000; {
			end := offset + size
			if end > 13000 {
				end = 13000
			}
			input := append([]float32(nil), pcm[offset:end]...)
			before := append([]float32(nil), input...)
			tokens, frames, err := s.AppendPCM(context.Background(), input)
			if err != nil || !reflect.DeepEqual(input, before) {
				t.Fatalf("size=%d offset=%d err=%v or PCM changed", size, offset, err)
			}
			if len(tokens) != 0 && first == nil {
				first = tokens
			}
			decisions = append(decisions, tokens...)
			positions = append(positions, frames...)
			offset = end
		}
		tokens, frames, err := s.Finish(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		decisions = append(decisions, tokens...)
		positions = append(positions, frames...)
		if len(decisions) != 12 || len(positions) != 12 || !s.closed || s.greedy.frames != 12 || len(first) == 0 {
			t.Fatalf("size=%d decisions=%v positions=%v", size, decisions, positions)
		}
		for i, token := range decisions {
			if token != rnntBlank || positions[i] != int64(i) {
				t.Fatalf("size=%d decision=%d token=%d frame=%d", size, i, token, positions[i])
			}
		}
		if first[0] != rnntBlank {
			t.Fatal("earlier owned decisions changed")
		}
		vocab, err := tokenizer.Load(filepath.Join("..", "..", "checkpoints", "nemotron", "asr", "tokenizer.json"))
		if err != nil {
			t.Fatal(err)
		}
		text, err := DecodeRNNTText(vocab, decisions)
		if err != nil || text != "" {
			t.Fatalf("short recording text=%q err=%v", text, err)
		}
		if _, _, err := s.AppendPCM(context.Background(), pcm[:1]); err == nil {
			t.Fatal("append after finish")
		}
	}
}

func TestPCMGenerationStreamRejectsAndCancels(t *testing.T) {
	model := releasedPCMGenerationModel(t)
	s := &PCMGenerationStream{Model: model}
	for _, pcm := range [][]float32{nil, make([]float32, 80001), {float32(math.NaN())}} {
		if _, _, err := s.AppendPCM(context.Background(), pcm); err == nil || s.frontend.first || len(s.frontend.pending) != 0 || s.closed {
			t.Fatal("invalid PCM changed stream")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := s.AppendPCM(ctx, make([]float32, 1)); err != context.Canceled || !s.closed {
		t.Fatalf("cancellation err=%v closed=%v", err, s.closed)
	}
	if _, _, err := s.AppendPCM(context.Background(), make([]float32, 1)); err == nil {
		t.Fatal("accepted append after cancellation")
	}
	short := &PCMGenerationStream{Model: model}
	if _, _, err := short.AppendPCM(context.Background(), make([]float32, 1)); err != nil {
		t.Fatal(err)
	}
	tokens, frames, err := short.Finish(context.Background())
	if err != nil || len(tokens) != 0 || len(frames) != 0 {
		t.Fatalf("short recording decisions=%v frames=%v err=%v", tokens, frames, err)
	}
}
