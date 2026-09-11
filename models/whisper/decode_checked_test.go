package whisper

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"
)

// Complete synthetic vocabulary geometry; this is not a model/tokenizer fixture.
func checkedTestTokenizer(size int) *Tokenizer {
	vocab := make(map[int]string, size)
	for i := 0; i < size; i++ {
		vocab[i] = fmt.Sprintf("text%d", i)
	}
	shift := size - 51865
	for id, text := range map[int]string{50257: "<|endoftext|>", 50258: "<|startoftranscript|>", 50259: "<|en|>", 50267: "<|pt|>", 50358 + shift: "<|translate|>", 50359 + shift: "<|transcribe|>", 50363 + shift: "<|notimestamps|>"} {
		vocab[id] = text
	}
	for i := 0; i <= 1500; i++ {
		vocab[50364+shift+i] = fmt.Sprintf("<|%.2f|>", float64(i)/50)
	}
	vocab[42] = "hello"
	return &Tokenizer{Vocab: vocab, VocabSize: size}
}

func TestCheckedTimestampVocabulary(t *testing.T) {
	for _, cfg := range []Config{Tiny(), LargeV3Turbo()} {
		tok := checkedTestTokenizer(cfg.VocabSize)
		v, err := checkedTimestampVocabulary(cfg, tok, "pt")
		if err != nil {
			t.Fatal(err)
		}
		if v.transcribe != cfg.VocabSize-1506 || v.timestampBegin != cfg.VocabSize-1501 || v.language != 50267 {
			t.Fatal(v)
		}
		for _, lang := range []string{"", "unknown", "transcribe", "translate", "0.00", "endoftext"} {
			if _, err := checkedTimestampVocabulary(cfg, tok, lang); err == nil {
				t.Fatalf("accepted language %q", lang)
			}
		}
		delete(tok.Vocab, 42)
		if _, err := checkedTimestampVocabulary(cfg, tok, "pt"); err == nil {
			t.Fatal("missing text token accepted")
		}
		tok = checkedTestTokenizer(cfg.VocabSize)
		tok.Vocab[42] = "<|pt|>"
		if _, err := checkedTimestampVocabulary(cfg, tok, "pt"); err == nil {
			t.Fatal("duplicate language accepted")
		}
		tok = checkedTestTokenizer(cfg.VocabSize)
		tok.Vocab[v.timestampBegin+1] = "wrong"
		if _, err := checkedTimestampVocabulary(cfg, tok, "pt"); err == nil {
			t.Fatal("wrong timestamp accepted")
		}
	}
}

func TestCheckedTimestampMaskPairingAndMass(t *testing.T) {
	cfg := LargeV3Turbo()
	v, err := checkedTimestampVocabulary(cfg, checkedTestTokenizer(cfg.VocabSize), "pt")
	if err != nil {
		t.Fatal(err)
	}
	scores := func() []float32 {
		x := make([]float32, cfg.VocabSize)
		for i := range x {
			x[i] = float32(math.Inf(-1))
		}
		x[42] = 10
		x[v.eot] = 9
		for i := v.timestampBegin; i < len(x); i++ {
			x[i] = 0
		}
		return x
	}
	tests := []struct {
		name             string
		seq              []int
		allowed, blocked []int
	}{
		{"initial", nil, []int{v.timestampBegin}, []int{42, v.eot, v.timestampBegin + 1}},
		{"initial_timestamp", []int{v.timestampBegin}, []int{42, v.eot}, []int{v.timestampBegin, v.timestampBegin + 1}},
		{"closing", []int{v.timestampBegin, 42, v.timestampBegin + 5}, []int{v.eot, v.timestampBegin + 5}, []int{42, v.timestampBegin + 4}},
		{"new_start", []int{v.timestampBegin, 42, v.timestampBegin + 5, v.timestampBegin + 5}, []int{42, v.eot}, []int{v.timestampBegin + 5, v.timestampBegin + 6}},
		{"text", []int{v.timestampBegin, 42}, []int{42, v.eot, v.timestampBegin + 1}, []int{v.timestampBegin}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := scores()
			checkedTimestampMask(x, tt.seq, v, 0)
			for _, id := range tt.allowed {
				if math.IsInf(float64(x[id]), -1) {
					t.Fatalf("masked %d", id)
				}
			}
			for _, id := range tt.blocked {
				if !math.IsInf(float64(x[id]), -1) {
					t.Fatalf("allowed %d", id)
				}
			}
		})
	}
	x := scores()
	x[42] = 1
	x[v.eot] = 0
	checkedTimestampMask(x, []int{v.timestampBegin, 42}, v, 50)
	if !math.IsInf(float64(x[42]), -1) {
		t.Fatal("aggregate timestamp probability not applied")
	}
	x = scores()
	checkedTimestampMask(x, nil, v, 50)
	if math.IsInf(float64(x[v.timestampBegin+50]), -1) || !math.IsInf(float64(x[v.timestampBegin+51]), -1) {
		t.Fatal("initial limit")
	}
}

func scriptedCheckedDecode(ctx context.Context, cfg Config, tok *Tokenizer, v timestampVocabulary, opts PCMTranscribeOptions, sequence []int, mutate func(int, []float32)) ([]Segment, []int, error) {
	var calls []int
	forward := func(id int) ([]float32, error) {
		calls = append(calls, id)
		x := make([]float32, cfg.VocabSize)
		for i := range x {
			x[i] = -1000
		}
		index := len(calls) - 3
		if index >= 0 && index < len(sequence) {
			x[sequence[index]] = 100
		}
		if mutate != nil {
			mutate(len(calls), x)
		}
		return x, nil
	}
	segments, err := decodeCheckedTimestamps(ctx, cfg, tok, v, opts, nil, nil, forward)
	return segments, calls, err
}

func TestCheckedTimestampDecodePromptParsingAndLimit(t *testing.T) {
	for _, cfg := range []Config{Tiny(), LargeV3Turbo()} {
		tok := checkedTestTokenizer(cfg.VocabSize)
		v, err := checkedTimestampVocabulary(cfg, tok, "pt")
		if err != nil {
			t.Fatal(err)
		}
		sequence := []int{v.timestampBegin, 42, v.timestampBegin + 25, v.timestampBegin + 30, 42, v.timestampBegin + 50, v.eot}
		opts := PCMTranscribeOptions{Language: "pt", MaxNewTokens: len(sequence), MaxInitialTimestampIndex: 50}
		segs, calls, err := scriptedCheckedDecode(context.Background(), cfg, tok, v, opts, sequence, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(segs) != 2 || segs[0].Start != 0 || segs[0].End != 0.5 || segs[1].Start != 0.6 || segs[1].End != 1 || segs[0].Text != "hello" {
			t.Fatalf("segments %+v", segs)
		}
		want := append([]int{v.sot, v.language, v.transcribe}, sequence[:len(sequence)-1]...)
		if fmt.Sprint(calls) != fmt.Sprint(want) {
			t.Fatalf("prompt/position drift: %v want %v", calls, want)
		}
		opts.MaxNewTokens = 2
		segs, calls, err = scriptedCheckedDecode(context.Background(), cfg, tok, v, opts, sequence, nil)
		if !errors.Is(err, ErrGenerationLimit) || segs != nil || len(calls) != 4 {
			t.Fatalf("silent generation truncation: %v %d", err, len(calls))
		}
		cfg.MaxDecoderLength = 4
		opts.MaxNewTokens = 1
		_, calls, err = scriptedCheckedDecode(context.Background(), cfg, tok, v, opts, sequence, nil)
		if !errors.Is(err, ErrGenerationLimit) || len(calls) != 3 {
			t.Fatalf("position limit exceeded %d %v", len(calls), err)
		}
	}
}

func TestCheckedTimestampDecodeFailures(t *testing.T) {
	cfg := Tiny()
	tok := checkedTestTokenizer(cfg.VocabSize)
	v, _ := checkedTimestampVocabulary(cfg, tok, "pt")
	opts := PCMTranscribeOptions{Language: "pt", MaxNewTokens: 6}
	sequence := []int{v.timestampBegin, 42, v.eot}
	segs, _, err := scriptedCheckedDecode(context.Background(), cfg, tok, v, opts, sequence, nil)
	if !errors.Is(err, ErrIncompleteTimestamp) || segs != nil {
		t.Fatalf("invented end timestamp: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, calls, err := scriptedCheckedDecode(ctx, cfg, tok, v, opts, sequence, nil)
	if !errors.Is(err, context.Canceled) || len(calls) != 0 {
		t.Fatal("precancel")
	}
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	_, calls, err = scriptedCheckedDecode(ctx, cfg, tok, v, opts, sequence, func(n int, _ []float32) {
		if n == 4 {
			cancel()
		}
	})
	if !errors.Is(err, context.Canceled) || len(calls) != 4 {
		t.Fatal("cancel during token")
	}
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		_, _, err = scriptedCheckedDecode(context.Background(), cfg, tok, v, opts, sequence, func(_ int, x []float32) { x[0] = bad })
		if err == nil {
			t.Fatal("nonfinite logit accepted")
		}
	}
	_, _, err = scriptedCheckedDecode(context.Background(), cfg, tok, v, opts, sequence, func(_ int, x []float32) {
		for i := range x {
			x[i] = float32(math.Inf(-1))
		}
	})
	if err == nil {
		t.Fatal("all-masked logits accepted")
	}
	_, err = decodeCheckedTimestamps(context.Background(), cfg, tok, v, opts, []int{-1}, nil, func(int) ([]float32, error) { t.Fatal("invalid suppression called model"); return nil, nil })
	if err == nil {
		t.Fatal("invalid suppression accepted")
	}
	_, err = decodeCheckedTimestamps(context.Background(), cfg, tok, v, opts, nil, nil, func(int) ([]float32, error) { return nil, nil })
	if err == nil {
		t.Fatal("empty logits accepted")
	}
}
