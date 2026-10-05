package whisper

import (
	"context"
	"reflect"
	"testing"
)

func TestCheckedPreviousTextPrompt(t *testing.T) {
	cfg := LargeV3Turbo()
	tok := checkedTestTokenizer(cfg.VocabSize)
	plain, err := checkedTimestampVocabulary(cfg, tok, "pt")
	if err != nil || plain.previous != 0 {
		t.Fatal(plain.previous, err)
	}
	tok.Vocab[plain.transcribe+2] = "<|startofprev|>"
	v, err := checkedTimestampVocabulary(cfg, tok, "pt")
	if err != nil || v.previous != plain.transcribe+2 {
		t.Fatal(v.previous, err)
	}
	bad := checkedTestTokenizer(cfg.VocabSize)
	bad.Vocab[plain.transcribe+3] = "<|startofprev|>"
	if _, err := checkedTimestampVocabulary(cfg, bad, "pt"); err == nil {
		t.Fatal("misplaced <|startofprev|> accepted")
	}
	// Scripted decoder: <|0.00|> 42 <|0.10|> EOT after any prompt.
	script := []int{v.timestampBegin, 42, v.timestampBegin + 5, v.eot}
	run := func(v timestampVocabulary, previous []int) ([]int, []Segment, []int, error) {
		var fed []int
		step := 0
		promptLen := -1
		segs, generated, err := decodeCheckedTimestampsPrompted(context.Background(), cfg, tok, v, PCMTranscribeOptions{MaxInitialTimestampIndex: 50}, nil, nil, previous, func(token int) ([]float32, error) {
			fed = append(fed, token)
			logits := make([]float32, cfg.VocabSize)
			for i := range logits {
				logits[i] = -100
			}
			if token == v.transcribe {
				promptLen = len(fed)
			}
			if promptLen >= 0 {
				logits[script[step]] = 100
				step++
			}
			return logits, nil
		}, nil)
		return fed, segs, generated, err
	}
	fed, segs, generated, err := run(v, nil)
	if err != nil || !reflect.DeepEqual(fed[:3], []int{v.sot, v.language, v.transcribe}) || len(segs) != 1 || segs[0].End != 0.1 || !reflect.DeepEqual(generated, script[:3]) {
		t.Fatal("no prompt", fed, segs, generated, err)
	}
	long := make([]int, 300)
	for i := range long {
		long[i] = i // text tokens; last 223 retained
	}
	long[299] = v.timestampBegin + 7 // timestamps are valid previous tokens
	fed, _, generated, err = run(v, long)
	want := append(append([]int{v.previous}, long[300-maxPreviousTextTokens:]...), v.sot, v.language, v.transcribe)
	if err != nil || !reflect.DeepEqual(fed[:len(want)], want) || !reflect.DeepEqual(generated, script[:3]) {
		t.Fatal("prompt order", len(fed), err)
	}
	for _, id := range []int{-1, v.eot, v.sot, v.language, v.transcribe, v.previous, v.noTimestamps, cfg.VocabSize} {
		if _, _, _, err := run(v, []int{1, id}); err == nil {
			t.Fatal("invalid previous token accepted", id)
		}
	}
	// Advance path: every prompt token but the last skips logits; generation
	// and the logits actually read are unchanged.
	var advanced, forwarded []int
	step, calls := 0, 0
	_, generated2, err := decodeCheckedTimestampsPrompted(context.Background(), cfg, tok, v, PCMTranscribeOptions{MaxInitialTimestampIndex: 50}, nil, nil, long, func(token int) ([]float32, error) {
		forwarded = append(forwarded, token)
		logits := make([]float32, cfg.VocabSize)
		for i := range logits {
			logits[i] = -100
		}
		if token == v.transcribe || len(forwarded) > 1 {
			logits[script[step]] = 100
			step++
		}
		return logits, nil
	}, func(tokens []int) error { advanced = append(advanced, tokens...); calls++; return nil })
	if err != nil || calls != 1 || !reflect.DeepEqual(advanced, want[:len(want)-1]) || forwarded[0] != v.transcribe || !reflect.DeepEqual(generated2, script[:3]) {
		t.Fatal("advance path", advanced, forwarded, generated2, err)
	}
	if _, _, _, err := run(plain, []int{1}); err == nil {
		t.Fatal("previous prompt without marker accepted")
	}
}
