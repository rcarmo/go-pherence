package whisper

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
)

func modelJSONFixture(c Config) map[string]any {
	return map[string]any{"model_type": "whisper", "activation_function": "gelu", "architectures": []string{"WhisperForConditionalGeneration"}, "is_encoder_decoder": true, "scale_embedding": false, "num_mel_bins": c.NumMelBins, "d_model": c.EncoderDModel, "encoder_layers": c.EncoderLayers, "decoder_layers": c.DecoderLayers, "encoder_attention_heads": c.EncoderHeads, "decoder_attention_heads": c.DecoderHeads, "encoder_ffn_dim": c.EncoderFFNDim, "decoder_ffn_dim": c.DecoderFFNDim, "vocab_size": c.VocabSize, "max_target_positions": c.MaxDecoderLength, "max_source_positions": c.MaxLength / 2, "bos_token_id": 50257, "eos_token_id": 50257, "pad_token_id": 50257, "decoder_start_token_id": 50258}
}
func marshalConfig(t *testing.T, v any) []byte {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func generationJSONFixture(t *testing.T, cfg Config) (*Tokenizer, map[string]any) {
	t.Helper()
	tok := checkedTestTokenizer(cfg.VocabSize)
	transcribe := cfg.VocabSize - 1506
	langs := map[string]int{}
	for id := 50259; id < transcribe-1; id++ {
		name := fmt.Sprintf("<|language%d|>", id)
		if id == 50259 {
			name = "<|en|>"
		}
		if id == 50267 {
			name = "<|pt|>"
		}
		tok.Vocab[id] = name
		langs[name] = id
	}
	return tok, map[string]any{"bos_token_id": 50257, "eos_token_id": 50257, "pad_token_id": 50257, "decoder_start_token_id": 50258, "no_timestamps_token_id": cfg.VocabSize - 1502, "is_multilingual": true, "max_length": cfg.MaxDecoderLength, "max_initial_timestamp_index": 50, "lang_to_id": langs, "task_to_id": map[string]int{"transcribe": transcribe, "translate": transcribe - 1}, "suppress_tokens": []int{1, 2, 42}, "begin_suppress_tokens": []int{220, 50257}, "forced_decoder_ids": [][]any{{1, nil}, {2, transcribe}, {3, cfg.VocabSize - 1502}}, "return_timestamps": false}
}

func TestCheckedConfigJSONBoundsDuplicatesAndNulls(t *testing.T) {
	for _, data := range []string{"", "null", "[]", "{} {}", `{"a":1,"a":2}`, `{"x":{"a":1,"a":2}}`, strings.Repeat("[", 34) + "0" + strings.Repeat("]", 34), strings.Repeat(" ", maxSpeechConfigJSON+1)} {
		if _, err := speechJSONObject([]byte(data)); err == nil {
			t.Fatalf("accepted invalid JSON %.80q", data)
		}
	}
	fields, err := speechJSONObject([]byte(`{"a":[0,null],"b":{"pt":null}}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := configField[[]int](fields, "a"); err == nil {
		t.Fatal("null slice value became zero")
	}
	if _, err := configField[map[string]int](fields, "b"); err == nil {
		t.Fatal("null map value became zero")
	}
}

func TestCheckedConfigModelMappingAndRejections(t *testing.T) {
	for _, want := range []Config{Tiny(), Base(), LargeV3Turbo()} {
		got, err := ParseModelConfigChecked(marshalConfig(t, modelJSONFixture(want)))
		if err != nil || got != want {
			t.Fatalf("geometry %+v %v", got, err)
		}
	}
	for _, tt := range []struct {
		key   string
		value any
	}{
		{"unsupported_encoder_scale", true}, {"d_model", 1 << 60}, {"max_source_positions", 1 << 60}, {"max_source_positions", 0}, {"max_target_positions", 449}, {"encoder_attention_heads", 0}, {"decoder_attention_heads", 7}, {"vocab_size", 51864}, {"num_mel_bins", 64}, {"activation_function", "relu"}, {"scale_embedding", true}, {"is_encoder_decoder", false}, {"model_type", "bart"}, {"architectures", []string{"WhisperModel"}}, {"decoder_start_token_id", 50259}, {"tie_word_embeddings", false}, {"layer_norm_eps", 1e-6}, {"num_mel_bins", nil}, {"num_mel_bins", 80.5},
	} {
		t.Run(tt.key+fmt.Sprint(tt.value), func(t *testing.T) {
			m := modelJSONFixture(Tiny())
			m[tt.key] = tt.value
			if _, err := ParseModelConfigChecked(marshalConfig(t, m)); err == nil {
				t.Fatal("accepted unsupported model config")
			}
		})
	}
	m := modelJSONFixture(Tiny())
	delete(m, "max_target_positions")
	if _, err := ParseModelConfigChecked(marshalConfig(t, m)); err == nil {
		t.Fatal("missing position limit")
	}
}

func TestCheckedGenerationWordAlignmentCapability(t *testing.T) {
	cfg := Tiny()
	tok, raw := generationJSONFixture(t, cfg)
	g, err := ParseGenerationConfigChecked(marshalConfig(t, raw), cfg, tok)
	if err != nil || g.SupportsWordAlignment() {
		t.Fatal("unexpected alignment capability", err)
	}
	raw["alignment_heads"] = [][]int{{0, 0}}
	g, err = ParseGenerationConfigChecked(marshalConfig(t, raw), cfg, tok)
	if err != nil || !g.SupportsWordAlignment() {
		t.Fatal("missing alignment capability", err)
	}
}

func TestCheckedConfigGenerationImportAndPrecedence(t *testing.T) {
	cfg := Tiny()
	tok, m := generationJSONFixture(t, cfg)
	g, err := ParseGenerationConfigChecked(marshalConfig(t, m), cfg, tok)
	if err != nil {
		t.Fatal(err)
	}
	v, _ := checkedTimestampVocabulary(cfg, tok, "pt")
	opts, suppress, begin, err := resolvePCMGeneration(cfg, v, PCMTranscribeOptions{Language: "pt", Generation: g}, []int{-1}, []int{-1})
	if err != nil {
		t.Fatal(err)
	}
	if opts.MaxNewTokens != 445 || opts.MaxInitialTimestampIndex != 50 || fmt.Sprint(suppress) != "[1 2 42]" || fmt.Sprint(begin) != "[220 50257]" {
		t.Fatal("generation precedence", opts, suppress, begin)
	}
	// Source object mutation cannot affect the parsed immutable policy.
	m["suppress_tokens"].([]int)[0] = 999
	if g.suppress[0] != 1 {
		t.Fatal("policy aliases JSON source")
	}
	for _, o := range []PCMTranscribeOptions{{Language: "pt", Generation: g, MaxInitialTimestampIndex: 1}, {Language: "pt", Generation: g, MaxNewTokens: 446}, {Language: "fr", Generation: g}} {
		if _, _, _, err := resolvePCMGeneration(cfg, v, o, nil, nil); err == nil {
			t.Fatal("accepted conflicting options")
		}
	}
	short, _, _, err := resolvePCMGeneration(cfg, v, PCMTranscribeOptions{Language: "pt", Generation: g, MaxNewTokens: 5}, nil, nil)
	if err != nil || short.MaxNewTokens != 5 {
		t.Fatal("cannot shorten generation budget")
	}
	cfg.MaxLength = 2
	if _, _, _, err := resolvePCMGeneration(cfg, v, PCMTranscribeOptions{Language: "pt", Generation: g}, nil, nil); err == nil {
		t.Fatal("policy used for different model")
	}
}

func TestCheckedConfigGenerationRejections(t *testing.T) {
	cfg := Tiny()
	for _, tt := range []struct {
		key   string
		value any
	}{
		{"do_sample", true}, {"num_beams", 2}, {"temperature", 0.5}, {"top_p", 0.9}, {"repetition_penalty", 1.2}, {"no_repeat_ngram_size", 3}, {"unknown_control", true}, {"is_multilingual", false}, {"max_length", 449}, {"max_length", 3}, {"max_initial_timestamp_index", 1501}, {"max_initial_timestamp_index", -1}, {"no_timestamps_token_id", 50364}, {"pad_token_id", -1}, {"suppress_tokens", []int{-1}}, {"suppress_tokens", []int{50257}}, {"suppress_tokens", []int{50364}}, {"begin_suppress_tokens", []int{50364}}, {"suppress_tokens", []any{nil}}, {"lang_to_id", map[string]int{"<|en|>": 50259}}, {"task_to_id", map[string]int{"transcribe": 50360, "translate": 50359}}, {"forced_decoder_ids", [][]any{{1, 42}}}, {"forced_decoder_ids", [][]any{{2, nil}}}, {"forced_decoder_ids", [][]any{{1, 50259}, {1, 50267}}}, {"forced_decoder_ids", [][]any{{4, 42}}}, {"forced_decoder_ids", [][]any{{3, 50364}}}, {"alignment_heads", [][]int{{4, 0}}}, {"alignment_heads", [][]any{{nil, 0}}}, {"return_timestamps", nil}, {"language", "bad"}, {"task", "summarize"},
	} {
		t.Run(tt.key+fmt.Sprint(tt.value), func(t *testing.T) {
			tok, m := generationJSONFixture(t, cfg)
			m[tt.key] = tt.value
			if _, err := ParseGenerationConfigChecked(marshalConfig(t, m), cfg, tok); err == nil {
				t.Fatal("accepted unsupported generation")
			}
		})
	}
	tok, m := generationJSONFixture(t, cfg)
	delete(m, "suppress_tokens")
	if _, err := ParseGenerationConfigChecked(marshalConfig(t, m), cfg, tok); err == nil {
		t.Fatal("missing suppression accepted")
	}
}

func TestCheckedConfigConfiguredLoaderBeforeTensorReads(t *testing.T) {
	cfg := checkedLoadConfig()
	cfg.MaxLength = 4
	tok, gen := generationJSONFixture(t, cfg)
	source := checkedLoadFixture(cfg, "F32")
	model := modelJSONFixture(cfg)
	w, g, err := LoadConfiguredModelSourceChecked(context.Background(), source, marshalConfig(t, model), marshalConfig(t, gen), tok)
	if err != nil || w == nil || g == nil || source.reads == 0 {
		t.Fatalf("configured load %v", err)
	}
	source.reads = 0
	gen["num_beams"] = 2
	w, g, err = LoadConfiguredModelSourceChecked(context.Background(), source, marshalConfig(t, model), marshalConfig(t, gen), tok)
	if err == nil || w != nil || g != nil || source.reads != 0 {
		t.Fatal("unsupported generation reached tensors")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, err = LoadConfiguredModelSourceChecked(ctx, source, marshalConfig(t, model), marshalConfig(t, gen), tok)
	if !errors.Is(err, context.Canceled) || source.reads != 0 {
		t.Fatal("pre-cancel load")
	}
}

func TestCheckedConfigPCMPolicyDoesNotMutateDecoder(t *testing.T) {
	w := toyPCMModel()
	tok, gen := generationJSONFixture(t, w.Config)
	gen["max_initial_timestamp_index"] = 0
	g, err := ParseGenerationConfigChecked(marshalConfig(t, gen), w.Config, tok)
	if err != nil {
		t.Fatal(err)
	}
	// Invalid legacy settings must not influence the supplied checked policy.
	w.Decoder.SuppressTokens = []int{-1}
	w.Decoder.BeginSuppressTokens = []int{-1}
	calls := 0
	reader := sampleReadFunc(func(_ context.Context, dst []float32, _ int64) (int, error) { clear(dst); return len(dst), nil })
	err = w.TranscribePCMWindows(context.Background(), reader, 321, tok, PCMTranscribeOptions{Language: "pt", Generation: g}, func(out WindowTranscript) error {
		calls++
		if len(out.Segments) != 0 {
			t.Fatal("expected toy EOT")
		}
		return nil
	})
	if err != nil || calls != 2 {
		t.Fatalf("configured PCM %d %v", calls, err)
	}
	if w.Decoder.SuppressTokens[0] != -1 || w.Decoder.BeginSuppressTokens[0] != -1 {
		t.Fatal("mutated model suppression")
	}
	// Without a policy, invalid legacy lists now fail before reading PCM.
	reader = sampleReadFunc(func(context.Context, []float32, int64) (int, error) {
		t.Fatal("bad suppression read audio")
		return 0, nil
	})
	if err := w.TranscribePCMWindows(context.Background(), reader, 1, tok, PCMTranscribeOptions{Language: "pt"}, func(WindowTranscript) error { return nil }); err == nil {
		t.Fatal("invalid legacy suppression accepted")
	}
}

func TestCheckedConfigSuppressionReachesTokenSelection(t *testing.T) {
	cfg := Tiny()
	tok, raw := generationJSONFixture(t, cfg)
	tok.Vocab[43] = "allowed"
	g, err := ParseGenerationConfigChecked(marshalConfig(t, raw), cfg, tok)
	if err != nil {
		t.Fatal(err)
	}
	v, err := checkedTimestampVocabulary(cfg, tok, "pt")
	if err != nil {
		t.Fatal(err)
	}
	opts, suppress, begin, err := resolvePCMGeneration(cfg, v, PCMTranscribeOptions{Language: "pt", Generation: g}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	forward := func(id int) ([]float32, error) {
		calls++
		logits := make([]float32, cfg.VocabSize)
		for i := range logits {
			logits[i] = -1000
		}
		switch calls {
		case 1:
			if id != v.sot {
				t.Fatal("wrong SOT")
			}
		case 2:
			if id != v.language {
				t.Fatal("wrong language")
			}
		case 3:
			if id != v.transcribe {
				t.Fatal("source prompt defaults leaked")
			}
			logits[v.timestampBegin+10] = 100
		case 4:
			if id != v.timestampBegin+10 {
				t.Fatal("generation initial timestamp bound not applied")
			}
			logits[42] = 100
			logits[43] = 90
		case 5:
			if id != 43 {
				t.Fatal("suppressed text token reached decoder")
			}
			logits[v.timestampBegin+25] = 100
		case 6:
			logits[v.eot] = 100
		default:
			t.Fatal("decoder exceeded expected call count")
		}
		return logits, nil
	}
	segments, err := decodeCheckedTimestamps(context.Background(), cfg, tok, v, opts, suppress, begin, forward)
	if err != nil || calls != 6 || len(segments) != 1 || segments[0].Text != "allowed" || segments[0].Start != 0.2 || segments[0].End != 0.5 {
		t.Fatalf("suppression/config semantics %+v %v", segments, err)
	}
}
