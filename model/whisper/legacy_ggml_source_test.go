package whisper

import (
	"context"
	"fmt"
	legacy "github.com/rcarmo/go-pherence/loader/whisperggml"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLegacyWhisperNameMapping(t *testing.T) {
	for _, tc := range [][2]string{{"encoder.positional_embedding", "model.encoder.embed_positions.weight"}, {"decoder.blocks.3.cross_attn.key.weight", "model.decoder.layers.3.encoder_attn.k_proj.weight"}, {"encoder.blocks.31.mlp.2.bias", "model.encoder.layers.31.fc2.bias"}, {"decoder.ln.bias", "model.decoder.layer_norm.bias"}, {"encoder.blocks.4.attn_ln.weight", "model.encoder.layers.4.self_attn_layer_norm.weight"}, {"decoder.token_embedding.weight", "model.decoder.embed_tokens.weight"}} {
		got, err := legacyWhisperHFName(tc[0])
		if err != nil || got != tc[1] {
			t.Fatal(tc, got, err)
		}
	}
	for _, n := range []string{"unknown", "encoder.blocks.-1.mlp.0.weight", "encoder.blocks.01.mlp.0.weight", "encoder.blocks.65.mlp.0.weight", "decoder.blocks.1.attn.query.notweight", "encoder.blocks.2.mlp.3.weight"} {
		if _, err := legacyWhisperHFName(n); err == nil {
			t.Fatal("badname", n)
		}
	}
}
func pinnedLegacyWhisperModel(t *testing.T, ctx context.Context) (*Whisper, *Tokenizer, *CheckedGenerationConfig) {
	t.Helper()
	m, tok, policy, file := pinnedLegacyWhisperModelOpen(t, ctx)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	return m, tok, policy
}

// Construction callers can use the same verified open file for packed reads;
// avoid reopening and hashing the model twice. Close remains caller-owned.
func pinnedLegacyWhisperModelOpen(t *testing.T, ctx context.Context) (*Whisper, *Tokenizer, *CheckedGenerationConfig, *legacy.File) {
	t.Helper()
	dir := os.Getenv("GO_PHERENCE_WHISPER_TURBO_DIR")
	if dir == "" {
		t.Fatal("HF config/tokenizer provenance required")
	}
	for _, item := range [][2]string{{"config.json", "c5b526b3e3cd64cd8940dabb45e8ba726629e22d8ed389c29b552f9140daf04a"}, {"generation_config.json", "cce11bfe3aaa6ae9e072ea2637caaec8795e68d9b67e655a5af16ee509681a4c"}, {"tokenizer.json", "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd"}} {
		pinnedSpeechFile(t, filepath.Join(dir, item[0]), item[1])
	}
	config, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseModelConfigChecked(config)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := LoadTokenizer(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	gen, err := os.ReadFile(filepath.Join(dir, "generation_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	policy, err := ParseGenerationConfigChecked(gen, cfg, tok)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	file, err := legacy.Open(ctx, os.Getenv("GO_PHERENCE_WHISPER_GGML"), os.Getenv("GO_PHERENCE_WHISPER_GGML_SHA256"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := file.Close(); err != nil {
			t.Error(err)
		}
	})
	vocab := file.Vocabulary()
	if len(vocab) != TokenEOT {
		t.Fatal("legacy vocabulary incomplete")
	}
	for i := 0; i < TokenEOT; i++ {
		if tok.decodeRaw([]int{i}) != string(vocab[i]) {
			t.Fatalf("legacy/HF text vocabulary mismatch id=%d", i)
		}
	}
	source, err := newLegacyGGMLSource(ctx, file, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(source.infos) != 587 {
		t.Fatal("inventory", len(source.infos))
	}
	model, err := LoadModelSourceChecked(ctx, source, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("LEGACY_LOAD seconds=%g storage=%d tensors=%d pin=%s explicit_widened_values=true", time.Since(start).Seconds(), file.Header().FileType, len(source.infos), file.SHA256())
	return model, tok, policy, file
}
func TestPinnedLegacyWhisperValueLoad(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_LEGACY_WHISPER_LOAD") != "1" {
		t.Skip("explicit retained-value load")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 2*time.Minute {
		t.Fatal("bounded2min")
	}
	ctx, cancel := context.WithDeadline(context.Background(), d.Add(-time.Second))
	defer cancel()
	model, _, _ := pinnedLegacyWhisperModel(t, ctx)
	fmt.Println("LEGACY_LOADED", model.Config)
}

func TestLegacyWhisperSourceRejectsNilBeforeLoading(t *testing.T) {
	cfg := Config{}
	if _, err := newLegacyGGMLSource(nil, nil, cfg); err == nil {
		t.Fatal("nilctx")
	}
	if _, err := newLegacyGGMLSource(context.Background(), nil, cfg); err == nil {
		t.Fatal("nilfile")
	}
	var s *legacyGGMLSource
	if s.TensorInfos() != nil {
		t.Fatal("nilmetadata")
	}
	if _, _, err := s.GetFloat32("x"); err == nil {
		t.Fatal("nilread")
	}
}
