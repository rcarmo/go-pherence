package model

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
	gemmacfg "github.com/rcarmo/go-pherence/model/gemma"
)

func TestGemma4GGUF12BExactArtifactCompatibility(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_GEMMA4_12B")
	if path == "" {
		t.Skip("set GO_PHERENCE_GEMMA4_12B to the pinned Gemma 4 12B GGUF")
	}
	const (
		wantBytes  = int64(7366423360)
		wantSHA256 = "90fd944d227e9d9b68e7e2c7d5b57b79d4c66ed521b0919fbbd932cf834f6f8e"
	)
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		t.Fatal(err)
	}
	if info.Size() != wantBytes {
		f.Close()
		t.Fatalf("12B GGUF bytes=%d want %d", info.Size(), wantBytes)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		f.Close()
		t.Fatal(err)
	}
	f.Close()
	if got := hex.EncodeToString(h.Sum(nil)); got != wantSHA256 {
		t.Fatalf("12B GGUF SHA-256=%s want %s", got, wantSHA256)
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := gemma4GGUFConfig(g)
	g.Close()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.NumLayers != 48 || cfg.HiddenSize != 3840 || cfg.VocabSize != 262144 || cfg.MaxSeqLen != 262144 {
		t.Fatalf("12B config=%+v", cfg)
	}
	if len(cfg.KVHeadsPerLayer) != cfg.NumLayers || gemmacfg.LayerKVHeads(cfg, 0) != 8 || gemmacfg.LayerKVHeads(cfg, 5) != 1 {
		t.Fatalf("12B KV heads=%v", cfg.KVHeadsPerLayer)
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	if m.EmbedTokensGGUF == nil || m.EmbedTokensGGUF.QType != gguf.QuantQ5_K || m.LMHeadGGUF != m.EmbedTokensGGUF {
		t.Fatalf("12B tied embedding/head=%+v/%v", m.EmbedTokensGGUF, m.LMHeadGGUF == m.EmbedTokensGGUF)
	}
}
