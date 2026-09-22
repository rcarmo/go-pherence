package pockettts

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"testing"
)

const (
	releasedModelSHA256     = "916ccd2686e9311cb40054893a3c4284393d658825ffc714a276f3e9b152344f"
	releasedModelBytes      = int64(219029196)
	releasedTokenizerSHA256 = "f498428e1eafee50492f7be13dc9bfafcfc12e508cd0eb1b01c92ecd5d8c6687"
	releasedTokenizerBytes  = int64(245020)
	releasedVoiceSHA256     = "69c32db63ca56843d994f81f343f62e0bf2d73f7e4c9bc73e44bb1110b1d8845"
	releasedVoiceBytes      = int64(6194424)
)

func releasedAsset(tb testing.TB, env, wantHash string, wantBytes int64) string {
	tb.Helper()
	path := os.Getenv(env)
	if path == "" {
		tb.Skipf("set %s to the pinned Pocket TTS artifact", env)
	}
	file, err := os.Open(path)
	if err != nil {
		tb.Fatalf("open %s: %v", env, err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		tb.Fatal(err)
	}
	if info.Size() != wantBytes {
		tb.Fatalf("%s bytes=%d want=%d", env, info.Size(), wantBytes)
	}
	h := sha256.New()
	if _, err := io.Copy(h, file); err != nil {
		tb.Fatal(err)
	}
	if got := fmt.Sprintf("%x", h.Sum(nil)); got != wantHash {
		tb.Fatalf("%s sha256=%s want=%s", env, got, wantHash)
	}
	return path
}
func releasedModel(tb testing.TB) string {
	return releasedAsset(tb, "GO_PHERENCE_POCKETTTS_MODEL", releasedModelSHA256, releasedModelBytes)
}
func releasedTokenizerPath(tb testing.TB) string {
	return releasedAsset(tb, "GO_PHERENCE_POCKETTTS_TOKENIZER", releasedTokenizerSHA256, releasedTokenizerBytes)
}
func releasedVoice(tb testing.TB) string {
	return releasedAsset(tb, "GO_PHERENCE_POCKETTTS_VOICE", releasedVoiceSHA256, releasedVoiceBytes)
}
