package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmission(t *testing.T) {
	for _, args := range [][]string{nil, {"-config", "x"}, {"-config", "x", "-model", "y", "-tokenizer", "z", "-voice", "v", "-max-frames", "0"}} {
		if err := run(args, new(bytes.Buffer)); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}
func TestReleasedCLISmoke(t *testing.T) {
	model := os.Getenv("GO_PHERENCE_POCKETTTS_MODEL")
	tokenizer := os.Getenv("GO_PHERENCE_POCKETTTS_TOKENIZER")
	voice := os.Getenv("GO_PHERENCE_POCKETTTS_VOICE")
	if model == "" || tokenizer == "" || voice == "" {
		t.Skip("set Pocket TTS released artifact variables")
	}
	out := filepath.Join(t.TempDir(), "out.wav")
	var stdout bytes.Buffer
	err := run([]string{"-config", "../../../model/pockettts/testdata/english-upstream.yaml", "-model", model, "-tokenizer", tokenizer, "-voice", voice, "-text", "Hello world!", "-max-frames", "1", "-seed", "42", "-out", out}, &stdout)
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(out)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() != 44+2*1920 || !strings.Contains(stdout.String(), `"samples":1920`) {
		t.Fatalf("size=%d output=%s", info.Size(), stdout.String())
	}
}
