package main

import (
	"bytes"
	"encoding/json"
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
	if info.Size() != 44+2*1920 || !strings.Contains(stdout.String(), `"samples":1920`) || !strings.Contains(stdout.String(), `"capacity_reached":true`) {
		t.Fatalf("size=%d output=%s", info.Size(), stdout.String())
	}
	// The same pinned prompt and seed has room to finish below this cap.
	out2 := filepath.Join(t.TempDir(), "longer.wav")
	stdout.Reset()
	err = run([]string{"-config", "../../../model/pockettts/testdata/english-upstream.yaml", "-model", model, "-tokenizer", tokenizer, "-voice", voice, "-text", "Hello world!", "-max-frames", "64", "-seed", "42", "-out", out2}, &stdout)
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		Samples         int  `json:"samples"`
		CapacityReached bool `json:"capacity_reached"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Samples <= 0 || result.Samples >= 64*1920 || result.CapacityReached {
		t.Fatalf("longer generation samples=%d output=%s", result.Samples, stdout.String())
	}
}
