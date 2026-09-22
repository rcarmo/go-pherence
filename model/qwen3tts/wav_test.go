package qwen3tts

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
)

func TestWritePCM16Mono24k(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.wav")
	input := []float32{-1, -.5, 0, .5, 1}
	if err := WritePCM16Mono24k(path, input); err != nil {
		t.Fatal(err)
	}
	got, rate, err := audio.WAV(path)
	if err != nil || rate != 24000 || len(got) != len(input) {
		t.Fatalf("wav rate=%d samples=%d err=%v", rate, len(got), err)
	}
	for i := range input {
		if math.Abs(float64(got[i]-input[i])) > 1.0/32767+1e-6 {
			t.Fatalf("sample[%d]=%g want %g", i, got[i], input[i])
		}
	}
	before, _ := os.ReadFile(path)
	if err := WritePCM16Mono24k(path, []float32{.1}); err == nil {
		t.Fatal("overwrote existing WAV")
	}
	after, _ := os.ReadFile(path)
	if string(before) != string(after) {
		t.Fatal("existing WAV changed")
	}
}

func TestWritePCM16Mono24kRejectsMalformed(t *testing.T) {
	for _, samples := range [][]float32{nil, {float32(math.NaN())}, {float32(math.Inf(1))}, {-1.01}, {1.01}} {
		path := filepath.Join(t.TempDir(), "bad.wav")
		if err := WritePCM16Mono24k(path, samples); err == nil {
			t.Fatalf("accepted %v", samples)
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("partial output remains for %v", samples)
		}
	}
}
