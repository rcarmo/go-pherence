package main

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestProbeRefusesInvalidArtifactsAndOutput(t *testing.T) {
	dir := t.TempDir()
	output := filepath.Join(dir, "probe.wav")
	if err := run(dir, output); err == nil || !os.IsNotExist(err) {
		t.Fatalf("accepted absent model: %v", err)
	}
	if _, err := os.Stat(output); !os.IsNotExist(err) {
		t.Fatalf("created output on rejected model: %v", err)
	}
	path := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(path, []byte("bad"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := run(dir, output); err == nil || !strings.Contains(err.Error(), "size/type mismatch") {
		t.Fatalf("accepted short model: %v", err)
	}
	if err := verifyArtifact(path, modelSHA256, 3); err == nil || !strings.Contains(err.Error(), "SHA-256 mismatch") {
		t.Fatalf("accepted wrong hash: %v", err)
	}
}

func TestReleasedTwoFrameWAV(t *testing.T) {
	const envName = "GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR"
	dir := os.Getenv(envName)
	if dir == "" {
		t.Skipf("set %s to pinned 0.6B CustomVoice directory", envName)
	}
	output := filepath.Join(t.TempDir(), "two-frame.wav")
	if err := run(dir, output); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 44+3840*2 || string(data[:4]) != "RIFF" || string(data[8:16]) != "WAVEfmt " || binary.LittleEndian.Uint32(data[24:]) != 24000 || binary.LittleEndian.Uint16(data[22:]) != 1 || binary.LittleEndian.Uint16(data[34:]) != 16 || binary.LittleEndian.Uint32(data[40:]) != 3840*2 {
		t.Fatalf("unexpected WAV size/header: %d", len(data))
	}
	if err := run(dir, output); err == nil || !os.IsExist(err) {
		t.Fatalf("overwrote existing output: %v", err)
	}
	if again, err := os.ReadFile(output); err != nil || string(again) != string(data) {
		t.Fatalf("changed original on failed second write: %v", err)
	}
}
