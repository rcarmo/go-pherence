package main

import (
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"os"
	"path/filepath"
	"testing"
)

func TestPreparedOutputExclusive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompt.json")
	p := loader.PreparedPrompt{Text: "Test.", TargetFrames: 1}
	if err := writePreparedPrompt(path, p); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(path)
	if err := writePreparedPrompt(path, p); err == nil {
		t.Fatal("overwrote output")
	}
	after, _ := os.ReadFile(path)
	if string(before) != string(after) {
		t.Fatal("changed existing file")
	}
	if err := writePreparedPrompt(filepath.Join(t.TempDir(), "bad.wav"), p); err == nil {
		t.Fatal("accepted non JSON output")
	}
}

func TestGenerationPromptPreservesRMS(t *testing.T) {
	rms := .05
	p := loader.PreparedPrompt{Text: "Text.", TargetFrames: 2, RefRMS: &rms, Conditional: loader.PreparedInput{Tokens: 3, IDs: []int{1, 2, 3}, AudioMask: []bool{false, true, true}}, Unconditional: loader.PreparedInput{Tokens: 2, IDs: []int{2, 3}, AudioMask: []bool{true, true}}}
	got := generationPrompt(p)
	if got.RefRMS == nil || *got.RefRMS != rms || got.Conditional.Tokens != 3 || got.Target != 2 || got.Text != p.Text {
		t.Fatal("lost prompt metadata")
	}
}

func TestPreparationPreflight(t *testing.T) {
	output := filepath.Join(t.TempDir(), "new.wav")
	if err := validatePreparationFlags("synthesize", "", output, "text", "voice.wav", "transcript", "", 75, 8); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"-mode", "synthesize", "-model", "/nonexistent", "-output", output, "-reference", "a.wav", "-text", "text", "-steps", "0"},
		{"-mode", "prepare", "-model", "/nonexistent", "-output", output},
		{"-mode", "encode-reference", "-model", "/nonexistent", "-output", ""},
	} {
		if err := run(args); err == nil {
			t.Fatal("accepted invalid flags")
		}
	}
}
