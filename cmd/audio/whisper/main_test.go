package main

import (
	"testing"

	"github.com/rcarmo/go-pherence/models/whisper"
)

func TestFilterTimestampSegmentsDropsPunctuationOnly(t *testing.T) {
	segments := []whisper.Segment{
		{Start: 0, End: 1, Text: "Hello"},
		{Start: 1, End: 2, Text: ",,,,"},
		{Start: 2, End: 3, Text: "  world  "},
	}
	out := filterTimestampSegments(segments)
	if len(out) != 2 || out[0].Text != "Hello" || out[1].Text != "world" {
		t.Fatalf("filtered=%+v", out)
	}
}

func TestDefaultWhisperTurboPromptContract(t *testing.T) {
	if defaultWhisperModelPath != "models/whisper-large-v3-turbo-hf/model.safetensors" {
		t.Fatalf("default model=%q", defaultWhisperModelPath)
	}
	if defaultWhisperSize != "turbo" {
		t.Fatalf("default size=%q", defaultWhisperSize)
	}
	if defaultWhisperLanguage != "en" {
		t.Fatalf("default language=%q", defaultWhisperLanguage)
	}
}
