package main

import (
	"testing"

	"github.com/rcarmo/go-pherence/models/whisper"
)

func TestShouldChunkSimple(t *testing.T) {
	if shouldChunkSimple(make([]float32, 16000*10), 30) {
		t.Fatal("short audio should not chunk")
	}
	if !shouldChunkSimple(make([]float32, 16000*31), 30) {
		t.Fatal("long audio should chunk")
	}
}

func TestJoinTexts(t *testing.T) {
	if got := joinTexts([]string{" hello ", "", "world"}); got != "hello world" {
		t.Fatalf("joinTexts=%q", got)
	}
}

func TestTextFromSegments(t *testing.T) {
	text := textFromSegments([]whisper.Segment{{Text: " hello "}, {Text: ""}, {Text: "world"}})
	if text != "hello world" {
		t.Fatalf("text=%q", text)
	}
}

func TestFilterTimestampSegmentsDropsPunctuationOnly(t *testing.T) {
	segments := []whisper.Segment{
		{Start: 0, End: 1, Text: "Hello"},
		{Start: 1, End: 2, Text: ",,,,"},
		{Start: 2, End: 3, Text: "  world  "},
		{Start: 3, End: 4, Text: ": This one here"},
	}
	out := filterTimestampSegments(segments)
	if len(out) != 3 || out[0].Text != "Hello" || out[1].Text != "world" || out[2].Text != "This one here" {
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
