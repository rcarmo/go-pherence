package main

import "testing"

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
