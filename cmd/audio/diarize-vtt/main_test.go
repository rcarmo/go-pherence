package main

import (
	"testing"

	"github.com/rcarmo/go-pherence/models/speaker"
)

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

func TestDegenerateCueText(t *testing.T) {
	if !degenerateCueText("and I'm going to go back and I'm going to go back and I'm going to go back") {
		t.Fatal("missed repeated phrase cue")
	}
	if !degenerateCueText("and") {
		t.Fatal("missed low-value short cue")
	}
	if degenerateCueText("go field") {
		t.Fatal("false positive on meaningful short cue")
	}
	if degenerateCueText("the recommendation is to confirm if it is necessary to go out into the field") {
		t.Fatal("false positive on normal cue")
	}
}

func TestSegmentsFromResultsCleansLowValuePunctuationCue(t *testing.T) {
	segments := segmentsFromResults([]result{
		{idx: 0, startSec: 0, endSec: 1, text: ".... I,"},
		{idx: 1, startSec: 1, endSec: 2, text: ": Hello world"},
	})
	if len(segments) != 1 || segments[0].Text != "Hello world" {
		t.Fatalf("segments=%+v", segments)
	}
}

func TestSegmentsFromResultsSortsByTime(t *testing.T) {
	segments := segmentsFromResults([]result{
		{idx: 9, startSec: 10, endSec: 11, text: "later"},
		{idx: 1, startSec: 1, endSec: 2, text: "earlier"},
	})
	if len(segments) != 2 || segments[0].Text != "earlier" || segments[1].Text != "later" {
		t.Fatalf("segments not time sorted: %+v", segments)
	}
}

func TestParseVTTMillisAndFilterCompletedJobs(t *testing.T) {
	ms, ok := parseVTTMillis("01:02:03.456")
	if !ok || ms != 3723456 {
		t.Fatalf("parse=%d,%v", ms, ok)
	}
	jobs := []job{
		{idx: 0, cueStart: 0, cueEnd: 16000},
		{idx: 1, cueStart: 16000, cueEnd: 32000},
	}
	remaining := filterCompletedJobs(jobs, map[cueKey]bool{{startMS: 0, endMS: 1000}: true})
	if len(remaining) != 1 || remaining[0].idx != 1 {
		t.Fatalf("remaining=%+v", remaining)
	}
	// Older partial VTTs may not match current VAD-packed cue boundaries exactly;
	// sufficient overlap should still count as completed.
	remaining = filterCompletedJobs(jobs, map[cueKey]bool{{startMS: 100, endMS: 950}: true})
	if len(remaining) != 1 || remaining[0].idx != 1 {
		t.Fatalf("fuzzy remaining=%+v", remaining)
	}
	remaining = filterCompletedJobs(jobs, map[cueKey]bool{{startMS: 700, endMS: 1000}: true})
	if len(remaining) != 2 {
		t.Fatalf("low-overlap remaining=%+v", remaining)
	}
}

func TestSpeakerLabelsFallback(t *testing.T) {
	vad := []speaker.VADSegment{{Start: 0, End: 1}, {Start: 2, End: 3}}
	labels := speakerLabels(make([]float32, 16000*3), vad, "", 0)
	if len(labels) != len(vad) {
		t.Fatalf("labels len=%d want %d", len(labels), len(vad))
	}
	for i, label := range labels {
		if label != 0 {
			t.Fatalf("label[%d]=%d want fallback 0", i, label)
		}
	}
}

func TestDynamicMaxTokens(t *testing.T) {
	if got := dynamicMaxTokens(40, 0.5, 4); got != 12 {
		t.Fatalf("short cue budget=%d want 12", got)
	}
	if got := dynamicMaxTokens(40, 5, 4); got != 28 {
		t.Fatalf("5s cue budget=%d want 28", got)
	}
	if got := dynamicMaxTokens(40, 20, 4); got != 40 {
		t.Fatalf("capped budget=%d want 40", got)
	}
	if got := dynamicMaxTokens(40, 5, 0); got != 40 {
		t.Fatalf("disabled budget=%d want 40", got)
	}
}
