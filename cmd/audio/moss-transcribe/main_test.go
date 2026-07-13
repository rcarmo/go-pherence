package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/model/mosstranscribe"
)

func TestCapabilitiesNeedsNoModel(t *testing.T) {
	var out, stderr bytes.Buffer
	if err := run([]string{"-capabilities"}, &out, &stderr); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "native=true") || !strings.Contains(out.String(), "avx2_fma=") {
		t.Fatalf("capabilities=%q", out.String())
	}
}

func TestCLIRejectsUnsupportedControlsEarly(t *testing.T) {
	if err := run([]string{"-model-dir", "x", "-audio", "a.mp3"}, &bytes.Buffer{}, &bytes.Buffer{}); err == nil || !strings.Contains(err.Error(), "PCM WAV only") {
		t.Fatalf("audio error=%v", err)
	}
	if err := run([]string{"-model-dir", "x", "-audio", "a.wav", "-format", "vtt"}, &bytes.Buffer{}, &bytes.Buffer{}); err == nil || !strings.Contains(err.Error(), "unsupported format") {
		t.Fatalf("format error=%v", err)
	}
}

func TestRenderJSONAndText(t *testing.T) {
	segments := []mosstranscribe.TranscriptSegment{{Start: 0, End: 1, Speaker: "S01", Text: "hi"}}
	text, err := render("text", "raw", segments)
	if err != nil || text != "[0.00-1.00][S01] hi\n" {
		t.Fatalf("text=%q err=%v", text, err)
	}
	payload, err := render("json", "raw", segments)
	if err != nil || !strings.Contains(payload, `"text": "raw"`) || !strings.Contains(payload, `"speaker": "S01"`) {
		t.Fatalf("json=%q err=%v", payload, err)
	}
}
